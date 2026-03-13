#!/bin/bash
# run_remote_benchmarks.sh
# Runner script to set up envs and run benchmarks on a fresh cloud instance.

# 1. Safety Check: Ensure a device name was passed
if [ -z "$1" ]; then
    echo "❌ Error: No device name provided."
    echo "Usage: bash run_remote_benchmarks.sh <device_name>"
    echo "Example: bash run_remote_benchmarks.sh RTX_4090"
    exit 1
fi

# Parse device name
DEVICE_NAME=$1

# Grab the current date in YYYYMMDD format (e.g., 20260311)
CURRENT_DATE=$(date +"%Y%m%d")

# Exit immediately if a command exits with a non-zero status
set -e 

echo "========================================"
echo " Starting Cloud Benchmark Pipeline"
echo " Target Device: ${DEVICE_NAME}"
echo "========================================"
date

# 1. Ensure Logs Directory Exists
mkdir -p logs

# 2. Initialize Conda for Bash Scripts
if [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniforge3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
else
    # Fallback just in case
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

echo "----------------------------------------"
echo " Phase 1: Environment Setup"
echo "----------------------------------------"
# Assuming these scripts handle their own conda create/install logic
echo "Setting up PtyRAD environment..."
bash envs/create_ptyrad_env.sh

echo "Setting up Phaser environment..."
bash envs/create_phaser_env.sh

# Turn off strict error checking so one failed benchmark doesn't kill the whole script
set +e 

echo "----------------------------------------"
echo " Phase 2: Benchmark Execution"
echo "----------------------------------------"

# --- PTYRAD ---
echo "Starting PtyRAD Benchmark..."
conda activate bench_ptyrad

echo "### Log nvidia-smi and pip list info ###"
nvidia-smi
pip list

# Redirecting both stdout and stderr (2>&1) and append >> to the log file
python -u benchlib/diagnostics.py > logs/log_cloud_${CURRENT_DATE}_ptyrad_reduce_${DEVICE_NAME}_tBL_WSe2.txt 2>&1
python -u ./runners/run_ptyrad_loop.py \
    --device "${DEVICE_NAME}" \
    --date "${CURRENT_DATE}" \
    --label "ptyrad_reduce" \
    --round_idx 1 \
    --batches 512 256 128 64 32 16 8 4 \
    --pmodes 6 \
    --slices 6 \
    --niter 10 \
    --save 10 \
    --compile 'reduce-overhead'\
    >> logs/log_cloud_${CURRENT_DATE}_ptyrad_reduce_${DEVICE_NAME}_tBL_WSe2.txt 2>&1
date

echo "PtyRAD Benchmark Complete."
conda deactivate

# --- PHASER ---
echo "Starting Phaser Benchmark..."
conda activate bench_phaser

echo "### Log nvidia-smi and pip list info ###"
nvidia-smi
pip list

# Redirecting both stdout and stderr (2>&1) and append >> to the log file
python benchlib/diagnostics.py > logs/log_cloud_${CURRENT_DATE}_phaser_${DEVICE_NAME}_tBL_WSe2.txt 2>&1
python -u ./runners/run_phaser_loop.py \
    --device "${DEVICE_NAME}" \
    --date "${CURRENT_DATE}" \
    --label "phaser" \
    --round_idx 1 \
    --batches 512 256 128 64 32 16 8 4 \
    --pmodes 6 \
    --slices 6 \
    --niter 10 \
    --save 10 \
    >> logs/log_cloud_${CURRENT_DATE}_phaser_${DEVICE_NAME}_tBL_WSe2.txt 2>&1

echo "Phaser Benchmark Complete."
conda deactivate

echo "========================================"
echo " All Benchmarks Finished!"
echo "========================================"
date
echo "Reminder: Don't forget to rsync the 'logs/' directory back to your local machine before destroying the instance!"