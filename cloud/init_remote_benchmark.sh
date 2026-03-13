#!/bin/bash
# init_remote_benchmark.sh
# Master script to launch the benchmark on remote machines

# Run this from your LOCAL machine (e.g., your laptop or workstation)
# This will (1) connect to the fresh remote host rented from cloud services like vast.ai, 
# (2) push local working directory of `ptycho_benchmark` (data and scripts) to it,
# then (3) trigger the `run_remote_benchmark.sh` and terminate the ssh session.
# The `run_remote_benchmark.sh` script will create environments and then perform benchmarks.

if [ "$#" -ne 3 ]; then
    echo "❌ Usage: bash ./cloud/init_remote_benchmark.sh <REMOTE_IP> <REMOTE_PORT> <DEVICE_NAME>"
    echo "Example: bash ./cloud/init_remote_benchmark.sh 192.168.1.1 23456 RTX_4090"
    exit 1
fi

REMOTE_IP=$1
REMOTE_PORT=$2
DEVICE_NAME=$3

# Vast.ai usually defaults to the 'root' user in their containers
REMOTE_USER="root"
REMOTE_DIR="/workspace/ptycho_benchmark"

echo "========================================"
echo " 1. Pushing repository to remote host machine..."
echo "========================================"
# -a: archive mode (preserves permissions/times)
# -v: verbose
# -z: compress data during transfer
# --exclude: don't upload heavy, unnecessary local files
rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'data/PSO' \
    --exclude 'figures' \
    --exclude 'logs' \
    --exclude 'notebooks' \
    --exclude 'output' \
    --exclude 'params' \
    ./ ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/

echo "========================================"
echo " 2. Triggering remote benchmark execution..."
echo "========================================"
# We use nohup and & to run the script in the background. 
# This means if your local SSH connection drops, the benchmark keeps running!
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_IP} << EOF
    cd ${REMOTE_DIR}
    
    # Create logs/
    mkdir -p logs

    # Make the cloud script executable
    chmod +x ./cloud/run_remote_benchmark.sh
    
    # Run in the background and detach
    nohup bash ./cloud/run_remote_benchmark.sh ${DEVICE_NAME} > logs/nohup_master_log_${DEVICE_NAME}.txt 2>&1 &
    
    echo "✅ Benchmark triggered successfully in the background! (PID: \$!)"
EOF

echo "========================================"
echo " Pipeline Initiated!"
echo "========================================"
echo "The benchmarks are now running on the cloud."
echo "To check the live progress, SSH into the machine and run:"
echo "  tail -f ${REMOTE_DIR}/logs/nohup_master_log_${DEVICE_NAME}.txt"
echo ""
echo "When it finishes, run this command to pull your logs back down:"
echo "  rsync -avz -e 'ssh -p ${REMOTE_PORT}' ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/logs/ ./logs/"