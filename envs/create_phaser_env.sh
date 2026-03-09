#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# CONFIGURATION
# ==========================================
PACKAGE_NAME="phaser" # Change this to phaser or quantem for the other scripts
ENV_NAME="bench_${PACKAGE_NAME}"
PYTHON_VERSION="3.12"
FORCE=false
LOCAL=false
GIT_REF=""

# ==========================================
# ARGUMENT PARSING
# ==========================================
# A cleaner way to parse arguments, allowing for flags with values
while [[ $# -gt 0 ]]; do
    case $1 in
        --force) 
            FORCE=true
            shift 
            ;;
        --local) 
            LOCAL=true
            shift 
            ;;
        --git) 
            GIT_REF="$2"
            shift 2 # Shift past both the flag and the value
            ;;
        *) 
            shift 
            ;;
    esac
done

# Initialize conda for use within a bash script
eval "$(conda shell.bash hook)"

# ==========================================
# ENVIRONMENT CHECK & CLEANUP
# ==========================================
# Check if the environment already exists by looking for it in the env list
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    if [ "$FORCE" = true ]; then
        echo "⚠️ Environment '${ENV_NAME}' exists. --force flag detected. Removing..."
        conda remove -n "${ENV_NAME}" --all -y
    else
        echo "❌ Error: Environment '${ENV_NAME}' already exists."
        echo "💡 Run the script with --force to delete and recreate it."
        exit 1
    fi
fi

echo "🧹 Purging Conda and Pip caches to ensure a clean build..."
conda clean --all -y
# We run pip cache purge here; appending || true ensures it doesn't fail the script 
# if the base environment's pip cache is already empty or inaccessible.
pip cache purge || true 

# ==========================================
# ENVIRONMENT CREATION
# ==========================================
echo "✨ Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo "🔄 Activating '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# ==========================================
# PACKAGE SPECIFIC INSTALLATION
# ==========================================
echo "📦 Installing dependencies for ${PACKAGE_NAME}..."

pip install --upgrade "jax[cuda12]"
pip install optax

# Get the absolute path of the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ "$LOCAL" = true ]; then
    echo "Installing ${PACKAGE_NAME} from existing local sources"
    # Assuming this script is in ptycho_benchmark/envs/
    # Adjust the relative path below if your repo structure is different
    cd "$SCRIPT_DIR/../../phaser" 
    echo "Current directory is: $(pwd)" 
    pip install -e .

elif [ -n "$GIT_REF" ]; then
    echo "Installing ${PACKAGE_NAME} from Git reference: ${GIT_REF}"
    # Replace with your actual repo URL
    pip install "git+https://github.com/hexane360/phaser.git@${GIT_REF}"

else
    echo "Installing ${PACKAGE_NAME} from PyPI"
    pip install phaserEM

fi

# ==========================================
# PACKAGE SPECIFIC TESTING
# ==========================================

# Using a heredoc to run a multi-line Python script
python <<EOF
import sys

print("--------------------------------------------------")
print("Environment : ${ENV_NAME}")
print("Python Path : " + sys.executable)

try:
    import ${PACKAGE_NAME}
    print("✅ Status     : Successfully imported ${PACKAGE_NAME}!")
    
    # Optional: Try to print the version if the package supports it
    if hasattr(${PACKAGE_NAME}, "__version__"):
        print("🏷️ Version    : " + getattr(${PACKAGE_NAME}, "__version__"))
        
except ImportError as e:
    print("❌ Status     : Failed to import ${PACKAGE_NAME}!")
    print("Details      : " + str(e))
    sys.exit(1)
    
print("--------------------------------------------------")
EOF

echo "✅ Environment '${ENV_NAME}' setup successfully!"
echo "To use it, run: conda activate ${ENV_NAME}"