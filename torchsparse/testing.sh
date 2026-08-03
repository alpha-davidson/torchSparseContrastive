#!/bin/bash
#SBATCH --job-name=compile_torchsparse
#SBATCH --output=torchsparse_compile_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -e
echo "=== Job Started ==="
# 1. Initialize and activate the main environment
source /opt/pub/apps/generic/Miniconda3/24.3.0-0/etc/profile.d/conda.sh
conda activate Torch
# 2. Extract environment paths dynamically
ENV_PYTHON=$(which python)
CONDA_PREFIX=$(dirname $(dirname $ENV_PYTHON))
# 3. Explicitly lock down the CUDA and library paths
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/include/sparsehash:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# --- COMPILER OVERRIDE: Point to the Isolated GCC 11 Bucket ---
# REMINDER: Change 'noguggenheimer' to your specific username if needed!
export CC=/home/DAVIDSON/tomallenntiador/gcc11_bin/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/DAVIDSON/tomallenntiador/gcc11_bin/bin/x86_64-conda-linux-gnu-g++
echo "Using Secret C Compiler: $CC"
echo "Using Secret C++ Compiler: $CXX"
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"
# 4. Inject the compiler bypass flags to ignore NVCC version warnings
export EXTRA_NVCCFLAGS="-allow-unsupported-compiler"
export NVCC_FLAGS="-allow-unsupported-compiler"
# 5. Restrict parallel jobs to prevent crashing cluster node memory
export MAX_JOBS=2
# 6. Navigate to your repository folder
cd /home/DAVIDSON/tomallenntiador/SparseTPCNet/torchsparse
echo "=== Wiping workspace cache ==="
$ENV_PYTHON setup.py clean --all
rm -rf build/ dist/ *.egg-info
echo "=== Launching Pip Compilation ==="
$ENV_PYTHON -m pip install --no-build-isolation --no-cache-dir .
echo "=== Running Post-Build Diagnostic ==="
$ENV_PYTHON -c "import torch; import torchsparse; print('System Verification: Fully
Operational')"