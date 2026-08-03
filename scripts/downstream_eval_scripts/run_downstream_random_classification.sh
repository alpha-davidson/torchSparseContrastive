#!/bin/bash
#SBATCH --job-name="ATTPC_RANDOM_BASELINE"
#SBATCH --mem=32G
#SBATCH --gpus=rtx_a6000:1
#SBATCH --output=logs/downstream_random_%j.log

source activate contrastive

type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load CUDA/11.8.0
module load GCC/11.3.0

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

cd ../torchSparseContrastive

DATA_DIR="data"
PREFIX="O16_UNSAMPLED"
CONFIG="checkpoints/combined_o16_ar46_c16_mg22/run_config.json"
OUTPUT_DIR="results/downstream"

TRIALS=3
EPOCHS=100
BATCH_SIZE=16
NUM_WORKERS=0

mkdir -p logs "$OUTPUT_DIR" "$MPLCONFIGDIR"

python -u -m src.downstream.benchmark.classification_no_pretrain \
    --data-dir       "$DATA_DIR" \
    --prefix         "$PREFIX" \
    --config         "$CONFIG" \
    --output-dir     "$OUTPUT_DIR" \
    --trials         "$TRIALS" \
    --epochs         "$EPOCHS" \
    --batch-size     "$BATCH_SIZE" \
    --num-workers    "$NUM_WORKERS"
