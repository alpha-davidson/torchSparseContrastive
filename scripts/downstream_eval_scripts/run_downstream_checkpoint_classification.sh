#!/bin/bash
#SBATCH --job-name="ATTPC_DOWNSTREAM"
#SBATCH --mem=32G
#SBATCH --gpus=rtx_a6000:1
#SBATCH --output=logs/downstream_checkpoint_%j.log

source activate contrastive

type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load CUDA/11.8.0
module load GCC/11.3.0

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

cd ../torchSparseContrastive

DATA_DIR="data"
PREFIX="O16_UNSAMPLED"
CHECKPOINT_DIR="checkpoints/combined_o16_ar46_c16_mg22"
CHECKPOINT_PATTERN="epoch_*.pt"
OUTPUT_DIR="results/downstream"

TRIALS=3
EPOCHS=50
FREEZE_EPOCHS=10
BATCH_SIZE=16
NUM_WORKERS=0

# Set to 1 to validate files/configs/backbones without training.
DRY_RUN=0
DRY_RUN_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
    DRY_RUN_ARGS=(--dry-run)
fi

mkdir -p logs "$OUTPUT_DIR" "$MPLCONFIGDIR"

python -u -m src.downstream.training.training_pipeline \
    --data-dir            "$DATA_DIR" \
    --prefix              "$PREFIX" \
    --checkpoint-dir      "$CHECKPOINT_DIR" \
    --checkpoint-pattern  "$CHECKPOINT_PATTERN" \
    --output-dir          "$OUTPUT_DIR" \
    --trials              "$TRIALS" \
    --epochs              "$EPOCHS" \
    --freeze-epochs       "$FREEZE_EPOCHS" \
    --batch-size          "$BATCH_SIZE" \
    --num-workers         "$NUM_WORKERS" \
    "${DRY_RUN_ARGS[@]}"
