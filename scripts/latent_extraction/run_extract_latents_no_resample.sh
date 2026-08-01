#!/bin/bash
#SBATCH --job-name="O16_COMBINED_LATENTS"
#SBATCH --mem=32G
#SBATCH --gpus=rtx_a6000:1
#SBATCH --output=logs/extract_combined_latents_%j.log

source activate contrastive

type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load CUDA/11.8.0
module load GCC/11.3.0

export PYTHONNOUSERSITE=1

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

# Combined-pretraining checkpoint; downstream extraction uses the canonical
# labeled variable-length O16 splits for an apples-to-apples probe comparison.
CHECKPOINT="checkpoints/combined_o16_ar46_c16_mg22/best.pt"
CONFIG="checkpoints/combined_o16_ar46_c16_mg22/run_config.json"
OUTPUT_DIR="embeddings/O16_combined_ar46_c16_mg22"

USE_SPLITS=1
SPLIT_DIR="data"

# Used only if USE_SPLITS=0.
DATA="data/O16_w_event_keys.npy"
LENS="data/O16_event_lens.npy"
MIN_HITS=10

# Empty values fall back to the training config or extractor defaults.
BATCH_SIZE=""
NUM_WORKERS=0
VOXEL_SIZE=""
HASH_RSV_RATIO=8

mkdir -p logs "$OUTPUT_DIR"
LOG="logs/extract_combined_latents_$(date +%Y%m%d_%H%M%S).log"

MODE_ARGS=()
if [[ "$USE_SPLITS" == "1" ]]; then
    MODE_ARGS=(--split-dir "$SPLIT_DIR")
else
    MODE_ARGS=(
        --no-splits
        --data "$DATA"
        --lens "$LENS"
        --min-hits "$MIN_HITS"
    )
fi

BATCH_ARGS=()
if [[ -n "$BATCH_SIZE" ]]; then
    BATCH_ARGS=(--batch-size "$BATCH_SIZE")
fi

VOXEL_ARGS=()
if [[ -n "$VOXEL_SIZE" ]]; then
    VOXEL_ARGS=(--voxel-size "$VOXEL_SIZE")
fi

python -u -m src.evaluation.extract_latents_no_resample \
    --checkpoint       "$CHECKPOINT" \
    --config           "$CONFIG" \
    --output-dir       "$OUTPUT_DIR" \
    --num-workers      "$NUM_WORKERS" \
    --hash-rsv-ratio   "$HASH_RSV_RATIO" \
    "${BATCH_ARGS[@]}" \
    "${VOXEL_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    2>&1 | tee "$LOG"
