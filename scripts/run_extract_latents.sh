#!/bin/bash
#SBATCH --job-name "O16_LATENTS"
#SBATCH --mem 32G
#SBATCH --gpus rtx_a6000:1
#SBATCH --output=logs/extract_latents_%j.log

source activate contrastive

# Match the toolchain used for PyTorch and the in-place TorchSparse backend.
type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load CUDA/11.8.0
module load GCC/11.3.0

export PYTHONNOUSERSITE=1

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

# --- Checkpoint / output ---
CHECKPOINT="checkpoints/best.pt"
CONFIG="checkpoints/run_config.json"
OUTPUT_DIR="embeddings/O16_simclr_best"

# --- Extraction mode ---
# Use split mode for labeled latent vectors after O16_downstream_pipeline.py
# creates data/O16_size512_{train,val,test}.npy.
# Set USE_SPLITS=0 to extract from raw O16_w_event_keys.npy with labels=-1.
USE_SPLITS=1
SPLIT_DIR="data"
SAMPLE_SIZE=512

# --- Raw-mode paths ---
DATA="data/O16_w_event_keys.npy"
LENS="data/O16_event_lens.npy"
MIN_HITS=10

# --- Runtime overrides ---
# Empty values fall back to checkpoints/run_config.json or extract_latents.py defaults.
BATCH_SIZE=""
NUM_WORKERS=0
VOXEL_SIZE=""
HASH_RSV_RATIO=8

mkdir -p logs "$OUTPUT_DIR"
LOG="logs/extract_latents_$(date +%Y%m%d_%H%M%S).log"

MODE_ARGS=()
if [[ "$USE_SPLITS" == "1" ]]; then
    MODE_ARGS+=(--split-dir "$SPLIT_DIR" --sample-size "$SAMPLE_SIZE")
else
    MODE_ARGS+=(--no-splits --data "$DATA" --lens "$LENS" --min-hits "$MIN_HITS")
fi

python -u -m src.evaluation.extract_latents \
    --checkpoint       "$CHECKPOINT" \
    --config           "$CONFIG" \
    --output-dir       "$OUTPUT_DIR" \
    --num-workers      "$NUM_WORKERS" \
    --hash-rsv-ratio   "$HASH_RSV_RATIO" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${VOXEL_SIZE:+--voxel-size "$VOXEL_SIZE"} \
    "${MODE_ARGS[@]}" \
    2>&1 | tee "$LOG"
