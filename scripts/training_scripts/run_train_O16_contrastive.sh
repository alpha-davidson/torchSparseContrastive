#!/bin/bash
#SBATCH --job-name="ATTPC_SIMCLR"
#SBATCH --mem=32G
#SBATCH --gpus=rtx_a6000:1
#SBATCH --output=logs/train_ATTPC_%j.log

source activate contrastive

# Match the toolchain used for PyTorch and the in-place TorchSparse backend.
type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load CUDA/11.8.0
module load GCC/11.3.0

export PYTHONNOUSERSITE=1

cd ../torchSparseContrastive

# ---------------------------------------------------------------------------
# Dataset selection
# Set DATASET_MODE="o16" for the original loader and "combined" for
# virtual O16+Ar46+C16 training.
# In combined mode, DATA, LENS, and DATASET_NAMES must have matching orders.
# ---------------------------------------------------------------------------

DATASET_MODE="combined"

if [[ "$DATASET_MODE" == "o16" ]]; then
    DATA=("data/O16_w_event_keys.npy")
    LENS=("data/O16_event_lens.npy")
    DATASET_NAMES=()
elif [[ "$DATASET_MODE" == "combined" ]]; then
    DATA=(
        "data/O16_w_event_keys.npy"
        "data/Ar46_w_event_keys.npy"
        "data/C16_w_event_keys.npy"
        "data/Mg22_w_event_keys.npy"
    )
    LENS=(
        "data/O16_event_lens.npy"
        "data/Ar46_event_lens.npy"
        "data/C16_event_lens.npy"
        "data/Mg22_event_lens.npy"
    )
    # These names select columns in data/dataset_loaders/combined_dataset.py:
    # O16 amplitude=4, Ar46 charge=3, C16 charge=3.
    DATASET_NAMES=("O16" "Ar46" "C16" "Mg22")
else
    echo "Unknown DATASET_MODE: $DATASET_MODE" >&2
    exit 2
fi

# Set RESUME to a checkpoint path to resume training.
if [[ "$DATASET_MODE" == "combined" ]]; then
    SAVE_DIR="checkpoints/combined_o16_ar46_c16_mg22"
else
    SAVE_DIR="checkpoints/o16_combined"
fi
RESUME=""

# Training
EPOCHS=100
BATCH_SIZE=16
LR=3e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=10
NUM_WORKERS=0
VOXEL_SIZE=0.00390625  # 1/256
HASH_RSV_RATIO=8
# Set to a positive integer for a short smoke test.
MAX_BATCHES=""

# Model
IN_CHANNELS=1
PROJ_OUT_DIM=128
PROJ_HIDDEN_DIM=512
TEMPERATURE=0.1

mkdir -p logs "$SAVE_DIR"
LOG="logs/train_${DATASET_MODE}_$(date +%Y%m%d_%H%M%S).log"

DATASET_NAME_ARGS=()
if [[ "$DATASET_MODE" == "combined" ]]; then
    DATASET_NAME_ARGS=(--dataset-names "${DATASET_NAMES[@]}")
fi

MAX_BATCH_ARGS=()
if [[ -n "$MAX_BATCHES" ]]; then
    MAX_BATCH_ARGS=(--max-batches "$MAX_BATCHES")
fi

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
    RESUME_ARGS=(--resume "$RESUME")
fi

python -u -m src.training.train_contrastive \
    --dataset          "$DATASET_MODE" \
    --data             "${DATA[@]}" \
    --lens             "${LENS[@]}" \
    "${DATASET_NAME_ARGS[@]}" \
    --save-dir         "$SAVE_DIR" \
    --epochs           "$EPOCHS" \
    --batch-size       "$BATCH_SIZE" \
    --lr               "$LR" \
    --weight-decay     "$WEIGHT_DECAY" \
    --grad-clip        "$GRAD_CLIP" \
    --save-every       "$SAVE_EVERY" \
    --num-workers      "$NUM_WORKERS" \
    --voxel-size       "$VOXEL_SIZE" \
    --hash-rsv-ratio   "$HASH_RSV_RATIO" \
    "${MAX_BATCH_ARGS[@]}" \
    --in-channels      "$IN_CHANNELS" \
    --proj-out-dim     "$PROJ_OUT_DIM" \
    --proj-hidden-dim  "$PROJ_HIDDEN_DIM" \
    --temperature      "$TEMPERATURE" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "$LOG"
