#!/bin/bash
#SBATCH --job-name "O16_SIMCLR"
#SBATCH --mem 32G
#SBATCH --gpus rtx_a6000:1
#SBATCH --output=logs/train_contrastive_%j.log

source activate contrastive

# Match the toolchain used for PyTorch and the in-place TorchSparse backend.
type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load CUDA/11.8.0
module load GCC/11.3.0

export PYTHONNOUSERSITE=1

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

# --- Paths ---
# IF YOU WOULD LIKE TO RESUME FROM A CHECKPOINT, SET RESUME TO THE CHECKPOINT PATH (e.g. checkpoints/best.pt)
DATA="data/O16_w_event_keys.npy"
LENS="data/O16_event_lens.npy"
SAVE_DIR="checkpoints"
RESUME="checkpoints/best.pt"

# --- Training ---
EPOCHS=10
BATCH_SIZE=16
LR=3e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=10
NUM_WORKERS=0
VOXEL_SIZE=0.00390625 # 1/256
HASH_RSV_RATIO=8
# Set to a positive integer for a short smoke test.
MAX_BATCHES=""

# --- Model ---
# IN_CHANNELS=1 because feature is amplitude A only
IN_CHANNELS=1
PROJ_OUT_DIM=128
PROJ_HIDDEN_DIM=512
TEMPERATURE=0.1

mkdir -p logs
LOG="logs/train_contrastive_$(date +%Y%m%d_%H%M%S).log"

python -u -m src.training.train_contrastive \
    --data              "$DATA" \
    --lens              "$LENS" \
    --save-dir          "$SAVE_DIR" \
    --epochs            $EPOCHS \
    --batch-size        $BATCH_SIZE \
    --lr                $LR \
    --weight-decay      $WEIGHT_DECAY \
    --grad-clip         $GRAD_CLIP \
    --save-every        $SAVE_EVERY \
    --num-workers       $NUM_WORKERS \
    --voxel-size        $VOXEL_SIZE \
    --hash-rsv-ratio    $HASH_RSV_RATIO \
    ${MAX_BATCHES:+--max-batches "$MAX_BATCHES"} \
    --in-channels       $IN_CHANNELS \
    --proj-out-dim      $PROJ_OUT_DIM \
    --proj-hidden-dim   $PROJ_HIDDEN_DIM \
    --temperature       $TEMPERATURE \
    ${RESUME:+--resume "$RESUME"} \
    2>&1 | tee "$LOG"
