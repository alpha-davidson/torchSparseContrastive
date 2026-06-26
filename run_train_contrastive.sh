#!/bin/bash
#SBATCH --job-name "SPARSE_SIMCLR"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --output=logs/train_contrastive_%j.log

source activate torchsparse

# --- Paths ---
# IF YOU WOULD LIKE TO RESUME FROM A CHECKPOINT, SET RESUME TO THE CHECKPOINT PATH (e.g. checkpoints/best.pt)
DATA="shapenet_simple_large.pt"
SAVE_DIR="checkpoints"
RESUME=""

# --- Training ---
EPOCHS=100
BATCH_SIZE=16
LR=3e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=10
NUM_WORKERS=0

# --- Model ---
# IN_CHANNELS: 3 = xyz only, 6 = xyz + normals
IN_CHANNELS=3
PROJ_OUT_DIM=128
PROJ_HIDDEN_DIM=512
TEMPERATURE=0.1

python train_contrastive.py \
    --data              "$DATA" \
    --save-dir          "$SAVE_DIR" \
    --epochs            $EPOCHS \
    --batch-size        $BATCH_SIZE \
    --lr                $LR \
    --weight-decay      $WEIGHT_DECAY \
    --grad-clip         $GRAD_CLIP \
    --save-every        $SAVE_EVERY \
    --num-workers       $NUM_WORKERS \
    --in-channels       $IN_CHANNELS \
    --proj-out-dim      $PROJ_OUT_DIM \
    --proj-hidden-dim   $PROJ_HIDDEN_DIM \
    --temperature       $TEMPERATURE \
    ${RESUME:+--resume "$RESUME"}
