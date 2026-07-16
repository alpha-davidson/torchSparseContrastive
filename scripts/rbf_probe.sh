#!/bin/bash
#SBATCH --job-name "O16_RBF_KERNEL_PROBE"
#SBATCH --mem 32G
#SBATCH --output=logs/rbf_probe_copy_%j.log

source activate contrastive
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

python -u src/evaluation/rbf_probe.py \
    --name O16_simclr_best \
    --seed 0 \
    --c 1.0 \
    --gamma scale \
    --cache-size 4096 \
    --min-train-size 50 \
    --max-train-size 1475 \
    --num-size-points 20 \
    --cv-folds 3 \
    embeddings/O16_simclr_best/latent_vectors.npy \
    embeddings/O16_simclr_best/labels.npy
