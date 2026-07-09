#!/bin/bash
#SBATCH --job-name "O16_LINEAR_PROBE"
#SBATCH --mem 32G
#SBATCH --output=logs/probe_copy_%j.log

source activate contrastive
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

python -u src/evaluation/linear_probe.py \
    --name O16_simclr_best \
    --seed 0 \
    --max-train-size 1475 \
    embeddings/O16_simclr_best/latent_vectors.npy \
    embeddings/O16_simclr_best/labels.npy
