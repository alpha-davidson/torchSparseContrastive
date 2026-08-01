#!/bin/bash
#SBATCH --job-name "O16_LINEAR_PROBE"
#SBATCH --mem 32G
#SBATCH --output=logs/linear_probe_copy_%j.log

source activate contrastive
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

python -u -m src.evaluation.probing_files.linear_probe \
    --name O16_simclr_best \
    --seed 0 \
    --max-train-size 1475 \
    embeddings/O16_combined_ar46_c16_mg22/latent_vectors.npy \
    embeddings/O16_combined_ar46_c16_mg22/labels.npy
