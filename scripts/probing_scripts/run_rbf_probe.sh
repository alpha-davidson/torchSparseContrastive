#!/bin/bash
#SBATCH --job-name "O16_RBF_KERNEL_PROBE"
#SBATCH --mem 32G
#SBATCH --output=logs/rbf_probe_copy_%j.log

source activate contrastive
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"

cd ../torchSparseContrastive

python -u -m src.evaluation.probing_files.rbf_probe \
    --name O16_simclr_best \
    --seed 0 \
    --c 1.0 \
    --gamma scale \
    --cache-size 4096 \
    --min-train-size 50 \
    --max-train-size 1475 \
    --num-size-points 20 \
    --cv-folds 3 \
    embeddings/O16_combined_ar46_c16_mg22/latent_vectors.npy \
    embeddings/O16_combined_ar46_c16_mg22/labels.npy
