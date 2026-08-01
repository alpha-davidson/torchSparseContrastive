#!/bin/bash
#SBATCH --job-name="DOWNSTREAM_ANALYSIS"
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/downstream_analysis_%j.log

source activate contrastive
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/cache-${SLURM_JOB_ID:-manual}"

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

TASK="classification"
PREFIX="O16_UNSAMPLED"
RESULTS_DIR="results/downstream"

mkdir -p logs "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python -u -m src.downstream.plotting.benchmark_analysis \
    --task        "$TASK" \
    --prefix      "$PREFIX" \
    --results-dir "$RESULTS_DIR"
