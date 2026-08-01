#!/bin/bash
#SBATCH --job-name="DOWNSTREAM_CLASS_PLOTS"
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/downstream_class_plots_%j.log

source activate contrastive
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/cache-${SLURM_JOB_ID:-manual}"

cd ../torchSparseContrastive

PREFIX="O16_UNSAMPLED"
RESULTS_DIR="results/downstream"
TITLE='$^{16}$O track-count classification'

mkdir -p logs "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python -u -m src.downstream.plotting.comparison_plotting \
    --prefix      "$PREFIX" \
    --results-dir "$RESULTS_DIR" \
    --title       "$TITLE"
