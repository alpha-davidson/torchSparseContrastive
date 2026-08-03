#!/bin/bash
#SBATCH --job-name="DOWNSTREAM_REG_PLOTS"
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/downstream_reg_plots_%j.log

source activate contrastive
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}"
export XDG_CACHE_HOME="/tmp/cache-${SLURM_JOB_ID:-manual}"

cd ../torchSparseContrastive

# Replace this with the prefix used when pointwise regression data/evaluation
# is available. O16_UNSAMPLED currently contains event-classification labels,
# not pointwise regression labels.
PREFIX="C16"
RESULTS_DIR="results/downstream"
TITLE='$^{16}$C pointwise kinematics regression'

mkdir -p logs "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python -u -m src.downstream.plotting.regression_comparison_plotting \
    --prefix      "$PREFIX" \
    --results-dir "$RESULTS_DIR" \
    --title       "$TITLE"
