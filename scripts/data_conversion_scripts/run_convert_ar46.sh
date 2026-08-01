#!/bin/bash
#SBATCH --job-name="AR46_CONVERT"
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/convert_AR46_%j.log

source activate contrastive

cd ../torchSparseContrastive

mkdir -p logs data

python -u -m src.data.data_conversion_files.Ar46_convert
