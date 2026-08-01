#!/bin/bash
#SBATCH --job-name="O16_CONVERT"
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/convert_O16_%j.log

source activate contrastive

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

mkdir -p logs data

python -u src/data/data_conversion_files/convert-data.py
