#!/bin/bash
#SBATCH --job-name="MG22_CONVERT"
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/convert_MG22_%j.log

source activate contrastive

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

mkdir -p logs data

python -u -m src.data.data_conversion_files.mg22_convert
