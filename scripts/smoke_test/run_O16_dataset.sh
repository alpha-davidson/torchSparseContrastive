#!/bin/bash
#SBATCH --job-name="O16_DATASET"
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/o16_dataset_%j.log

source activate contrastive

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

mkdir -p logs data

python -u -m src.data.dataset_loaders.o16_dataset
