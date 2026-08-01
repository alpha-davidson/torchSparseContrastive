#!/bin/bash
#SBATCH --job-name "O16_DataDownstream"
#SBATCH --mem 32G
#SBATCH --output=logs/O16_DataDownstream_%j.log

source activate contrastive

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

mkdir -p logs data

python -u -m src.data.legacy.O16_downstream_pipeline
