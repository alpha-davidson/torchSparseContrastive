#!/bin/bash
#SBATCH --job-name "O16_DataDownstream_No_Resample"
#SBATCH --mem 32G
#SBATCH --output=logs/O16_DataDownstream_No_Resample_%j.log

source activate contrastive

cd /home/DAVIDSON/tomallenntiador/torchSparseContrastive

mkdir -p logs data

python -u -m src.data.downstream_data_loader.O16_downstream_no_resample
