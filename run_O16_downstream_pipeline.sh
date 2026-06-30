#!/bin/bash
#SBATCH --job-name "O16_DataDownstream"
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --output "O16_DataDownstream-out.log"      # output file
#SBATCH --error "O16_DataDownstream-err.log"       # error message file

python3 O16_downstream_pipeline.py 