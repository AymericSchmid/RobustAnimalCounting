#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable
#SBATCH --account=gratis
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --output=/storage/homefs/as26q834/RobustAnimalCounting/sbatch_scripts/logs/%x_%j.logs
#SBATCH --job-name=preprocess_delplanque

PYTHON_SCRIPT=/storage/homefs/as26q834/RobustAnimalCounting/scripts/datasets_processing/preprocess_delplanque.py

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate animal_counting

cd /storage/homefs/as26q834/RobustAnimalCounting
python $PYTHON_SCRIPT
