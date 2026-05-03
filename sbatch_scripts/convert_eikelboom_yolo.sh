#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable
#SBATCH --account=gratis
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --output=/storage/homefs/as26q834/RobustAnimalCounting/sbatch_scripts/logs/%x_%j.logs
#SBATCH --job-name=convert_eikelboom_yolo

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate animal_counting

cd /storage/homefs/as26q834/RobustAnimalCounting
python scripts/datasets_processing/convert_dataset.py \
    --dataset eikelboom --format yolo \
    --root data/splits/eikelboom --output data/yolo/eikelboom
