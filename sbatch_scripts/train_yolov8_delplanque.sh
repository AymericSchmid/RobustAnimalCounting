#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/storage/homefs/as26q834/RobustAnimalCounting/sbatch_scripts/logs/%x_%j.logs
#SBATCH --job-name=train_yolov8_eikelboom

PYTHON_SCRIPT=/storage/homefs/as26q834/RobustAnimalCounting/scripts/train/yolov8/train.py

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate animal_counting

python $PYTHON_SCRIPT --dataset delplanque