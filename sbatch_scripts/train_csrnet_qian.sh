#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/storage/homefs/as26q834/RobustAnimalCounting/sbatch_scripts/logs/%x_%j.logs
#SBATCH --job-name=train_csrnet_qian

PYTHON_SCRIPT=/storage/homefs/as26q834/RobustAnimalCounting/scripts/train/csrnet/train_qian.py

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate animal_counting

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /storage/homefs/as26q834/RobustAnimalCounting
PYTHONPATH=src python $PYTHON_SCRIPT
