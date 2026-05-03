#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --account=gratis
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/storage/homefs/as26q834/RobustAnimalCounting/sbatch_scripts/logs/%x_%j.logs
#SBATCH --job-name=train_csrnet_waid

PYTHON_SCRIPT=/storage/homefs/as26q834/RobustAnimalCounting/scripts/train/csrnet/train_waid.py

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate animal_counting

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /storage/homefs/as26q834/RobustAnimalCounting

echo "Copy data to local scratch..."
DATA_ROOT=/storage/homefs/as26q834/RobustAnimalCounting/data/splits/waid
LOCAL_DATA_ROOT=$TMPDIR/data/splits/waid
mkdir -p $LOCAL_DATA_ROOT
cp -r $DATA_ROOT/* $LOCAL_DATA_ROOT/

PYTHONPATH=src python $PYTHON_SCRIPT --override_data_root $LOCAL_DATA_ROOT
