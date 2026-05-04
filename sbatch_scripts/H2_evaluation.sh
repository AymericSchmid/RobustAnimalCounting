#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --partition=cpu-invest
#SBATCH --qos=job_cpu_preemptable
#SBATCH --account=gratis
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=/storage/homefs/as26q834/RobustAnimalCounting/sbatch_scripts/logs/%x_%j.logs
#SBATCH --job-name=h2_evaluation

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate animal_counting

echo "=============================================================="
echo "CSRNet Fine-tuned on Qian Penguins evaluating on Qian Penguins"
echo "=============================================================="

python scripts/eval/csrnet/evaluate.py \
    --train-dataset qian_penguins --test-dataset qian_penguins \
    --weights results/csrnet/qian_penguins/best.pth --mode density

echo "=============================================================="
echo "YoloV8 Fine-tuned on Qian Penguins evaluating on Qian Penguins"
echo "=============================================================="

python scripts/eval/yolov8/evaluate.py \
    --train-dataset qian_penguins --test-dataset qian_penguins \
    --weights results/yolov8/qian_penguins/weights/best.pt --mode density