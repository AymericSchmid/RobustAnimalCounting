# RobustAnimalCounting

animal-counting/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── datasets/
│   │   ├── eikelboom.yaml
│   │   ├── qian_penguins.yaml
│   │   ├── waid.yaml
│   │   ├── delplanque.yaml
│   │   └── aed.yaml
│   ├── models/
│   │   ├── yolov8.yaml
│   │   ├── csrnet.yaml
│   │   └── p2pnet.yaml
│   ├── experiments/
│   │   ├── in_domain/
│   │   ├── cross_domain/
│   │   └── density_buckets/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_annotation_conversion.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_cross_domain.py
│   └── make_density_buckets.py
├── src/
│   └── animal_counting/
│       ├── __init__.py
│       ├── datasets/
│       │   ├── base.py
│       │   ├── eikelboom.py
│       │   ├── qian_penguins.py
│       │   ├── waid.py
│       │   ├── delplanque.py
│       │   ├── aed.py
│       │   ├── transforms.py
│       │   └── converters.py
│       ├── models/
│       │   ├── detection/
│       │   │   ├── yolov8_wrapper.py
│       │   │   └── utils.py
│       │   ├── density/
│       │   │   ├── csrnet.py
│       │   │   ├── density_maps.py
│       │   │   └── losses.py
│       │   ├── transformer/
│       │   │   ├── p2pnet.py
│       │   │   └── matcher.py
│       │   └── common/
│       │       ├── backbones.py
│       │       └── checkpoints.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── loops.py
│       │   ├── optimizers.py
│       │   └── early_stopping.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── detection_metrics.py
│       │   ├── counting_metrics.py
│       │   ├── cross_domain.py
│       │   └── density_bucket_eval.py
│       ├── experiments/
│       │   ├── runner.py
│       │   └── registry.py
│       ├── visualization/
│       │   ├── predictions.py
│       │   ├── density_maps.py
│       │   └── plots.py
│       └── utils/
│           ├── io.py
│           ├── logging.py
│           ├── seed.py
│           └── config.py
├── outputs/
│   ├── models/
│   ├── logs/
│   ├── predictions/
│   └── figures/
├── tests/
│   ├── test_datasets.py
│   ├── test_metrics.py
│   ├── test_density_maps.py
│   └── test_splits.py
└── docs/
    └── project_notes.md
---

## Current Status

### Datasets

| Dataset | Annotations | Wrapper | Preprocessed | YOLO-converted | Status |
|---------|-------------|---------|-------------|----------------|--------|
| Eikelboom | 4,305 boxes | ✅ `eikelboom.py` | ✅ `preprocess_eikelboom.py` | ✅ | Ready |
| Delplanque | 10,239 boxes | ✅ `delplanque.py` | ✅ `preprocess_delplanque.py` | ✅ | Ready |
| WAID | 233,806 boxes | ✅ `waid.py` | ✅ `preprocess_waid.py` | ✅ | Ready |
| Qian Penguins | 134,767 points | ✅ `qian_penguins.py` | ✅ `preprocess_qian.py` | ✅ | Ready |
| AED | ~15,581 points | ⬚ | ⬚ | ⬚ | Deprioritized |

### Models

**Note: P2PNet has been dropped due to project timeline.** The comparison is now between YOLOv8 (detection) and CSRNet (density map) across the datasets.

| Model | Paradigm | Wrapper | Training script | Sbatch script | Status |
|-------|----------|---------|----------------|---------------|--------|
| YOLOv8 | Detection | ✅ `models/yolov8.py` | ✅ `train/yolov8/train_eikelboom.py` | ✅ | Working |
| CSRNet | Density map | ✅ `models/csrnet.py` | ✅ `train/csrnet/train_qian.py` | ✅ | Ready to train |
| P2PNet | Transformer | — | — | — | **Dropped** |

### Evaluation

The `evaluation/` module is fully implemented:
- `counting_metrics.py` — MAE, RMSE, relative error
- `density_buckets.py` — splits images into sparse (≤10), medium (10–50), crowded (>50) buckets
- `density_map_metrics.py` — SSIM between predicted and ground-truth density maps
- `paradigm_runners.py` — `evaluate_csrnet_density`, `evaluate_yolo_density`, `evaluate_yolo_cross`, `evaluate_csrnet_cross`
- `scripts/eval/yolov8/evaluate.py` — ready-to-run CLI evaluation script

`YOLOv8CountingModel.predict()` is also now fully implemented.

---

## What remains to do

### Must-do before results

- [ ] **Run the smoke-test notebook** (`notebooks/2_test_csrnet_pipeline.ipynb`) on the cluster to confirm everything works before submitting jobs
- [ ] **Train YOLOv8** on Eikelboom and Delplanque — submit the existing sbatch scripts
- [ ] **Train CSRNet** on Qian Penguins — submit `sbatch_scripts/train_csrnet_qian.sh`
- [ ] **Run per-density evaluation** for YOLOv8 (script ready: `scripts/eval/yolov8/evaluate.py --mode density`)
- [ ] **Run cross-domain evaluation** for YOLOv8 (same script: `--mode cross`, e.g. Eikelboom → Delplanque)
- [ ] **Run CSRNet evaluation** — script is ready at `scripts/eval/csrnet/evaluate.py` (see cluster workflow below)

### Nice to have

- [ ] **CSRNet on box-annotation datasets** — `DensityMapDataset` already supports box→centroid conversion; a training script for Eikelboom/WAID would enable cross-domain CSRNet experiments
- [ ] **Results notebook** — visualize density maps, detection boxes, and per-bucket bar charts for the paper

---

## Cluster workflow (step by step)

Everything GPU-related runs on the cluster via Slurm. All commands below assume you are in the repo root.

### 0 · Pull the latest code

```bash
git pull origin main
conda activate animal_counting
```

### 1 · Verify everything works before submitting jobs

Open `notebooks/2_test_csrnet_pipeline.ipynb`, set `ROOT` in the first cell to your cluster path, then run all cells top to bottom. Every cell should print ✓ or display a plot. If anything fails, do not submit training jobs yet.

### 2 · Submit training jobs

```bash
sbatch sbatch_scripts/train_csrnet_qian.sh         # CSRNet on Qian Penguins (~12 h)
sbatch sbatch_scripts/train_yolov8_eikelboom.sh    # YOLOv8 on Eikelboom    (~5 h)
sbatch sbatch_scripts/train_yolov8_delplanque.sh   # YOLOv8 on Delplanque   (~5 h)
```

Monitor progress:

```bash
squeue -u $USER                                         # see running jobs
tail -f sbatch_scripts/logs/<job_name>_<job_id>.logs    # live output
```

CSRNet prints `val_MAE` and `val_RMSE` every epoch and saves `results/csrnet/qian_penguins/best.pth` whenever the MAE improves. It stops automatically after 20 epochs without improvement.

### 3 · Run evaluation (after training completes)

#### YOLOv8 — in-domain with density buckets (for H1)

```bash
PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset eikelboom \
    --test-dataset  eikelboom \
    --weights results/yolov8/eikelboom/weights/best.pt \
    --mode density
```

#### YOLOv8 — cross-domain (for H4)

```bash
PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset eikelboom \
    --test-dataset  delplanque \
    --weights results/yolov8/eikelboom/weights/best.pt \
    --mode cross
```

Results are saved as JSON in `results/yolov8/`. The script prints a summary table of MAE, RMSE, relative error, mAP@0.5, precision, and recall — overall and per density bucket.

#### CSRNet — in-domain with density buckets (for H2)

```bash
PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \
    --train-dataset qian_penguins \
    --test-dataset  qian_penguins \
    --weights results/csrnet/qian_penguins/best.pth \
    --mode density
```

#### CSRNet — cross-domain (for H4)

```bash
PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \
    --train-dataset qian_penguins \
    --test-dataset  eikelboom \
    --weights results/csrnet/qian_penguins/best.pth \
    --mode cross
```

Results are saved as JSON in `results/csrnet/`. The script prints MAE, RMSE, relative error, and SSIM — overall and per density bucket.

**Memory note:** large aerial images are automatically resized to fit within 1024px on the longest side before inference (the `--max-size` flag, default 1024). Dimensions are then rounded up to the nearest multiple of 8, which is a hard requirement from the three VGG max-pool layers. Pass `--max-size 0` to use full resolution if GPU memory allows.

**How GT density maps are generated:** the script builds them on-the-fly from point annotations (or bounding-box centroids for box-only datasets) using the same adaptive Gaussian formula as during training (`σ = 0.3 × mean distance to 3 nearest neighbours`). They are then downsampled by 8 to match the network output size before SSIM is computed.

---

## Setup from scratch

### Environment

```bash
conda env create -f environment.yml
conda activate animal_counting
```

### Important: PYTHONPATH

The package is not installed via pip. Always prefix scripts with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/datasets_processing/convert_dataset.py ...
```

### Dataset pipeline

Every dataset goes through the same three steps.

**1 · Download** into `data/raw/<name>/`

```bash
python scripts/datasets_processing/download_datasets.py
# Some datasets need manual download — see comments in datasets_list.py
```

**2 · Preprocess** — creates `data/splits/<name>/annotations.csv` and copies images

```bash
python scripts/datasets_processing/preprocess_eikelboom.py
python scripts/datasets_processing/preprocess_qian.py
python scripts/datasets_processing/preprocess_waid.py
python scripts/datasets_processing/preprocess_delplanque.py
```

**3 · Convert to YOLO format** (YOLOv8 only — CSRNet does not need this step)

```bash
PYTHONPATH=src python scripts/datasets_processing/convert_dataset.py \
    --dataset eikelboom --format yolo \
    --root data/splits/eikelboom --output data/yolo/eikelboom
```

Repeat with `--dataset delplanque`, `--dataset waid`, `--dataset qian_penguins` as needed.

#### Annotation formats

| Dataset type | File columns | Used by |
|---|---|---|
| Box (Eikelboom, Delplanque, WAID) | `image_path, x1, y1, x2, y2, species, split` | YOLOv8 directly; CSRNet via box centroids |
| Point (Qian Penguins) | `image_path, x, y, species, colony, split` | CSRNet directly; YOLOv8 via synthetic 10×10 px boxes |

---

## Key design notes

- **All species → single class.** Both models treat every animal as class 0/1 — the task is counting, not classification.
- **Density map generation.** Adaptive Gaussian sigma per point: `σ = 0.3 × mean_distance_to_3_nearest_neighbours` (Hamrouni et al. 2020). Single-point images use a fixed σ = 15 px.
- **CSRNet output resolution.** 1/8 of input (VGG pool1 × pool2 × pool3). A 512×512 patch produces a 64×64 density map. `density_map.sum() = predicted count`.
- **Density map downsampling.** Full-resolution GT map is average-pooled by 8 and multiplied by 64 to preserve the integral before computing MSE loss.
- **Box-only datasets with CSRNet.** `DensityMapDataset` automatically computes bounding-box centroids as pseudo-points when no point annotations are available.