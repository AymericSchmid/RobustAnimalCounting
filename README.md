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

## Current status

### Datasets

| Dataset | Annotations | Wrapper | Preprocessed | YOLO-converted | Role | Status |
|---------|-------------|---------|-------------|----------------|------|--------|
| Eikelboom | 4,305 boxes | ✅ | ✅ | ✅ | Train + test | Ready |
| Delplanque | 10,239 boxes | ✅ | ✅ | ✅ | Train + test | Ready |
| WAID | 233,806 boxes | ✅ | ✅ | ✅ | **Test only** | Ready |
| Qian Penguins | 134,767 points | ✅ | ✅ | ✅ | Train + test | Ready |
| AED | ~15,581 points | ⬚ | ⬚ | ⬚ | Test only | Deprioritized |

**Why WAID is test-only:** WAID is the "unseen environment" dataset. Training on it would consume the only dataset we have to measure generalization degradation (H4). It stays held out.

### Models

P2PNet has been dropped due to project timeline. The comparison is YOLOv8 (detection) vs CSRNet (density map).

| Model | Paradigm | Wrapper | Status |
|-------|----------|---------|--------|
| YOLOv8 | Detection | ✅ `models/yolov8.py` | Implemented |
| CSRNet | Density map | ✅ `models/csrnet.py` | Implemented |
| P2PNet | Transformer | — | **Dropped** |

### Evaluation module

| File | What it provides |
|------|-----------------|
| `evaluation/counting_metrics.py` | MAE, RMSE, relative error |
| `evaluation/density_buckets.py` | Splits images into sparse (≤10), medium (10–50), crowded (>50) |
| `evaluation/density_map_metrics.py` | SSIM between predicted and GT density maps |
| `evaluation/paradigm_runners.py` | `evaluate_yolo_density/cross`, `evaluate_csrnet_density/cross` |
| `scripts/eval/yolov8/evaluate.py` | CLI for YOLOv8 evaluation |
| `scripts/eval/csrnet/evaluate.py` | CLI for CSRNet evaluation |

---

## Experimental design

### Why these datasets, why these training runs

The project compares two models (YOLOv8, CSRNet) across different density regimes and environments. For a fair comparison, both models need to be trained and tested **on the same data**. That drives the training choices:

| Training run | Script | Why |
|---|---|---|
| YOLOv8 on Eikelboom | `train/yolov8/train_eikelboom.py` | Primary sparse/detection dataset (H1) |
| CSRNet on Eikelboom | `train/csrnet/train_eikelboom.py` | Same data as above so models are directly comparable on H1 |
| CSRNet on Qian Penguins | `train/csrnet/train_qian.py` | Primary dense/crowd dataset (H2) |
| YOLOv8 on Delplanque | `train/yolov8/train_eikelboom.py` (Delplanque version) | Needed to run the reverse cross-domain experiment (H4) |

### How each hypothesis is tested

| Hypothesis | What we measure | Train on | Test on | Eval mode |
|---|---|---|---|---|
| **H1** — Detection best in sparse scenes | MAE in the sparse bucket (1–10 animals), YOLOv8 vs CSRNet | Eikelboom | Eikelboom (test split) | `--mode density` |
| **H2** — Density maps best in crowded scenes | MAE in the crowded bucket (50+ animals), YOLOv8 vs CSRNet | Qian Penguins | Qian Penguins (test split) | `--mode density` |
| **H4** — Cross-domain gap > in-domain gap | In-domain MAE vs MAE on a completely different environment | Eikelboom | WAID, Delplanque | `--mode cross` |
| **H4 (reverse)** | Same, from the other direction | Delplanque | Eikelboom, WAID | `--mode cross` |

> H3 (transformer robustness) was dropped along with P2PNet.

### Full experiment matrix

Once all models are trained, run this full set of evaluations:

| Train | Test | Answers |
|---|---|---|
| YOLOv8 (Eikelboom) | Eikelboom | H1 — in-domain sparse |
| CSRNet (Eikelboom) | Eikelboom | H1 — comparison on same data |
| CSRNet (Qian Penguins) | Qian Penguins | H2 — in-domain crowded |
| YOLOv8 (Eikelboom) | Qian Penguins | H2 — comparison on same data |
| YOLOv8 (Eikelboom) | Delplanque | H4 — cross-domain (same habitat, different species) |
| YOLOv8 (Eikelboom) | WAID | H4 — cross-domain (different environment) |
| CSRNet (Eikelboom) | Delplanque | H4 — CSRNet cross-domain comparison |
| CSRNet (Eikelboom) | WAID | H4 — CSRNet cross-domain comparison |
| YOLOv8 (Delplanque) | Eikelboom | H4 — reverse direction |

---

## What remains to do

- [ ] Run the smoke-test notebook on the cluster (`notebooks/2_test_csrnet_pipeline.ipynb`) before submitting any jobs
- [ ] Train all 4 models — submit sbatch scripts (see cluster workflow below)
- [ ] Run the full experiment matrix above once training is complete
- [ ] Results notebook — visualize density maps, detection boxes, per-bucket bar charts for the paper

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
sbatch sbatch_scripts/train_yolov8_eikelboom.sh    # YOLOv8 on Eikelboom      (~5 h)
sbatch sbatch_scripts/train_yolov8_delplanque.sh   # YOLOv8 on Delplanque     (~5 h)
sbatch sbatch_scripts/train_csrnet_qian.sh         # CSRNet on Qian Penguins  (~12 h)
sbatch sbatch_scripts/train_csrnet_eikelboom.sh    # CSRNet on Eikelboom      (~12 h)
```

All four can run in parallel if enough GPUs are available. Monitor progress:

```bash
squeue -u $USER                                         # see running jobs
tail -f sbatch_scripts/logs/<job_name>_<job_id>.logs    # live output per job
```

CSRNet prints `val_MAE` and `val_RMSE` every epoch and saves `best.pth` whenever the MAE improves. It stops automatically after 20 epochs without improvement.

### 3 · Run the full experiment matrix (after training completes)

Run every command below — each corresponds to one row in the experiment matrix above. Results are saved automatically as JSON files in `results/yolov8/` and `results/csrnet/`.

#### H1 — sparse scene comparison (both models on Eikelboom)

```bash
# YOLOv8 in-domain on Eikelboom, per density bucket
PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset eikelboom --test-dataset eikelboom \
    --weights results/yolov8/eikelboom/weights/best.pt --mode density

# CSRNet in-domain on Eikelboom, per density bucket
PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \
    --train-dataset eikelboom --test-dataset eikelboom \
    --weights results/csrnet/eikelboom/best.pth --mode density
```

#### H2 — crowded scene comparison (both models on Qian Penguins)

```bash
# CSRNet in-domain on Qian Penguins, per density bucket
PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \
    --train-dataset qian_penguins --test-dataset qian_penguins \
    --weights results/csrnet/qian_penguins/best.pth --mode density

# YOLOv8 cross-domain on Qian Penguins (trained on Eikelboom — comparison point)
PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset eikelboom --test-dataset qian_penguins \
    --weights results/yolov8/eikelboom/weights/best.pt --mode density
```

#### H4 — cross-domain generalization

```bash
# YOLOv8 (Eikelboom) → Delplanque and WAID
PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset eikelboom --test-dataset delplanque \
    --weights results/yolov8/eikelboom/weights/best.pt --mode cross

PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset eikelboom --test-dataset waid \
    --weights results/yolov8/eikelboom/weights/best.pt --mode cross

# CSRNet (Eikelboom) → Delplanque and WAID
PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \
    --train-dataset eikelboom --test-dataset delplanque \
    --weights results/csrnet/eikelboom/best.pth --mode cross

PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \
    --train-dataset eikelboom --test-dataset waid \
    --weights results/csrnet/eikelboom/best.pth --mode cross

# Reverse direction: YOLOv8 (Delplanque) → Eikelboom and WAID
PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset delplanque --test-dataset eikelboom \
    --weights results/yolov8/delplanque/weights/best.pt --mode cross

PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \
    --train-dataset delplanque --test-dataset waid \
    --weights results/yolov8/delplanque/weights/best.pt --mode cross
```

**YOLOv8 metrics:** MAE, RMSE, relative error, mAP@0.5, precision, recall — overall and per density bucket.

**CSRNet metrics:** MAE, RMSE, relative error, SSIM — overall and per density bucket.

**CSRNet memory note:** images are automatically resized to fit within 1024px on the longest side before inference (`--max-size` flag, default 1024). Both dimensions are rounded up to the nearest multiple of 8, required by the three VGG max-pool layers. Pass `--max-size 0` for full resolution if GPU memory allows.

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