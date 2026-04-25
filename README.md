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

The `evaluation/` module is currently a placeholder. A teammate is implementing it on a separate branch. The `BaseCountingModel` base class already provides `evaluate_counts()` (MAE, RMSE, bias, MAPE) and `evaluate_dataset()`, which will be the foundation for the full evaluation pipeline.

---

## What remains to do

### High priority (needed to produce results)

- [ ] **Train YOLOv8** on Eikelboom, WAID, Delplanque — run existing sbatch scripts
- [ ] **Train CSRNet** on Qian Penguins — `sbatch sbatch_scripts/train_csrnet_qian.sh`
- [ ] **Merge evaluation branch** and wire it up to both models' `predict()` output
- [ ] **Per-density bucket evaluation** — bin test images into sparse (1–10), medium (10–50), crowded (50+) and report MAE/RMSE per bucket (H1 and H2)
- [ ] **Cross-domain evaluation** — train on dataset A, test on dataset B (e.g., Eikelboom → Delplanque for H4)

### Medium priority

- [ ] **Complete `YOLOv8CountingModel.predict()`** — currently a `pass`; needs to wrap ultralytics results into `PredictionResult`
- [ ] **CSRNet on box-annotation datasets** — `DensityMapDataset` already supports box→centroid conversion; add a training script for Eikelboom/WAID to enable cross-domain comparison
- [ ] **SSIM reporting** for CSRNet density maps (paper metric for spatial accuracy)

### Low priority

- [ ] **AED dataset** — wrapper + preprocessing (useful for cross-domain elephant experiments)
- [ ] **Results notebook** — visualize density maps, detection boxes, per-bucket bar charts

---

## Important note: PYTHONPATH

The project package isn't installed via pip yet. To run any script that imports
from `animal_counting`, prefix your command with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/datasets_processing/convert_dataset.py ...
```

### How to set up a dataset

Every dataset follows the same 3-step pipeline:

1. **Download** raw data into `data/raw/<name>/`
2. **Preprocess** (creates `data/splits/<name>/annotations.csv` + copies images)
3. **Convert** to model-specific format (e.g., YOLO)

#### Box-annotation datasets (Eikelboom, Delplanque, WAID)

The annotations.csv has columns: `image_path, x1, y1, x2, y2, species, split`.
These work directly with YOLO and can also derive points (box centers) for CSRNet.

```bash
# Example: WAID
git clone https://github.com/xiaohuicui/WAID.git data/raw/waid
python scripts/datasets_processing/preprocess_waid.py
PYTHONPATH=src python scripts/datasets_processing/convert_dataset.py \
    --dataset waid --format yolo --root data/splits/waid --output data/yolo/waid
```

#### Point-annotation datasets (Qian Penguins)

The annotations.csv has columns: `image_path, x, y, species, colony, split`.
These are the native format for CSRNet. For YOLO, the wrapper generates synthetic
10×10 px bounding boxes around each point.

```bash
# Download all files from https://doi.org/10.5061/dryad.8931zcrv8
# into data/raw/qian_penguins/ (4 zips + 4 JSONs), then:
python scripts/datasets_processing/preprocess_qian.py
PYTHONPATH=src python scripts/datasets_processing/convert_dataset.py \
    --dataset qian_penguins --format yolo \
    --root data/splits/qian_penguins --output data/yolo/qian_penguins
```

### How to train

#### YOLOv8

```bash
# Locally
PYTHONPATH=src python scripts/train/yolov8/train_eikelboom.py

# On the cluster
sbatch sbatch_scripts/train_yolov8_eikelboom.sh
```

Results are written to `results/yolov8/<dataset>/`.

#### CSRNet

```bash
# Locally
PYTHONPATH=src python scripts/train/csrnet/train_qian.py

# On the cluster
sbatch sbatch_scripts/train_csrnet_qian.sh
```

Best checkpoint is saved to `results/csrnet/qian_penguins/best.pth`. Training prints `val_MAE` and `val_RMSE` each epoch and stops early after 20 epochs without improvement (patience configurable in the script).

---

## Key design notes

- **All species → single class.** Both models treat every animal as class 0/1 — the task is counting, not species classification.
- **Density map generation.** Adaptive Gaussian sigma per point: `σ = 0.3 × mean_distance_to_3_nearest_neighbours` (Hamrouni et al. 2020). Single-point images use a fixed σ = 15 px.
- **CSRNet output resolution.** 1/8 of input (VGG pool1 × pool2 × pool3). A 512×512 patch produces a 64×64 density map. `density_map.sum() = predicted count`.
- **Box-only datasets with CSRNet.** `DensityMapDataset` automatically computes bounding-box centroids and uses them as pseudo-points when no point annotations are available.