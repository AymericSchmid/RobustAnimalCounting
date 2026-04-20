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
| AED | ~15,581 points | ⬚ | ⬚ | ⬚ | To do |
 
### Models
 
| Model | Wrapper | Training script | Status |
|-------|---------|----------------|--------|
| YOLOv8 | ✅ `yolov8.py` | ✅ `train_eikelboom.py` | Working |
| CSRNet | ⬚ | ⬚ | To do — needs density map converter |
| P2PNet | ⬚ | ⬚ | To do |
 
### Important note: PYTHONPATH
 
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
These work directly with YOLO and can also derive points (box centers) for CSRNet/P2PNet.
 
```bash
# Example: WAID
git clone https://github.com/xiaohuicui/WAID.git data/raw/waid
python scripts/datasets_processing/preprocess_waid.py
PYTHONPATH=src python scripts/datasets_processing/convert_dataset.py \
    --dataset waid --format yolo --root data/splits/waid --output data/yolo/waid
```
 
#### Point-annotation datasets (Qian Penguins, AED)
 
The annotations.csv has columns: `image_path, x, y, species, colony, split`.
These are the native format for CSRNet and P2PNet. For YOLO, the wrapper
generates synthetic 10×10 px bounding boxes around each point.
 
```bash
# Example: Qian Penguins
# Download all files from https://doi.org/10.5061/dryad.8931zcrv8
# into data/raw/qian_penguins/ (4 zips + 4 JSONs)
python scripts/datasets_processing/preprocess_qian.py
PYTHONPATH=src python scripts/datasets_processing/convert_dataset.py \
    --dataset qian_penguins --format yolo \
    --root data/splits/qian_penguins --output data/yolo/qian_penguins
```