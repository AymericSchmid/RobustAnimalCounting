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