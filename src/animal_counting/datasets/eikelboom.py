from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .base import BaseAnimalCountingDataset

class EikelboomDataset(BaseAnimalCountingDataset):
    """
    Wrapper for the Eikelboom dataset.

    Expected structure:
        data/raw/eikelboom/
        ├── train/
        ├── val/
        ├── test/
        ├── annotations_train.csv
        ├── annotations_val.csv
        └── annotations_test.csv

    This loader assumes one row per bounding box.
    """

    def __init__(self, root, split, transform=None, target_transform=None, return_image_path=True):
        super().__init__(root, split, transform, target_transform, return_image_path=return_image_path)

    def _get_split_paths(self):
        split_map = {
            "train": "annotations_train.csv",
            "val": "annotations_val.csv",
            "test": "annotations_test.csv",
        }

        annotation_file = split_map[self.split]
        annotation_path = self.root / annotation_file

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        return annotation_path
    
    def _load_samples(self):
        annotation_path = self._get_split_paths()

        df = pd.read_csv(annotation_path, names=['image_path', 'x1', 'y1', 'x2', 'y2', 'species'])
        self.annotations_df = df
        
        grouped = df.groupby('image_path')

        samples = []
        for image_path, group in grouped:
            samples.append({
                "image_id": image_path,
                "image_path": self.root / image_path,
                "rows": group.reset_index(drop=True),
            })
        return samples

    def load_image(self, image_path: Path) -> Any:
        return Image.open(image_path).convert("RGB")

    def load_annotation(self, sample_info):
        rows = sample_info["rows"]

        image = Image.open(sample_info["image_path"])
        width, height = image.size

        boxes = rows[['x1', 'y1', 'x2', 'y2']].values.tolist()
        labels = [1] * len(boxes)

        return self.build_annotation(
            boxes=boxes,
            labels=labels,
            count=len(boxes),
            image_size=(height, width),
            metadata={"dataset": "eikelboom"},
        )