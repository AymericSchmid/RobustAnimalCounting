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
        ├── images/
        ├── annotations.csv

    This loader assumes one row per bounding box.
    """

    def __init__(self, root, split=None, transform=None, target_transform=None, return_image_path=True):
        super().__init__(root, split, transform, target_transform, return_image_path=return_image_path)
    
    def _load_samples(self):
        annotation_path = self.root / "annotations.csv"

        df = pd.read_csv(annotation_path, names=['image_path', 'x1', 'y1', 'x2', 'y2', 'species', 'split'])
        if self.split:
            df = df[df['split'] == self.split]
        
        self.annotations_df = df
        
        grouped = df.groupby('image_path')

        samples = []
        for image_path, group in grouped:
            samples.append({
                "image_id": Path(image_path).stem,
                "image_path": self.root / "images" / image_path,
                "rows": group.reset_index(drop=True),
            })
        return samples

    def load_image(self, image_path: Path) -> Any:
        return Image.open(image_path).convert("RGB")

    def load_annotation(self, sample_info):
        rows = sample_info["rows"]

        image = Image.open(sample_info["image_path"])
        width, height = image.size

        # convert to numbers
        boxes = rows[['x1', 'y1', 'x2', 'y2']].values.tolist()
        boxes = [[float(x) for x in box] for box in boxes]
        labels = [1] * len(boxes)  # all animals mapped to label 1

        return self.build_annotation(
            boxes=boxes,
            labels=labels,
            count=len(boxes),
            image_size=(height, width),
            metadata={"dataset": "eikelboom"},
        )