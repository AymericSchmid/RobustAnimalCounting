from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .base import BaseAnimalCountingDataset


class QianPenguinsDataset(BaseAnimalCountingDataset):
    """
    Wrapper for the Qian Penguins dataset.

    Expected structure after preprocessing:
        data/splits/qian_penguins/
        ├── images/
        └── annotations.csv

    annotations.csv has columns (no header):
        image_path, x, y, species, colony, split

    Unlike Eikelboom/WAID, this dataset uses POINT annotations
    (penguin head locations) rather than bounding boxes.
    This is the format needed for density-map models (CSRNet)
    and point-based models (P2PNet).
    """

    def __init__(self, root, split=None, transform=None, target_transform=None, return_image_path=True):
        super().__init__(root, split, transform, target_transform, return_image_path=return_image_path)

    def _load_samples(self):
        annotation_path = self.root / "annotations.csv"

        df = pd.read_csv(
            annotation_path,
            names=["image_path", "x", "y", "species", "colony", "split"],
        )
        if self.split:
            df = df[df["split"] == self.split]

        self.annotations_df = df

        grouped = df.groupby("image_path")

        samples = []
        for image_path, group in grouped:
            samples.append(
                {
                    "image_id": Path(image_path).stem,
                    "image_path": self.root / "images" / image_path,
                    "rows": group.reset_index(drop=True),
                }
            )
        return samples

    def load_image(self, image_path: Path) -> Any:
        return Image.open(image_path).convert("RGB")

    # Size of synthetic bounding box (in pixels) centered on each point.
    # Used so detection models like YOLO can train on this point-annotated data.
    SYNTHETIC_BOX_SIZE = 10

    def load_annotation(self, sample_info):
        rows = sample_info["rows"]

        image = Image.open(sample_info["image_path"])
        width, height = image.size

        # Extract point coordinates
        points = rows[["x", "y"]].values.tolist()
        points = [[float(x), float(y)] for x, y in points]
        labels = [1] * len(points)  # all penguins mapped to label 1

        # Generate synthetic bounding boxes around each point
        # so detection models (YOLO) can also use this dataset
        half = self.SYNTHETIC_BOX_SIZE / 2.0
        boxes = []
        for x, y in points:
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(width, x + half)
            y2 = min(height, y + half)
            boxes.append([x1, y1, x2, y2])

        return self.build_annotation(
            points=points,
            boxes=boxes,
            labels=labels,
            count=len(points),
            image_size=(height, width),
            metadata={"dataset": "qian_penguins"},
        )