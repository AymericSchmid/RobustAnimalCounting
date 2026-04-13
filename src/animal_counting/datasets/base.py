from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

@dataclass
class SampleAnnotation:

    boxes = None
    labels = None
    points = None
    count = None
    image_size = None
    metadata = {}

    def to_dict(self):
        """Convert the annotation to a dictionary format"""
        return {
            "boxes": self.boxes,
            "labels": self.labels,
            "points": self.points,
            "count": self.count,
            "image_size": self.image_size,
            "metadata": self.metadata
        }
    
class BaseAnimalCountingDataset(Dataset, ABC):

    def __init__(self, root, split, transform=None, target_transform=None, split_file=None, return_image_path=True):
        self.root = Path(root)
        self.split = split.lower().strip()
        self.transform = transform
        self.target_transform = target_transform
        self.split_file = Path(split_file) if split_file else None
        self.return_image_path = return_image_path

        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(f"No samples found for dataset={self.root}, split={self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]

        image_path = Path(sample_info["image_path"])
        image_id = str(sample_info["image_id"])

        image = self.load_image(image_path)
        annotation = self.load_annotation(sample_info)
        target = annotation.to_dict()

        # fill count automaticall when possible
        if target["count"] is None:
            if target["points"] is not None:
                target["count"] = int(target["points"].shape[0])
            elif target["boxes"] is not None:
                target["count"] = int(target["boxes"].shape[0])
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            image, target = self.transform(image, target)
        
        output = {
            "image": image,
            "target": target,
            "image_id": image_id,
            "split": self.split
        }

        if self.return_image_path:
            output["path"] = image_path

        return output
    
    @abstractmethod
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Return a list of sample descriptors.

        Each entry should contain at least:
        {
            "image_id": str,
            "image_path": str or Path,
            ...
        }

        Additional dataset-specific fields may be included and later used by
        load_annotation().
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_image(self, image_path: Path) -> Any:
        """
        Load one image from disk.

        Subclasses can return:
        - a PIL image
        - a torch Tensor
        - another structure expected by their transforms
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_annotation(self, sample_info: Dict[str, Any]) -> SampleAnnotation:
        """
        Load and convert one raw annotation into the unified format.
        """
        raise NotImplementedError
    
    def resolve_split_files(self):
        if self.split_file is not None:
            return self.split_file
        
        candidate = self.root / f"{self.split}.txt"
        if not candidate.exists():
            raise FileNotFoundError(f"Split file not found: {candidate}")
        return candidate
    
    def read_split_file(self, split_file=None):
        path = Path(split_file) if split_file is not None else self.resolve_split_files()
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        
        entries = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append(line)
        
        return entries
    
    @staticmethod
    def ensure_tensor_boxes(boxes):
        """Convert boxes to float32 tensor of shape [N, 4]"""
        if boxes is None:
            return None
        
        if isinstance(boxes, Tensor):
            result = boxes.to(dtype=torch.float32)
        else:
            result = torch.tensor(boxes, dtype=torch.float32)
        
        if result.numel() == 0:
            return torch.zeros((0,4), dtype=torch.float32)
    
        if result.ndim != 2 or result.shape[1] != 4:
            raise ValueError(f"Boxes should have shape [N, 4], got {result.shape}")
        return result
    
    @staticmethod
    def ensure_tensor_points(points):
        """Convert points to float32 tensor of shape [N,2] in XY format"""
        if points is None:
            return None
        
        if isinstance(points, Tensor):
            result = points.to(dtype=torch.float32)
        else:
            result = torch.tensor(points, dtype=torch.float32)
        
        if result.numel() == 0:
            return torch.zeros((0,2), dtype=torch.float32)
        
        if result.ndim != 2 or result.shape[1] != 2:
            raise ValueError(f"Points should have shape [N, 2], got {result.shape}")
        return result
    
    @staticmethod
    def ensure_tensor_labels(labels, num_instances=None, default_label=-1):
        """Convert labels to int64 tensor of shape [N]"""
        if labels is None:
            if num_instances is None:
                return None
            return torch.full((num_instances,), fill_value=default_label, dtype=torch.int64)
        
        if isinstance(labels, Tensor):
            result = labels.to(dtype=torch.int64)
        else:
            result = torch.tensor(labels, dtype=torch.int64)
        
        if result.ndim != 1:
            raise ValueError(f"Labels should have shape [N], got {result.shape}")
        return result
    
    @staticmethod
    def xywh_to_xyxy(boxes):
        """Convert boxes from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]"""
        boxes_tensor = BaseAnimalCountingDataset.ensure_tensor_boxes(boxes)
        if boxes_tensor is None:
            return torch.zeros((0,4), dtype=torch.float32)
        
        converted = boxes_tensor.clone()
        converted[:, 2] = converted[:, 0] + converted[:, 2]
        converted[:, 3] = converted[:, 1] + converted[:, 3]
        return converted
    
    @classmethod
    def build_annotation(cls, *, boxes=None, labels=None, points=None, count=None, image_size=None, metadata=None, default_label=1):
        """Helper to build a SampleAnnotation with proper tensor conversions and defaults"""
        boxes_tensor = cls.ensure_tensor_boxes(boxes)
        points_tensor = cls.ensure_tensor_points(points)

        num_instances = None
        if boxes_tensor is not None:
            num_instances = boxes_tensor.shape[0]
        elif points_tensor is not None:
            num_instances = points_tensor.shape[0]
        
        labels_tensor = cls.ensure_tensor_labels(labels, num_instances=num_instances, default_label=default_label)

        annotation = SampleAnnotation(
            boxes=boxes_tensor,
            labels=labels_tensor,
            points=points_tensor,
            count=count,
            image_size=image_size,
            metadata=metadata or {}
        )
        return annotation