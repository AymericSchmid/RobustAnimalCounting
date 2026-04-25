from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .base import BaseAnimalCountingDataset


def generate_density_map(
    points: np.ndarray,
    image_size: tuple[int, int],
    beta: float = 0.3,
    k: int = 3,
    min_sigma: float = 1.0,
    fixed_sigma: float = 15.0,
) -> np.ndarray:
    """
    Generate a density map from point annotations using adaptive Gaussian kernels.

    Each point contributes a Gaussian that integrates to 1, so the density map
    sum equals the total animal count. Sigma is estimated per-point from the mean
    distance to its k nearest neighbours, following Hamrouni et al. (2020):
        sigma_n = beta * mean_k_dist,  beta=0.3, k=3

    Args:
        points: (N, 2) float array of (x, y) pixel coordinates.
        image_size: (H, W) of the output density map.
        beta: Scale factor for the adaptive sigma.
        k: Number of nearest neighbours used to estimate sigma.
        min_sigma: Lower bound on sigma to avoid collapsed Gaussians.
        fixed_sigma: Fallback sigma used when N == 1.
    """
    H, W = image_size
    density = np.zeros((H, W), dtype=np.float32)

    if len(points) == 0:
        return density

    points = np.asarray(points, dtype=np.float32)

    if len(points) == 1:
        x, y = int(round(points[0, 0])), int(round(points[0, 1]))
        if 0 <= x < W and 0 <= y < H:
            density[y, x] = 1.0
        return gaussian_filter(density, sigma=fixed_sigma)

    tree = KDTree(points)
    n_query = min(k + 1, len(points))  # +1 because the point itself is always the closest
    distances, _ = tree.query(points, k=n_query)

    for i, point in enumerate(points):
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < W and 0 <= y < H):
            continue

        # distances[i, 0] is always 0 (self); skip it
        neighbor_dists = distances[i, 1:] if n_query > 1 else distances[i]
        d_bar = float(neighbor_dists.mean()) if len(neighbor_dists) > 0 else fixed_sigma / beta
        sigma = max(beta * d_bar, min_sigma)

        pt_map = np.zeros((H, W), dtype=np.float32)
        pt_map[y, x] = 1.0
        density += gaussian_filter(pt_map, sigma=sigma)

    return density


class DensityMapDataset(Dataset):
    """
    Wraps any BaseAnimalCountingDataset to produce (image, density_map) pairs
    for training CSRNet.

    Point annotations are used directly when available; bounding-box centres
    are used as pseudo-points for box-only datasets (Eikelboom, WAID, Delplanque).

    The density map is generated at full patch resolution, then downsampled by
    `density_scale` (default 8, matching VGG pool1+pool2+pool3) via average
    pooling. Values are rescaled by density_scale² to preserve the integral
    (i.e. density_map.sum() == animal count).
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        dataset: BaseAnimalCountingDataset,
        patch_size: int = 512,
        augment: bool = False,
        density_scale: int = 8,
        beta: float = 0.3,
        k: int = 3,
    ) -> None:
        self.dataset = dataset
        self.patch_size = patch_size
        self.augment = augment
        self.density_scale = density_scale
        self.beta = beta
        self.k = k

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        image: Any = sample["image"]
        target: dict = sample["target"]

        # --- Extract (x, y) point coordinates ---
        if target["points"] is not None and len(target["points"]) > 0:
            points = target["points"].numpy().astype(np.float32)  # (N, 2)
        elif target["boxes"] is not None and len(target["boxes"]) > 0:
            boxes = target["boxes"].numpy()  # xyxy
            points = np.stack(
                [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2],
                axis=1,
            ).astype(np.float32)
        else:
            points = np.zeros((0, 2), dtype=np.float32)

        # --- Ensure PIL image ---
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)

        W, H = image.size

        # --- Crop or resize to patch_size ---
        if W >= self.patch_size and H >= self.patch_size:
            left = random.randint(0, W - self.patch_size)
            top = random.randint(0, H - self.patch_size)
            image = TF.crop(image, top, left, self.patch_size, self.patch_size)
            if len(points) > 0:
                mask = (
                    (points[:, 0] >= left)
                    & (points[:, 0] < left + self.patch_size)
                    & (points[:, 1] >= top)
                    & (points[:, 1] < top + self.patch_size)
                )
                points = points[mask].copy()
                if len(points) > 0:
                    points[:, 0] -= left
                    points[:, 1] -= top
        else:
            scale_x = self.patch_size / W
            scale_y = self.patch_size / H
            image = TF.resize(image, (self.patch_size, self.patch_size))
            if len(points) > 0:
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y

        crop_H = crop_W = self.patch_size

        # --- Augmentation: random horizontal flip ---
        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            if len(points) > 0:
                points[:, 0] = crop_W - 1 - points[:, 0]

        # --- Generate density map at full patch resolution ---
        density_full = generate_density_map(
            points, (crop_H, crop_W), beta=self.beta, k=self.k
        )

        # --- Downsample by density_scale; scale values to preserve sum ---
        density_tensor = torch.from_numpy(density_full).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        density_down = F.avg_pool2d(
            density_tensor,
            kernel_size=self.density_scale,
            stride=self.density_scale,
        ) * (self.density_scale ** 2)
        density_down = density_down.squeeze(0)  # (1, H/8, W/8)

        # --- Normalize image with ImageNet statistics ---
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD)

        return img_tensor, density_down
