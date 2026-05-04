"""Evaluate a trained CSRNet checkpoint on one of the project datasets.

Usage (from the repo root):

    # In-domain with density buckets (Qian Penguins, tests H2)
    PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \\
        --train-dataset qian_penguins \\
        --test-dataset  qian_penguins \\
        --weights results/csrnet/qian_penguins/best.pth \\
        --mode density

    # Cross-domain (trained on Qian Penguins, tested on Eikelboom, tests H4)
    PYTHONPATH=src python scripts/eval/csrnet/evaluate.py \\
        --train-dataset qian_penguins \\
        --test-dataset  eikelboom \\
        --weights results/csrnet/qian_penguins/best.pth \\
        --mode cross

Notes
-----
- Images are resized to be divisible by 8 before inference (required by VGG pooling).
  Use --max-size to cap very large images for memory reasons (default: 1024).
- GT density maps are generated on-the-fly from point/box annotations and
  downsampled by 8 to match the network output, so they are comparable for SSIM.
- Results are saved as JSON in results/csrnet/<train>_<test>_<mode>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from animal_counting.datasets.delplanque import DelplanqueDataset
from animal_counting.datasets.density_map import generate_density_map
from animal_counting.datasets.eikelboom import EikelboomDataset
from animal_counting.datasets.qian_penguins import QianPenguinsDataset
from animal_counting.datasets.waid import WAIDDataset
from animal_counting.datasets.aed import AEDDataset
from animal_counting.evaluation import (
    evaluate_csrnet_cross,
    evaluate_csrnet_density,
)
from animal_counting.models.csrnet import CSRNetCountingModel

DATASET_REGISTRY = {
    "eikelboom": EikelboomDataset,
    "delplanque": DelplanqueDataset,
    "waid": WAIDDataset,
    "qian_penguins": QianPenguinsDataset,
    "aed": AEDDataset,
}

DENSITY_SCALE = 8  # must match the CSRNet frontend stride (pool1 * pool2 * pool3)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CSRNet on an animal counting dataset")
    p.add_argument("--train-dataset", choices=list(DATASET_REGISTRY.keys()),
                   help="Dataset used for training (for labelling output files)")
    p.add_argument("--test-dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--weights", required=True, help="Path to best.pth checkpoint")
    p.add_argument("--split", default="test")
    p.add_argument("--mode", choices=["density", "cross"], default="density",
                   help="'density' = per-bucket results. 'cross' = overall only.")
    p.add_argument("--max-size", type=int, default=1024,
                   help="Cap the longest image dimension before inference "
                        "(reduces memory use on large aerial images). "
                        "Set to 0 to use full resolution.")
    p.add_argument("--beta", type=float, default=0.3,
                   help="Gaussian sigma scale factor for GT density map generation")
    p.add_argument("--k", type=int, default=3,
                   help="Number of nearest neighbours for adaptive sigma")
    return p.parse_args()


def to_eval_size(pil_image: Image.Image, max_size: int) -> tuple[Image.Image, float, float]:
    """
    Resize a PIL image so that:
      1. Neither dimension exceeds max_size (if max_size > 0).
      2. Both dimensions are divisible by DENSITY_SCALE (= 8),
         so the VGG pooling layers produce an integer output size.

    Returns the resized image and the (scale_x, scale_y) factors applied,
    needed to rescale annotation coordinates to the new image size.
    """
    W, H = pil_image.size
    scale = 1.0

    if max_size > 0 and max(H, W) > max_size:
        scale = max_size / max(H, W)
        H = int(H * scale)
        W = int(W * scale)

    # Round up to nearest multiple of DENSITY_SCALE
    H_new = ((H + DENSITY_SCALE - 1) // DENSITY_SCALE) * DENSITY_SCALE
    W_new = ((W + DENSITY_SCALE - 1) // DENSITY_SCALE) * DENSITY_SCALE

    scale_x = W_new / pil_image.size[0]
    scale_y = H_new / pil_image.size[1]

    if H_new != pil_image.size[1] or W_new != pil_image.size[0]:
        pil_image = pil_image.resize((W_new, H_new), Image.BILINEAR)

    return pil_image, scale_x, scale_y


def get_points(target: dict) -> np.ndarray:
    """
    Extract (x, y) point coordinates from a sample target dict.
    Uses point annotations when available; falls back to bounding-box centroids.
    Returns a (N, 2) float32 array (may be empty).
    """
    if target["points"] is not None and len(target["points"]) > 0:
        return target["points"].numpy().astype(np.float32)
    if target["boxes"] is not None and len(target["boxes"]) > 0:
        boxes = target["boxes"].numpy()
        return np.stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2],
            axis=1,
        ).astype(np.float32)
    return np.zeros((0, 2), dtype=np.float32)


def make_gt_density_map(points: np.ndarray, H: int, W: int, beta: float, k: int) -> np.ndarray:
    """
    Generate a ground-truth density map at (H/8, W/8) — the same resolution
    as the CSRNet output — by creating a full-resolution map and average-pooling.
    """
    dm_full = generate_density_map(points, (H, W), beta=beta, k=k)
    t = torch.from_numpy(dm_full).unsqueeze(0).unsqueeze(0)
    dm_down = F.avg_pool2d(t, kernel_size=DENSITY_SCALE, stride=DENSITY_SCALE) * (DENSITY_SCALE ** 2)
    return dm_down.squeeze().numpy()  # (H/8, W/8)


def run_inference(model, dataset, args):
    """
    Loop over the dataset, run CSRNet on each image, and collect:
      - image_ids, pred_counts, gt_counts  (for counting metrics)
      - pred_maps, gt_maps                 (for SSIM — 2-D numpy arrays)
    """
    image_ids, pred_counts, gt_counts = [], [], []
    pred_maps, gt_maps = [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        pil_image = sample["image"]
        target = sample["target"]

        if not isinstance(pil_image, Image.Image):
            from torchvision.transforms.functional import to_pil_image
            pil_image = to_pil_image(pil_image)

        # Resize to eval dimensions (divisible by 8, capped by max_size)
        pil_resized, scale_x, scale_y = to_eval_size(pil_image, args.max_size)
        W_new, H_new = pil_resized.size

        # Scale annotation points to match the resized image
        points = get_points(target)
        if len(points) > 0:
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y

        # GT density map at network output resolution
        gt_dm = make_gt_density_map(points, H_new, W_new, args.beta, args.k)
        gt_count = float(gt_dm.sum())

        # Predicted density map
        result = model.predict(pil_resized)
        pred_dm = result.density_map.squeeze().numpy()  # (H/8, W/8)
        pred_count = float(result.count)

        image_ids.append(sample["image_id"])
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        pred_maps.append(pred_dm)
        gt_maps.append(gt_dm)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(dataset)}] processed")

    return image_ids, pred_counts, gt_counts, pred_maps, gt_maps


def print_summary(results: dict):
    print("\n=== CSRNet evaluation results ===")
    for bucket, m in results.items():
        n = m.get("n_images", "?")
        print(f"\n[{bucket}]  n={n}")
        for k, v in m.items():
            if k == "n_images":
                continue
            print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")


def save_results(out_path: Path, args, results: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "weights": args.weights,
            "split": args.split,
            "mode": args.mode,
            "max_size": args.max_size,
            "beta": args.beta,
            "k": args.k,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Evaluation parameters ===")
    print(f"  Train dataset : {args.train_dataset}")
    print(f"  Test dataset  : {args.test_dataset} (split={args.split})")
    print(f"  Weights       : {args.weights}")
    print(f"  Mode          : {args.mode}")
    print(f"  Max image size: {args.max_size or 'full resolution'}")
    print(f"  Device        : {device}")

    dataset = DATASET_REGISTRY[args.test_dataset](
        root=ROOT / "data" / "splits" / args.test_dataset,
        split=args.split,
    )
    print(f"\nLoaded {len(dataset)} samples from {args.test_dataset}/{args.split}")

    model = CSRNetCountingModel(device=device, pretrained=False)
    model.load(args.weights)
    print(f"Loaded weights from {args.weights}")

    print("\nRunning inference...")
    image_ids, pred_counts, gt_counts, pred_maps, gt_maps = run_inference(
        model, dataset, args
    )

    print("\nComputing metrics...")
    if args.mode == "density":
        results = evaluate_csrnet_density(
            image_ids=image_ids,
            pred_counts=pred_counts,
            gt_counts=gt_counts,
            pred_maps=pred_maps,
            gt_maps=gt_maps,
        )
    else:
        results = evaluate_csrnet_cross(
            pred_counts=pred_counts,
            gt_counts=gt_counts,
            pred_maps=pred_maps,
            gt_maps=gt_maps,
        )

    print_summary(results)

    out_path = (
        ROOT / "results" / "csrnet"
        / f"{args.train_dataset or 'unknown'}_{args.test_dataset}_{args.mode}.json"
    )
    save_results(out_path, args, results)


if __name__ == "__main__":
    main()
