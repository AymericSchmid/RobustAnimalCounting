"""Evaluate a trained YOLOv8 checkpoint on one of the project datasets.

Usage (from the repo root):

    # In-domain (trained and tested on Eikelboom, with density buckets)
    PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \\
        --train-dataset eikelboom \\
        --test-dataset eikelboom \\
        --weights results/yolov8/eikelboom/weights/best.pt \\
        --mode density

    # Cross-domain (trained on Eikelboom, tested on WAID, overall only)
    PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \\
        --train-dataset eikelboom \\
        --test-dataset waid \\
        --weights results/yolov8/eikelboom/weights/best.pt \\
        --mode cross
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from animal_counting.datasets.delplanque import DelplanqueDataset
from animal_counting.datasets.eikelboom import EikelboomDataset
from animal_counting.datasets.qian_penguins import QianPenguinsDataset
from animal_counting.datasets.waid import WAIDDataset
from animal_counting.evaluation import evaluate_yolo_cross, evaluate_yolo_density
from animal_counting.models.yolov8 import YOLOv8CountingModel


DATASET_REGISTRY = {
    "eikelboom": EikelboomDataset,
    "delplanque": DelplanqueDataset,
    "waid": WAIDDataset,
    "qian_penguins": QianPenguinsDataset,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 on an animal counting dataset")
    p.add_argument("--train-dataset", required=True, choices=list(DATASET_REGISTRY.keys()),
                   help="Name of the dataset used for training (used for output folder naming).")
    p.add_argument("--test-dataset", required=True, choices=list(DATASET_REGISTRY.keys()),
                   help="Name of the dataset used for evaluation.")
    p.add_argument("--weights", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--mode", choices=["density", "cross"], default="density",
                   help="'density' = per-bucket counting metrics (in-domain). "
                        "'cross' = overall-only (cross-dataset generalization).")
    return p.parse_args()


def run_inference(model, dataset, conf, iou, imgsz):
    """Loop over the dataset, call model.predict() per image, collect counts."""
    image_ids, pred_counts, gt_counts = [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        pred = model.predict(
            sample["path"], conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )

        image_ids.append(sample["image_id"])
        pred_counts.append(int(pred.count))
        gt_counts.append(int(sample["target"]["count"]))

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(dataset)}] processed")

    return image_ids, pred_counts, gt_counts


def print_summary(val_metrics, counting_results):
    print("\n=== Detection metrics (Ultralytics val) ===")
    box = val_metrics.box
    print(f"  mAP@0.5       {box.map50:.4f}")
    print(f"  mAP@0.5:0.95  {box.map:.4f}")
    print(f"  precision     {box.mp:.4f}")
    print(f"  recall        {box.mr:.4f}")

    print("\n=== Counting metrics ===")
    for bucket, m in counting_results.items():
        print(f"\n[{bucket}]  n={m['n_images']}")
        for k in ("MAE", "RMSE", "relative_error"):
            print(f"  {k:16s} {m[k]:.4f}")


def save_results(out_dir, args, val_metrics, counting_results,
                 image_ids, pred_counts, gt_counts, n_samples):
    """Save detection/counting metrics, per-image predictions and config to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_dump = {
        "detection": {
            "mAP50": float(val_metrics.box.map50),
            "mAP50_95": float(val_metrics.box.map),
            "precision": float(val_metrics.box.mp),
            "recall": float(val_metrics.box.mr),
        },
        "counting": counting_results,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics_dump, f, indent=2)

    per_image = [
        {"image_id": iid, "pred_count": pc, "gt_count": gc}
        for iid, pc, gc in zip(image_ids, pred_counts, gt_counts)
    ]
    with (out_dir / "per_image.json").open("w") as f:
        json.dump(per_image, f, indent=2)

    config_log = {
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
        "split": args.split,
        "weights": str(args.weights),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "mode": args.mode,
        "n_samples": n_samples,
    }
    with (out_dir / "config.json").open("w") as f:
        json.dump(config_log, f, indent=2)

    print(f"\nResults written to: {out_dir}")


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:        {device}")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Test dataset:  {args.test_dataset} (split={args.split})")
    print(f"Weights:       {args.weights}")
    print(f"Mode:          {args.mode}")

    # -- load dataset ---------------------------------------------------------
    dataset_cls = DATASET_REGISTRY[args.test_dataset]
    dataset_root = ROOT / "data" / "splits" / args.test_dataset
    dataset = dataset_cls(root=dataset_root, split=args.split)
    print(f"Loaded {len(dataset)} samples")

    # -- load model -----------------------------------------------------------
    model = YOLOv8CountingModel(
        device=device, config={"model_path": args.weights}
    )

    # -- (1) detection metrics via Ultralytics val ----------------------------
    print("\nRunning Ultralytics val()...")
    data_yaml = ROOT / "data" / "yolo" / args.test_dataset / "data.yaml"
    val_metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz)

    # -- (2) counting metrics (per-bucket or overall-only) --------------------
    print("\nRunning per-image predictions for counting metrics...")
    image_ids, pred_counts, gt_counts = run_inference(
        model, dataset, args.conf, args.iou, args.imgsz
    )

    if args.mode == "density":
        counting_results = evaluate_yolo_density(
            image_ids=image_ids,
            pred_counts=pred_counts,
            gt_counts=gt_counts,
        )
    else:  # cross
        counting_results = evaluate_yolo_cross(
            pred_counts=pred_counts,
            gt_counts=gt_counts,
        )

    print_summary(val_metrics, counting_results)

    # -- save -----------------------------------------------------------------
    run_name = f"{args.train_dataset}_to_{args.test_dataset}_{args.mode}"
    out_dir = ROOT / "results" / "yolov8" / "eval" / run_name
    save_results(
        out_dir, args, val_metrics, counting_results,
        image_ids, pred_counts, gt_counts, len(dataset),
    )


if __name__ == "__main__":
    main()