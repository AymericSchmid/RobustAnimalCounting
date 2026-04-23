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
import shutil
import tempfile
from pathlib import Path
import yaml
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
    p.add_argument("--train-dataset", choices=list(DATASET_REGISTRY.keys()), help="Name of the dataset used for training")
    p.add_argument("--test-dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--weights", required=True, help="Path to weights of the trained model (best.pt checkpoint)")
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--mode", choices=["density", "cross"], default="density",
                   help="'density' = per-bucket counting metrics (in-domain). "
                        "'cross' = overall-only (cross-dataset generalization).")
    p.add_argument("--output-dir", default=None,
                   help="Directory where results JSON will be saved. "
                        "Default: results/yolov8/<train>_to_<test>/")
    return p.parse_args()


def run_inference(model, dataset, conf, iou, imgsz):
    """Loop over the dataset, call model.predict() per image, collect counts."""
    image_ids, pred_counts, gt_counts = [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        pred = model.predict(
            sample["path"], conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )

        # Store results
        image_ids.append(sample["image_id"])
        pred_counts.append(int(pred.count))
        gt_counts.append(int(sample["target"]["count"]))

        # Print progress every 50 images
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(dataset)}] processed")

    return image_ids, pred_counts, gt_counts


def split_yaml_by_bucket(data_yaml, split, bucket_image_ids, tmp_root, bucket_name):
    """Create a temporary data.yaml pointing to the subset of images in a bucket.

    This is needed to run the Ultralytics val() method on just the images in a bucket, so we can get per-bucket detection metrics.

    args :
    - data_yaml: Path to the original data.yaml
    - split: "train", "val", or "test"
    - bucket_image_ids: set of image IDs that belong to the bucket
    - tmp_root: root dir where the new bucket-specific data.yaml and images/labels dirs will be created
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    # Resolve original images/labels dirs
    base = Path(cfg.get("path", data_yaml.parent)).resolve()
    orig_images = (base / cfg[split]).resolve()
    orig_labels = orig_images.parent.parent / "labels" / orig_images.name

    # Build new subset dirs
    bucket_dir = tmp_root / bucket_name
    new_images = bucket_dir / "images" / split
    new_labels = bucket_dir / "labels" / split
    new_images.mkdir(parents=True, exist_ok=True)
    new_labels.mkdir(parents=True, exist_ok=True)

    wanted = set(bucket_image_ids)
    n = 0
    for img in orig_images.iterdir():
        if img.stem not in wanted:
            continue
        shutil.copy2(img, new_images / img.name)
        lbl = orig_labels / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, new_labels / lbl.name)
        n += 1

    if n == 0:
        return None

    # Write new yaml
    new_cfg = {**cfg, "path": str(bucket_dir), split: f"images/{split}"}
    for k in ("train", "val", "test"):
        if k != split:
            new_cfg.pop(k, None)
    new_yaml = bucket_dir / "data.yaml"
    with open(new_yaml, "w") as f:
        yaml.safe_dump(new_cfg, f)
    return new_yaml


def val_metrics_to_dict(m):
    """Extract the fields we care about from an Ultralytics val() result."""
    return {
        "mAP@0.5": float(m.box.map50),
        "mAP@0.5:0.95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }


def print_summary(val_overall, val_per_bucket, counting_results):
    """Print a summary of the evaluation results to the console."""
    print("\n=== Detection metrics (Ultralytics val) — overall ===")
    for k, v in val_overall.items():
        print(f"  {k:16s} {v:.4f}")

    if val_per_bucket:
        print("\n=== Detection metrics (Ultralytics val) — per bucket ===")
        for bucket, metrics in val_per_bucket.items():
            if metrics is None:
                print(f"\n[{bucket}]  (empty — skipped)")
                continue
            print(f"\n[{bucket}]")
            for k, v in metrics.items():
                print(f"  {k:16s} {v:.4f}")

    print("\n=== Counting metrics ===")
    for bucket, m in counting_results.items():
        print(f"\n[{bucket}]  n={m['n_images']}")
        for k in ("MAE", "RMSE", "relative_error"):
            print(f"  {k:16s} {m[k]:.4f}")


def save_results(output_dir, args, val_overall, val_per_bucket, counting_results):
    """Save all evaluation results to a JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "params": {
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "split": args.split,
            "weights": args.weights,
            "mode": args.mode,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        },
        "detection_overall": val_overall,
        "detection_per_bucket": val_per_bucket,
        "counting": counting_results,
    }

    out_file = output_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nResults saved to {out_file}")
    return out_file


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Summary of the parameters for this run
    print("=== Evaluation parameters ===")
    print(f"  Train dataset: {args.train_dataset}")
    print(f"  Test dataset:  {args.test_dataset} (split={args.split})")
    print(f"  Weights:       {args.weights}")
    print(f"  Mode:          {args.mode}")

    # Load the test dataset
    dataset_cls = DATASET_REGISTRY[args.test_dataset]
    dataset_root = ROOT / "data" / "splits" / args.test_dataset
    dataset = dataset_cls(root=dataset_root, split=args.split)
    print(f"Loaded {len(dataset)} samples")

    # Load the trained YOLOv8 counting model
    model = YOLOv8CountingModel(
        device=device, config={"model_path": args.weights}
    )

    # Per-image inference for counting metrics
    print("\nRunning per-image inference...")
    image_ids, pred_counts, gt_counts = run_inference(
        model, dataset, args.conf, args.iou, args.imgsz
    )

    # Counting metrics (MAE, RMSE, relative error)
    print("\nRunning counting evaluation...")
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

    # Overall Ultralytics val()
    data_yaml = ROOT / "data" / "yolo" / args.test_dataset / "data.yaml"
    print("\nRunning Ultralytics val() — overall...")
    overall_raw = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz)
    val_overall = val_metrics_to_dict(overall_raw)

    # Per-bucket val() in density mode
    val_per_bucket = {}
    if args.mode == "density":
        print("\nRunning Ultralytics val() — per density bucket...")
        with tempfile.TemporaryDirectory(prefix="yolo_bucket_") as tmp:
            tmp_root = Path(tmp)
            for bucket, m in counting_results.items():
                if bucket == "overall" or "image_ids" not in m:
                    continue
                sub_yaml = split_yaml_by_bucket(
                    data_yaml, args.split, m["image_ids"], tmp_root, bucket
                )
                if sub_yaml is None:
                    print(f"  [{bucket}] empty bucket — skipping")
                    val_per_bucket[bucket] = None
                    continue
                print(f"  [{bucket}] running val() on {len(m['image_ids'])} images...")
                raw = model.val(data=str(sub_yaml), split=args.split, imgsz=args.imgsz)
                val_per_bucket[bucket] = val_metrics_to_dict(raw)

    # Print and save
    print_summary(val_overall, val_per_bucket, counting_results)

    output_dir = args.output_dir or (
        ROOT / "results" / "yolov8" / f"{args.mode}_{args.train_dataset}_to_{args.test_dataset}"
    )
    save_results(output_dir, args, val_overall, val_per_bucket, counting_results)


if __name__ == "__main__":
    main()