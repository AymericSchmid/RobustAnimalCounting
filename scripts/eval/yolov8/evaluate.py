"""Evaluate a trained YOLOv8 checkpoint on one of the project datasets.

Usage (from the repo root):

    # In-domain (trained and tested on Eikelboom, with density buckets)
    PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \\
        --train-dataset eikelboom \\
        --test-dataset eikelboom \\
        --weights results/yolov8/eikelboom/weights/best.pt \\
        --mode density

    # Force regenerate the per-bucket yamls (e.g. after changing split_density)
    PYTHONPATH=src python scripts/eval/yolov8/evaluate.py \\
        --train-dataset eikelboom \\
        --test-dataset eikelboom \\
        --weights results/yolov8/eikelboom/weights/best.pt \\
        --mode density \\
        --rebuild-buckets

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
import yaml

from animal_counting.datasets.delplanque import DelplanqueDataset
from animal_counting.datasets.eikelboom import EikelboomDataset
from animal_counting.datasets.qian_penguins import QianPenguinsDataset
from animal_counting.datasets.waid import WAIDDataset
from animal_counting.datasets.aed import AEDDataset
from animal_counting.evaluation import (
    evaluate_yolo_cross,
    evaluate_yolo_density,
    split_by_density,
)
from animal_counting.models.yolov8 import YOLOv8CountingModel


DATASET_REGISTRY = {
    "eikelboom": EikelboomDataset,
    "delplanque": DelplanqueDataset,
    "waid": WAIDDataset,
    "qian_penguins": QianPenguinsDataset,
    "aed": AEDDataset,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 on an animal counting dataset")
    p.add_argument("--train-dataset", choices=list(DATASET_REGISTRY.keys()),
                   help="Name of the dataset used for training")
    p.add_argument("--test-dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--weights", required=True, help="Path to weights of the trained model (best.pt checkpoint)")
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--mode", choices=["density", "cross"], default="density",
                   help="'density' = per-bucket counting metrics (in-domain). "
                        "'cross' = overall-only (cross-dataset generalization).")
    p.add_argument("--rebuild-buckets", action="store_true",
                   help="Force regeneration of the per-bucket yamls, even if they "
                        "already exist on disk. Use this after changing split_density().")
    return p.parse_args()


def run_inference(model, dataset, conf, iou, imgsz):
    """Loop over the dataset, call model.predict() per image, collect counts
    AND image paths so we can build per-bucket yamls later."""
    image_ids, pred_counts, gt_counts, image_paths = [], [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        pred = model.predict(
            sample["path"], conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )

        image_ids.append(sample["image_id"])
        pred_counts.append(int(pred.count))
        gt_counts.append(int(sample["target"]["count"]))
        image_paths.append(sample["path"])

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(dataset)}] processed")

    return image_ids, pred_counts, gt_counts, image_paths


def get_or_make_bucket_yaml(data_yaml, split, image_ids, bucket_root, bucket_name, force=False):
    """Return path to a per-bucket data.yaml, creating it only if missing.

    Writes two files under bucket_root:
      - {split}_{bucket}.txt : one image path per line (absolute paths, pointing
        into the YOLO dataset structure so labels are auto-resolved)
      - {split}_{bucket}.yaml: data.yaml pointing to the .txt for this split

    YOLO auto-resolves labels via the /images/ -> /labels/ convention on each
    listed path, so the paths must point into the YOLO dataset structure
    (data/yolo/<dataset>/images/<split>/...), NOT into the Python dataset
    structure (data/splits/...).

    args:
        data_yaml: path to the original data.yaml (used as template)
        split: "train", "val", or "test"
        image_ids: list of image IDs (stems, no extension) for this bucket
        bucket_root: directory where the bucket yamls live (persisted on disk)
        bucket_name: name of the bucket (used as filename suffix)
        force: if True, regenerate the files even if they already exist
    """
    bucket_root.mkdir(parents=True, exist_ok=True)
    list_file = bucket_root / f"{split}_{bucket_name}.txt"
    bucket_yaml = bucket_root / f"{split}_{bucket_name}.yaml"

    if bucket_yaml.exists() and list_file.exists() and not force:
        return bucket_yaml

    # Load the original yaml as template (names, nc, etc.)
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    # Resolve the YOLO images dir from the original yaml so we point to the
    # right place (data/yolo/...), where labels exist alongside.
    base = Path(cfg.get("path", data_yaml.parent)).resolve()
    yolo_images_dir = (base / cfg[split]).resolve()

    # Build {stem -> full path} for all images in the YOLO split dir
    stem_to_path = {p.stem: p for p in yolo_images_dir.iterdir() if p.is_file()}
    bucket_paths = [stem_to_path[i] for i in image_ids if i in stem_to_path]

    # Write the image list
    list_file.write_text("\n".join(str(p) for p in bucket_paths))

    # Build the per-bucket yaml: same nc/names/train/val/path as the original,
    # but the `split` entry points to our list file (absolute path). Ultralytics
    # requires `train` and `val` keys to be present even when validating on
    # `test`, so we leave them untouched.
    new_cfg = {**cfg, split: str(list_file)}

    with open(bucket_yaml, "w") as f:
        yaml.safe_dump(new_cfg, f)

    return bucket_yaml


def val_to_dict(m):
    """Extract a plain dict of detection metrics from Ultralytics val() output."""
    return {
        "mAP@0.5": float(m.box.map50),
        "mAP@0.5:0.95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }


def print_summary(detection_results, counting_results):
    print("\n=== Detection metrics (Ultralytics val) ===")
    for bucket, m in detection_results.items():
        if m is None:
            print(f"\n[{bucket}]  (empty — skipped)")
            continue
        print(f"\n[{bucket}]")
        for k, v in m.items():
            print(f"  {k:16s} {v:.4f}")

    print("\n=== Counting metrics ===")
    for bucket, m in counting_results.items():
        print(f"\n[{bucket}]  n={m['n_images']}")
        for k in ("MAE", "RMSE", "relative_error"):
            print(f"  {k:16s} {m[k]:.4f}")


def save_results(out_path, args, detection_results, counting_results):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "weights": args.weights,
            "split": args.split,
            "mode": args.mode,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        },
        "detection": detection_results,
        "counting": counting_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Evaluation parameters ===")
    print(f"  Train dataset: {args.train_dataset}")
    print(f"  Test dataset:  {args.test_dataset} (split={args.split})")
    print(f"  Weights:       {args.weights}")
    print(f"  Mode:          {args.mode}")
    if args.rebuild_buckets:
        print("  Rebuild buckets: yes")

    # Load dataset + model
    dataset = DATASET_REGISTRY[args.test_dataset](
        root=ROOT / "data" / "splits" / args.test_dataset, split=args.split
    )
    print(f"Loaded {len(dataset)} samples")

    model = YOLOv8CountingModel(device=device, config={"model_path": args.weights})

    # Per-image inference: counts + paths for bucketing
    print("\nRunning per-image inference...")
    image_ids, pred_counts, gt_counts, image_paths = run_inference(
        model, dataset, args.conf, args.iou, args.imgsz
    )

    # Counting metrics
    print("\nComputing counting metrics...")
    if args.mode == "density":
        counting_results = evaluate_yolo_density(
            image_ids=image_ids, pred_counts=pred_counts, gt_counts=gt_counts
        )
    else:
        counting_results = evaluate_yolo_cross(
            pred_counts=pred_counts, gt_counts=gt_counts
        )

    # Overall detection metrics via Ultralytics val()
    detection_results = {}
    data_yaml = ROOT / "data" / "yolo" / args.test_dataset / "data.yaml"
    print("\nRunning Ultralytics val() — overall...")
    overall = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz)
    detection_results["overall"] = val_to_dict(overall)

    # Per-bucket detection metrics (density mode only)
    if args.mode == "density":
        buckets_ids = split_by_density(image_ids, pred_counts, gt_counts)
        bucket_root = ROOT / "data" / "yolo" / args.test_dataset / "buckets"

        print("\nRunning Ultralytics val() — per bucket...")
        for bucket_name, bucket_data in buckets_ids.items():
            ids = bucket_data["image_ids"]
            if not ids:
                print(f"  [{bucket_name}] empty — skipped")
                detection_results[bucket_name] = None
                continue
            sub_yaml = get_or_make_bucket_yaml(
                data_yaml, args.split, ids,
                bucket_root, bucket_name,
                force=args.rebuild_buckets,
            )
            print(f"  [{bucket_name}] running val() on {len(ids)} images...")
            sub = model.val(data=str(sub_yaml), split=args.split, imgsz=args.imgsz)
            detection_results[bucket_name] = val_to_dict(sub)

    # Print and save
    print_summary(detection_results, counting_results)

    out_path = (ROOT / "results" / "yolov8"
                / f"{args.train_dataset or 'unknown'}_{args.test_dataset}_{args.mode}.json")
    save_results(out_path, args, detection_results, counting_results)


if __name__ == "__main__":
    main()