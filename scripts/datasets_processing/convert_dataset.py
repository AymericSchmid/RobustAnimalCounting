from __future__ import annotations

import argparse
from pathlib import Path

from animal_counting.datasets.eikelboom import EikelboomDataset
from animal_counting.datasets.delplanque import DelplanqueDataset
from animal_counting.datasets.waid import WAIDDataset
from animal_counting.datasets.qian_penguins import QianPenguinsDataset
from animal_counting.datasets.converters import export_to_yolo

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a wrapped dataset to a model-specific format.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, for example: eikelboom")
    parser.add_argument("--format", type=str, required=True, choices=["yolo"], help="Target format")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the wrapped dataset")
    parser.add_argument("--output", type=str, required=True, help="Output directory for the converted dataset",)
    return parser.parse_args()

def get_dataset(name, root, split):
    if name == "eikelboom":
        return EikelboomDataset(root, split=split)
    elif name == "delplanque":
        return DelplanqueDataset(root, split=split)
    elif name == "waid":
        return WAIDDataset(root, split=split)
    elif name == "qian_penguins":
        return QianPenguinsDataset(root, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    
def get_labels_map(name):
    return 0 # all animals mapped to label 0 for counting task

def convert_to_yolo(dataset_name, root, output):
    dataset_dir = Path(output)
    image_dir = dataset_dir / "images"

    for split in ["train", "val", "test"]:
        (image_dir / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        dataset = get_dataset(dataset_name, root=root, split=split)
        export_to_yolo(dataset, dataset_dir)

    print(f"Conversion to YOLO format completed. Output saved to: {dataset_dir}")

def main():
    args = parse_args()

    if args.format == "yolo":
        convert_to_yolo(
            dataset_name=args.dataset,
            root=args.root,
            output=args.output,
        )
    else:
        raise ValueError(f"Unsupported format: {args.format}")

if __name__ == "__main__":
    main()