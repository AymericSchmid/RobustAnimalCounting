import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tiling_utils import tile_image_and_annotations


def _load_points(csv_path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"image_name", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    result = df[["image_name", "x", "y"]].copy()
    result["image_name"] = result["image_name"].astype(str).map(lambda n: f"{n}.jpg")
    result["species"] = "elephant"
    result["split"] = split
    return result


def _build_split_map(train_df: pd.DataFrame, val_ratio: float, seed: int) -> dict[str, str]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    train_images = train_df["image_name"].drop_duplicates().to_numpy()
    if len(train_images) < 2:
        raise ValueError("Not enough training images to create a validation split.")

    rng = np.random.RandomState(seed)
    rng.shuffle(train_images)

    n_val = max(1, int(round(len(train_images) * val_ratio)))
    n_val = min(n_val, len(train_images) - 1)

    val_set = set(train_images[:n_val])

    split_map: dict[str, str] = {}
    for image_name in train_images:
        split_map[image_name] = "val" if image_name in val_set else "train"
    return split_map


def _copy_all_images(
    train_images_dir: Path,
    test_images_dir: Path,
    out_images_dir: Path,
    expected_images: set[str],
) -> int:
    out_images_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = set(expected_images)

    for src_dir in (train_images_dir, test_images_dir):
        if not src_dir.exists():
            continue

        for src_path in src_dir.iterdir():
            if src_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            image_name = src_path.name
            if image_name not in expected_images:
                continue

            dst_path = out_images_dir / image_name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

            missing.discard(image_name)

    if missing:
        preview = "\n".join(sorted(missing)[:10])
        raise FileNotFoundError(
            f"{len(missing)} expected images are missing in train/test directories. "
            f"First missing files:\n{preview}"
        )

    return copied


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data/raw/aed/aerial-elephant-dataset/aerial-elephant-dataset"
    out_dir = root / "data/splits/aed"

    train_points_csv = raw_dir / "training_elephants.csv"
    test_points_csv = raw_dir / "test_elephants.csv"

    train_images_dir = raw_dir / "training_images"
    test_images_dir = raw_dir / "test_images"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    train_points = _load_points(train_points_csv, split="train")
    test_points = _load_points(test_points_csv, split="test")

    split_map = _build_split_map(train_points, val_ratio=0.1, seed=42)
    train_points["split"] = train_points["image_name"].map(split_map)

    all_points = pd.concat([train_points, test_points], ignore_index=True)

    all_points = all_points.rename(columns={"image_name": "image_path"})
    all_points = all_points[["image_path", "x", "y", "species", "split"]]

    # perform tiling instead of simple copying
    df_tiled_rows = []

    for split in ("train", "val", "test"):
        split_points = all_points[all_points["split"] == split]
        split_images = split_points["image_path"].drop_duplicates().tolist()

        for image_name in split_images:
            # find image in raw directory
            src_path = None
            for src_dir in (train_images_dir, test_images_dir):
                candidate = src_dir / image_name
                if candidate.exists():
                    src_path = candidate
                    break

            if src_path is None:
                print(f"Warning: {image_name} not found in raw directories")
                continue

            # select annotations for this image
            ann_rows = split_points[split_points["image_path"] == image_name]

            try:
                # perform tiling and collect annotations (using point columns)
                tiled_anns = tile_image_and_annotations(
                    src_image_path=src_path,
                    annotations=ann_rows,
                    image_name=image_name,
                    output_dir=out_dir,
                    split=split,
                    tile_size=1024,
                    overlap=0.2,
                    save_empty_tiles=False,
                    bbox_columns=None,
                    point_columns=("x", "y"),
                )
                df_tiled_rows.extend(tiled_anns)
            except (OSError, IOError) as e:
                print(f"Warning: could not process {image_name}: {e}")
                continue

    out_ann_path = out_dir / "annotations.csv"
    if df_tiled_rows:
        df_tiled = pd.DataFrame(df_tiled_rows)
        df_tiled.to_csv(out_ann_path, index=False)
    else:
        print("Warning: no tiled annotations generated")

    print("AED preprocessing completed.")
    print(f"Annotations saved to: {out_ann_path}")
    print(f"Images directory: {out_images_dir}")

    for split in ("train", "val", "test"):
        if df_tiled_rows:
            split_ann = [r for r in df_tiled_rows if r["split"] == split]
            split_images = set(r["image_path"] for r in split_ann)
            print(
                f"- {split}: {len(split_images)} tiles, "
                f"{len(split_ann)} points"
            )
        else:
            print(f"- {split}: (no data)")


if __name__ == "__main__":
    main()
