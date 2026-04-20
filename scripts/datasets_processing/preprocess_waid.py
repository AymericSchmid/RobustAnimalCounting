"""
Preprocess the WAID dataset.

WAID comes in standard YOLO format:
    data/raw/waid/
    ├── images/
    │   ├── train/   (jpg files)
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/   (txt files, one per image)
        ├── val/
        └── test/

Each .txt label file has lines like:
    class_id  center_x  center_y  width  height
    (all normalized 0-1 relative to image size)

This script reads everything and produces:
    data/splits/waid/
    ├── images/        (all images copied here, flat)
    └── annotations.csv   (one row per bounding box, with absolute pixel coords)

The annotations.csv has columns:
    image_path, x1, y1, x2, y2, species, split

This matches the exact same format as Eikelboom and Delplanque,
so the same dataset wrapper pattern works.
"""

import shutil
from pathlib import Path
from PIL import Image
import pandas as pd

# ── Adjust these paths to your setup ─────────────────────────────
RAW_DIR = Path("data/raw/waid/WAID")
OUT_DIR = Path("data/splits/waid")
# ─────────────────────────────────────────────────────────────────

# WAID species mapping (class_id → species name)
# From the WAID paper: 0=sheep, 1=cattle, 2=seal, 3=camel, 4=kiang, 5=zebra
SPECIES_MAP = {
    0: "sheep",
    1: "cattle",
    2: "seal",
    3: "camelus",
    4: "kiang",
    5: "zebra",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_out = OUT_DIR / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows = []

    # WAID uses "valid" folder name, but the project standardizes on "val"
    SPLIT_FOLDER_MAP = {
        "train": "train",
        "val": "valid",    # folder is called "valid", we store as "val"
        "test": "test",
    }

    for split_name, folder_name in SPLIT_FOLDER_MAP.items():
        img_folder = RAW_DIR / "images" / folder_name
        lbl_folder = RAW_DIR / "labels" / folder_name

        if not img_folder.exists():
            print(f"Warning: {img_folder} does not exist, skipping.")
            continue

        for img_path in sorted(img_folder.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Copy image to flat output folder
            dst = img_out / img_path.name
            if not dst.exists():
                shutil.copy(img_path, dst)

            # Read the corresponding label file
            lbl_path = lbl_folder / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                # Image with no annotations (background image) — skip or keep
                continue

            # We need the image size to convert normalized → absolute coords
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            with open(lbl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h

                    # Convert center+size → corner coords (x1, y1, x2, y2)
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    species = SPECIES_MAP.get(class_id, f"class_{class_id}")

                    rows.append({
                        "image_path": img_path.name,
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "species": species,
                        "split": split_name,
                    })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "annotations.csv"
    df.to_csv(out_csv, index=False, header=False)

    # Print summary
    print(f"Done! Saved {len(df)} annotations to {out_csv}")
    for split in ["train", "val", "test"]:
        n = len(df[df["split"] == split])
        imgs = df[df["split"] == split]["image_path"].nunique()
        print(f"  {split}: {n} boxes across {imgs} images")


if __name__ == "__main__":
    main()