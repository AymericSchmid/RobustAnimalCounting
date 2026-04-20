"""
Preprocess the Qian Penguins dataset.

Raw data (from Dryad: https://doi.org/10.5061/dryad.8931zcrv8):
    data/raw/qian_penguins/
    ├── Jack.zip              (images for Jack colony)
    ├── JACK_export-*.json    (Labelbox annotations)
    ├── Luke.zip
    ├── LUKE_export-*.json
    ├── Maisie.zip
    ├── MAISIE_export-*.json
    ├── Thomas.zip
    └── THOMAS_export-*.json

Each JSON is a Labelbox export: a list of entries with:
  - "External ID": image filename
  - "Label": {"objects": [{"bbox": {"top", "left", "height", "width"}, ...}]}

The bboxes are tiny (~5x5 px) — essentially point annotations on penguin heads.
We extract the center of each bbox as a point.

Output:
    data/splits/qian_penguins/
    ├── images/
    └── annotations.csv   (image_path, x, y, species, split)

NOTE: This dataset uses POINTS, not boxes. The annotations.csv format
is therefore different from Eikelboom/WAID: columns are x, y (not x1,y1,x2,y2).
"""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Adjust these paths to your setup ─────────────────────────────
RAW_DIR = Path("data/raw/qian_penguins")
OUT_DIR = Path("data/splits/qian_penguins")
# ─────────────────────────────────────────────────────────────────

# Map colony names to their JSON and zip files
# The JSON filenames have timestamps — we match by prefix
COLONIES = ["Jack", "Luke", "Maisie", "Thomas"]


def find_file(directory, prefix, suffix):
    """Find a file matching a prefix and suffix (case-insensitive on prefix)."""
    for f in directory.iterdir():
        if f.name.upper().startswith(prefix.upper()) and f.name.endswith(suffix):
            return f
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_out = OUT_DIR / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows = []

    for colony in COLONIES:
        print(f"\n--- Processing {colony} colony ---")

        # Find the JSON annotation file
        json_file = find_file(RAW_DIR, colony, ".json")
        if json_file is None:
            # Try with uppercase prefix (JACK_export-...)
            json_file = find_file(RAW_DIR, colony.upper(), ".json")
        if json_file is None:
            print(f"  WARNING: No JSON file found for {colony}, skipping.")
            continue
        print(f"  JSON: {json_file.name}")

        # Find and extract the zip file
        zip_file = find_file(RAW_DIR, colony, ".zip")
        if zip_file is None:
            print(f"  WARNING: No zip file found for {colony}, skipping.")
            continue
        print(f"  ZIP: {zip_file.name}")

        # Extract zip to a temporary folder
        extract_dir = RAW_DIR / f"{colony}_images"
        if not extract_dir.exists():
            print(f"  Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(extract_dir)
        else:
            print(f"  Already extracted to {extract_dir}")

        # Read JSON annotations
        with open(json_file, "r") as f:
            data = json.load(f)
        print(f"  {len(data)} images in JSON")

        n_annotations = 0
        n_images_with_annotations = 0

        for entry in data:
            image_name = entry.get("External ID", "")
            if not image_name:
                continue

            label = entry.get("Label", {})
            if not isinstance(label, dict) or "objects" not in label:
                # Empty/skipped image — skip
                continue

            objects = label["objects"]
            if len(objects) == 0:
                continue

            # Find the actual image file (might be in a subfolder)
            img_path = None
            for candidate in extract_dir.rglob(image_name):
                img_path = candidate
                break

            if img_path is None:
                # Try without the full path, just filename
                for candidate in extract_dir.rglob("*.png"):
                    if candidate.name == image_name:
                        img_path = candidate
                        break

            if img_path is None:
                continue

            # Copy image to output folder (prefix with colony to avoid name clashes)
            out_name = f"{colony}_{image_name}"
            dst = img_out / out_name
            if not dst.exists():
                shutil.copy(img_path, dst)

            n_images_with_annotations += 1

            for obj in objects:
                bbox = obj.get("bbox", {})
                if not bbox:
                    continue

                # Extract center point from the tiny bbox
                x = bbox["left"] + bbox["width"] / 2.0
                y = bbox["top"] + bbox["height"] / 2.0

                rows.append({
                    "image_path": out_name,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "species": "penguin",
                    "colony": colony,
                })
                n_annotations += 1

        print(f"  {n_annotations} point annotations across {n_images_with_annotations} images")

    # Create train/val/test splits (70/10/20)
    print("\n--- Creating splits ---")
    df = pd.DataFrame(rows)
    images = df["image_path"].unique()

    rng = np.random.RandomState(42)
    rng.shuffle(images)

    n = len(images)
    train_imgs = set(images[: int(0.7 * n)])
    val_imgs = set(images[int(0.7 * n) : int(0.8 * n)])
    test_imgs = set(images[int(0.8 * n) :])

    def assign_split(img):
        if img in train_imgs:
            return "train"
        elif img in val_imgs:
            return "val"
        else:
            return "test"

    df["split"] = df["image_path"].apply(assign_split)

    # Save
    out_csv = OUT_DIR / "annotations.csv"
    df.to_csv(out_csv, index=False, header=False)

    print(f"\nDone! Saved {len(df)} annotations to {out_csv}")
    for split in ["train", "val", "test"]:
        n_ann = len(df[df["split"] == split])
        n_img = df[df["split"] == split]["image_path"].nunique()
        print(f"  {split}: {n_ann} points across {n_img} images")


if __name__ == "__main__":
    main()