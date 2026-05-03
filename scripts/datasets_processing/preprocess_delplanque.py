import shutil
from pathlib import Path
import pandas as pd

from tiling_utils import tile_image_and_annotations

SPLITS = ("train", "val", "test")


def _load_split_annotations(raw_dir: Path, split: str) -> pd.DataFrame:
    csv_path = raw_dir / "groundtruth" / "csv" / f"{split}_big_size_A_B_E_K_WH_WB.csv"

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    df = df.rename(columns={"Image": "image_path", "Label": "species"})
    df["image_path"] = df["image_path"].astype(str).map(lambda p: Path(p).name)
    df["split"] = split

    return df[["image_path", "x1", "y1", "x2", "y2", "species", "split"]]


def _copy_split_images(raw_dir: Path, out_images_dir: Path, annotations: pd.DataFrame) -> int:
    copied = 0
    missing_files = []

    for split in SPLITS:
        split_df = annotations[annotations["split"] == split]
        split_images = split_df["image_path"].drop_duplicates().tolist()

        for image_name in split_images:
            src_path = raw_dir / split / image_name
            dst_path = out_images_dir / image_name

            if not src_path.exists():
                missing_files.append(str(src_path))
                continue

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} image files listed in annotations were not found. "
            f"First missing files:\n{preview}"
        )

    return copied

def main():
    ROOT = Path(__file__).resolve().parents[2]
    RAW_DIR = ROOT / "data/raw/delplanque/general_dataset"
    OUT_DIR = ROOT / "data/splits/delplanque"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR = OUT_DIR / "images"
    if IMG_DIR.exists():
        for path in IMG_DIR.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    annotations = []
    for split in SPLITS:
        split_df = _load_split_annotations(RAW_DIR, split)
        annotations.append(split_df)

    all_annotations = pd.concat(annotations, ignore_index=True)

    # perform tiling instead of simple copying
    df_tiled_rows = []

    for split in SPLITS:
        split_ann = all_annotations[all_annotations["split"] == split]
        split_images = split_ann["image_path"].drop_duplicates().tolist()

        for image_name in split_images:
            src_path = RAW_DIR / split / image_name

            if not src_path.exists():
                print(f"Warning: {src_path} not found")
                continue

            # select annotations for this image
            ann_rows = split_ann[split_ann["image_path"] == image_name]

            try:
                # perform tiling and collect annotations
                tiled_anns = tile_image_and_annotations(
                    src_image_path=src_path,
                    annotations=ann_rows,
                    image_name=image_name,
                    output_dir=OUT_DIR,
                    split=split,
                    tile_size=1024,
                    overlap=0.2,
                    save_empty_tiles=False,
                    bbox_columns=("x1", "y1", "x2", "y2"),
                )
                df_tiled_rows.extend(tiled_anns)
            except (OSError, IOError) as e:
                print(f"Warning: could not process {image_name}: {e}")
                continue

    out_ann_path = OUT_DIR / "annotations.csv"
    if df_tiled_rows:
        df_tiled = pd.DataFrame(df_tiled_rows)
        df_tiled.to_csv(out_ann_path, index=False)
    else:
        print("Warning: no tiled annotations generated")

    print("Delplanque preprocessing completed.")
    print(f"Annotations saved to: {out_ann_path}")
    print(f"Images directory: {IMG_DIR}")

    for split in SPLITS:
        if df_tiled_rows:
            split_ann = [r for r in df_tiled_rows if r["split"] == split]
            split_images = set(r["image_path"] for r in split_ann)
            print(
                f"- {split}: {len(split_images)} tiles, "
                f"{len(split_ann)} annotations"
            )
        else:
            print(f"- {split}: (no data)")
    
if __name__ == "__main__":
    main()