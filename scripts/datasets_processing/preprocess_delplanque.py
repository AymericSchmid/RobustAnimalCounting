import shutil
from pathlib import Path
import pandas as pd


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
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    annotations = []
    for split in SPLITS:
        split_df = _load_split_annotations(RAW_DIR, split)
        annotations.append(split_df)

    all_annotations = pd.concat(annotations, ignore_index=True)
    copied_count = _copy_split_images(RAW_DIR, IMG_DIR, all_annotations)

    out_ann_path = OUT_DIR / "annotations.csv"
    all_annotations.to_csv(out_ann_path, index=False)

    print("Delplanque preprocessing completed.")
    print(f"Annotations saved to: {out_ann_path}")
    print(f"Images directory: {IMG_DIR}")
    print(f"Copied {copied_count} images")

    for split in SPLITS:
        split_ann = all_annotations[all_annotations["split"] == split]
        print(
            f"- {split}: {split_ann['image_path'].nunique()} images, "
            f"{len(split_ann)} boxes"
        )
    
if __name__ == "__main__":
    main()