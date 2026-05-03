import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from tiling_utils import tile_image_and_annotations

RAW_DIR = Path("/storage/homefs/as26q834/RobustAnimalCounting/data/raw/eikelboom")
OUT_DIR = Path("/storage/homefs/as26q834/RobustAnimalCounting/data/splits/eikelboom")


def main(tile_size: int = 1024, overlap: float = 0.2, save_empty_tiles: bool = False):
    """Prepare Eikelboom splits and optionally tile images.

    Args:
        tile_size: Tile width/height in pixels.
        overlap: Fractional overlap between tiles (0.0-0.5).
        save_empty_tiles: If True, keep tiles without any annotations.
    """

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR = OUT_DIR / "images"

    # load raw annotations
    ann_path = RAW_DIR / "annotations_images.csv"
    df = pd.read_csv(ann_path, names=['image_path', 'x1', 'y1', 'x2', 'y2', 'species'], header=0)

    images = df['image_path'].unique()

    # shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(images)

    # split
    n = len(images)
    train_imgs = images[:int(0.7 * n)]
    val_imgs = images[int(0.7 * n):int(0.8 * n)]
    test_imgs = images[int(0.8 * n):]

    splits = {
        "train": set(train_imgs),
        "val": set(val_imgs),
        "test": set(test_imgs),
    }

    # create images folder (tiles will be saved here)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # collect tiled annotation rows here
    df_tiled_rows = []

    for img_name in images:
        found = False

        # images might be in train/val/test/ folders originally
        for folder in ["train", "val", "test"]:
            src_path = RAW_DIR / folder / img_name
            if src_path.exists():
                print(f"Found: {src_path}")
                found = True

                # determine split for this image
                split_col = ('train' if img_name in splits['train']
                             else 'val' if img_name in splits['val']
                             else 'test')

                # select annotations for this image
                ann_rows = df[df['image_path'] == img_name]

                try:
                    # perform tiling and collect annotations
                    tiled_anns = tile_image_and_annotations(
                        src_image_path=src_path,
                        annotations=ann_rows,
                        image_name=img_name,
                        output_dir=OUT_DIR,
                        split=split_col,
                        tile_size=tile_size,
                        overlap=overlap,
                        save_empty_tiles=save_empty_tiles,
                        bbox_columns=("x1", "y1", "x2", "y2"),
                    )
                    df_tiled_rows.extend(tiled_anns)
                except (OSError, IOError) as e:
                    print(f"Warning: could not process {img_name}: {e}")
                    continue

                break

        if not found:
            print(f"Warning: {img_name} not found in any source folder.")

    # Finalize and save tiled annotations
    if len(df_tiled_rows) > 0:
        df_tiled = pd.DataFrame(df_tiled_rows)
        out_ann = OUT_DIR / "annotations.csv"
        df_tiled.to_csv(out_ann, index=False)
        print(f"Wrote tiled annotations: {out_ann} (tiles in {IMG_DIR})")
    else:
        print("No tiled annotations were generated.")

    print("\nDone.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    # defaults: 1024 tile, 20% overlap, do not save empty tiles
    main(tile_size=1024, overlap=0.2, save_empty_tiles=False)
