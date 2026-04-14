import shutil
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("/storage/homefs/as26q834/RobustAnimalCounting/data/raw/eikelboom")
OUT_DIR = Path("/storage/homefs/as26q834/RobustAnimalCounting/data/splits/eikelboom")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # create output folders
    for split in splits:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    # copy images
    print("Copying images...")

    for img_name in images:
        found = False

        # images might be in train/val/test/ folders originally
        for folder in ["train", "val", "test"]:
            src_path = RAW_DIR / folder / img_name
            if src_path.exists():
                found = True
                
                for split, img_set in splits.items():
                    if img_name in img_set:
                        dst_path = OUT_DIR / split / img_name
                        shutil.copy(src_path, dst_path)
                        break
                break
        
        if not found:
            print(f"Warning: {img_name} not found in any source folder.")

    # add split column to annotations
    def assign_split(img):
        for split, img_set in splits.items():
            if img in img_set:
                return split
        return None
    
    df["split"] = df["image_path"].apply(assign_split)

    # save annotations
    out_ann = OUT_DIR / "annotations.csv"
    df.to_csv(out_ann, index=False)

    print("\nDone.")
    print(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()