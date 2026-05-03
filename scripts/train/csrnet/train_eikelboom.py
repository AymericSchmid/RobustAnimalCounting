from pathlib import Path

import torch
import argparse

from animal_counting.datasets.eikelboom import EikelboomDataset
from animal_counting.models.csrnet import CSRNetCountingModel


def main():
    arg_parser = argparse.ArgumentParser(description="Train CSRNet on Eikelboom dataset")
    arg_parser.add_argument("--override_data_root", type=str, default=None, help="Override default data root")

    ROOT = Path(__file__).resolve().parents[3]
    OUTPUT_DIR = ROOT / "results" / "csrnet" / "eikelboom"
    data_root = ROOT / "data" / "splits" / "eikelboom"
    if arg_parser.parse_args().override_data_root is not None:
        data_root = Path(arg_parser.parse_args().override_data_root)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Eikelboom is a box-annotation dataset. DensityMapDataset automatically
    # converts bounding-box centroids to pseudo-points for density map generation.
    CFG = {
        "epochs": 200,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patch_size": 512,
        "density_scale": 8,
        "beta": 0.3,
        "k": 3,
        "patience": 20,
        "num_workers": 8,
    }

    print(f"Device : {DEVICE}")
    print(f"Data   : {data_root}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Config : {CFG}")

    train_dataset = EikelboomDataset(root=data_root, split="train")
    val_dataset = EikelboomDataset(root=data_root, split="val")

    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    model = CSRNetCountingModel(device=DEVICE, pretrained=True)

    results = model.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=OUTPUT_DIR,
        **CFG,
    )

    print(f"Done. Best val MAE: {results['best_val_mae']:.2f}")


if __name__ == "__main__":
    main()
