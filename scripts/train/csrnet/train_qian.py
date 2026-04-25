from pathlib import Path

import torch

from animal_counting.datasets.qian_penguins import QianPenguinsDataset
from animal_counting.models.csrnet import CSRNetCountingModel


def main():
    ROOT = Path(__file__).resolve().parents[3]
    DATA_ROOT = ROOT / "data" / "splits" / "qian_penguins"
    OUTPUT_DIR = ROOT / "results" / "csrnet" / "qian_penguins"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CFG = {
        "epochs": 200,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patch_size": 512,
        "density_scale": 8,   # must match CSRNet frontend stride (pool1*pool2*pool3)
        "beta": 0.3,           # sigma = beta * mean_k_nearest_dist
        "k": 3,                # number of neighbours for adaptive sigma
        "patience": 20,
        "num_workers": 4,
    }

    print(f"Device : {DEVICE}")
    print(f"Data   : {DATA_ROOT}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Config : {CFG}")

    train_dataset = QianPenguinsDataset(root=DATA_ROOT, split="train")
    val_dataset = QianPenguinsDataset(root=DATA_ROOT, split="val")

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
