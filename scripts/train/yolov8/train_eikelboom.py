import yaml
from pathlib import Path
import torch

from animal_counting.models.yolov8 import YOLOv8CountingModel

def main():
    ROOT = Path(__file__).resolve().parents[3]

    CFG = {
        "MODEL": "yolov8s.pt",
        "DATA": str(ROOT / "data" / "yolo" / "eikelboom" / "data.yaml"),
        "PROJECT": str(ROOT / "results" / "yolov8"),
        "NAME": "eikelboom",
        "EPOCHS": 100,
        "BATCH": -1,  # Use default batch size
        "IMGSZ": 1280,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    }

    AUGMENTATION_CONFIG = {
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.2,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.3,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "cutmix": 0.0,
    }
    
    print(CFG)

    model = YOLOv8CountingModel(device=CFG["DEVICE"], config={"model_path": CFG["MODEL"]})

    print(f"Starting training with data config: {CFG['DATA']}, epochs: {CFG['EPOCHS']}, batch size: {CFG['BATCH']}, image size: {CFG['IMGSZ']}")
    results = model.fit(
        data=CFG["DATA"],
        imgsz=CFG["IMGSZ"],
        epochs=CFG["EPOCHS"],
        batch=CFG["BATCH"],
        project=CFG["PROJECT"],
        name=CFG["NAME"],
        **AUGMENTATION_CONFIG,
    )

    print(f"Training completed. Results: {results}")

if __name__ == "__main__":
    main()