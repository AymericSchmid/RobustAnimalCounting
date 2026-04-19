from pathlib import Path
import shutil
import yaml


def export_to_yolo(dataset, dataset_dir):
    for i in range(len(dataset)):
        sample = dataset[i]
        image_id = sample["image_id"]
        image_path = sample["path"]
        target = sample["target"]
        split = sample["split"]
        extension = Path(image_path).suffix

        image_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        # copy image to output folder
        dst_image_path = image_dir / f"{image_id}{extension}"
        shutil.copy(image_path, dst_image_path)

        # create annotation file in YOLO normalized xywh format
        boxes = target.get("boxes")
        if boxes is not None and len(boxes) > 0:
            image_height, image_width = target["image_size"]
            label_path = labels_dir / f"{image_id}.txt"

            with open(label_path, "w") as f:
                for box, label in zip(target["boxes"], target["labels"]):
                    x1, y1, x2, y2 = [float(v) for v in box]

                    class_idx = 0  # all animals mapped to class index 0 in YOLO
                    x_center = ((x1 + x2) / 2.0) / float(image_width)
                    y_center = ((y1 + y2) / 2.0) / float(image_height)
                    width = (x2 - x1) / float(image_width)
                    height = (y2 - y1) / float(image_height)

                    # Clamp values so labels always stay in the valid YOLO range.
                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    width = min(max(width, 0.0), 1.0)
                    height = min(max(height, 0.0), 1.0)

                    f.write(
                        f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

        # create the yaml file for YOLOv8
        yaml_path = dataset_dir / "data.yaml"

        data = {
            "path": str(dataset_dir),
            "train": "images" + "/train",
            "val": "images" + "/val",
            "test": "images" + "/test",
            "nc": 1,             # only one class in YOLO
            "names": ["animal"]  # name for the single class
        }

        with open(yaml_path, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        