from ultralytics import YOLO

from animal_counting.models.base import BaseCountingModel, CountingParadigm,PredictionResult

class YOLOv8CountingModel(BaseCountingModel):

    def __init__(self, device, config):
        super().__init__(name="YOLOv8", paradigm=CountingParadigm.DETECTION, device=device, config=config)

        if 'model_path' not in config:
            raise ValueError("YOLOv8CountingModel requires 'model_path' in config")
        
        self.model = YOLO(config['model_path'])
        self.results = None

    def fit(self, data, imgsz=1280, epochs=100, batch=-1, **kwargs):
        """Train the YOLOv8 model using the provided configuration file."""
        self.results = self.model.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, **kwargs)
        return self.results
    
    def val(self, data, split="test", imgsz=1280, **kwargs):
        """Run Ultralytics' built-in validation.

        Returns the raw Ultralytics metrics object. Useful attributes:
        - .box.map     -> mAP@0.5:0.95
        - .box.map50   -> mAP@0.5
        - .box.mp      -> mean precision
        - .box.mr      -> mean recall
        """
        return self.model.val(data=data, split=split, imgsz=imgsz, **kwargs)

    def predict(self, image, conf=0.25, iou=0.7, imgsz=1280, verbose=False, **kwargs):
        """Run inference on one image and return a PredictionResult."""
        outputs = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=verbose,
            **kwargs,
        )
        result = outputs[0]

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        return PredictionResult(
            count=int(len(boxes_xyxy)),
            boxes=boxes_xyxy,
            scores=scores,
            labels=labels,
            metadata={"conf": conf, "iou": iou, "imgsz": imgsz},
            raw=result,
        )

    def save(self, path):
        raise NotImplementedError("Model saving not implemented for YOLOv8CountingModel as it relies on ultralytics' internal saving mechanism during training.")