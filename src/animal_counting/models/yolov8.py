from ultralytics import YOLO

from animal_counting.models.base import BaseCountingModel, CountingParadigm

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
    
    def predict(self, image, **kwargs):
        pass

    def save(self, path):
        raise NotImplementedError("Model saving not implemented for YOLOv8CountingModel as it relies on ultralytics' internal saving mechanism during training.")