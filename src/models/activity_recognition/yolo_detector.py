from ultralytics import YOLO
import cv2
import numpy as np

class ConstructionYOLO:
    def __init__(self, model_path=None):
        """Initialize YOLO model for construction site detection."""
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')  # Pre-trained model
        
        # Construction-specific classes
        self.construction_classes = {
            'person': 0,
            'hard_hat': 1,
            'safety_vest': 2,
            'tools': 3,
            'materials': 4
        }
    
    def detect(self, frame, confidence=0.5):
        """Detect objects in frame."""
        results = self.model(frame, conf=confidence)
        return results
    
    def annotate_frame(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        annotated_frame = results[0].plot()
        return annotated_frame