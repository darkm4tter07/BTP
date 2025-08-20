from ultralytics import YOLO
import cv2
import numpy as np
import torch
from pathlib import Path
import yaml

class ModernConstructionYOLO:
    """Modern YOLOv8/v11 implementation (much better than YOLOv3)"""
    
    def __init__(self, model_path=None, model_size='yolov8n'):
        """
        Initialize with modern YOLO models
        model_size options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use pre-trained model (much better than training from scratch)
            self.model = YOLO(f'{model_size}.pt')
        
        # Construction-specific classes for fine-tuning
        self.construction_classes = {
            'person': 0, 'hard_hat': 1, 'safety_vest': 2, 'gloves': 3,
            'safety_boots': 4, 'scaffolding': 5, 'crane': 6, 'excavator': 7,
            'concrete_mixer': 8, 'rebar': 9, 'formwork': 10, 'brick': 11,
            'shovel': 12, 'hammer': 13, 'drill': 14, 'measuring_tape': 15
        }
        
        # Performance settings
        self.model.fuse()  # Fuse layers for faster inference
    
    def detect_objects(self, source, conf=0.5, iou=0.4, save=False):
        """
        Modern detection with tracking capabilities
        source: image, video, or webcam
        """
        results = self.model.track(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            tracker="bytetrack.yaml",  # Modern tracking
            persist=True  # Maintain track IDs
        )
        return results
    
    def detect_frame(self, frame, conf=0.5):
        """Process single frame (for real-time applications)"""
        results = self.model(frame, conf=conf, verbose=False)
        return self._parse_results(results[0])
    
    def _parse_results(self, result):
        """Extract structured data from YOLO results"""
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                detection = {
                    'bbox': box.xyxy[0],  # x1, y1, x2, y2
                    'confidence': box.conf[0],
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])],
                    'track_id': box.id[0] if box.id is not None else None
                }
                detections.append(detection)
        return detections
    
    def train_custom_model(self, data_yaml_path, epochs=100):
        """Train on construction-specific dataset"""
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device='auto',  # Automatically use GPU if available
            project='construction_models',
            name='yolo_construction'
        )
        return results
    
    def export_optimized(self, format='onnx'):
        """Export for deployment (ONNX, TensorRT, etc.)"""
        self.model.export(format=format, dynamic=True, simplify=True)