from src.models.temporal.advanced_classifier import TemporalActivityClassifier
from src.models.temporal.activity_buffer import ActivityBuffer
from src.models.temporal.temporal_utils import TemporalUtils

class RealTimeProcessor:
    def __init__(self):
        # Your existing code...
        
        # Add temporal analysis
        self.temporal_classifier = TemporalActivityClassifier()
        self.activity_buffer = ActivityBuffer()
        self.temporal_utils = TemporalUtils()
    
    def process_frame(self, frame, worker_detections):
        # Your existing YOLO + pose estimation...
        
        for worker_id, pose_data in worker_detections.items():
            # Add pose to temporal buffer
            self.activity_buffer.add_pose(worker_id, pose_data['landmarks'])
            
            # Predict activity if enough frames
            if self.activity_buffer.is_ready_for_prediction(worker_id):
                sequence = self.activity_buffer.get_sequence(worker_id)
                self.temporal_classifier.pose_buffer = deque(sequence, maxlen=30)
                
                activity, confidence = self.temporal_classifier.predict_activity()
                
                if activity:
                    print(f"Worker {worker_id}: {activity} (confidence: {confidence:.2f})")
                    self.activity_buffer.add_activity_prediction(worker_id, activity, confidence)