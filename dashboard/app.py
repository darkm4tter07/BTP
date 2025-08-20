import gradio as gr
import cv2
from src.models.activity_recognition.yolo_detector import ModernConstructionYOLO
from src.models.pose_estimation.mediapipe_pose import AdvancedPoseEstimator

# Initialize models
yolo_detector = ModernConstructionYOLO()
pose_estimator = AdvancedPoseEstimator()

def process_video(video_path):
    """Process video with modern pipeline"""
    cap = cv2.VideoCapture(video_path)
    results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Object detection
        detections = yolo_detector.detect_frame(frame)
        
        # Pose estimation
        pose_data = pose_estimator.estimate_pose(frame)
        
        # Visualize
        if pose_data:
            frame = pose_estimator.visualize_pose_with_risks(frame, pose_data)
        
        results.append({
            'detections': detections,
            'pose_data': pose_data
        })
    
    cap.release()
    return results

# Create Gradio interface
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs=gr.JSON(),
    title="🏗️ Modern Construction Safety AI"
)

if __name__ == "__main__":
    demo.launch()