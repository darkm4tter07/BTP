import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class AdvancedPoseEstimator:
    """Enhanced pose estimation with 3D capabilities and ergonomic analysis"""
    
    def __init__(self, model_complexity=2, min_detection_confidence=0.7):
        """
        Initialize with latest MediaPipe Pose
        model_complexity: 0, 1, or 2 (2 is most accurate)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize with best settings
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=True,  # Get body segmentation mask
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Define body part groups for ergonomic analysis
        self.body_parts = {
            'head': [0, 1, 2, 3, 4, 7, 8, 9, 10],
            'torso': [11, 12, 23, 24],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }
        
        # Joint angle thresholds for risk assessment
        self.risk_thresholds = {
            'back_bend': 20,    # degrees
            'neck_bend': 25,
            'arm_elevation': 60,
            'knee_bend': 90,
            'ankle_flex': 15
        }
    
    def estimate_pose(self, frame):
        """Enhanced pose estimation with 3D coordinates"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract 3D landmarks
            landmarks_3d = self._extract_3d_landmarks(results.pose_landmarks)
            
            # Calculate joint angles
            joint_angles = self._calculate_joint_angles(landmarks_3d)
            
            # Assess ergonomic risks
            risk_scores = self._assess_ergonomic_risks(joint_angles)
            
            return {
                'landmarks_3d': landmarks_3d,
                'joint_angles': joint_angles,
                'risk_scores': risk_scores,
                'segmentation_mask': results.segmentation_mask,
                'raw_results': results
            }
        return None
    
    def _extract_3d_landmarks(self, landmarks):
        """Extract 3D coordinates from MediaPipe landmarks"""
        landmarks_array = []
        for landmark in landmarks.landmark:
            landmarks_array.append([
                landmark.x,  # Normalized x coordinate
                landmark.y,  # Normalized y coordinate
                landmark.z,  # Depth (relative to hip)
                landmark.visibility  # Visibility score
            ])
        return np.array(landmarks_array)
    
    def _calculate_joint_angles(self, landmarks_3d):
        """Calculate important joint angles for ergonomic assessment"""
        angles = {}
        
        # Back angle (spine)
        if self._landmarks_visible(landmarks_3d, [11, 12, 23, 24]):
            shoulder_center = (landmarks_3d[11] + landmarks_3d[12]) / 2
            hip_center = (landmarks_3d[23] + landmarks_3d[24]) / 2
            vertical = np.array([0, 1, 0])
            spine_vector = shoulder_center[:3] - hip_center[:3]
            angles['back_angle'] = self._angle_between_vectors(spine_vector, vertical)
        
        # Neck angle
        if self._landmarks_visible(landmarks_3d, [0, 11, 12]):
            nose = landmarks_3d[0][:3]
            shoulder_center = (landmarks_3d[11] + landmarks_3d[12]) / 2
            neck_vector = nose - shoulder_center[:3]
            vertical = np.array([0, -1, 0])
            angles['neck_angle'] = self._angle_between_vectors(neck_vector, vertical)
        
        # Arm elevation angles
        for side, shoulder_idx, elbow_idx in [('left', 11, 13), ('right', 12, 14)]:
            if self._landmarks_visible(landmarks_3d, [shoulder_idx, elbow_idx]):
                shoulder = landmarks_3d[shoulder_idx][:3]
                elbow = landmarks_3d[elbow_idx][:3]
                arm_vector = elbow - shoulder
                horizontal = np.array([1, 0, 0])
                angles[f'{side}_arm_elevation'] = self._angle_between_vectors(arm_vector, horizontal)
        
        return angles
    
    def _assess_ergonomic_risks(self, joint_angles):
        """Assess ergonomic risks based on joint angles"""
        risk_scores = {}
        
        for angle_name, angle_value in joint_angles.items():
            if 'back_angle' in angle_name:
                risk_scores['back_risk'] = self._calculate_risk_score(
                    angle_value, self.risk_thresholds['back_bend']
                )
            elif 'neck_angle' in angle_name:
                risk_scores['neck_risk'] = self._calculate_risk_score(
                    angle_value, self.risk_thresholds['neck_bend']
                )
            elif 'arm_elevation' in angle_name:
                risk_scores[f'{angle_name}_risk'] = self._calculate_risk_score(
                    angle_value, self.risk_thresholds['arm_elevation']
                )
        
        # Overall risk score
        if risk_scores:
            risk_scores['overall_risk'] = np.mean(list(risk_scores.values()))
        
        return risk_scores
    
    def _landmarks_visible(self, landmarks_3d, indices, threshold=0.5):
        """Check if landmarks are visible enough for analysis"""
        return all(landmarks_3d[i][3] > threshold for i in indices)
    
    def _angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _calculate_risk_score(self, angle, threshold):
        """Calculate risk score based on angle deviation from safe range"""
        if angle < threshold:
            return 1.0  # Low risk
        elif angle < threshold * 1.5:
            return 2.0  # Medium risk
        elif angle < threshold * 2:
            return 3.0  # High risk
        else:
            return 4.0  # Very high risk
    
    def visualize_pose_with_risks(self, frame, pose_data):
        """Enhanced visualization with risk indicators"""
        if pose_data is None:
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            pose_data['raw_results'].pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Add risk indicators
        risk_scores = pose_data['risk_scores']
        y_offset = 30
        
        for risk_name, risk_score in risk_scores.items():
            color = self._get_risk_color(risk_score)
            text = f"{risk_name}: {risk_score:.1f}"
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return annotated_frame
    
    def _get_risk_color(self, risk_score):
        """Get color based on risk level"""
        if risk_score <= 1.5:
            return (0, 255, 0)  # Green - Low risk
        elif risk_score <= 2.5:
            return (0, 255, 255)  # Yellow - Medium risk
        elif risk_score <= 3.5:
            return (0, 165, 255)  # Orange - High risk
        else:
            return (0, 0, 255)  # Red - Very high risk