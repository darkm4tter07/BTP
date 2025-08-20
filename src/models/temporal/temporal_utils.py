"""
Utility functions for temporal analysis and activity classification.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from collections import Counter
import cv2

class TemporalUtils:
    """Utility functions for temporal analysis."""
    
    @staticmethod
    def visualize_activity_timeline(activities: List[str], 
                                   confidences: List[float],
                                   timestamps: List[float] = None,
                                   save_path: str = None) -> None:
        """
        Visualize activity timeline with confidence scores.
        
        Args:
            activities: List of predicted activities
            confidences: Confidence scores for each activity
            timestamps: Optional timestamps for each prediction
            save_path: Optional path to save the plot
        """
        if timestamps is None:
            timestamps = list(range(len(activities)))
        
        # Create activity color mapping
        unique_activities = list(set(activities))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_activities)))
        activity_colors = dict(zip(unique_activities, colors))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Plot 1: Activity timeline
        for i, (activity, confidence, timestamp) in enumerate(zip(activities, confidences, timestamps)):
            color = activity_colors[activity]
            ax1.barh(0, 1, left=timestamp, height=0.5, 
                    color=color, alpha=confidence, label=activity if i == 0 else "")
            
        ax1.set_ylabel('Activity')
        ax1.set_title('Activity Timeline')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Confidence over time
        ax2.plot(timestamps, confidences, 'b-', linewidth=2)
        ax2.fill_between(timestamps, confidences, alpha=0.3)
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Prediction Confidence Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def analyze_activity_patterns(activities: List[str]) -> Dict[str, any]:
        """
        Analyze patterns in activity sequences.
        
        Args:
            activities: List of activities in temporal order
            
        Returns:
            Dict containing pattern analysis results
        """
        if not activities:
            return {}
        
        # Activity frequency
        activity_counts = Counter(activities)
        total_activities = len(activities)
        
        # Activity transitions
        transitions = []
        for i in range(len(activities) - 1):
            transitions.append((activities[i], activities[i + 1]))
        
        transition_counts = Counter(transitions)
        
        # Activity durations (consecutive same activities)
        durations = {}
        current_activity = activities[0]
        current_duration = 1
        
        for i in range(1, len(activities)):
            if activities[i] == current_activity:
                current_duration += 1
            else:
                if current_activity not in durations:
                    durations[current_activity] = []
                durations[current_activity].append(current_duration)
                current_activity = activities[i]
                current_duration = 1
        
        # Add last duration
        if current_activity not in durations:
            durations[current_activity] = []
        durations[current_activity].append(current_duration)
        
        # Calculate statistics
        avg_durations = {}
        for activity, duration_list in durations.items():
            avg_durations[activity] = {
                'mean': np.mean(duration_list),
                'std': np.std(duration_list),
                'min': np.min(duration_list),
                'max': np.max(duration_list)
            }
        
        return {
            'activity_frequencies': dict(activity_counts),
            'activity_percentages': {k: v/total_activities*100 for k, v in activity_counts.items()},
            'common_transitions': transition_counts.most_common(10),
            'average_durations': avg_durations,
            'total_activities': total_activities,
            'unique_activities': len(activity_counts)
        }
    
    @staticmethod
    def detect_unsafe_patterns(activities: List[str], 
                              confidences: List[float],
                              unsafe_activities: List[str] = None) -> Dict[str, any]:
        """
        Detect potentially unsafe activity patterns.
        
        Args:
            activities: List of activities
            confidences: Confidence scores
            unsafe_activities: List of activities considered unsafe
            
        Returns:
            Dict containing safety analysis
        """
        if unsafe_activities is None:
            unsafe_activities = ['bending', 'lifting', 'reaching_up', 'reaching_down']
        
        safety_issues = []
        
        # Detect prolonged unsafe activities
        consecutive_unsafe = 0
        max_consecutive_unsafe = 0
        
        for i, activity in enumerate(activities):
            if activity in unsafe_activities:
                consecutive_unsafe += 1
                max_consecutive_unsafe = max(max_consecutive_unsafe, consecutive_unsafe)
                
                # Flag if unsafe activity lasts too long
                if consecutive_unsafe > 10:  # 10 frames = ~0.33 seconds
                    safety_issues.append({
                        'type': 'prolonged_unsafe_activity',
                        'activity': activity,
                        'duration': consecutive_unsafe,
                        'frame': i,
                        'confidence': confidences[i] if i < len(confidences) else 0
                    })
            else:
                consecutive_unsafe = 0
        
        # Detect rapid activity changes (might indicate loss of control)
        rapid_changes = 0
        for i in range(1, len(activities)):
            if activities[i] != activities[i-1]:
                rapid_changes += 1
        
        change_rate = rapid_changes / len(activities) if activities else 0
        
        if change_rate > 0.5:  # More than 50% frame-to-frame changes
            safety_issues.append({
                'type': 'excessive_activity_changes',
                'change_rate': change_rate,
                'description': 'Worker activities changing too frequently'
            })
        
        # Count total unsafe activities
        unsafe_count = sum(1 for activity in activities if activity in unsafe_activities)
        unsafe_percentage = unsafe_count / len(activities) * 100 if activities else 0
        
        return {
            'safety_issues': safety_issues,
            'unsafe_activity_percentage': unsafe_percentage,
            'max_consecutive_unsafe': max_consecutive_unsafe,
            'total_unsafe_activities': unsafe_count,
            'change_rate': change_rate
        }
    
    @staticmethod
    def create_pose_heatmap(pose_sequences: np.ndarray, save_path: str = None) -> None:
        """
        Create heatmap visualization of pose keypoint activations.
        
        Args:
            pose_sequences: Array of shape (num_sequences, sequence_length, 75)
            save_path: Optional path to save the heatmap
        """
        if len(pose_sequences.shape) != 3:
            raise ValueError("pose_sequences should have shape (num_sequences, sequence_length, 75)")
        
        # Calculate average activation for each keypoint across all sequences
        avg_poses = np.mean(pose_sequences, axis=(0, 1))  # Shape: (75,)
        
        # Reshape to (25, 3) for 25 keypoints with x,y,z coordinates
        keypoint_data = avg_poses.reshape(25, 3)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(keypoint_data.T, 
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   xticklabels=[f'Keypoint_{i}' for i in range(25)],
                   yticklabels=['X', 'Y', 'Z'])
        
        plt.title('Average Pose Keypoint Activations')
        plt.xlabel('Body Keypoints')
        plt.ylabel('Coordinates')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()