"""
Activity buffer for managing pose sequences in real-time processing.
"""

from collections import deque
import numpy as np
from typing import List, Optional, Dict, Any
import threading
import time

class ActivityBuffer:
    """Thread-safe buffer for managing pose sequences."""
    
    def __init__(self, sequence_length: int = 30, max_workers: int = 10):
        """
        Initialize activity buffer for multiple workers.
        
        Args:
            sequence_length: Number of frames to keep in sequence
            max_workers: Maximum number of workers to track simultaneously
        """
        self.sequence_length = sequence_length
        self.max_workers = max_workers
        self.worker_buffers = {}  # worker_id -> deque of poses
        self.worker_metadata = {}  # worker_id -> {last_update, activity_history}
        self.lock = threading.Lock()
        
    def add_pose(self, worker_id: int, pose_landmarks: np.ndarray, timestamp: float = None) -> bool:
        """
        Add pose data for a specific worker.
        
        Args:
            worker_id: Unique identifier for the worker
            pose_landmarks: Array of pose landmarks (shape: [33, 4] or similar)
            timestamp: Optional timestamp for the pose
            
        Returns:
            bool: True if pose was added successfully
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            # Initialize buffer for new worker
            if worker_id not in self.worker_buffers:
                if len(self.worker_buffers) >= self.max_workers:
                    # Remove oldest worker if at capacity
                    oldest_worker = min(
                        self.worker_metadata.keys(),
                        key=lambda w: self.worker_metadata[w]['last_update']
                    )
                    del self.worker_buffers[oldest_worker]
                    del self.worker_metadata[oldest_worker]
                
                self.worker_buffers[worker_id] = deque(maxlen=self.sequence_length)
                self.worker_metadata[worker_id] = {
                    'last_update': timestamp,
                    'activity_history': deque(maxlen=100)  # Keep last 100 activities
                }
            
            # Process pose landmarks
            if pose_landmarks is not None and len(pose_landmarks) >= 25:
                # Take first 25 keypoints with x,y,z coordinates
                pose_vector = pose_landmarks[:25, :3].flatten()  # Shape: (75,)
                
                # Add to buffer
                self.worker_buffers[worker_id].append({
                    'pose': pose_vector,
                    'timestamp': timestamp
                })
                
                # Update metadata
                self.worker_metadata[worker_id]['last_update'] = timestamp
                return True
                
        return False
    
    def get_sequence(self, worker_id: int) -> Optional[np.ndarray]:
        """
        Get the current pose sequence for a worker.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            np.ndarray: Pose sequence of shape (sequence_length, 75) or None
        """
        with self.lock:
            if worker_id not in self.worker_buffers:
                return None
            
            buffer = self.worker_buffers[worker_id]
            if len(buffer) < self.sequence_length:
                return None
            
            # Extract pose vectors
            poses = [frame['pose'] for frame in buffer]
            return np.array(poses)
    
    def is_ready_for_prediction(self, worker_id: int) -> bool:
        """Check if worker has enough frames for prediction."""
        with self.lock:
            if worker_id not in self.worker_buffers:
                return False
            return len(self.worker_buffers[worker_id]) >= self.sequence_length
    
    def add_activity_prediction(self, worker_id: int, activity: str, confidence: float):
        """Store activity prediction for a worker."""
        with self.lock:
            if worker_id in self.worker_metadata:
                self.worker_metadata[worker_id]['activity_history'].append({
                    'activity': activity,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
    
    def get_recent_activities(self, worker_id: int, num_activities: int = 10) -> List[Dict]:
        """Get recent activity predictions for a worker."""
        with self.lock:
            if worker_id not in self.worker_metadata:
                return []
            
            history = self.worker_metadata[worker_id]['activity_history']
            return list(history)[-num_activities:]
    
    def cleanup_inactive_workers(self, timeout_seconds: float = 30.0):
        """Remove workers that haven't been updated recently."""
        current_time = time.time()
        inactive_workers = []
        
        with self.lock:
            for worker_id, metadata in self.worker_metadata.items():
                if current_time - metadata['last_update'] > timeout_seconds:
                    inactive_workers.append(worker_id)
            
            for worker_id in inactive_workers:
                del self.worker_buffers[worker_id]
                del self.worker_metadata[worker_id]
        
        return len(inactive_workers)
    
    def get_worker_count(self) -> int:
        """Get number of active workers being tracked."""
        return len(self.worker_buffers)
    
    def get_buffer_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status information for all workers."""
        with self.lock:
            status = {}
            for worker_id in self.worker_buffers:
                buffer = self.worker_buffers[worker_id]
                metadata = self.worker_metadata[worker_id]
                
                status[worker_id] = {
                    'buffer_length': len(buffer),
                    'is_ready': len(buffer) >= self.sequence_length,
                    'last_update': metadata['last_update'],
                    'recent_activities': len(metadata['activity_history'])
                }
            
            return status