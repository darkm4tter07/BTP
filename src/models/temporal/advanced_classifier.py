import torch
import torch.nn as nn
import numpy as np
from collections import deque
import joblib

class TemporalActivityClassifier:
    """Advanced temporal analysis using modern techniques"""
    
    def __init__(self, sequence_length=30, num_features=75):
        """
        sequence_length: Number of frames to consider
        num_features: Pose features (25 keypoints * 3 coordinates)
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.pose_buffer = deque(maxlen=sequence_length)
        
        # Activity classes
        self.activity_classes = [
            'standing', 'walking', 'bending', 'lifting', 'carrying',
            'hammering', 'drilling', 'welding', 'painting', 'measuring',
            'climbing', 'kneeling', 'sitting', 'reaching_up', 'reaching_down'
        ]
        
        # Initialize model
        self.model = self._build_modern_model()
        self.scaler = None
        
    def _build_modern_model(self):
        """Build modern temporal classifier"""
        model = nn.Sequential(
            # 1D Convolutional layers for feature extraction
            nn.Conv1d(self.num_features, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Attention mechanism
            AttentionLayer(256),
            
            # LSTM for temporal modeling
            nn.LSTM(256, 128, num_layers=2, batch_first=True, dropout=0.2),
            
            # Classification head
            nn.Flatten(),
            nn.Linear(128 * self.sequence_length, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.activity_classes))
        )
        return model
    
    def add_pose_frame(self, pose_landmarks):
        """Add new pose frame to temporal buffer"""
        if pose_landmarks is not None and len(pose_landmarks) == 33:
            # Flatten 3D landmarks
            pose_vector = pose_landmarks[:, :3].flatten()  # x, y, z coordinates
            self.pose_buffer.append(pose_vector)
            return True
        return False
    
    def predict_activity(self):
        """Predict current activity based on pose sequence"""
        if len(self.pose_buffer) < self.sequence_length:
            return None, 0.0
        
        # Prepare sequence
        sequence = np.array(list(self.pose_buffer))
        
        # Normalize if scaler is available
        if self.scaler:
            sequence = self.scaler.transform(sequence)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        sequence_tensor = sequence_tensor.transpose(1, 2)  # (batch, features, seq_len)
        
        # Predict
        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return self.activity_classes[predicted_class], confidence
    
    def train_model(self, training_data, labels, validation_data=None):
        """Train the temporal classifier"""
        # Prepare data
        X = torch.FloatTensor(training_data).transpose(1, 2)
        y = torch.LongTensor(labels)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            scheduler.step(loss)
        
        return self.model

class AttentionLayer(nn.Module):
    """Simple attention mechanism for temporal features"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, hidden_dim, seq_len)
        attention_weights = torch.softmax(self.attention(x.transpose(1, 2)), dim=1)
        attended = torch.sum(x * attention_weights.transpose(1, 2), dim=2, keepdim=True)
        return attended.expand_as(x)