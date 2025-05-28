# argus_track/filters/kalman.py

"""Kalman filter implementation for object tracking"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Optional

from ..core import Detection


class KalmanBoxTracker:
    """
    Kalman filter implementation optimized for static/slow-moving objects
    
    State vector: [x, y, w, h, vx, vy, vw, vh]
    where (x, y) is center position, (w, h) is width/height,
    and v* are the corresponding velocities
    """
    
    def __init__(self, initial_detection: Detection):
        """
        Initialize Kalman filter with detection
        
        Args:
            initial_detection: First detection to initialize the filter
        """
        # 8-dimensional state, 4-dimensional measurement
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ])
        
        # Measurement matrix (we only measure position and size)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # w
            [0, 0, 0, 1, 0, 0, 0, 0]   # h
        ])
        
        xywh = initial_detection.xywh
        self.kf.x[0] = xywh[0]  # center x
        self.kf.x[1] = xywh[1]  # center y
        self.kf.x[2] = xywh[2]  # width
        self.kf.x[3] = xywh[3]  # height
        self.kf.x[4:] = 0       # Zero initial velocity (static assumption)
        
        # Initial uncertainty (higher for velocities)
        self.kf.P[4:, 4:] *= 1000  # High uncertainty for velocities
        self.kf.P[:4, :4] *= 10    # Lower uncertainty for position
        
        # Process noise (very low for static objects)
        self.kf.Q[4:, 4:] *= 0.01  # Minimal velocity changes expected
        self.kf.Q[:4, :4] *= 0.1   # Small position changes expected
        
        # Measurement noise
        self.kf.R *= 10.0
        
        self.time_since_update = 0
        self.history: List[Detection] = []
        self.hits = 1
        self.age = 1
        
    def predict(self) -> np.ndarray:
        """
        Predict next state
        
        Returns:
            Predicted bounding box in tlbr format
        """
        # Handle numerical stability
        if np.trace(self.kf.P[4:, 4:]) > 1e4:
            self.kf.P[4:, 4:] *= 0.01
            
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        return self.get_state()
    
    def update(self, detection: Detection) -> None:
        """
        Update filter with new detection
        
        Args:
            detection: New detection measurement
        """
        self.time_since_update = 0
        self.history.append(detection)
        self.hits += 1
        
        # Perform Kalman update
        self.kf.update(detection.xywh)

    def get_state(self) -> np.ndarray:
        """
        Get current state estimate
        
        Returns:
            Bounding box in tlbr format
        """
        x, y, w, h = self.kf.x[:4].flatten()  # Add .flatten() to fix shape
        return np.array([
            x - w/2,  # x1
            y - h/2,  # y1
            x + w/2,  # x2
            y + h/2   # y2
        ])
    
    def get_velocity(self) -> np.ndarray:
        """
        Get current velocity estimate
        
        Returns:
            Velocity vector [vx, vy]
        """
        return self.kf.x[4:6]


def batch_predict_kalman(kalman_trackers: List[KalmanBoxTracker]) -> np.ndarray:
    """
    Batch prediction for multiple Kalman filters
    
    Args:
        kalman_trackers: List of KalmanBoxTracker instances
        
    Returns:
        Array of predicted bounding boxes in tlbr format
    """
    if not kalman_trackers:
        return np.array([])
    
    # Collect predicted states
    predictions = np.zeros((len(kalman_trackers), 4))
    
    for i, tracker in enumerate(kalman_trackers):
        # Predict and get state
        tracker.predict()
        predictions[i] = tracker.get_state()
    
    return predictions