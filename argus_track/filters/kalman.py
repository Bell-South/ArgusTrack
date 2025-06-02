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
        
        # Gap handling
        self.consecutive_gaps = 0
        self.prediction_confidence = 1.0
        
    def predict(self) -> np.ndarray:
        """
        Predict next state with gap tolerance
        
        Returns:
            Predicted bounding box in tlbr format
        """
        # Adapt process noise based on gap duration
        if self.time_since_update > 0:
            self.consecutive_gaps += 1
            
            # Increase uncertainty during longer gaps
            gap_factor = min(3.0, 1.0 + self.consecutive_gaps * 0.1)
            
            # For static objects, increase only slightly
            self.kf.Q[:4, :4] *= gap_factor * 0.5
            self.kf.Q[4:, 4:] *= gap_factor * 0.2
        else:
            self.consecutive_gaps = 0
        
        # Handle numerical stability
        if np.trace(self.kf.P[4:, 4:]) > 1e4:
            self.kf.P[4:, 4:] *= 0.01
            
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        # Reduce prediction confidence during gaps
        if self.consecutive_gaps > 0:
            confidence_decay = 0.95 ** self.consecutive_gaps
            self.prediction_confidence = max(0.1, confidence_decay)
        else:
            self.prediction_confidence = 1.0
        
        return self.get_state()
    
    def update(self, detection: Detection) -> None:
        """
        Update filter with new detection
        
        Args:
            detection: New detection measurement
        """
        self.time_since_update = 0
        self.consecutive_gaps = 0
        self.history.append(detection)
        self.hits += 1
        self.prediction_confidence = 1.0
        
        # Perform Kalman update
        self.kf.update(detection.xywh)

    def get_state(self) -> np.ndarray:
        """
        Get current state estimate with confidence weighting
        
        Returns:
            Bounding box in tlbr format
        """
        x, y, w, h = self.kf.x[:4].flatten()
        
        # Apply confidence-based adjustment for predictions during gaps
        if self.prediction_confidence < 1.0 and len(self.history) > 0:
            # Blend prediction with last known good position
            last_detection = self.history[-1]
            last_center = last_detection.center
            last_size = np.array([last_detection.bbox[2] - last_detection.bbox[0],
                                 last_detection.bbox[3] - last_detection.bbox[1]])
            
            # Weighted blend based on confidence
            blend_factor = 1.0 - self.prediction_confidence
            x = x * self.prediction_confidence + last_center[0] * blend_factor
            y = y * self.prediction_confidence + last_center[1] * blend_factor
            w = w * self.prediction_confidence + last_size[0] * blend_factor
            h = h * self.prediction_confidence + last_size[1] * blend_factor
        
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
    
    def get_prediction_confidence(self) -> float:
        """Get current prediction confidence"""
        return self.prediction_confidence
    
    def is_stable(self, threshold: float = 5.0) -> bool:
        """Check if track represents a stable/static object"""
        if len(self.history) < 3:
            return False
        
        recent_centers = [det.center for det in self.history[-5:]]
        positions = np.array(recent_centers)
        position_std = np.std(positions, axis=0)
        max_std = np.max(position_std)
        
        return max_std < threshold


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


class StaticOptimizedKalmanBoxTracker(KalmanBoxTracker):
    """
    Enhanced Kalman filter specifically optimized for static objects
    """
    
    def __init__(self, initial_detection: Detection, static_mode: bool = True):
        """
        Initialize enhanced Kalman filter
        
        Args:
            initial_detection: First detection
            static_mode: Whether to optimize for static objects
        """
        super().__init__(initial_detection)
        
        self.static_mode = static_mode
        
        if static_mode:
            # Apply static object optimizations
            self.kf.P[4:, 4:] *= 0.1      # Much lower velocity uncertainty
            self.kf.P[:4, :4] *= 0.5      # Lower position uncertainty
            
            # Very low process noise for static objects
            self.kf.Q[4:, 4:] *= 0.001    # Almost no velocity changes
            self.kf.Q[:4, :4] *= 0.01     # Minimal position changes
            
            # Lower measurement noise for consistent detections
            self.kf.R *= 0.5
    
    def predict(self) -> np.ndarray:
        """Enhanced prediction for static objects"""
        # For static objects, be more conservative with uncertainty growth
        if self.static_mode and self.time_since_update > 0:
            # Slower uncertainty growth for static objects
            gap_factor = min(2.0, 1.0 + self.consecutive_gaps * 0.05)
            self.kf.Q[:4, :4] *= gap_factor * 0.3
            self.kf.Q[4:, 4:] *= gap_factor * 0.1
        
        return super().predict()


# Enhanced batch prediction for static objects
def batch_predict_kalman_enhanced(kalman_trackers: List[KalmanBoxTracker]) -> np.ndarray:
    """
    Enhanced batch prediction with confidence information
    
    Args:
        kalman_trackers: List of Kalman trackers
        
    Returns:
        Array of predicted bounding boxes with confidence
    """
    if not kalman_trackers:
        return np.array([])
    
    predictions = np.zeros((len(kalman_trackers), 5))  # Include confidence
    
    for i, tracker in enumerate(kalman_trackers):
        bbox = tracker.predict()
        confidence = tracker.get_prediction_confidence() if hasattr(tracker, 'get_prediction_confidence') else 1.0
        
        predictions[i, :4] = bbox
        predictions[i, 4] = confidence
    
    return predictions


# Utility functions for gap tolerance analysis
def analyze_tracking_gaps(kalman_trackers: List[KalmanBoxTracker]) -> dict:
    """
    Analyze tracking gaps across all trackers
    
    Args:
        kalman_trackers: List of Kalman trackers
        
    Returns:
        Dictionary with gap analysis
    """
    total_gaps = 0
    max_gap = 0
    trackers_with_gaps = 0
    total_predictions = 0
    
    for tracker in kalman_trackers:
        if hasattr(tracker, 'consecutive_gaps'):
            if tracker.consecutive_gaps > 0:
                total_gaps += tracker.consecutive_gaps
                max_gap = max(max_gap, tracker.consecutive_gaps)
                trackers_with_gaps += 1
        
        total_predictions += tracker.age
    
    return {
        'total_gaps': total_gaps,
        'max_gap': max_gap,
        'trackers_with_gaps': trackers_with_gaps,
        'total_trackers': len(kalman_trackers),
        'avg_gap_per_tracker': total_gaps / len(kalman_trackers) if kalman_trackers else 0,
        'gap_ratio': total_gaps / total_predictions if total_predictions > 0 else 0
    }