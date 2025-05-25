"""Mock detector for testing"""

import numpy as np
from typing import List, Dict, Any
import random

from .base import ObjectDetector


class MockDetector(ObjectDetector):
    """Mock detector for testing purposes"""
    
    def __init__(self, target_classes: List[str] = None):
        """
        Initialize mock detector
        
        Args:
            target_classes: List of class names to detect
        """
        self.class_names = [
            'light_post', 'street_light', 'pole', 
            'traffic_light', 'stop_sign', 'person'
        ]
        self.target_classes = target_classes or self.class_names
        self.frame_count = 0
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate mock detections for testing
        
        Args:
            frame: Input image
            
        Returns:
            List of mock detections
        """
        h, w = frame.shape[:2]
        detections = []
        
        # Generate stable mock detections with slight variations
        base_positions = [
            (100, 100, 150, 300),
            (400, 120, 450, 320),
            (700, 90, 750, 290)
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(base_positions):
            # Add some noise to make it more realistic
            noise = 5 * np.sin(self.frame_count * 0.1 + i)
            
            x1 += int(noise)
            y1 += int(noise * 0.5)
            x2 += int(noise)
            y2 += int(noise * 0.5)
            
            # Ensure bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            
            # Random class from target classes
            class_name = random.choice(self.target_classes)
            class_id = self.class_names.index(class_name) if class_name in self.class_names else 0
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': 0.85 + random.uniform(-0.1, 0.1),
                'class_name': class_name,
                'class_id': class_id
            })
        
        self.frame_count += 1
        return detections
    
    def get_class_names(self) -> List[str]:
        """Get list of detectable class names"""
        return self.class_names.copy()