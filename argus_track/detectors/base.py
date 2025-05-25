"""Base detector interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class ObjectDetector(ABC):
    """Abstract base class for object detection modules"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of detections with keys: bbox, score, class_name, class_id
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get list of detectable class names"""
        pass
    
    def set_target_classes(self, target_classes: List[str]) -> None:
        """Set specific classes to detect"""
        self.target_classes = target_classes