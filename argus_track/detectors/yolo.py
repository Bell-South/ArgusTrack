"""YOLO detector implementation"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .base import ObjectDetector


class YOLODetector(ObjectDetector):
    """YOLO-based object detector implementation"""
    
    def __init__(self, model_path: str, config_path: str, 
                 target_classes: Optional[List[str]] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO weights
            config_path: Path to YOLO config
            target_classes: List of class names to detect (None for all)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.logger = logging.getLogger(f"{__name__}.YOLODetector")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        try:
            # Initialize YOLO model (using OpenCV's DNN module)
            self.net = cv2.dnn.readNet(model_path, config_path)
            
            # Set backend preference (CUDA if available)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] 
                                for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            names_path = Path(config_path).with_suffix('.names')
            if names_path.exists():
                with open(names_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                self.logger.warning(f"Class names file not found: {names_path}")
                self.class_names = [f"class_{i}" for i in range(80)]  # Default COCO classes
            
            self.target_classes = target_classes or self.class_names
            self.logger.info(f"Initialized YOLO detector with {len(self.class_names)} classes")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO detector: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame using YOLO
        
        Args:
            frame: Input image
            
        Returns:
            List of detections
        """
        height, width = frame.shape[:2]
        
        # Prepare input
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), 
                                   True, crop=False)
        self.net.setInput(blob)
        
        # Run inference
        outputs = self.net.forward(self.output_layers)
        
        # Extract detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter by confidence and target classes
                class_name = self.class_names[class_id]
                if confidence > self.confidence_threshold and class_name in self.target_classes:
                    # Convert YOLO format to pixel coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 
                                  self.confidence_threshold, 
                                  self.nms_threshold)
        
        # Format results
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'bbox': [x, y, x + w, y + h],  # Convert to tlbr format
                    'score': confidences[i],
                    'class_name': self.class_names[class_ids[i]],
                    'class_id': class_ids[i]
                })
        
        return detections
    
    def get_class_names(self) -> List[str]:
        """Get list of detectable class names"""
        return self.class_names.copy()
    
    def set_backend(self, backend: str = 'cpu') -> None:
        """
        Set computation backend
        
        Args:
            backend: 'cpu', 'cuda', or 'opencl'
        """
        if backend.lower() == 'cuda':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif backend.lower() == 'opencl':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.logger.info(f"Set backend to: {backend}")