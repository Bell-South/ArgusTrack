# argus_track/detectors/yolov11.py (NEW FILE)

"""YOLOv11 detector implementation with improved architecture support"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import torch
import torchvision.transforms as transforms

from .base import ObjectDetector


class YOLOv11Detector(ObjectDetector):
    """YOLOv11-based object detector implementation with PyTorch backend"""
    
    def __init__(self, 
                 model_path: str,
                 target_classes: Optional[List[str]] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 device: str = 'auto',
                 input_size: int = 640):
        """
        Initialize YOLOv11 detector
        
        Args:
            model_path: Path to YOLOv11 model file (.pt)
            target_classes: List of class names to detect (None for all)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to use ('cpu', 'cuda', or 'auto')
            input_size: Model input size (typically 640)
        """
        self.logger = logging.getLogger(f"{__name__}.YOLOv11Detector")
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        # Get class names directly from the loaded model
        self.class_names = list(self.model.names.values())

        # If no target_classes specified, use ALL classes from the model
        if target_classes is None:
            self.target_classes = self.class_names.copy()
            self.logger.info(f"Using all model classes: {self.target_classes}")
        else:
            # Filter to only valid classes that exist in the model
            valid_classes = [cls for cls in target_classes if cls in self.class_names]
            if not valid_classes:
                self.logger.warning(f"None of the target classes {target_classes} found in model. Using all model classes.")
                self.target_classes = self.class_names.copy()
            else:
                self.target_classes = valid_classes
                self.logger.info(f"Using filtered target classes: {self.target_classes}")
    
        
        # Target class indices
        self.target_class_indices = [
            i for i, name in enumerate(self.class_names) 
            if name in self.target_classes
        ]
        
        self.logger.info(f"Initialized YOLOv11 detector with {len(self.target_classes)} target classes")
    
    def _load_model(self):
        """Load YOLOv11 model"""
        try:
            # Try to load with ultralytics (if available)
            try:
                from ultralytics import YOLO
                model = YOLO(self.model_path)
                model.to(self.device)
                self.logger.info("Loaded YOLOv11 model using ultralytics")
                return model
            except ImportError:
                self.logger.warning("ultralytics not available, falling back to torch.hub")
                
            # Fallback to torch.hub or direct torch loading
            if self.model_path.endswith('.pt'):
                model = torch.jit.load(self.model_path, map_location=self.device)
                model.eval()
                self.logger.info("Loaded YOLOv11 model using torch.jit")
                return model
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv11 model: {e}")
            raise
    
    def _get_coco_classes(self) -> List[str]:
        """Get COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame using YOLOv11
        
        Args:
            frame: Input image
            
        Returns:
            List of detections
        """
        try:
            # Check if using ultralytics YOLO
            if hasattr(self.model, 'predict'):
                return self._detect_ultralytics(frame)
            else:
                return self._detect_torch(frame)
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
        
    # Edit argus_track/detectors/yolov11.py
    def _detect_ultralytics(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detection using ultralytics YOLO"""
        
        # print(f"🔍 DETECTION CALLED: Frame shape {frame.shape}")
        # self.logger.info(f"🔍 DETECTION CALLED: Frame shape {frame.shape}")
        
        # Run inference
        results = self.model.predict(
            frame, 
            conf=0.001,  # Force very low confidence
            iou=self.nms_threshold,
            verbose=False
        )
        
        detections = []
        
        # print(f"🔍 MODEL RESULTS: {len(results)} results")
        # self.logger.info(f"🔍 MODEL RESULTS: {len(results)} results")
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                # print(f"🔍 RAW DETECTIONS: {len(boxes)} boxes")
                # self.logger.info(f"🔍 RAW DETECTIONS: {len(boxes)} boxes")
                
                for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
                    if cls_id in [0, 1]:  # Led-150 or Led-240
                        class_name = f"Led-{150 if cls_id == 0 else 240}"
                        
                        if score >= self.confidence_threshold:
                            detections.append({
                                'bbox': box.tolist(),
                                'score': float(score),
                                'class_name': class_name,
                                'class_id': cls_id
                            })
                            
                            # print(f"✅ KEPT: {class_name}, Conf: {score:.4f}")
                            # self.logger.info(f"✅ KEPT: {class_name}, Conf: {score:.4f}")
        
        # print(f"🔍 FINAL: {len(detections)} detections returned")
        # self.logger.info(f"🔍 FINAL: {len(detections)} detections returned")
        return detections

    def _detect_torch(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detection using pure PyTorch model"""
        # Preprocess image
        input_tensor = self._preprocess_image(frame)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Post-process results
        detections = self._postprocess_predictions(predictions, frame.shape)
        
        return detections
    
    def _preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess image for YOLOv11"""
        # Resize to model input size
        height, width = frame.shape[:2]
        
        # Calculate scale factor
        scale = min(self.input_size / width, self.input_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Pad to square
        top = (self.input_size - new_height) // 2
        bottom = self.input_size - new_height - top
        left = (self.input_size - new_width) // 2
        right = self.input_size - new_width - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to tensor
        image_tensor = torch.from_numpy(padded).permute(2, 0, 1).float()
        image_tensor /= 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _postprocess_predictions(self, 
                                predictions: torch.Tensor, 
                                original_shape: tuple) -> List[Dict[str, Any]]:
        """Post-process YOLOv11 predictions"""
        detections = []
        
        # Assuming predictions shape: [batch, num_boxes, 85] (x, y, w, h, conf, classes...)
        pred = predictions[0]  # Remove batch dimension
        
        # Filter by confidence
        conf_mask = pred[:, 4] >= self.confidence_threshold
        pred = pred[conf_mask]
        
        if len(pred) == 0:
            return detections
        
        # Convert boxes from center format to corner format
        boxes = pred[:, :4].clone()
        boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2  # x2 = cx + w/2
        boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2  # y2 = cy + h/2
        
        # Scale boxes back to original image size
        scale_x = original_shape[1] / self.input_size
        scale_y = original_shape[0] / self.input_size
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Get class predictions
        class_probs = pred[:, 5:]
        class_ids = torch.argmax(class_probs, dim=1)
        max_class_probs = torch.max(class_probs, dim=1)[0]
        
        # Apply NMS
        keep_indices = torchvision.ops.nms(
            boxes, 
            pred[:, 4] * max_class_probs,  # Combined confidence
            self.nms_threshold
        )
        
        # Filter results
        final_boxes = boxes[keep_indices]
        final_scores = pred[keep_indices, 4]
        final_classes = class_ids[keep_indices]
        
        # Convert to detection format
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            cls_id = int(cls_id.item())
            
            # Filter by target classes
            if cls_id in self.target_class_indices:
                class_name = self.class_names[cls_id]
                
                detections.append({
                    'bbox': box.cpu().numpy().tolist(),
                    'score': float(score.item()),
                    'class_name': class_name,
                    'class_id': cls_id
                })
        
        return detections
    
    def get_class_names(self) -> List[str]:
        """Get list of detectable class names"""
        return self.class_names.copy()
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set detection confidence threshold"""
        self.confidence_threshold = threshold
        self.logger.info(f"Updated confidence threshold to {threshold}")
    
    def set_nms_threshold(self, threshold: float) -> None:
        """Set NMS threshold"""
        self.nms_threshold = threshold
        self.logger.info(f"Updated NMS threshold to {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'input_size': self.input_size,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'target_classes': self.target_classes,
            'num_classes': len(self.class_names)
        }