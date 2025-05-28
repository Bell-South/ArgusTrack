"""Object detectors for ByteTrack system"""

from .base import ObjectDetector
from .yolo import YOLODetector
from .mock import MockDetector
from .yolov11 import YOLOv11Detector

__all__ = ["ObjectDetector", "YOLODetector", "MockDetector", "YOLOv11Detector"]