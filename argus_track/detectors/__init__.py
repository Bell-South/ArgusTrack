"""Object detectors for ByteTrack system"""

from .base import ObjectDetector
from .yolov11 import YOLOv11Detector

__all__ = ["ObjectDetector", "YOLOv11Detector"]