"""Object detectors for ByteTrack system"""

from .base import ObjectDetector
from .yolo import YOLODetector
from .mock import MockDetector

__all__ = ["ObjectDetector", "YOLODetector", "MockDetector"]