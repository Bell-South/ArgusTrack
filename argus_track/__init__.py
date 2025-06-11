# argus_track/__init__.py - FIXED imports

"""
Argus Track: Stereo ByteTrack Light Post Tracking System
========================================================

A specialized implementation of ByteTrack for tracking light posts in stereo video sequences
with GPS integration for precise 3D geolocation estimation.

Key Features:
- Stereo vision processing with 3D triangulation
- Optimized for static/slow-moving objects
- GPS data integration for geolocation
- YOLOv11 support for advanced object detection
- Modular architecture with clear separation of concerns
- Comprehensive logging and error handling
- Type hints and documentation throughout

Author: Argus Track Team
Date: 2025
License: MIT
"""

from argus_track.__version__ import __version__
from argus_track.config import StaticCarConfig, TrackerConfig
from argus_track.core import Detection, GPSData, Track
from argus_track.detectors import ObjectDetector
from argus_track.detectors.yolov11 import YOLOv11Detector
from argus_track.trackers import UnifiedLightPostTracker

__all__ = [
    "__version__",
    "TrackerConfig",
    "StaticCarConfig",
    "Detection",
    "Track",
    "GPSData",
    "UnifiedLightPostTracker",
    "YOLOv11Detector",
    "ObjectDetector",
]
