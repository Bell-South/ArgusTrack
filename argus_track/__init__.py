# argus_track/__init__.py (UPDATED)

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
from argus_track.config import TrackerConfig, StereoCalibrationConfig, DetectorConfig
from argus_track.core import Detection, Track, GPSData
from argus_track.core.stereo import StereoDetection, StereoFrame, StereoTrack
from argus_track.trackers import ByteTrack, LightPostTracker
from argus_track.trackers.stereo_lightpost_tracker import StereoLightPostTracker
from argus_track.detectors import YOLODetector, ObjectDetector
from argus_track.detectors.yolov11 import YOLOv11Detector
from argus_track.stereo import StereoMatcher, StereoTriangulator, StereoCalibrationManager
from argus_track.exceptions import (
    ArgusTrackError, 
    DetectorError, 
    TrackerError,
    ConfigurationError,
    GPSError,
    VideoError
)
from argus_track.analysis import StaticObjectAnalyzer

__all__ = [
    "__version__",
    "TrackerConfig",
    "StereoCalibrationConfig", 
    "DetectorConfig",
    "Detection",
    "Track",
    "GPSData",
    "StereoDetection",
    "StereoFrame", 
    "StereoTrack",
    "ByteTrack",
    "LightPostTracker",
    "StereoLightPostTracker",
    "YOLODetector",
    "YOLOv11Detector",
    "ObjectDetector",
    "StereoMatcher",
    "StereoTriangulator", 
    "StereoCalibrationManager",
    "StaticObjectAnalyzer",
    "ArgusTrackError",
    "DetectorError",
    "TrackerError",
    "ConfigurationError", 
    "GPSError",
    "VideoError"
]