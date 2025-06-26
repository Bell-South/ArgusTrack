# argus_track/__init__.py

"""
Argus Track: Enhanced GPS ByteTrack Light Post Tracking System
============================================================

A specialized implementation of ByteTrack for tracking light posts in stereo video sequences
with enhanced GPS integration and state-of-the-art heading calculation for precise 3D geolocation.

Key Features:
- Enhanced GPS extraction with fallback heading calculation
- Stereo vision processing with 3D triangulation
- Optimized for static/slow-moving objects
- Advanced GPS data integration for geolocation
- YOLOv11 support for advanced object detection
- Proper frame naming convention (0, 6, 12, 18...)
- State-of-the-art vehicle heading estimation
- Modular architecture with clear separation of concerns
- Comprehensive logging and error handling
- Type hints and documentation throughout

Enhanced GPS Features:
- Automatic heading calculation when GPS metadata is missing
- Vehicle-specific motion models for improved accuracy
- Multi-point bearing calculation for stability
- Context-adaptive processing (highway, city, parking)
- Vincenty's formulae for high-precision calculations

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

# Enhanced GPS extraction (with fallback for compatibility)
try:
    from argus_track.utils.enhanced_gps_extraction import (
        EnhancedGoProGPSExtractor,
        VehicleHeadingCalculator,
        extract_gps_with_enhanced_heading,
    )
    GPS_ENHANCEMENT_AVAILABLE = True
except ImportError:
    GPS_ENHANCEMENT_AVAILABLE = False
    # Fallback imports for basic functionality
    EnhancedGoProGPSExtractor = None
    VehicleHeadingCalculator = None
    extract_gps_with_enhanced_heading = None

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
    "GPS_ENHANCEMENT_AVAILABLE",
]

# Add enhanced GPS exports if available
if GPS_ENHANCEMENT_AVAILABLE:
    __all__.extend([
        "EnhancedGoProGPSExtractor",
        "VehicleHeadingCalculator", 
        "extract_gps_with_enhanced_heading",
    ])

# Version and capability info
def get_capabilities():
    """Get available capabilities of this Argus Track installation"""
    capabilities = {
        "version": __version__,
        "enhanced_gps": GPS_ENHANCEMENT_AVAILABLE,
        "yolov11_support": True,
        "real_time_visualization": True,
        "stereo_processing": True,
        "frame_naming_fixed": True,
    }
    
    return capabilities

def get_gps_info():
    """Get information about GPS enhancement capabilities"""
    if GPS_ENHANCEMENT_AVAILABLE:
        return {
            "enhanced_gps_available": True,
            "fallback_heading_calculation": True,
            "vehicle_motion_models": True,
            "vincenty_calculations": True,
            "multi_point_bearing": True,
            "context_adaptive_processing": True,
        }
    else:
        return {
            "enhanced_gps_available": False,
            "basic_gps_only": True,
            "install_requirements": [
                "beautifulsoup4>=4.9.0",
                "lxml>=4.6.0", 
                "gopro-overlay>=0.10.0",
            ]
        }