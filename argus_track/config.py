# argus_track/config.py (COMPLETE FIX)

"""Configuration classes for ArgusTrack with StereoCalibrationConfig"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import json
import pickle
import numpy as np
from pathlib import Path

@dataclass
class TrackerConfig:
    """Configuration for Ultralytics-based light post tracker"""
    
    # === DETECTION PARAMETERS (OPTIMIZED FOR YOUR ISSUES) ===
    detection_conf: float = 0.15      # LOWERED: Better continuity, less track loss
    detection_iou: float = 0.8       
    tracker_type: str = "bytetrack.yaml"  # Ultralytics tracker config
    max_detections: int = 20         # ADDED: Limit max detections per frame
    
    # === GEOLOCATION PARAMETERS ===
    static_threshold: float = 25.0      # Pixel movement threshold for static objects
    min_static_frames: int = 5          # Frames needed to confirm static object
    min_gps_points: int = 2             # Minimum GPS detections for geolocation
    max_geolocation_std: float = 0.0005 # GPS coordinate standard deviation limit
    
    # === GPS SYNCHRONIZATION ===
    gps_frame_interval: int = 6         # Process every 6th frame for GPS sync
    enable_gps_extraction: bool = True  # Auto-extract GPS from video metadata
    
    # === DEPTH ESTIMATION ===
    lightpost_height_meters: float = 4.0      # Assumed LED light height
    camera_focal_length_px: float = 1400.0    # Camera focal length in pixels
    max_detection_distance_m: float = 50.0    # Maximum reasonable detection distance
    min_detection_distance_m: float = 1.0     # Minimum reasonable detection distance
    
    # === EXPORT SETTINGS ===
    min_detections_for_export: int = 3  # LOWERED: Capture more LEDs
    export_geojson: bool = True         # Export GeoJSON file
    export_json: bool = True            # Export JSON results
    export_csv: bool = True             # Export CSV GPS data
    
    # === STATIC CAR DETECTION ===
    enable_static_car_detection: bool = True    # Enable static car frame skipping
    static_movement_threshold_m: float = 0.9    # Minimum movement to consider moving
    static_time_threshold_s: float = 5.0        # LOWERED: Faster response to stops

    @classmethod
    def create_ultralytics_optimized(cls) -> 'TrackerConfig':
        """Create configuration optimized for LED tracking issues"""
        return cls(
            # OPTIMIZED FOR YOUR SPECIFIC ISSUES
            detection_conf=0.30,           # Lower = better track continuity
            detection_iou=0.5,            # Higher = less overlapping boxes
            tracker_type="bytetrack.yaml", 
            max_detections=50,             # Reasonable limit
            
            # Geolocation settings
            static_threshold=55.0,         # Tighter static requirement
            min_static_frames=5,           # Reasonable confirmation
            min_gps_points=2,              # Keep at 2 for more exports
            max_geolocation_std=0.0002,    # Stricter GPS consistency
            
            # Depth estimation - OPTIMIZED for closer LEDs
            lightpost_height_meters=4.0,   # Standard LED light height
            camera_focal_length_px=1400.0, # GoPro approximate
            max_detection_distance_m=100.0, # LOWERED: Focus on closer LEDs
            
            # Export settings - MORE PERMISSIVE
            min_detections_for_export=3,   # Capture more tracks
            export_geojson=True,
            export_json=True,
            export_csv=True,
            
            # Static car detection - FASTER RESPONSE
            enable_static_car_detection=True,
            static_movement_threshold_m=0.9,  # LOWERED: More sensitive
            static_time_threshold_s=5.0       # LOWERED: Faster skip response
        )
    
    def get_ultralytics_track_params(self) -> dict:
        """Get parameters for model.track() call - OPTIMIZED"""
        return {
            'persist': True,
            'tracker': self.tracker_type,
            'conf': self.detection_conf,      # 0.15 - Lower for continuity
            'iou': self.detection_iou,        # 0.25 - Higher for less overlaps
            'max_det': self.max_detections,   # 80 - Reasonable limit
            'verbose': True
        }

@dataclass
class DetectorConfig:
    """Configuration for object detectors"""
    model_path: str
    config_path: str
    target_classes: Optional[list] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    model_type: str = "yolov11"        # Support for YOLOv11


@dataclass
class StereoCalibrationConfig:
    """Stereo camera calibration parameters"""
    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray                      # Rotation matrix between cameras
    T: np.ndarray                      # Translation vector between cameras
    E: Optional[np.ndarray] = None     # Essential matrix
    F: Optional[np.ndarray] = None     # Fundamental matrix
    P1: Optional[np.ndarray] = None    # Left camera projection matrix
    P2: Optional[np.ndarray] = None    # Right camera projection matrix
    Q: Optional[np.ndarray] = None     # Disparity-to-depth mapping matrix
    baseline: float = 0.0              # Distance between cameras (meters)
    image_width: int = 1920
    image_height: int = 1080
    
    @classmethod
    def from_pickle(cls, calibration_path: str) -> 'StereoCalibrationConfig':
        """Load stereo calibration from pickle file"""
        try:
            with open(calibration_path, 'rb') as f:
                calib_data = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load calibration file {calibration_path}: {e}")
        
        # Calculate baseline if not provided
        baseline = calib_data.get('baseline', 0.0)
        if baseline == 0.0 and 'T' in calib_data:
            baseline = float(np.linalg.norm(calib_data['T']))
        
        return cls(
            camera_matrix_left=calib_data['camera_matrix_left'],
            camera_matrix_right=calib_data['camera_matrix_right'],
            dist_coeffs_left=calib_data['dist_coeffs_left'],
            dist_coeffs_right=calib_data['dist_coeffs_right'],
            R=calib_data['R'],
            T=calib_data['T'],
            E=calib_data.get('E'),
            F=calib_data.get('F'),
            P1=calib_data.get('P1'),
            P2=calib_data.get('P2'),
            Q=calib_data.get('Q'),
            baseline=baseline,
            image_width=calib_data.get('image_width', 1920),
            image_height=calib_data.get('image_height', 1080)
        )
    
    @classmethod
    def create_sample_calibration(cls, 
                                 image_width: int = 1920,
                                 image_height: int = 1080,
                                 baseline: float = 0.12) -> 'StereoCalibrationConfig':
        """
        Create sample calibration for testing (GoPro Hero 11 approximate values)
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            baseline: Baseline distance in meters
            
        Returns:
            Sample calibration configuration
        """
        # Approximate GoPro Hero 11 parameters
        focal_length = 1400  # pixels
        cx = image_width / 2
        cy = image_height / 2
        
        # Camera matrices
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (approximate for GoPro)
        dist_coeffs = np.array([-0.3, 0.1, 0, 0, 0], dtype=np.float64)
        
        # Stereo parameters (assuming cameras are aligned horizontally)
        R = np.eye(3, dtype=np.float64)  # No rotation between cameras
        T = np.array([[baseline], [0], [0]], dtype=np.float64)  # Horizontal translation
        
        return cls(
            camera_matrix_left=camera_matrix,
            camera_matrix_right=camera_matrix,
            dist_coeffs_left=dist_coeffs,
            dist_coeffs_right=dist_coeffs,
            R=R,
            T=T,
            baseline=baseline,
            image_width=image_width,
            image_height=image_height
        )
    
    def save_pickle(self, output_path: str) -> None:
        """Save calibration to pickle file"""
        calib_data = {
            'camera_matrix_left': self.camera_matrix_left,
            'camera_matrix_right': self.camera_matrix_right,
            'dist_coeffs_left': self.dist_coeffs_left,
            'dist_coeffs_right': self.dist_coeffs_right,
            'R': self.R,
            'T': self.T,
            'E': self.E,
            'F': self.F,
            'P1': self.P1,
            'P2': self.P2,
            'Q': self.Q,
            'baseline': self.baseline,
            'image_width': self.image_width,
            'image_height': self.image_height
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(calib_data, f)


@dataclass
class CameraConfig:
    """Camera calibration parameters (backward compatibility)"""
    camera_matrix: list
    distortion_coeffs: list
    image_width: int
    image_height: int
    
    @classmethod
    def from_file(cls, calibration_path: str) -> 'CameraConfig':
        """Load camera configuration from file"""
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
