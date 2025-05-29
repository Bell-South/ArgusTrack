# argus_track/config.py (UPDATED TRACKING PARAMETERS)
"""Configuration with improved tracking parameters for static objects"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import json
import pickle
import numpy as np
from pathlib import Path


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack light post tracker - OPTIMIZED FOR STATIC OBJECTS"""
    
    # DETECTION THRESHOLDS - More restrictive to reduce noise
    track_thresh: float = 0.3          # Higher threshold to avoid weak detections
    match_thresh: float = 0.3          # Lower IoU threshold for better matching of static objects
    
    # TRACK MANAGEMENT - Extended for static objects
    track_buffer: int = 100            # Keep lost tracks longer (static objects may be occluded)
    min_box_area: float = 200.0        # Larger minimum area to filter small noise
    
    # STATIC OBJECT DETECTION - More conservative
    static_threshold: float = 5.0      # Pixels - allow slightly more movement for static detection
    min_static_frames: int = 15        # Require more frames to confirm static object
    max_track_age: int = 1000          # Maximum age before removing track
    
    # TRACK CONFIRMATION - More strict requirements
    min_hits: int = 5                  # Require more hits before confirming track
    max_time_lost: int = 50            # Maximum frames without detection before marking as lost
    
    # DUPLICATE TRACK MERGING - New parameters
    merge_distance_threshold: float = 30.0    # Pixel distance to consider tracks duplicates
    merge_iou_threshold: float = 0.5          # IoU threshold for merging tracks
    enable_track_merging: bool = True         # Enable automatic duplicate track merging
    
    # GPS AND GEOLOCATION
    gps_frame_interval: int = 6        # Process every 6th frame for GPS (10fps from 60fps)
    min_gps_points: int = 5            # Minimum GPS points needed for geolocation
    max_geolocation_std: float = 0.0001  # Maximum standard deviation for reliable geolocation
    
    # STEREO PARAMETERS
    stereo_mode: bool = False          # Default to monocular
    stereo_match_threshold: float = 0.7
    max_stereo_distance: float = 100.0
    
    @classmethod
    def create_optimized_config(cls) -> 'TrackerConfig':
        """Create optimized configuration for static LED detection"""
        return cls(
            track_thresh=0.1,              # Higher confidence threshold
            match_thresh=0.6,              # More lenient matching for static objects
            track_buffer=150,              # Longer buffer for static objects
            min_box_area=300.0,            # Filter out small detections
            static_threshold=8.0,          # Allow some camera shake
            min_static_frames=20,          # More frames needed for static confirmation
            max_track_age=2000,            # Very long track lifetime
            min_hits=8,                    # More hits required for confirmation
            max_time_lost=75,              # Longer time before considering lost
            merge_distance_threshold=50.0, # Merge nearby duplicate tracks
            merge_iou_threshold=0.4,       # IoU threshold for merging
            enable_track_merging=True,     # Enable merging
            gps_frame_interval=6,          # GPS sync
            min_gps_points=8,              # More GPS points for better accuracy
            max_geolocation_std=0.00005    # Stricter geolocation reliability
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrackerConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrackerConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def save_json(self, output_path: str) -> None:
        """Save configuration to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

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
        with open(calibration_path, 'rb') as f:
            calib_data = pickle.load(f)
        
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