# argus_track/config.py (UPDATE - Add stereo configuration)

"""Configuration classes for ByteTrack Light Post Tracking System"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import json
import pickle
import numpy as np
from pathlib import Path


@dataclass
class TrackerConfig:
    """Configuration for ByteTrack light post tracker"""
    track_thresh: float = 0.5          # Minimum detection confidence
    match_thresh: float = 0.8          # Minimum IoU for matching
    track_buffer: int = 50             # Frames to keep lost tracks
    min_box_area: float = 100.0        # Minimum detection area
    static_threshold: float = 2.0      # Pixel movement threshold for static detection
    min_static_frames: int = 5         # Frames needed to confirm static object
    
    # Stereo-specific parameters
    stereo_mode: bool = True           # Enable stereo processing
    stereo_match_threshold: float = 0.7  # IoU threshold for stereo matching
    max_stereo_distance: float = 100.0   # Max pixel distance for stereo matching
    gps_frame_interval: int = 6          # Process every Nth frame (60fps -> 10fps GPS)
    
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