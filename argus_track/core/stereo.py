# argus_track/core/stereo.py (NEW FILE)

"""Stereo vision data structures and utilities"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from .detection import Detection


@dataclass
class StereoDetection:
    """Stereo detection pair from left and right cameras"""
    left_detection: Detection
    right_detection: Detection
    disparity: float                   # Pixel disparity between left/right
    depth: float                       # Estimated depth in meters
    world_coordinates: np.ndarray      # 3D coordinates in camera frame
    stereo_confidence: float           # Confidence of stereo match [0,1]
    
    @property
    def center_3d(self) -> np.ndarray:
        """Get 3D center point"""
        return self.world_coordinates
    
    @property
    def left_center(self) -> np.ndarray:
        """Get left camera center point"""
        return self.left_detection.center
    
    @property
    def right_center(self) -> np.ndarray:
        """Get right camera center point"""
        return self.right_detection.center
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'left_detection': self.left_detection.to_dict(),
            'right_detection': self.right_detection.to_dict(),
            'disparity': self.disparity,
            'depth': self.depth,
            'world_coordinates': self.world_coordinates.tolist(),
            'stereo_confidence': self.stereo_confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StereoDetection':
        """Create from dictionary representation"""
        return cls(
            left_detection=Detection.from_dict(data['left_detection']),
            right_detection=Detection.from_dict(data['right_detection']),
            disparity=data['disparity'],
            depth=data['depth'],
            world_coordinates=np.array(data['world_coordinates']),
            stereo_confidence=data['stereo_confidence']
        )


@dataclass
class StereoFrame:
    """Stereo frame pair with synchronized detections"""
    frame_id: int
    timestamp: float
    left_frame: np.ndarray
    right_frame: np.ndarray
    left_detections: List[Detection]
    right_detections: List[Detection]
    stereo_detections: List[StereoDetection]
    gps_data: Optional['GPSData'] = None
    
    @property
    def has_gps(self) -> bool:
        """Check if frame has GPS data"""
        return self.gps_data is not None
    
    def get_stereo_count(self) -> int:
        """Get number of successful stereo matches"""
        return len(self.stereo_detections)


@dataclass
class StereoTrack:
    """Extended track with stereo 3D information"""
    track_id: int
    stereo_detections: List[StereoDetection]
    world_trajectory: List[np.ndarray]  # 3D trajectory in world coordinates
    gps_trajectory: List[np.ndarray]    # GPS coordinate trajectory
    estimated_location: Optional['GeoLocation'] = None
    depth_consistency: float = 0.0      # Measure of depth consistency
    
    @property
    def is_static_3d(self) -> bool:
        """Check if object is static in 3D space"""
        if len(self.world_trajectory) < 3:
            return False
        
        positions = np.array(self.world_trajectory)
        std_dev = np.std(positions, axis=0)
        
        # Object is static if movement in any axis is < 1 meter
        return np.all(std_dev < 1.0)
    
    @property
    def average_depth(self) -> float:
        """Get average depth of all detections"""
        if not self.stereo_detections:
            return 0.0
        return np.mean([det.depth for det in self.stereo_detections])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'track_id': self.track_id,
            'stereo_detections': [det.to_dict() for det in self.stereo_detections[-10:]],  # Last 10
            'world_trajectory': [pos.tolist() for pos in self.world_trajectory[-20:]],     # Last 20
            'gps_trajectory': [pos.tolist() for pos in self.gps_trajectory[-20:]],        # Last 20
            'estimated_location': self.estimated_location.__dict__ if self.estimated_location else None,
            'depth_consistency': self.depth_consistency,
            'is_static_3d': self.is_static_3d,
            'average_depth': self.average_depth
        }