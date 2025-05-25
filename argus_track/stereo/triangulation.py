# argus_track/stereo/triangulation.py (NEW FILE)

"""3D triangulation and coordinate transformation"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from ..core.stereo import StereoDetection
from ..config import StereoCalibrationConfig
from ..core import GPSData
from ..utils.gps_utils import CoordinateTransformer, GeoLocation


class StereoTriangulator:
    """
    Handles 3D triangulation and coordinate system transformations
    from camera coordinates to world coordinates to GPS coordinates.
    """
    
    def __init__(self, 
                 calibration: StereoCalibrationConfig,
                 coordinate_transformer: Optional[CoordinateTransformer] = None):
        """
        Initialize triangulator
        
        Args:
            calibration: Stereo camera calibration
            coordinate_transformer: GPS coordinate transformer
        """
        self.calibration = calibration
        self.coordinate_transformer = coordinate_transformer
        self.logger = logging.getLogger(f"{__name__}.StereoTriangulator")
        
        # Camera extrinsics (if available)
        self.camera_position = None  # GPS position of camera
        self.camera_orientation = None  # Camera orientation relative to world
        
    def set_camera_pose(self, 
                       gps_position: GPSData, 
                       orientation_angles: Optional[Tuple[float, float, float]] = None):
        """
        Set camera pose for world coordinate transformation
        
        Args:
            gps_position: GPS position of the camera
            orientation_angles: (roll, pitch, yaw) in degrees
        """
        self.camera_position = gps_position
        if orientation_angles:
            self.camera_orientation = np.array(orientation_angles) * np.pi / 180  # Convert to radians
        
        # Update coordinate transformer
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(
                reference_lat=gps_position.latitude,
                reference_lon=gps_position.longitude
            )
    
    def triangulate_points(self, stereo_detections: List[StereoDetection]) -> List[np.ndarray]:
        """
        Triangulate 3D points from stereo detections
        
        Args:
            stereo_detections: List of stereo detection pairs
            
        Returns:
            List of 3D points in camera coordinate system
        """
        points_3d = []
        
        for stereo_det in stereo_detections:
            # Get 2D points
            left_point = stereo_det.left_detection.center
            right_point = stereo_det.right_detection.center
            
            # Triangulate
            point_3d = self._triangulate_single_point(left_point, right_point)
            points_3d.append(point_3d)
        
        return points_3d
    
    def _triangulate_single_point(self, 
                                 left_point: np.ndarray, 
                                 right_point: np.ndarray) -> np.ndarray:
        """Triangulate single 3D point from stereo pair"""
        
        # Prepare points for OpenCV triangulation
        left_pt = left_point.reshape(2, 1).astype(np.float32)
        right_pt = right_point.reshape(2, 1).astype(np.float32)
        
        # Use projection matrices if available
        if self.calibration.P1 is not None and self.calibration.P2 is not None:
            points_4d = cv2.triangulatePoints(
                self.calibration.P1,
                self.calibration.P2,
                left_pt,
                right_pt
            )
            
            # Convert from homogeneous coordinates
            if points_4d[3, 0] != 0:
                point_3d = points_4d[:3, 0] / points_4d[3, 0]
            else:
                point_3d = points_4d[:3, 0]
                
            return point_3d
        else:
            # Fallback triangulation using basic stereo geometry
            return self._basic_triangulation(left_point, right_point)
    
    def _basic_triangulation(self, left_point: np.ndarray, right_point: np.ndarray) -> np.ndarray:
        """Basic triangulation without projection matrices"""
        # Calculate disparity
        disparity = left_point[0] - right_point[0]
        
        if disparity <= 0:
            return np.array([0, 0, float('inf')])
        
        # Camera parameters
        fx = self.calibration.camera_matrix_left[0, 0]
        fy = self.calibration.camera_matrix_left[1, 1]
        cx = self.calibration.camera_matrix_left[0, 2]
        cy = self.calibration.camera_matrix_left[1, 2]
        baseline = self.calibration.baseline
        
        # Calculate depth
        depth = (baseline * fx) / disparity
        
        # Calculate 3D coordinates
        x = (left_point[0] - cx) * depth / fx
        y = (left_point[1] - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def camera_to_world_coordinates(self, 
                                   camera_points: List[np.ndarray],
                                   gps_data: GPSData) -> List[np.ndarray]:
        """
        Transform camera coordinates to world coordinates
        
        Args:
            camera_points: 3D points in camera coordinate system
            gps_data: GPS data for camera pose
            
        Returns:
            3D points in world coordinate system
        """
        world_points = []
        
        for cam_point in camera_points:
            # Apply camera rotation and translation
            world_point = self._transform_camera_to_world(cam_point, gps_data)
            world_points.append(world_point)
        
        return world_points
    
    def _transform_camera_to_world(self, 
                                  camera_point: np.ndarray, 
                                  gps_data: GPSData) -> np.ndarray:
        """Transform single point from camera to world coordinates"""
        
        # If we have camera orientation, apply rotation
        if self.camera_orientation is not None:
            # Create rotation matrix from Euler angles
            roll, pitch, yaw = self.camera_orientation
            
            # Rotation matrices
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(roll), -np.sin(roll)],
                          [0, np.sin(roll), np.cos(roll)]])
            
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                          [0, 1, 0],
                          [-np.sin(pitch), 0, np.cos(pitch)]])
            
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
            
            # Combined rotation matrix
            R = Rz @ Ry @ Rx
            
            # Apply rotation
            world_point = R @ camera_point
        else:
            # Assume camera is level and facing forward
            # Simple transformation: camera Z -> world X, camera X -> world Y, camera Y -> world Z
            world_point = np.array([camera_point[2], camera_point[0], -camera_point[1]])
        
        return world_point
    
    def world_to_gps_coordinates(self, 
                                world_points: List[np.ndarray],
                                reference_gps: GPSData) -> List[GeoLocation]:
        """
        Convert world coordinates to GPS coordinates
        
        Args:
            world_points: 3D points in world coordinate system
            reference_gps: Reference GPS position
            
        Returns:
            List of GPS locations
        """
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(
                reference_lat=reference_gps.latitude,
                reference_lon=reference_gps.longitude
            )
        
        gps_locations = []
        
        for world_point in world_points:
            # Use X, Y coordinates for GPS conversion (ignore Z/altitude)
            local_x = world_point[0]
            local_y = world_point[1]
            
            # Convert to GPS
            lat, lon = self.coordinate_transformer.local_to_gps(local_x, local_y)
            
            # Create GeoLocation with estimated accuracy
            location = GeoLocation(
                latitude=lat,
                longitude=lon,
                accuracy=self._estimate_gps_accuracy(world_point),
                reliability=0.8,  # Base reliability for stereo triangulation
                timestamp=reference_gps.timestamp
            )
            
            gps_locations.append(location)
        
        return gps_locations
    
    def _estimate_gps_accuracy(self, world_point: np.ndarray) -> float:
        """Estimate GPS accuracy based on triangulation quality"""
        # Accuracy degrades with distance
        distance = np.linalg.norm(world_point)
        
        # Base accuracy (1m) + distance-dependent error
        base_accuracy = 1.0
        distance_error = distance * 0.01  # 1cm per meter of distance
        
        estimated_accuracy = base_accuracy + distance_error
        
        # Cap at reasonable maximum
        return min(estimated_accuracy, 10.0)
    
    def estimate_object_location(self, 
                                stereo_track: 'StereoTrack',
                                gps_history: List[GPSData]) -> Optional[GeoLocation]:
        """
        Estimate final GPS location for a static object track
        
        Args:
            stereo_track: Stereo track with 3D trajectory
            gps_history: GPS data history for the track
            
        Returns:
            Estimated GPS location or None
        """
        if not stereo_track.is_static_3d or len(gps_history) < 3:
            return None
        
        # Get all world coordinates for the track
        world_coords = stereo_track.world_trajectory
        
        if len(world_coords) < 3:
            return None
        
        # Calculate average world position
        avg_world_pos = np.mean(world_coords, axis=0)
        
        # Use middle GPS point as reference
        mid_gps = gps_history[len(gps_history) // 2]
        
        # Convert to GPS
        gps_locations = self.world_to_gps_coordinates([avg_world_pos], mid_gps)
        
        if gps_locations:
            location = gps_locations[0]
            
            # Calculate reliability based on trajectory consistency
            if len(world_coords) > 1:
                positions = np.array(world_coords)
                std_dev = np.std(positions, axis=0)
                max_std = np.max(std_dev)
                
                # High reliability if standard deviation is low
                reliability = 1.0 / (1.0 + max_std)
                location.reliability = min(1.0, max(0.1, reliability))
            
            return location
        
        return None
    
    def validate_triangulation(self, 
                              stereo_detection: StereoDetection,
                              max_depth: float = 100.0,
                              min_depth: float = 1.0) -> bool:
        """
        Validate triangulation result
        
        Args:
            stereo_detection: Stereo detection to validate
            max_depth: Maximum reasonable depth
            min_depth: Minimum reasonable depth
            
        Returns:
            True if triangulation is valid
        """
        depth = stereo_detection.depth
        
        # Check depth range
        if not (min_depth <= depth <= max_depth):
            return False
        
        # Check if 3D coordinates are reasonable
        world_coords = stereo_detection.world_coordinates
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(world_coords)):
            return False
        
        # Check if coordinates are within reasonable bounds
        max_coord = 1000.0  # 1km from camera
        if np.any(np.abs(world_coords) > max_coord):
            return False
        
        return True