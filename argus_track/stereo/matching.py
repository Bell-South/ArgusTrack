"""Stereo matching for object detections"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment
import logging

from ..core import Detection
from ..core.stereo import StereoDetection
from ..config import StereoCalibrationConfig
from ..utils.iou import calculate_iou


class StereoMatcher:
    """
    Matches detections between left and right camera views using epipolar constraints
    and appearance similarity for robust stereo correspondence.
    """
    
    def __init__(self, 
                 calibration: StereoCalibrationConfig,
                 max_disparity: float = 1000.0,
                 min_disparity: float = -50.0,
                 epipolar_threshold: float = 16.0,
                 iou_threshold: float = 0.0,
                 cost_threshold: float = 0.8):  # Added cost threshold parameter
        """
        Initialize stereo matcher
        
        Args:
            calibration: Stereo camera calibration
            max_disparity: Maximum disparity in pixels
            min_disparity: Minimum disparity in pixels
            epipolar_threshold: Maximum distance from epipolar line in pixels
            iou_threshold: Minimum IoU for detection matching
            cost_threshold: Maximum matching cost to accept (lower is better)
        """
        self.calibration = calibration
        self.max_disparity = max_disparity
        self.min_disparity = min_disparity
        self.epipolar_threshold = epipolar_threshold
        self.iou_threshold = iou_threshold
        self.cost_threshold = cost_threshold  # Added this line
        self.logger = logging.getLogger(f"{__name__}.StereoMatcher")
        
        # Precompute rectification maps if available
        self.has_rectification = (calibration.P1 is not None and 
                                calibration.P2 is not None and 
                                calibration.Q is not None)
        
        if self.has_rectification:
            self.logger.info("Using rectified stereo matching")
        else:
            self.logger.info("Using unrectified stereo matching with epipolar constraints")
    
    def match_detections(self, 
                        left_detections: List[Detection], 
                        right_detections: List[Detection]) -> List[StereoDetection]:
        """
        Match detections between left and right cameras
        
        Args:
            left_detections: Detections from left camera
            right_detections: Detections from right camera
            
        Returns:
            List of stereo detection pairs
        """
        if not left_detections or not right_detections:
            return []
        
        # Calculate matching costs
        cost_matrix = self._calculate_matching_costs(left_detections, right_detections)
        
        # Apply Hungarian algorithm for optimal matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        stereo_detections = []
        
        for left_idx, right_idx in zip(row_indices, col_indices):
            cost = cost_matrix[left_idx, right_idx]
            
            # Filter matches by cost threshold
            if cost < 1.0:  # Cost of 1.0 means no valid match
                left_det = left_detections[left_idx]
                right_det = right_detections[right_idx]
                
                # Calculate disparity and depth
                disparity = self._calculate_disparity(left_det, right_det)
                depth = self._estimate_depth(disparity)
                
                # Calculate 3D world coordinates
                world_coords = self._triangulate_point(left_det, right_det)
                
                # Calculate stereo confidence
                confidence = self._calculate_stereo_confidence(left_det, right_det, cost)
                
                stereo_detection = StereoDetection(
                    left_detection=left_det,
                    right_detection=right_det,
                    disparity=disparity,
                    depth=depth,
                    world_coordinates=world_coords,
                    stereo_confidence=confidence
                )
                
                stereo_detections.append(stereo_detection)
        
        self.logger.debug(f"Matched {len(stereo_detections)} stereo pairs from "
                         f"{len(left_detections)} left and {len(right_detections)} right detections")
        
        return stereo_detections
    
    def _calculate_matching_costs(self, 
                                 left_detections: List[Detection], 
                                 right_detections: List[Detection]) -> np.ndarray:
        """Calculate cost matrix for detection matching"""
        n_left = len(left_detections)
        n_right = len(right_detections)
        cost_matrix = np.ones((n_left, n_right))  # Initialize with high cost
        
        for i, left_det in enumerate(left_detections):
            for j, right_det in enumerate(right_detections):
                # Check epipolar constraint
                if not self._check_epipolar_constraint(left_det, right_det):
                    continue
                
                # Check disparity range
                disparity = self._calculate_disparity(left_det, right_det)
                if not (self.min_disparity <= disparity <= self.max_disparity):
                    continue
                
                # Calculate geometric cost
                geometric_cost = self._calculate_geometric_cost(left_det, right_det)
                
                # Calculate appearance cost (simplified - could use features)
                appearance_cost = self._calculate_appearance_cost(left_det, right_det)
                
                # Combine costs
                total_cost = 0.7 * geometric_cost + 0.3 * appearance_cost
                cost_matrix[i, j] = total_cost
        
        return cost_matrix
    
    def _check_epipolar_constraint(self, left_det: Detection, right_det: Detection) -> bool:
        """Check if detections satisfy epipolar constraint"""
        if self.has_rectification:
            # For rectified images, epipolar lines are horizontal
            left_center = left_det.center
            right_center = right_det.center
            
            y_diff = abs(left_center[1] - right_center[1])
            return y_diff <= self.epipolar_threshold
        else:
            # Use fundamental matrix for unrectified images
            if self.calibration.F is None:
                # Fallback: assume roughly horizontal epipolar lines
                left_center = left_det.center
                right_center = right_det.center
                y_diff = abs(left_center[1] - right_center[1])
                return y_diff <= self.epipolar_threshold * 2
            
            # Proper epipolar constraint using fundamental matrix
            left_point = np.array([left_det.center[0], left_det.center[1], 1])
            right_point = np.array([right_det.center[0], right_det.center[1], 1])
            
            # Calculate epipolar line
            epipolar_line = self.calibration.F @ left_point
            
            # Distance from point to line
            distance = abs(np.dot(epipolar_line, right_point)) / np.sqrt(epipolar_line[0]**2 + epipolar_line[1]**2)
            
            return distance <= self.epipolar_threshold
    
    def _calculate_disparity(self, left_det: Detection, right_det: Detection) -> float:
        """Calculate disparity between matched detections"""
        left_center = left_det.center
        right_center = right_det.center
        
        # Disparity is the horizontal difference
        disparity = left_center[0] - right_center[0]
        return max(0, disparity)  # Ensure positive disparity
    
    def _estimate_depth(self, disparity: float) -> float:
        """Estimate depth from disparity"""
        if disparity <= 0:
            return float('inf')
        
        # Use calibration baseline and focal length
        baseline = self.calibration.baseline
        focal_length = self.calibration.camera_matrix_left[0, 0]  # fx
        
        if baseline == 0 or focal_length == 0:
            self.logger.warning("Invalid calibration parameters for depth estimation")
            return float('inf')
        
        depth = (baseline * focal_length) / disparity
        return depth
    
    def _triangulate_point(self, left_det: Detection, right_det: Detection) -> np.ndarray:
        """Triangulate 3D point from stereo detections"""
        left_center = left_det.center
        right_center = right_det.center
        
        # Prepare points for triangulation
        left_point = np.array([left_center[0], left_center[1]], dtype=np.float32)
        right_point = np.array([right_center[0], right_center[1]], dtype=np.float32)
        
        if self.has_rectification and self.calibration.Q is not None:
            # Use Q matrix for rectified images
            disparity = self._calculate_disparity(left_det, right_det)
            
            # Create homogeneous point
            point_2d = np.array([left_center[0], left_center[1], disparity, 1])
            
            # Transform to 3D
            point_3d = self.calibration.Q @ point_2d
            
            if point_3d[3] != 0:
                point_3d = point_3d / point_3d[3]
            
            return point_3d[:3]
        else:
            # Use projection matrices if available
            if self.calibration.P1 is not None and self.calibration.P2 is not None:
                # Triangulate using OpenCV
                points_4d = cv2.triangulatePoints(
                    self.calibration.P1,
                    self.calibration.P2,
                    left_point.reshape(2, 1),
                    right_point.reshape(2, 1)
                )
                
                # Convert from homogeneous coordinates
                if points_4d[3, 0] != 0:
                    point_3d = points_4d[:3, 0] / points_4d[3, 0]
                else:
                    point_3d = points_4d[:3, 0]
                
                return point_3d
            else:
                # Fallback: simple depth estimation
                depth = self._estimate_depth(self._calculate_disparity(left_det, right_det))
                
                # Convert to 3D using camera intrinsics
                fx = self.calibration.camera_matrix_left[0, 0]
                fy = self.calibration.camera_matrix_left[1, 1]
                cx = self.calibration.camera_matrix_left[0, 2]
                cy = self.calibration.camera_matrix_left[1, 2]
                
                x = (left_center[0] - cx) * depth / fx
                y = (left_center[1] - cy) * depth / fy
                z = depth
                
                return np.array([x, y, z])
    
    def _calculate_geometric_cost(self, left_det: Detection, right_det: Detection) -> float:
        """Calculate geometric matching cost"""
        # Size similarity
        left_area = left_det.area
        right_area = right_det.area
        
        if left_area == 0 or right_area == 0:
            size_cost = 1.0
        else:
            size_ratio = min(left_area, right_area) / max(left_area, right_area)
            size_cost = 1.0 - size_ratio
        
        # Y-coordinate difference (should be small for good stereo)
        y_diff = abs(left_det.center[1] - right_det.center[1])
        y_cost = min(1.0, y_diff / 50.0)  # Normalize by expected max difference
        
        # Disparity reasonableness
        disparity = self._calculate_disparity(left_det, right_det)
        if disparity < self.min_disparity or disparity > self.max_disparity:
            disparity_cost = 1.0
        else:
            # Prefer moderate disparities
            normalized_disparity = disparity / self.max_disparity
            disparity_cost = abs(normalized_disparity - 0.3)  # Prefer ~30% of max disparity
        
        return (size_cost + y_cost + disparity_cost) / 3.0
    
    def _calculate_appearance_cost(self, left_det: Detection, right_det: Detection) -> float:
        """Calculate appearance-based matching cost (simplified)"""
        # For now, use detection confidence similarity
        conf_diff = abs(left_det.score - right_det.score)
        
        # Class consistency
        class_cost = 0.0 if left_det.class_id == right_det.class_id else 1.0
        
        return (conf_diff + class_cost) / 2.0
    
    def _calculate_stereo_confidence(self, 
                                   left_det: Detection, 
                                   right_det: Detection, 
                                   matching_cost: float) -> float:
        """Calculate confidence for stereo match"""
        # Base confidence from detection scores
        base_confidence = (left_det.score + right_det.score) / 2.0
        
        # Reduce confidence based on matching cost
        matching_confidence = 1.0 - matching_cost
        
        # Combine confidences
        stereo_confidence = base_confidence * matching_confidence
        
        return min(1.0, max(0.0, stereo_confidence))