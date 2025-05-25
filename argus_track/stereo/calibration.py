# argus_track/stereo/calibration.py (NEW FILE)

"""Stereo camera calibration management"""

import cv2
import numpy as np
import pickle
from typing import Optional, Tuple, List
import logging
from pathlib import Path

from ..config import StereoCalibrationConfig


class StereoCalibrationManager:
    """
    Manages stereo camera calibration data and provides rectification utilities
    """
    
    def __init__(self, calibration: Optional[StereoCalibrationConfig] = None):
        """
        Initialize calibration manager
        
        Args:
            calibration: Pre-loaded calibration data
        """
        self.calibration = calibration
        self.logger = logging.getLogger(f"{__name__}.StereoCalibrationManager")
        
        # Rectification maps (computed when needed)
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        
    @classmethod
    def from_pickle_file(cls, calibration_path: str) -> 'StereoCalibrationManager':
        """
        Load calibration from pickle file
        
        Args:
            calibration_path: Path to calibration pickle file
            
        Returns:
            StereoCalibrationManager instance
        """
        calibration = StereoCalibrationConfig.from_pickle(calibration_path)
        return cls(calibration)
    
    def compute_rectification_maps(self, 
                                  image_size: Optional[Tuple[int, int]] = None,
                                  alpha: float = 0.0) -> bool:
        """
        Compute rectification maps for stereo pair
        
        Args:
            image_size: (width, height) of images, uses calibration size if None
            alpha: Free scaling parameter (0=crop, 1=no crop)
            
        Returns:
            True if successful
        """
        if self.calibration is None:
            self.logger.error("No calibration data available")
            return False
        
        if image_size is None:
            image_size = (self.calibration.image_width, self.calibration.image_height)
        
        try:
            # Compute rectification transforms
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                self.calibration.camera_matrix_left,
                self.calibration.dist_coeffs_left,
                self.calibration.camera_matrix_right,
                self.calibration.dist_coeffs_right,
                image_size,
                self.calibration.R,
                self.calibration.T,
                alpha=alpha
            )
            
            # Update calibration with computed matrices
            self.calibration.P1 = P1
            self.calibration.P2 = P2
            self.calibration.Q = Q
            
            # Compute rectification maps
            self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
                self.calibration.camera_matrix_left,
                self.calibration.dist_coeffs_left,
                R1, P1, image_size,
                cv2.CV_32FC1
            )
            
            self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
                self.calibration.camera_matrix_right,
                self.calibration.dist_coeffs_right,
                R2, P2, image_size,
                cv2.CV_32FC1
            )
            
            self.logger.info("Successfully computed rectification maps")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compute rectification maps: {e}")
            return False
    
    def rectify_image_pair(self, 
                          left_image: np.ndarray, 
                          right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            (rectified_left, rectified_right) images
        """
        if self.left_map1 is None or self.left_map2 is None:
            # Compute maps if not available
            image_size = (left_image.shape[1], left_image.shape[0])
            if not self.compute_rectification_maps(image_size):
                self.logger.warning("Using non-rectified images")
                return left_image, right_image
        
        # Apply rectification
        left_rectified = cv2.remap(
            left_image, self.left_map1, self.left_map2, cv2.INTER_LINEAR
        )
        right_rectified = cv2.remap(
            right_image, self.right_map1, self.right_map2, cv2.INTER_LINEAR
        )
        
        return left_rectified, right_rectified
    
    def validate_calibration(self) -> Tuple[bool, List[str]]:
        """
        Validate calibration data
        
        Returns:
            (is_valid, error_messages)
        """
        if self.calibration is None:
            return False, ["No calibration data loaded"]
        
        errors = []
        
        # Check camera matrices
        if self.calibration.camera_matrix_left.shape != (3, 3):
            errors.append("Invalid left camera matrix shape")
        
        if self.calibration.camera_matrix_right.shape != (3, 3):
            errors.append("Invalid right camera matrix shape")
        
        # Check distortion coefficients
        if len(self.calibration.dist_coeffs_left) < 4:
            errors.append("Invalid left distortion coefficients")
        
        if len(self.calibration.dist_coeffs_right) < 4:
            errors.append("Invalid right distortion coefficients")
        
        # Check rotation and translation
        if self.calibration.R.shape != (3, 3):
            errors.append("Invalid rotation matrix shape")
        
        if self.calibration.T.shape != (3, 1) and self.calibration.T.shape != (3,):
            errors.append("Invalid translation vector shape")
        
        # Check baseline
        if self.calibration.baseline <= 0:
            # Try to compute from translation vector
            if self.calibration.T.shape == (3, 1):
                baseline = float(np.linalg.norm(self.calibration.T))
            else:
                baseline = float(np.linalg.norm(self.calibration.T))
            
            if baseline <= 0:
                errors.append("Invalid baseline distance")
            else:
                self.calibration.baseline = baseline
                self.logger.info(f"Computed baseline: {baseline:.3f}m")
        
        # Check image dimensions
        if self.calibration.image_width <= 0 or self.calibration.image_height <= 0:
            errors.append("Invalid image dimensions")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info("Calibration validation passed")
        else:
            self.logger.error(f"Calibration validation failed: {errors}")
        
        return is_valid, errors
    
    def get_calibration_summary(self) -> dict:
        """Get summary of calibration parameters"""
        if self.calibration is None:
            return {"status": "No calibration loaded"}
        
        return {
            "baseline": f"{self.calibration.baseline:.3f}m",
            "image_size": f"{self.calibration.image_width}x{self.calibration.image_height}",
            "left_focal_length": f"{self.calibration.camera_matrix_left[0,0]:.1f}px",
            "right_focal_length": f"{self.calibration.camera_matrix_right[0,0]:.1f}px",
            "has_rectification": self.calibration.P1 is not None,
            "has_maps": self.left_map1 is not None
        }
    
    def create_sample_calibration(self, 
                                 image_width: int = 1920,
                                 image_height: int = 1080,
                                 baseline: float = 0.12) -> StereoCalibrationConfig:
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
        
        calibration = StereoCalibrationConfig(
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
        
        self.logger.info(f"Created sample calibration with {baseline}m baseline")
        return calibration
    
    def save_calibration(self, output_path: str) -> bool:
        """
        Save calibration to pickle file
        
        Args:
            output_path: Path for output file
            
        Returns:
            True if successful
        """
        if self.calibration is None:
            self.logger.error("No calibration to save")
            return False
        
        try:
            self.calibration.save_pickle(output_path)
            self.logger.info(f"Saved calibration to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
            return False