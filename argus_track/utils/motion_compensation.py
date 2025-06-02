# argus_track/utils/motion_compensation.py

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from ..core import Detection, Track, GPSData

class MotionCompensationTracker:
    """
    Handles camera motion compensation to improve track continuity
    """
    
    def __init__(self, 
                 feature_detector: str = 'ORB',
                 max_features: int = 500,
                 match_threshold: float = 0.7):
        """
        Initialize motion compensation tracker
        
        Args:
            feature_detector: Type of feature detector ('ORB', 'SIFT', 'AKAZE')
            max_features: Maximum number of features to track
            match_threshold: Threshold for feature matching
        """
        self.logger = logging.getLogger(f"{__name__}.MotionCompensationTracker")
        
        # Initialize feature detector
        if feature_detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif feature_detector == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif feature_detector == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector: {feature_detector}")
        
        self.matcher = cv2.BFMatcher()
        self.match_threshold = match_threshold
        
        # Store previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.camera_motion_history = []
        
        self.logger.info(f"Initialized motion compensation with {feature_detector}")
    
    def estimate_camera_motion(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Estimate camera motion between frames using feature matching
        
        Args:
            current_frame: Current frame
            
        Returns:
            Homography matrix representing camera motion
        """
        if self.prev_frame is None:
            self._update_reference_frame(current_frame)
            return np.eye(3)  # Identity matrix for first frame
        
        # Detect features in current frame
        keypoints, descriptors = self.detector.detectAndCompute(current_frame, None)
        
        if descriptors is None or self.prev_descriptors is None:
            self.logger.warning("No features detected for motion estimation")
            return np.eye(3)
        
        # Match features between frames
        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            self.logger.warning(f"Insufficient matches for motion estimation: {len(good_matches)}")
            return np.eye(3)
        
        # Extract matched points
        prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        
        # Estimate homography
        try:
            homography, mask = cv2.findHomography(
                prev_pts, curr_pts,
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.99
            )
            
            if homography is not None:
                # Store motion for smoothing
                self.camera_motion_history.append(homography)
                if len(self.camera_motion_history) > 5:
                    self.camera_motion_history = self.camera_motion_history[-5:]
                
                # Update reference frame
                self._update_reference_frame(current_frame, keypoints, descriptors)
                
                return homography
            else:
                self.logger.warning("Failed to compute homography")
                return np.eye(3)
                
        except Exception as e:
            self.logger.error(f"Error in motion estimation: {e}")
            return np.eye(3)
    
    def compensate_track_positions(self, 
                                  tracks: List[Track], 
                                  homography: np.ndarray) -> List[Track]:
        """
        Compensate track positions for camera motion
        
        Args:
            tracks: List of tracks to compensate
            homography: Camera motion homography
            
        Returns:
            List of tracks with compensated positions
        """
        if np.allclose(homography, np.eye(3)):
            return tracks  # No motion compensation needed
        
        compensated_tracks = []
        
        for track in tracks:
            # Get current predicted position
            current_bbox = track.to_tlbr()
            
            # Convert bbox to center point
            center_x = (current_bbox[0] + current_bbox[2]) / 2
            center_y = (current_bbox[1] + current_bbox[3]) / 2
            
            # Apply inverse homography to compensate for camera motion
            try:
                homography_inv = np.linalg.inv(homography)
                
                # Transform center point
                point = np.array([[center_x, center_y]], dtype=np.float32)
                point = point.reshape(-1, 1, 2)
                
                compensated_point = cv2.perspectiveTransform(point, homography_inv)
                compensated_center = compensated_point.reshape(-1, 2)[0]
                
                # Update track's Kalman filter with compensated position
                if track.kalman_filter:
                    # Calculate offset
                    offset_x = compensated_center[0] - center_x
                    offset_y = compensated_center[1] - center_y
                    
                    # Apply offset to Kalman state
                    track.kalman_filter.kf.x[0] += offset_x  # x position
                    track.kalman_filter.kf.x[1] += offset_y  # y position
                
                compensated_tracks.append(track)
                
            except Exception as e:
                self.logger.error(f"Error compensating track {track.track_id}: {e}")
                compensated_tracks.append(track)  # Use original track
        
        return compensated_tracks
    
    def predict_detection_positions(self, 
                                   detections: List[Detection],
                                   homography: np.ndarray) -> List[Detection]:
        """
        Predict where detections should be based on camera motion
        
        Args:
            detections: Current detections
            homography: Camera motion homography
            
        Returns:
            Detections with motion-compensated positions
        """
        if np.allclose(homography, np.eye(3)):
            return detections
        
        compensated_detections = []
        
        for detection in detections:
            try:
                # Get detection center
                center = detection.center
                
                # Transform center using homography
                point = np.array([[center[0], center[1]]], dtype=np.float32)
                point = point.reshape(-1, 1, 2)
                
                transformed_point = cv2.perspectiveTransform(point, homography)
                new_center = transformed_point.reshape(-1, 2)[0]
                
                # Calculate offset and apply to bbox
                offset_x = new_center[0] - center[0]
                offset_y = new_center[1] - center[1]
                
                new_bbox = detection.bbox.copy()
                new_bbox[0] += offset_x  # x1
                new_bbox[1] += offset_y  # y1
                new_bbox[2] += offset_x  # x2
                new_bbox[3] += offset_y  # y2
                
                # Create new detection with adjusted position
                compensated_detection = Detection(
                    bbox=new_bbox,
                    score=detection.score,
                    class_id=detection.class_id,
                    frame_id=detection.frame_id
                )
                
                compensated_detections.append(compensated_detection)
                
            except Exception as e:
                self.logger.error(f"Error compensating detection: {e}")
                compensated_detections.append(detection)  # Use original
        
        return compensated_detections
    
    def estimate_motion_from_gps(self, 
                                current_gps: GPSData,
                                previous_gps: GPSData,
                                camera_focal_length: float = 1400) -> np.ndarray:
        """
        Estimate camera motion from GPS data
        
        Args:
            current_gps: Current GPS position
            previous_gps: Previous GPS position
            camera_focal_length: Camera focal length in pixels
            
        Returns:
            Estimated motion transformation matrix
        """
        try:
            # Calculate GPS displacement
            lat_diff = current_gps.latitude - previous_gps.latitude
            lon_diff = current_gps.longitude - previous_gps.longitude
            heading_diff = current_gps.heading - previous_gps.heading
            
            # Convert to meters (approximate)
            R = 6378137.0  # Earth radius
            lat_offset = lat_diff * R * np.pi / 180
            lon_offset = lon_diff * R * np.pi / 180 * np.cos(np.radians(current_gps.latitude))
            
            # Convert to pixels (rough approximation)
            # This needs calibration for your specific camera setup
            pixels_per_meter = camera_focal_length / 10.0  # Adjust based on your setup
            
            pixel_offset_x = lon_offset * pixels_per_meter
            pixel_offset_y = -lat_offset * pixels_per_meter  # Y is inverted in image coordinates
            
            # Create translation matrix
            translation_matrix = np.array([
                [1, 0, pixel_offset_x],
                [0, 1, pixel_offset_y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Add rotation for heading change if significant
            if abs(heading_diff) > 1.0:  # degrees
                heading_rad = np.radians(heading_diff)
                cos_h = np.cos(heading_rad)
                sin_h = np.sin(heading_rad)
                
                rotation_matrix = np.array([
                    [cos_h, -sin_h, 0],
                    [sin_h, cos_h, 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Combine rotation and translation
                motion_matrix = rotation_matrix @ translation_matrix
            else:
                motion_matrix = translation_matrix
            
            return motion_matrix
            
        except Exception as e:
            self.logger.error(f"Error estimating motion from GPS: {e}")
            return np.eye(3)
    
    def _update_reference_frame(self, 
                               frame: np.ndarray,
                               keypoints: Optional[List] = None,
                               descriptors: Optional[np.ndarray] = None):
        """Update reference frame for motion tracking"""
        self.prev_frame = frame.copy()
        
        if keypoints is not None and descriptors is not None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
        else:
            # Detect features if not provided
            self.prev_keypoints, self.prev_descriptors = self.detector.detectAndCompute(frame, None)
    
    def get_motion_statistics(self) -> dict:
        """Get motion tracking statistics"""
        if not self.camera_motion_history:
            return {"motion_detected": False}
        
        # Analyze recent motion
        recent_motions = self.camera_motion_history[-3:]
        translations = []
        rotations = []
        
        for H in recent_motions:
            # Extract translation
            tx, ty = H[0, 2], H[1, 2]
            translation_magnitude = np.sqrt(tx**2 + ty**2)
            translations.append(translation_magnitude)
            
            # Estimate rotation (simplified)
            rotation_angle = np.arctan2(H[1, 0], H[0, 0])
            rotations.append(abs(rotation_angle))
        
        return {
            "motion_detected": True,
            "avg_translation": np.mean(translations),
            "max_translation": np.max(translations),
            "avg_rotation": np.mean(rotations),
            "motion_frames": len(self.camera_motion_history)
        }