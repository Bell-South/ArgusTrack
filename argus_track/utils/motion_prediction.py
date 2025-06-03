"""
Motion Predictor for Enhanced Tracking
=====================================
Provides advanced motion prediction capabilities for object tracking,
including camera motion compensation and track position prediction.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import cv2
import logging

from ..core import Detection, Track, GPSData

@dataclass
class CameraMotion:
    """Represents camera motion between frames"""
    translation: np.ndarray  # [dx, dy]
    rotation: float         # rotation angle in radians
    scale: float           # scale factor
    confidence: float      # motion estimation confidence
    timestamp: float       # when motion was detected

@dataclass 
class PredictedPosition:
    """Predicted position for an object"""
    bbox: np.ndarray       # predicted bounding box
    confidence: float      # prediction confidence
    velocity: np.ndarray   # estimated velocity [vx, vy]
    timestamp: float       # prediction timestamp

class MotionPredictor:
    """
    Advanced motion predictor for object tracking with camera motion compensation
    """
    
    def __init__(self, 
                 history_length: int = 10,
                 motion_threshold: float = 2.0,
                 prediction_steps: int = 5):
        """
        Initialize motion predictor
        
        Args:
            history_length: Number of frames to maintain in history
            motion_threshold: Minimum motion to consider significant
            prediction_steps: Number of frames to predict ahead
        """
        self.history_length = history_length
        self.motion_threshold = motion_threshold
        self.prediction_steps = prediction_steps
        self.logger = logging.getLogger(f"{__name__}.MotionPredictor")
        
        # Motion history
        self.camera_motion_history: List[CameraMotion] = []
        self.track_histories: Dict[int, List[Detection]] = {}
        
        # Feature detector for motion estimation
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher()
        
        # Previous frame data for motion estimation
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        self.logger.info("Motion Predictor initialized")
    
    # Add these methods to your MotionPredictor class in motion_prediction.py

    def estimate_motion_from_gps_enhanced(self, 
                                        current_gps: GPSData,
                                        previous_gps: GPSData,
                                        camera_focal_length: float = 1400,
                                        vehicle_speed_threshold: float = 1.0) -> np.ndarray:
        """
        Enhanced GPS-based motion estimation with vehicle state awareness
        
        Args:
            current_gps: Current GPS position
            previous_gps: Previous GPS position  
            camera_focal_length: Camera focal length in pixels
            vehicle_speed_threshold: Speed threshold to determine if vehicle is moving (m/s)
            
        Returns:
            Enhanced motion transformation matrix
        """
        try:
            # Calculate time difference
            dt = current_gps.timestamp - previous_gps.timestamp
            if dt <= 0:
                return np.eye(3)
            
            # Calculate GPS displacement in meters
            R = 6378137.0  # Earth radius
            lat1, lon1 = np.radians(previous_gps.latitude), np.radians(previous_gps.longitude)
            lat2, lon2 = np.radians(current_gps.latitude), np.radians(current_gps.longitude)
            
            # Haversine distance
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c
            
            # Calculate speed
            speed = distance / dt  # m/s
            
            # Calculate bearing change
            bearing_change = current_gps.heading - previous_gps.heading
            
            # Normalize bearing change to [-180, 180]
            while bearing_change > 180:
                bearing_change -= 360
            while bearing_change < -180:
                bearing_change += 360
            
            # === VEHICLE STATE ANALYSIS ===
            if speed < vehicle_speed_threshold:
                # Vehicle is stationary or very slow
                # Use minimal motion compensation
                motion_matrix = np.eye(3)
                motion_matrix[0, 2] = 0  # No translation
                motion_matrix[1, 2] = 0
                
                self.logger.debug(f"Vehicle stationary (speed: {speed:.2f} m/s)")
                return motion_matrix
            
            # === MOVING VEHICLE COMPENSATION ===
            
            # Convert GPS motion to pixel motion
            # This needs calibration for your specific camera setup
            meters_per_pixel = 0.01  # Rough estimate - needs calibration
            
            # Calculate forward motion (in vehicle direction)
            forward_motion_m = distance  # Total distance moved
            
            # Project GPS motion onto image plane
            # Vehicle heading determines primary motion direction
            vehicle_heading_rad = np.radians(current_gps.heading)
            
            # Calculate pixel displacement based on GPS motion
            # This assumes camera is aligned with vehicle heading
            forward_pixels = forward_motion_m / meters_per_pixel
            
            # Project onto image coordinates
            pixel_displacement_x = forward_pixels * np.sin(vehicle_heading_rad)
            pixel_displacement_y = -forward_pixels * np.cos(vehicle_heading_rad)  # Y is inverted
            
            # Apply scaling based on speed (faster motion = more compensation needed)
            speed_factor = min(2.0, 1.0 + speed / 10.0)  # Cap at 2x compensation
            pixel_displacement_x *= speed_factor
            pixel_displacement_y *= speed_factor
            
            # Create translation matrix
            translation_matrix = np.array([
                [1, 0, pixel_displacement_x],
                [0, 1, pixel_displacement_y], 
                [0, 0, 1]
            ], dtype=np.float32)
            
            # === ROTATION COMPENSATION ===
            if abs(bearing_change) > 1.0:  # Significant rotation
                # Convert bearing change to rotation matrix
                rotation_rad = np.radians(bearing_change)
                cos_r = np.cos(rotation_rad)
                sin_r = np.sin(rotation_rad)
                
                rotation_matrix = np.array([
                    [cos_r, -sin_r, 0],
                    [sin_r, cos_r, 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Combine rotation and translation
                motion_matrix = rotation_matrix @ translation_matrix
                
                self.logger.debug(f"GPS motion: {distance:.1f}m, {bearing_change:.1f}°, speed: {speed:.1f}m/s")
            else:
                motion_matrix = translation_matrix
                self.logger.debug(f"GPS motion: {distance:.1f}m, speed: {speed:.1f}m/s (no rotation)")
            
            return motion_matrix
            
        except Exception as e:
            self.logger.error(f"Error in GPS-enhanced motion estimation: {e}")
            return np.eye(3)

    def compensate_tracks_for_gps_motion(self, 
                                        tracks: List[Track], 
                                        gps_motion_matrix: np.ndarray,
                                        vehicle_speed: float) -> List[Track]:
        """
        Compensate track positions for GPS-based vehicle motion
        
        Args:
            tracks: List of tracks to compensate
            gps_motion_matrix: GPS-derived motion transformation
            vehicle_speed: Current vehicle speed in m/s
            
        Returns:
            Motion-compensated tracks
        """
        if np.allclose(gps_motion_matrix, np.eye(3)) or vehicle_speed < 0.5:
            # No compensation needed for stationary vehicle
            return tracks
        
        compensated_tracks = []
        
        for track in tracks:
            try:
                # Get current track position
                current_bbox = track.to_tlbr()
                center_x = (current_bbox[0] + current_bbox[2]) / 2
                center_y = (current_bbox[1] + current_bbox[3]) / 2
                
                # Apply inverse GPS motion to compensate
                gps_motion_inv = np.linalg.inv(gps_motion_matrix)
                
                # Transform center point
                point = np.array([center_x, center_y, 1])
                compensated_point = gps_motion_inv @ point
                
                # Calculate offset
                offset_x = compensated_point[0] - center_x
                offset_y = compensated_point[1] - center_y
                
                # Apply compensation strength based on speed
                compensation_strength = min(1.0, vehicle_speed / 5.0)  # Full compensation at 5 m/s
                offset_x *= compensation_strength
                offset_y *= compensation_strength
                
                # Update track's Kalman filter if available
                if hasattr(track, 'kalman_filter') and track.kalman_filter:
                    track.kalman_filter.kf.x[0] += offset_x  # x position
                    track.kalman_filter.kf.x[1] += offset_y  # y position
                    
                    # Also update velocity prediction based on GPS
                    gps_velocity_x = gps_motion_matrix[0, 2]
                    gps_velocity_y = gps_motion_matrix[1, 2]
                    
                    # Blend GPS velocity with Kalman prediction
                    blend_factor = 0.3  # 30% GPS, 70% Kalman
                    track.kalman_filter.kf.x[4] = (1 - blend_factor) * track.kalman_filter.kf.x[4] + blend_factor * gps_velocity_x
                    track.kalman_filter.kf.x[5] = (1 - blend_factor) * track.kalman_filter.kf.x[5] + blend_factor * gps_velocity_y
                
                compensated_tracks.append(track)
                
                self.logger.debug(f"Track {track.track_id}: GPS compensation applied ({offset_x:.1f}, {offset_y:.1f})")
                
            except Exception as e:
                self.logger.error(f"Error compensating track {track.track_id}: {e}")
                compensated_tracks.append(track)  # Use original if compensation fails
        
        return compensated_tracks

    def predict_track_position_gps_enhanced(self, 
                                        track: Track, 
                                        steps_ahead: int = 1,
                                        gps_velocity: Optional[Tuple[float, float]] = None) -> PredictedPosition:
        """
        Enhanced track position prediction using GPS velocity information
        
        Args:
            track: Track to predict
            steps_ahead: Number of frames to predict ahead
            gps_velocity: GPS velocity (speed_m/s, heading_deg)
            
        Returns:
            Enhanced predicted position
        """
        # Get base prediction from motion history
        base_prediction = self.predict_track_position(track, steps_ahead)
        
        if gps_velocity is None:
            return base_prediction
        
        speed_ms, heading_deg = gps_velocity
        
        # If vehicle is moving significantly, adjust prediction
        if speed_ms > 1.0:  # Vehicle moving > 1 m/s
            try:
                # Convert GPS velocity to pixel velocity
                # This conversion needs calibration for your setup
                pixels_per_meter = 100  # Rough estimate - needs calibration
                heading_rad = np.radians(heading_deg)
                
                # Calculate GPS-based velocity in pixels per frame
                # Assuming 10fps processing rate
                frame_rate = 10.0
                distance_per_frame = speed_ms / frame_rate  # meters per frame
                pixels_per_frame = distance_per_frame * pixels_per_meter
                
                # Project GPS velocity onto image coordinates
                gps_velocity_x = pixels_per_frame * np.sin(heading_rad)
                gps_velocity_y = -pixels_per_frame * np.cos(heading_rad)  # Y inverted
                
                # Get current track position
                current_bbox = base_prediction.bbox
                current_center = np.array([
                    (current_bbox[0] + current_bbox[2]) / 2,
                    (current_bbox[1] + current_bbox[3]) / 2
                ])
                
                # Apply GPS-based motion compensation
                # Objects should appear to move opposite to vehicle motion
                compensated_center = current_center - np.array([gps_velocity_x, gps_velocity_y]) * steps_ahead
                
                # Maintain bounding box size
                bbox_width = current_bbox[2] - current_bbox[0]
                bbox_height = current_bbox[3] - current_bbox[1]
                
                enhanced_bbox = np.array([
                    compensated_center[0] - bbox_width / 2,
                    compensated_center[1] - bbox_height / 2,
                    compensated_center[0] + bbox_width / 2,
                    compensated_center[1] + bbox_height / 2
                ])
                
                # Blend base prediction with GPS-enhanced prediction
                blend_factor = min(0.4, speed_ms / 10.0)  # More GPS influence at higher speeds
                
                final_bbox = (1 - blend_factor) * base_prediction.bbox + blend_factor * enhanced_bbox
                
                # Enhanced confidence based on GPS reliability
                gps_confidence_bonus = min(0.2, speed_ms / 20.0)  # Up to 20% bonus
                enhanced_confidence = min(1.0, base_prediction.confidence + gps_confidence_bonus)
                
                return PredictedPosition(
                    bbox=final_bbox,
                    confidence=enhanced_confidence,
                    velocity=base_prediction.velocity,  # Keep original velocity estimate
                    timestamp=base_prediction.timestamp
                )
                
            except Exception as e:
                self.logger.error(f"Error in GPS-enhanced prediction: {e}")
                return base_prediction
        
        return base_prediction

    def get_gps_motion_statistics(self) -> Dict[str, Any]:
        """Get GPS-enhanced motion statistics"""
        base_stats = self.get_motion_statistics()
        
        # Add GPS-specific statistics
        gps_stats = {
            **base_stats,
            'gps_enhanced': True,
            'vehicle_motion_compensation': True,
            'speed_adaptive_compensation': True
        }
        
        return gps_stats

    def update_camera_motion(self, current_frame: np.ndarray, timestamp: float) -> CameraMotion:
        """
        Estimate camera motion from consecutive frames
        
        Args:
            current_frame: Current video frame
            timestamp: Frame timestamp
            
        Returns:
            Estimated camera motion
        """
        if self.prev_frame is None:
            self._update_reference_frame(current_frame)
            return CameraMotion(
                translation=np.array([0.0, 0.0]),
                rotation=0.0,
                scale=1.0,
                confidence=1.0,
                timestamp=timestamp
            )
        
        try:
            # Detect and match features
            keypoints, descriptors = self.feature_detector.detectAndCompute(current_frame, None)
            
            if descriptors is None or self.prev_descriptors is None:
                self.logger.warning("No features detected for motion estimation")
                return self._create_zero_motion(timestamp)
            
            # Match features
            matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                self.logger.warning(f"Insufficient matches for motion estimation: {len(good_matches)}")
                return self._create_zero_motion(timestamp)
            
            # Extract matched points
            prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
            curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
            
            # Estimate affine transformation
            transform_matrix = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
            
            if transform_matrix is not None:
                motion = self._extract_motion_parameters(transform_matrix, timestamp)
                
                # Update reference frame
                self._update_reference_frame(current_frame, keypoints, descriptors)
                
                # Add to history
                self.camera_motion_history.append(motion)
                if len(self.camera_motion_history) > self.history_length:
                    self.camera_motion_history = self.camera_motion_history[-self.history_length:]
                
                return motion
            else:
                return self._create_zero_motion(timestamp)
                
        except Exception as e:
            self.logger.error(f"Error in camera motion estimation: {e}")
            return self._create_zero_motion(timestamp)
        
    def estimate_camera_motion(self, frame: np.ndarray) -> CameraMotion:
        """Backward compatibility method"""
        import time
        timestamp = time.time()
        return self.update_camera_motion(frame, timestamp)

    def predict_track_position(self, track: Track, steps_ahead: int = 1) -> PredictedPosition:
        """
        Predict future position of a track
        
        Args:
            track: Track to predict
            steps_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted position
        """
        if track.track_id not in self.track_histories:
            self.track_histories[track.track_id] = []
        
        # Update track history
        if track.detections:
            self.track_histories[track.track_id].extend(track.detections)
            # Keep only recent history
            if len(self.track_histories[track.track_id]) > self.history_length:
                self.track_histories[track.track_id] = self.track_histories[track.track_id][-self.history_length:]
        
        history = self.track_histories[track.track_id]
        
        if len(history) < 2:
            # Not enough history for prediction
            current_bbox = track.to_tlbr()
            return PredictedPosition(
                bbox=current_bbox,
                confidence=0.5,
                velocity=np.array([0.0, 0.0]),
                timestamp=0.0
            )
        
        # Calculate velocity from recent positions
        velocity = self._calculate_track_velocity(history)
        
        # Apply camera motion compensation
        compensated_velocity = self._compensate_velocity_for_camera_motion(velocity)
        
        # Predict future position
        current_bbox = track.to_tlbr()
        current_center = np.array([(current_bbox[0] + current_bbox[2]) / 2,
                                  (current_bbox[1] + current_bbox[3]) / 2])
        
        predicted_center = current_center + compensated_velocity * steps_ahead
        
        # Maintain bbox size
        bbox_width = current_bbox[2] - current_bbox[0]
        bbox_height = current_bbox[3] - current_bbox[1]
        
        predicted_bbox = np.array([
            predicted_center[0] - bbox_width / 2,
            predicted_center[1] - bbox_height / 2,
            predicted_center[0] + bbox_width / 2,
            predicted_center[1] + bbox_height / 2
        ])
        
        # Calculate prediction confidence
        confidence = self._calculate_prediction_confidence(history, velocity)
        
        return PredictedPosition(
            bbox=predicted_bbox,
            confidence=confidence,
            velocity=compensated_velocity,
            timestamp=0.0
        )
    
    def compensate_tracks_for_motion(self, tracks: List[Track]) -> List[Track]:
        """
        Compensate track positions for camera motion
        
        Args:
            tracks: List of tracks to compensate
            
        Returns:
            List of motion-compensated tracks
        """
        if not self.camera_motion_history:
            return tracks
        
        recent_motion = self.camera_motion_history[-1]
        
        compensated_tracks = []
        for track in tracks:
            compensated_track = self._compensate_track_for_motion(track, recent_motion)
            compensated_tracks.append(compensated_track)
        
        return compensated_tracks
    
    def _extract_motion_parameters(self, transform_matrix: np.ndarray, timestamp: float) -> CameraMotion:
        """Extract motion parameters from transformation matrix"""
        # Extract translation
        translation = transform_matrix[:2, 2]
        
        # Extract rotation and scale
        a = transform_matrix[0, 0]
        b = transform_matrix[0, 1]
        
        scale = np.sqrt(a*a + b*b)
        rotation = np.arctan2(b, a)
        
        # Calculate confidence based on motion magnitude
        motion_magnitude = np.linalg.norm(translation)
        confidence = min(1.0, max(0.1, 1.0 / (1.0 + motion_magnitude / 10.0)))
        
        return CameraMotion(
            translation=translation,
            rotation=rotation,
            scale=scale,
            confidence=confidence,
            timestamp=timestamp
        )
    
    def _calculate_track_velocity(self, history: List[Detection]) -> np.ndarray:
        """Calculate track velocity from detection history"""
        if len(history) < 2:
            return np.array([0.0, 0.0])
        
        # Get recent positions
        recent_positions = []
        for detection in history[-5:]:  # Last 5 detections
            center = detection.center
            recent_positions.append(center)
        
        if len(recent_positions) < 2:
            return np.array([0.0, 0.0])
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            vel = recent_positions[i] - recent_positions[i-1]
            velocities.append(vel)
        
        return np.mean(velocities, axis=0)
    
    def _compensate_velocity_for_camera_motion(self, object_velocity: np.ndarray) -> np.ndarray:
        """Compensate object velocity for camera motion"""
        if not self.camera_motion_history:
            return object_velocity
        
        # Get recent camera motion
        recent_motion = self.camera_motion_history[-1]
        camera_velocity = recent_motion.translation
        
        # Subtract camera motion from object motion
        compensated_velocity = object_velocity - camera_velocity
        
        return compensated_velocity
    
    def _calculate_prediction_confidence(self, history: List[Detection], velocity: np.ndarray) -> float:
        """Calculate confidence for motion prediction"""
        if len(history) < 3:
            return 0.5
        
        # Calculate motion consistency
        recent_velocities = []
        positions = [det.center for det in history[-5:]]
        
        for i in range(1, len(positions)):
            vel = positions[i] - positions[i-1]
            recent_velocities.append(vel)
        
        if len(recent_velocities) < 2:
            return 0.5
        
        # Calculate velocity standard deviation
        velocities_array = np.array(recent_velocities)
        velocity_std = np.std(velocities_array, axis=0)
        max_std = np.max(velocity_std)
        
        # Higher consistency = higher confidence
        consistency = 1.0 / (1.0 + max_std)
        
        # Factor in motion magnitude (very fast motion is less reliable)
        velocity_magnitude = np.linalg.norm(velocity)
        magnitude_factor = 1.0 / (1.0 + velocity_magnitude / 20.0)
        
        confidence = consistency * magnitude_factor
        return np.clip(confidence, 0.1, 1.0)
    
    def _compensate_track_for_motion(self, track: Track, camera_motion: CameraMotion) -> Track:
        """Compensate individual track for camera motion"""
        # This is a simplified implementation
        # In practice, you'd modify the track's internal state
        return track
    
    def _create_zero_motion(self, timestamp: float) -> CameraMotion:
        """Create a zero motion object"""
        return CameraMotion(
            translation=np.array([0.0, 0.0]),
            rotation=0.0,
            scale=1.0,
            confidence=1.0,
            timestamp=timestamp
        )
    
    def _update_reference_frame(self, frame: np.ndarray, 
                               keypoints: Optional[List] = None,
                               descriptors: Optional[np.ndarray] = None):
        """Update reference frame for motion tracking"""
        self.prev_frame = frame.copy()
        
        if keypoints is not None and descriptors is not None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
        else:
            # Detect features if not provided
            self.prev_keypoints, self.prev_descriptors = self.feature_detector.detectAndCompute(frame, None)
    
    def get_motion_statistics(self) -> Dict[str, Any]:
        """Get motion tracking statistics"""
        if not self.camera_motion_history:
            return {"motion_detected": False}
        
        # Analyze recent motion
        recent_motions = self.camera_motion_history[-5:]
        translations = [motion.translation for motion in recent_motions]
        rotations = [motion.rotation for motion in recent_motions]
        confidences = [motion.confidence for motion in recent_motions]
        
        return {
            "motion_detected": True,
            "avg_translation": np.mean([np.linalg.norm(t) for t in translations]),
            "max_translation": np.max([np.linalg.norm(t) for t in translations]),
            "avg_rotation": np.mean([abs(r) for r in rotations]),
            "avg_confidence": np.mean(confidences),
            "motion_frames": len(self.camera_motion_history)
        }

class EnhancedTrackMatcher:
    """
    Enhanced track matching using motion prediction
    """
    
    def __init__(self, motion_predictor: MotionPredictor):
        """
        Initialize enhanced track matcher
        
        Args:
            motion_predictor: Motion predictor instance
        """
        self.motion_predictor = motion_predictor
        self.logger = logging.getLogger(f"{__name__}.EnhancedTrackMatcher")
    
    def match_tracks_with_predictions(self, 
                                    tracks: List[Track], 
                                    detections: List[Detection]) -> List[Tuple[int, int]]:
        """
        Match tracks to detections using motion predictions
        
        Args:
            tracks: List of existing tracks
            detections: List of new detections
            
        Returns:
            List of (track_idx, detection_idx) matches
        """
        if not tracks or not detections:
            return []
        
        # Generate predictions for all tracks
        predictions = []
        for track in tracks:
            prediction = self.motion_predictor.predict_track_position(track)
            predictions.append(prediction)
        
        # Calculate cost matrix using predictions
        cost_matrix = self._calculate_prediction_cost_matrix(predictions, detections)
        
        # Use Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by cost threshold
        matches = []
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, det_idx] < 0.5:  # Cost threshold
                matches.append((track_idx, det_idx))
        
        return matches
    
    def _calculate_prediction_cost_matrix(self, 
                                        predictions: List[PredictedPosition],
                                        detections: List[Detection]) -> np.ndarray:
        """Calculate cost matrix using motion predictions"""
        cost_matrix = np.ones((len(predictions), len(detections)))
        
        for i, prediction in enumerate(predictions):
            for j, detection in enumerate(detections):
                # Calculate IoU between predicted and actual position
                iou = self._calculate_iou(prediction.bbox, detection.bbox)
                
                # Calculate distance between centers
                pred_center = np.array([(prediction.bbox[0] + prediction.bbox[2]) / 2,
                                       (prediction.bbox[1] + prediction.bbox[3]) / 2])
                det_center = detection.center
                distance = np.linalg.norm(pred_center - det_center)
                
                # Combine IoU and distance with prediction confidence
                cost = (1.0 - iou) + (distance / 100.0)  # Normalize distance
                cost = cost * (2.0 - prediction.confidence)  # Weight by confidence
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

# Factory functions for easy creation
def create_motion_predictor(history_length: int = 10,
                           motion_threshold: float = 2.0,
                           prediction_steps: int = 5) -> MotionPredictor:
    """Create a motion predictor with specified parameters"""
    return MotionPredictor(history_length, motion_threshold, prediction_steps)

def create_enhanced_track_matcher(motion_predictor: MotionPredictor) -> EnhancedTrackMatcher:
    """Create an enhanced track matcher"""
    return EnhancedTrackMatcher(motion_predictor)