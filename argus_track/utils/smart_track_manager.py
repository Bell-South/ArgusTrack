# argus_track/utils/clean_track_manager.py

"""
Clean Track Manager - GPS-informed tracking without complexity
============================================================

Focused track management that uses GPS movement context to solve:
- Track fragmentation 
- Track ID resurrection
- Duplicate ID assignment

GPS Movement Logic:
- If vehicle moved forward > 50m, don't reuse old track IDs
- If vehicle is stationary, allow limited track reappearance  
- Conservative ID management for forward motion
"""

import numpy as np
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
import logging
import time
from ..utils.gps_motion_prediction import MotionPredictor, create_motion_prediction_config, VehicleMovement
from ..core import Detection
from ..config import TrackerConfig


@dataclass
class TrackMemory:
    """Simple track memory without complex features"""
    track_id: int
    last_seen_frame: int
    last_position: np.ndarray
    detection_count: int = 0
    confidence_history: List[float] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    
    def update(self, detection: Detection, frame_id: int):
        """Update track memory with new detection"""
        self.last_position = detection.center
        self.last_seen_frame = frame_id
        self.detection_count += 1
        self.last_update_time = time.time()
        
        self.confidence_history.append(detection.score)
        if len(self.confidence_history) > 10:
            self.confidence_history = self.confidence_history[-10:]


class CleanTrackManager:
    """
    Clean track manager with GPS movement context
    
    Key principles:
    1. Use GPS to understand vehicle movement
    2. Prevent impossible track resurrections in forward motion
    3. Conservative ID management 
    4. Simple, predictable behavior
    """
    # Temporal resurrection constants
    MIN_DISTANCE_DIFFERENT_OBJECT_M = 12.0      # Vehicle travel distance to consider new object(15)
    FORBID_DISTANCE_THRESHOLD_M = 20.0          # Distance to proactively forbid resurrections
    MAX_TIME_SAME_OBJECT_S = 2.5                # Max time for same object resurrection(3)
    FAST_SPEED_THRESHOLD_MS = 5.0               # Speed threshold for restrictive policy
    STATIONARY_THRESHOLD_MS = 0.5               # Speed threshold for lenient policy
    MAX_FRAMES_FAST_MOVING = 3                  # Max frames gap at high speed
    MAX_FRAMES_SAME_POSITION = 5                # Max frames for same position resurrection
    DETECTION_POSITION_TOLERANCE_PX = 50.0      # Tolerance for "same position"
    
    def __init__(self, config: TrackerConfig):
        """Initialize clean track manager"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CleanTrackManager")
        
        # Track memory storage
        self.track_memories: Dict[int, TrackMemory] = {}
        self.active_track_ids: Set[int] = set()
        self.lost_track_ids: Set[int] = set()
        
        # GPS movement context
        self.vehicle_speed: float = 0.0
        self.total_distance_moved: float = 0.0
        self.distance_since_last_cleanup: float = 0.0
        
        # ID management
        self.next_track_id = 1
        self.id_reuse_forbidden: Set[int] = set()  # IDs we should never reuse
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.total_resurrections_prevented = 0
        self.fragmentation_fixes = 0
        
        self.track_positions = {}  # track_id -> last_known_position  
        self.track_death_positions = {}  # track_id -> position_when_lost
        self.forbidden_resurrections = set()  # track_ids that should never resurrect

        self.motion_config = create_motion_prediction_config(
        object_distance_m=30.0,
        gps_accuracy_threshold_m=3.0, 
        prediction_tolerance_px=15.0,
        enable_debug=True
        )
        self.motion_predictor = MotionPredictor(self.motion_config)

        # GPS context
        self.current_gps = None
        self.previous_gps = None
        # Initialize vehicle movement tracking
        self.current_vehicle_movement = None
        self.previous_vehicle_movement = None

        self.motion_predictions = {}
        
        self.logger.info("Motion prediction enabled in CleanTrackManager")

        self.logger.info("Clean Track Manager initialized")
    
    def update_gps_context(self, current_gps, previous_gps=None):
        """Update GPS context for motion prediction"""
        # Don't overwrite previous_gps if we already have one from the actual previous frame
        if self.current_gps is not None:
            self.previous_gps = self.current_gps
        
        self.current_gps = current_gps
        
        # Only use passed previous_gps if we don't have context yet
        if self.previous_gps is None and previous_gps is not None:
            self.previous_gps = previous_gps
        
        self.logger.debug(f"GPS context updated: prev_time={getattr(self.previous_gps, 'timestamp', 'None')}, "
                        f"curr_time={getattr(self.current_gps, 'timestamp', 'None')}")

    def update_movement_context_try(self, gps_data):
        """Update movement context from GPS data"""
        if gps_data:
            # Calculate speed in m/s
            speed_ms = getattr(gps_data, 'speed', 0.0) if gps_data else 0.0
            
            # Create a simple movement object
            from collections import namedtuple
            VehicleMovement = namedtuple('VehicleMovement', ['speed_ms'])
            self.current_vehicle_movement = VehicleMovement(speed_ms=speed_ms)
        else:
            self.current_vehicle_movement = None

    def update_movement_context(self, vehicle_speed: float, 
                               distance_moved: float, total_distance: float):
        """Update GPS movement context for tracking decisions"""
        self.vehicle_speed = vehicle_speed
        self.total_distance_moved = total_distance
        self.distance_since_last_cleanup += distance_moved
        
        # If we've moved forward significantly, mark old IDs as forbidden
        if distance_moved > 10.0:  # 10 meters forward
            self._mark_distant_tracks_as_forbidden()
        
        # Periodic cleanup based on distance
        if self.distance_since_last_cleanup > 50.0:  # Every 50 meters
            self._cleanup_distant_tracks()
            self.distance_since_last_cleanup = 0.0
    
    def _calculate_association_tolerance(self) -> float:
        """Calculate dynamic tolerance based on vehicle motion"""
        base_tolerance = 300.0
        
        if self.vehicle_speed > 1.0:  # Vehicle moving
            # Increase tolerance based on speed (max 2x for 10+ m/s)
            speed_factor = min(2.0, 1.0 + (self.vehicle_speed / 10.0))
            return base_tolerance * speed_factor
        
        return base_tolerance

    def _generate_motion_predictions(self, frame_id: int) -> Dict[int, np.ndarray]:
        """Generate pixel-space predictions and maintain track continuity"""
        self.motion_predictions.clear()
        
        if self.previous_gps is None or self.current_gps is None:
            self.logger.debug(f"Frame {frame_id}: No GPS data for motion prediction")
            return {}
        
        # ENHANCED: Include lost tracks that might reappear
        current_positions = {}
        
        # Add active tracks
        for track_id in self.active_track_ids:
            if track_id in self.track_positions:
                current_positions[track_id] = self.track_positions[track_id]
        
        # NEW: Add recently lost tracks (within 60 frames = ~6 seconds)
        recently_lost_tracks = {}
        for track_id in list(self.lost_track_ids):
            if track_id in self.track_memories:
                memory = self.track_memories[track_id]
                frames_since_lost = frame_id - memory.last_seen_frame
                
                # Keep recently lost tracks alive with predictions
                if frames_since_lost <= 60 and track_id in self.track_positions:
                    recently_lost_tracks[track_id] = self.track_positions[track_id]
                    self.logger.debug(f"Frame {frame_id}: Including lost track {track_id} "
                                    f"in predictions (lost {frames_since_lost} frames ago)")
        
        # Combine active and recently lost
        all_positions = {**current_positions, **recently_lost_tracks}
        
        if not all_positions:
            self.logger.debug(f"Frame {frame_id}: No track positions for prediction")
            return {}
        
        try:
            # Use existing MotionPredictor
            predictions = self.motion_predictor.predict_object_positions(
                self.previous_gps, self.current_gps, all_positions
            )
            
            # Store for debugging and extract positions
            self.motion_predictions = predictions
            predicted_positions = {}
            
            for track_id, info in predictions.items():
                predicted_positions[track_id] = info['predicted_position']
                confidence = info['confidence']
                
                # NEW: Reactivate lost tracks with good predictions
                if track_id in recently_lost_tracks and confidence > 0.6:
                    self.logger.info(f"Frame {frame_id}: Reactivating lost track {track_id} "
                                f"with prediction confidence {confidence:.2f}")
                    self.lost_track_ids.discard(track_id)
                    self.active_track_ids.add(track_id)
                    
                    # Update position with prediction
                    self.track_positions[track_id] = predicted_positions[track_id]
                
                self.logger.debug(f"Frame {frame_id}: Track {track_id} predicted at "
                                f"{predicted_positions[track_id]} (conf: {confidence:.2f})")
            
            return predicted_positions
            
        except Exception as e:
            self.logger.error(f"Motion prediction failed: {e}")
            return {}

    def _generate_motion_predictions_bkp(self, frame_id: int) -> Dict[int, np.ndarray]:
        """Generate pixel-space predictions for active tracks"""
        self.motion_predictions.clear()
        
        if self.previous_gps is None or self.current_gps is None:
            self.logger.debug(f"Frame {frame_id}: No GPS data for motion prediction")
            return {}
        
        # Get current track positions
        current_positions = {}

        for track_id in self.active_track_ids:
            if track_id in self.track_positions:
                current_positions[track_id] = self.track_positions[track_id]
        
        if not current_positions:
            self.logger.debug(f"Frame {frame_id}: No active track positions")
            return {}
        
        try:
            # Use existing MotionPredictor
            predictions = self.motion_predictor.predict_object_positions(
                self.previous_gps, self.current_gps, current_positions
            )
            
            # Store for debugging and extract positions
            self.motion_predictions = predictions
            predicted_positions = {}
            
            for track_id, info in predictions.items():
                predicted_positions[track_id] = info['predicted_position']
                confidence = info['confidence']
                self.logger.debug(f"Frame {frame_id}: Track {track_id} predicted at {predicted_positions[track_id]} (conf: {confidence:.2f})")
            
            return predicted_positions
            
        except Exception as e:
            self.logger.error(f"Motion prediction failed: {e}")
            return {}

    def process_frame_detections(self, detections: List[Detection], 
                                frame_id: int, timestamp: float) -> List[Detection]:
        """
        Process frame detections with GPS-informed track management
        
        Args:
            detections: Raw detections from overlap fixer
            frame_id: Current frame ID
            timestamp: Frame timestamp
            
        Returns:
            Detections with clean track IDs
        """
        processed_detections = []
        
        for detection in detections:
            original_track_id = detection.track_id
            
            # Check if this track ID should be reused
            clean_track_id = self._get_clean_track_id(original_track_id, detection, frame_id)
            
            # Update detection with clean ID
            detection.track_id = clean_track_id
            processed_detections.append(detection)
            
            # Update track memory
            self._update_track_memory(detection, frame_id)
        
        if frame_id % 20 == 0:
            self._cleanup_old_tracks()

        self._handle_lost_tracks(frame_id)
        
        return processed_detections
    
    def _handle_resurrection_and_new_tracks(self, original_id: int, detection: Detection, frame_id: int) -> int:
        """Handle resurrection logic and new track creation with more lenient rules"""
        
        # Check if this is a forbidden resurrection
        if original_id in getattr(self, 'forbidden_resurrections', set()):
            new_id = self._assign_new_track_id()
            self.total_resurrections_prevented += 1
            self.logger.info(f"Frame {frame_id}: Prevented forbidden resurrection {original_id} -> {new_id}")
            return new_id
        
        # Enhanced resurrection check - be more lenient for recent tracks
        if original_id in self.lost_track_ids:
            if self._resurrection_makes_spatial_sense_enhanced(original_id, detection, frame_id):
                # Allow resurrection
                self.lost_track_ids.discard(original_id)
                self.active_track_ids.add(original_id)
                self.logger.info(f"Frame {frame_id}: Allowed logical resurrection {original_id}")
                return original_id
            else:
                # Block resurrection but be less aggressive
                self.forbidden_resurrections.add(original_id)
                new_id = self._assign_new_track_id()
                self.total_resurrections_prevented += 1
                self.logger.info(f"Frame {frame_id}: Prevented illogical resurrection {original_id} -> {new_id}")
                return new_id
        
        # Create new track
        new_id = self._assign_new_track_id()
        return new_id

    def _resurrection_makes_temporal_sense(self, track_id: int, detection: Detection, frame_id: int) -> bool:

        if track_id not in self.track_memories:
            return False
        
        memory = self.track_memories[track_id]
        frames_since_death = frame_id - memory.last_seen_frame
        time_since_death = frames_since_death / 10.0  # seconds (assuming 10fps)
        
        # Calculate how far vehicle has traveled since track death
        distance_traveled = self.vehicle_speed * time_since_death
        
        # CORE LOGIC: If vehicle has moved far enough, this CANNOT be the same object
        
        # For light posts and street infrastructure:
        # - Typical spacing: 30-100 meters apart
        # - If vehicle traveled >15m, likely a different object
        
        min_distance_for_new_object = 15.0  # meters
        
        if distance_traveled > min_distance_for_new_object:
            self.logger.warning(f"Resurrection blocked: track {track_id} - vehicle traveled {distance_traveled:.1f}m "
                            f"since death (>{min_distance_for_new_object}m = different object)")
            return False
        
        # Additional check: Time-based cutoff
        # Even at slow speeds, after enough time it's likely a different object
        max_time_same_object = 3.0  # seconds
        
        if time_since_death > max_time_same_object:
            self.logger.warning(f"Resurrection blocked: track {track_id} - {time_since_death:.1f}s since death "
                            f"(>{max_time_same_object}s = different object)")
            return False
        
        # If we get here, resurrection might be legitimate
        return True

    def _resurrection_makes_spatial_sense_enhanced(self, track_id: int, detection: Detection, frame_id: int) -> bool:
        """
        ENHANCED: Spatial check that considers screen position reuse
        """
        
        if track_id not in self.track_death_positions:
            return False
        
        death_position = self.track_death_positions[track_id]
        new_position = detection.center
        distance = np.linalg.norm(new_position - death_position)
        
        # For same screen position resurrections, be more restrictive
        if distance < 50.0:  # Very close to death position
            
            # This could be legitimate (same object reappearing) OR
            # different object at same screen position
            
            # Use temporal logic to decide
            if not self._resurrection_makes_temporal_sense(track_id, detection, frame_id):
                return False
            
            # Additional check: if vehicle is moving, same position resurrections are suspicious
            if self.vehicle_speed > 1.0:
                
                memory = self.track_memories[track_id]
                frames_since_death = frame_id - memory.last_seen_frame
                
                # For moving vehicle, only allow very quick resurrections at same position
                if frames_since_death > 5:  # 0.5 seconds
                    self.logger.warning(f"Resurrection blocked: track {track_id} - same position after {frames_since_death} "
                                    f"frames with moving vehicle (likely different object)")
                    return False
        
        # For distant resurrections, use standard distance check
        max_distance = 100.0  # Reasonable limit
        if distance > max_distance:
            self.logger.debug(f"Resurrection blocked: track {track_id} moved {distance:.1f}px (max: {max_distance})")
            return False
        
        return True

    def _resurrection_makes_spatial_sense_enhanced_bkp(self, track_id: int, detection: Detection, frame_id: int) -> bool:
        """Enhanced spatial sense check for resurrections"""
        
        if track_id not in self.track_death_positions:
            # If we don't know where it died, be more lenient
            return True
        
        death_position = self.track_death_positions[track_id]
        new_position = detection.center
        distance = np.linalg.norm(new_position - death_position)
        
        max_allowed_distance = 100.0  # Increased from 30px to 100px
        
        # Be even more lenient when vehicle is moving fast
        if self.vehicle_speed > 2.0:
            max_allowed_distance = 150.0  # Even higher tolerance for moving vehicle
        
        spatial_ok = distance <= max_allowed_distance
        
        if not spatial_ok:
            self.logger.warning(f"Resurrection blocked: track {track_id} moved {distance:.1f}px "
                            f"from death position (threshold: {max_allowed_distance:.1f}px)")
        else:
            self.logger.info(f"Resurrection allowed: track {track_id} moved {distance:.1f}px "
                            f"within threshold {max_allowed_distance:.1f}px")
        
        return spatial_ok

    def _proactive_forbid_distant_tracks(self, frame_id: int):
        """
        PROACTIVE: Forbid resurrections for tracks that are definitely behind us
        """
        
        tracks_to_forbid = []
        
        for track_id in list(self.lost_track_ids):
            if track_id in self.track_memories:
                memory = self.track_memories[track_id]
                frames_since_death = frame_id - memory.last_seen_frame
                time_since_death = frames_since_death / 10.0
                
                # Calculate distance traveled since death
                distance_traveled = self.vehicle_speed * time_since_death
                
                # If we've traveled far enough, this track is definitely behind us
                forbid_distance = 20.0  # meters
                
                if distance_traveled > forbid_distance:
                    tracks_to_forbid.append(track_id)
        
        # Move to forbidden resurrections
        for track_id in tracks_to_forbid:
            self.lost_track_ids.discard(track_id)
            self.forbidden_resurrections.add(track_id)
            self.logger.debug(f"Track {track_id} proactively forbidden - vehicle traveled "
                            f"{self.vehicle_speed * (frame_id - self.track_memories[track_id].last_seen_frame) / 10.0:.1f}m")

    def _resurrection_vehicle_context_check(self, track_id: int, detection: Detection, frame_id: int) -> bool:
        """
        Check resurrection against vehicle movement context
        """
        
        # If vehicle is stationary, allow more resurrections
        if self.vehicle_speed < 0.5:
            return True
        
        # If vehicle is moving fast, be very restrictive
        if self.vehicle_speed > 5.0:
            memory = self.track_memories[track_id]
            frames_since_death = frame_id - memory.last_seen_frame
            
            # At high speed, only allow immediate resurrections (tracking gaps)
            if frames_since_death > 3:  # 0.3 seconds
                self.logger.debug(f"Resurrection blocked: track {track_id} - vehicle too fast ({self.vehicle_speed:.1f}m/s) "
                                f"for {frames_since_death}-frame gap")
                return False
        
        return True

    def _enhanced_resurrection_check(self, track_id: int, detection: Detection, frame_id: int) -> bool:
        """
        COMPREHENSIVE: Combine all resurrection checks
        """
        
        # Check 1: Basic spatial sense (enhanced)
        if not self._resurrection_makes_spatial_sense_enhanced(track_id, detection, frame_id):
            return False
        
        # Check 2: Temporal sense (NEW - this is the key for your problem)
        if not self._resurrection_makes_temporal_sense(track_id, detection, frame_id):
            return False
        
        # Check 3: Vehicle context
        if not self._resurrection_vehicle_context_check(track_id, detection, frame_id):
            return False
        
        return True

    def _handle_resurrection_and_new_tracks_bkp(self, original_id: int, detection: Detection, frame_id: int) -> int:
        """Handle resurrection logic and new track creation (existing logic)"""

        if original_id in self.forbidden_resurrections:
            new_id = self._assign_new_track_id()
            self.total_resurrections_prevented += 1
            self.logger.info(f"Frame {frame_id}: Prevented forbidden resurrection {original_id} -> {new_id}")
            return new_id
        
        # ENHANCED RESURRECTION CHECK: Only resurrect if spatially makes sense
        if original_id in self.lost_track_ids:
            if not self._resurrection_makes_spatial_sense(original_id, detection, frame_id):
                # Block this resurrection and forbid future ones
                self.forbidden_resurrections.add(original_id)
                new_id = self._assign_new_track_id()
                self.total_resurrections_prevented += 1
                self.logger.info(f"Frame {frame_id}: Prevented illogical resurrection {original_id} -> {new_id}")
                return new_id
            else:
                # Resurrection is OK - remove from lost tracks
                self.lost_track_ids.discard(original_id)
                self.active_track_ids.add(original_id)
                self.logger.info(f"Frame {frame_id}: Allowed logical resurrection {original_id}")
                return original_id
        
        # If this ID is forbidden (was problematic before), assign new one
        if original_id in self.id_reuse_forbidden:
            new_id = self._assign_new_track_id()
            self.total_resurrections_prevented += 1
            self.logger.info(f"Frame {frame_id}: Prevented reuse of forbidden ID {original_id} -> {new_id}")
            return new_id
        
        # This appears to be a new track - use original ID if available
        if original_id not in self.track_memories and original_id not in self.active_track_ids:
            self.active_track_ids.add(original_id)
            self.total_tracks_created += 1
            return original_id
        
        # ID conflict - assign new one
        new_id = self._assign_new_track_id()
        self.total_tracks_created += 1
        return new_id
    
    def _get_clean_track_id(self, original_id: int, detection: Detection, frame_id: int) -> int:
        """Enhanced track ID assignment with motion prediction"""
        detection_center = detection.center
        
        motion_predictions = self._generate_motion_predictions(frame_id)
        
        if motion_predictions:
            best_match_id = None
            best_distance = float('inf')
            
            for track_id, predicted_pos in motion_predictions.items():
                distance = np.linalg.norm(detection_center - predicted_pos)
                
               
                tolerance = self.motion_config.prediction_tolerance_px
                if self.current_gps and hasattr(self.current_gps, 'speed') and self.current_gps.speed > 2.0:
                    tolerance *= 2.0  # Double tolerance for fast movement
                
                if distance < tolerance and distance < best_distance:
                    best_match_id = track_id
                    best_distance = distance
            
            if best_match_id is not None:
                # IMPORTANT: Reactivate the track if it was lost
                if best_match_id in self.lost_track_ids:
                    self.lost_track_ids.discard(best_match_id)
                    self.active_track_ids.add(best_match_id)
                    self.logger.info(f"Frame {frame_id}: Motion prediction reactivated track {best_match_id}")
                
                self.logger.info(f"Frame {frame_id}: Motion prediction match {original_id} -> {best_match_id} "
                            f"(error: {best_distance:.1f}px)")
                return best_match_id
        
        # Continue with existing fallback logic...
        return self._get_fallback_track_id(original_id, detection, frame_id)

    def _get_clean_track_id_bkp(self, original_id: int, detection: Detection, frame_id: int) -> int:
        """Enhanced track ID assignment with motion prediction"""
        detection_center = detection.center
        
        # PHASE 1: Try motion prediction matching first
        motion_predictions = self._generate_motion_predictions(frame_id)
        
        if motion_predictions:
            best_match_id = None
            best_distance = float('inf')
            
            for track_id, predicted_pos in motion_predictions.items():
                distance = np.linalg.norm(detection_center - predicted_pos)
                
                # Use motion prediction tolerance
                if distance < self.motion_config.prediction_tolerance_px and distance < best_distance:
                    best_match_id = track_id
                    best_distance = distance
            
            if best_match_id is not None:
                self.logger.info(f"Frame {frame_id}: Motion prediction match {original_id} -> {best_match_id} (error: {best_distance:.1f}px)")
                return best_match_id
        
        # FALLBACK: Enhanced anti-fragmentation with dynamic tolerance
        dynamic_tolerance = self._calculate_association_tolerance()
        
        for active_id in self.active_track_ids:
            if active_id in self.track_positions:
                distance = np.linalg.norm(detection_center - self.track_positions[active_id])
                if distance < dynamic_tolerance:
                    self.logger.info(f"Frame {frame_id}: Position-based match {original_id} -> {active_id} (distance: {distance:.1f}px)")
                    return active_id
        
        # Continue with existing logic for resurrections and new tracks
        return self._handle_resurrection_and_new_tracks(original_id, detection, frame_id)

    def _should_allow_track_reappearance(self, track_id: int, detection: Detection, 
                                       frame_id: int) -> bool:
        """
        Determine if track reappearance makes sense given GPS movement
        
        Logic:
        - If vehicle is stationary or slow: allow reappearance
        - If vehicle moved forward significantly: block reappearance
        - Conservative approach for forward motion
        """
        if track_id not in self.track_memories:
            return False
        
        track_memory = self.track_memories[track_id]
        frames_since_lost = frame_id - track_memory.last_seen_frame
        
        # Don't allow very old tracks to reappear
        if frames_since_lost > 30:  # ~3 seconds at 10fps
            return False
        
        # GPS-based logic
        if self.vehicle_speed < 1.0:  # Vehicle stationary/very slow
            # Allow reappearance for stationary vehicle
            # Object might have been temporarily occluded
            return frames_since_lost <= 15  # 1.5 seconds
        
        elif self.vehicle_speed < 5.0:  # Vehicle moving slowly 
            # More restrictive for slow movement
            return frames_since_lost <= 10  # 1 second
        
        else:  # Vehicle moving fast (>5 m/s = 18 km/h)
            # Very restrictive for fast movement
            # Object should be behind the vehicle quickly
            return frames_since_lost <= 5  # 0.5 seconds
    
    def _assign_new_track_id(self) -> int:
        """Assign a new track ID that hasn't been used"""
        while self.next_track_id in self.track_memories or self.next_track_id in self.id_reuse_forbidden:
            self.next_track_id += 1
        
        new_id = self.next_track_id
        self.next_track_id += 1
        self.total_tracks_created += 1
        
        return new_id

    def _resurrection_makes_spatial_sense_final(self, track_id: int, detection: Detection, frame_id: int) -> bool:
        """
        FINAL VERSION: Replace your existing _resurrection_makes_spatial_sense method with this
        """
        return self._enhanced_resurrection_check(track_id, detection, frame_id)

    def process_frame_detections_enhanced(self, detections: List[Detection], 
                                        frame_id: int, timestamp: float) -> List[Detection]:
        """
        ENHANCED: Add proactive track forbidding to your existing method
        """
        
        # ADDITION: Proactively forbid distant tracks
        if frame_id % 10 == 0:  # Every 10 frames
            self._proactive_forbid_distant_tracks(frame_id)
        
        # Continue with your existing logic...
        processed_detections = []
        
        for detection in detections:
            original_track_id = detection.track_id
            clean_track_id = self._get_clean_track_id(original_track_id, detection, frame_id)
            detection.track_id = clean_track_id
            processed_detections.append(detection)
            self._update_track_memory(detection, frame_id)
        
        if frame_id % 20 == 0:
            self._cleanup_old_tracks()
        
        self._handle_lost_tracks(frame_id)
        
        return processed_detections

    def _resurrection_makes_spatial_sense(self, track_id: int, detection: Detection, frame_id: int) -> bool:
        """Check if resurrecting this track makes spatial sense"""
        
        if track_id not in self.track_death_positions:
            return False  # Don't know where it died, don't resurrect
        
        death_position = self.track_death_positions[track_id]
        new_position = detection.center
        
        # Calculate distance from death position
        distance = np.linalg.norm(new_position - death_position)
        
        # For light posts (static objects), should appear very close to where they "died"
        max_allowed_distance = 30.0  # 30 pixels tolerance
        
        # If vehicle moved significantly, be more restrictive
        if self.total_distance_moved > 10.0:  # 10 meters
            max_allowed_distance = 15.0  # Only 15 pixels tolerance
        
        spatial_ok = distance <= max_allowed_distance
        
        if not spatial_ok:
            self.logger.warning(f"Resurrection blocked: track {track_id} moved {distance:.1f}px from death position")
        
        return spatial_ok

    def _update_track_memory(self, detection: Detection, frame_id: int):
        """Enhanced track memory update with position tracking"""
        
        track_id = detection.track_id
        
        # Call original memory update logic
        if track_id not in self.track_memories:
            self.track_memories[track_id] = TrackMemory(
                track_id=track_id,
                last_seen_frame=frame_id,
                last_position=detection.center
            )
            self.active_track_ids.add(track_id)
        
        # Update existing memory
        self.track_memories[track_id].update(detection, frame_id)
        
        # ENHANCEMENT: Always track current position
        self.track_positions[track_id] = detection.center.copy()

    def _should_keep_track_alive(self, track_id: int, frame_id: int) -> bool:
        """Determine if a track should be kept alive based on motion prediction"""
        
        if track_id not in self.track_memories:
            return False
        
        memory = self.track_memories[track_id]
        frames_since_seen = frame_id - memory.last_seen_frame
        
        # Keep tracks alive longer when vehicle is moving (they might reappear)
        if self.vehicle_speed > 1.0:
            max_frames_alive = 60  # 6 seconds at 10fps
        else:
            max_frames_alive = 30  # 3 seconds when stationary
        
        return frames_since_seen <= max_frames_alive

    def _handle_lost_tracks(self, current_frame: int):
        """Enhanced lost track handling with motion prediction"""
        
        # Find tracks that were updated this frame
        current_active = set()
        for track_id, memory in self.track_memories.items():
            if memory.last_seen_frame == current_frame:
                current_active.add(track_id)
        
        # Find tracks that should be marked as lost
        potentially_lost = self.active_track_ids - current_active
        
        for track_id in potentially_lost:
            # Check if we should keep this track alive
            if self._should_keep_track_alive(track_id, current_frame):
                self.logger.debug(f"Frame {current_frame}: Keeping track {track_id} alive for motion prediction")
                continue  # Don't mark as lost yet
            
            # Mark as truly lost
            self.active_track_ids.discard(track_id)
            self.lost_track_ids.add(track_id)
            
            # Remember where this track "died"
            if track_id in self.track_positions:
                self.track_death_positions[track_id] = self.track_positions[track_id].copy()
            
            self.total_tracks_lost += 1
            self.logger.debug(f"Track {track_id} lost at frame {current_frame}")

    def _handle_lost_tracks_bkp(self, current_frame: int):
        """Enhanced lost track handling with death position tracking"""
        
        # Find tracks that were updated this frame
        current_active = set()
        for track_id, memory in self.track_memories.items():
            if memory.last_seen_frame == current_frame:
                current_active.add(track_id)
        
        # Find newly lost tracks
        newly_lost = self.active_track_ids - current_active
        
        for track_id in newly_lost:
            # Move to lost tracks
            self.active_track_ids.discard(track_id)
            self.lost_track_ids.add(track_id)
            
            # ENHANCEMENT: Remember where this track "died"
            if track_id in self.track_positions:
                self.track_death_positions[track_id] = self.track_positions[track_id].copy()
            
            self.total_tracks_lost += 1
            self.logger.debug(f"Track {track_id} lost at frame {current_frame}")

    def _cleanup_old_tracks(self):
        """Cleanup old track data to prevent memory bloat"""
        
        # Move very old lost tracks to forbidden resurrections
        very_old_tracks = []
        for track_id in self.lost_track_ids:
            if track_id in self.track_memories:
                memory = self.track_memories[track_id]
                if hasattr(memory, 'last_seen_frame'):
                    # If track was lost more than 30 frames ago, forbid resurrection
                    current_frame = max([m.last_seen_frame for m in self.track_memories.values()])
                    if current_frame - memory.last_seen_frame > 30:
                        very_old_tracks.append(track_id)
        
        for track_id in very_old_tracks:
            self.lost_track_ids.discard(track_id)
            self.forbidden_resurrections.add(track_id)
            self.logger.debug(f"Track {track_id} moved to forbidden resurrections (too old)")
    
    def _mark_distant_tracks_as_forbidden(self):
        """Mark tracks as forbidden for reuse when vehicle moves forward significantly"""
        # Mark lost tracks as forbidden if vehicle moved forward
        if self.vehicle_speed > 2.0:  # Moving forward at reasonable speed
            newly_forbidden = set()
            
            for track_id in self.lost_track_ids.copy():
                if track_id in self.track_memories:
                    # This track is now behind the vehicle, forbid reuse
                    self.id_reuse_forbidden.add(track_id)
                    newly_forbidden.add(track_id)
            
            # Remove from lost tracks (they're now forbidden)
            self.lost_track_ids -= newly_forbidden
            
            if newly_forbidden:
                self.logger.debug(f"Marked {len(newly_forbidden)} track IDs as forbidden due to forward movement")
    
    def _cleanup_distant_tracks(self):
        """Cleanup tracks that are definitely behind the vehicle"""
        # Remove very old tracks from memory
        current_time = time.time()
        to_remove = []
        
        for track_id, memory in self.track_memories.items():
            age = current_time - memory.last_update_time
            
            # Remove tracks older than 30 seconds
            if age > 30.0:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_memories[track_id]
            self.active_track_ids.discard(track_id)
            self.lost_track_ids.discard(track_id)
            self.id_reuse_forbidden.add(track_id)  # Never reuse
        
        if to_remove:
            self.logger.debug(f"Cleaned up {len(to_remove)} old tracks")
        
        # Limit forbidden set size (keep memory reasonable)
        if len(self.id_reuse_forbidden) > 500:
            # Remove oldest forbidden IDs
            forbidden_list = sorted(list(self.id_reuse_forbidden))
            self.id_reuse_forbidden = set(forbidden_list[-400:])  # Keep newest 400
    
    def _get_fallback_track_id_bkp(self, original_id: int, detection, frame_id: int) -> int:
        """
        Fallback track ID assignment when motion prediction fails
        Uses existing position-based matching and resurrection logic
        """
        detection_center = detection.center
        
        # Step 1: Try to match with existing active tracks (anti-fragmentation)
        for active_id in self.active_track_ids:
            if active_id in self.track_positions:
                distance = np.linalg.norm(detection_center - self.track_positions[active_id])
                
                # Use dynamic tolerance based on vehicle speed if available
                base_tolerance = 300.0  # Base tolerance in pixels
                speed_factor = getattr(self, 'vehicle_speed', 0) * 2.0  # Extra tolerance for movement
                dynamic_tolerance = base_tolerance + min(speed_factor, 25.0)  # Cap the adjustment
                
                if distance < dynamic_tolerance:
                    self.logger.info(f"Frame {frame_id}: Prevented fragmentation {original_id} -> {active_id}")
                    return active_id
        
        # Step 2: Check for logical resurrection of lost tracks
        if hasattr(self, 'previous_track_positions') and self.previous_track_positions:
            for prev_id, prev_pos in self.previous_track_positions.items():
                if prev_id not in self.active_track_ids:  # Track is currently lost
                    distance = np.linalg.norm(detection_center - prev_pos)
                    
                    # Use resurrection distance threshold (should be defined in your class)
                    resurrection_threshold = getattr(self, 'resurrection_distance_threshold', 50.0)
                    
                    if distance < resurrection_threshold:
                        # Additional check: ensure track hasn't moved too far from death position
                        if hasattr(self, 'track_death_positions') and prev_id in self.track_death_positions:
                            death_pos = self.track_death_positions[prev_id]
                            death_distance = np.linalg.norm(detection_center - death_pos)
                            
                            # Logical resurrection threshold (should be defined in your class)
                            logical_threshold = getattr(self, 'logical_resurrection_threshold', 75.0)
                            
                            if death_distance < logical_threshold:
                                self.logger.info(f"Frame {frame_id}: Allowed logical resurrection {prev_id}")
                                # Remove from death positions since it's resurrected
                                if hasattr(self, 'track_death_positions'):
                                    del self.track_death_positions[prev_id]
                                return prev_id
                            else:
                                self.logger.warning(f"Resurrection blocked: track {prev_id} moved {death_distance:.1f}px from death position")
                                self.logger.info(f"Frame {frame_id}: Prevented illogical resurrection {prev_id} -> {original_id}")
                                return original_id
                        else:
                            # Simple resurrection without death position check
                            self.logger.info(f"Frame {frame_id}: Allowed logical resurrection {prev_id}")
                            return prev_id
        
        # Step 3: Return original ID (new track)
        self.logger.info(f"Frame {frame_id}: New track {original_id}")
        return original_id

    def _get_fallback_track_id(self, original_id: int, detection, frame_id: int) -> int:
        """
        FIXED: Fallback track ID assignment that USES temporal logic
        """
        detection_center = detection.center
        
        # Step 1: Try to match with existing active tracks (anti-fragmentation)
        for active_id in self.active_track_ids:
            if active_id in self.track_positions:
                distance = np.linalg.norm(detection_center - self.track_positions[active_id])
                
                # Use dynamic tolerance based on vehicle speed
                base_tolerance = 300.0  # Your working value
                speed_factor = self.vehicle_speed * 2.0
                dynamic_tolerance = base_tolerance + min(speed_factor, 25.0)
                
                if distance < dynamic_tolerance:
                    self.logger.info(f"Frame {frame_id}: Prevented fragmentation {original_id} -> {active_id}")
                    return active_id
        
        # Step 2: FIXED - Use your enhanced resurrection logic instead of bypassing it
        return self._handle_resurrection_and_new_tracks(original_id, detection, frame_id)

    def get_statistics(self) -> Dict[str, Any]:

        """Enhanced statistics with fragmentation and resurrection info"""
        
        # Your existing statistics
        base_stats = {
            'active_tracks': len(self.active_track_ids),
            'lost_tracks': len(self.lost_track_ids),
            'tracks_in_memory': len(self.track_memories),
            'forbidden_ids': len(self.id_reuse_forbidden),
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_lost': self.total_tracks_lost,
            'resurrections_prevented': self.total_resurrections_prevented,
            'next_track_id': self.next_track_id,
            'vehicle_speed': self.vehicle_speed,
            'distance_moved': self.total_distance_moved
        }
        
        # ADD THESE NEW METRICS:
        enhanced_stats = {
            **base_stats,
            'forbidden_resurrections': len(self.forbidden_resurrections),
            'death_positions_tracked': len(self.track_death_positions),
            'current_positions_tracked': len(self.track_positions),
            'avg_tracks_per_frame': len(self.active_track_ids),
            'resurrection_prevention_rate': len(self.forbidden_resurrections) / max(1, self.total_tracks_created)
        }
        
        return enhanced_stats