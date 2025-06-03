# Complete enhanced SmartTrackManager - replace the existing class

import numpy as np
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import logging
import time

from ..core import Detection, Track, GPSData
from .motion_prediction import MotionPredictor, CameraMotion, PredictedPosition, EnhancedTrackMatcher
from .visual_feature_extractor import VisualFeatureExtractor, VisualFeatures

@dataclass
class TrackMemory:
    """Enhanced memory container with motion prediction and visual features"""
    track_id: int
    last_seen_frame: int
    last_position: np.ndarray
    velocity_history: List[np.ndarray] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    detection_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    gps_positions: List[Tuple[float, float]] = field(default_factory=list)
    is_static: bool = False
    reliability_score: float = 0.5
    
    # Motion prediction attributes
    predicted_position: Optional[np.ndarray] = None
    prediction_confidence: float = 0.0
    prediction_accuracy_history: List[float] = field(default_factory=list)
    motion_compensated: bool = False
    camera_motion_history: List[Dict] = field(default_factory=list)
    
    # Visual feature attributes
    visual_features: Optional[VisualFeatures] = None
    feature_history: List[VisualFeatures] = field(default_factory=list)
    feature_match_history: List[float] = field(default_factory=list)
    appearance_stability: float = 0.5
    
    # GPS-enhanced motion attributes
    gps_velocity_history: List[Tuple[float, float]] = field(default_factory=list)  # (speed, heading)
    world_motion_compensation: bool = False
    
    def update(self, detection: Detection, frame_id: int, 
               gps_pos: Optional[Tuple[float, float]] = None,
               gps_velocity: Optional[Tuple[float, float]] = None,
               visual_features: Optional[VisualFeatures] = None):
        """Enhanced update with visual features and GPS motion"""
        
        # === MOTION UPDATES ===
        if self.last_position is not None:
            velocity = detection.center - self.last_position
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > 10:
                self.velocity_history = self.velocity_history[-10:]
        
        # GPS velocity tracking
        if gps_velocity:
            self.gps_velocity_history.append(gps_velocity)
            if len(self.gps_velocity_history) > 20:
                self.gps_velocity_history = self.gps_velocity_history[-20:]
        
        # === VISUAL FEATURE UPDATES ===
        if visual_features:
            # Calculate feature similarity with previous features
            if self.visual_features:
                from .visual_feature_extractor import VisualFeatureExtractor
                feature_extractor = VisualFeatureExtractor()
                similarity = feature_extractor.compare_features(self.visual_features, visual_features)
                self.feature_match_history.append(similarity)
                
                if len(self.feature_match_history) > 15:
                    self.feature_match_history = self.feature_match_history[-15:]
                
                # Update appearance stability
                if len(self.feature_match_history) >= 3:
                    self.appearance_stability = np.mean(self.feature_match_history[-5:])
            
            # Store current features
            self.visual_features = visual_features
            self.feature_history.append(visual_features)
            if len(self.feature_history) > 5:
                self.feature_history = self.feature_history[-5:]
        
        # === PREDICTION ACCURACY ===
        if self.predicted_position is not None:
            predicted_center = np.array([
                (self.predicted_position[0] + self.predicted_position[2]) / 2,
                (self.predicted_position[1] + self.predicted_position[3]) / 2
            ])
            actual_center = detection.center
            prediction_error = np.linalg.norm(actual_center - predicted_center)
            
            normalized_error = prediction_error / 1000.0
            accuracy = max(0.0, 1.0 - normalized_error)
            self.prediction_accuracy_history.append(accuracy)
            
            if len(self.prediction_accuracy_history) > 20:
                self.prediction_accuracy_history = self.prediction_accuracy_history[-20:]
        
        # === STANDARD UPDATES ===
        self.last_position = detection.center
        self.last_seen_frame = frame_id
        self.detection_count += 1
        self.last_update_time = time.time()
        
        self.confidence_history.append(detection.score)
        if len(self.confidence_history) > 20:
            self.confidence_history = self.confidence_history[-20:]
        
        if gps_pos:
            self.gps_positions.append(gps_pos)
            if len(self.gps_positions) > 50:
                self.gps_positions = self.gps_positions[-50:]
        
        if hasattr(detection, 'motion_compensated'):
            self.motion_compensated = detection.motion_compensated
        
        self._update_static_analysis()
        self._update_reliability_score()
        
        # Clear prediction for next frame
        self.predicted_position = None
        self.prediction_confidence = 0.0
    
    def get_combined_similarity(self, other_features: VisualFeatures, 
                               motion_distance: float, config) -> float:
        """Calculate combined visual + motion similarity"""
        visual_sim = 0.0
        motion_sim = 0.0
        
        # Visual similarity
        if self.visual_features and other_features:
            from .visual_feature_extractor import VisualFeatureExtractor
            feature_extractor = VisualFeatureExtractor()
            visual_sim = feature_extractor.compare_features(self.visual_features, other_features)
        
        # Motion similarity (inverse of distance, normalized)
        if motion_distance < 200:  # Within reasonable range
            motion_sim = max(0.0, 1.0 - motion_distance / 200.0)
        
        # Weighted combination
        visual_weight = config.feature_weight if config.enable_visual_features else 0.0
        motion_weight = config.motion_weight if config.enable_motion_prediction else 1.0
        
        # Normalize weights
        total_weight = visual_weight + motion_weight
        if total_weight > 0:
            visual_weight /= total_weight
            motion_weight /= total_weight
        
        combined_similarity = visual_weight * visual_sim + motion_weight * motion_sim
        return combined_similarity
    
    def _update_static_analysis(self):
        """Enhanced static analysis with GPS motion"""
        if len(self.velocity_history) < 5:
            return
        
        # Camera motion analysis
        recent_velocities = self.velocity_history[-5:]
        velocity_magnitudes = [np.linalg.norm(v) for v in recent_velocities]
        avg_camera_velocity = np.mean(velocity_magnitudes)
        
        # GPS motion analysis
        gps_motion_stable = True
        if len(self.gps_velocity_history) >= 3:
            recent_gps_speeds = [v[0] for v in self.gps_velocity_history[-3:]]
            gps_speed_variation = np.std(recent_gps_speeds)
            gps_motion_stable = gps_speed_variation < 2.0  # m/s
        
        # Object is static if camera motion is low AND GPS motion is stable
        self.is_static = avg_camera_velocity < 2.0 and gps_motion_stable
    
    def _update_reliability_score(self):
        """Enhanced reliability with visual and motion factors"""
        score = 0.0
        
        # Base factors (40%)
        detection_factor = min(1.0, self.detection_count / 10.0)
        score += 0.2 * detection_factor
        
        if self.confidence_history:
            avg_confidence = np.mean(self.confidence_history)
            score += 0.2 * avg_confidence
        
        # Motion consistency (30%)
        if len(self.velocity_history) > 3:
            velocities = np.array(self.velocity_history[-5:])
            velocity_std = np.std(velocities, axis=0)
            max_std = np.max(velocity_std)
            motion_consistency = 1.0 / (1.0 + max_std / 5.0)
            score += 0.15 * motion_consistency
        
        # Prediction accuracy (15%)
        if self.prediction_accuracy_history:
            avg_prediction_accuracy = np.mean(self.prediction_accuracy_history)
            score += 0.15 * avg_prediction_accuracy
        
        # Visual stability (10%)
        score += 0.1 * self.appearance_stability
        
        # GPS factor (5%)
        if self.gps_positions:
            gps_factor = min(1.0, len(self.gps_positions) / 5.0)
            score += 0.05 * gps_factor
        
        self.reliability_score = min(1.0, score)


class SmartTrackManager:
    """Enhanced Smart Track Manager with Motion Prediction and Visual Features"""
    
    def __init__(self, 
                 config,
                 max_memory_age: int = 300,
                 min_detection_count: int = 3,
                 similarity_threshold: float = 50.0):
        """Initialize enhanced track manager"""
        self.config = config
        self.max_memory_age = max_memory_age
        self.min_detection_count = min_detection_count
        self.similarity_threshold = similarity_threshold
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedSmartTrackManager")
        
        # Track memory storage
        self.track_memories: Dict[int, TrackMemory] = {}
        self.active_track_ids: Set[int] = set()
        self.lost_track_ids: Set[int] = set()
        
        # Motion prediction
        try:
            self.motion_predictor = MotionPredictor(history_length=10)
            self.enhanced_matcher = EnhancedTrackMatcher(self.motion_predictor)
            self.logger.info("Motion predictor enabled")
        except Exception as e:
            self.logger.warning(f"Failed to initialize motion predictor: {e}")
            self.motion_predictor = None
            self.enhanced_matcher = None

        self.feature_extractor = None
        try:
            if not hasattr(self, 'feature_extractor') or self.feature_extractor is None:
                self.feature_extractor = VisualFeatureExtractor(
                    min_bbox_size=20,
                    hist_bins=32
                )
                self.logger.info("Visual feature extractor initialized ONCE")
        except Exception as e:
            self.logger.warning(f"Failed to initialize feature extractor: {e}")
            self.feature_extractor = None
        
        # Current frame data
        self.current_camera_motion = None
        self.current_gps_velocity = None
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.total_tracks_recovered = 0
        self.total_consolidations = 0
        self.total_reappearances = 0
        
        self.logger.info("Enhanced Smart Track Manager initialized")
    
    def process_frame_detections(self, 
                                detections: List[Detection],
                                frame_id: int,
                                timestamp: float = None,
                                frame: Optional[np.ndarray] = None,
                                current_gps: Optional[GPSData] = None) -> List[Detection]:
        """Enhanced frame processing with motion and visual features"""
        
        # Update camera motion
        if self.motion_predictor and frame is not None:
            frame_timestamp = current_gps.timestamp if current_gps else frame_id / 30.0
            self.current_camera_motion = self.motion_predictor.update_camera_motion(frame, frame_timestamp)
        
        # Calculate GPS velocity
        self.current_gps_velocity = self._calculate_gps_velocity(current_gps)
        
        # Update position predictions
        self._update_position_predictions()
        
        # Extract visual features for all detections
        detection_features = {}
        if self.feature_extractor and frame is not None:
            for i, detection in enumerate(detections):
                features = self.feature_extractor.extract_features(frame, detection)
                if features:
                    detection_features[i] = features
        
        # Enhanced matching with motion and visual features
        processed_detections = self._enhanced_detection_matching(
            detections, detection_features, frame_id
        )
        
        # Update track memories
        self._update_track_memories(processed_detections, frame_id, current_gps, frame)
        
        # Handle lost tracks and cleanup
        self._handle_lost_tracks(frame_id)
        self._cleanup_old_memories(frame_id)
        
        return processed_detections
    
    def _calculate_gps_velocity(self, current_gps: Optional[GPSData]) -> Optional[Tuple[float, float]]:
        """Calculate GPS velocity (speed, heading) from GPS history"""
        if not current_gps or not hasattr(self, 'previous_gps'):
            self.previous_gps = current_gps
            return None
        
        if self.previous_gps is None:
            self.previous_gps = current_gps
            return None
        
        # Calculate time difference
        dt = current_gps.timestamp - self.previous_gps.timestamp
        if dt <= 0:
            return None
        
        # Calculate distance (simplified)
        R = 6378137.0  # Earth radius
        lat1, lon1 = np.radians(self.previous_gps.latitude), np.radians(self.previous_gps.longitude)
        lat2, lon2 = np.radians(current_gps.latitude), np.radians(current_gps.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        # Speed in m/s
        speed = distance / dt
        
        # Use GPS heading
        heading = current_gps.heading
        
        self.previous_gps = current_gps
        return (speed, heading)
    
    def _update_position_predictions(self):
        """Update position predictions for active tracks"""
        if not self.motion_predictor:
            return
        
        for track_id in self.active_track_ids:
            if track_id in self.track_memories:
                memory = self.track_memories[track_id]
                mock_track = self._create_mock_track_from_memory(memory)
                
                if mock_track:
                    prediction = self.motion_predictor.predict_track_position(mock_track, steps_ahead=1)
                    memory.predicted_position = prediction.bbox
                    memory.prediction_confidence = prediction.confidence
    
    def _enhanced_detection_matching(self, 
                                   detections: List[Detection], 
                                   detection_features: Dict[int, VisualFeatures],
                                   frame_id: int) -> List[Detection]:
        """Enhanced matching using motion prediction and visual features"""
        
        matched_detections = []
        unmatched_detections = list(enumerate(detections))
        
        # Phase 1: Match with active tracks using predictions and features
        for track_id in self.active_track_ids.copy():
            if track_id not in self.track_memories:
                continue
            
            memory = self.track_memories[track_id]
            best_match = None
            best_score = 0.0
            best_detection_idx = None
            
            for det_idx, detection in unmatched_detections:
                # Calculate motion-based similarity
                motion_distance = float('inf')
                if memory.predicted_position is not None:
                    predicted_center = np.array([
                        (memory.predicted_position[0] + memory.predicted_position[2]) / 2,
                        (memory.predicted_position[1] + memory.predicted_position[3]) / 2
                    ])
                    motion_distance = np.linalg.norm(detection.center - predicted_center)
                
                # Get visual features for this detection
                visual_features = detection_features.get(det_idx)
                
                # Calculate combined similarity
                if self.config.track_consolidation.enable_visual_features or self.config.track_consolidation.enable_motion_prediction:
                    similarity = memory.get_combined_similarity(
                        visual_features, motion_distance, self.config.track_consolidation
                    )
                else:
                    # Fallback to distance-based matching
                    similarity = max(0.0, 1.0 - motion_distance / self.similarity_threshold) if motion_distance != float('inf') else 0.0
                
                # Check thresholds
                motion_ok = motion_distance < self.similarity_threshold
                visual_ok = True
                if visual_features and memory.visual_features and self.config.track_consolidation.enable_visual_features:
                    visual_similarity = self.feature_extractor.compare_features(memory.visual_features, visual_features)
                    visual_ok = visual_similarity > self.config.track_consolidation.feature_similarity_threshold
                
                if motion_ok and visual_ok and similarity > best_score:
                    best_score = similarity
                    best_match = detection
                    best_detection_idx = det_idx
            
            # Assign best match
            if best_match and best_score > 0.5:  # Minimum similarity threshold
                enhanced_detection = Detection(
                    bbox=best_match.bbox,
                    score=best_match.score,
                    class_id=best_match.class_id,
                    frame_id=frame_id
                )
                enhanced_detection.track_id = track_id
                enhanced_detection.prediction_match = True
                enhanced_detection.match_score = best_score
                enhanced_detection.motion_compensated = self.current_camera_motion is not None
                
                matched_detections.append(enhanced_detection)
                unmatched_detections = [(i, d) for i, d in unmatched_detections if i != best_detection_idx]
                
                self.logger.debug(f"Track {track_id}: Matched with score {best_score:.3f}")
        
        # Phase 2: Handle unmatched detections (new tracks or reappearances)
        for det_idx, detection in unmatched_detections:
            # Check for track reappearance
            reappeared_track_id = self._check_track_reappearance(detection, detection_features.get(det_idx))
            
            if reappeared_track_id:
                enhanced_detection = Detection(
                    bbox=detection.bbox,
                    score=detection.score,
                    class_id=detection.class_id,
                    frame_id=frame_id
                )
                enhanced_detection.track_id = reappeared_track_id
                enhanced_detection.reappearance_match = True
                enhanced_detection.motion_compensated = self.current_camera_motion is not None
                
                matched_detections.append(enhanced_detection)
                self.total_reappearances += 1
                
                # Move track back to active
                self.lost_track_ids.discard(reappeared_track_id)
                self.active_track_ids.add(reappeared_track_id)
                
                self.logger.info(f"Track {reappeared_track_id}: Reappeared at frame {frame_id}")
            else:
                # New track
                new_track_id = self._get_next_track_id()
                enhanced_detection = Detection(
                    bbox=detection.bbox,
                    score=detection.score,
                    class_id=detection.class_id,
                    frame_id=frame_id
                )
                enhanced_detection.track_id = new_track_id
                enhanced_detection.new_track = True
                enhanced_detection.motion_compensated = self.current_camera_motion is not None
                
                matched_detections.append(enhanced_detection)
        
        return matched_detections
    
    def _check_track_reappearance(self, detection: Detection, visual_features: Optional[VisualFeatures]) -> Optional[int]:
        """Check if detection matches a recently lost track"""
        if not self.config.track_consolidation.enable_reappearance_detection:
            return None
        
        best_track_id = None
        best_score = 0.0
        
        for track_id in self.lost_track_ids.copy():
            if track_id not in self.track_memories:
                continue
            
            memory = self.track_memories[track_id]
            
            # Check if track was lost recently enough
            frames_since_lost = abs(detection.frame_id - memory.last_seen_frame)
            if frames_since_lost > self.config.track_consolidation.max_gap_frames:
                continue
            
            # Calculate spatial distance
            if memory.last_position is not None:
                spatial_distance = np.linalg.norm(detection.center - memory.last_position)
                if spatial_distance > self.config.track_consolidation.reappearance_spatial_threshold:
                    continue
            
            # Calculate combined similarity for reappearance
            motion_distance = spatial_distance if memory.last_position is not None else float('inf')
            similarity = memory.get_combined_similarity(
                visual_features, motion_distance, self.config.track_consolidation
            )
            
            if similarity > best_score:
                best_score = similarity
                best_track_id = track_id
        
        # Return track ID if similarity is high enough
        reappearance_threshold = 0.6  # Higher threshold for reappearance
        return best_track_id if best_score > reappearance_threshold else None
    
    def _update_track_memories(self, detections: List[Detection], frame_id: int, 
                              current_gps: Optional[GPSData], frame: Optional[np.ndarray]):
        """Update track memories with enhanced features"""
        
        for detection in detections:
            if hasattr(detection, 'track_id') and detection.track_id is not None:
                track_id = detection.track_id
                
                # Calculate GPS position
                gps_pos = None
                if current_gps:
                    gps_pos = self._calculate_gps_position(detection, current_gps)
                
                # Extract visual features
                visual_features = None
                if self.feature_extractor and frame is not None:
                    visual_features = self.feature_extractor.extract_features(frame, detection)
                
                # Create or update memory
                if track_id not in self.track_memories:
                    self.track_memories[track_id] = TrackMemory(
                        track_id=track_id,
                        last_seen_frame=frame_id,
                        last_position=detection.center
                    )
                    self.total_tracks_created += 1
                    self.active_track_ids.add(track_id)
                
                # Update memory with all features
                self.track_memories[track_id].update(
                    detection, frame_id, gps_pos, self.current_gps_velocity, visual_features
                )
    
    def _calculate_gps_position(self, detection: Detection, gps_data: GPSData) -> Optional[Tuple[float, float]]:
        """Calculate GPS position for detection using enhanced depth estimation"""
        try:
            bbox = detection.bbox
            bbox_height = bbox[3] - bbox[1]
            
            if bbox_height <= 0:
                return None
            
            # Enhanced depth estimation considering camera motion
            focal_length = 1400
            object_height = 4.0  # meters
            base_depth = (object_height * focal_length) / bbox_height
            
            # Adjust depth based on camera motion
            if self.current_camera_motion and self.current_camera_motion.translation is not None:
                motion_magnitude = np.linalg.norm(self.current_camera_motion.translation)
                motion_factor = 1.0 + motion_magnitude * 0.001  # Small adjustment
                estimated_depth = base_depth * motion_factor
            else:
                estimated_depth = base_depth
            
            # Enhanced bearing calculation with GPS velocity
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            image_width = 1920
            pixels_from_center = bbox_center_x - (image_width / 2)
            degrees_per_pixel = 60.0 / image_width
            bearing_offset = pixels_from_center * degrees_per_pixel
            
            # Adjust bearing based on GPS heading and camera motion
            base_bearing = gps_data.heading + bearing_offset
            
            if self.current_gps_velocity:
                speed, gps_heading = self.current_gps_velocity
                # Use GPS heading for moving vehicle, camera heading for stationary
                if speed > 1.0:  # Moving
                    object_bearing = gps_heading + bearing_offset
                else:
                    object_bearing = base_bearing
            else:
                object_bearing = base_bearing
            
            # Convert to GPS coordinates
            import math
            lat_offset = (estimated_depth * math.cos(math.radians(object_bearing))) / 111000
            lon_offset = (estimated_depth * math.sin(math.radians(object_bearing))) / (111000 * math.cos(math.radians(gps_data.latitude)))
            
            object_lat = gps_data.latitude + lat_offset
            object_lon = gps_data.longitude + lon_offset
            
            return (object_lat, object_lon)
            
        except Exception as e:
            self.logger.error(f"Error calculating GPS position: {e}")
            return None
    
    def _get_next_track_id(self) -> int:
        """Get next available track ID"""
        if not hasattr(self, '_next_track_id'):
            self._next_track_id = 1
        
        self._next_track_id += 1
        return self._next_track_id
    
    def _create_mock_track_from_memory(self, memory: TrackMemory):
        """Create mock track for motion prediction"""
        from ..core import Track, Detection
        
        if memory.last_position is not None:
            estimated_size = 50
            center = memory.last_position
            
            bbox = np.array([
                center[0] - estimated_size/2,
                center[1] - estimated_size/2,
                center[0] + estimated_size/2,
                center[1] + estimated_size/2
            ])
            
            detection = Detection(
                bbox=bbox,
                score=0.8,
                class_id=0,
                frame_id=memory.last_seen_frame
            )
            
            track = Track(
                track_id=memory.track_id,
                detections=[detection],
                state='confirmed'
            )
            
            return track
        
        return None
    
    def _handle_lost_tracks(self, frame_id: int):
        """Handle tracks that weren't detected this frame"""
        current_active = set()
        
        for track_id, memory in self.track_memories.items():
            if memory.last_seen_frame == frame_id:
                current_active.add(track_id)
        
        lost_tracks = self.active_track_ids - current_active
        for lost_track_id in lost_tracks:
            self.active_track_ids.discard(lost_track_id)
            self.lost_track_ids.add(lost_track_id)
            self.total_tracks_lost += 1
            self.logger.debug(f"Track {lost_track_id} lost at frame {frame_id}")
    
    def _cleanup_old_memories(self, current_frame: int):
        """Cleanup old track memories"""
        to_remove = []
        
        for track_id, memory in self.track_memories.items():
            age = current_frame - memory.last_seen_frame
            if age > self.max_memory_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_memories[track_id]
            self.lost_track_ids.discard(track_id)
            self.active_track_ids.discard(track_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced statistics"""
        exportable_tracks = len([m for m in self.track_memories.values() 
                                if m.detection_count >= self.min_detection_count and m.reliability_score > 0.3])
        
        # Visual feature statistics
        tracks_with_features = len([m for m in self.track_memories.values() if m.visual_features])
        avg_feature_stability = np.mean([m.appearance_stability for m in self.track_memories.values()]) if self.track_memories else 0
        
        # Motion prediction statistics
        tracks_with_predictions = len([m for m in self.track_memories.values() if m.prediction_accuracy_history])
        avg_prediction_accuracy = np.mean([
            np.mean(m.prediction_accuracy_history) for m in self.track_memories.values() 
            if m.prediction_accuracy_history
        ]) if tracks_with_predictions > 0 else 0
        
        return {
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_lost': self.total_tracks_lost,
            'total_tracks_recovered': self.total_tracks_recovered,
            'total_consolidations': self.total_consolidations,
            'total_reappearances': self.total_reappearances,
            'active_tracks': len(self.active_track_ids),
            'lost_tracks': len(self.lost_track_ids),
            'tracks_in_memory': len(self.track_memories),
            'exportable_tracks': exportable_tracks,
            'recovery_rate': self.total_tracks_recovered / max(1, self.total_tracks_lost),
            'visual_features': {
                'tracks_with_features': tracks_with_features,
                'avg_appearance_stability': avg_feature_stability,
                'feature_extraction_enabled': self.feature_extractor is not None
            },
            'motion_prediction': {
                'tracks_with_predictions': tracks_with_predictions,
                'avg_prediction_accuracy': avg_prediction_accuracy,
                'motion_predictor_enabled': self.motion_predictor is not None,
                'camera_motion_detected': self.current_camera_motion is not None
            },
            'gps_enhanced': {
                'gps_velocity_tracking': self.current_gps_velocity is not None,
                'world_motion_compensation': True
            }
        }