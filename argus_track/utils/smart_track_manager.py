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
        
        self.logger.info("Clean Track Manager initialized")
    
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
    
    def _get_clean_track_id(self, original_id: int, detection: Detection, frame_id: int) -> int:
        """Enhanced version that prevents fragmentation and resurrection"""
        
        detection_center = detection.center
        
        # ANTI-FRAGMENTATION: Check if this detection is very close to an existing active track
        for active_id in self.active_track_ids:
            if active_id in self.track_positions:
                distance = np.linalg.norm(detection_center - self.track_positions[active_id])
                if distance < 25.0:  # Within 25 pixels = same object (fragmentation)
                    self.logger.info(f"Frame {frame_id}: Prevented fragmentation {original_id} -> {active_id}")
                    return active_id
        
        # ANTI-RESURRECTION: Check forbidden resurrections
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

    def _handle_lost_tracks(self, current_frame: int):
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