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
        unmatched_detections = []
        
        # Phase 1: Match detections to active tracks
        for detection in detections:
            original_track_id = detection.track_id
            
            # Check if this track ID should be reused
            clean_track_id = self._get_clean_track_id(original_track_id, detection, frame_id)
            
            # Update detection with clean ID
            detection.track_id = clean_track_id
            processed_detections.append(detection)
            
            # Update track memory
            self._update_track_memory(detection, frame_id)
        
        # Phase 2: Handle lost tracks
        self._handle_lost_tracks(frame_id)
        
        return processed_detections
    
    def _get_clean_track_id(self, original_id: int, detection: Detection, 
                           frame_id: int) -> int:
        """
        Get clean track ID using GPS movement logic
        
        Prevents resurrection of tracks that should be behind the vehicle
        """
        # If this is a forbidden ID (too far behind), assign new ID
        if original_id in self.id_reuse_forbidden:
            new_id = self._assign_new_track_id()
            self.total_resurrections_prevented += 1
            self.logger.debug(f"Frame {frame_id}: Prevented resurrection of ID {original_id} → {new_id}")
            return new_id
        
        # If track is in active memory, continue using it
        if original_id in self.track_memories:
            return original_id
        
        # Check if track was recently lost and could reappear
        if original_id in self.lost_track_ids:
            # Only allow reappearance if vehicle hasn't moved too far
            if self._should_allow_track_reappearance(original_id, detection, frame_id):
                # Move back to active
                self.lost_track_ids.discard(original_id)
                self.active_track_ids.add(original_id)
                self.logger.debug(f"Frame {frame_id}: Track {original_id} reappeared")
                return original_id
            else:
                # Vehicle moved too far, assign new ID
                new_id = self._assign_new_track_id()
                self.total_resurrections_prevented += 1
                self.logger.debug(f"Frame {frame_id}: Blocked distant reappearance {original_id} → {new_id}")
                return new_id
        
        # This is a completely new track
        self.active_track_ids.add(original_id)
        return original_id
    
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
    
    def _update_track_memory(self, detection: Detection, frame_id: int):
        """Update or create track memory"""
        track_id = detection.track_id
        
        if track_id not in self.track_memories:
            # Create new track memory
            self.track_memories[track_id] = TrackMemory(
                track_id=track_id,
                last_seen_frame=frame_id,
                last_position=detection.center
            )
            self.active_track_ids.add(track_id)
        
        # Update existing memory
        self.track_memories[track_id].update(detection, frame_id)
    
    def _handle_lost_tracks(self, current_frame: int):
        """Identify and handle tracks that weren't detected this frame"""
        current_active = set()
        
        # Find tracks that were updated this frame
        for track_id, memory in self.track_memories.items():
            if memory.last_seen_frame == current_frame:
                current_active.add(track_id)
        
        # Find tracks that were lost
        lost_tracks = self.active_track_ids - current_active
        
        for lost_track_id in lost_tracks:
            self.active_track_ids.discard(lost_track_id)
            self.lost_track_ids.add(lost_track_id)
            self.total_tracks_lost += 1
            
            self.logger.debug(f"Track {lost_track_id} lost at frame {current_frame}")
    
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
        """Get comprehensive tracking statistics"""
        return {
            'active_tracks': len(self.active_track_ids),
            'lost_tracks': len(self.lost_track_ids),
            'tracks_in_memory': len(self.track_memories),
            'forbidden_ids': len(self.id_reuse_forbidden),
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_lost': self.total_tracks_lost,
            'resurrections_prevented': self.total_resurrections_prevented,
            'fragmentation_fixes': self.fragmentation_fixes,
            'next_track_id': self.next_track_id,
            'vehicle_speed': self.vehicle_speed,
            'distance_moved': self.total_distance_moved
        }