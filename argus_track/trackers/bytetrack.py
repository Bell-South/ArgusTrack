# argus_track/trackers/bytetrack.py

"""ByteTrack core implementation"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import TrackerConfig
from ..core import Detection, Track
from ..filters import KalmanBoxTracker, batch_predict_kalman
from ..utils import calculate_iou, calculate_iou_matrix


class ByteTrack:
    """
    ByteTrack multi-object tracker optimized for light posts
    
    This implementation uses a two-stage association strategy:
    1. Match high-confidence detections with existing tracks
    2. Match low-confidence detections with unmatched tracks
    
    Optimizations for static objects:
    - Extended track buffer for better persistence
    - Higher IoU thresholds for matching
    - Reduced process noise in Kalman filter
    - Vectorized operations for better performance
    """
    
    def __init__(self, config: TrackerConfig):
        """
        Initialize ByteTrack with configuration
        
        Args:
            config: Tracker configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ByteTrack")
        
        # Track management
        self.tracks: Dict[int, Track] = {}
        self.track_id_counter = 0
        self.frame_id = 0
        
        # Track pools
        self.active_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections for current frame
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        
        # Filter detections by size
        valid_detections = [d for d in detections 
                          if d.area >= self.config.min_box_area]
        
        # Split into high and low confidence
        high_conf_dets = []
        low_conf_dets = []
        
        for det in valid_detections:
            if det.score >= self.config.track_thresh:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)
        
        # Predict current tracks - use vectorized prediction if possible
        if self.active_tracks:
            # Batch prediction for better performance
            kalman_trackers = [track.kalman_filter for track in self.active_tracks]
            batch_predict_kalman(kalman_trackers)
        
        # First association: high confidence detections with active tracks
        matches1, unmatched_tracks1, unmatched_dets1 = self._associate(
            self.active_tracks, 
            high_conf_dets,
            thresh=self.config.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches1:
            track = self.active_tracks[track_idx]
            detection = high_conf_dets[det_idx]
            self._update_track(track, detection)
        
        # Second association: low confidence detections with unmatched tracks
        remaining_tracks = [self.active_tracks[i] for i in unmatched_tracks1]
        matches2, unmatched_tracks2, unmatched_dets2 = self._associate(
            remaining_tracks,
            low_conf_dets,
            thresh=0.5  # Lower threshold for second stage
        )
        
        # Update with low confidence matches
        for track_idx, det_idx in matches2:
            track = remaining_tracks[track_idx]
            detection = low_conf_dets[det_idx]
            self._update_track(track, detection)
        
        # Handle unmatched tracks
        for idx in unmatched_tracks2:
            track = remaining_tracks[idx]
            self._mark_lost(track)
        
        # Create new tracks from unmatched high confidence detections
        for idx in unmatched_dets1:
            detection = high_conf_dets[idx]
            self._create_track(detection)
        
        # Update track lists
        self._update_track_lists()
        
        return self.active_tracks
    
    def _associate(self, tracks: List[Track], detections: List[Detection],
                   thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU
        
        Args:
            tracks: List of tracks
            detections: List of detections
            thresh: IoU threshold for matching
            
        Returns:
            (matches, unmatched_tracks, unmatched_detections)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix - using optimized function
        iou_matrix = calculate_iou_matrix(tracks, detections)
        
        # Apply Hungarian algorithm
        cost_matrix = 1 - iou_matrix  # Convert to cost
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= thresh:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_detections.discard(col)
        
        return matches, list(unmatched_tracks), list(unmatched_detections)
    
    def _update_track(self, track: Track, detection: Detection) -> None:
        """
        Update track with new detection
        
        Args:
            track: Track to update
            detection: New detection
        """
        track.kalman_filter.update(detection)
        track.detections.append(detection)
        track.hits += 1
        track.time_since_update = 0
        
        # Update track state
        if track.state == 'tentative' and track.hits >= 3:
            track.state = 'confirmed'
            self.logger.debug(f"Track {track.track_id} confirmed")
    
    def _create_track(self, detection: Detection) -> Track:
        """
        Create new track from detection
        
        Args:
            detection: Initial detection
            
        Returns:
            New track
        """
        track_id = self.track_id_counter
        self.track_id_counter += 1
        
        track = Track(
            track_id=track_id,
            detections=[detection],
            kalman_filter=KalmanBoxTracker(detection),
            state='tentative',
            hits=1,
            age=1,
            time_since_update=0,
            start_frame=self.frame_id
        )
        
        self.tracks[track_id] = track
        self.active_tracks.append(track)
        
        self.logger.debug(f"Created new track {track_id}")
        return track
    
    def _mark_lost(self, track: Track) -> None:
        """Mark track as lost"""
        track.state = 'lost'
        track.time_since_update += 1
        self.lost_tracks.append(track)
        
    def _update_track_lists(self) -> None:
        """Update track lists based on current states"""
        # Separate active and lost tracks
        new_active = []
        new_lost = []
        
        for track in self.active_tracks:
            if track.state in ['tentative', 'confirmed']:
                new_active.append(track)
            else:
                new_lost.append(track)
        
        # Handle lost tracks
        for track in self.lost_tracks:
            track.time_since_update += 1
            if track.time_since_update > self.config.track_buffer:
                track.state = 'removed'
                self.removed_tracks.append(track)
            else:
                new_lost.append(track)
        
        self.active_tracks = new_active
        self.lost_tracks = new_lost
    
    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all tracks (active, lost, and removed)"""
        return self.tracks.copy()
    
    def reset(self) -> None:
        """Reset tracker to initial state"""
        self.tracks.clear()
        self.track_id_counter = 0
        self.frame_id = 0
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.logger.info("Tracker reset")
    
    def merge_duplicate_tracks(self, distance_threshold: float = 10.0) -> Dict[int, List[int]]:
        """
        Identify and merge duplicate tracks that likely belong to the same object
        
        Args:
            distance_threshold: Maximum center distance to consider duplicates
            
        Returns:
            Dictionary mapping primary track_id to list of duplicate track_ids
        """
        duplicates = {}
        processed = set()
        
        # First, identify duplicates
        for i, track1 in enumerate(self.active_tracks):
            if track1.track_id in processed:
                continue
                
            similar_tracks = []
            
            for j, track2 in enumerate(self.active_tracks[i+1:], i+1):
                if track2.track_id in processed:
                    continue
                
                # Skip if tracks have very different ages
                if abs(track1.age - track2.age) > 30:
                    continue
                
                # Get current positions
                bbox1 = track1.to_tlbr()
                bbox2 = track2.to_tlbr()
                
                # Calculate center points
                center1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
                center2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])
                
                # Calculate distance
                distance = np.linalg.norm(center1 - center2)
                
                # If tracks are close, mark as duplicates
                if distance < distance_threshold:
                    similar_tracks.append(track2.track_id)
                    processed.add(track2.track_id)
            
            if similar_tracks:
                duplicates[track1.track_id] = similar_tracks
        
        # Now merge the duplicates (keep the one with more hits)
        for main_id, duplicate_ids in duplicates.items():
            main_track = self.tracks[main_id]
            
            for dup_id in duplicate_ids:
                dup_track = self.tracks[dup_id]
                
                # Keep the track with more hits
                if dup_track.hits > main_track.hits:
                    # Move the duplicate's detections to the main track
                    main_track.detections.extend(dup_track.detections)
                    main_track.hits += dup_track.hits
                
                # Mark duplicate as removed
                dup_track.state = 'removed'
                
                # Remove from active tracks
                if dup_track in self.active_tracks:
                    self.active_tracks.remove(dup_track)
        
        # Update track lists to reflect changes
        self._update_track_lists()
                
        return duplicates