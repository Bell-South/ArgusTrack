# argus_track/trackers/bytetrack.py (UPDATED WITH TRACK MERGING)
"""ByteTrack implementation with enhanced track merging for static objects"""

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
    Enhanced ByteTrack multi-object tracker optimized for static LED detection
    
    Key improvements:
    - Automatic duplicate track merging
    - Extended track lifetimes for static objects
    - Better handling of occlusions
    - More conservative track confirmation
    """
    
    def __init__(self, config: TrackerConfig):
        """Initialize ByteTrack with enhanced configuration"""
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
        
        # Track merging history to avoid re-merging
        self.merged_tracks: Dict[int, int] = {}  # merged_id -> main_id
        
        self.logger.info(f"Initialized ByteTrack with enhanced tracking parameters")
        self.logger.info(f"Track confirmation requires {config.min_hits} hits")
        self.logger.info(f"Track merging enabled: {config.enable_track_merging}")
        
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections and perform track merging"""
        self.frame_id += 1
        
        # Filter detections by size and confidence
        valid_detections = [d for d in detections 
                          if (d.area >= self.config.min_box_area and 
                              d.score >= self.config.track_thresh)]
        
        if len(valid_detections) == 0:
            self.logger.debug(f"Frame {self.frame_id}: No valid detections")
            return self._handle_no_detections()
        
        self.logger.debug(f"Frame {self.frame_id}: {len(valid_detections)} valid detections")
        
        # Split into high and low confidence
        high_conf_dets = []
        low_conf_dets = []
        
        for det in valid_detections:
            if det.score >= self.config.track_thresh * 1.5:  # Higher threshold for high confidence
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)
        
        # Predict current tracks
        if self.active_tracks:
            kalman_trackers = [track.kalman_filter for track in self.active_tracks if track.kalman_filter]
            if kalman_trackers:
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
            thresh=self.config.match_thresh * 0.8  # More lenient for second stage
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
        new_tracks = []
        for idx in unmatched_dets1:
            detection = high_conf_dets[idx]
            new_track = self._create_track(detection)
            new_tracks.append(new_track)
        
        # Update track lists
        self._update_track_lists()
        
        # Perform track merging if enabled
        if self.config.enable_track_merging and self.frame_id % 30 == 0:  # Every 30 frames
            self._merge_duplicate_tracks()
        
        return self.active_tracks
    
    def _handle_no_detections(self) -> List[Track]:
        """Handle frames with no valid detections"""
        # Mark all active tracks as lost
        for track in self.active_tracks:
            self._mark_lost(track)
        
        self._update_track_lists()
        return self.active_tracks
    
    def _associate(self, tracks: List[Track], detections: List[Detection],
                   thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks with detections using IoU with distance penalty"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix
        iou_matrix = calculate_iou_matrix(tracks, detections)
        
        # Add distance penalty for better static object tracking
        distance_matrix = self._calculate_distance_matrix(tracks, detections)
        
        # Combine IoU and distance (normalize distance and subtract from IoU)
        max_distance = np.max(distance_matrix) if np.max(distance_matrix) > 0 else 1.0
        normalized_distance = distance_matrix / max_distance
        combined_matrix = iou_matrix - 0.1 * normalized_distance  # Small distance penalty
        
        # Apply Hungarian algorithm
        cost_matrix = 1 - combined_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if combined_matrix[row, col] >= thresh:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_detections.discard(col)
        
        return matches, list(unmatched_tracks), list(unmatched_detections)
    
    def _calculate_distance_matrix(self, tracks: List[Track], 
                                  detections: List[Detection]) -> np.ndarray:
        """Calculate center-to-center distance matrix"""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))
        
        track_centers = []
        for track in tracks:
            bbox = track.to_tlbr()
            center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            track_centers.append(center)
        
        det_centers = [det.center for det in detections]
        
        distance_matrix = np.zeros((len(tracks), len(detections)))
        for i, track_center in enumerate(track_centers):
            for j, det_center in enumerate(det_centers):
                distance_matrix[i, j] = np.linalg.norm(track_center - det_center)
        
        return distance_matrix
    
    def _update_track(self, track: Track, detection: Detection) -> None:
        """Update track with new detection"""
        track.kalman_filter.update(detection)
        track.detections.append(detection)
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        
        # Update track state with stricter confirmation requirements
        if (track.state == 'tentative' and 
            track.hits >= self.config.min_hits and 
            self._is_track_stable(track)):
            track.state = 'confirmed'
            self.logger.debug(f"Track {track.track_id} confirmed (hits: {track.hits})")
    
    def _is_track_stable(self, track: Track) -> bool:
        """Check if track has stable detections (for static objects)"""
        if len(track.detections) < 3:
            return False
        
        # Check position stability
        recent_centers = [det.center for det in track.detections[-5:]]
        if len(recent_centers) < 3:
            return True  # Not enough data, assume stable
        
        centers_array = np.array(recent_centers)
        std_dev = np.std(centers_array, axis=0)
        max_std = np.max(std_dev)
        
        return max_std < self.config.static_threshold
    
    def _create_track(self, detection: Detection) -> Track:
        """Create new track from detection"""
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
        if track.state == 'confirmed':
            track.state = 'lost'
        track.time_since_update += 1
        track.age += 1
        
        # Remove from active tracks
        if track in self.active_tracks:
            self.active_tracks.remove(track)
        
        # Add to lost tracks if not already there
        if track not in self.lost_tracks:
            self.lost_tracks.append(track)
    
    def _update_track_lists(self) -> None:
        """Update track lists based on current states"""
        # Handle lost tracks
        tracks_to_remove = []
        for track in self.lost_tracks:
            track.time_since_update += 1
            track.age += 1
            
            if track.time_since_update > self.config.max_time_lost:
                track.state = 'removed'
                tracks_to_remove.append(track)
                self.removed_tracks.append(track)
        
        # Remove from lost tracks
        for track in tracks_to_remove:
            self.lost_tracks.remove(track)
        
        # Handle very old tracks
        old_tracks = []
        for track in self.active_tracks:
            track.age += 1
            if track.age > self.config.max_track_age:
                old_tracks.append(track)
        
        for track in old_tracks:
            self._mark_lost(track)
    
    def _merge_duplicate_tracks(self) -> int:
        """Merge duplicate tracks that likely represent the same object"""
        if not self.config.enable_track_merging:
            return 0
        
        merged_count = 0
        confirmed_tracks = [t for t in self.active_tracks if t.state == 'confirmed']
        
        if len(confirmed_tracks) < 2:
            return 0
        
        # Find pairs of tracks to merge
        tracks_to_merge = []
        processed_tracks = set()
        
        for i, track1 in enumerate(confirmed_tracks):
            if track1.track_id in processed_tracks:
                continue
                
            for j, track2 in enumerate(confirmed_tracks[i+1:], i+1):
                if track2.track_id in processed_tracks:
                    continue
                
                if self._should_merge_tracks(track1, track2):
                    tracks_to_merge.append((track1, track2))
                    processed_tracks.add(track2.track_id)
                    break
        
        # Perform merging
        for track1, track2 in tracks_to_merge:
            if self._merge_tracks(track1, track2):
                merged_count += 1
                self.logger.info(f"Merged track {track2.track_id} into track {track1.track_id}")
        
        if merged_count > 0:
            self.logger.info(f"Merged {merged_count} duplicate tracks")
        
        return merged_count
    
    def _should_merge_tracks(self, track1: Track, track2: Track) -> bool:
        """Determine if two tracks should be merged"""
        # Both must be confirmed
        if track1.state != 'confirmed' or track2.state != 'confirmed':
            return False
        
        # Check if already merged
        if (track1.track_id in self.merged_tracks or 
            track2.track_id in self.merged_tracks):
            return False
        
        # Get current positions
        bbox1 = track1.to_tlbr()
        bbox2 = track2.to_tlbr()
        
        # Calculate IoU
        iou = calculate_iou(bbox1, bbox2)
        if iou < self.config.merge_iou_threshold:
            return False
        
        # Calculate center distance
        center1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
        center2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])
        distance = np.linalg.norm(center1 - center2)
        
        if distance > self.config.merge_distance_threshold:
            return False
        
        # Check if both tracks are stable (static)
        if not (self._is_track_stable(track1) and self._is_track_stable(track2)):
            return False
        
        return True
    
    def _merge_tracks(self, main_track: Track, merge_track: Track) -> bool:
        """Merge merge_track into main_track"""
        try:
            # Keep the track with more hits as main
            if merge_track.hits > main_track.hits:
                main_track, merge_track = merge_track, main_track
            
            # Merge detections (keep recent ones from both)
            all_detections = main_track.detections + merge_track.detections
            # Sort by frame and keep unique frames
            frame_dict = {}
            for det in all_detections:
                if det.frame_id not in frame_dict or det.score > frame_dict[det.frame_id].score:
                    frame_dict[det.frame_id] = det
            
            main_track.detections = sorted(frame_dict.values(), key=lambda x: x.frame_id)
            
            # Update track statistics
            main_track.hits += merge_track.hits
            main_track.age = max(main_track.age, merge_track.age)
            
            # Remove merged track
            merge_track.state = 'removed'
            if merge_track in self.active_tracks:
                self.active_tracks.remove(merge_track)
            
            # Record the merge
            self.merged_tracks[merge_track.track_id] = main_track.track_id
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging tracks: {e}")
            return False
    
    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all tracks (active, lost, and removed)"""
        return self.tracks.copy()
    
    def get_tracking_stats(self) -> Dict[str, int]:
        """Get current tracking statistics"""
        return {
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks),
            'removed_tracks': len(self.removed_tracks),
            'total_tracks': len(self.tracks),
            'merged_tracks': len(self.merged_tracks),
            'frame_id': self.frame_id
        }
    
    def reset(self) -> None:
        """Reset tracker to initial state"""
        self.tracks.clear()
        self.track_id_counter = 0
        self.frame_id = 0
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.merged_tracks.clear()
        self.logger.info("Tracker reset")