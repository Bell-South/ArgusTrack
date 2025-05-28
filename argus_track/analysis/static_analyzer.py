"""Enhanced static object analysis"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN

from ..core import Track, Detection


class StaticObjectAnalyzer:
    """Advanced analysis of static objects in tracking data"""
    
    def __init__(self, 
                 position_threshold: float = 2.0,
                 velocity_threshold: float = 0.5,
                 min_observations: int = 10,
                 stability_window: int = 30):
        """
        Initialize static object analyzer
        
        Args:
            position_threshold: Maximum position variance for static classification
            velocity_threshold: Maximum velocity for static classification
            min_observations: Minimum observations required
            stability_window: Window size for stability analysis
        """
        self.position_threshold = position_threshold
        self.velocity_threshold = velocity_threshold
        self.min_observations = min_observations
        self.stability_window = stability_window
    
    def analyze_track(self, track: Track) -> Dict[str, float]:
        """
        Analyze a single track for static behavior
        
        Args:
            track: Track to analyze
            
        Returns:
            Dictionary with analysis metrics
        """
        if len(track.detections) < self.min_observations:
            return {
                'is_static': False,
                'confidence': 0.0,
                'position_variance': float('inf'),
                'velocity_mean': float('inf'),
                'stability_score': 0.0
            }
        
        # Extract positions
        positions = np.array([det.center for det in track.detections])
        
        # Calculate position variance
        position_variance = np.std(positions, axis=0).mean()
        
        # Calculate velocities
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0)
            velocity_mean = np.abs(velocities).mean()
        else:
            velocity_mean = 0.0
        
        # Calculate stability score using sliding window
        stability_scores = []
        for i in range(len(positions) - self.stability_window + 1):
            window = positions[i:i + self.stability_window]
            window_variance = np.std(window, axis=0).mean()
            stability_scores.append(1.0 / (1.0 + window_variance))
        
        stability_score = np.mean(stability_scores) if stability_scores else 0.0
        
        # Determine if static
        is_static = (
            position_variance < self.position_threshold and
            velocity_mean < self.velocity_threshold
        )
        
        # Calculate confidence
        confidence = min(1.0, stability_score * (1.0 - velocity_mean / self.velocity_threshold))
        
        return {
            'is_static': is_static,
            'confidence': confidence,
            'position_variance': position_variance,
            'velocity_mean': velocity_mean,
            'stability_score': stability_score
        }
    
    def find_clusters(self, tracks: Dict[int, Track], 
                     eps: float = 50.0, 
                     min_samples: int = 2) -> Dict[int, int]:
        """
        Find clusters of static objects
        
        Args:
            tracks: Dictionary of tracks
            eps: Maximum distance between points in a cluster
            min_samples: Minimum samples in a cluster
            
        Returns:
            Dictionary mapping track_id to cluster_id
        """
        # Filter static tracks
        static_tracks = {}
        positions = []
        track_ids = []
        
        for track_id, track in tracks.items():
            analysis = self.analyze_track(track)
            if analysis['is_static']:
                static_tracks[track_id] = track
                positions.append(track.trajectory[-1])  # Use last position
                track_ids.append(track_id)
        
        if len(positions) < min_samples:
            return {}
        
        # Perform clustering
        positions = np.array(positions)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        
        # Map track IDs to clusters
        cluster_mapping = {}
        for track_id, cluster_id in zip(track_ids, clustering.labels_):
            if cluster_id != -1:  # Ignore noise points
                cluster_mapping[track_id] = cluster_id
        
        return cluster_mapping
    
    def merge_duplicate_tracks(self, tracks: Dict[int, Track],
                              distance_threshold: float = 30.0) -> Dict[int, List[int]]:
        """
        Identify duplicate tracks of the same static object
        
        Args:
            tracks: Dictionary of tracks
            distance_threshold: Maximum distance to consider duplicates
            
        Returns:
            Dictionary mapping primary track_id to list of duplicate track_ids
        """
        static_tracks = []
        for track_id, track in tracks.items():
            analysis = self.analyze_track(track)
            if analysis['is_static']:
                static_tracks.append((track_id, track))
        
        duplicates = {}
        processed = set()
        
        for i, (track_id1, track1) in enumerate(static_tracks):
            if track_id1 in processed:
                continue
                
            duplicates[track_id1] = []
            
            for j, (track_id2, track2) in enumerate(static_tracks[i+1:], i+1):
                if track_id2 in processed:
                    continue
                
                # Calculate distance between average positions
                pos1 = np.mean([det.center for det in track1.detections], axis=0)
                pos2 = np.mean([det.center for det in track2.detections], axis=0)
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < distance_threshold:
                    duplicates[track_id1].append(track_id2)
                    processed.add(track_id2)
        
        # Remove entries with no duplicates
        return {k: v for k, v in duplicates.items() if v}
    
    def calculate_persistence_score(self, track: Track, 
                                   total_frames: int) -> float:
        """
        Calculate persistence score for a track
        
        Args:
            track: Track to analyze
            total_frames: Total frames in video
            
        Returns:
            Persistence score between 0 and 1
        """
        if total_frames == 0:
            return 0.0
        
        # Calculate presence ratio
        presence_ratio = track.age / total_frames
        
        # Calculate detection density
        if track.age > 0:
            detection_density = len(track.detections) / track.age
        else:
            detection_density = 0.0
        
        # Combine metrics
        persistence_score = presence_ratio * detection_density
        
        return min(1.0, persistence_score)