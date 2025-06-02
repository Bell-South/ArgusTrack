# argus_track/utils/gps_movement_filter.py - NEW FILE

"""
GPS Movement Filter - Skip frames when GPS position doesn't change significantly
"""

import numpy as np
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..core import GPSData

@dataclass
class MovementFilterConfig:
    """Configuration for GPS movement filtering"""
    min_movement_meters: float = 1.0      # Minimum movement to consider as "moving"
    stationary_frame_threshold: int = 10   # Frames before considering stationary
    max_stationary_skip: int = 300         # Maximum frames to skip when stationary
    enable_altitude_check: bool = False    # Whether to include altitude in movement calculation
    movement_smoothing_window: int = 3     # Frames to average for movement calculation


class GPSMovementFilter:
    """
    Filter GPS data to skip frames when camera is stationary
    
    This helps focus processing on frames where the camera is moving,
    which provides better triangulation angles for static object detection.
    """
    
    def __init__(self, config: MovementFilterConfig):
        """
        Initialize GPS movement filter
        
        Args:
            config: Movement filter configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GPSMovementFilter")
        
        # State tracking
        self.gps_history: List[GPSData] = []
        self.movement_history: List[float] = []
        self.stationary_frame_count = 0
        self.total_frames_processed = 0
        self.total_frames_skipped = 0
        
        # Statistics
        self.movement_stats = {
            'total_distance': 0.0,
            'max_movement': 0.0,
            'avg_movement': 0.0,
            'stationary_periods': 0,
            'moving_periods': 0
        }
        
        self.logger.info(f"GPS Movement Filter initialized:")
        self.logger.info(f"  Min movement: {config.min_movement_meters}m")
        self.logger.info(f"  Stationary threshold: {config.stationary_frame_threshold} frames")
        self.logger.info(f"  Max skip frames: {config.max_stationary_skip}")
    
    def should_process_frame(self, gps_data: GPSData, frame_id: int) -> bool:
        """
        Determine if frame should be processed based on GPS movement
        
        Args:
            gps_data: Current GPS data
            frame_id: Current frame number
            
        Returns:
            True if frame should be processed, False if it should be skipped
        """
        # Always process first few frames
        if len(self.gps_history) < 2:
            self._add_gps_data(gps_data)
            self.total_frames_processed += 1
            return True
        
        # Calculate movement from recent position
        movement_distance = self._calculate_movement_distance(gps_data)
        self.movement_history.append(movement_distance)
        
        # Keep movement history manageable
        if len(self.movement_history) > 50:
            self.movement_history = self.movement_history[-50:]
        
        # Update statistics
        self.movement_stats['total_distance'] += movement_distance
        self.movement_stats['max_movement'] = max(self.movement_stats['max_movement'], movement_distance)
        
        # Calculate smoothed movement (average over recent frames)
        recent_movements = self.movement_history[-self.config.movement_smoothing_window:]
        smoothed_movement = np.mean(recent_movements)
        
        # Determine if camera is moving significantly
        is_moving = smoothed_movement >= self.config.min_movement_meters
        
        if is_moving:
            # Camera is moving - always process these frames
            if self.stationary_frame_count > 0:
                self.logger.debug(f"Frame {frame_id}: Movement detected ({smoothed_movement:.2f}m) - "
                               f"ending stationary period of {self.stationary_frame_count} frames")
                self.movement_stats['stationary_periods'] += 1
                self.stationary_frame_count = 0
            
            self.movement_stats['moving_periods'] += 1
            self._add_gps_data(gps_data)
            self.total_frames_processed += 1
            return True
        
        else:
            # Camera is stationary or moving very little
            self.stationary_frame_count += 1
            
            if self.stationary_frame_count <= self.config.stationary_frame_threshold:
                # Still within threshold - continue processing
                self.logger.debug(f"Frame {frame_id}: Low movement ({smoothed_movement:.2f}m) - "
                               f"stationary count: {self.stationary_frame_count}/{self.config.stationary_frame_threshold}")
                self._add_gps_data(gps_data)
                self.total_frames_processed += 1
                return True
            
            elif self.stationary_frame_count <= self.config.max_stationary_skip:
                # Skip this frame - camera has been stationary too long
                if self.stationary_frame_count == self.config.stationary_frame_threshold + 1:
                    self.logger.info(f"Frame {frame_id}: Camera stationary - skipping frames until movement detected")
                
                self.total_frames_skipped += 1
                return False
            
            else:
                # Force process a frame occasionally even when stationary
                if self.stationary_frame_count % self.config.max_stationary_skip == 0:
                    self.logger.debug(f"Frame {frame_id}: Forced processing after {self.stationary_frame_count} stationary frames")
                    self._add_gps_data(gps_data)
                    self.total_frames_processed += 1
                    return True
                else:
                    self.total_frames_skipped += 1
                    return False
    
    def _calculate_movement_distance(self, current_gps: GPSData) -> float:
        """Calculate movement distance from previous GPS position"""
        if not self.gps_history:
            return 0.0
        
        # Use most recent GPS position for comparison
        previous_gps = self.gps_history[-1]
        
        # Calculate distance using Haversine formula
        distance = self._haversine_distance(
            previous_gps.latitude, previous_gps.longitude,
            current_gps.latitude, current_gps.longitude
        )
        
        # Include altitude if enabled
        if self.config.enable_altitude_check:
            altitude_diff = abs(current_gps.altitude - previous_gps.altitude)
            distance = np.sqrt(distance**2 + altitude_diff**2)
        
        return distance
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        # Earth's radius in meters
        R = 6378137.0
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        # Haversine formula
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _add_gps_data(self, gps_data: GPSData):
        """Add GPS data to history"""
        self.gps_history.append(gps_data)
        
        # Keep history manageable
        if len(self.gps_history) > 100:
            self.gps_history = self.gps_history[-100:]
    
    def get_movement_statistics(self) -> dict:
        """Get movement filtering statistics"""
        total_frames = self.total_frames_processed + self.total_frames_skipped
        
        if self.movement_history:
            self.movement_stats['avg_movement'] = np.mean(self.movement_history)
        
        return {
            'total_frames': total_frames,
            'processed_frames': self.total_frames_processed,
            'skipped_frames': self.total_frames_skipped,
            'skip_ratio': self.total_frames_skipped / total_frames if total_frames > 0 else 0,
            'movement_stats': self.movement_stats,
            'current_stationary_count': self.stationary_frame_count,
            'efficiency_gain': f"{self.total_frames_skipped / total_frames * 100:.1f}%" if total_frames > 0 else "0%"
        }
    
    def reset(self):
        """Reset filter state"""
        self.gps_history.clear()
        self.movement_history.clear()
        self.stationary_frame_count = 0
        self.total_frames_processed = 0
        self.total_frames_skipped = 0
        self.movement_stats = {
            'total_distance': 0.0,
            'max_movement': 0.0,
            'avg_movement': 0.0,
            'stationary_periods': 0,
            'moving_periods': 0
        }


def create_movement_filter(min_movement_meters: float = 1.0,
                          stationary_threshold: int = 10,
                          max_skip_frames: int = 300) -> GPSMovementFilter:
    """
    Create a GPS movement filter with specified parameters
    
    Args:
        min_movement_meters: Minimum movement to consider as moving (default: 1.0m)
        stationary_threshold: Frames before considering stationary (default: 10)
        max_skip_frames: Maximum frames to skip when stationary (default: 300)
        
    Returns:
        Configured GPS movement filter
    """
    config = MovementFilterConfig(
        min_movement_meters=min_movement_meters,
        stationary_frame_threshold=stationary_threshold,
        max_stationary_skip=max_skip_frames
    )
    
    return GPSMovementFilter(config)
