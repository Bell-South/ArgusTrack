# argus_track/utils/static_car_detector.py - NEW FILE

"""
Static Car Detector - Skip frames when GPS position doesn't change
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core import GPSData


@dataclass
class StaticCarConfig:
    """Configuration for static car detection"""

    movement_threshold_meters: float = 2.0  # Minimum movement to consider as "moving"
    stationary_time_threshold: float = 10.0  # Seconds before considering stationary
    gps_frame_interval: int = 6  # Normal GPS frame processing interval


class StaticCarDetector:
    """
    Detects when car is stationary and skips frames for processing efficiency

    Logic:
    1. Car stops moving (< 2m movement for 10+ seconds) → Skip all frames
    2. Car starts moving again → Resume normal frame processing
    3. Purpose: Speed up processing, we already captured objects when we first stopped
    """

    def __init__(self, config: StaticCarConfig):
        """
        Initialize static car detector

        Args:
            config: Static car detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.StaticCarDetector")

        # State tracking
        self.gps_history: List[GPSData] = []
        self.last_movement_time: float = 0.0
        self.is_currently_stationary: bool = False
        self.total_frames_processed: int = 0
        self.total_frames_skipped: int = 0

        # Statistics
        self.stationary_periods: List[float] = []  # Duration of each stationary period
        self.current_stationary_start: Optional[float] = None

        self.logger.info(f"Static Car Detector initialized:")
        self.logger.info(f"  Movement threshold: {config.movement_threshold_meters}m")
        self.logger.info(f"  Stationary threshold: {config.stationary_time_threshold}s")

    def should_process_frame(self, gps_data: GPSData, frame_id: int) -> bool:
        """
        Determine if frame should be processed based on car movement

        Args:
            gps_data: Current GPS data
            frame_id: Current frame number

        Returns:
            True if frame should be processed, False if it should be skipped
        """
        # Always process first frame
        if len(self.gps_history) == 0:
            self._add_gps_data(gps_data)
            self.total_frames_processed += 1
            self.logger.debug(f"Frame {frame_id}: First frame - processing")
            return True

        # Check if car has moved significantly
        has_moved = self._has_moved_enough(gps_data)
        current_time = gps_data.timestamp

        if has_moved:
            # Car is moving
            self._handle_movement_detected(current_time, frame_id)
            self._add_gps_data(gps_data)
            self.total_frames_processed += 1
            return True
        else:
            # Car hasn't moved much - check if we should skip
            if self._should_skip_stationary_frame(current_time, frame_id):
                self.total_frames_skipped += 1
                return False
            else:
                # Still in grace period - keep processing
                self._add_gps_data(gps_data)
                self.total_frames_processed += 1
                return True

    def _has_moved_enough(self, current_gps: GPSData) -> bool:
        """Check if car has moved beyond the threshold"""
        if not self.gps_history:
            return True

        # Calculate distance from most recent position
        last_gps = self.gps_history[-1]
        distance = self._calculate_distance(last_gps, current_gps)

        moved_enough = distance >= self.config.movement_threshold_meters

        if moved_enough:
            self.logger.debug(f"Movement detected: {distance:.1f}m")

        return moved_enough

    def _should_skip_stationary_frame(self, current_time: float, frame_id: int) -> bool:
        """Determine if we should skip this frame due to stationary car"""
        if self.last_movement_time == 0:
            self.last_movement_time = current_time
            return False

        time_stationary = current_time - self.last_movement_time

        if time_stationary >= self.config.stationary_time_threshold:
            # Car has been stationary long enough - start skipping frames
            if not self.is_currently_stationary:
                self.logger.info(
                    f"Frame {frame_id}: Car stationary for {time_stationary:.1f}s - "
                    f"starting to skip frames for efficiency"
                )
                self.is_currently_stationary = True
                self.current_stationary_start = current_time

            return True  # Skip this frame

        return False  # Still in grace period

    def _handle_movement_detected(self, current_time: float, frame_id: int):
        """Handle when movement is detected after being stationary"""
        if self.is_currently_stationary:
            # End of stationary period
            if self.current_stationary_start:
                stationary_duration = current_time - self.current_stationary_start
                self.stationary_periods.append(stationary_duration)

                self.logger.info(
                    f"Frame {frame_id}: Movement resumed after "
                    f"{stationary_duration:.1f}s stationary period"
                )

            self.is_currently_stationary = False
            self.current_stationary_start = None

        # Update last movement time
        self.last_movement_time = current_time

    def _calculate_distance(self, gps1: GPSData, gps2: GPSData) -> float:
        """
        Calculate distance between two GPS points using Haversine formula

        Returns:
            Distance in meters
        """
        # Earth's radius in meters
        R = 6378137.0

        # Convert to radians
        lat1_rad = np.radians(gps1.latitude)
        lat2_rad = np.radians(gps2.latitude)
        dlat = np.radians(gps2.latitude - gps1.latitude)
        dlon = np.radians(gps2.longitude - gps1.longitude)

        # Haversine formula
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def _add_gps_data(self, gps_data: GPSData):
        """Add GPS data to history"""
        self.gps_history.append(gps_data)

        # Keep history manageable (last 50 points)
        if len(self.gps_history) > 50:
            self.gps_history = self.gps_history[-50:]

    def get_statistics(self) -> dict:
        """Get static car detection statistics"""
        total_frames = self.total_frames_processed + self.total_frames_skipped

        return {
            "total_frames": total_frames,
            "processed_frames": self.total_frames_processed,
            "skipped_frames": self.total_frames_skipped,
            "skip_ratio": (
                self.total_frames_skipped / total_frames if total_frames > 0 else 0
            ),
            "stationary_periods_count": len(self.stationary_periods),
            "total_stationary_time": sum(self.stationary_periods),
            "avg_stationary_duration": (
                np.mean(self.stationary_periods) if self.stationary_periods else 0
            ),
            "max_stationary_duration": (
                max(self.stationary_periods) if self.stationary_periods else 0
            ),
            "currently_stationary": self.is_currently_stationary,
            "efficiency_gain": (
                f"{self.total_frames_skipped / total_frames * 100:.1f}%"
                if total_frames > 0
                else "0%"
            ),
        }

    def reset(self):
        """Reset detector state"""
        self.gps_history.clear()
        self.last_movement_time = 0.0
        self.is_currently_stationary = False
        self.total_frames_processed = 0
        self.total_frames_skipped = 0
        self.stationary_periods.clear()
        self.current_stationary_start = None


def create_static_car_detector(
    movement_threshold_m: float = 2.0,
    stationary_time_s: float = 10.0,
    gps_frame_interval: int = 6,
) -> StaticCarDetector:
    """
    Create a static car detector with specified parameters

    Args:
        movement_threshold_m: Minimum movement in meters to consider as moving
        stationary_time_s: Time in seconds before starting to skip frames
        gps_frame_interval: Normal GPS frame processing interval

    Returns:
        Configured static car detector
    """
    config = StaticCarConfig(
        movement_threshold_meters=movement_threshold_m,
        stationary_time_threshold=stationary_time_s,
        gps_frame_interval=gps_frame_interval,
    )

    return StaticCarDetector(config)
