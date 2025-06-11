"""
GPS-synchronized frame processing implementation for Argus Track
FIXED: Only process frames when GPS data is actually available
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..core import GPSData

# Configure logging
logger = logging.getLogger(__name__)


class GPSSynchronizer:
    """
    Synchronizes video frame processing with actual GPS data points
    ONLY processes frames that have real GPS measurements
    """

    def __init__(
        self, gps_data: List[GPSData], video_fps: float, gps_fps: float = 10.0
    ):
        """
        Initialize GPS synchronizer - FIXED VERSION

        Args:
            gps_data: List of actual GPS data points
            video_fps: Video frame rate in FPS
            gps_fps: Expected GPS data rate in Hz (for validation only)
        """
        if not gps_data:
            self.gps_data = []
            self.sync_frames = set()
            self.frame_to_gps = {}
            return

        self.gps_data = sorted(gps_data, key=lambda x: x.timestamp)
        self.video_fps = video_fps
        self.gps_fps = gps_fps

        # Calculate video start time (assume GPS and video start together)
        self.video_start_time = self.gps_data[0].timestamp

        # FIXED: Only create mappings for frames where we have actual GPS data
        self.frame_to_gps: Dict[int, int] = {}  # frame_idx -> gps_data_idx
        self.sync_frames: Set[int] = set()  # Only frames with GPS data

        # Generate the mapping - ONLY for actual GPS points
        self._generate_gps_frame_mapping()

        logger.info(f"GPS Synchronizer initialized:")
        logger.info(f"  📊 GPS points available: {len(self.gps_data)}")
        logger.info(f"  🎬 Video FPS: {video_fps}")
        logger.info(f"  📍 Frames to process: {len(self.sync_frames)}")
        logger.info(
            f"  ⏱️  Processing ratio: {len(self.sync_frames)}/{int(video_fps * self._get_video_duration()):.0f} frames"
        )

    def _get_video_duration(self) -> float:
        """Calculate video duration based on GPS data"""
        if len(self.gps_data) < 2:
            return 0.0
        return self.gps_data[-1].timestamp - self.gps_data[0].timestamp

    def _generate_gps_frame_mapping(self) -> None:
        """Generate mapping ONLY for frames with actual GPS data"""
        if not self.gps_data:
            return

        for gps_idx, gps_point in enumerate(self.gps_data):
            # Calculate which video frame corresponds to this GPS timestamp
            time_offset = gps_point.timestamp - self.video_start_time
            frame_number = int(time_offset * self.video_fps)

            # Only map frames that are valid
            if frame_number >= 0:
                self.frame_to_gps[frame_number] = gps_idx
                self.sync_frames.add(frame_number)

        logger.info(f"GPS-Video Mapping:")
        logger.info(f"  📍 GPS points: {len(self.gps_data)}")
        logger.info(f"  🎬 Mapped frames: {len(self.sync_frames)}")

        if self.sync_frames:
            min_frame = min(self.sync_frames)
            max_frame = max(self.sync_frames)
            logger.info(f"  📊 Frame range: {min_frame} to {max_frame}")

            # Show actual GPS frequency
            frame_intervals = []
            sorted_frames = sorted(self.sync_frames)
            for i in range(1, len(sorted_frames)):
                interval = sorted_frames[i] - sorted_frames[i - 1]
                frame_intervals.append(interval)

            if frame_intervals:
                avg_interval = np.mean(frame_intervals)
                actual_gps_freq = (
                    self.video_fps / avg_interval if avg_interval > 0 else 0
                )
                logger.info(f"  🔄 Actual GPS frequency: {actual_gps_freq:.1f} Hz")
                logger.info(f"  📏 Average frame interval: {avg_interval:.1f} frames")

    def should_process_frame(self, frame_idx: int) -> bool:
        """
        FIXED: Only process frames that have actual GPS data

        Args:
            frame_idx: Frame index

        Returns:
            True ONLY if frame has GPS data, False otherwise
        """
        return frame_idx in self.sync_frames

    def get_gps_for_frame(self, frame_idx: int) -> Optional[GPSData]:
        """
        Get GPS data for a specific frame

        Args:
            frame_idx: Frame index

        Returns:
            GPS data for the frame or None if not available
        """
        gps_idx = self.frame_to_gps.get(frame_idx)
        if gps_idx is not None and gps_idx < len(self.gps_data):
            return self.gps_data[gps_idx]
        return None

    def get_all_sync_frames(self) -> List[int]:
        """
        Get all frames that should be processed (only GPS frames)

        Returns:
            Sorted list of frame indices with GPS data
        """
        return sorted(list(self.sync_frames))

    def get_sync_frames_count(self) -> int:
        """
        Get number of frames to process (only GPS frames)

        Returns:
            Number of frames with GPS data
        """
        return len(self.sync_frames)

    def get_next_sync_frame(self, current_frame: int) -> Optional[int]:
        """
        Get the next frame with GPS data

        Args:
            current_frame: Current frame index

        Returns:
            Next frame index with GPS data or None if no more
        """
        sync_frames = sorted(list(self.sync_frames))
        for frame in sync_frames:
            if frame > current_frame:
                return frame
        return None

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about GPS-synchronized processing

        Returns:
            Dictionary with processing statistics
        """
        if not self.sync_frames:
            return {
                "gps_points": 0,
                "sync_frames": 0,
                "processing_ratio": 0.0,
                "avg_gps_frequency": 0.0,
            }

        sync_frames = sorted(list(self.sync_frames))
        frame_intervals = np.diff(sync_frames) if len(sync_frames) > 1 else [0]

        avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 0
        actual_gps_freq = self.video_fps / avg_interval if avg_interval > 0 else 0

        # Calculate total video frames in the GPS time range
        if len(sync_frames) >= 2:
            frame_span = max(sync_frames) - min(sync_frames)
            processing_ratio = (
                len(sync_frames) / (frame_span + 1) if frame_span > 0 else 1.0
            )
        else:
            processing_ratio = 1.0 if sync_frames else 0.0

        return {
            "gps_points": len(self.gps_data),
            "sync_frames": len(self.sync_frames),
            "processing_ratio": processing_ratio,
            "avg_gps_frequency": actual_gps_freq,
            "frame_range": (
                (min(sync_frames), max(sync_frames)) if sync_frames else (0, 0)
            ),
            "avg_frame_interval": avg_interval,
        }


def create_gps_synchronizer(
    gps_data: List[GPSData], video_fps: float, gps_fps: float = 10.0
) -> GPSSynchronizer:
    """
    Create a GPS synchronizer for frame processing
    FIXED: Only processes frames with actual GPS data

    Args:
        gps_data: List of actual GPS data points
        video_fps: Video frame rate in FPS
        gps_fps: Expected GPS data rate in Hz (for validation)

    Returns:
        GPS synchronizer instance
    """
    return GPSSynchronizer(gps_data, video_fps, gps_fps)
