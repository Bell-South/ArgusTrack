# argus_track/utils/gps_sync_tracker.py

"""
GPS-synchronized frame processing implementation for Argus Track
FIXED: Frame naming convention to use proper 6-frame intervals (0, 6, 12, 18...)
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
    FIXED: Proper 6-frame interval naming convention
    """

    def __init__(
        self, gps_data: List[GPSData], video_fps: float, gps_fps: float = 10.0, frame_interval: int = 6
    ):
        """
        Initialize GPS synchronizer - FIXED VERSION with proper frame intervals

        Args:
            gps_data: List of actual GPS data points
            video_fps: Video frame rate in FPS
            gps_fps: Expected GPS data rate in Hz (for validation only)
            frame_interval: Process every N frames (default: 6 for 60fps->10fps)
        """
        if not gps_data:
            self.gps_data = []
            self.sync_frames = set()
            self.frame_to_gps = {}
            self.frame_interval = frame_interval
            return

        self.gps_data = sorted(gps_data, key=lambda x: x.timestamp)
        self.video_fps = video_fps
        self.gps_fps = gps_fps
        self.frame_interval = frame_interval  # NEW: Store frame interval

        # Calculate video start time (assume GPS and video start together)
        self.video_start_time = self.gps_data[0].timestamp

        # FIXED: Only create mappings for frames at proper intervals with GPS data
        self.frame_to_gps: Dict[int, int] = {}  # frame_idx -> gps_data_idx
        self.sync_frames: Set[int] = set()  # Only frames with GPS data at proper intervals

        # Generate the mapping - FIXED for proper 6-frame intervals
        self._generate_gps_frame_mapping()

        logger.info(f"GPS Synchronizer initialized:")
        logger.info(f"  📊 GPS points available: {len(self.gps_data)}")
        logger.info(f"  🎬 Video FPS: {video_fps}")
        logger.info(f"  📅 Frame interval: {frame_interval} (every {frame_interval} frames)")
        logger.info(f"  📍 Frames to process: {len(self.sync_frames)}")
        logger.info(
            f"  ⏱️  Processing ratio: {len(self.sync_frames)}/{int(video_fps * self._get_video_duration() / frame_interval):.0f} expected frames"
        )

    def _get_video_duration(self) -> float:
        """Calculate video duration based on GPS data"""
        if len(self.gps_data) < 2:
            return 0.0
        return self.gps_data[-1].timestamp - self.gps_data[0].timestamp

    def _generate_gps_frame_mapping(self) -> None:
        """
        FIXED: Generate mapping for frames at proper intervals with actual GPS data
        Frame naming: 0, 6, 12, 18, 24... (every frame_interval frames)
        """
        if not self.gps_data:
            return

        for gps_idx, gps_point in enumerate(self.gps_data):
            # Calculate which video frame corresponds to this GPS timestamp
            time_offset = gps_point.timestamp - self.video_start_time
            actual_frame_number = int(time_offset * self.video_fps)

            # FIXED: Map to the nearest frame that follows our interval pattern
            # Find the closest frame that is a multiple of frame_interval
            target_frame = round(actual_frame_number / self.frame_interval) * self.frame_interval

            # Only map frames that are valid and follow our naming convention
            if target_frame >= 0:
                self.frame_to_gps[target_frame] = gps_idx
                self.sync_frames.add(target_frame)

        logger.info(f"GPS-Video Mapping (Fixed Frame Intervals):")
        logger.info(f"  📍 GPS points: {len(self.gps_data)}")
        logger.info(f"  🎬 Mapped frames: {len(self.sync_frames)}")

        if self.sync_frames:
            min_frame = min(self.sync_frames)
            max_frame = max(self.sync_frames)
            logger.info(f"  📊 Frame range: {min_frame} to {max_frame} (step: {self.frame_interval})")

            # Verify proper frame interval naming
            sorted_frames = sorted(self.sync_frames)
            proper_intervals = all(
                (sorted_frames[i] - sorted_frames[i-1]) % self.frame_interval == 0 
                for i in range(1, min(5, len(sorted_frames)))  # Check first few intervals
            )
            
            if proper_intervals:
                logger.info(f"  ✅ Frame naming follows proper {self.frame_interval}-frame intervals")
            else:
                logger.warning(f"  ⚠️  Frame naming may not follow perfect {self.frame_interval}-frame intervals")

            # Show actual frame sequence (first 10 for verification)
            sample_frames = sorted_frames[:10]
            logger.info(f"  📋 Sample frames: {sample_frames}")

            # Calculate actual effective GPS frequency
            if len(sorted_frames) > 1:
                avg_frame_interval = np.mean(np.diff(sorted_frames))
                actual_gps_freq = self.video_fps / avg_frame_interval if avg_frame_interval > 0 else 0
                logger.info(f"  🔄 Effective GPS frequency: {actual_gps_freq:.1f} Hz")

    def should_process_frame(self, frame_idx: int) -> bool:
        """
        FIXED: Only process frames that follow our naming convention AND have GPS data

        Args:
            frame_idx: Frame index

        Returns:
            True ONLY if frame follows 6-frame interval AND has GPS data
        """
        # FIXED: Check if frame follows our interval naming convention
        if frame_idx % self.frame_interval != 0:
            return False  # Not a frame we should process based on interval
            
        # Check if this specific frame has GPS data
        return frame_idx in self.sync_frames

    def get_gps_for_frame(self, frame_idx: int) -> Optional[GPSData]:
        """
        Get GPS data for a specific frame (must follow naming convention)

        Args:
            frame_idx: Frame index (should be multiple of frame_interval)

        Returns:
            GPS data for the frame or None if not available
        """
        gps_idx = self.frame_to_gps.get(frame_idx)
        if gps_idx is not None and gps_idx < len(self.gps_data):
            return self.gps_data[gps_idx]
        return None

    def get_all_sync_frames(self) -> List[int]:
        """
        Get all frames that should be processed (following proper intervals)

        Returns:
            Sorted list of frame indices with GPS data (0, 6, 12, 18...)
        """
        return sorted(list(self.sync_frames))

    def get_sync_frames_count(self) -> int:
        """
        Get number of frames to process (following proper intervals)

        Returns:
            Number of frames with GPS data at proper intervals
        """
        return len(self.sync_frames)

    def get_next_sync_frame(self, current_frame: int) -> Optional[int]:
        """
        Get the next frame with GPS data that follows our naming convention

        Args:
            current_frame: Current frame index

        Returns:
            Next frame index with GPS data (multiple of frame_interval) or None
        """
        sync_frames = sorted(list(self.sync_frames))
        for frame in sync_frames:
            if frame > current_frame:
                return frame
        return None

    def get_expected_frame_sequence(self, max_frames: Optional[int] = None) -> List[int]:
        """
        Get the expected frame sequence based on our naming convention
        
        Args:
            max_frames: Maximum number of frames to generate
            
        Returns:
            List of expected frame numbers (0, 6, 12, 18...)
        """
        if not self.sync_frames:
            return []
            
        max_frame = max(self.sync_frames) if max_frames is None else max_frames
        expected_frames = list(range(0, max_frame + 1, self.frame_interval))
        
        return expected_frames

    def validate_frame_naming_convention(self) -> Dict[str, Any]:
        """
        Validate that our frames follow the proper naming convention
        
        Returns:
            Dictionary with validation results
        """
        if not self.sync_frames:
            return {
                "valid": False,
                "error": "No sync frames available",
                "expected_pattern": f"multiples of {self.frame_interval}",
                "actual_frames": []
            }
        
        sorted_frames = sorted(list(self.sync_frames))
        expected_frames = self.get_expected_frame_sequence()
        
        # Check if all frames are multiples of frame_interval
        all_multiples = all(frame % self.frame_interval == 0 for frame in sorted_frames)
        
        # Check if we have expected intervals
        intervals = np.diff(sorted_frames) if len(sorted_frames) > 1 else []
        expected_interval = self.frame_interval
        proper_intervals = all(interval % expected_interval == 0 for interval in intervals)
        
        # Find missing frames
        expected_set = set(expected_frames)
        actual_set = set(sorted_frames)
        missing_frames = expected_set - actual_set
        extra_frames = actual_set - expected_set
        
        return {
            "valid": all_multiples and proper_intervals,
            "all_multiples_of_interval": all_multiples,
            "proper_intervals": proper_intervals,
            "expected_pattern": f"multiples of {self.frame_interval}",
            "actual_frames": sorted_frames[:20],  # First 20 for display
            "missing_frames": sorted(list(missing_frames))[:10],  # First 10 missing
            "extra_frames": sorted(list(extra_frames))[:10],  # First 10 extra
            "coverage_percentage": len(actual_set) / len(expected_set) * 100 if expected_set else 0,
            "total_expected": len(expected_set),
            "total_actual": len(actual_set)
        }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about GPS-synchronized processing with frame validation

        Returns:
            Dictionary with processing statistics including frame naming validation
        """
        if not self.sync_frames:
            return {
                "gps_points": 0,
                "sync_frames": 0,
                "processing_ratio": 0.0,
                "avg_gps_frequency": 0.0,
                "frame_interval": self.frame_interval,
                "naming_convention_valid": False
            }

        sync_frames = sorted(list(self.sync_frames))
        frame_intervals = np.diff(sync_frames) if len(sync_frames) > 1 else [0]

        avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 0
        actual_gps_freq = self.video_fps / avg_interval if avg_interval > 0 else 0

        # Calculate total video frames in the GPS time range
        if len(sync_frames) >= 2:
            frame_span = max(sync_frames) - min(sync_frames)
            processing_ratio = (
                len(sync_frames) / (frame_span / self.frame_interval + 1) if frame_span > 0 else 1.0
            )
        else:
            processing_ratio = 1.0 if sync_frames else 0.0

        # Validate naming convention
        validation = self.validate_frame_naming_convention()

        return {
            "gps_points": len(self.gps_data),
            "sync_frames": len(self.sync_frames),
            "processing_ratio": processing_ratio,
            "avg_gps_frequency": actual_gps_freq,
            "frame_range": (
                (min(sync_frames), max(sync_frames)) if sync_frames else (0, 0)
            ),
            "avg_frame_interval": avg_interval,
            "expected_frame_interval": self.frame_interval,
            "naming_convention_valid": validation["valid"],
            "frame_coverage_percentage": validation["coverage_percentage"]
        }


def create_gps_synchronizer(
    gps_data: List[GPSData], 
    video_fps: float, 
    gps_fps: float = 10.0,
    frame_interval: int = 6
) -> GPSSynchronizer:
    """
    Create a GPS synchronizer for frame processing with proper frame intervals
    FIXED: Now properly handles 6-frame intervals (0, 6, 12, 18...)

    Args:
        gps_data: List of actual GPS data points
        video_fps: Video frame rate in FPS
        gps_fps: Expected GPS data rate in Hz (for validation)
        frame_interval: Process every N frames (default: 6)

    Returns:
        GPS synchronizer instance with proper frame naming
    """
    return GPSSynchronizer(gps_data, video_fps, gps_fps, frame_interval)