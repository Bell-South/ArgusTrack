"""Track data structure"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .detection import Detection


@dataclass
class Track:
    """Represents a tracked object through multiple frames"""

    track_id: int
    detections: List[Detection] = field(default_factory=list)
    kalman_filter: Optional["KalmanBoxTracker"] = None
    state: str = "tentative"  # tentative, confirmed, lost, removed
    hits: int = 0  # Number of successful updates
    age: int = 0  # Total frames since creation
    time_since_update: int = 0  # Frames since last update
    start_frame: int = 0

    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (has enough hits)"""
        return self.state == "confirmed"

    @property
    def is_active(self) -> bool:
        """Check if track is currently active"""
        return self.state in ["tentative", "confirmed"]

    def to_tlbr(self) -> np.ndarray:
        """Get current position in tlbr format"""
        if self.kalman_filter is None:
            return self.detections[-1].tlbr if self.detections else np.zeros(4)
        return self.kalman_filter.get_state()

    @property
    def last_detection(self) -> Optional[Detection]:
        """Get the most recent detection"""
        return self.detections[-1] if self.detections else None

    @property
    def trajectory(self) -> List[np.ndarray]:
        """Get trajectory as list of center points"""
        return [det.center for det in self.detections]

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "track_id": self.track_id,
            "state": self.state,
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "start_frame": self.start_frame,
            "detections": [
                det.to_dict() for det in self.detections[-10:]
            ],  # Last 10 detections
        }
