"""Detection data structure"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Single object detection"""

    bbox: np.ndarray  # [x1, y1, x2, y2] format
    score: float  # Confidence score [0, 1]
    class_id: int  # Object class ID
    frame_id: int  # Frame number

    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box in top-left, bottom-right format"""
        return self.bbox

    @property
    def xywh(self) -> np.ndarray:
        """Get bounding box in center-x, center-y, width, height format"""
        x1, y1, x2, y2 = self.bbox
        return np.array(
            [
                (x1 + x2) / 2,  # center x
                (y1 + y2) / 2,  # center y
                x2 - x1,  # width
                y2 - y1,  # height
            ]
        )

    @property
    def area(self) -> float:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> np.ndarray:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "bbox": self.bbox.tolist(),
            "score": self.score,
            "class_id": self.class_id,
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Detection":
        """Create from dictionary representation"""
        return cls(
            bbox=np.array(data["bbox"]),
            score=data["score"],
            class_id=data["class_id"],
            frame_id=data["frame_id"],
        )
