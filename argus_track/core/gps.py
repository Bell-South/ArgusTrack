"""GPS data structure"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GPSData:
    """GPS data for a single frame"""

    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    heading: float
    accuracy: float = 1.0  # GPS accuracy in meters

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "heading": self.heading,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPSData":
        """Create from dictionary representation"""
        return cls(**data)

    @classmethod
    def from_csv_line(cls, line: str) -> "GPSData":
        """Create from CSV line"""
        parts = line.strip().split(",")
        if len(parts) < 5:
            raise ValueError(f"Invalid GPS data line: {line}")

        return cls(
            timestamp=float(parts[0]),
            latitude=float(parts[1]),
            longitude=float(parts[2]),
            altitude=float(parts[3]),
            heading=float(parts[4]),
            accuracy=float(parts[5]) if len(parts) > 5 else 1.0,
        )
