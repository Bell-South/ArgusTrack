# argus_track/config.py (SIMPLIFIED VERSION)

"""Simplified Configuration for Unified Tracker"""

from dataclasses import dataclass


@dataclass
class TrackerConfig:
    """Simplified configuration for unified light post tracker"""

    # === DETECTION PARAMETERS ===
    detection_conf: float = 0.20  # Detection confidence threshold
    detection_iou: float = 0.5  # NMS IoU threshold
    tracker_type: str = "bytetrack.yaml"  # Ultralytics tracker config
    max_detections: int = 10  # Max detections per frame

    # === GPS SYNCHRONIZATION ===
    gps_frame_interval: int = 6  # Process every 6th frame for GPS sync

    # === TRACK MANAGEMENT ===
    max_track_memory_age: int = 30  # Max frames to remember lost tracks

    # === STATIC CAR DETECTION ===
    enable_static_car_detection: bool = True  # Enable static car frame skipping
    static_movement_threshold_m: float = 0.3  # Minimum movement to consider moving
    static_time_threshold_s: float = 5.0  # Time before starting to skip frames

    # === OUTPUT SETTINGS ===
    export_json: bool = True  # Export JSON frame data
    export_csv: bool = True  # Export CSV GPS data

    @classmethod
    def create_for_unified_tracker(cls) -> "TrackerConfig":
        """Create optimized configuration for unified tracker"""
        return cls(
            # Conservative detection settings
            detection_conf=0.20,
            detection_iou=0.5,
            tracker_type="bytetrack.yaml",
            max_detections=20,
            # GPS settings
            gps_frame_interval=6,
            # Conservative track management
            max_track_memory_age=20,  # Short memory to prevent resurrection
            # Static car detection
            enable_static_car_detection=True,
            static_movement_threshold_m=0.05,  # Very sensitive
            static_time_threshold_s=5.0,  # Start skipping quickly
            # Output settings
            export_json=True,
            export_csv=True,
            min_detections_for_export=3,  # Only export stable tracks
        )

    def get_ultralytics_track_params(self) -> dict:
        """Get parameters for model.track() call"""
        return {
            "persist": True,
            "tracker": self.tracker_type,
            "conf": self.detection_conf,
            "iou": self.detection_iou,
            "max_det": self.max_detections,
            "verbose": False,
        }


@dataclass
class StaticCarConfig:
    """Configuration for static car detection"""

    movement_threshold_meters: float = 0.9
    stationary_time_threshold: float = 5.0
    gps_frame_interval: int = 6
