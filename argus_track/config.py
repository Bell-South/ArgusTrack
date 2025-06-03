# argus_track/config.py (UPDATED WITH TRACK CONSOLIDATION)

"""Configuration classes for ArgusTrack with Track Consolidation"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import json
import pickle
import numpy as np
from pathlib import Path

@dataclass
class TrackConsolidationConfig:
    """Configuration for track ID consolidation and reappearance handling"""
    
    # === REAPPEARANCE DETECTION ===
    max_gap_frames: int = 15                    # Max frames to remember lost tracks
    reappearance_iou_threshold: float = 0.3     # IoU threshold for reappearance
    reappearance_spatial_threshold: float = 80  # Pixel distance for reappearance
    
    # === DUPLICATE ID PREVENTION ===
    duplicate_spatial_threshold: float = 50     # Pixels apart to consider different objects
    duplicate_size_threshold: float = 0.7       # Size ratio threshold (±30%)
    duplicate_iou_threshold: float = 0.5        # IoU threshold for duplicate detection
    
    # === ID MANAGEMENT ===
    prefer_lower_id: bool = True                # Always keep lower track ID
    enable_id_consolidation: bool = True        # Enable real-time ID fixing
    enable_reappearance_detection: bool = True  # Enable gap-bridging

    # === MOTION PREDICTION FEATURES === (ADD THESE)
    enable_motion_prediction: bool = True       # Enable motion prediction
    enable_visual_features: bool = True         # Enable visual feature matching
    feature_similarity_threshold: float = 0.6  # Visual similarity threshold
    motion_weight: float = 0.4                 # Motion prediction weight
    feature_weight: float = 0.6                # Visual feature weight

    # === MEMORY MANAGEMENT ===
    max_track_memory: int = 100                 # Max tracks to keep in memory
    cleanup_interval_frames: int = 30           # Clean old tracks every N frames
    
    # === DEBUGGING ===
    log_consolidations: bool = True             # Log ID consolidations
    log_reappearances: bool = True              # Log track reappearances

# Add performance optimization settings
@dataclass
class PerformanceConfig:
    """Performance optimization settings for 10fps processing"""
    target_fps: float = 10.0                   # Target processing speed
    max_processing_time_ms: float = 100.0      # Max time per frame
    enable_gpu_acceleration: bool = True       # Use GPU if available
    
    # Memory management
    max_memory_usage_mb: float = 2048.0        # Max memory usage
    cleanup_frequency: int = 100               # Cleanup every N frames
    
    # Feature extraction optimization
    visual_feature_decimation: int = 1         # Extract features every N detections
    motion_prediction_decimation: int = 1      # Predict motion every N frames
    
    # Multi-threading settings
    enable_parallel_processing: bool = False   # Parallel processing (experimental)
    num_worker_threads: int = 2                # Worker threads if enabled

@dataclass
class TrackerConfig:
    """Configuration for simplified light post tracker (monocular)"""
    
    # === DETECTION PARAMETERS ===
    detection_conf: float = 0.15               # Detection confidence threshold
    detection_iou: float = 0.8                 # NMS IoU threshold
    tracker_type: str = "bytetrack.yaml"       # Ultralytics tracker config
    max_detections: int = 50                   # Max detections per frame
    
    # === GPS SYNCHRONIZATION ===
    gps_frame_interval: int = 6                # Process every 6th frame for GPS sync
    enable_gps_extraction: bool = True         # Auto-extract GPS from video metadata
    
    # === TRACK CONSOLIDATION ===
    track_consolidation: TrackConsolidationConfig = field(
        default_factory=TrackConsolidationConfig
    )
    
    # === OUTPUT SETTINGS ===
    export_json: bool = True                   # Export JSON frame data
    export_csv: bool = True                    # Export CSV GPS data
    min_detections_for_export: int = 2         # Minimum detections to include in output
    
    # === STATIC CAR DETECTION ===
    enable_static_car_detection: bool = True   # Enable static car frame skipping
    static_movement_threshold_m: float = 0.9   # Minimum movement to consider moving
    static_time_threshold_s: float = 5.0       # Time before starting to skip frames
    
    # === REAL-TIME DISPLAY ===
    display_gps_frames_only: bool = True       # Only display GPS-synchronized frames
    display_track_history: bool = True         # Show track trajectories
    display_consolidation_info: bool = True    # Show ID consolidation info
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Enhanced processing flags
    enable_enhanced_visualization: bool = True  # Show motion/visual info
    enable_performance_monitoring: bool = True  # Monitor performance
    enable_adaptive_quality: bool = True        # Adapt quality based on performance
    
    def optimize_for_realtime(self):
        """Optimize configuration for real-time processing at 10fps"""
        # Adjust detection parameters for speed
        self.detection_conf = max(0.2, self.detection_conf)  # Higher threshold for speed
        self.max_detections = min(25, self.max_detections)   # Fewer detections for speed
        
        # Optimize track consolidation for speed
        tc = self.track_consolidation
        tc.max_gap_frames = min(10, tc.max_gap_frames)       # Shorter memory for speed
        tc.cleanup_interval_frames = min(30, tc.cleanup_interval_frames)
        
        # Optimize visual features for speed
        tc.feature_similarity_threshold = min(0.7, tc.feature_similarity_threshold)  # Higher threshold
        
        # Set performance targets
        self.performance.target_fps = 10.0
        self.performance.max_processing_time_ms = 100.0
        
        self.logger.info("Configuration optimized for real-time 10fps processing")
    
    def optimize_for_accuracy(self):
        """Optimize configuration for maximum accuracy (slower processing)"""
        # Lower thresholds for better detection
        self.detection_conf = 0.1
        self.max_detections = 50
        
        # Enhanced track consolidation for accuracy
        tc = self.track_consolidation
        tc.max_gap_frames = 20
        tc.feature_similarity_threshold = 0.5  # Lower for more matches
        tc.enable_visual_features = True
        tc.enable_motion_prediction = True
        
        # Relaxed performance targets
        self.performance.target_fps = 5.0
        self.performance.max_processing_time_ms = 200.0
        
        self.logger.info("Configuration optimized for maximum accuracy")
    
    @classmethod
    def create_simplified_tracker(cls) -> 'TrackerConfig':
        """Create configuration for simplified tracking - STATIONARY OBJECTS"""
        return cls(
            # Optimized detection settings
            detection_conf=0.15,
            detection_iou=0.6,
            tracker_type="bytetrack.yaml",
            max_detections=10,  # Lower limit
            
            # GPS settings
            gps_frame_interval=6,
            enable_gps_extraction=True,
            
            # SIMPLIFIED track consolidation - DISABLE ADVANCED FEATURES
            track_consolidation=TrackConsolidationConfig(
                max_gap_frames=5,                      # Short memory
                reappearance_iou_threshold=0.5,
                reappearance_spatial_threshold=30,
                duplicate_spatial_threshold=20,
                duplicate_size_threshold=0.8,
                duplicate_iou_threshold=0.4,
                prefer_lower_id=True,
                enable_id_consolidation=True,
                enable_reappearance_detection=False,   # DISABLE - causing issues
                
                # === DISABLE ADVANCED FEATURES FOR STATIONARY ===
                enable_motion_prediction=False,        # DISABLE - not needed for stationary
                enable_visual_features=False,          # DISABLE - causing re-initialization
                feature_similarity_threshold=0.7,
                motion_weight=0.0,                     # No motion weight
                feature_weight=0.0,                    # No visual weight
                
                max_track_memory=30,
                cleanup_interval_frames=10,
                log_consolidations=True,
                log_reappearances=True
            ),
            
            # More aggressive export threshold
            min_detections_for_export=10,            # Only export stable tracks
            
            # Static car detection
            enable_static_car_detection=True,
            static_movement_threshold_m=0.3,         # Very sensitive
            static_time_threshold_s=2.0,             # Start skipping quickly
            
            # Display settings
            display_gps_frames_only=True,
            display_track_history=True,
            display_consolidation_info=True
        )

    def get_ultralytics_track_params(self) -> dict:
        """Get enhanced parameters for model.track() call with motion prediction"""
        base_params = self.get_ultralytics_track_params()
        
        # Enhanced parameters for motion prediction and visual features
        enhanced_params = {
            **base_params,
            'persist': True,
            'tracker': self.tracker_type,
            'conf': self.detection_conf,
            'iou': self.detection_iou,
            'max_det': self.max_detections,
            'verbose': False,
            # Enhanced tracking parameters
            'half': False,                    # Full precision for better features
            'device': None,                   # Auto-detect best device
            'classes': None,                  # Track all classes
            'agnostic_nms': False,           # Class-specific NMS
            'retina_masks': False,           # No masks needed
            'embed': None,                   # No embedding override
        }
        
        return enhanced_params

    @classmethod
    def create_simplified_tracker(cls) -> 'TrackerConfig':
        """Create configuration for simplified tracking (no depth/geolocation)"""
        return cls(
            # Optimized detection settings
            detection_conf=0.20,
            detection_iou=0.5,
            tracker_type="bytetrack.yaml",
            max_detections=30,
            
            # GPS settings
            gps_frame_interval=6,
            enable_gps_extraction=True,
            
            # Track consolidation with LESS AGGRESSIVE settings
            track_consolidation=TrackConsolidationConfig(
                max_gap_frames=8,                      # REDUCED: Less aggressive reappearance
                reappearance_iou_threshold=0.5,        # HIGHER: More strict reappearance
                reappearance_spatial_threshold=40,     # SMALLER: Closer proximity required
                duplicate_spatial_threshold=30,        # SMALLER: Tighter duplicate detection
                duplicate_size_threshold=0.8,          # HIGHER: More similar size required
                duplicate_iou_threshold=0.6,           # HIGHER: More overlap required
                prefer_lower_id=True,
                enable_id_consolidation=True,
                enable_reappearance_detection=False,   # DISABLE: Causing old ID reuse
                log_consolidations=True,
                log_reappearances=True
            ),
            
            # Output settings
            export_json=True,
            export_csv=True,
            min_detections_for_export=2,
            
            # Static car detection
            enable_static_car_detection=True,
            static_movement_threshold_m=0.9,
            static_time_threshold_s=5.0,
            
            # Real-time display
            display_gps_frames_only=True,
            display_track_history=True,
            display_consolidation_info=True
        )
    
    def get_ultralytics_track_params(self) -> dict:
        """Get parameters for model.track() call"""
        return {
            'persist': True,
            'tracker': self.tracker_type,
            'conf': self.detection_conf,
            'iou': self.detection_iou,
            'max_det': self.max_detections,
            'verbose': False  # Reduce verbosity for cleaner output
        }

@dataclass
class DetectorConfig:
    """Configuration for object detectors"""
    model_path: str
    config_path: str = ""
    target_classes: Optional[list] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    model_type: str = "yolov11"

@dataclass
class StereoCalibrationConfig:
    """Stereo camera calibration parameters (kept for backward compatibility)"""
    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    P1: Optional[np.ndarray] = None
    P2: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    baseline: float = 0.0
    image_width: int = 1920
    image_height: int = 1080
    
    @classmethod
    def from_pickle(cls, calibration_path: str) -> 'StereoCalibrationConfig':
        """Load stereo calibration from pickle file"""
        try:
            with open(calibration_path, 'rb') as f:
                calib_data = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to load calibration file {calibration_path}: {e}")
        
        baseline = calib_data.get('baseline', 0.0)
        if baseline == 0.0 and 'T' in calib_data:
            baseline = float(np.linalg.norm(calib_data['T']))
        
        return cls(
            camera_matrix_left=calib_data['camera_matrix_left'],
            camera_matrix_right=calib_data['camera_matrix_right'],
            dist_coeffs_left=calib_data['dist_coeffs_left'],
            dist_coeffs_right=calib_data['dist_coeffs_right'],
            R=calib_data['R'],
            T=calib_data['T'],
            E=calib_data.get('E'),
            F=calib_data.get('F'),
            P1=calib_data.get('P1'),
            P2=calib_data.get('P2'),
            Q=calib_data.get('Q'),
            baseline=baseline,
            image_width=calib_data.get('image_width', 1920),
            image_height=calib_data.get('image_height', 1080)
        )

@dataclass
class CameraConfig:
    """Camera calibration parameters (backward compatibility)"""
    camera_matrix: list
    distortion_coeffs: list
    image_width: int
    image_height: int
    
    @classmethod
    def from_file(cls, calibration_path: str) -> 'CameraConfig':
        """Load camera configuration from file"""
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        return cls(**data)