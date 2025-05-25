# Argus Track: Complete Usage Guide

This comprehensive guide covers all aspects of using Argus Track for stereo light post tracking and geolocation.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation & Setup](#installation--setup)
3. [Stereo Camera Calibration](#stereo-camera-calibration)
4. [Data Preparation](#data-preparation)
5. [Basic Usage](#basic-usage)
6. [Advanced Configuration](#advanced-configuration)
7. [API Reference](#api-reference)
8. [Output Analysis](#output-analysis)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

## System Overview

Argus Track processes stereo video sequences to track and geolocate light posts with 1-2 meter accuracy using:

- **Stereo Vision**: 3D depth estimation from camera pairs
- **ByteTrack Algorithm**: Robust multi-object tracking
- **GPS Synchronization**: Frame-accurate positioning data
- **YOLOv11 Detection**: State-of-the-art object detection
- **3D Triangulation**: Camera-to-world coordinate transformation

### Processing Workflow

```
Stereo Videos (60fps) + GPS Data (10fps) → Frame Sync → Detection → 
Stereo Matching → 3D Triangulation → Tracking → Geolocation → Export
```

## Installation & Setup

### Requirements

- **Python**: 3.8 or newer
- **GPU**: NVIDIA GPU with CUDA 11.0+ (recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: SSD recommended for video processing

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/Bell-South/ArgusTrack.git
cd ArgusTrack

# 2. Create virtual environment
python -m venv argus_env
source argus_env/bin/activate  # Linux/Mac
# or
argus_env\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r argus_track/requirements.txt

# 4. Install package
pip install -e .

# 5. Verify installation
argus_track --help
```

### GPU Setup (Optional but Recommended)

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Stereo Camera Calibration

### Hardware Setup

**Recommended: GoPro Hero 11 Stereo Rig**
- **Baseline**: 10-15cm separation
- **Mounting**: Rigid connection, parallel cameras
- **Synchronization**: Manual start with clap/flash for sync

### Calibration Process

1. **Capture Calibration Images**:
   ```bash
   # Take 20-30 image pairs of checkerboard pattern
   # Vary positions: center, corners, different distances
   # Ensure pattern is visible in both cameras
   ```

2. **Create Calibration Script**:
   ```bash
   # Save as scripts/calibrate_stereo.py
   python scripts/calibrate_stereo.py \
       --left-pattern "calibration/left/*.jpg" \
       --right-pattern "calibration/right/*.jpg" \
       --board-size 9 6 \
       --square-size 25.0 \
       --output stereo_calibration.pkl
   ```

3. **Validate Calibration**:
   ```python
   from argus_track.stereo import StereoCalibrationManager
   
   calib = StereoCalibrationManager.from_pickle_file('stereo_calibration.pkl')
   is_valid, errors = calib.validate_calibration()
   
   if is_valid:
       print("✅ Calibration valid")
       print(calib.get_calibration_summary())
   else:
       print("❌ Calibration issues:", errors)
   ```

### Calibration Quality Metrics

Good calibration should have:
- **Reprojection Error**: < 1.0 pixels
- **Baseline**: 10-20cm for outdoor scenes
- **Focal Length**: Consistent between cameras (±5%)

## Data Preparation

### Video Requirements

- **Format**: MP4, AVI, or MOV
- **Resolution**: 1920x1080 minimum
- **Frame Rate**: 30-60 fps
- **Synchronization**: Sub-second timing between left/right cameras

### GPS Data Format

Create CSV file with GPS data:
```csv
timestamp,latitude,longitude,altitude,heading,accuracy
1623456789.123,40.712345,-74.006789,10.5,45.0,1.2
1623456789.223,40.712346,-74.006790,10.6,45.2,1.1
```

**Field Descriptions**:
- `timestamp`: Unix timestamp or seconds from start
- `latitude/longitude`: Decimal degrees (WGS84)
- `altitude`: Meters above sea level
- `heading`: Degrees (0-360, optional)
- `accuracy`: GPS accuracy in meters

### Data Synchronization

Align video and GPS timestamps:
```python
from argus_track.utils.gps_utils import sync_gps_with_frames

# Synchronize GPS with video frames
synced_gps = sync_gps_with_frames(
    gps_data=raw_gps_data,
    video_fps=60.0,
    start_timestamp=video_start_time
)
```

## Basic Usage

### Command Line Interface

**Stereo Tracking**:
```bash
# Basic stereo tracking
argus_track --stereo left.mp4 right.mp4 \
    --calibration stereo_calibration.pkl \
    --detector yolov11 \
    --model yolov11n.pt

# With GPS data
argus_track --stereo left.mp4 right.mp4 \
    --calibration stereo_calibration.pkl \
    --gps gps_data.csv \
    --output tracking_result.mp4

# Custom configuration
argus_track --stereo left.mp4 right.mp4 \
    --calibration stereo_calibration.pkl \
    --config stereo_config.yaml \
    --verbose
```

**Legacy Monocular Mode**:
```bash
# Single camera tracking (legacy)
argus_track input_video.mp4 \
    --detector yolo \
    --model yolov4.weights \
    --gps gps_data.csv
```

### Python API

**Basic Stereo Processing**:
```python
from argus_track import (
    TrackerConfig, StereoCalibrationConfig, 
    StereoLightPostTracker, YOLOv11Detector
)
from argus_track.utils.io import load_gps_data

# Load configuration
config = TrackerConfig(
    track_thresh=0.5,
    match_thresh=0.8,
    stereo_mode=True,
    gps_frame_interval=6
)

# Load calibration
stereo_calibration = StereoCalibrationConfig.from_pickle(
    'stereo_calibration.pkl'
)

# Initialize detector
detector = YOLOv11Detector(
    model_path='yolov11n.pt',
    target_classes=['traffic light', 'stop sign', 'pole'],
    device='auto'
)

# Initialize tracker
tracker = StereoLightPostTracker(
    config=config,
    detector=detector,
    stereo_calibration=stereo_calibration
)

# Load GPS data
gps_data = load_gps_data('gps_data.csv')

# Process video
tracks = tracker.process_stereo_video(
    left_video_path='left.mp4',
    right_video_path='right.mp4',
    gps_data=gps_data,
    save_results=True
)

# Get results
stats = tracker.get_tracking_statistics()
locations = tracker.estimated_locations

print(f"Processed {stats['total_stereo_tracks']} tracks")
print(f"Found {len(locations)} static objects")
```

## Advanced Configuration

### Configuration Files

**Complete Stereo Configuration** (`stereo_config.yaml`):
```yaml
# Tracking parameters
track_thresh: 0.5
match_thresh: 0.8
track_buffer: 50
min_box_area: 100.0
static_threshold: 2.0
min_static_frames: 10

# Stereo processing
stereo_mode: true
stereo_match_threshold: 0.7
max_stereo_distance: 100.0
gps_frame_interval: 6

# Detection
detector:
  model_type: "yolov11"
  model_path: "models/yolov11s.pt"
  confidence_threshold: 0.4
  nms_threshold: 0.45
  target_classes:
    - "traffic light"
    - "stop sign"
    - "pole"
  device: "auto"

# Calibration
stereo_calibration:
  calibration_file: "calibration/stereo_calibration.pkl"
  baseline: 0.12
  image_width: 1920
  image_height: 1080

# GPS processing
gps:
  accuracy_threshold: 5.0
  outlier_threshold: 30.0
  coordinate_system: "WGS84"

# Output options
output:
  save_video: true
  save_geojson: true
  save_json: true
  video_codec: "mp4v"
  geojson_precision: 6

# Performance tuning
performance:
  gpu_backend: "auto"
  batch_size: 1
  max_track_age: 200
  min_track_length: 5
```

### Custom Detector Integration

```python
from argus_track.detectors import ObjectDetector

class CustomInfrastructureDetector(ObjectDetector):
    """Custom detector for infrastructure objects"""
    
    def __init__(self, model_path: str):
        # Initialize your custom model
        self.model = load_custom_model(model_path)
        self.class_names = [
            'light_post', 'traffic_signal', 'utility_pole',
            'street_lamp', 'camera_mount', 'sign_post'
        ]
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        # Implement detection logic
        detections = self.model.predict(frame)
        
        return [{
            'bbox': det.bbox,
            'score': det.confidence,
            'class_name': self.class_names[det.class_id],
            'class_id': det.class_id
        } for det in detections]
    
    def get_class_names(self) -> List[str]:
        return self.class_names
```

### Advanced Stereo Processing

```python
from argus_track.stereo import StereoMatcher, StereoTriangulator

# Custom stereo matching
stereo_matcher = StereoMatcher(
    calibration=stereo_calibration,
    max_disparity=250.0,
    min_disparity=10.0,
    epipolar_threshold=1.5,
    iou_threshold=0.3
)

# Custom triangulation with coordinate transformation
triangulator = StereoTriangulator(
    calibration=stereo_calibration
)

# Set camera pose for improved world coordinates
triangulator.set_camera_pose(
    gps_position=initial_gps_position,
    orientation_angles=(0, 5, 0)  # Slight pitch adjustment
)

# Process with custom components
tracker = StereoLightPostTracker(config, detector, stereo_calibration)
tracker.stereo_matcher = stereo_matcher
tracker.triangulator = triangulator
```

## API Reference

### Core Classes

#### `StereoLightPostTracker`
Main stereo tracking class.

```python
class StereoLightPostTracker:
    def __init__(self, config: TrackerConfig, 
                 detector: ObjectDetector,
                 stereo_calibration: StereoCalibrationConfig)
    
    def process_stereo_video(self, 
                           left_video_path: str,
                           right_video_path: str,
                           gps_data: Optional[List[GPSData]] = None,
                           output_path: Optional[str] = None,
                           save_results: bool = True) -> Dict[int, StereoTrack]
    
    def get_tracking_statistics(self) -> Dict[str, Any]
```

#### `StereoCalibrationConfig`
Stereo camera calibration data.

```python
@dataclass
class StereoCalibrationConfig:
    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray              # Rotation between cameras
    T: np.ndarray              # Translation between cameras
    baseline: float            # Distance between cameras (m)
    
    @classmethod
    def from_pickle(cls, path: str) -> 'StereoCalibrationConfig'
```

#### `StereoTrack`
Individual tracked object with 3D information.

```python
@dataclass
class StereoTrack:
    track_id: int
    stereo_detections: List[StereoDetection]
    world_trajectory: List[np.ndarray]
    gps_trajectory: List[np.ndarray]
    estimated_location: Optional[GeoLocation]
    
    @property
    def is_static_3d(self) -> bool
    @property
    def average_depth(self) -> float
```

## Output Analysis

### Result Files

After processing, you'll get:

1. **`results.json`**: Complete tracking data
2. **`results.geojson`**: Locations for GIS software
3. **`output_video.mp4`**: Visualization video
4. **`processing_log.txt`**: Detailed processing log

### Analyzing Results

```python
import json
import geopandas as gpd

# Load tracking results
with open('results.json', 'r') as f:
    results = json.load(f)

# Analyze tracking quality
metadata = results['metadata']
print(f"Processed {metadata['total_frames']} frames")
print(f"Average processing time: {metadata['processing_times']['mean']:.3f}s")

# Load locations in GIS software
gdf = gpd.read_file('results.geojson')
print(f"Found {len(gdf)} locations")
print(f"Average reliability: {gdf['reliability'].mean():.2f}")

# Filter high-confidence locations
high_conf = gdf[gdf['reliability'] > 0.8]
print(f"High confidence locations: {len(high_conf)}")
```

### Quality Metrics

```python
# Analyze track quality
for track_id, track_data in results['stereo_tracks'].items():
    print(f"Track {track_id}:")
    print(f"  - Static: {track_data['is_static_3d']}")
    print(f"  - Depth: {track_data['average_depth']:.1f}m")
    print(f"  - Consistency: {track_data['depth_consistency']:.2f}")
    
    if track_data['estimated_location']:
        loc = track_data['estimated_location']
        print(f"  - Location: ({loc['latitude']:.6f}, {loc['longitude']:.6f})")
        print(f"  - Accuracy: {loc['accuracy']:.1f}m")
        print(f"  - Reliability: {loc['reliability']:.2f}")
```

## Troubleshooting

### Common Issues

**1. Poor Stereo Calibration**
```bash
# Symptoms: High reprojection error, inconsistent depth
# Solutions:
- Recalibrate with more images
- Ensure pattern is sharp and well-lit
- Check camera synchronization

# Test calibration:
python -c "
from argus_track.stereo import StereoCalibrationManager
ca