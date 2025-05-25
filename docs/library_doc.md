## Proposed Library Structure

```
ArgusTrack/
├── README.md
├── setup.py
├── requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_detectors.py
│   ├── test_filters.py
│   ├── test_tracker.py
│   └── test_utils.py
├── examples/
│   ├── basic_tracking.py
│   ├── video_tracking_with_gps.py
│   └── config_examples/
│       └── default_config.yaml
├── docs/
│   ├── conf.py
│   ├── index.md
│   ├── api/
│   │   ├── core.md
│   │   ├── detectors.md
│   │   └── trackers.md
│   └── tutorials/
│       ├── getting_started.md
│       └── advanced_usage.md
└── argus_track/
    ├── __init__.py
    ├── __version__.py
    ├── config.py
    ├── core/
    │   ├── __init__.py
    │   ├── track.py
    │   ├── detection.py
    │   └── gps.py
    ├── filters/
    │   ├── __init__.py
    │   └── kalman.py
    ├── detectors/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── yolo.py
    │   └── mock.py
    ├── trackers/
    │   ├── __init__.py
    │   ├── bytetrack.py
    │   └── lightpost_tracker.py
    ├── utils/
    │   ├── __init__.py
    │   ├── iou.py
    │   ├── visualization.py
    │   └── io.py
    └── main.py
```

## Module Breakdown

### Core Data Classes (`core/`)
- `track.py`: Track class
- `detection.py`: Detection class
- `gps.py`: GPSData class

### Configuration (`config.py`)
- TrackerConfig and other configuration classes

### Filters (`filters/`)
- `kalman.py`: KalmanBoxTracker implementation

### Detectors (`detectors/`)
- `base.py`: ObjectDetector protocol/base class
- `yolo.py`: YOLODetector implementation
- `mock.py`: MockDetector for testing

### Trackers (`trackers/`)
- `bytetrack.py`: ByteTrack core implementation
- `lightpost_tracker.py`: LightPostTracker with GPS integration

### Utilities (`utils/`)
- `iou.py`: IoU calculation utilities
- `visualization.py`: Visualization functions
- `io.py`: I/O operations, GPS data loading

### Main Entry Point (`main.py`)
- Command-line interface and main execution logic