# 🎯 Stereo Geolocation Tracker

**ByteTrack + Kalman + Stereo Vision for Light Post Geolocation**

A specialized computer vision system that tracks light posts and infrastructure elements in stereo video sequences and determines their precise geographic coordinates with 1-2 meter accuracy.

## 📋 Project Overview

### Objective
Track light posts, poles, and similar static infrastructure in stereo camera video and determine their precise geographic coordinates (latitude/longitude) for mapping and asset management.

### Key Specifications
- **Accuracy Target**: 1-2 meter precision
- **Hardware**: GoPro 11 stereo camera setup  
- **Processing Strategy**: GPS-synchronized frames only (no interpolation)
- **Video Format**: 60fps video with 10fps GPS data (process every 6th frame)

## 🏗️ System Architecture

### Core Components
- **ByteTrack Algorithm**: Two-stage association for robust object tracking
- **Kalman Filtering**: Motion prediction optimized for static objects and 6-frame gaps
- **Stereo Matching**: Associate tracked objects between left/right cameras
- **3D Triangulation**: Calculate real-world positions using stereo geometry
- **Geographic Conversion**: Transform camera coordinates to GPS coordinates

### Processing Pipeline

```
GPS Frame Selection → YOLO Detection → ByteTrack → Stereo Matching → Triangulation → Geolocation
    (every 6th)         (both cameras)   (tracking)   (L/R association)  (3D positions)  (lat/lon)
```

## 📥 System Inputs

| Input Type | Description | Format |
|------------|-------------|---------|
| **Video Files** | Left and right camera recordings | `.mp4`, `.avi` |
| **Detection Model** | User's trained YOLOv11 model | Model file |
| **Calibration** | Stereo camera calibration data | `.pkl` file |
| **GPS Data** | Extracted from GoPro metadata | JSON/CSV |

### GPS Synchronization Strategy
- **Video**: 60 FPS (16.67ms intervals)
- **GPS**: 10 FPS (100ms intervals) 
- **Solution**: Process only frames with actual GPS coordinates
- **Advantage**: Higher reliability, no interpolation errors

## 🎯 Key Technical Decisions

### ✅ Confirmed Approaches
- **Stereo Primary**: Better accuracy than monocular through triangulation
- **No GPS Interpolation**: Process only real GPS frames to avoid uncertainty
- **ByteTrack Algorithm**: Proven two-stage association method
- **Static Object Focus**: Optimized for light posts, poles, traffic signs
- **6-Frame Processing**: Kalman filter adapted for 100ms intervals

### 🔄 Fallback Options
- **Monocular Mode**: Available as fallback if stereo fails
- **Modular Design**: Easy to swap between stereo/monocular processing

## 📤 Output Format

### JSON Structure
```json
{
  "detected_objects": [
    {
      "track_id": 1,
      "class": "light_post",
      "latitude": 40.712345,
      "longitude": -74.006789,
      "confidence": 0.92,
      "reliability": 0.88,
      "frames_tracked": 45,
      "first_seen_frame": 10,
      "last_seen_frame": 55
    }
  ],
  "metadata": {
    "total_frames": 100,
    "processing_time": 45.2,
    "objects_detected": 2,
    "static_objects": 2
  }
}
```

### Export Options
- **JSON**: Complete tracking and geolocation data
- **GeoJSON**: Ready for mapping applications and GIS tools
- **CSV**: Simplified format for spreadsheet analysis

## ⚠️ Technical Challenges Identified

### Primary Challenges
- **6-Frame Gaps**: Maintaining tracking continuity with sparse frame processing
- **Stereo Matching**: Robust association of objects between left/right cameras  
- **Static Detection**: Reliable identification of non-moving infrastructure
- **Triangulation Accuracy**: Handling detection noise and calibration errors
- **Geographic Precision**: Preserving accuracy through coordinate transformations

### Mitigation Strategies
- Extended Kalman prediction for larger time gaps
- Epipolar constraint validation for stereo matching
- Position variance analysis for static object detection
- Outlier filtering and statistical averaging
- High-quality stereo calibration requirements

## 🚀 Implementation Status

### ✅ Completed Design Decisions
- [x] Architecture definition
- [x] Algorithm selection (ByteTrack + Kalman)
- [x] GPS synchronization strategy  
- [x] Input/output format specification
- [x] Processing pipeline design

### 🔧 Next Steps (Priority Order)

#### High Priority
1. **YOLO Integration**: Connect user's YOLOv11 model with detection pipeline
2. **Calibration Format**: Define exact `.pkl` file structure and requirements  
3. **GPS Data Format**: Specify extracted GPS data structure and timing
4. **Core Implementation**: ByteTrack + Kalman filter for 6-frame gaps

#### Medium Priority  
5. **Stereo Matching**: Implement robust left/right track association
6. **Static Analysis**: Position variance calculation for object classification
7. **Sample Pipeline**: Create test data and validation framework
8. **Performance Optimization**: Processing speed and memory efficiency

#### Future Enhancements
9. **Monocular Fallback**: Implement single-camera processing mode
10. **Real-time Processing**: Adapt for live video streams
11. **Advanced Filtering**: Additional outlier detection methods
12. **Visualization Tools**: Interactive result viewing and validation

## 🔧 Technical Requirements

### Calibration Data Structure
```python
# Required in .pkl file
{
    "camera_matrix_left": np.array(...),   # 3x3 intrinsic matrix
    "camera_matrix_right": np.array(...),  # 3x3 intrinsic matrix  
    "dist_coeffs_left": np.array(...),     # Distortion coefficients
    "dist_coeffs_right": np.array(...),    # Distortion coefficients
    "R": np.array(...),                    # 3x3 rotation matrix
    "T": np.array(...),                    # 3x1 translation vector
    "baseline": float,                     # Distance between cameras (meters)
    # Optional: P1, P2, Q matrices for optimization
}
```

### GPS Data Format
```json
[
    {
        "frame_id": 0,
        "timestamp": 1000.0,
        "latitude": 40.7128,
        "longitude": -74.0060,
        "altitude": 10.0,
        "heading": 45.0,
        "accuracy": 1.0
    }
]
```

## 📊 Performance Expectations

### Processing Metrics
- **Frame Rate**: 10 effective FPS (every 6th frame of 60fps video)
- **Tracking**: Optimized for static objects with minimal movement
- **Accuracy**: 1-2 meter geolocation precision under optimal conditions
- **Reliability**: Position confidence scoring for each detected object

### Accuracy Dependencies
- Stereo calibration quality
- GPS accuracy from GoPro (typically 1-3 meters)
- YOLO detection precision
- Camera baseline distance (wider = better for distant objects)

## 📝 Usage Example

```bash
# Future command-line interface
python stereo_tracker.py \
    --left-video left_camera.mp4 \
    --right-video right_camera.mp4 \
    --model yolov11_lightposts.pt \
    --calibration stereo_calibration.pkl \
    --gps gps_data.json \
    --output results.json
```

## 🔗 Integration Points

### User-Provided Components
- **YOLOv11 Model**: Pre-trained object detection model
- **Calibration Data**: Stereo camera calibration from user's system
- **GPS Extraction**: User handles GPS metadata extraction from GoPro

### System-Provided Components  
- **Tracking Algorithm**: ByteTrack implementation with Kalman filtering
- **Stereo Processing**: Triangulation and coordinate transformation
- **Geolocation Engine**: Camera-to-GPS coordinate conversion
- **Output Generation**: JSON/GeoJSON formatting for mapping tools

---

*This project represents a complete pipeline from stereo video input to precise geographic coordinates, specifically optimized for infrastructure mapping and asset management applications.*