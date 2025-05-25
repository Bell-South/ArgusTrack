"""Additional comprehensive tests"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from argus_track import (
    TrackerConfig,
    LightPostTracker,
    ByteTrack,
    MockDetector
)
from argus_track.core import Detection, Track, GPSData
from argus_track.utils import calculate_iou, calculate_iou_matrix


class TestByteTrackIntegration:
    """Integration tests for ByteTrack"""
    
    def test_track_lifecycle(self):
        """Test complete track lifecycle"""
        config = TrackerConfig(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=30
        )
        tracker = ByteTrack(config)
        
        # Frame 1: New detection
        detections = [
            Detection(
                bbox=np.array([100, 100, 200, 200]),
                score=0.9,
                class_id=0,
                frame_id=0
            )
        ]
        tracks = tracker.update(detections)
        
        assert len(tracks) == 1
        assert tracks[0].state == 'tentative'
        assert tracks[0].track_id == 0
        
        # Frame 2-4: Confirm track
        for frame_id in range(1, 4):
            detections = [
                Detection(
                    bbox=np.array([102, 102, 202, 202]),
                    score=0.9,
                    class_id=0,
                    frame_id=frame_id
                )
            ]
            tracks = tracker.update(detections)
        
        assert len(tracks) == 1
        assert tracks[0].state == 'confirmed'
        assert tracks[0].hits == 4
        
        # Frame 5-35: No detections (test lost state)
        for frame_id in range(4, 35):
            tracks = tracker.update([])
        
        # Track should be removed after buffer
        assert len(tracker.active_tracks) == 0
        assert len(tracker.removed_tracks) == 1
    
    def test_two_stage_association(self):
        """Test two-stage association strategy"""
        config = TrackerConfig(
            track_thresh=0.5,
            match_thresh=0.8
        )
        tracker = ByteTrack(config)
        
        # Create initial track
        initial_det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            score=0.9,
            class_id=0,
            frame_id=0
        )
        tracker.update([initial_det])
        
        # Test high and low confidence detections
        high_conf = Detection(
            bbox=np.array([105, 105, 205, 205]),
            score=0.8,
            class_id=0,
            frame_id=1
        )
        low_conf = Detection(
            bbox=np.array([110, 110, 210, 210]),
            score=0.3,
            class_id=0,
            frame_id=1
        )
        
        # Both detections should associate with the track
        tracks = tracker.update([high_conf, low_conf])
        
        # Only high confidence should match in first stage
        assert len(tracks) == 1
        assert tracks[0].hits == 2
    
    def test_track_overlap_handling(self):
        """Test handling of overlapping tracks"""
        config = TrackerConfig()
        tracker = ByteTrack(config)
        
        # Create two overlapping detections
        det1 = Detection(
            bbox=np.array([100, 100, 200, 200]),
            score=0.9,
            class_id=0,
            frame_id=0
        )
        det2 = Detection(
            bbox=np.array([150, 150, 250, 250]),
            score=0.9,
            class_id=0,
            frame_id=0
        )
        
        tracks = tracker.update([det1, det2])
        assert len(tracks) == 2
        
        # Update with single detection in overlap area
        overlap_det = Detection(
            bbox=np.array([140, 140, 210, 210]),
            score=0.9,
            class_id=0,
            frame_id=1
        )
        
        tracks = tracker.update([overlap_det])
        # Should maintain both tracks (one matched, one lost)
        assert len(tracker.active_tracks) == 1
        assert len(tracker.lost_tracks) == 1


class TestStaticObjectAnalysis:
    """Test static object detection"""
    
    def test_static_detection(self):
        """Test detection of static objects"""
        config = TrackerConfig(
            static_threshold=2.0,
            min_static_frames=5
        )
        
        detector = MockDetector()
        tracker = LightPostTracker(config, detector)
        
        # Create track with minimal movement
        track = Track(track_id=0)
        base_pos = np.array([100, 100])
        
        for i in range(10):
            # Add small noise to simulate real detection
            noise = np.random.normal(0, 0.5, 2)
            bbox = np.array([
                base_pos[0] + noise[0],
                base_pos[1] + noise[1],
                base_pos[0] + noise[0] + 50,
                base_pos[1] + noise[1] + 100
            ])
            
            detection = Detection(
                bbox=bbox,
                score=0.9,
                class_id=0,
                frame_id=i
            )
            track.detections.append(detection)
        
        # Add to tracker
        tracker.tracker.tracks[0] = track
        
        # Analyze static objects
        static_analysis = tracker.analyze_static_objects()
        
        assert 0 in static_analysis
        assert static_analysis[0] == True


class TestGPSIntegration:
    """Test GPS functionality"""
    
    def test_gps_interpolation(self):
        """Test GPS data interpolation"""
        gps_data = [
            GPSData(
                timestamp=0.0,
                latitude=40.0,
                longitude=-74.0,
                altitude=10.0,
                heading=90.0
            ),
            GPSData(
                timestamp=1.0,
                latitude=40.001,
                longitude=-74.001,
                altitude=11.0,
                heading=92.0
            )
        ]
        
        from argus_track.utils.gps_utils import GPSInterpolator
        interpolator = GPSInterpolator(gps_data)
        
        # Test interpolation at 0.5 seconds
        interpolated = interpolator.interpolate(0.5)
        
        assert abs(interpolated.latitude - 40.0005) < 1e-6
        assert abs(interpolated.longitude - (-74.0005)) < 1e-6
        assert abs(interpolated.altitude - 10.5) < 0.01
        assert abs(interpolated.heading - 91.0) < 0.01
    
    def test_coordinate_transformation(self):
        """Test GPS coordinate transformations"""
        from argus_track.utils.gps_utils import CoordinateTransformer
        
        transformer = CoordinateTransformer(
            reference_lat=40.0,
            reference_lon=-74.0
        )
        
        # Test conversion to local coordinates
        local_x, local_y = transformer.gps_to_local(40.001, -74.001)
        
        # Should be approximately 111m north and 85m west
        assert abs(local_x - (-85)) < 10  # meters
        assert abs(local_y - 111) < 10  # meters
        
        # Test round-trip conversion
        lat2, lon2 = transformer.local_to_gps(local_x, local_y)
        assert abs(lat2 - 40.001) < 1e-6
        assert abs(lon2 - (-74.001)) < 1e-6


class TestPerformance:
    """Test performance monitoring"""
    
    def test_performance_monitor(self):
        """Test performance monitoring functionality"""
        from argus_track.utils.performance import PerformanceMonitor
        
        monitor = PerformanceMonitor(
            monitor_memory=True,
            monitor_gpu=False,
            log_interval=10
        )
        
        # Simulate processing
        with monitor.timer('frame'):
            time.sleep(0.01)  # Simulate 10ms processing
        
        with monitor.timer('detection'):
            time.sleep(0.005)  # Simulate 5ms detection
        
        with monitor.timer('tracking'):
            time.sleep(0.003)  # Simulate 3ms tracking
        
        monitor.update()
        
        # Check metrics
        assert len(monitor.metrics.frame_times) == 1
        assert monitor.metrics.frame_times[0] >= 0.01
        assert len(monitor.metrics.detection_times) == 1
        assert monitor.metrics.detection_times[0] >= 0.005
        
        # Generate report
        report = monitor.generate_report()
        assert 'summary' in report
        assert 'statistics' in report
        assert report['summary']['total_frames'] == 1


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test validation of valid configuration"""
        from argus_track.utils.config_validator import ConfigValidator
        
        config = TrackerConfig()
        errors = ConfigValidator.validate_tracker_config(config)
        assert len(errors) == 0
    
    def test_invalid_config(self):
        """Test validation of invalid configuration"""
        from argus_track.utils.config_validator import ConfigValidator
        
        config = TrackerConfig(
            track_thresh=1.5,  # Invalid: > 1
            match_thresh=-0.1,  # Invalid: < 0
            track_buffer=0,  # Invalid: < 1
            min_box_area=-10  # Invalid: < 0
        )
        
        errors = ConfigValidator.validate_tracker_config(config)
        assert len(errors) >= 4
        assert any('track_thresh' in error for error in errors)
        assert any('match_thresh' in error for error in errors)