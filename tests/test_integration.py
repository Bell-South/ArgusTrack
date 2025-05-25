    # tests/test_integration.py

"""Integration tests for ByteTrack Light Post Tracking system"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json
import os

from argus_track import (
    TrackerConfig,
    Detection,
    Track,
    GPSData,
    ByteTrack, 
    LightPostTracker, 
    MockDetector
)
from argus_track.utils.gps_utils import GeoLocation


class TestBasicTracking:
    """Test basic tracking functionality"""
    
    def test_track_lifecycle(self):
        """Test full track lifecycle from creation to removal"""
        config = TrackerConfig(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=10  # Short buffer for testing
        )
        tracker = ByteTrack(config)
        
        # Frame 1: Create track
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
        track_id = tracks[0].track_id
        
        # Frames 2-3: Confirm track
        for i in range(1, 3):
            detections = [
                Detection(
                    bbox=np.array([102, 102, 202, 202]),
                    score=0.9,
                    class_id=0,
                    frame_id=i
                )
            ]
            tracks = tracker.update(detections)
        
        assert len(tracks) == 1
        assert tracks[0].state == 'confirmed'
        assert tracks[0].track_id == track_id
        
        # Frames 4-14: No detections, track should be lost then removed
        for i in range(3, 15):
            tracks = tracker.update([])
        
        # Track should be removed after buffer expires
        assert len(tracker.active_tracks) == 0
        assert len(tracker.lost_tracks) == 0
        assert len(tracker.removed_tracks) == 1
        assert tracker.removed_tracks[0].track_id == track_id
    
    def test_two_stage_association(self):
        """Test two-stage association strategy"""
        config = TrackerConfig(
            track_thresh=0.5,
            match_thresh=0.8
        )
        tracker = ByteTrack(config)
        
        # Create initial track
        init_det = Detection(
            bbox=np.array([100, 100, 200, 200]),
            score=0.9,
            class_id=0,
            frame_id=0
        )
        tracker.update([init_det])
        
        # Next frame: provide high and low confidence detections
        high_conf = Detection(
            bbox=np.array([102, 102, 202, 202]),  # Close to previous
            score=0.9,
            class_id=0,
            frame_id=1
        )
        
        low_conf = Detection(
            bbox=np.array([300, 300, 400, 400]),  # Far from previous
            score=0.3,
            class_id=0,
            frame_id=1
        )
        
        # Update with both detections
        tracks = tracker.update([high_conf, low_conf])
        
        # Should have 2 tracks: one matched with high conf, one new from low conf
        assert len(tracks) == 2
        assert any(t.hits == 2 for t in tracks)  # One track matched twice
        assert any(t.hits == 1 for t in tracks)  # One new track


class TestLightPostTrackerWithGPS:
    """Test LightPostTracker with GPS integration"""
    
    def test_gps_integration(self):
        """Test GPS data integration with tracks"""
        config = TrackerConfig()
        detector = MockDetector(target_classes=['light_post'])
        tracker = LightPostTracker(config, detector)
        
        # Create sample frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create sample GPS data
        gps_data = GPSData(
            timestamp=1000.0,
            latitude=40.7128,
            longitude=-74.0060,
            altitude=10.0,
            heading=0.0
        )
        
        # Process frame with GPS
        tracks = tracker.process_frame(frame, 0, gps_data)
        
        # Detector should have created some tracks
        assert len(tracks) > 0
        
        # GPS data should be associated with tracks
        for track in tracks:
            assert track.track_id in tracker.gps_tracks
            assert len(tracker.gps_tracks[track.track_id]) == 1
            assert tracker.gps_tracks[track.track_id][0].latitude == 40.7128
    
    def test_location_estimation(self):
        """Test location estimation from GPS data"""
        config = TrackerConfig()
        detector = MockDetector(target_classes=['light_post'])
        tracker = LightPostTracker(config, detector)
        
        # Create sample frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create GPS data sequence with slight movement
        gps_sequence = [
            GPSData(timestamp=1000.0, latitude=40.7128, longitude=-74.0060, altitude=10.0, heading=0.0),
            GPSData(timestamp=1033.0, latitude=40.7129, longitude=-74.0061, altitude=10.0, heading=0.0),
            GPSData(timestamp=1066.0, latitude=40.7127, longitude=-74.0059, altitude=10.0, heading=0.0),
        ]
        
        # Process multiple frames with GPS
        for i, gps in enumerate(gps_sequence):
            tracker.process_frame(frame, i, gps)
        
        # Make some tracks static
        for track in tracker.tracker.active_tracks:
            # Mock static detection logic
            track.age = 10  # Enough frames to be considered static
            track.detections = [track.detections[0]] * max(3, len(track.detections))
        
        # Estimate locations
        locations = tracker.estimate_track_locations()
        
        # Check if locations were computed
        assert len(locations) > 0
        
        # Check location properties
        for track_id, location in locations.items():
            assert isinstance(location, GeoLocation)
            assert 40.7 < location.latitude < 40.8
            assert -74.1 < location.longitude < -74.0
            assert 0.0 <= location.reliability <= 1.0
    
    @pytest.mark.skipif(not os.path.exists('/tmp'), reason="Requires /tmp directory")
    def test_geojson_export(self):
        """Test exporting locations to GeoJSON"""
        config = TrackerConfig()
        detector = MockDetector(target_classes=['light_post'])
        tracker = LightPostTracker(config, detector)
        
        # Create some fake track locations
        tracker.track_locations = {
            1: GeoLocation(latitude=40.7128, longitude=-74.0060, reliability=0.9, accuracy=1.0),
            2: GeoLocation(latitude=40.7130, longitude=-74.0065, reliability=0.8, accuracy=2.0),
            3: GeoLocation(latitude=40.7135, longitude=-74.0070, reliability=0.6, accuracy=5.0)
        }
        
        # Export to GeoJSON
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            tracker.export_locations_to_geojson(output_path)
            
            # Verify file exists
            assert Path(output_path).exists()
            
            # Check content
            with open(output_path, 'r') as f:
                geojson = json.load(f)
            
            assert geojson['type'] == 'FeatureCollection'
            assert len(geojson['features']) == 3
            
            # Check coordinates
            for feature in geojson['features']:
                assert feature['type'] == 'Feature'
                assert feature['geometry']['type'] == 'Point'
                assert len(feature['geometry']['coordinates']) == 2
                assert 'track_id' in feature['properties']
                assert 'reliability' in feature['properties']
        
        finally:
            # Clean up
            if Path(output_path).exists():
                Path(output_path).unlink()


class TestVectorizedOperations:
    """Test vectorized operations for performance"""
    
    def test_batch_kalman_predict(self):
        """Test batch Kalman prediction"""
        from argus_track.filters import batch_predict_kalman
        
        # Create multiple detections
        detections = [
            Detection(bbox=np.array([100, 100, 200, 200]), score=0.9, class_id=0, frame_id=0),
            Detection(bbox=np.array([300, 300, 400, 400]), score=0.9, class_id=0, frame_id=0),
            Detection(bbox=np.array([500, 500, 600, 600]), score=0.9, class_id=0, frame_id=0)
        ]
        
        # Create Kalman trackers
        from argus_track.filters import KalmanBoxTracker
        trackers = [KalmanBoxTracker(det) for det in detections]
        
        # Test batch prediction
        predictions = batch_predict_kalman(trackers)
        
        # Should return array of predictions
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (3, 4)  # 3 trackers, 4 coordinates each
        
        # Check if predictions updated the trackers
        for tracker in trackers:
            assert tracker.age == 2
            assert tracker.time_since_update == 1
    
    def test_numba_iou_calculation(self):
        """Test numba-accelerated IoU calculation"""
        from argus_track.utils.iou import calculate_iou, calculate_iou_matrix_jit
        
        # Create bounding boxes
        bbox1 = np.array([100, 100, 200, 200])
        bbox2 = np.array([150, 150, 250, 250])
        
        # Calculate IoU
        iou = calculate_iou(bbox1, bbox2)
        assert 0.1 < iou < 0.2  # Roughly 1/9 overlap
        
        # Test with batches
        bboxes1 = np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400]
        ])
        
        bboxes2 = np.array([
            [150, 150, 250, 250],
            [350, 350, 450, 450]
        ])
        
        iou_matrix = calculate_iou_matrix_jit(bboxes1, bboxes2)
        assert iou_matrix.shape == (2, 2)
        assert 0.1 < iou_matrix[0, 0] < 0.2
        assert 0.1 < iou_matrix[1, 1] < 0.2
        assert iou_matrix[0, 1] == 0
        assert iou_matrix[1, 0] == 0


class TestErrorHandling:
    """Test error handling in the tracking system"""
    
    def test_invalid_video_path(self):
        """Test handling of invalid video path"""
        config = TrackerConfig()
        detector = MockDetector()
        tracker = LightPostTracker(config, detector)
        
        with pytest.raises(IOError):
            tracker.process_video("nonexistent_video.mp4")
    
    def test_gps_data_gaps(self):
        """Test handling of gaps in GPS data"""
        config = TrackerConfig()
        detector = MockDetector()
        tracker = LightPostTracker(config, detector)
        
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frames with alternating GPS (simulating gaps)
        for i in range(10):
            if i % 2 == 0:
                # Even frames have GPS
                gps = GPSData(
                    timestamp=1000.0 + i * 33.3,
                    latitude=40.7128 + i * 0.0001,
                    longitude=-74.0060 - i * 0.0001,
                    altitude=10.0,
                    heading=0.0
                )
                tracker.process_frame(frame, i, gps)
            else:
                # Odd frames don't have GPS
                tracker.process_frame(frame, i, None)
        
        # Should still have GPS data for tracks
        assert len(tracker.gps_tracks) > 0
        
        # Can still estimate locations
        locations = tracker.estimate_track_locations()
        assert len(locations) > 0