"""Test module for core data structures"""

import pytest
import numpy as np

from argus_track.core import Detection, Track, GPSData


class TestDetection:
    """Test Detection class"""
    
    def test_detection_creation(self):
        """Test creating a detection"""
        det = Detection(
            bbox=np.array([100, 200, 150, 300]),
            score=0.95,
            class_id=0,
            frame_id=1
        )
        
        assert det.score == 0.95
        assert det.class_id == 0
        assert det.frame_id == 1
        np.testing.assert_array_equal(det.bbox, [100, 200, 150, 300])
    
    def test_detection_properties(self):
        """Test detection property calculations"""
        det = Detection(
            bbox=np.array([100, 200, 200, 400]),
            score=0.9,
            class_id=0,
            frame_id=1
        )
        
        # Test tlbr property
        np.testing.assert_array_equal(det.tlbr, [100, 200, 200, 400])
        
        # Test xywh property
        xywh = det.xywh
        assert xywh[0] == 150  # center x
        assert xywh[1] == 300  # center y
        assert xywh[2] == 100  # width
        assert xywh[3] == 200  # height
        
        # Test area
        assert det.area == 20000
        
        # Test center
        center = det.center
        assert center[0] == 150
        assert center[1] == 300
    
    def test_detection_serialization(self):
        """Test detection to/from dict"""
        det = Detection(
            bbox=np.array([10, 20, 30, 40]),
            score=0.85,
            class_id=1,
            frame_id=5
        )
        
        # To dict
        det_dict = det.to_dict()
        assert det_dict['bbox'] == [10, 20, 30, 40]
        assert det_dict['score'] == 0.85
        assert det_dict['class_id'] == 1
        assert det_dict['frame_id'] == 5
        
        # From dict
        det2 = Detection.from_dict(det_dict)
        assert det2.score == det.score
        assert det2.class_id == det.class_id
        np.testing.assert_array_equal(det2.bbox, det.bbox)


class TestTrack:
    """Test Track class"""
    
    def test_track_creation(self):
        """Test creating a track"""
        track = Track(track_id=1)
        
        assert track.track_id == 1
        assert track.state == 'tentative'
        assert track.hits == 0
        assert track.age == 0
        assert len(track.detections) == 0
    
    def test_track_properties(self):
        """Test track properties"""
        track = Track(track_id=1, state='confirmed')
        
        # Test is_confirmed
        assert track.is_confirmed is True
        
        # Test is_active
        assert track.is_active is True
        
        track.state = 'lost'
        assert track.is_confirmed is False
        assert track.is_active is False
    
    def test_track_with_detections(self):
        """Test track with detections"""
        det1 = Detection(
            bbox=np.array([10, 20, 30, 40]),
            score=0.9,
            class_id=0,
            frame_id=1
        )
        det2 = Detection(
            bbox=np.array([12, 22, 32, 42]),
            score=0.85,
            class_id=0,
            frame_id=2
        )
        
        track = Track(track_id=1, detections=[det1, det2])
        
        assert len(track.detections) == 2
        assert track.last_detection == det2
        
        # Test trajectory
        trajectory = track.trajectory
        assert len(trajectory) == 2
        np.testing.assert_array_equal(trajectory[0], [20, 30])
        np.testing.assert_array_equal(trajectory[1], [22, 32])
    
    def test_track_to_tlbr(self):
        """Test getting track bounding box"""
        det = Detection(
            bbox=np.array([100, 200, 150, 300]),
            score=0.9,
            class_id=0,
            frame_id=1
        )
        
        track = Track(track_id=1, detections=[det])
        bbox = track.to_tlbr()
        np.testing.assert_array_equal(bbox, [100, 200, 150, 300])


class TestGPSData:
    """Test GPSData class"""
    
    def test_gps_creation(self):
        """Test creating GPS data"""
        gps = GPSData(
            timestamp=1234567890.0,
            latitude=40.7128,
            longitude=-74.0060,
            altitude=10.5,
            heading=45.0,
            accuracy=1.5
        )
        
        assert gps.timestamp == 1234567890.0
        assert gps.latitude == 40.7128
        assert gps.longitude == -74.0060
        assert gps.altitude == 10.5
        assert gps.heading == 45.0
        assert gps.accuracy == 1.5
    
    def test_gps_serialization(self):
        """Test GPS to/from dict"""
        gps = GPSData(
            timestamp=1234567890.0,
            latitude=40.7128,
            longitude=-74.0060,
            altitude=10.5,
            heading=45.0
        )
        
        # To dict
        gps_dict = gps.to_dict()
        assert gps_dict['timestamp'] == 1234567890.0
        assert gps_dict['latitude'] == 40.7128
        assert gps_dict['accuracy'] == 1.0  # default value
        
        # From dict
        gps2 = GPSData.from_dict(gps_dict)
        assert gps2.timestamp == gps.timestamp
        assert gps2.latitude == gps.latitude
        assert gps2.longitude == gps.longitude
    
    def test_gps_from_csv(self):
        """Test creating GPS from CSV line"""
        line = "1234567890.0,40.7128,-74.0060,10.5,45.0,2.0"
        gps = GPSData.from_csv_line(line)
        
        assert gps.timestamp == 1234567890.0
        assert gps.latitude == 40.7128
        assert gps.longitude == -74.0060
        assert gps.altitude == 10.5
        assert gps.heading == 45.0
        assert gps.accuracy == 2.0
        
        # Test without accuracy
        line2 = "1234567890.0,40.7128,-74.0060,10.5,45.0"
        gps2 = GPSData.from_csv_line(line2)
        assert gps2.accuracy == 1.0  # default
        
        # Test invalid line
        with pytest.raises(ValueError):
            GPSData.from_csv_line("invalid,data")