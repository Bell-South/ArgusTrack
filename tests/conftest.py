"""
Pytest configuration file
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_detection():
    """Provide sample detection for testing"""
    import numpy as np
    from argus_track.core import Detection
    
    return Detection(
        bbox=np.array([100, 200, 150, 300]),
        score=0.95,
        class_id=0,
        frame_id=1
    )


@pytest.fixture
def sample_track():
    """Provide sample track for testing"""
    from argus_track.core import Track
    
    return Track(
        track_id=1,
        state='confirmed',
        hits=5,
        age=10
    )


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing"""
    from argus_track import TrackerConfig
    
    return TrackerConfig(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=50
    )


@pytest.fixture
def mock_video_capture(monkeypatch):
    """Mock cv2.VideoCapture for testing"""
    import cv2
    import numpy as np
    
    class MockVideoCapture:
        def __init__(self, path):
            self.path = path
            self.frame_count = 100
            self.current_frame = 0
            
        def read(self):
            if self.current_frame < self.frame_count:
                self.current_frame += 1
                return True, np.zeros((720, 1280, 3), dtype=np.uint8)
            return False, None
            
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            elif prop == cv2.CAP_PROP_FRAME_COUNT:
                return self.frame_count
            elif prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            return 0
            
        def release(self):
            pass
    
    monkeypatch.setattr('cv2.VideoCapture', MockVideoCapture)
    return MockVideoCapture 