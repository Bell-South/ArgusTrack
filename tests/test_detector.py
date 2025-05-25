"""Test module for detectors"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from argus_track.detectors import (
    ObjectDetector,
    YOLODetector,
    MockDetector
)


class TestMockDetector:
    """Test MockDetector class"""
    
    def test_mock_detector_creation(self):
        """Test creating mock detector"""
        detector = MockDetector(target_classes=['light_post'])
        
        assert 'light_post' in detector.target_classes
        assert detector.frame_count == 0
    
    def test_mock_detector_detection(self):
        """Test mock detection generation"""
        detector = MockDetector(target_classes=['light_post'])
        
        # Create dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Get detections
        detections = detector.detect(frame)
        
        assert len(detections) > 0
        
        for det in detections:
            assert 'bbox' in det
            assert 'score' in det
            assert 'class_name' in det
            assert 'class_id' in det
            assert det['class_name'] in detector.target_classes
            assert 0.7 <= det['score'] <= 1.0
    
    def test_mock_detector_consistency(self):
        """Test that mock detector produces consistent results"""
        detector = MockDetector(target_classes=['light_post'])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Get detections from multiple frames
        detections1 = detector.detect(frame)
        detections2 = detector.detect(frame)
        
        # Should produce similar but slightly different results
        assert len(detections1) == len(detections2)
        
        # Check that positions are slightly different (due to noise)
        for det1, det2 in zip(detections1, detections2):
            bbox1 = np.array(det1['bbox'])
            bbox2 = np.array(det2['bbox'])
            diff = np.abs(bbox1 - bbox2)
            assert np.all(diff < 20)  # Small movement


class TestYOLODetector:
    """Test YOLODetector class"""
    
    @patch('cv2.dnn.readNet')
    def test_yolo_detector_creation(self, mock_readnet):
        """Test creating YOLO detector"""
        # Mock cv2.dnn.readNet
        mock_net = MagicMock()
        mock_readnet.return_value = mock_net
        mock_net.getLayerNames.return_value = ['layer1', 'layer2', 'layer3']
        mock_net.getUnconnectedOutLayers.return_value = np.array([2, 3])
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = [
                'light_post\n', 'street_light\n'
            ]
            
            detector = YOLODetector(
                model_path='yolo.weights',
                config_path='yolo.cfg',
                target_classes=['light_post']
            )
        
        assert 'light_post' in detector.target_classes
        assert len(detector.class_names) == 2
        mock_readnet.assert_called_once()
    
    @patch('cv2.dnn.readNet')
    def test_yolo_detector_detection(self, mock_readnet):
        """Test YOLO detection process"""
        # Mock network
        mock_net = MagicMock()
        mock_readnet.return_value = mock_net
        mock_net.getLayerNames.return_value = ['layer1', 'layer2', 'layer3']
        mock_net.getUnconnectedOutLayers.return_value = np.array([2, 3])
        
        # Mock detection output
        mock_output = np.zeros((1, 85))  # YOLO output format
        mock_output[0, 0] = 0.5  # center x
        mock_output[0, 1] = 0.5  # center y
        mock_output[0, 2] = 0.1  # width
        mock_output[0, 3] = 0.2  # height
        mock_output[0, 4] = 0.9  # objectness
        mock_output[0, 5] = 0.95  # class 0 score (light_post)
        
        mock_net.forward.return_value = [mock_output]
        
        # Mock NMS
        with patch('cv2.dnn.NMSBoxes') as mock_nms:
            mock_nms.return_value = np.array([0])
            
            # Create detector
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.readlines.return_value = [
                    'light_post\n'
                ]
                
                detector = YOLODetector(
                    model_path='yolo.weights',
                    config_path='yolo.cfg'
                )
            
            # Test detection
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)
            
            assert len(detections) == 1
            assert detections[0]['class_name'] == 'light_post'
            assert detections[0]['score'] > 0.9
    
    def test_yolo_detector_backend(self):
        """Test setting YOLO backend"""
        with patch('cv2.dnn.readNet') as mock_readnet:
            mock_net = MagicMock()
            mock_readnet.return_value = mock_net
            mock_net.getLayerNames.return_value = ['layer']
            mock_net.getUnconnectedOutLayers.return_value = np.array([1])
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.readlines.return_value = []
                
                detector = YOLODetector(
                    model_path='yolo.weights',
                    config_path='yolo.cfg'
                )
                
                # Test setting backend
                detector.set_backend('cuda')
                mock_net.setPreferableBackend.assert_called()
                mock_net.setPreferableTarget.assert_called()


class TestObjectDetectorInterface:
    """Test ObjectDetector abstract interface"""
    
    def test_detector_interface(self):
        """Test that concrete detectors implement required methods"""
        # MockDetector should implement all required methods
        detector = MockDetector()
        
        assert hasattr(detector, 'detect')
        assert hasattr(detector, 'get_class_names')
        assert callable(detector.detect)
        assert callable(detector.get_class_names)
        
        # Test that methods work
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        class_names = detector.get_class_names()
        
        assert isinstance(detections, list)
        assert isinstance(class_names, list)