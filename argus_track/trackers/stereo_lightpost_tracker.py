# argus_track/trackers/stereo_lightpost_tracker.py (UPDATED)

"""Enhanced Stereo Light Post Tracker with Integrated GPS Extraction"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json

from ..config import TrackerConfig, StereoCalibrationConfig
from ..core import Detection, GPSData
from ..core.stereo import StereoDetection, StereoFrame, StereoTrack
from ..detectors import ObjectDetector
from ..stereo import StereoMatcher, StereoTriangulator, StereoCalibrationManager
from ..trackers import ByteTrack
from ..utils.visualization import draw_tracks
from ..utils.io import save_tracking_results
from ..utils.gps_utils import sync_gps_with_frames, GeoLocation
from ..utils.gps_extraction import extract_gps_from_stereo_videos, save_gps_to_csv


class EnhancedStereoLightPostTracker:
    """
    Enhanced stereo light post tracking system with integrated GPS extraction
    
    This class includes:
    1. Automatic GPS extraction from GoPro videos
    2. Stereo object detection and matching
    3. 3D triangulation for depth estimation
    4. Multi-object tracking with ByteTrack
    5. GPS data synchronization (every 6th frame)
    6. 3D to GPS coordinate transformation
    7. Static object analysis and geolocation
    """
    
    def __init__(self, 
                 config: TrackerConfig,
                 detector: ObjectDetector,
                 stereo_calibration: StereoCalibrationConfig):
        """
        Initialize enhanced stereo light post tracker
        
        Args:
            config: Tracker configuration
            detector: Object detection module
            stereo_calibration: Stereo camera calibration
        """
        self.config = config
        self.detector = detector
        self.logger = logging.getLogger(f"{__name__}.EnhancedStereoLightPostTracker")
        
        # Stereo processing components
        self.calibration_manager = StereoCalibrationManager(stereo_calibration)
        self.stereo_matcher = StereoMatcher(
            calibration=stereo_calibration,
            epipolar_threshold=16.0,
            iou_threshold=config.stereo_match_threshold
        )
        self.triangulator = StereoTriangulator(stereo_calibration)
        
        # Tracking components
        self.left_tracker = ByteTrack(config)
        self.right_tracker = ByteTrack(config)
        
        # Stereo tracking data
        self.stereo_tracks: Dict[int, StereoTrack] = {}
        self.stereo_frames: List[StereoFrame] = []
        self.track_id_counter = 0
        
        # GPS and geolocation
        self.gps_data_history: List[GPSData] = []
        self.estimated_locations: Dict[int, GeoLocation] = {}
        self.gps_extraction_method: str = 'none'
        
        # Performance monitoring
        self.processing_times = []
        self.frame_count = 0
        
        # Validate calibration
        is_valid, errors = self.calibration_manager.validate_calibration()
        if not is_valid:
            self.logger.warning(f"Calibration validation failed: {errors}")
        
        self.logger.info("Initialized enhanced stereo light post tracker")
        self.logger.info(f"Calibration: {self.calibration_manager.get_calibration_summary()}")
    
    def process_stereo_video_with_auto_gps(self, 
                                          left_video_path: str,
                                          right_video_path: str,
                                          output_path: Optional[str] = None,
                                          save_results: bool = True,
                                          gps_extraction_method: str = 'auto',
                                          save_extracted_gps: bool = True) -> Dict[int, StereoTrack]:
        """
        Process stereo video pair with automatic GPS extraction
        
        Args:
            left_video_path: Path to left camera video
            right_video_path: Path to right camera video
            output_path: Optional path for output video
            save_results: Whether to save tracking results
            gps_extraction_method: GPS extraction method ('auto', 'exiftool', 'gopro_api')
            save_extracted_gps: Whether to save extracted GPS data to CSV
            
        Returns:
            Dictionary of stereo tracks
        """
        self.logger.info("=== Enhanced Stereo Processing with GPS Extraction ===")
        self.logger.info(f"Left video: {left_video_path}")
        self.logger.info(f"Right video: {right_video_path}")
        
        # Step 1: Extract GPS data from videos
        self.logger.info("Step 1: Extracting GPS data from videos...")
        gps_data, method_used = extract_gps_from_stereo_videos(
            left_video_path, right_video_path, gps_extraction_method
        )
        
        self.gps_extraction_method = method_used
        
        if gps_data:
            self.logger.info(f"✅ Successfully extracted {len(gps_data)} GPS points using {method_used}")
            
            # Save extracted GPS data if requested
            if save_extracted_gps:
                gps_csv_path = Path(left_video_path).with_suffix('.csv')
                save_gps_to_csv(gps_data, str(gps_csv_path))
                self.logger.info(f"Saved GPS data to: {gps_csv_path}")
        else:
            self.logger.warning("⚠️  No GPS data extracted - proceeding without geolocation")
            gps_data = None
        
        # Step 2: Process stereo video with extracted GPS data
        self.logger.info("Step 2: Processing stereo video with tracking...")
        return self.process_stereo_video(
            left_video_path=left_video_path,
            right_video_path=right_video_path,
            gps_data=gps_data,
            output_path=output_path,
            save_results=save_results
        )
    
    def process_stereo_video(self, 
                            left_video_path: str,
                            right_video_path: str,
                            gps_data: Optional[List[GPSData]] = None,
                            output_path: Optional[str] = None,
                            save_results: bool = True) -> Dict[int, StereoTrack]:
        """
        Process stereo video pair with tracking and geolocation
        
        Args:
            left_video_path: Path to left camera video
            right_video_path: Path to right camera video
            gps_data: Optional GPS data synchronized with frames
            output_path: Optional path for output video
            save_results: Whether to save tracking results
            
        Returns:
            Dictionary of stereo tracks
        """
        self.logger.info(f"Processing stereo videos: {left_video_path}, {right_video_path}")
        
        # Open video captures
        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)
        
        if not left_cap.isOpened() or not right_cap.isOpened():
            error_msg = "Could not open one or both video files"
            self.logger.error(error_msg)
            raise IOError(error_msg)
        
        # Get video properties
        fps = left_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / fps
        
        self.logger.info(f"Video properties: {total_frames} frames, {fps} FPS, {width}x{height}")
        self.logger.info(f"Video duration: {video_duration:.1f} seconds")
        
        # Setup video writer if output requested
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Create side-by-side output
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        # Synchronize GPS data with frame rate if available
        if gps_data:
            from ..utils.gps_extraction import GoProGPSExtractor
            extractor = GoProGPSExtractor(fps_video=fps, fps_gps=10.0)
            
            # Synchronize GPS data to match video timeline
            gps_frame_data = extractor.synchronize_with_video(
                gps_data, video_duration, target_fps=10.0
            )
            self.logger.info(f"Synchronized {len(gps_frame_data)} GPS points with video timeline")
        else:
            gps_frame_data = []
        
        # Process frames
        processed_frames = 0
        
        try:
            while True:
                # Read frame pair
                left_ret, left_frame = left_cap.read()
                right_ret, right_frame = right_cap.read()
                
                if not left_ret or not right_ret:
                    break
                
                start_time = time.time()
                
                # Process every 6th frame (GPS synchronization strategy)
                if processed_frames % self.config.gps_frame_interval == 0:
                    gps_index = processed_frames // self.config.gps_frame_interval
                    current_gps = gps_frame_data[gps_index] if gps_index < len(gps_frame_data) else None
                    
                    # Process stereo frame
                    stereo_frame = self._process_stereo_frame_pair(
                        left_frame, right_frame, processed_frames, current_gps
                    )
                    
                    if stereo_frame:
                        self.stereo_frames.append(stereo_frame)
                        
                        # Update tracking
                        self._update_stereo_tracking(stereo_frame)
                
                # Visualize if output requested
                if out_writer:
                    vis_frame = self._create_stereo_visualization(left_frame, right_frame)
                    out_writer.write(vis_frame)
                
                # Performance monitoring
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                
                # Progress logging
                if processed_frames % 300 == 0:  # Every 10 seconds at 30fps
                    avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
                    progress = processed_frames / total_frames * 100
                    self.logger.info(
                        f"Processed {processed_frames}/{total_frames} frames "
                        f"({progress:.1f}%) Avg time: {avg_time*1000:.1f}ms"
                    )
                
                processed_frames += 1
                
        except Exception as e:
            self.logger.error(f"Error processing stereo video: {e}")
            raise
        finally:
            # Cleanup
            left_cap.release()
            right_cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
        
        # Post-processing: estimate locations for static tracks
        self._estimate_stereo_track_locations()
        
        # Save results if requested
        if save_results:
            self._save_enhanced_stereo_results(left_video_path, fps, width, height)
        
        self.logger.info(f"Processing complete. Tracked {len(self.stereo_tracks)} stereo objects")
        return self.stereo_tracks
    
    def _process_stereo_frame_pair(self, 
                                  left_frame: np.ndarray, 
                                  right_frame: np.ndarray,
                                  frame_id: int,
                                  gps_data: Optional[GPSData]) -> Optional[StereoFrame]:
        """Process a single stereo frame pair"""

        # Rectify images if calibration supports it
        if self.config.stereo_mode:
            left_rect, right_rect = self.calibration_manager.rectify_image_pair(
                left_frame, right_frame
            )
        else:
            left_rect, right_rect = left_frame, right_frame
        
        # Detect objects in both frames
        left_detections = self._detect_objects(left_rect, frame_id, 'left')
        
        right_detections = self._detect_objects(right_rect, frame_id, 'right')
        
        if not left_detections and not right_detections:
            return None
        
        # Match detections between left and right views
        stereo_detections = []
        if left_detections and right_detections:
            stereo_detections = self.stereo_matcher.match_detections(
                left_detections, right_detections
            )
            
            # Validate triangulation results
            valid_stereo_detections = []
            for stereo_det in stereo_detections:
                if self.triangulator.validate_triangulation(stereo_det):
                    valid_stereo_detections.append(stereo_det)
                else:
                    self.logger.debug(f"Invalid triangulation for detection at frame {frame_id}")
            
            stereo_detections = valid_stereo_detections
        
        # Create stereo frame
        stereo_frame = StereoFrame(
            frame_id=frame_id,
            timestamp=gps_data.timestamp if gps_data else frame_id / 30.0,  # Assume 30fps fallback
            left_frame=left_rect,
            right_frame=right_rect,
            left_detections=left_detections,
            right_detections=right_detections,
            stereo_detections=stereo_detections,
            gps_data=gps_data
        )
        
        return stereo_frame
    
    def _detect_objects(self, frame: np.ndarray, frame_id: int, camera: str) -> List[Detection]:
        """Detect objects in a single frame"""
        raw_detections = self.detector.detect(frame)
        
        detections = []
        for det in raw_detections:
            detection = Detection(
                bbox=np.array(det['bbox']),
                score=det['score'],
                class_id=det['class_id'],
                frame_id=frame_id
            )
            detections.append(detection)
        
        return detections
    
    def _update_stereo_tracking(self, stereo_frame: StereoFrame) -> None:
        """Update stereo tracking with new frame"""
        
        # Update individual camera trackers (for robustness)
        left_tracks = self.left_tracker.update(stereo_frame.left_detections)
        right_tracks = self.right_tracker.update(stereo_frame.right_detections)
        
        # Process stereo detections for 3D tracking
        for stereo_det in stereo_frame.stereo_detections:
            # Find corresponding tracks in left/right trackers
            left_track_id = self._find_matching_track(stereo_det.left_detection, left_tracks)
            right_track_id = self._find_matching_track(stereo_det.right_detection, right_tracks)
            
            if left_track_id is not None and right_track_id is not None:
                # Find existing stereo track or create new one
                stereo_track_id = self._get_or_create_stereo_track(left_track_id, right_track_id)
                
                if stereo_track_id in self.stereo_tracks:
                    # Update existing stereo track
                    stereo_track = self.stereo_tracks[stereo_track_id]
                    stereo_track.stereo_detections.append(stereo_det)
                    stereo_track.world_trajectory.append(stereo_det.world_coordinates)
                    
                    # Add GPS coordinate if available
                    if stereo_frame.gps_data:
                        # Transform to GPS coordinates
                        gps_locations = self.triangulator.world_to_gps_coordinates(
                            [stereo_det.world_coordinates], stereo_frame.gps_data
                        )
                        if gps_locations:
                            stereo_track.gps_trajectory.append(
                                np.array([gps_locations[0].latitude, gps_locations[0].longitude])
                            )
                    
                    # Update depth consistency
                    self._update_depth_consistency(stereo_track)
    
    def _find_matching_track(self, detection: Detection, tracks: List) -> Optional[int]:
        """Find track that matches the given detection"""
        best_track_id = None
        best_iou = 0.0
        
        for track in tracks:
            if track.last_detection:
                from ..utils.iou import calculate_iou
                iou = calculate_iou(detection.bbox, track.last_detection.bbox)
                if iou > best_iou and iou > 0.5:  # Minimum IoU threshold
                    best_iou = iou
                    best_track_id = track.track_id
        
        return best_track_id
    
    def _get_or_create_stereo_track(self, left_track_id: int, right_track_id: int) -> int:
        """Get existing stereo track or create new one"""
        # Look for existing stereo track that matches either left or right track
        for stereo_id, stereo_track in self.stereo_tracks.items():
            # For simplicity, use left track ID as primary identifier
            if stereo_id == left_track_id:
                return stereo_id
        
        # Create new stereo track
        stereo_track = StereoTrack(
            track_id=left_track_id,  # Use left track ID
            stereo_detections=[],
            world_trajectory=[],
            gps_trajectory=[]
        )
        
        self.stereo_tracks[left_track_id] = stereo_track
        return left_track_id
    
    def _update_depth_consistency(self, stereo_track: StereoTrack) -> None:
        """Update depth consistency metric for a stereo track"""
        if len(stereo_track.stereo_detections) < 3:
            return
        
        # Calculate depth variance over recent detections
        recent_depths = [det.depth for det in stereo_track.stereo_detections[-10:]]
        depth_std = np.std(recent_depths)
        
        # Consistency is inversely related to standard deviation
        stereo_track.depth_consistency = 1.0 / (1.0 + depth_std)
    
    def _estimate_stereo_track_locations(self) -> None:
        """Estimate final GPS locations for static stereo tracks"""
        for track_id, stereo_track in self.stereo_tracks.items():
            if stereo_track.is_static_3d and len(stereo_track.gps_trajectory) >= 3:
                # Get GPS history for this track
                gps_points = []
                for gps_coord in stereo_track.gps_trajectory:
                    # Convert back to GPSData format
                    gps_point = GPSData(
                        timestamp=0.0,  # Timestamp not needed for location estimation
                        latitude=gps_coord[0],
                        longitude=gps_coord[1],
                        altitude=0.0,
                        heading=0.0
                    )
                    gps_points.append(gps_point)
                
                # Estimate location using triangulator
                estimated_location = self.triangulator.estimate_object_location(
                    stereo_track, gps_points
                )
                
                if estimated_location:
                    stereo_track.estimated_location = estimated_location
                    self.estimated_locations[track_id] = estimated_location
                    
                    self.logger.debug(
                        f"Track {track_id} located at ({estimated_location.latitude:.6f}, {estimated_location.longitude:.6f}) "
                        f"reliability: {estimated_location.reliability:.2f}"
                    )
    
    def _create_stereo_visualization(self, 
                                   left_frame: np.ndarray, 
                                   right_frame: np.ndarray) -> np.ndarray:
        """Create side-by-side visualization of stereo tracking"""
        # Draw tracks on both frames
        left_vis = draw_tracks(left_frame, self.left_tracker.active_tracks)
        right_vis = draw_tracks(right_frame, self.right_tracker.active_tracks)
        
        # Create side-by-side visualization
        stereo_vis = np.hstack([left_vis, right_vis])
        
        # Add stereo information overlay
        self._add_stereo_info_overlay(stereo_vis)
        
        return stereo_vis
    
    def _add_stereo_info_overlay(self, stereo_frame: np.ndarray) -> None:
        """Add information overlay to stereo visualization"""
        # Add text information
        info_text = [
            f"Stereo Tracks: {len(self.stereo_tracks)}",
            f"GPS Method: {self.gps_extraction_method}",
            f"GPS Points: {len(self.gps_data_history)}",
            f"Locations: {len(self.estimated_locations)}",
            f"Frame: {self.frame_count}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(stereo_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
    
    def _save_enhanced_stereo_results(self, video_path: str, fps: float, width: int, height: int) -> None:
        """Save enhanced stereo tracking results with GPS extraction info"""
        results_path = Path(video_path).with_suffix('.json')
        
        # Prepare results data
        results = {
            'metadata': {
                'total_frames': len(self.stereo_frames),
                'fps': fps,
                'width': width,
                'height': height,
                'stereo_mode': self.config.stereo_mode,
                'gps_frame_interval': self.config.gps_frame_interval,
                'gps_extraction_method': self.gps_extraction_method,
                'gps_points_extracted': len(self.gps_data_history),
                'processing_times': {
                    'mean': np.mean(self.processing_times) if self.processing_times else 0,
                    'std': np.std(self.processing_times) if self.processing_times else 0,
                    'min': np.min(self.processing_times) if self.processing_times else 0,
                    'max': np.max(self.processing_times) if self.processing_times else 0
                }
            },
            'stereo_tracks': {
                str(track_id): track.to_dict() 
                for track_id, track in self.stereo_tracks.items()
            },
            'estimated_locations': {
                str(track_id): location.__dict__ 
                for track_id, location in self.estimated_locations.items()
            },
            'calibration_summary': self.calibration_manager.get_calibration_summary()
        }
        
        # Save to JSON
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved enhanced stereo tracking results to {results_path}")
        
        # Also save GeoJSON for mapping
        geojson_path = Path(video_path).with_suffix('.geojson')
        self._export_locations_to_geojson(geojson_path)
    
    def _export_locations_to_geojson(self, output_path: Path) -> None:
        """Export estimated locations to GeoJSON format"""
        features = []
        
        for track_id, location in self.estimated_locations.items():
            if location.reliability > 0.5:  # Only export reliable locations
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [location.longitude, location.latitude]
                    },
                    "properties": {
                        "track_id": track_id,
                        "reliability": location.reliability,
                        "accuracy": location.accuracy,
                        "method": "stereo_triangulation_with_auto_gps",
                        "gps_extraction_method": self.gps_extraction_method
                    }
                }
                features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generator": "Argus Track Enhanced Stereo Tracker",
                "gps_extraction_method": self.gps_extraction_method,
                "total_locations": len(features)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        self.logger.info(f"Exported {len(features)} locations to GeoJSON: {output_path}")
    
    def get_enhanced_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced stereo tracking statistics"""
        static_count = sum(1 for track in self.stereo_tracks.values() if track.is_static_3d)
        
        return {
            'total_stereo_tracks': len(self.stereo_tracks),
            'static_tracks': static_count,
            'estimated_locations': len(self.estimated_locations),
            'processed_frames': len(self.stereo_frames),
            'gps_extraction_method': self.gps_extraction_method,
            'gps_points_used': len(self.gps_data_history),
            'avg_depth': np.mean([track.average_depth for track in self.stereo_tracks.values()]) if self.stereo_tracks else 0,
            'avg_depth_consistency': np.mean([track.depth_consistency for track in self.stereo_tracks.values()]) if self.stereo_tracks else 0,
            'calibration_baseline': self.calibration_manager.calibration.baseline if self.calibration_manager.calibration else 0,
            'accuracy_achieved': np.mean([loc.accuracy for loc in self.estimated_locations.values()]) if self.estimated_locations else 0,
            'avg_reliability': np.mean([loc.reliability for loc in self.estimated_locations.values()]) if self.estimated_locations else 0
        }