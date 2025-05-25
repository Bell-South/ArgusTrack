# argus_track/trackers/lightpost_tracker.py

"""Light Post Tracker with GPS integration"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import cv2

from ..config import TrackerConfig, CameraConfig
from ..core import Detection, Track, GPSData
from ..detectors import ObjectDetector
from .bytetrack import ByteTrack
from ..utils.visualization import draw_tracks, create_track_overlay
from ..utils.io import save_tracking_results, load_gps_data
from ..utils.gps_utils import compute_average_location, filter_gps_outliers, GeoLocation


class LightPostTracker:
    """
    Complete light post tracking system with GPS integration
    
    This class orchestrates the entire tracking pipeline:
    1. Object detection using configurable detector
    2. Multi-object tracking with ByteTrack
    3. GPS data synchronization
    4. Geolocation estimation
    5. Results visualization and saving
    """
    
    def __init__(self, config: TrackerConfig, 
                 detector: ObjectDetector,
                 camera_config: Optional[CameraConfig] = None):
        """
        Initialize light post tracker
        
        Args:
            config: Tracker configuration
            detector: Object detection module
            camera_config: Camera calibration configuration
        """
        self.config = config
        self.detector = detector
        self.tracker = ByteTrack(config)
        self.logger = logging.getLogger(f"{__name__}.LightPostTracker")
        
        # Camera calibration
        self.camera_config = camera_config
        
        # GPS tracking
        self.gps_tracks: Dict[int, List[GPSData]] = {}
        
        # Track locations
        self.track_locations: Dict[int, GeoLocation] = {}
        
        # Performance monitoring
        self.processing_times = []
        
    def process_video(self, video_path: str, 
                     gps_data: Optional[List[GPSData]] = None,
                     output_path: Optional[str] = None,
                     save_results: bool = True) -> Dict[int, List[Dict]]:
        """
        Process complete video with tracking
        
        Args:
            video_path: Path to input video
            gps_data: Optional GPS data synchronized with frames
            output_path: Optional path for output video
            save_results: Whether to save tracking results
            
        Returns:
            Dictionary of tracks with their complete history
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video: {video_path}"
            self.logger.error(error_msg)
            raise IOError(error_msg)
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        all_tracks = {}
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Detect objects
                raw_detections = self.detector.detect(frame)
                
                # Convert to Detection objects
                detections = []
                for i, det in enumerate(raw_detections):
                    detections.append(Detection(
                        bbox=np.array(det['bbox']),
                        score=det['score'],
                        class_id=det['class_id'],
                        frame_id=frame_idx
                    ))
                
                # Update tracker - use batched Kalman prediction for efficiency
                tracks = self.tracker.update(detections)
                
                # Update GPS data if available
                if gps_data and frame_idx < len(gps_data):
                    self._update_gps_tracks(tracks, gps_data[frame_idx])
                
                # Store track data
                for track in tracks:
                    if track.track_id not in all_tracks:
                        all_tracks[track.track_id] = []
                    
                    all_tracks[track.track_id].append({
                        'frame': frame_idx,
                        'bbox': track.to_tlbr().tolist(),
                        'score': track.detections[-1].score if track.detections else 0,
                        'state': track.state,
                        'hits': track.hits
                    })
                
                # Visualize if output requested
                if out_writer:
                    vis_frame = draw_tracks(frame, tracks)
                    out_writer.write(vis_frame)
                
                # Performance monitoring
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                
                # Progress logging
                if frame_idx % 100 == 0:
                    avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
                    self.logger.info(
                        f"Processed {frame_idx}/{frame_count} frames "
                        f"({frame_idx/frame_count*100:.1f}%) "
                        f"Avg time: {avg_time*1000:.1f}ms"
                    )
                
                frame_idx += 1
                
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
            
        finally:
            # Cleanup
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
        
        # Estimate geolocations for static tracks
        if gps_data:
            self.estimate_track_locations()
        
        # Save results if requested
        if save_results:
            results_path = Path(video_path).with_suffix('.json')
            save_tracking_results(
                all_tracks, 
                results_path,
                metadata={
                    'total_frames': frame_idx,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'config': self.config.__dict__,
                    'processing_times': {
                        'mean': np.mean(self.processing_times) if self.processing_times else 0,
                        'std': np.std(self.processing_times) if self.processing_times else 0,
                        'min': np.min(self.processing_times) if self.processing_times else 0,
                        'max': np.max(self.processing_times) if self.processing_times else 0
                    }
                },
                gps_tracks=self.gps_tracks,
                track_locations={
                    k: v.__dict__ for k, v in self.track_locations.items()
                }
            )
        
        self.logger.info(f"Processing complete. Tracked {len(all_tracks)} objects")
        return all_tracks
    
    def process_frame(self, frame: np.ndarray, frame_idx: int,
                     gps_data: Optional[GPSData] = None) -> List[Track]:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            gps_data: Optional GPS data for this frame
            
        Returns:
            List of active tracks
        """
        # Detect objects
        raw_detections = self.detector.detect(frame)
        
        # Convert to Detection objects
        detections = []
        for det in raw_detections:
            detections.append(Detection(
                bbox=np.array(det['bbox']),
                score=det['score'],
                class_id=det['class_id'],
                frame_id=frame_idx
            ))
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Update GPS data if available
        if gps_data:
            self._update_gps_tracks(tracks, gps_data)
        
        return tracks
    
    def _update_gps_tracks(self, tracks: List[Track], gps_data: GPSData) -> None:
        """
        Update GPS data for active tracks
        
        Args:
            tracks: Current active tracks
            gps_data: GPS data for current frame
        """
        for track in tracks:
            if track.track_id not in self.gps_tracks:
                self.gps_tracks[track.track_id] = []
            self.gps_tracks[track.track_id].append(gps_data)
    
    def estimate_track_locations(self) -> Dict[int, GeoLocation]:
        """
        Estimate geolocation for all tracks
        
        Returns:
            Dictionary mapping track_id to GeoLocation
        """
        static_objects = self.analyze_static_objects()
        
        for track_id, is_static in static_objects.items():
            # Only compute locations for static objects
            if not is_static or track_id not in self.gps_tracks:
                continue
            
            gps_points = self.gps_tracks[track_id]
            
            # Filter outliers
            filtered_points = filter_gps_outliers(gps_points)
            
            # Compute average location
            location = compute_average_location(filtered_points)
            
            self.track_locations[track_id] = location
            
            self.logger.debug(
                f"Track {track_id} located at ({location.latitude:.6f}, {location.longitude:.6f}) "
                f"reliability: {location.reliability:.2f}"
            )
        
        return self.track_locations
    
    def get_static_locations(self) -> Dict[int, GeoLocation]:
        """
        Get geolocations of all static objects
        
        Returns:
            Dictionary mapping track_id to GeoLocation
        """
        # Ensure locations are estimated
        if not self.track_locations:
            self.estimate_track_locations()
            
        return {k: v for k, v in self.track_locations.items() if v.reliability > 0.5}
    
    def analyze_static_objects(self) -> Dict[int, bool]:
        """
        Analyze which tracked objects are static
        
        Returns:
            Dictionary mapping track_id to is_static boolean
        """
        static_analysis = {}
        
        for track_id, track in self.tracker.tracks.items():
            if len(track.detections) < self.config.min_static_frames:
                static_analysis[track_id] = False
                continue
            
            # Calculate movement over recent frames
            recent_detections = track.detections[-self.config.min_static_frames:]
            positions = np.array([det.xywh[:2] for det in recent_detections])
            
            # Calculate standard deviation of positions
            movement = np.std(positions, axis=0)
            
            # Check if movement is below threshold
            is_static = np.all(movement < self.config.static_threshold)
            static_analysis[track_id] = is_static
            
            if is_static:
                self.logger.debug(f"Track {track_id} identified as static object")
        
        return static_analysis
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        tracks = self.tracker.get_all_tracks()
        
        return {
            'total_tracks': len(tracks),
            'active_tracks': len(self.tracker.active_tracks),
            'lost_tracks': len(self.tracker.lost_tracks),
            'removed_tracks': len(self.tracker.removed_tracks),
            'total_frames': self.tracker.frame_id,
            'avg_track_length': np.mean([track.age for track in tracks.values()]) if tracks else 0,
            'static_objects': sum(1 for is_static in self.analyze_static_objects().values() if is_static),
            'located_objects': len(self.track_locations)
        }
    
    def export_locations_to_geojson(self, output_path: str) -> None:
        """
        Export static object locations to GeoJSON format
        
        Args:
            output_path: Path to output GeoJSON file
        """
        if not self.track_locations:
            self.estimate_track_locations()
            
        # Filter to only include reliable locations
        reliable_locations = {k: v for k, v in self.track_locations.items() 
                             if v.reliability > 0.5}
            
        features = []
        for track_id, location in reliable_locations.items():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [location.longitude, location.latitude]
                },
                "properties": {
                    "track_id": track_id,
                    "reliability": location.reliability,
                    "accuracy": location.accuracy
                }
            }
            features.append(feature)
            
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
            
        self.logger.info(f"Exported {len(features)} locations to GeoJSON: {output_path}")