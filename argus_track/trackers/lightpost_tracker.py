"""Light Post Tracker with Enhanced GPS integration and Real-time Display"""

import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from ..config import TrackerConfig, CameraConfig
from ..core import Detection, Track, GPSData
from ..detectors import ObjectDetector
from .bytetrack import ByteTrack
from ..utils.visualization import draw_tracks, create_track_overlay, RealTimeVisualizer
from ..utils.io import save_tracking_results, load_gps_data
from ..utils.gps_utils import compute_average_location, filter_gps_outliers, GeoLocation, CoordinateTransformer


class EnhancedLightPostTracker:
    """
    Enhanced light post tracking system with GPS-based geolocation calculation and real-time display
    """
    
    def __init__(self, config: TrackerConfig, 
                 detector: ObjectDetector,
                 camera_config: Optional[CameraConfig] = None,
                 show_realtime: bool = True):
        """
        Initialize enhanced light post tracker
        
        Args:
            config: Tracker configuration
            detector: Object detection module
            camera_config: Camera calibration configuration
            show_realtime: Whether to show real-time visualization
        """
        self.config = config
        self.detector = detector
        self.tracker = ByteTrack(config)
        self.logger = logging.getLogger(f"{__name__}.EnhancedLightPostTracker")
        
        # Real-time visualization
        self.show_realtime = show_realtime
        self.visualizer = None
        
        # Camera parameters for distance estimation
        self.camera_config = camera_config
        self.focal_length_px = 1400  # Default GoPro approximation
        self.image_width = 2704
        self.image_height = 2028
        self.camera_height = 1.5  # Estimated camera height in meters
        
        # GPS tracking and geolocation
        self.gps_tracks: Dict[int, List[GPSData]] = {}
        self.frame_to_gps: Dict[int, GPSData] = {}
        self.track_locations: Dict[int, GeoLocation] = {}
        
        # Performance monitoring
        self.processing_times = []
        
        self.logger.info("Initialized enhanced light post tracker with GPS geolocation")
        if self.show_realtime:
            self.logger.info("Real-time visualization enabled")
        
    def process_video(self, video_path: str, 
                     gps_data: Optional[List[GPSData]] = None,
                     output_path: Optional[str] = None,
                     save_results: bool = True) -> Dict[int, List[Dict]]:
        """
        Process complete video with tracking, GPS-based geolocation, and real-time display
        
        Args:
            video_path: Path to input video
            gps_data: GPS data synchronized with frames
            output_path: Optional path for output video
            save_results: Whether to save tracking results
            
        Returns:
            Dictionary of tracks with their complete history
        """
        self.logger.info(f"Processing video with GPS geolocation: {video_path}")
        
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
        
        # Update camera parameters
        self.image_width = width
        self.image_height = height
        
        # Initialize real-time visualizer
        if self.show_realtime:
            self.visualizer = RealTimeVisualizer(
                window_name="Argus Track - LED Detection",
                display_size=(1280, 720),
                show_info_panel=True
            )
        
        # Setup video writer if output path provided
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Synchronize GPS data with frames
        if gps_data:
            self._synchronize_gps_with_frames(gps_data, frame_count, fps)
            self.logger.info(f"Synchronized GPS data with {frame_count} video frames")
        
        # Process frames
        all_tracks = {}
        frame_idx = 0
        user_quit = False
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                print(f"[DEBUG] Frame index {frame_idx}, ret: {ret}, frame is None: {frame is None}, shape: {getattr(frame, 'shape', None)}")
                if not ret or frame is None:
                    print("No frame, breaking.")
                    break
                # if not ret:
                #     break
                    # Save first frame for manual check
                if frame_idx == 0:
                    cv2.imwrite("first_frame_debug.jpg", frame)
                start_time = time.time()
                
                # Get current GPS data
                current_gps = self.frame_to_gps.get(frame_idx) if frame_idx in self.frame_to_gps else None
                
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
                
                # Update tracker
                tracks = self.tracker.update(detections)
                
                # Update GPS data for active tracks (if GPS available)
                if current_gps:
                    self._update_gps_tracks(tracks, current_gps, frame_idx)
                
                # Store track data
                for track in tracks:
                    if track.track_id not in all_tracks:
                        all_tracks[track.track_id] = []
                    
                    # Enhanced track data with GPS frame info
                    track_data = {
                        'frame': frame_idx,
                        'bbox': track.to_tlbr().tolist(),
                        'score': track.detections[-1].score if track.detections else 0,
                        'state': track.state,
                        'hits': track.hits,
                        'has_gps': current_gps is not None,
                        'timestamp': current_gps.timestamp if current_gps else frame_idx / fps
                    }
                    all_tracks[track.track_id].append(track_data)
                
                # Real-time visualization
                if self.show_realtime and self.visualizer:
                    gps_info = None
                    if current_gps:
                        gps_info = {
                            'latitude': current_gps.latitude,
                            'longitude': current_gps.longitude,
                            'heading': current_gps.heading
                        }
                    
                    frame_info = {
                        'frame_idx': frame_idx,
                        'total_frames': frame_count,
                        'fps': fps
                    }
                    
                    # Show real-time visualization
                    should_continue = self.visualizer.visualize_frame(
                        frame, detections, tracks, gps_info, frame_info
                    )
                    frame_idx += 1
                    if not should_continue:
                        user_quit = True
                        self.logger.info("User requested quit from visualization")
                        break
                
                # Save to output video if requested
                if out_writer:
                    vis_frame = draw_tracks(frame, tracks)
                    # Add GPS info overlay
                    if current_gps:
                        self._add_gps_overlay(vis_frame, current_gps, frame_idx)
                    out_writer.write(vis_frame)
                
                # Performance monitoring
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                
                # Progress logging (less frequent when showing real-time)
                log_interval = 600 if self.show_realtime else 300  # Every 20s vs 10s
                if frame_idx % log_interval == 0:
                    avg_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
                    progress = frame_idx / frame_count * 100
                    self.logger.info(
                        f"Processed {frame_idx}/{frame_count} frames "
                        f"({progress:.1f}%) Avg time: {avg_time*1000:.1f}ms"
                    )
                
                frame_idx += 1
                
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user (Ctrl+C)")
            user_quit = True
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
            
        finally:
            # Cleanup
            cap.release()
            if out_writer:
                out_writer.release()
            if self.visualizer:
                self.visualizer.close()
            cv2.destroyAllWindows()
        
        if user_quit:
            self.logger.info("Processing stopped by user request")
        else:
            self.logger.info(f"Processing completed successfully")
        
        # Calculate geolocations for tracked objects
        if gps_data:
            self.logger.info("Calculating object geolocations...")
            self._calculate_object_geolocations(all_tracks)
        
        # Save results if requested
        if save_results:
            results_path = Path(video_path).with_suffix('.json')
            geojson_path = Path(video_path).with_suffix('.geojson')
            
            self._save_enhanced_results(
                all_tracks, results_path, geojson_path,
                metadata={
                    'total_frames': frame_idx,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'config': self.config.__dict__,
                    'has_gps_data': gps_data is not None,
                    'gps_points_used': len(gps_data) if gps_data else 0,
                    'geolocated_objects': len(self.track_locations),
                    'user_quit': user_quit,
                    'processing_times': {
                        'mean': np.mean(self.processing_times) if self.processing_times else 0,
                        'std': np.std(self.processing_times) if self.processing_times else 0,
                        'min': np.min(self.processing_times) if self.processing_times else 0,
                        'max': np.max(self.processing_times) if self.processing_times else 0
                    }
                }
            )
        
        processed_tracks = len([t for t in all_tracks.values() if len(t) >= 3])
        geolocated_count = len(self.track_locations)
        
        self.logger.info(
            f"Final results: {processed_tracks} well-tracked objects, "
            f"{geolocated_count} geolocated static objects"
        )
        
        return all_tracks
    
    def _synchronize_gps_with_frames(self, gps_data: List[GPSData], 
                                    total_frames: int, fps: float) -> None:
        """Synchronize GPS data with video frames"""
        if not gps_data:
            return
        
        # Convert GPS timestamps to relative time
        start_time = gps_data[0].timestamp
        gps_times = [(gps.timestamp - start_time) for gps in gps_data]
        
        # Map each frame to closest GPS point
        for frame in range(total_frames):
            frame_time = frame / fps
            
            # Find closest GPS point
            if gps_times:
                closest_idx = min(range(len(gps_times)), 
                                 key=lambda i: abs(gps_times[i] - frame_time))
                self.frame_to_gps[frame] = gps_data[closest_idx]
    
    def _update_gps_tracks(self, tracks: List[Track], gps_data: GPSData, frame_idx: int) -> None:
        """Update GPS data for active tracks"""
        for track in tracks:
            if track.track_id not in self.gps_tracks:
                self.gps_tracks[track.track_id] = []
            
            # Only add GPS data every few frames to avoid redundancy
            if frame_idx % self.config.gps_frame_interval == 0:
                self.gps_tracks[track.track_id].append(gps_data)
    
    def _calculate_object_geolocations(self, all_tracks: Dict[int, List[Dict]]) -> None:
        """Calculate real-world geolocations for tracked objects"""
        
        for track_id, track_history in all_tracks.items():
            # Filter for static, well-tracked objects
            if len(track_history) < self.config.min_static_frames:
                continue
            
            # Check if track has GPS data
            gps_frames = [t for t in track_history if t.get('has_gps', False)]
            if len(gps_frames) < 3:
                continue
            
            # Analyze if object is static
            if not self._is_static_track(track_history):
                continue
            
            # Calculate geolocation
            geolocation = self._estimate_object_geolocation(track_id, track_history)
            
            if geolocation and geolocation.reliability > 0.3:
                self.track_locations[track_id] = geolocation
                self.logger.debug(
                    f"Track {track_id} geolocated: ({geolocation.latitude:.6f}, {geolocation.longitude:.6f}) "
                    f"reliability: {geolocation.reliability:.2f}"
                )
    
    def _is_static_track(self, track_history: List[Dict]) -> bool:
        """Determine if a track represents a static object"""
        if len(track_history) < 5:
            return False
        
        # Calculate movement of bounding box centers
        centers = []
        for detection in track_history:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
        
        centers = np.array(centers)
        
        # Calculate standard deviation of movement
        std_movement = np.std(centers, axis=0)
        max_movement = np.max(std_movement)
        
        # Object is static if movement is below threshold
        return max_movement < self.config.static_threshold
    
    def _estimate_object_geolocation(self, track_id: int, 
                                   track_history: List[Dict]) -> Optional[GeoLocation]:
        """Estimate geolocation for a single tracked object"""
        
        # Get frames with GPS data
        gps_frames = [t for t in track_history if t.get('has_gps', False)]
        
        if len(gps_frames) < 3:
            return None
        
        estimated_positions = []
        
        for detection in gps_frames:
            # Get GPS data for this frame
            frame_idx = detection['frame']
            if frame_idx not in self.frame_to_gps:
                continue
            
            gps_point = self.frame_to_gps[frame_idx]
            bbox = detection['bbox']
            
            # Estimate distance to object
            distance = self._estimate_object_distance(bbox)
            
            # Calculate object offset from camera
            lateral_offset, forward_offset = self._calculate_object_offset(bbox, distance)
            
            # Convert to GPS coordinates
            obj_lat, obj_lon = self._gps_offset_to_coordinates(
                gps_point, lateral_offset, forward_offset
            )
            
            estimated_positions.append({
                'lat': obj_lat,
                'lon': obj_lon,
                'distance': distance,
                'confidence': detection['score'],
                'frame': frame_idx
            })
        
        if not estimated_positions:
            return None
        
        # Calculate average position (static object assumption)
        avg_lat = np.mean([p['lat'] for p in estimated_positions])
        avg_lon = np.mean([p['lon'] for p in estimated_positions])
        avg_confidence = np.mean([p['confidence'] for p in estimated_positions])
        
        # Calculate reliability based on position consistency
        lat_std = np.std([p['lat'] for p in estimated_positions])
        lon_std = np.std([p['lon'] for p in estimated_positions])
        position_std = np.sqrt(lat_std**2 + lon_std**2)
        
        # Convert position standard deviation to reliability score
        reliability = 1.0 / (1.0 + position_std * 10000)
        
        # Estimate accuracy (in meters)
        earth_radius = 6378137.0
        lat_error = lat_std * earth_radius * np.pi / 180
        lon_error = lon_std * earth_radius * np.pi / 180 * np.cos(np.radians(avg_lat))
        accuracy = np.sqrt(lat_error**2 + lon_error**2)
        
        return GeoLocation(
            latitude=avg_lat,
            longitude=avg_lon,
            accuracy=max(1.0, accuracy),
            reliability=reliability,
            timestamp=estimated_positions[-1]['frame']
        )
    
    def _estimate_object_distance(self, bbox: List[float]) -> float:
        """Estimate distance to object based on bounding box size"""
        real_object_width = 0.3  # meters
        bbox_width = bbox[2] - bbox[0]
        
        if bbox_width > 0:
            distance = (real_object_width * self.focal_length_px) / bbox_width
            return max(1.0, min(distance, 200.0))
        
        return 50.0
    
    def _calculate_object_offset(self, bbox: List[float], distance: float) -> Tuple[float, float]:
        """Calculate object offset from camera position"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        dx_px = center_x - (self.image_width / 2)
        dy_px = center_y - (self.image_height / 2)
        
        angle_per_pixel = 0.0005
        angle_x = dx_px * angle_per_pixel
        angle_y = dy_px * angle_per_pixel
        
        lateral_offset = distance * np.sin(angle_x)
        forward_offset = distance * np.cos(angle_x)
        
        return lateral_offset, forward_offset
    
    def _gps_offset_to_coordinates(self, base_gps: GPSData, 
                                  lateral_offset: float, 
                                  forward_offset: float) -> Tuple[float, float]:
        """Convert local offsets to GPS coordinates"""
        R = 6378137.0
        heading_rad = np.radians(base_gps.heading)
        
        east_offset = lateral_offset * np.cos(heading_rad) + forward_offset * np.sin(heading_rad)
        north_offset = -lateral_offset * np.sin(heading_rad) + forward_offset * np.cos(heading_rad)
        
        lat_offset = north_offset / R * 180 / np.pi
        lon_offset = east_offset / (R * np.cos(np.radians(base_gps.latitude))) * 180 / np.pi
        
        object_lat = base_gps.latitude + lat_offset
        object_lon = base_gps.longitude + lon_offset
        
        return object_lat, object_lon
    
    def _add_gps_overlay(self, frame: np.ndarray, gps_data: GPSData, frame_idx: int) -> None:
        """Add GPS information overlay to frame"""
        overlay_text = [
            f"Frame: {frame_idx}",
            f"GPS: {gps_data.latitude:.6f}, {gps_data.longitude:.6f}",
            f"Heading: {gps_data.heading:.1f}°",
            f"Objects tracked: {len(self.tracker.active_tracks)}"
        ]
        
        y_offset = 30
        for text in overlay_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
    
    def _save_enhanced_results(self, all_tracks: Dict[int, List[Dict]], 
                              json_path: Path, geojson_path: Path,
                              metadata: Dict[str, Any]) -> None:
        """Save enhanced results with geolocations"""
        
        # Save standard JSON results
        results = {
            'metadata': metadata,
            'tracks': all_tracks,
            'gps_tracks': {
                str(track_id): [gps.to_dict() for gps in gps_list]
                for track_id, gps_list in self.gps_tracks.items()
            },
            'track_locations': {
                str(track_id): {
                    'latitude': loc.latitude,
                    'longitude': loc.longitude,
                    'accuracy': loc.accuracy,
                    'reliability': loc.reliability,
                    'timestamp': loc.timestamp
                }
                for track_id, loc in self.track_locations.items()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved tracking results to {json_path}")
        
        # Save GeoJSON with geolocated objects
        self._export_geojson(geojson_path)
    
    def _export_geojson(self, output_path: Path, min_reliability: float = 0.3) -> None:
        """Export geolocated objects to GeoJSON format"""
        
        reliable_locations = {
            track_id: location for track_id, location in self.track_locations.items()
            if location.reliability >= min_reliability
        }
        
        features = []
        
        for track_id, location in reliable_locations.items():
            class_name = f"Led-{150 if track_id % 2 == 0 else 240}"
            track_info = self._get_track_summary(track_id)
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [location.longitude, location.latitude]
                },
                "properties": {
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": track_info.get('avg_confidence', 0.0),
                    "reliability": round(location.reliability, 3),
                    "accuracy_meters": round(location.accuracy, 1),
                    "detection_count": track_info.get('detection_count', 0),
                    "first_frame": track_info.get('first_frame', 0),
                    "last_frame": track_info.get('last_frame', 0),
                    "duration_frames": track_info.get('duration_frames', 0),
                    "estimated_distance_m": track_info.get('avg_distance', 0.0)
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generator": "Argus Track Enhanced Light Post Tracker",
                "total_locations": len(features),
                "min_reliability_threshold": min_reliability,
                "coordinate_system": "WGS84",
                "gps_based_geolocation": True
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        self.logger.info(f"Exported {len(features)} geolocated objects to GeoJSON: {output_path}")
    
    def _get_track_summary(self, track_id: int) -> Dict[str, Any]:
        """Get summary statistics for a track"""
        if track_id not in self.gps_tracks:
            return {}
        
        return {
            'avg_confidence': 0.8,
            'detection_count': len(self.gps_tracks[track_id]),
            'first_frame': 0,
            'last_frame': 100,
            'duration_frames': 100,
            'avg_distance': 25.0
        }

    def get_enhanced_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics with geolocation info"""
        base_stats = self.get_track_statistics()
        
        enhanced_stats = {
            **base_stats,
            'geolocated_objects': len(self.track_locations),
            'avg_reliability': np.mean([loc.reliability for loc in self.track_locations.values()]) if self.track_locations else 0.0,
            'avg_accuracy_meters': np.mean([loc.accuracy for loc in self.track_locations.values()]) if self.track_locations else 0.0,
            'gps_data_available': len(self.frame_to_gps) > 0,
            'gps_coverage_percent': len(self.frame_to_gps) / max(1, len(self.tracker.get_all_tracks())) * 100
        }
        
        return enhanced_stats

    def get_track_statistics(self) -> Dict[str, Any]:
        """Get basic tracking statistics"""
        tracks = self.tracker.get_all_tracks()
        
        return {
            'total_tracks': len(tracks),
            'active_tracks': len(self.tracker.active_tracks),
            'lost_tracks': len(self.tracker.lost_tracks),
            'removed_tracks': len(self.tracker.removed_tracks),
            'total_frames': self.tracker.frame_id,
            'avg_track_length': np.mean([track.age for track in tracks.values()]) if tracks else 0,
            'static_objects': len([track_id for track_id in self.track_locations.keys()]),
            'located_objects': len(self.track_locations)
        }