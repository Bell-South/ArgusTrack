"""
Enhanced Light Post Tracker using Ultralytics' ByteTrack
=======================================================

This module integrates Ultralytics' ByteTrack with our GPS and geolocation system.
It replaces the custom ByteTrack implementation with Ultralytics' optimized version.
"""

import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
import json

# Ultralytics imports
from ultralytics import YOLO

# Project imports
from argus_track.config import TrackerConfig
from argus_track.core import Detection, GPSData
from argus_track.utils.gps_utils import compute_average_location, GeoLocation, CoordinateTransformer
from argus_track.utils.io import save_tracking_results
from argus_track.utils.gps_sync_tracker import GPSSynchronizer
from argus_track.utils.visualization import RealTimeVisualizer


class UltralyticsTracker:
    """
    Enhanced Light Post Tracker using Ultralytics' ByteTrack implementation
    for more efficient and accurate tracking with GPS integration.
    """
    
    def __init__(self, 
                 config: TrackerConfig,
                 model_path: str,
                 show_realtime: bool = False,
                 display_size: Tuple[int, int] = (1280, 720),
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.5):
        """
        Initialize enhanced tracker with Ultralytics' ByteTrack
        
        Args:
            config: Tracker configuration
            model_path: Path to YOLO model file (.pt)
            show_realtime: Whether to show real-time visualization
            display_size: Display size for visualization
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for detections
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.UltralyticsTracker")
        
        # Real-time visualization settings
        self.show_realtime = show_realtime
        self.display_size = display_size
        self.visualizer = None
        
        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Resolution scaling factor
        self.resolution_scale = 1.0
        
        # GPS tracking and geolocation
        self.gps_synchronizer: Optional[GPSSynchronizer] = None
        self.gps_tracks: Dict[int, List[GPSData]] = {}
        self.track_locations: Dict[int, GeoLocation] = {}
        
        # Track history
        self.track_history: Dict[int, List[Dict]] = {}
        
        # Performance monitoring
        self.processing_times = []
        
        # Load Ultralytics model
        self.logger.info(f"Loading YOLO model from {model_path}")
        try:
            self.model = YOLO(model_path)
            self.logger.info(f"Successfully loaded model: {self.model.names}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Initialize visualization if requested
        if self.show_realtime:
            self.visualizer = RealTimeVisualizer(
                window_name="Argus Track - Ultralytics Tracker",
                display_size=self.display_size,
                show_info_panel=True
            )
            self.logger.info("Real-time visualization enabled")
    
    def process_video(self, 
                      video_path: str, 
                      gps_data: Optional[List[GPSData]] = None,
                      output_path: Optional[str] = None,
                      save_results: bool = True,
                      resolution_scale: float = 1.0) -> Dict[int, List[Dict]]:
        """
        Process video with Ultralytics' ByteTrack and GPS integration
        
        Args:
            video_path: Path to input video
            gps_data: Optional GPS data for geolocation
            output_path: Optional path for output video
            save_results: Whether to save tracking results
            resolution_scale: Resolution scaling factor
            
        Returns:
            Dictionary of track histories
        """
        self.logger.info(f"Processing video with Ultralytics ByteTrack: {video_path}")
        
        # Set resolution scale
        self.resolution_scale = resolution_scale
        if resolution_scale < 1.0:
            self.logger.warning(f"Resolution scaling is set to {resolution_scale:.2f}x, which may reduce detection accuracy")
        
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
        
        self.logger.info(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Initialize GPS synchronizer if GPS data available
        if gps_data:
            self.gps_synchronizer = GPSSynchronizer(gps_data, fps, gps_fps=10.0)
            sync_stats = self.gps_synchronizer.get_processing_statistics()
            
            self.logger.info("🎯 GPS SYNCHRONIZATION ACTIVE:")
            self.logger.info(f"   📍 GPS points available: {sync_stats['gps_points']}")
            self.logger.info(f"   🎬 Frames to process: {sync_stats['sync_frames']}")
            self.logger.info(f"   📊 Processing ratio: {sync_stats['processing_ratio']:.3f}")
            self.logger.info(f"   🔄 GPS frequency: {sync_stats['avg_gps_frequency']:.1f} Hz")
            
            if sync_stats['sync_frames'] == 0:
                self.logger.error("❌ No frames to process - GPS synchronization failed!")
                return {}
        else:
            self.logger.warning("⚠️  No GPS data provided - processing all frames")
            self.gps_synchronizer = None
        
        # Setup video writer if output path provided
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        processed_frames = 0
        skipped_frames = 0
        
        # Create custom tracker config file
        tracker_config = self._create_tracker_config()
        
        try:
            current_frame_idx = 0
            
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                # Check if this frame should be processed
                should_process = False
                current_gps = None
                
                if self.gps_synchronizer:
                    # Only process frames with GPS data
                    should_process = self.gps_synchronizer.should_process_frame(current_frame_idx)
                    if should_process:
                        current_gps = self.gps_synchronizer.get_gps_for_frame(current_frame_idx)
                else:
                    # Process every frame if no GPS synchronizer
                    should_process = True
                
                if not should_process:
                    current_frame_idx += 1
                    skipped_frames += 1
                    continue
                
                # Apply resolution scaling if enabled
                if self.resolution_scale < 1.0:
                    try:
                        scaled_width = int(frame.shape[1] * self.resolution_scale)
                        scaled_height = int(frame.shape[0] * self.resolution_scale)
                        frame = cv2.resize(frame, (scaled_width, scaled_height), 
                                         interpolation=cv2.INTER_AREA)
                    except Exception as e:
                        self.logger.error(f"Error scaling frame: {e}")
                
                # Process frame with ByteTrack
                start_time = time.time()
                
                # Run tracking with Ultralytics model
                results = self.model.track(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    tracker=tracker_config,
                    persist=True,  # Keep track history
                    verbose=False
                )
                
                # Extract tracking results
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Get boxes, track IDs, and classes
                    if result.boxes is not None and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        track_ids = result.boxes.id.int().cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.int().cpu().numpy()
                        
                        # Update track history
                        for i, track_id in enumerate(track_ids):
                            track_id = int(track_id)
                            if track_id not in self.track_history:
                                self.track_history[track_id] = []
                            
                            # Create track data
                            box = boxes[i]
                            score = scores[i]
                            class_id = int(classes[i])
                            class_name = self.model.names[class_id]
                            
                            track_data = {
                                'frame': current_frame_idx,
                                'bbox': box.tolist(),
                                'score': float(score),
                                'class_id': class_id,
                                'class_name': class_name,
                                'has_gps': current_gps is not None,
                                'timestamp': current_gps.timestamp if current_gps else current_frame_idx / fps
                            }
                            
                            self.track_history[track_id].append(track_data)
                            
                            # Update GPS data for active tracks
                            if current_gps:
                                self._update_gps_tracks(track_id, current_gps)
                
                # Real-time visualization
                if self.show_realtime and self.visualizer:
                    # Show frame with tracking results
                    annotated_frame = None
                    if results and len(results) > 0:
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame.copy()
                    
                    # Add GPS info if available
                    gps_info = None
                    if current_gps:
                        gps_info = {
                            'latitude': current_gps.latitude,
                            'longitude': current_gps.longitude,
                            'heading': current_gps.heading
                        }
                    
                    frame_info = {
                        'frame_idx': current_frame_idx,
                        'total_frames': frame_count,
                        'fps': fps,
                        'skipped_frames': skipped_frames,
                        'processed_frames': processed_frames,
                        'gps_sync': current_gps is not None
                    }
                    
                    # Visualize frame
                    detections = []  # Convert Ultralytics format to our Detection format
                    tracks = []      # Convert to our Track format for visualization
                    
                    should_continue = self.visualizer.visualize_frame(
                        annotated_frame, detections, tracks, gps_info, frame_info
                    )
                    
                    if not should_continue:
                        self.logger.info("User requested quit from visualization")
                        break
                
                # Save to output video if requested
                if out_writer:
                    # Scale back to original size if needed
                    if self.resolution_scale < 1.0:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                    # Use annotated frame from Ultralytics if available
                    if results and len(results) > 0:
                        out_frame = results[0].plot()
                        out_writer.write(out_frame)
                    else:
                        out_writer.write(frame)
                
                # Performance monitoring
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                
                processed_frames += 1
                current_frame_idx += 1
                
                # Progress logging
                if processed_frames % 50 == 0:
                    avg_time = np.mean(self.processing_times[-50:]) if self.processing_times else 0
                    effective_fps = 1.0 / avg_time if avg_time > 0 else 0
                    
                    if self.gps_synchronizer:
                        total_gps_frames = self.gps_synchronizer.get_sync_frames_count()
                        progress = processed_frames / total_gps_frames * 100
                        self.logger.info(f"📍 GPS-synced progress: {processed_frames}/{total_gps_frames} "
                                      f"({progress:.1f}%) at {effective_fps:.1f} FPS")
                    else:
                        progress = current_frame_idx / frame_count * 100
                        self.logger.info(f"🎬 Frame progress: {current_frame_idx}/{frame_count} "
                                      f"({progress:.1f}%) - processed: {processed_frames}")
        
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user (Ctrl+C)")
        
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
            
        finally:
            # Cleanup
            cap.release()
            if out_writer:
                out_writer.release()
            if self.visualizer:
                self.visualizer.close()
            cv2.destroyAllWindows()
            
            # Remove temporary tracker config
            if os.path.exists(tracker_config):
                try:
                    os.remove(tracker_config)
                except:
                    pass
        
        # Calculate geolocations for tracked objects
        if self.gps_synchronizer and processed_frames > 0:
            self.logger.info("📍 Calculating object geolocations from GPS-synced data...")
            self._calculate_object_geolocations()
        
        # Performance summary
        if self.processing_times:
            avg_processing = np.mean(self.processing_times) * 1000  # ms
            effective_fps = 1.0 / np.mean(self.processing_times)
            
            self.logger.info("📊 PROCESSING SUMMARY:")
            self.logger.info(f"   🎬 Total video frames: {frame_count}")
            self.logger.info(f"   📍 Processed frames: {processed_frames}")
            self.logger.info(f"   ⏭️  Skipped frames: {skipped_frames}")
            self.logger.info(f"   ⚡ Processing FPS: {effective_fps:.1f}")
            self.logger.info(f"   ⏱️  Avg frame time: {avg_processing:.1f}ms")
        
        # Save results if requested
        if save_results:
            self._save_results(video_path, fps, width, height, processed_frames, skipped_frames, frame_count)
        
        return self.track_history
    
    def _create_tracker_config(self) -> str:
        """
        Create custom ByteTrack configuration file
        
        Returns:
            Path to tracker configuration file
        """
        # Create temporary file for tracker config
        config_path = "custom_bytetrack.yaml"
        
        # Create config content
        config_content = f"""# Custom ByteTrack configuration for Stereo Geolocation Tracker

tracker_type: bytetrack
track_high_thresh: {self.confidence_threshold}
track_low_thresh: {max(0.1, self.confidence_threshold / 2)}
new_track_thresh: {self.confidence_threshold}
track_buffer: {self.config.track_buffer}
match_thresh: {self.config.match_thresh}
fuse_score: True
"""
        
        # Write config to file
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        self.logger.info(f"Created custom ByteTrack configuration: {config_path}")
        return config_path
    
    def _update_gps_tracks(self, track_id: int, gps_data: GPSData) -> None:
        """Update GPS data for a track"""
        if track_id not in self.gps_tracks:
            self.gps_tracks[track_id] = []
        
        # Add GPS data for this track
        self.gps_tracks[track_id].append(gps_data)
    
    def _calculate_object_geolocations(self) -> None:
        """Calculate real-world geolocations for tracked objects using GPS data"""
        
        for track_id, track_history in self.track_history.items():
            # Filter for static, well-tracked objects
            if len(track_history) < self.config.min_static_frames:
                continue
            
            # Check if track has GPS data
            gps_frames = [t for t in track_history if t.get('has_gps', False)]
            if len(gps_frames) < 3:  # Need at least 3 GPS points for reliable geolocation
                continue
            
            # Analyze if object is static
            if not self._is_static_track(track_history):
                continue
            
            # Calculate geolocation
            geolocation = self._estimate_object_geolocation(track_id, track_history)
            
            if geolocation and geolocation.reliability > 0.3:
                self.track_locations[track_id] = geolocation
                self.logger.debug(
                    f"Track {track_id} ({track_history[0].get('class_name', 'unknown')}) geolocated: "
                    f"({geolocation.latitude:.6f}, {geolocation.longitude:.6f}) "
                    f"reliability: {geolocation.reliability:.2f}"
                )
    
    def _is_static_track(self, track_history: List[Dict]) -> bool:
        """
        Determine if a track represents a static object
        
        Args:
            track_history: Track history data
            
        Returns:
            True if track is static, False otherwise
        """
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
        """
        Estimate geolocation for a single tracked object
        
        Args:
            track_id: Track ID
            track_history: Track history data
            
        Returns:
            Estimated geolocation or None if not possible
        """
        # Get track's GPS data
        if track_id not in self.gps_tracks or len(self.gps_tracks[track_id]) < 3:
            return None
        
        gps_points = self.gps_tracks[track_id]
        
        # For static objects, we can simply average GPS positions
        avg_lat = np.mean([gps.latitude for gps in gps_points])
        avg_lon = np.mean([gps.longitude for gps in gps_points])
        
        # Calculate reliability based on position consistency
        lat_std = np.std([gps.latitude for gps in gps_points])
        lon_std = np.std([gps.longitude for gps in gps_points])
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
            timestamp=gps_points[-1].timestamp
        )
    
    def _save_results(self, video_path: str, fps: float, width: int, height: int,
                    processed_frames: int, skipped_frames: int, frame_count: int) -> None:
        """
        Save tracking and geolocation results
        
        Args:
            video_path: Input video path
            fps: Video frame rate
            width: Video width
            height: Video height
            processed_frames: Number of processed frames
            skipped_frames: Number of skipped frames
            frame_count: Total frame count
        """
        results_path = Path(video_path).with_suffix('.json')
        geojson_path = Path(video_path).with_suffix('.geojson')
        
        # Get GPS synchronization stats
        sync_stats = self.gps_synchronizer.get_processing_statistics() if self.gps_synchronizer else {}
        
        # Prepare results
        results = {
            'metadata': {
                'total_frames': frame_count,
                'processed_frames': processed_frames,
                'skipped_frames': skipped_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'resolution_scale': self.resolution_scale,
                'config': self.config.__dict__,
                'gps_synchronization': sync_stats,
                'geolocated_objects': len(self.track_locations),
                'tracker': 'ultralytics_bytetrack',
                'model': str(self.model),
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold
            },
            'tracks': self.track_history,
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
        
        # Save JSON results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved tracking results to {results_path}")
        
        # Save GeoJSON
        self._export_geojson(geojson_path)
    
    def _export_geojson(self, output_path: Path, min_reliability: float = 0.3) -> None:
        """
        Export geolocated objects to GeoJSON format
        
        Args:
            output_path: Output file path
            min_reliability: Minimum reliability threshold
        """
        reliable_locations = {
            track_id: location for track_id, location in self.track_locations.items()
            if location.reliability >= min_reliability
        }
        
        features = []
        
        for track_id, location in reliable_locations.items():
            # Get class name from track history
            class_name = "unknown"
            if track_id in self.track_history and self.track_history[track_id]:
                class_name = self.track_history[track_id][0].get('class_name', "unknown")
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [location.longitude, location.latitude]
                },
                "properties": {
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": sum(t.get('score', 0) for t in self.track_history.get(track_id, [])) / 
                                len(self.track_history.get(track_id, [1])),
                    "reliability": round(location.reliability, 3),
                    "accuracy_meters": round(location.accuracy, 1),
                    "detection_count": len(self.track_history.get(track_id, [])),
                    "gps_synced": True,
                    "processing_method": "ultralytics_bytetrack"
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generator": "Argus Track with Ultralytics ByteTrack",
                "total_locations": len(features),
                "min_reliability_threshold": min_reliability,
                "coordinate_system": "WGS84",
                "gps_based_geolocation": True,
                "processing_mode": "gps_synchronized_frames",
                "sync_statistics": self.gps_synchronizer.get_processing_statistics() if self.gps_synchronizer else {}
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        self.logger.info(f"Exported {len(features)} geolocated objects to GeoJSON: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tracking and geolocation statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_tracks': len(self.track_history),
            'geolocated_objects': len(self.track_locations),
            'avg_reliability': np.mean([loc.reliability for loc in self.track_locations.values()]) if self.track_locations else 0.0,
            'avg_accuracy_meters': np.mean([loc.accuracy for loc in self.track_locations.values()]) if self.track_locations else 0.0,
            'static_objects': sum(1 for track_id in self.track_history if self._is_static_track(self.track_history[track_id])),
            'avg_track_length': np.mean([len(track) for track in self.track_history.values()]) if self.track_history else 0,
            'processing_fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
            'avg_processing_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0,
        }
        
        # Add GPS synchronization stats if available
        if self.gps_synchronizer:
            stats['gps_synchronization'] = self.gps_synchronizer.get_processing_statistics()
        
        return stats