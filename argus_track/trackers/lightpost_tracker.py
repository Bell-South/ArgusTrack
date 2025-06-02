# argus_track/trackers/lightpost_tracker_v2.py

"""Enhanced Light Post Tracker with Motion Compensation"""

import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from ..config import TrackerConfig
from ..core import Detection, Track, GPSData
from ultralytics import YOLO
from ..utils.static_car_detector import StaticCarDetector, StaticCarConfig
from ..utils.visualization import draw_tracks, RealTimeVisualizer
from ..utils.io import save_tracking_results, load_gps_data
from ..utils.gps_utils import compute_average_location, GeoLocation
from ..utils.gps_sync_tracker import GPSSynchronizer
from ..utils.overlap_fixer import OverlapFixer

class EnhancedLightPostTracker:
    """
    Enhanced Light Post Tracker with Motion Compensation for Moving Cameras
    """
    
    def __init__(self, 
                 config: TrackerConfig,
                 model_path: str,
                 show_realtime: bool = False,
                 display_size: Tuple[int, int] = (1280, 720),
                 auto_adjust_motion: bool = True):
        """
        Initialize motion-aware light post tracker
        
        Args:
            config: Motion-aware tracker configuration
            detector: Object detection module
            show_realtime: Whether to show real-time visualization
            display_size: Size of real-time display
            auto_adjust_motion: Automatically adjust parameters based on motion
        """
        self.config = config
        # self.detector = detector
        self.show_realtime = show_realtime
        self.display_size = display_size
        self.auto_adjust_motion = auto_adjust_motion

        self.overlap_fixer = OverlapFixer(overlap_threshold=0.9, distance_threshold=1.0)

        # Initialize YOLO model with tracking
        self.model = YOLO(model_path)
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.MotionAwareLightPostTracker")
        
        # Real-time visualization
        self.visualizer = None
        if show_realtime:
            self.visualizer = RealTimeVisualizer(
                window_name="Motion-Aware Light Post Tracking",
                display_size=display_size,
                show_info_panel=True
            )
        
        # GPS and motion tracking
        self.gps_synchronizer: Optional[GPSSynchronizer] = None
        self.gps_tracks: Dict[int, List[GPSData]] = {}
        self.track_locations: Dict[int, GeoLocation] = {}
        
        # Motion analysis
        self.motion_history = []
        self.adaptive_config = config
        self.motion_adjustment_frequency = 60  # Adjust every 60 frames
        
        # Performance monitoring
        self.processing_times = []
        self.motion_compensation_times = []
        self.frame_count = 0
        
        self.logger.info("Initialized Motion-Aware Light Post Tracker")
        self.logger.info(f"Auto motion adjustment: {auto_adjust_motion}")

        # Import Kalman deduplicator
        from ..utils.kalman_gps_filter import create_kalman_gps_deduplicator
        self.kalman_deduplicator = create_kalman_gps_deduplicator(merge_distance_m=3.0)
        self.logger.info("Kalman GPS deduplication enabled (3m threshold)")

        self.static_car_detector = None
        if hasattr(config, 'enable_static_car_detection') and config.enable_static_car_detection:
            from ..utils.static_car_detector import StaticCarDetector, StaticCarConfig
            static_config = StaticCarConfig(
                movement_threshold_meters=getattr(config, 'static_movement_threshold_m', 2.0),
                stationary_time_threshold=getattr(config, 'static_time_threshold_s', 10.0),
                gps_frame_interval=6
            )
            self.static_car_detector = StaticCarDetector(static_config)
            self.logger.info("Static car detection enabled")
        else:
            self.logger.info("Static car detection disabled")

        self._analyze_tracking_issues()

    def process_video(self, 
                     video_path: str,
                     gps_data: Optional[List[GPSData]] = None,
                     output_path: Optional[str] = None,
                     save_results: bool = True,
                     resolution_scale: float = 1.0) -> Dict[int, List[Dict]]:
        """
        Process video with motion-aware tracking
        
        Args:
            video_path: Path to input video
            gps_data: Optional GPS data
            output_path: Optional output video path
            save_results: Whether to save results
            resolution_scale: Resolution scaling factor
            
        Returns:
            Dictionary of tracked objects
        """
        self.logger.info(f"Processing video with motion-aware tracking: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Could not open video: {video_path}"
            self.logger.error(error_msg)
            raise IOError(error_msg)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Initialize GPS synchronizer if available
        if gps_data:
            self.gps_synchronizer = GPSSynchronizer(gps_data, fps, gps_fps=10.0)
            sync_stats = self.gps_synchronizer.get_processing_statistics()
            self.logger.info(f"GPS sync: {sync_stats['sync_frames']} frames to process")
        
        # Setup video writer
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        all_tracks = {}
        current_frame_idx = 0
        processed_frames = 0
        motion_estimates = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if frame should be processed (GPS sync or all frames)
                should_process = True
                current_gps = None
                
                if self.gps_synchronizer:
                    should_process = self.gps_synchronizer.should_process_frame(current_frame_idx)
                    if should_process:
                        current_gps = self.gps_synchronizer.get_gps_for_frame(current_frame_idx)
                        
                        # Additional check: Static car detection
                        if self.static_car_detector and current_gps:
                            should_process = self.static_car_detector.should_process_frame(current_gps, current_frame_idx)
                            if not should_process:
                                self.logger.debug(f"Frame {current_frame_idx}: Skipped due to stationary car")

                if not should_process:
                    current_frame_idx += 1
                    continue
                
                # Apply resolution scaling
                if resolution_scale < 1.0:
                    scaled_width = int(frame.shape[1] * resolution_scale)
                    scaled_height = int(frame.shape[0] * resolution_scale)
                    frame = cv2.resize(frame, (scaled_width, scaled_height))
                
                # Start timing
                start_time = time.time()
                
                track_params = self.config.get_ultralytics_track_params()
                results = self.model.track(frame, **track_params)

                # Convert Ultralytics results to our format
                detections = []
                tracks = []
               
                # APPLY OVERLAP FIXING - CORRECTED METHOD NAME
                if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    # Fix overlapping boxes and consolidate IDs
                    fixed_results = self.overlap_fixer.fix_tracking_results(
                        results, current_gps, current_frame_idx
                    )
                    
                    # Process fixed results
                    for fixed_detection in fixed_results:
                        # Create Detection object
                        detection = Detection(
                            bbox=fixed_detection['bbox'],
                            score=fixed_detection['score'],
                            class_id=fixed_detection['class_id'],
                            frame_id=current_frame_idx
                        )
                        detections.append(detection)
                        
                        # Use consolidated track ID
                        track_id = fixed_detection['track_id']
                        
                        # Create Track object for visualization
                        track = Track(
                            track_id=track_id,
                            detections=[detection],
                            state='confirmed'
                        )
                        tracks.append(track)
                        
                        # Calculate depth using lightpost height (4 meters)
                        bbox_height = detection.bbox[3] - detection.bbox[1]
                        focal_length = 1400
                        lightpost_height = 4.0
                        estimated_depth = (lightpost_height * focal_length) / bbox_height
                        
                        # Store depth information
                        if not hasattr(self, 'track_depths'):
                            self.track_depths = {}
                        if track_id not in self.track_depths:
                            self.track_depths[track_id] = []
                        self.track_depths[track_id].append(estimated_depth)
                        
                        # Convert to GPS coordinates if we have GPS data
                        if current_gps:
                            bbox_center_x = (detection.bbox[0] + detection.bbox[2]) / 2
                            image_width = frame.shape[1]
                            
                            pixels_from_center = bbox_center_x - (image_width / 2)
                            degrees_per_pixel = 60.0 / image_width
                            bearing_offset = pixels_from_center * degrees_per_pixel
                            object_bearing = current_gps.heading + bearing_offset
                            
                            import math
                            lat_offset = (estimated_depth * math.cos(math.radians(object_bearing))) / 111000
                            lon_offset = (estimated_depth * math.sin(math.radians(object_bearing))) / (111000 * math.cos(math.radians(current_gps.latitude)))
                            
                            object_lat = current_gps.latitude + lat_offset
                            object_lon = current_gps.longitude + lon_offset
                            
                            # Store GPS location with CONSOLIDATED track ID
                            if not hasattr(self, 'track_gps_locations'):
                                self.track_gps_locations = {}
                            if track_id not in self.track_gps_locations:
                                self.track_gps_locations[track_id] = []
                            
                            location_data = {
                                'latitude': object_lat,
                                'longitude': object_lon,
                                'depth': estimated_depth,
                                'bearing': object_bearing,
                                'frame': current_frame_idx,
                                'confidence': fixed_detection['score'],
                                'class_id': fixed_detection['class_id'],
                                'original_track_id': fixed_detection.get('original_track_id', track_id)  # Keep original for debugging
                            }
                            self.track_gps_locations[track_id].append(location_data)
                            
                            # Log with both original and consolidated IDs
                            orig_id = fixed_detection.get('original_track_id', track_id)
                            id_msg = f" (was {orig_id})" if orig_id != track_id else ""
                            print(f"Track {track_id}{id_msg}: GPS ({object_lat:.6f}, {object_lon:.6f}) depth {estimated_depth:.1f}m")

                # Get motion statistics
                motion_stats = {
                    'motion_detected': False,
                    'avg_translation': 0,
                    'total_tracks': len(tracks) if tracks else 0
                }
                
                # Auto-adjust configuration based on motion
                if (self.auto_adjust_motion and 
                    processed_frames % self.motion_adjustment_frequency == 0 and
                    motion_estimates):
                    
                    avg_motion = np.mean(motion_estimates[-self.motion_adjustment_frequency:])
                    
                    self.logger.info(f"Adjusted config for motion level: {avg_motion:.1f}px/frame")
                
                # Update GPS tracks
                if current_gps:
                    self._update_gps_tracks(tracks, current_gps, current_frame_idx)
                
                # Store track data
                for track in tracks:
                    if track.track_id not in all_tracks:
                        all_tracks[track.track_id] = []
                    
                    track_data = {
                        'frame': current_frame_idx,
                        'bbox': track.to_tlbr().tolist(),
                        'score': track.detections[-1].score if track.detections else 0,
                        'state': track.state,
                        'hits': track.hits,
                        'has_gps': current_gps is not None,
                        'motion_compensated': motion_stats.get('motion_detected', False)
                    }
                    all_tracks[track.track_id].append(track_data)
                
                # Real-time visualization
                if self.show_realtime and self.visualizer:
                    frame_info = {
                        'frame_idx': current_frame_idx,
                        'total_frames': frame_count,
                        'fps': fps,
                        'processed_frames': processed_frames,
                        'motion_detected': motion_stats.get('motion_detected', False),
                        'avg_motion': motion_stats.get('avg_translation', 0),
                        'config_adjusted': hasattr(self, 'adaptive_config')
                    }
                    
                    gps_info = None
                    if current_gps:
                        gps_info = {
                            'latitude': current_gps.latitude,
                            'longitude': current_gps.longitude,
                            'heading': current_gps.heading
                        }
                    
                    should_continue = self.visualizer.visualize_frame(
                        frame, detections, tracks, gps_info, frame_info
                    )
                    
                    if not should_continue:
                        self.logger.info("User requested quit")
                        break
                
                # Save to output video
                if out_writer:
                    vis_frame = self._create_motion_aware_visualization(
                        frame, tracks, motion_stats, current_gps
                    )
                    
                    if resolution_scale < 1.0:
                        vis_frame = cv2.resize(vis_frame, (width, height))
                    
                    out_writer.write(vis_frame)
                
                # Performance monitoring
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                
                # Progress logging
                processed_frames += 1
                if processed_frames % 100 == 0:
                    avg_time = np.mean(self.processing_times[-50:])
                    avg_motion_time = np.mean(self.motion_compensation_times[-50:])
                    
                    self.logger.info(
                        f"Processed {processed_frames} frames - "
                        f"Avg: {avg_time*1000:.1f}ms/frame, "
                        f"Motion: {avg_motion_time*1000:.1f}ms"
                    )
                    
                    if motion_estimates:
                        recent_motion = np.mean(motion_estimates[-50:])
                        self.logger.info(f"Recent motion: {recent_motion:.1f}px/frame")
                
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if self.visualizer:
                self.visualizer.close()
            cv2.destroyAllWindows()
        
        # Calculate geolocations with motion compensation
        if self.gps_synchronizer and processed_frames > 0:
            self.logger.info("Calculating motion-compensated geolocations...")
            self._calculate_motion_aware_geolocations(all_tracks)
        
        # Performance summary
        if self.processing_times:
            avg_processing = np.mean(self.processing_times) * 1000
            avg_motion_comp = np.mean(self.motion_compensation_times) * 1000
            effective_fps = 1.0 / np.mean(self.processing_times)
            
            self.logger.info("=== MOTION-AWARE PROCESSING SUMMARY ===")
            self.logger.info(f"Processed frames: {processed_frames}")
            self.logger.info(f"Processing FPS: {effective_fps:.1f}")
            self.logger.info(f"Avg frame time: {avg_processing:.1f}ms")
            self.logger.info(f"Motion compensation: {avg_motion_comp:.1f}ms")
            
            if motion_estimates:
                avg_motion = np.mean(motion_estimates)
                max_motion = np.max(motion_estimates)
                self.logger.info(f"Average motion: {avg_motion:.1f}px/frame")
                self.logger.info(f"Maximum motion: {max_motion:.1f}px/frame")

        # Static car detection summary - ADD THIS BLOCK
        if self.static_car_detector:
            static_stats = self.static_car_detector.get_statistics()
            self.logger.info("=== STATIC CAR DETECTION SUMMARY ===")
            self.logger.info(f"Frames skipped due to stationary car: {static_stats['skipped_frames']}")
            self.logger.info(f"Stationary periods detected: {static_stats['stationary_periods_count']}")
            if static_stats['stationary_periods_count'] > 0:
                self.logger.info(f"Total stationary time: {static_stats['total_stationary_time']:.1f}s")
                self.logger.info(f"Avg stationary duration: {static_stats['avg_stationary_duration']:.1f}s")
            self.logger.info(f"Processing efficiency gain: {static_stats['efficiency_gain']}")

        # Save results
        if save_results:
            self._save_motion_aware_results(all_tracks, video_path, motion_estimates)
        
        return all_tracks
    
    def _create_motion_aware_visualization(self, 
                                          frame: np.ndarray,
                                          tracks: List[Track],
                                          motion_stats: Dict,
                                          gps_data: Optional[GPSData]) -> np.ndarray:
        """Create visualization with motion information"""
        
        # Draw basic tracks
        vis_frame = draw_tracks(frame, tracks, show_trajectory=True)
        
        # Add motion information overlay
        info_lines = [
            f"Tracks: {len(tracks)}",
            f"Motion: {motion_stats.get('avg_translation', 0):.1f}px",
            f"Config: {'Adaptive' if self.auto_adjust_motion else 'Fixed'}",
        ]
        
        if gps_data:
            info_lines.extend([
                f"GPS: {gps_data.latitude:.5f}",
                f"     {gps_data.longitude:.5f}"
            ])
        
        # Add text overlay
        y_offset = 30
        for line in info_lines:
            cv2.putText(vis_frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Add motion indicator
        if motion_stats.get('motion_detected', False):
            cv2.putText(vis_frame, "MOTION DETECTED", (10, vis_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame
    
    def _update_gps_tracks(self, tracks: List[Track], gps_data: GPSData, frame_idx: int):
        """Update GPS data for active tracks"""
        for track in tracks:
            if track.track_id not in self.gps_tracks:
                self.gps_tracks[track.track_id] = []
            self.gps_tracks[track.track_id].append(gps_data)
    
    def _calculate_motion_aware_geolocations(self, all_tracks: Dict[int, List[Dict]]):
        """Calculate geolocations considering camera motion"""
        
        for track_id, track_history in all_tracks.items():
            # Filter for well-tracked objects
            if len(track_history) < self.config.min_static_frames:
                continue
            
            # Check GPS availability
            gps_frames = [t for t in track_history if t.get('has_gps', False)]
            if len(gps_frames) < self.config.min_gps_points:
                continue
            
            # Check if object is motion-compensated static
            if not self._is_motion_compensated_static(track_history):
                continue
            
            # Calculate geolocation with motion awareness
            geolocation = self._estimate_motion_aware_geolocation(track_id, track_history)
            
            if geolocation and geolocation.reliability > 0.2:
                self.track_locations[track_id] = geolocation
                self.logger.debug(
                    f"Track {track_id} geolocated (motion-aware): "
                    f"({geolocation.latitude:.6f}, {geolocation.longitude:.6f}) "
                    f"reliability: {geolocation.reliability:.2f}"
                )
    
    def _is_motion_compensated_static(self, track_history: List[Dict]) -> bool:
        """Check if object is static considering motion compensation"""
        
        # If track has motion compensation applied, use relaxed criteria
        motion_compensated_frames = [t for t in track_history if t.get('motion_compensated', False)]
        
        if len(motion_compensated_frames) > len(track_history) * 0.5:
            # Majority of frames were motion compensated
            # Use more lenient static criteria
            return len(track_history) >= self.config.min_static_frames
        else:
            # Use standard static analysis
            if len(track_history) < 5:
                return False
            
            centers = []
            for detection in track_history:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                centers.append([center_x, center_y])
            
            centers = np.array(centers)
            std_movement = np.std(centers, axis=0)
            max_movement = np.max(std_movement)
            
            # Use motion-adjusted threshold
            return max_movement < self.adaptive_config.static_threshold
    
    def _estimate_motion_aware_geolocation(self, 
                                          track_id: int,
                                          track_history: List[Dict]) -> Optional[GeoLocation]:
        """Estimate geolocation with motion compensation awareness"""
        
        if track_id not in self.gps_tracks or len(self.gps_tracks[track_id]) < 2:
            return None
        
        gps_points = self.gps_tracks[track_id]
        
        # Use multiple GPS points for better accuracy in motion scenarios
        estimated_positions = []
        
        for i, gps_point in enumerate(gps_points[::2]):  # Use every other GPS point
            # Find corresponding detection
            detection = None
            for hist_entry in track_history:
                if hist_entry.get('has_gps', False):
                    detection = hist_entry
                    break
            
            if detection is None:
                continue
            
            bbox = detection['bbox']
            
            # Enhanced distance estimation for motion scenarios
            distance = self._estimate_object_distance_motion_aware(bbox, gps_point)
            
            # Calculate object offset with motion compensation
            lateral_offset, forward_offset = self._calculate_object_offset_motion_aware(
                bbox, distance, gps_point
            )
            
            # Convert to GPS coordinates
            obj_lat, obj_lon = self._gps_offset_to_coordinates(
                gps_point, lateral_offset, forward_offset
            )
            
            estimated_positions.append({
                'lat': obj_lat,
                'lon': obj_lon,
                'distance': distance,
                'confidence': detection['score'],
                'motion_compensated': detection.get('motion_compensated', False)
            })
        
        if not estimated_positions:
            return None
        
        # Calculate weighted average (favor motion-compensated positions)
        weights = []
        for pos in estimated_positions:
            weight = pos['confidence']
            if pos['motion_compensated']:
                weight *= 1.5  # Boost motion-compensated positions
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        avg_lat = np.sum([pos['lat'] * w for pos, w in zip(estimated_positions, weights)])
        avg_lon = np.sum([pos['lon'] * w for pos, w in zip(estimated_positions, weights)])
        
        # Enhanced reliability calculation for motion scenarios
        lat_std = np.std([pos['lat'] for pos in estimated_positions])
        lon_std = np.std([pos['lon'] for pos in estimated_positions])
        position_std = np.sqrt(lat_std**2 + lon_std**2)
        
        # Motion compensation factor
        motion_comp_ratio = len([p for p in estimated_positions if p['motion_compensated']]) / len(estimated_positions)
        motion_bonus = motion_comp_ratio * 0.2
        
        reliability = (1.0 / (1.0 + position_std * 5000)) + motion_bonus
        reliability = min(1.0, reliability)
        
        # Accuracy estimation
        earth_radius = 6378137.0
        lat_error = lat_std * earth_radius * np.pi / 180
        lon_error = lon_std * earth_radius * np.pi / 180 * np.cos(np.radians(avg_lat))
        accuracy = np.sqrt(lat_error**2 + lon_error**2)
        
        return GeoLocation(
            latitude=avg_lat,
            longitude=avg_lon,
            accuracy=max(0.5, accuracy),
            reliability=reliability,
            timestamp=estimated_positions[-1]['confidence']  # Use last confidence as timestamp placeholder
        )
    
    def _estimate_object_distance_motion_aware(self, bbox: List[float], gps_data: GPSData) -> float:
        """Estimate distance with motion compensation"""
        # Enhanced distance estimation considering camera motion
        real_object_width = 0.3  # meters (LED light post width)
        bbox_width = bbox[2] - bbox[0]
        
        if bbox_width > 0:
            # Base distance calculation
            focal_length = 1400  # Approximate for GoPro
            base_distance = (real_object_width * focal_length) / bbox_width
            
            # Adjust for camera motion (moving camera sees objects differently)
            # This is a simplified adjustment - could be more sophisticated
            motion_factor = 1.0  # Could be based on GPS speed/heading changes
            
            adjusted_distance = base_distance * motion_factor
            return max(1.0, min(adjusted_distance, 200.0))
        
        return 50.0  # Default distance
    
    def get_track_statistics(self):
        """Get tracking statistics from the underlying tracker"""
        return {
            'total_tracks': len(self.track_locations) if hasattr(self, 'track_locations') else 0,
            'active_tracks': 0,  # Ultralytics handles this internally
            'lost_tracks': 0,    # Ultralytics handles this internally  
            'removed_tracks': 0, # Ultralytics handles this internally
            'frame_id': getattr(self, 'frame_count', 0)
        }

    def _calculate_object_offset_motion_aware(self, 
                                            bbox: List[float], 
                                            distance: float,
                                            gps_data: GPSData) -> Tuple[float, float]:
        """Calculate object offset with motion awareness"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Camera parameters
        image_width = 1920  # Adjust based on actual resolution
        image_height = 1080
        
        dx_px = center_x - (image_width / 2)
        dy_px = center_y - (image_height / 2)
        
        # Angle calculation with motion compensation
        angle_per_pixel = 0.0005  # Calibrated value
        angle_x = dx_px * angle_per_pixel
        angle_y = dy_px * angle_per_pixel
        
        # Calculate offsets in camera coordinate system
        lateral_offset = distance * np.sin(angle_x)
        forward_offset = distance * np.cos(angle_x)
        
        return lateral_offset, forward_offset
    
    def _gps_offset_to_coordinates(self, 
                                  base_gps: GPSData,
                                  lateral_offset: float,
                                  forward_offset: float) -> Tuple[float, float]:
        """Convert offsets to GPS coordinates"""
        R = 6378137.0  # Earth radius
        heading_rad = np.radians(base_gps.heading)
        
        # Rotate offsets to world coordinates
        east_offset = lateral_offset * np.cos(heading_rad) + forward_offset * np.sin(heading_rad)
        north_offset = -lateral_offset * np.sin(heading_rad) + forward_offset * np.cos(heading_rad)
        
        # Convert to GPS
        lat_offset = north_offset / R * 180 / np.pi
        lon_offset = east_offset / (R * np.cos(np.radians(base_gps.latitude))) * 180 / np.pi
        
        return base_gps.latitude + lat_offset, base_gps.longitude + lon_offset
    
    def _save_motion_aware_results(self, 
                                  all_tracks: Dict[int, List[Dict]],
                                  video_path: str,
                                  motion_estimates: List[float]):
        """Save results with motion awareness information"""
        
        results_path = Path(video_path).with_suffix('.json')
        geojson_path = Path(video_path).with_suffix('.geojson')

        # Enhanced metadata with motion information
        metadata = {
            'total_tracks': len(all_tracks),
            'geolocated_objects': len(self.track_locations),
            'auto_motion_adjustment': self.auto_adjust_motion,
            'motion_statistics': {
                'avg_motion_per_frame': np.mean(motion_estimates) if motion_estimates else 0,
                'max_motion_per_frame': np.max(motion_estimates) if motion_estimates else 0,
                'motion_frames': len(motion_estimates)
            },
            'config_used': self.adaptive_config.__dict__
        }
        
        # Save JSON results
        results = {
            'metadata': metadata,
            'tracks': all_tracks,
            'track_locations': {
                str(track_id): {
                    'latitude': loc.latitude,
                    'longitude': loc.longitude,
                    'accuracy': loc.accuracy,
                    'reliability': loc.reliability,
                    'motion_compensated': True
                }
                for track_id, loc in self.track_locations.items()
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save GeoJSON
        self._export_motion_aware_geojson(geojson_path)
        
        self.logger.info(f"Saved motion-aware results to {results_path}")

    def _export_motion_aware_geojson(self, output_path: Path):
        """Export GeoJSON with Kalman-filtered GPS deduplication"""
        
        features = []
        
        # DEBUG: Analyze all tracks (your existing debug code)
        if hasattr(self, 'track_gps_locations'):
            self.logger.info("🔍 TRACK ANALYSIS DEBUG:")
            self.logger.info(f"   Total tracks with GPS data: {len(self.track_gps_locations)}")
            
            tracks_by_detection_count = {}
            for track_id, locations in self.track_gps_locations.items():
                detection_count = len(locations)
                if detection_count not in tracks_by_detection_count:
                    tracks_by_detection_count[detection_count] = 0
                tracks_by_detection_count[detection_count] += 1
                
                # Log individual track details
                if detection_count >= self.config.min_detections_for_export:
                    avg_lat = sum(loc['latitude'] for loc in locations) / len(locations)
                    avg_lon = sum(loc['longitude'] for loc in locations) / len(locations)
                    avg_confidence = sum(loc['confidence'] for loc in locations) / len(locations)
                    
                    status = "✅ EXPORTED" if detection_count >= self.config.min_detections_for_export else "❌ TOO FEW"
                    self.logger.info(f"   Track {track_id}: {detection_count} detections, "
                                f"avg_conf: {avg_confidence:.2f}, "
                                f"pos: ({avg_lat:.6f}, {avg_lon:.6f}) - {status}")
            
            # Summary by detection count
            self.logger.info("   Detection count summary:")
            for count in sorted(tracks_by_detection_count.keys()):
                self.logger.info(f"     {tracks_by_detection_count[count]} tracks with {count} detections")
            
            self.logger.info(f"   Current export threshold: >= {self.config.min_detections_for_export} detections")
            
            # Count how many would be exported with different thresholds
            thresholds_to_test = [1, 2, 3, 4, 5]
            for threshold in thresholds_to_test:
                count = len([t for t, locs in self.track_gps_locations.items() if len(locs) >= threshold])
                self.logger.info(f"     With threshold >= {threshold}: {count} tracks would be exported")

        # Export features (your existing export code)
        if hasattr(self, 'track_gps_locations'):
            for track_id, locations in self.track_gps_locations.items():
                if len(locations) >= self.config.min_detections_for_export:
                    # Average the GPS coordinates for this track
                    avg_lat = sum(loc['latitude'] for loc in locations) / len(locations)
                    avg_lon = sum(loc['longitude'] for loc in locations) / len(locations)
                    avg_depth = sum(loc['depth'] for loc in locations) / len(locations)
                    avg_confidence = sum(loc['confidence'] for loc in locations) / len(locations)
                    
                    feature = {
                        "type": "Feature", 
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(avg_lon), float(avg_lat)]
                        },
                        "properties": {
                            "track_id": int(track_id),
                            "confidence": round(float(avg_confidence), 3),
                            "estimated_distance_m": round(float(avg_depth), 1),
                            "detection_count": int(len(locations)),
                            "class_id": int(locations[0]['class_id']),
                            "processing_method": "ultralytics_tracking"
                        }
                    }
                    features.append(feature)
        
        # KALMAN DEDUPLICATION - NEW SECTION
        if features:
            # Apply Kalman GPS deduplication
            deduplicated_features = self.kalman_deduplicator.deduplicate_locations(features)
        else:
            deduplicated_features = features
        
        # Create GeoJSON with deduplication metadata
        geojson = {
            "type": "FeatureCollection",
            "features": deduplicated_features,
            "metadata": {
                "generator": "Argus Track with Kalman GPS Deduplication",
                "total_locations": len(deduplicated_features),
                "original_locations": len(features),
                "duplicates_removed": len(features) - len(deduplicated_features),
                "merge_distance_m": 3.0,
                "processing_method": "kalman_filtered_gps_deduplication"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        self.logger.info(f"Exported {len(deduplicated_features)} Kalman-filtered locations to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        return {
            'geolocated_objects': len(self.track_locations),
            'avg_reliability': np.mean([loc.reliability for loc in self.track_locations.values()]) if self.track_locations else 0,
            'config_adaptive': self.auto_adjust_motion
        }
    
    def _analyze_tracking_issues(self):
        """
        Analyze tracking data to identify common issues and patterns
        """
        if not hasattr(self, 'track_gps_locations') or not self.track_gps_locations:
            return
        
        self.logger.info("🔍 TRACKING ISSUE ANALYSIS:")
        
        # Issue 1: Short-lived tracks (potential fragmentation)
        short_tracks = [tid for tid, locs in self.track_gps_locations.items() if len(locs) <= 3]
        if short_tracks:
            self.logger.info(f"   📉 Short tracks (≤3 detections): {len(short_tracks)} tracks")
            self.logger.info(f"      Track IDs: {short_tracks[:10]}{'...' if len(short_tracks) > 10 else ''}")
        
        # Issue 2: Track GPS clustering analysis  
        track_positions = {}
        for track_id, locations in self.track_gps_locations.items():
            if len(locations) >= 2:
                avg_lat = sum(loc['latitude'] for loc in locations) / len(locations)
                avg_lon = sum(loc['longitude'] for loc in locations) / len(locations)
                track_positions[track_id] = (avg_lat, avg_lon)
        
        # Find tracks that are very close to each other (potential duplicates/fragments)
        close_track_pairs = []
        track_ids = list(track_positions.keys())
        
        for i, tid1 in enumerate(track_ids):
            for tid2 in track_ids[i+1:]:
                lat1, lon1 = track_positions[tid1]
                lat2, lon2 = track_positions[tid2]
                distance = self._calculate_gps_distance(lat1, lon1, lat2, lon2)
                
                if distance <= 5.0:  # Within 5 meters
                    close_track_pairs.append((tid1, tid2, distance))
        
        if close_track_pairs:
            self.logger.info(f"   🎯 Potentially related tracks (≤5m apart): {len(close_track_pairs)} pairs")
            for tid1, tid2, dist in close_track_pairs[:5]:  # Show first 5
                det1 = len(self.track_gps_locations[tid1])
                det2 = len(self.track_gps_locations[tid2])
                self.logger.info(f"      Tracks {tid1}({det1} det) ↔ {tid2}({det2} det): {dist:.1f}m apart")
        
        # Issue 3: Track detection count distribution
        detection_counts = [len(locs) for locs in self.track_gps_locations.values()]
        if detection_counts:
            avg_detections = sum(detection_counts) / len(detection_counts)
            max_detections = max(detection_counts)
            min_detections = min(detection_counts)
            
            self.logger.info(f"   📊 Detection count stats:")
            self.logger.info(f"      Average: {avg_detections:.1f}, Range: {min_detections}-{max_detections}")
            
            # Identify outliers (very high detection counts - possible stuck tracks)
            high_detection_tracks = [tid for tid, locs in self.track_gps_locations.items() 
                                if len(locs) > avg_detections * 3]
            if high_detection_tracks:
                self.logger.info(f"      High-detection tracks: {high_detection_tracks}")
        
        # Issue 4: Track ID gaps analysis (fragmentation indicator)
        all_track_ids = sorted(self.track_gps_locations.keys())
        if len(all_track_ids) > 1:
            max_id = max(all_track_ids)
            actual_tracks = len(all_track_ids)
            id_efficiency = actual_tracks / max_id if max_id > 0 else 1.0
            
            self.logger.info(f"   🔢 Track ID efficiency: {id_efficiency:.2f} ({actual_tracks}/{max_id})")
            if id_efficiency < 0.7:
                self.logger.info(f"      ⚠️  Low efficiency suggests track fragmentation")
        
        # Issue 5: Temporal analysis - look for time gaps
        track_time_spans = {}
        for track_id, locations in self.track_gps_locations.items():
            if len(locations) >= 2:
                frames = [loc['frame'] for loc in locations]
                frame_span = max(frames) - min(frames)
                frame_gaps = []
                
                sorted_frames = sorted(frames)
                for i in range(1, len(sorted_frames)):
                    gap = sorted_frames[i] - sorted_frames[i-1]
                    if gap > 10:  # Gap larger than expected
                        frame_gaps.append(gap)
                
                if frame_gaps:
                    track_time_spans[track_id] = max(frame_gaps)
        
        if track_time_spans:
            tracks_with_gaps = len([gap for gap in track_time_spans.values() if gap > 20])
            self.logger.info(f"   ⏱️  Tracks with temporal gaps (>20 frames): {tracks_with_gaps}")

    def _calculate_gps_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between GPS points in meters"""
        import numpy as np
        R = 6378137.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
            np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
class OverlapFixer:
    """
    Fixes Ultralytics tracking issues in real-time:
    1. Removes overlapping bounding boxes in same frame
    2. Prevents multiple IDs for same object
    3. Consolidates fragmented track IDs
    """
    
    def __init__(self, overlap_threshold: float = 0.5, distance_threshold: float = 3.0):
        """
        Initialize overlap fixer
        
        Args:
            overlap_threshold: IoU threshold for detecting overlaps (0.5 = 50% overlap)
            distance_threshold: GPS distance threshold for same object (meters)
        """
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.logger = logging.getLogger(f"{__name__}.OverlapFixer")
        
        # Track ID management
        self.id_mapping = {}  # original_id -> consolidated_id
        self.next_consolidated_id = 1
        self.track_positions = {}  # track_id -> recent GPS positions
        
    def fix_tracking_results(self, ultralytics_results, current_gps: Optional[GPSData], 
                           frame_id: int) -> List[Dict]:
        """
        Fix Ultralytics tracking results in real-time
        
        Args:
            ultralytics_results: Raw results from model.track()
            current_gps: Current GPS data
            frame_id: Current frame number
            
        Returns:
            Fixed list of detections with consolidated track IDs
        """
        if not ultralytics_results[0].boxes or ultralytics_results[0].boxes.id is None:
            return []
        
        # Extract raw detections
        raw_detections = self._extract_detections(ultralytics_results[0], frame_id)
        
        # Step 1: Remove overlapping bounding boxes in same frame
        non_overlapping = self._remove_overlapping_boxes(raw_detections)
        
        # Step 2: Consolidate track IDs (prevent multiple IDs for same object)
        consolidated = self._consolidate_track_ids(non_overlapping, current_gps)
        
        self.logger.debug(f"Frame {frame_id}: {len(raw_detections)} → {len(non_overlapping)} → {len(consolidated)} detections")
        
        return consolidated
    
    def _extract_detections(self, results, frame_id: int) -> List[Dict]:
        """Extract detections from Ultralytics results"""
        detections = []
        
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        
        for i, (box, score, cls_id, track_id) in enumerate(zip(boxes, scores, classes, track_ids)):
            detections.append({
                'bbox': box,
                'score': score,
                'class_id': cls_id,
                'track_id': track_id,
                'frame': frame_id
            })
        
        return detections
    
    def _remove_overlapping_boxes(self, detections: List[Dict]) -> List[Dict]:
        """Remove overlapping bounding boxes in same frame"""
        if len(detections) <= 1:
            return detections
        
        # Calculate IoU matrix
        n = len(detections)
        keep_indices = list(range(n))
        
        for i in range(n):
            if i not in keep_indices:
                continue
                
            for j in range(i + 1, n):
                if j not in keep_indices:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(detections[i]['bbox'], detections[j]['bbox'])
                
                if iou > self.overlap_threshold:
                    # Keep the detection with higher confidence
                    if detections[i]['score'] >= detections[j]['score']:
                        keep_indices.remove(j)
                        self.logger.debug(f"   Removed overlapping box: track {detections[j]['track_id']} "
                                        f"(IoU: {iou:.2f} with track {detections[i]['track_id']})")
                    else:
                        keep_indices.remove(i)
                        self.logger.debug(f"   Removed overlapping box: track {detections[i]['track_id']} "
                                        f"(IoU: {iou:.2f} with track {detections[j]['track_id']})")
                        break
        
        return [detections[i] for i in keep_indices]
    
    def _consolidate_track_ids(self, detections: List[Dict], current_gps: Optional[GPSData]) -> List[Dict]:
        """Consolidate track IDs to prevent multiple IDs for same object"""
        
        for detection in detections:
            original_id = detection['track_id']
            
            # Check if this is a new track that should be merged with existing
            consolidated_id = self._get_consolidated_id(detection, current_gps)
            
            # Update detection with consolidated ID
            detection['original_track_id'] = original_id
            detection['track_id'] = consolidated_id
            
            # Update track position history
            if current_gps and consolidated_id not in self.track_positions:
                self.track_positions[consolidated_id] = []
            
            if current_gps:
                # Calculate GPS position for this detection
                gps_pos = self._calculate_detection_gps(detection, current_gps)
                if gps_pos:
                    self.track_positions[consolidated_id].append({
                        'lat': gps_pos[0],
                        'lon': gps_pos[1],
                        'frame': detection['frame']
                    })
                    
                    # Keep only recent positions
                    if len(self.track_positions[consolidated_id]) > 10:
                        self.track_positions[consolidated_id] = self.track_positions[consolidated_id][-10:]
        
        return detections
    
    def _get_consolidated_id(self, detection: Dict, current_gps: Optional[GPSData]) -> int:
        """Get consolidated track ID for detection"""
        original_id = detection['track_id']
        
        # If we've seen this original ID before, return its mapping
        if original_id in self.id_mapping:
            return self.id_mapping[original_id]
        
        # Check if this detection is close to any existing tracks
        if current_gps:
            detection_gps = self._calculate_detection_gps(detection, current_gps)
            
            if detection_gps:
                # Find existing tracks within distance threshold
                for existing_id, positions in self.track_positions.items():
                    if not positions:
                        continue
                    
                    # Check distance to most recent position
                    recent_pos = positions[-1]
                    distance = self._gps_distance(
                        detection_gps[0], detection_gps[1],
                        recent_pos['lat'], recent_pos['lon']
                    )
                    
                    if distance <= self.distance_threshold:
                        # This detection is close to existing track - merge them
                        self.id_mapping[original_id] = existing_id
                        self.logger.info(f"   🔗 Merged track {original_id} into {existing_id} "
                                       f"(distance: {distance:.1f}m)")
                        return existing_id
        
        # This is a genuinely new track
        new_consolidated_id = self.next_consolidated_id
        self.id_mapping[original_id] = new_consolidated_id
        self.next_consolidated_id += 1
        
        return new_consolidated_id
    
    def _calculate_detection_gps(self, detection: Dict, gps: GPSData) -> Optional[Tuple[float, float]]:
        """Calculate GPS coordinates for detection"""
        try:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Simple depth estimation
            bbox_height = bbox[3] - bbox[1]
            if bbox_height > 0:
                focal_length = 1400
                lightpost_height = 4.0
                estimated_depth = (lightpost_height * focal_length) / bbox_height
                
                # Convert to GPS offset
                image_width = 1920  # Assume standard resolution
                pixels_from_center = center_x - (image_width / 2)
                degrees_per_pixel = 60.0 / image_width
                bearing_offset = pixels_from_center * degrees_per_pixel
                object_bearing = gps.heading + bearing_offset
                
                import math
                lat_offset = (estimated_depth * math.cos(math.radians(object_bearing))) / 111000
                lon_offset = (estimated_depth * math.sin(math.radians(object_bearing))) / (111000 * math.cos(math.radians(gps.latitude)))
                
                object_lat = gps.latitude + lat_offset
                object_lon = gps.longitude + lon_offset
                
                return (object_lat, object_lon)
        except:
            pass
        
        return None
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _gps_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between GPS points in meters"""
        import numpy as np
        R = 6378137.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c