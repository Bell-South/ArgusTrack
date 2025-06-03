# argus_track/trackers/unified_lightpost_tracker.py

"""
Unified Light Post Tracker - Clean, GPS-informed tracking
=========================================================

Combines best features from both trackers with GPS movement context
to solve track fragmentation and resurrection issues.
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from ultralytics import YOLO
from ..config import TrackerConfig
from ..core import Detection, GPSData
from ..utils.smart_track_manager import CleanTrackManager
from ..utils.output_manager import OutputManager
from ..utils.gps_sync_tracker import GPSSynchronizer
from ..utils.static_car_detector import StaticCarDetector, StaticCarConfig
from ..utils.visualization import RealTimeVisualizer
from ..utils.overlap_fixer import OverlapFixer


class UnifiedLightPostTracker:
    """
    Unified Light Post Tracker with GPS movement context
    
    Clean, focused tracker that uses GPS to:
    - Detect vehicle movement (skip stationary frames)
    - Prevent impossible track resurrections (forward motion logic)
    - Provide movement context for ID management
    
    Solves:
    - Track fragmentation 
    - Track ID resurrection
    - Duplicate ID assignment
    """

    def __init__(self, 
                 config: TrackerConfig,
                 model_path: str,
                 show_realtime: bool = False,
                 display_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize unified tracker
        
        Args:
            config: Tracker configuration
            model_path: Path to YOLOv11 model
            show_realtime: Show real-time visualization
            display_size: Display window size
        """
        self.config = config
        self.model_path = model_path
        self.show_realtime = show_realtime
        self.display_size = display_size
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.UnifiedLightPostTracker")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.class_names = list(self.model.names.values())
        self.logger.info(f"Model classes: {self.class_names}")
        
        # Initialize core components
        self.track_manager = CleanTrackManager(config)
        self.overlap_fixer = OverlapFixer(
            overlap_threshold=0.3,
            distance_threshold=1.0
        )
        
        # GPS and movement tracking
        self.gps_synchronizer: Optional[GPSSynchronizer] = None
        self.static_car_detector: Optional[StaticCarDetector] = None
        self.current_gps: Optional[GPSData] = None
        self.previous_gps: Optional[GPSData] = None
        self.vehicle_speed: float = 0.0
        self.vehicle_moved_distance: float = 0.0
        
        # Initialize static car detection
        if config.enable_static_car_detection:
            static_config = StaticCarConfig(
                movement_threshold_meters=config.static_movement_threshold_m,
                stationary_time_threshold=config.static_time_threshold_s,
                gps_frame_interval=config.gps_frame_interval
            )
            self.static_car_detector = StaticCarDetector(static_config)
            self.logger.info("Static car detection enabled")
        
        # Initialize real-time visualizer
        self.visualizer = None
        if show_realtime:
            self.visualizer = RealTimeVisualizer(
                window_name="Unified Light Post Tracking",
                display_size=display_size,
                show_info_panel=True
            )
            self.logger.info("Real-time visualization enabled")
        
        # Processing statistics
        self.processing_times = []
        self.frame_count = 0
        self.processed_frame_count = 0
        
        self.logger.info("Unified Light Post Tracker initialized")

    def process_video(self,
                     video_path: str,
                     gps_data: Optional[List[GPSData]] = None,
                     output_path: Optional[str] = None,
                     save_results: bool = True) -> Dict[str, Any]:
        """
        Process video with unified tracking
        
        Args:
            video_path: Path to input video
            gps_data: Optional GPS data
            output_path: Optional output video path
            save_results: Whether to save results
            
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Starting unified tracking: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
        
        # Initialize GPS synchronizer
        if gps_data:
            self.gps_synchronizer = GPSSynchronizer(gps_data, fps, gps_fps=10.0)
            sync_stats = self.gps_synchronizer.get_processing_statistics()
            self.logger.info(f"GPS sync: {sync_stats['sync_frames']} frames to process")
        else:
            self.logger.warning("No GPS data - processing all frames")
        
        # Initialize output manager
        output_manager = OutputManager(video_path, self.class_names)
        
        # Setup video writer
        out_writer = None
        if output_path and self.show_realtime:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_fps = fps / self.config.gps_frame_interval if self.gps_synchronizer else fps
            out_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Processing loop
        current_frame_idx = 0
        processed_frames = 0
        skipped_frames_gps = 0
        skipped_frames_static = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Check GPS synchronization
                should_process_gps = True
                current_gps = None
                
                if self.gps_synchronizer:
                    should_process_gps = self.gps_synchronizer.should_process_frame(current_frame_idx)
                    if should_process_gps:
                        current_gps = self.gps_synchronizer.get_gps_for_frame(current_frame_idx)
                    else:
                        skipped_frames_gps += 1
                
                # Check static car detection
                should_process_static = True
                if should_process_gps and self.static_car_detector and current_gps:
                    should_process_static = self.static_car_detector.should_process_frame(
                        current_gps, current_frame_idx
                    )
                    if not should_process_static:
                        skipped_frames_static += 1
                
                # Final decision: process frame?
                should_process = should_process_gps and should_process_static
                
                if not should_process:
                    current_frame_idx += 1
                    continue
                
                # Update GPS movement context
                self._update_gps_context(current_gps)
                
                # Process frame
                frame_timestamp = current_frame_idx / fps
                detections = self._process_frame(frame, current_frame_idx, frame_timestamp)
                
                # Add to output manager
                output_manager.add_frame_data(
                    frame_id=current_frame_idx,
                    timestamp=frame_timestamp,
                    detections=detections,
                    gps_data=current_gps
                )
                
                # Real-time visualization
                if self.show_realtime and self.visualizer:
                    should_continue = self._visualize_frame(
                        frame, detections, current_frame_idx, current_gps, 
                        total_frames, processed_frames
                    )
                    
                    if not should_continue:
                        self.logger.info("User requested quit")
                        break
                
                # Save frame to output video
                if out_writer and self.show_realtime:
                    vis_frame = self._create_visualization_frame(frame, detections, current_gps)
                    out_writer.write(vis_frame)
                
                # Performance tracking
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                processed_frames += 1
                
                # Progress logging
                if processed_frames % 30 == 0:
                    avg_time = np.mean(self.processing_times[-30:])
                    progress = current_frame_idx / total_frames * 100
                    
                    self.logger.info(
                        f"Progress: {progress:.1f}% | "
                        f"Processed: {processed_frames} | "
                        f"Skipped (GPS): {skipped_frames_gps} | "
                        f"Skipped (Static): {skipped_frames_static} | "
                        f"Avg time: {avg_time*1000:.1f}ms | "
                        f"Speed: {self.vehicle_speed:.1f}m/s"
                    )
                
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise
        finally:
            # Cleanup
            cap.release()
            if out_writer:
                out_writer.release()
            if self.visualizer:
                self.visualizer.close()
            cv2.destroyAllWindows()
        
        # Processing summary
        total_time = sum(self.processing_times)
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        
        results = {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'skipped_frames_gps': skipped_frames_gps,
            'skipped_frames_static': skipped_frames_static,
            'processing_time': total_time,
            'avg_fps': avg_fps,
            'track_manager_stats': self.track_manager.get_statistics(),
            'output_summary': output_manager.get_processing_summary()
        }
        
        self.logger.info("=== PROCESSING COMPLETE ===")
        self.logger.info(f"Total frames: {total_frames}")
        self.logger.info(f"Processed frames: {processed_frames}")
        self.logger.info(f"Processing time: {total_time:.1f}s")
        self.logger.info(f"Average FPS: {avg_fps:.1f}")
        
        # Track manager statistics
        track_stats = self.track_manager.get_statistics()
        self.logger.info(f"Active tracks: {track_stats['active_tracks']}")
        self.logger.info(f"Total tracks created: {track_stats['total_tracks_created']}")
        
        # Save results
        if save_results:
            json_path, csv_path = output_manager.export_both()
            results['json_output'] = json_path
            results['csv_output'] = csv_path
            
            # Print summary
            output_manager.print_summary()
        
        return results
    
    def _update_gps_context(self, current_gps: Optional[GPSData]):
        """Update GPS movement context for tracking decisions"""
        if current_gps is None:
            return
        
        # Calculate vehicle movement
        if self.previous_gps is not None:
            # Calculate distance moved
            distance = self._calculate_gps_distance(
                self.previous_gps.latitude, self.previous_gps.longitude,
                current_gps.latitude, current_gps.longitude
            )
            
            # Calculate time difference
            dt = current_gps.timestamp - self.previous_gps.timestamp
            if dt > 0:
                self.vehicle_speed = distance / dt
                self.vehicle_moved_distance += distance
            
            # Pass movement context to track manager
            self.track_manager.update_movement_context(
                vehicle_speed=self.vehicle_speed,
                distance_moved=distance,
                total_distance=self.vehicle_moved_distance
            )
        
        # Update GPS references
        self.current_gps = current_gps
        self.previous_gps = current_gps
    
    def _calculate_gps_distance(self, lat1: float, lon1: float, 
                               lat2: float, lon2: float) -> float:
        """Calculate distance between GPS points in meters"""
        R = 6378137.0  # Earth radius
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        """Process single frame with GPS-informed tracking"""
        
        # NEW: Update GPS context for motion prediction
        if hasattr(self.track_manager, 'update_gps_context'):
            self.track_manager.update_gps_context(self.current_gps, self.previous_gps)
        
        # Run YOLO tracking (existing code)
        track_params = self.config.get_ultralytics_track_params()
        results = self.model.track(frame, **track_params)
        
        # Apply overlap fixer (existing code)
        fixed_detections = self.overlap_fixer.fix_ultralytics_results(
            results[0], self.current_gps, frame_id
        )
        
        # Convert to Detection objects (existing code)
        raw_detections = self._convert_fixed_detections(fixed_detections, frame_id)
        
        # Apply GPS-informed track management (existing code)
        processed_detections = self.track_manager.process_frame_detections(
            raw_detections, frame_id, timestamp
        )
        
        return processed_detections
        
    def _convert_fixed_detections(self, fixed_detections: List[Dict], frame_id: int) -> List[Detection]:
        """Convert fixed detections to Detection objects"""
        detections = []
        
        for det_dict in fixed_detections:
            detection = Detection(
                bbox=np.array(det_dict['bbox']),
                score=float(det_dict['score']),
                class_id=int(det_dict['class_id']),
                frame_id=frame_id
            )
            detection.track_id = int(det_dict['track_id'])
            detections.append(detection)
        
        return detections
    
    def _visualize_frame(self, frame: np.ndarray, detections: List[Detection],
                        frame_id: int, gps_data: Optional[GPSData],
                        total_frames: int, processed_frames: int) -> bool:
        """Visualize frame with real-time display"""
        
        # Prepare frame info
        frame_info = {
            'frame_idx': frame_id,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'gps_available': gps_data is not None,
            'vehicle_speed': self.vehicle_speed,
            'distance_moved': self.vehicle_moved_distance
        }
        
        # Prepare GPS info
        gps_info = None
        if gps_data:
            gps_info = {
                'latitude': gps_data.latitude,
                'longitude': gps_data.longitude,
                'heading': gps_data.heading,
                'accuracy': gps_data.accuracy,
                'vehicle_speed_ms': self.vehicle_speed,
                'vehicle_speed_kmh': self.vehicle_speed * 3.6
            }
        
        # Convert detections to tracks for visualization
        tracks = []
        for detection in detections:
            class MockTrack:
                def __init__(self, detection):
                    self.track_id = detection.track_id
                    self.state = 'confirmed'
                    self.detections = [detection]
                    self.hits = 1
                    self.age = 1
                    self.time_since_update = 0
                    self._bbox = detection.bbox
                
                def to_tlbr(self):
                    return self._bbox
            
            track = MockTrack(detection)
            tracks.append(track)
        
        # Visualize
        return self.visualizer.visualize_frame(
            frame, detections, tracks, gps_info, frame_info
        )
    
    def _create_visualization_frame(self, frame: np.ndarray, 
                                   detections: List[Detection],
                                   gps_data: Optional[GPSData]) -> np.ndarray:
        """Create visualization frame for video output"""
        
        vis_frame = frame.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection.bbox.astype(int)
            class_name = self.class_names[detection.class_id] if detection.class_id < len(self.class_names) else f"class_{detection.class_id}"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"ID:{detection.track_id} {class_name} {detection.score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(vis_frame, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         (0, 255, 0), -1)
            
            # Text
            cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add GPS and movement info overlay
        if gps_data:
            info_lines = [
                f"GPS: {gps_data.latitude:.6f}, {gps_data.longitude:.6f}",
                f"Speed: {self.vehicle_speed * 3.6:.1f} km/h",
                f"Distance: {self.vehicle_moved_distance:.1f} m"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(vis_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_offset += 30
        
        # Add detection count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(vis_frame, det_text, (10, vis_frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add track statistics
        track_stats = self.track_manager.get_statistics()
        stats_text = f"Active Tracks: {track_stats['active_tracks']}"
        cv2.putText(vis_frame, stats_text, (10, vis_frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return vis_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        track_stats = self.track_manager.get_statistics()
        
        return {
            'processed_frames': self.processed_frame_count,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'vehicle_speed': self.vehicle_speed,
            'distance_moved': self.vehicle_moved_distance,
            'track_manager': track_stats,
            'model_classes': self.class_names
        }