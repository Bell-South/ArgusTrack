# argus_track/trackers/simplified_lightpost_tracker.py (NEW FILE)

"""
Simplified Light Post Tracker - Monocular tracking with ID consolidation
No depth estimation, no geolocation - just tracking with frame data output
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
from ..utils.smart_track_manager import SmartTrackManager
from ..utils.output_manager import OutputManager
from ..utils.gps_sync_tracker import GPSSynchronizer
from ..utils.static_car_detector import StaticCarDetector, StaticCarConfig
from ..utils.visualization import RealTimeVisualizer
from ..utils.overlap_fixer import OverlapFixer

class SimplifiedLightPostTracker:
    """
    Simplified Light Post Tracker for monocular tracking
    
    Features:
    - Single camera processing (no stereo)
    - GPS frame synchronization (every 6th frame)
    - Smart track ID consolidation and reappearance detection
    - JSON + CSV output (no geolocation)
    - Real-time visualization (GPS frames only)
    """

    def __init__(self, 
                config: TrackerConfig,
                model_path: str,
                show_realtime: bool = False,
                display_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize enhanced simplified tracker with motion prediction and visual features
        """
        self.config = config
        self.model_path = model_path
        self.show_realtime = show_realtime
        self.display_size = display_size
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.SimplifiedLightPostTracker")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Get class names from model
        self.class_names = list(self.model.names.values())
        self.logger.info(f"Model classes: {self.class_names}")

        self.track_manager = SmartTrackManager(
            config=config,
            max_memory_age=300,
            min_detection_count=3,
            similarity_threshold=50.0
        )
        
        # Initialize overlap fixer (from your original solution)
        self.overlap_fixer = OverlapFixer(
            overlap_threshold=0.3, #0.5
            distance_threshold=1.0 #3.0
        )
        
        # Initialize GPS synchronizer (will be set during processing)
        self.gps_synchronizer: Optional[GPSSynchronizer] = None
        
        # Initialize static car detector
        self.static_car_detector = None
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
                window_name="Enhanced Simplified Light Post Tracking (GPS + Motion + Visual)",
                display_size=display_size,
                show_info_panel=True
            )
            self.logger.info("Real-time visualization enabled (GPS frames only)")
        
        # Processing statistics
        self.processing_times = []
        self.frame_count = 0
        self.processed_frame_count = 0
        
        # GPS motion tracking
        self.previous_gps = None
        self.gps_motion_history = []
        
        self.logger.info("Enhanced Simplified Light Post Tracker initialized")
        self.logger.info(f"GPS frame interval: {config.gps_frame_interval}")
        self.logger.info(f"Motion prediction enabled: {config.track_consolidation.enable_motion_prediction}")
        self.logger.info(f"Visual features enabled: {config.track_consolidation.enable_visual_features}")
        self.logger.info(f"Track consolidation enabled: {config.track_consolidation.enable_id_consolidation}")

    def _process_frame(self,
                    frame: np.ndarray,
                    frame_id: int,
                    timestamp: float,
                    gps_data: Optional[GPSData]) -> List[Detection]:
        """Enhanced frame processing with motion prediction and visual features"""
        
        # Calculate GPS motion if available
        gps_motion_matrix = None
        vehicle_speed = 0.0
        
        if gps_data and self.previous_gps and self.track_manager.motion_predictor:
            gps_motion_matrix = self.track_manager.motion_predictor.estimate_motion_from_gps_enhanced(
                gps_data, self.previous_gps
            )
            
            # Calculate vehicle speed
            dt = gps_data.timestamp - self.previous_gps.timestamp
            if dt > 0:
                R = 6378137.0
                lat1, lon1 = np.radians(self.previous_gps.latitude), np.radians(self.previous_gps.longitude)
                lat2, lon2 = np.radians(gps_data.latitude), np.radians(gps_data.longitude)
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                distance = R * c
                vehicle_speed = distance / dt
        
        # Store current GPS for next frame
        self.previous_gps = gps_data
        
        # Run YOLO tracking
        track_params = self.config.get_ultralytics_track_params()
        results = self.model.track(frame, **track_params)
        
        # STEP 1: Apply overlap fixer first
        fixed_detections = self.overlap_fixer.fix_ultralytics_results(
            results[0], gps_data, frame_id
        )
        
        # STEP 2: Convert to Detection objects
        raw_detections = self._convert_fixed_detections(fixed_detections, frame_id)
        
        # STEP 3: Apply ENHANCED smart track management with all features
        processed_detections = self.track_manager.process_frame_detections(
            raw_detections, 
            frame_id, 
            timestamp,
            frame,      # For visual feature extraction and camera motion estimation
            gps_data    # For GPS-enhanced processing
        )
        
        # STEP 4: Apply GPS motion compensation if available
        if (gps_motion_matrix is not None and 
            vehicle_speed > 1.0 and 
            self.track_manager.motion_predictor):
            
            # Convert detections to tracks for compensation
            temp_tracks = []
            for detection in processed_detections:
                if hasattr(detection, 'track_id'):
                    from ..core import Track
                    track = Track(
                        track_id=detection.track_id,
                        detections=[detection],
                        state='confirmed'
                    )
                    temp_tracks.append(track)
            
            # Apply GPS motion compensation
            compensated_tracks = self.track_manager.motion_predictor.compensate_tracks_for_gps_motion(
                temp_tracks, gps_motion_matrix, vehicle_speed
            )
            
            # Convert back to detections
            for i, track in enumerate(compensated_tracks):
                if i < len(processed_detections):
                    compensated_bbox = track.to_tlbr()
                    processed_detections[i].bbox = compensated_bbox
                    processed_detections[i].motion_compensated = True
            
            self.logger.debug(f"Frame {frame_id}: Applied GPS motion compensation (speed: {vehicle_speed:.1f} m/s)")
        
        return processed_detections

    def _visualize_frame(self,
                        frame: np.ndarray,
                        detections: List[Detection],
                        frame_id: int,
                        gps_data: Optional[GPSData],
                        total_frames: int,
                        processed_frames: int) -> bool:
        """Enhanced visualization with motion and visual feature info"""
        
        # Get enhanced statistics
        track_stats = self.track_manager.get_statistics()
        
        # Prepare enhanced frame info
        frame_info = {
            'frame_idx': frame_id,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'gps_available': gps_data is not None,
            'track_consolidations': track_stats['total_consolidations'],
            'track_reappearances': track_stats['total_reappearances'],
            # Enhanced info
            'motion_prediction_enabled': track_stats['motion_prediction']['motion_predictor_enabled'],
            'visual_features_enabled': track_stats['visual_features']['feature_extraction_enabled'],
            'camera_motion_detected': track_stats['motion_prediction']['camera_motion_detected'],
            'tracks_with_features': track_stats['visual_features']['tracks_with_features'],
            'avg_prediction_accuracy': track_stats['motion_prediction']['avg_prediction_accuracy'],
            'avg_appearance_stability': track_stats['visual_features']['avg_appearance_stability']
        }
        
        # Prepare GPS info with vehicle motion
        gps_info = None
        if gps_data:
            vehicle_speed = 0.0
            if self.previous_gps:
                dt = gps_data.timestamp - self.previous_gps.timestamp
                if dt > 0:
                    # Calculate speed for display
                    R = 6378137.0
                    lat1, lon1 = np.radians(self.previous_gps.latitude), np.radians(self.previous_gps.longitude)
                    lat2, lon2 = np.radians(gps_data.latitude), np.radians(gps_data.longitude)
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    distance = R * c
                    vehicle_speed = distance / dt
            
            gps_info = {
                'latitude': gps_data.latitude,
                'longitude': gps_data.longitude,
                'heading': gps_data.heading,
                'accuracy': gps_data.accuracy,
                'vehicle_speed_ms': vehicle_speed,
                'vehicle_speed_kmh': vehicle_speed * 3.6
            }
        
        # Convert detections to tracks for visualization
        tracks = []
        for detection in detections:
            class MockTrack:
                def __init__(self, detection):
                    self.track_id = getattr(detection, 'track_id', 0)
                    self.state = 'confirmed'
                    self.detections = [detection]
                    self.hits = 1
                    self.age = 1
                    self.time_since_update = 0
                    self._bbox = detection.bbox
                    # Enhanced attributes
                    self.motion_compensated = getattr(detection, 'motion_compensated', False)
                    self.prediction_match = getattr(detection, 'prediction_match', False)
                    self.match_score = getattr(detection, 'match_score', 0.0)
                
                def to_tlbr(self):
                    return self._bbox
            
            track = MockTrack(detection)
            tracks.append(track)
        
        # Visualize with enhanced info
        return self.visualizer.visualize_frame(
            frame, detections, tracks, gps_info, frame_info
        )

    def _create_visualization_frame(self,
                                frame: np.ndarray,
                                detections: List[Detection],
                                gps_data: Optional[GPSData]) -> np.ndarray:
        """Enhanced visualization frame with motion and feature info"""
        
        vis_frame = frame.copy()
        
        # Draw detections with enhanced info
        for detection in detections:
            bbox = detection.bbox.astype(int)
            class_name = self.class_names[detection.class_id] if detection.class_id < len(self.class_names) else f"class_{detection.class_id}"
            
            # Choose color based on detection type
            if getattr(detection, 'motion_compensated', False):
                color = (0, 255, 255)  # Cyan for motion compensated
            elif getattr(detection, 'prediction_match', False):
                color = (255, 0, 255)  # Magenta for prediction match
            elif getattr(detection, 'reappearance_match', False):
                color = (0, 165, 255)  # Orange for reappearance
            else:
                color = (0, 255, 0)    # Green for normal
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Enhanced label with motion/feature info
            label_parts = [f"ID:{getattr(detection, 'track_id', '?')}"]
            label_parts.append(class_name)
            label_parts.append(f"{detection.score:.2f}")
            
            if getattr(detection, 'motion_compensated', False):
                label_parts.append("MC")  # Motion Compensated
            if getattr(detection, 'prediction_match', False):
                match_score = getattr(detection, 'match_score', 0)
                label_parts.append(f"P:{match_score:.2f}")  # Prediction match
            
            label = " ".join(label_parts)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(vis_frame, 
                        (bbox[0], bbox[1] - label_size[1] - 10),
                        (bbox[0] + label_size[0], bbox[1]),
                        color, -1)
            
            # Text
            cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Enhanced GPS and motion info overlay
        if gps_data:
            # Calculate vehicle speed
            vehicle_speed = 0.0
            if hasattr(self, 'previous_gps') and self.previous_gps:
                dt = gps_data.timestamp - self.previous_gps.timestamp
                if dt > 0:
                    R = 6378137.0
                    lat1, lon1 = np.radians(self.previous_gps.latitude), np.radians(self.previous_gps.longitude)
                    lat2, lon2 = np.radians(gps_data.latitude), np.radians(gps_data.longitude)
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    distance = R * c
                    vehicle_speed = distance / dt
            
            # GPS info
            gps_text = f"GPS: {gps_data.latitude:.6f}, {gps_data.longitude:.6f}"
            cv2.putText(vis_frame, gps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Vehicle speed
            speed_text = f"Speed: {vehicle_speed * 3.6:.1f} km/h"
            cv2.putText(vis_frame, speed_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Detection count with enhancement info
        motion_compensated = len([d for d in detections if getattr(d, 'motion_compensated', False)])
        prediction_matches = len([d for d in detections if getattr(d, 'prediction_match', False)])
        
        det_text = f"Det: {len(detections)} (MC:{motion_compensated}, P:{prediction_matches})"
        cv2.putText(vis_frame, det_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Enhanced track statistics
        track_stats = self.track_manager.get_statistics()
        stats_text = f"Tracks: {track_stats['active_tracks']} | Features: {track_stats['visual_features']['tracks_with_features']} | Pred: {track_stats['motion_prediction']['avg_prediction_accuracy']:.2f}"
        cv2.putText(vis_frame, stats_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return vis_frame

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced tracking statistics"""
        base_stats = self.get_statistics()
        track_stats = self.track_manager.get_statistics()
        
        enhanced_stats = {
            **base_stats,
            'enhanced_features': {
                'motion_prediction': track_stats['motion_prediction'],
                'visual_features': track_stats['visual_features'],
                'gps_enhanced_motion': track_stats.get('gps_enhanced', {}),
                'track_consolidation': {
                    'total_consolidations': track_stats['total_consolidations'],
                    'total_reappearances': track_stats['total_reappearances'],
                    'recovery_rate': track_stats['recovery_rate']
                }
            },
            'performance_metrics': {
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'gps_frames_processed': len([1 for _ in self.processing_times]),  # Simplified
                'motion_compensation_active': track_stats['motion_prediction']['camera_motion_detected'],
                'visual_matching_active': track_stats['visual_features']['feature_extraction_enabled']
            }
        }
        
        return enhanced_stats

    def _add_info_panel(self, frame_info: Dict) -> Dict:
        """Add motion prediction info to visualization panel"""
        enhanced_info = frame_info.copy()
        
        # Add motion prediction statistics
        motion_stats = self.track_manager.get_motion_statistics()
        
        enhanced_info.update({
            'motion_detected': motion_stats.get('motion_detection', {}).get('motion_detected', False),
            'prediction_matches': motion_stats.get('prediction_matches', 0),
            'camera_motion': motion_stats.get('motion_detection', {}).get('avg_translation', 0),
            'motion_compensation': 'ACTIVE' if motion_stats.get('motion_detection', {}).get('motion_detected') else 'INACTIVE'
        })
        
        return enhanced_info

    def process_video(self,
                     video_path: str,
                     gps_data: Optional[List[GPSData]] = None,
                     output_path: Optional[str] = None,
                     save_results: bool = True) -> Dict[str, Any]:
        """
        Process video with simplified tracking
        
        Args:
            video_path: Path to input video
            gps_data: Optional GPS data for synchronization
            output_path: Optional output video path
            save_results: Whether to save results
            
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Starting simplified tracking: {video_path}")
        
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
        
        # Setup video writer (if requested)
        out_writer = None
        if output_path and self.show_realtime:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Write only GPS frames, so use GPS frequency for output
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
                
                # Process frame
                frame_timestamp = current_frame_idx / fps
                detections = self._process_frame(frame, current_frame_idx, frame_timestamp, current_gps)
                
                # Add to output manager
                output_manager.add_frame_data(
                    frame_id=current_frame_idx,
                    timestamp=frame_timestamp,
                    detections=detections,
                    gps_data=current_gps
                )
                
                # Real-time visualization (GPS frames only)
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
                if processed_frames % 30 == 0:  # Every 30 processed frames
                    avg_time = np.mean(self.processing_times[-30:])
                    progress = current_frame_idx / total_frames * 100
                    
                    self.logger.info(
                        f"Progress: {progress:.1f}% | "
                        f"Processed: {processed_frames} | "
                        f"Skipped (GPS): {skipped_frames_gps} | "
                        f"Skipped (Static): {skipped_frames_static} | "
                        f"Avg time: {avg_time*1000:.1f}ms"
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
        self.logger.info(f"Track consolidations: {track_stats['total_consolidations']}")
        self.logger.info(f"Track reappearances: {track_stats['total_reappearances']}")
        
        # Save results
        if save_results:
            json_path, csv_path = output_manager.export_both()
            results['json_output'] = json_path
            results['csv_output'] = csv_path
            
            # Print summary
            output_manager.print_summary()
        
        return results
    
    def _convert_fixed_detections(self, fixed_detections: List[Dict], frame_id: int) -> List[Detection]:
        """Convert fixed detections to Detection objects"""
        detections = []
        # DEBUG: Log detection conversion
        if fixed_detections:
            self.logger.info(f"Frame {frame_id}: Converting {len(fixed_detections)} fixed detections")
            track_ids = [det['track_id'] for det in fixed_detections]
            self.logger.info(f"Frame {frame_id}: Track IDs: {track_ids}")
    
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
    
    def _extract_detections_from_results(self, results, frame_id: int) -> List[Detection]:
        """Extract Detection objects from YOLO results"""
        detections = []
        
        if not results or not results[0].boxes:
            return detections
        
        result = results[0]
        
        # Check if tracking is available
        if not hasattr(result.boxes, 'id') or result.boxes.id is None:
            self.logger.warning(f"Frame {frame_id}: No tracking IDs available")
            return detections
        
        # Extract data
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        # Create Detection objects
        for i, (box, score, cls_id, track_id) in enumerate(zip(boxes, scores, classes, track_ids)):
            detection = Detection(
                bbox=box,
                score=float(score),
                class_id=int(cls_id),
                frame_id=frame_id
            )
            detection.track_id = int(track_id)  # Add track ID
            detections.append(detection)
        
        return detections
    
    def _visualize_frame(self,
                        frame: np.ndarray,
                        detections: List[Detection],
                        frame_id: int,
                        gps_data: Optional[GPSData],
                        total_frames: int,
                        processed_frames: int) -> bool:
        """Visualize frame with real-time display"""
        
        # Prepare frame info
        frame_info = {
            'frame_idx': frame_id,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'gps_available': gps_data is not None,
            'track_consolidations': self.track_manager.get_statistics()['total_consolidations'],
            'track_reappearances': self.track_manager.get_statistics()['total_reappearances']
        }
        
        # Prepare GPS info
        gps_info = None
        if gps_data:
            gps_info = {
                'latitude': gps_data.latitude,
                'longitude': gps_data.longitude,
                'heading': gps_data.heading,
                'accuracy': gps_data.accuracy
            }
        
        # Convert detections to tracks for visualization
        tracks = []
        for detection in detections:
            # Create minimal track object for visualization
            class MockTrack:
                def __init__(self, detection):
                    self.track_id = detection.track_id
                    self.state = 'confirmed'
                    self.detections = [detection]
                    self.hits = 1  # Required by visualizer
                    self.age = 1   # Required by visualizer
                    self.time_since_update = 0  # Required by visualizer
                    self._bbox = detection.bbox
                
                def to_tlbr(self):
                    return self._bbox
            
            track = MockTrack(detection)
            tracks.append(track)
        
        # Visualize
        return self.visualizer.visualize_frame(
            frame, detections, tracks, gps_info, frame_info
        )
    
    def _create_visualization_frame(self,
                                   frame: np.ndarray,
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
        
        # Add GPS info overlay
        if gps_data:
            gps_text = f"GPS: {gps_data.latitude:.6f}, {gps_data.longitude:.6f}"
            cv2.putText(vis_frame, gps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add detection count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(vis_frame, det_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add track consolidation info
        track_stats = self.track_manager.get_statistics()
        consol_text = f"Consolidations: {track_stats['total_consolidations']} | Reappearances: {track_stats['total_reappearances']}"
        cv2.putText(vis_frame, consol_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return vis_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        track_stats = self.track_manager.get_statistics()
        
        return {
            'processed_frames': self.processed_frame_count,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'track_manager': track_stats,
            'model_classes': self.class_names
        }