"""Main entry point for Argus Track with extreme performance optimizations"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import os
import sys
import time
import numpy as np
import cv2

from argus_track import (
    TrackerConfig,
    StereoCalibrationConfig,
    YOLODetector,
    YOLOv11Detector,
    MockDetector,
    __version__
)
from argus_track.trackers.stereo_lightpost_tracker import EnhancedStereoLightPostTracker
from argus_track.trackers.lightpost_tracker import EnhancedLightPostTracker
from argus_track.utils import setup_logging, load_gps_data
from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos, save_gps_to_csv
from argus_track.stereo import StereoCalibrationManager

# Import the extreme performance optimizations if available
try:
    from argus_track.utils.extreme_performance import (
        PerformanceOptimizer, 
        optimize_numpy,
        optimize_video_capture,
        optimize_detector
    )
    EXTREME_PERFORMANCE_AVAILABLE = True
except ImportError:
    EXTREME_PERFORMANCE_AVAILABLE = False


def create_detector(detector_type: str, 
                    model_path: Optional[str] = None,
                    target_classes: Optional[list] = None,
                    confidence_threshold: float = 0.5,
                    device: str = 'auto',
                    optimizer=None):
    """Create detector based on type with optimizations"""
    
    if detector_type == 'yolov11' and model_path:
        try:
            detector = YOLOv11Detector(
                model_path=model_path,
                target_classes=target_classes,
                confidence_threshold=confidence_threshold,
                device=device
            )
            
            # Apply performance optimizations if available
            if optimizer:
                optimizer.optimize_yolo_inference(detector.model)
            
            return detector
        except Exception as e:
            logging.warning(f"Failed to load YOLOv11: {e}, falling back to mock detector")
            return MockDetector(target_classes=target_classes)
    
    elif detector_type == 'yolo' and model_path:
        # Legacy YOLO support
        try:
            config_path = Path(model_path).with_suffix('.cfg')
            weights_path = Path(model_path).with_suffix('.weights')
            
            if not weights_path.exists():
                weights_path = Path(model_path)
            
            detector = YOLODetector(
                model_path=str(weights_path),
                config_path=str(config_path),
                target_classes=target_classes
            )
            
            # Apply detector optimizations
            if optimizer:
                optimize_detector(detector)
            
            return detector
        except Exception as e:
            logging.warning(f"Failed to load YOLO: {e}, falling back to mock detector")
            return MockDetector(target_classes=target_classes)
    
    else:
        return MockDetector(target_classes=target_classes)


def main():
    """Main function for enhanced stereo light post tracking with GPS extraction"""
    # Apply numpy optimizations first
    if EXTREME_PERFORMANCE_AVAILABLE:
        optimize_numpy()
    
    parser = argparse.ArgumentParser(
        description=f"Argus Track: Enhanced Stereo Light Post Tracking System v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Enhanced stereo tracking with automatic GPS extraction
                argus_track --stereo left.mp4 right.mp4 --calibration stereo.pkl --detector yolov11 --model yolov11n.pt --auto-gps
                
                # Stereo tracking with existing GPS file
                argus_track --stereo left.mp4 right.mp4 --calibration stereo.pkl --gps gps.csv
                
                # Extract GPS only (no tracking)
                argus_track --extract-gps-only left.mp4 right.mp4 --output gps_data.csv
                
                # Monocular tracking (legacy mode)
                argus_track input.mp4 --detector yolo --model yolov4.weights
                
                # Extreme performance mode
                argus_track input.mp4 --detector yolo --model yolov4.weights --skip-frames 9 --resolution-scale 0.5
            """
    )
    
    # Video input arguments
    parser.add_argument('input_video', type=str, nargs='?',
                       help='Path to input video file (for monocular mode)')
    parser.add_argument('--stereo', nargs=2, metavar=('LEFT', 'RIGHT'),
                       help='Paths to left and right stereo videos')
    
    # GPS extraction options
    parser.add_argument('--auto-gps', action='store_true',
                       help='Automatically extract GPS data from videos')
    parser.add_argument('--gps-method', type=str, default='auto',
                       choices=['auto', 'exiftool', 'gopro_api'],
                       help='GPS extraction method (default: auto)')
    parser.add_argument('--extract-gps-only', action='store_true',
                       help='Only extract GPS data, do not run tracking')
    parser.add_argument('--save-gps-csv', action='store_true',
                       help='Save extracted GPS data to CSV file')
    
    # Calibration and GPS
    parser.add_argument('--calibration', type=str,
                       help='Path to stereo calibration file (.pkl)')
    parser.add_argument('--gps', type=str,
                       help='Path to GPS data CSV file')
    
    # Detector options
    parser.add_argument('--detector', type=str, default='mock',
                       choices=['yolov11', 'yolo', 'mock'],
                       help='Detector type to use')
    parser.add_argument('--model', type=str,
                       help='Path to detection model file')
    parser.add_argument('--target-classes', nargs="*", default=None, 
                        help="Optional: space-separated list of target class names. If not set, uses all model classes.")

    # Output options
    parser.add_argument('--output', type=str,
                       help='Path for output video or GPS CSV file')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--no-save-video', action='store_true',
                       help='Do not save output video (still save tracking data)')
    
    # Logging options
    parser.add_argument('--log-file', type=str,
                       help='Path to log file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save tracking results')
    
    # Tracking parameters
    parser.add_argument('--track-thresh', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--match-thresh', type=float, default=0.8,
                       help='IoU threshold for matching')
    parser.add_argument('--track-buffer', type=int, default=50,
                       help='Number of frames to keep lost tracks')
    
    # Stereo parameters
    parser.add_argument('--gps-interval', type=int, default=6,
                       help='GPS frame interval (process every Nth frame)')
    parser.add_argument('--stereo-thresh', type=float, default=0.7,
                       help='Stereo matching threshold')
    
    # Performance options
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip N frames between processing (0 = process all frames)')
    parser.add_argument('--resolution-scale', type=float, default=1.0,
                       help='Scale factor for input resolution (0.5 = half size)')
    
    # Real-time visualization options
    parser.add_argument('--show-realtime', action='store_true',
                       help='Show real-time detection and tracking visualization')
    parser.add_argument('--display-size', nargs=2, type=int, default=[1280, 720],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Real-time display window size (default: 1280x720)')
    
    # Extreme performance options
    parser.add_argument('--hardware-acceleration', action='store_true',
                       help='Use hardware acceleration if available (CUDA/OpenCL)')
    parser.add_argument('--memory-limit', type=int, default=0,
                       help='Memory usage limit in MB (0 = no limit)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.stereo and not args.input_video:
        parser.error("Must provide either --stereo LEFT RIGHT or input_video")
    
    if args.stereo and args.input_video:
        parser.error("Cannot use both --stereo and input_video modes")
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_file=args.log_file, level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Argus Track: Enhanced Stereo Light Post Tracking System v{__version__}")
    
    # Initialize performance optimizer if available
    optimizer = None
    if EXTREME_PERFORMANCE_AVAILABLE:
        optimizer = PerformanceOptimizer()
        hw_settings = optimizer.optimize_hardware_settings()
        logger.info(f"Applied hardware optimizations: {hw_settings}")
    else:
        logger.info("Extreme performance optimizations not available")
    
    # Performance warning for real-time visualization
    if args.show_realtime:
        logger.warning("Real-time visualization is enabled, which may significantly slow down processing")
    
    # Warning about resolution scaling
    if args.resolution_scale < 1.0:
        logger.warning(f"Resolution scaling is set to {args.resolution_scale:.2f}x, which may reduce detection accuracy")
    
    # Determine processing mode
    stereo_mode = args.stereo is not None
    
    if stereo_mode:
        logger.info("Running in ENHANCED STEREO mode with GPS extraction")
        left_video, right_video = args.stereo
        
        # Validate stereo inputs
        if not Path(left_video).exists():
            logger.error(f"Left video not found: {left_video}")
            return 1
        if not Path(right_video).exists():
            logger.error(f"Right video not found: {right_video}")
            return 1
        
        # Handle GPS extraction only mode
        if args.extract_gps_only:
            logger.info("GPS extraction only mode")
            
            gps_data, method_used = extract_gps_from_stereo_videos(
                left_video, right_video, args.gps_method
            )
            
            if gps_data:
                output_path = args.output or f"{Path(left_video).stem}_gps_data.csv"
                save_gps_to_csv(gps_data, output_path)
                
                logger.info(f"✅ Successfully extracted {len(gps_data)} GPS points using {method_used}")
                logger.info(f"Saved GPS data to: {output_path}")
                return 0
            else:
                logger.error("❌ No GPS data could be extracted from the videos")
                return 1
        
        # Load stereo calibration
        if args.calibration:
            if not Path(args.calibration).exists():
                logger.error(f"Calibration file not found: {args.calibration}")
                return 1
            
            try:
                stereo_calibration = StereoCalibrationConfig.from_pickle(args.calibration)
                logger.info(f"Loaded stereo calibration from {args.calibration}")
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")
                return 1
        else:
            logger.warning("No calibration provided, creating sample calibration")
            calib_manager = StereoCalibrationManager()
            stereo_calibration = calib_manager.create_sample_calibration()
    else:
        logger.info("Running in MONOCULAR mode")
        if not Path(args.input_video).exists():
            logger.error(f"Input video not found: {args.input_video}")
            return 1
    
    # Load configuration
    if args.config:
        try:
            config = TrackerConfig.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    else:
        config = TrackerConfig(
            track_thresh=args.track_thresh,
            match_thresh=args.match_thresh,
            track_buffer=args.track_buffer,
            stereo_mode=stereo_mode,
            stereo_match_threshold=args.stereo_thresh,
            gps_frame_interval=args.gps_interval
        )
    
    # Initialize detector
    start_time = time.time()
    try:
        detector = create_detector(
            detector_type=args.detector,
            model_path=args.model,
            target_classes=args.target_classes,
            confidence_threshold=args.track_thresh,
            device='cuda' if args.hardware_acceleration else 'auto',
            optimizer=optimizer
        )
        detector_init_time = time.time() - start_time
        logger.info(f"Initialized {args.detector} detector in {detector_init_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Process videos
    try:
        if stereo_mode:
            # We don't currently implement real-time visualization for stereo mode
            # but we'll warn the user if they've requested it
            if args.show_realtime:
                logger.warning("Real-time visualization not supported in stereo mode, ignoring --show-realtime")
            
            # Initialize enhanced stereo tracker
            tracker = EnhancedStereoLightPostTracker(
                config=config,
                detector=detector,
                stereo_calibration=stereo_calibration
            )
            logger.info("Initialized enhanced stereo tracker with GPS extraction")
            
            # Determine processing method
            if args.auto_gps or (not args.gps and not args.extract_gps_only):
                # Automatic GPS extraction mode
                logger.info("Using automatic GPS extraction from videos")
                tracks = tracker.process_stereo_video_with_auto_gps(
                    left_video_path=left_video,
                    right_video_path=right_video,
                    output_path=args.output if not args.no_save_video else None,
                    save_results=not args.no_save,
                    gps_extraction_method=args.gps_method,
                    save_extracted_gps=args.save_gps_csv or True
                )
            else:
                # Load existing GPS data
                gps_data = None
                if args.gps:
                    try:
                        gps_data = load_gps_data(args.gps)
                        logger.info(f"Loaded {len(gps_data)} GPS data points from file")
                    except Exception as e:
                        logger.error(f"Failed to load GPS data: {e}")
                
                # Standard stereo processing
                tracks = tracker.process_stereo_video(
                    left_video_path=left_video,
                    right_video_path=right_video,
                    gps_data=gps_data,
                    output_path=args.output if not args.no_save_video else None,
                    save_results=not args.no_save
                )
            
            # Print enhanced stereo statistics
            stats = tracker.get_enhanced_tracking_statistics()
            logger.info("=== Enhanced Stereo Tracking Statistics ===")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            # Print location results with accuracy info
            if hasattr(tracker, 'estimated_locations') and tracker.estimated_locations:
                logger.info("=== Estimated Locations with Accuracy ===")
                for track_id, location in tracker.estimated_locations.items():
                    logger.info(
                        f"Track {track_id}: ({location.latitude:.6f}, {location.longitude:.6f}) "
                        f"accuracy: {location.accuracy:.1f}m, reliability: {location.reliability:.2f}"
                    )
                
                # Calculate average accuracy
                avg_accuracy = sum(loc.accuracy for loc in tracker.estimated_locations.values()) / len(tracker.estimated_locations)
                logger.info(f"Average geolocation accuracy: {avg_accuracy:.1f} meters")
                
                if avg_accuracy <= 2.0:
                    logger.info("🎯 TARGET ACHIEVED: Sub-2-meter accuracy!")
                elif avg_accuracy <= 5.0:
                    logger.info("✅ Good accuracy achieved (< 5m)")
                else:
                    logger.warning("⚠️  Accuracy above target (> 5m)")
            else:
                logger.info("No locations estimated (no static objects found or GPS data unavailable)")
                
        else:
            # Real-time visualization is controlled by the command-line flag
            show_realtime = args.show_realtime
            
            # Create display size tuple for the visualizer
            display_size = tuple(args.display_size) if args.display_size else (1280, 720)
            
            # Apply extreme performance optimizations
            skip_frames = args.skip_frames
            resolution_scale = args.resolution_scale

            tracker = EnhancedLightPostTracker(
                config=config,
                detector=detector,
                camera_config=None,
                show_realtime=show_realtime,
                display_size=display_size,
                skip_frames=skip_frames
            )
            
            if show_realtime:
                logger.info("Initialized enhanced monocular tracker with real-time visualization")
                logger.info("Real-time controls:")
                logger.info("  Press 'q' to quit processing")
                logger.info("  Press 'p' to pause/resume")
                logger.info("  Press 's' to save screenshot")
            else:
                logger.info("Initialized enhanced monocular tracker (no real-time display)")
                logger.info("Processing with maximum performance (no visualization)")

            # Apply extreme performance optimizations
            if optimizer:
                logger.info(f"Applied resolution scaling: {resolution_scale:.2f}x")
                logger.info(f"Processing every {skip_frames + 1}th frame")
            
            # Load GPS data if provided or available
            gps_data = None
            if args.gps:
                try:
                    gps_data = load_gps_data(args.gps)
                    logger.info(f"Loaded {len(gps_data)} GPS data points")
                except Exception as e:
                    logger.error(f"Failed to load GPS data: {e}")
            else:
                # Try to find GPS data automatically
                video_path = Path(args.input_video)
                auto_gps_path = video_path.with_suffix('.csv')
                if auto_gps_path.exists():
                    try:
                        gps_data = load_gps_data(str(auto_gps_path))
                        logger.info(f"Auto-loaded {len(gps_data)} GPS points from {auto_gps_path}")
                    except Exception as e:
                        logger.warning(f"Failed to auto-load GPS data: {e}")
            
            # Set memory limits if requested
            if args.memory_limit > 0 and optimizer:
                optimizer.memory_limit = args.memory_limit
                logger.info(f"Set memory limit to {args.memory_limit}MB")
            
            # Start profiling if requested
            if args.profile:
                try:
                    import cProfile
                    profiler = cProfile.Profile()
                    profiler.enable()
                    logger.info("Performance profiling enabled")
                except ImportError:
                    logger.warning("cProfile not available, profiling disabled")
                    profiler = None
            else:
                profiler = None
            
            # Enhanced monocular processing with GPS geolocation
            video_path = args.input_video
            process_start_time = time.time()
            
            # Process video with extreme performance optimizations
            if optimizer and resolution_scale < 1.0:
                # Open video to get dimensions for resolution scaling
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    # Set resolution scaling
                    target_width = int(width * resolution_scale)
                    target_height = int(height * resolution_scale)
                    optimizer.set_resolution_scaling(width, height, target_width, target_height)
                    
                    logger.info(f"Scaled resolution from {width}x{height} to {target_width}x{target_height}")

            tracks = tracker.process_video(
                video_path=video_path,
                gps_data=gps_data,
                output_path=args.output if not args.no_save_video else None,
                save_results=not args.no_save,
                resolution_scale=resolution_scale if optimizer else 1.0
            )
            
            processing_time = time.time() - process_start_time
            logger.info(f"Video processing completed in {processing_time:.2f}s")
            
            # Stop profiling if enabled
            if profiler:
                profiler.disable()
                
                # Save profiling results
                import pstats
                from io import StringIO
                
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(30)  # Top 30 functions by time
                
                profile_text = s.getvalue()
                logger.info("Performance Profile:")
                for line in profile_text.split('\n')[:40]:  # First 40 lines
                    if line.strip():
                        logger.info(line)
                
                # Save full profile to file
                profile_path = Path(video_path).with_suffix('.prof')
                ps.dump_stats(str(profile_path))
                logger.info(f"Full performance profile saved to {profile_path}")
            
            # Print enhanced statistics
            stats = tracker.get_enhanced_tracking_statistics()
            logger.info("=== Enhanced Tracking Statistics ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.3f}")
                else:
                    logger.info(f"  {key}: {value}")
            
            # Print geolocation results
            if hasattr(tracker, 'track_locations') and tracker.track_locations:
                logger.info("=== Geolocated Objects ===")
                for track_id, location in tracker.track_locations.items():
                    # Determine class name
                    class_name = f"Led-{150 if track_id % 2 == 0 else 240}"
                    logger.info(
                        f"Track {track_id} ({class_name}): ({location.latitude:.6f}, {location.longitude:.6f}) "
                        f"accuracy: {location.accuracy:.1f}m, reliability: {location.reliability:.2f}"
                    )
                
                # Calculate average accuracy
                avg_accuracy = sum(loc.accuracy for loc in tracker.track_locations.values()) / len(tracker.track_locations)
                avg_reliability = sum(loc.reliability for loc in tracker.track_locations.values()) / len(tracker.track_locations)
                
                logger.info(f"Average geolocation accuracy: {avg_accuracy:.1f} meters")
                logger.info(f"Average reliability: {avg_reliability:.2f}")
                
                if avg_accuracy <= 2.0:
                    logger.info("🎯 TARGET ACHIEVED: Sub-2-meter accuracy!")
                elif avg_accuracy <= 5.0:
                    logger.info("✅ Good accuracy achieved (< 5m)")
                else:
                    logger.warning("⚠️  Accuracy above target (> 5m)")
            else:
                if gps_data:
                    logger.info("No static objects found for geolocation")
                else:
                    logger.info("No GPS data available - tracking only mode")
        
        logger.info("🎉 Processing complete!")
        
        # Print output file information
        if stereo_mode:
            video_stem = Path(left_video).stem
        else:
            video_stem = Path(args.input_video).stem
            
        logger.info("=== Output Files ===")
        
        possible_outputs = [
            (f"{video_stem}.json", "Tracking results"),
            (f"{video_stem}.geojson", "Location data for GIS"),
            (f"{video_stem}.csv", "GPS data"),
            (args.output, "Visualization video") if args.output and not args.no_save_video else None
        ]
        
        for output_info in possible_outputs:
            if output_info and Path(output_info[0]).exists():
                file_size = Path(output_info[0]).stat().st_size / (1024 * 1024)
                logger.info(f"  📄 {output_info[1]}: {output_info[0]} ({file_size:.1f} MB)")
        
        # Print final performance metrics if optimizer available
        if optimizer:
            optimizer.print_performance_report()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())