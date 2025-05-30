"""Main entry point for Argus Track with Ultralytics ByteTrack integration"""

import argparse
import logging
from pathlib import Path
import os
import sys
import time
import numpy as np
import cv2

# Add parent directory to path to allow importing from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_track import (
    TrackerConfig,
    __version__
)
from argus_track.utils import setup_logging, load_gps_data
from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos, save_gps_to_csv

# Import our new Ultralytics tracker
from ultralytics_tracker import UltralyticsTracker


def main():
    """Main function for Argus Track with Ultralytics ByteTrack integration"""
    parser = argparse.ArgumentParser(
        description=f"Argus Track: Enhanced Light Post Tracking with Ultralytics ByteTrack v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Basic tracking with Ultralytics ByteTrack
                python ultralytics_main.py input.mp4 --model yolov8n.pt --show-realtime
                
                # Tracking with GPS data
                python ultralytics_main.py input.mp4 --model yolov8n.pt --gps gps_data.csv --show-realtime
                
                # Tracking with automatic GPS extraction
                python ultralytics_main.py input.mp4 --model yolov8n.pt --auto-gps --show-realtime
                
                # Extract GPS only
                python ultralytics_main.py --extract-gps-only input.mp4 --output gps_data.csv
            """
    )
    
    # Video input arguments
    parser.add_argument('input_video', type=str, nargs='?',
                       help='Path to input video file')
    
    # GPS extraction options
    parser.add_argument('--auto-gps', action='store_true',
                       help='Automatically extract GPS data from video')
    parser.add_argument('--gps-method', type=str, default='auto',
                       choices=['auto', 'exiftool', 'gopro_api'],
                       help='GPS extraction method (default: auto)')
    parser.add_argument('--extract-gps-only', action='store_true',
                       help='Only extract GPS data, do not run tracking')
    parser.add_argument('--save-gps-csv', action='store_true',
                       help='Save extracted GPS data to CSV file')
    
    # Model and GPS data
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model file (.pt)')
    parser.add_argument('--gps', type=str,
                       help='Path to GPS data CSV file')
    
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
    parser.add_argument('--track-thresh', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--match-thresh', type=float, default=0.8,
                       help='IoU threshold for matching')
    parser.add_argument('--track-buffer', type=int, default=50,
                       help='Number of frames to keep lost tracks')
    parser.add_argument('--static-threshold', type=float, default=8.0,
                       help='Threshold for static object classification')
    
    # Performance options
    parser.add_argument('--resolution-scale', type=float, default=1.0,
                       help='Scale factor for input resolution (0.5 = half size)')
    
    # Real-time visualization options
    parser.add_argument('--show-realtime', action='store_true',
                       help='Show real-time detection and tracking visualization')
    parser.add_argument('--display-size', nargs=2, type=int, default=[1280, 720],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Real-time display window size (default: 1280x720)')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.input_video and not args.extract_gps_only:
        parser.error("Must provide input_video")
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file or f"{Path(args.input_video).stem}_ultralytics.log" if args.input_video else None
    setup_logging(log_file=log_file, level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Argus Track with Ultralytics ByteTrack v{__version__}")
    
    # Handle GPS extraction only mode
    if args.extract_gps_only and args.input_video:
        logger.info("GPS extraction only mode")
        
        gps_data, method_used = extract_gps_from_stereo_videos(
            args.input_video, args.input_video, args.gps_method
        )
        
        if gps_data:
            output_path = args.output or f"{Path(args.input_video).stem}_gps_data.csv"
            save_gps_to_csv(gps_data, output_path)
            
            logger.info(f"✅ Successfully extracted {len(gps_data)} GPS points using {method_used}")
            logger.info(f"Saved GPS data to: {output_path}")
            return 0
        else:
            logger.error("❌ No GPS data could be extracted from the video")
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
            static_threshold=args.static_threshold
        )
    
    # Check if model file exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    # Process videos
    try:
        # Real-time visualization is controlled by the command-line flag
        show_realtime = args.show_realtime
        
        # Create display size tuple for the visualizer
        display_size = tuple(args.display_size) if args.display_size else (1280, 720)
        
        # Initialize the Ultralytics tracker
        tracker = UltralyticsTracker(
            config=config,
            model_path=args.model,
            show_realtime=show_realtime,
            display_size=display_size,
            confidence_threshold=args.track_thresh,
            iou_threshold=0.5  # Default IoU threshold
        )
        
        if show_realtime:
            logger.info("Initialized tracker with real-time visualization")
            logger.info("Real-time controls:")
            logger.info("  Press 'q' to quit processing")
            logger.info("  Press 'p' to pause/resume")
            logger.info("  Press 's' to save screenshot")
        else:
            logger.info("Initialized tracker (no real-time display)")
        
        # Load GPS data if provided or extract automatically
        gps_data = None
        if args.gps:
            try:
                gps_data = load_gps_data(args.gps)
                logger.info(f"Loaded {len(gps_data)} GPS data points from {args.gps}")
            except Exception as e:
                logger.error(f"Failed to load GPS data: {e}")
        elif args.auto_gps:
            logger.info("Extracting GPS data automatically...")
            gps_data, method_used = extract_gps_from_stereo_videos(
                args.input_video, args.input_video, args.gps_method
            )
            
            if gps_data:
                logger.info(f"Extracted {len(gps_data)} GPS data points using {method_used}")
                if args.save_gps_csv:
                    gps_csv_path = f"{Path(args.input_video).stem}_gps_data.csv"
                    save_gps_to_csv(gps_data, gps_csv_path)
                    logger.info(f"Saved GPS data to: {gps_csv_path}")
            else:
                logger.warning("No GPS data could be extracted, continuing without GPS")
        
        # Process video
        video_path = args.input_video
        process_start_time = time.time()
        
        # Determine output path
        output_path = None
        if args.output and not args.no_save_video:
            output_path = args.output
        elif not args.no_save_video:
            output_path = f"{Path(video_path).stem}_tracked.mp4"
        
        # Process video with Ultralytics ByteTrack
        tracks = tracker.process_video(
            video_path=video_path,
            gps_data=gps_data,
            output_path=output_path,
            save_results=not args.no_save,
            resolution_scale=args.resolution_scale
        )
        
        processing_time = time.time() - process_start_time
        logger.info(f"Video processing completed in {processing_time:.2f}s")
        
        # Print statistics
        stats = tracker.get_statistics()
        logger.info("=== Tracking Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            elif isinstance(value, dict):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("🎉 Processing complete!")
        
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