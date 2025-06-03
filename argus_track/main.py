# argus_track/main.py (UPDATED FOR SIMPLIFIED TRACKING)

import argparse
import logging
from pathlib import Path
from typing import Optional
import time
import numpy as np

from argus_track.config import TrackerConfig
from argus_track import __version__
from argus_track.trackers.simplified_lightpost_tracker import SimplifiedLightPostTracker
from argus_track.utils import setup_logging


def main():
    """Main function for simplified tracking with ID consolidation"""
    parser = argparse.ArgumentParser(
        description=f"Argus Track: Simplified Light Post Tracking System v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic tracking with automatic GPS extraction
    python main.py input.mp4 --model model.pt --show-realtime
    
    # Batch processing without visualization
    python main.py input.mp4 --model model.pt --no-realtime
    
    # Custom output paths
    python main.py input.mp4 --model model.pt --json-output results.json --csv-output gps.csv
        """
    )
    
    # Basic arguments
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model file')
    
    # Output arguments
    parser.add_argument('--output-video', type=str, help='Path for output visualization video')
    parser.add_argument('--json-output', type=str, help='Path for JSON output file')
    parser.add_argument('--csv-output', type=str, help='Path for CSV output file')
    
    # Processing arguments
    parser.add_argument('--show-realtime', action='store_true', default=True,
                       help='Show real-time visualization (default: True)')
    parser.add_argument('--no-realtime', action='store_true',
                       help='Disable real-time visualization')
    
    # Tracking parameters
    parser.add_argument('--detection-conf', type=float, default=0.20,
                       help='Detection confidence threshold (default: 0.20)')
    parser.add_argument('--gps-interval', type=int, default=6,
                       help='GPS frame interval (default: 6 - every 6th frame)')
    
    # Track consolidation parameters
    parser.add_argument('--disable-consolidation', action='store_true',
                       help='Disable track ID consolidation')
    parser.add_argument('--disable-reappearance', action='store_true',
                       help='Disable track reappearance detection')
    parser.add_argument('--max-gap-frames', type=int, default=15,
                       help='Max frames to remember lost tracks (default: 15)')
    
    # Static car detection
    parser.add_argument('--disable-static-car', action='store_true',
                       help='Disable static car detection')
    
    # Logging
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input_video).exists():
        print(f"❌ Error: Input video not found: {args.input_video}")
        return 1
    
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return 1
    
    # Handle real-time display settings
    show_realtime = args.show_realtime and not args.no_realtime
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_file=args.log_file, level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Argus Track v{__version__} - Simplified Tracking")
    logger.info(f"📹 Input video: {args.input_video}")
    logger.info(f"🤖 Model: {args.model}")
    logger.info(f"📺 Real-time display: {show_realtime}")
    
    try:
        # Create configuration
        config = TrackerConfig.create_simplified_tracker()
        
        # Apply command line overrides
        config.detection_conf = args.detection_conf
        config.gps_frame_interval = args.gps_interval
        
        # Track consolidation settings
        if args.disable_consolidation:
            config.track_consolidation.enable_id_consolidation = False
            logger.info("🔧 ID consolidation disabled")
        
        if args.disable_reappearance:
            config.track_consolidation.enable_reappearance_detection = False
            logger.info("🔧 Reappearance detection disabled")
        
        config.track_consolidation.max_gap_frames = args.max_gap_frames
        
        # Static car detection
        if args.disable_static_car:
            config.enable_static_car_detection = False
            logger.info("🔧 Static car detection disabled")
        
        logger.info("📷 Configuration created")
        logger.info(f"   Detection confidence: {config.detection_conf}")
        logger.info(f"   GPS frame interval: {config.gps_frame_interval}")
        logger.info(f"   ID consolidation: {config.track_consolidation.enable_id_consolidation}")
        logger.info(f"   Reappearance detection: {config.track_consolidation.enable_reappearance_detection}")
        logger.info(f"   Max gap frames: {config.track_consolidation.max_gap_frames}")
        
        # Extract GPS data from video
        logger.info("🗺️ Extracting GPS data from video metadata...")
        gps_data = None
        
        try:
            from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos
            
            # Use the same video file for both left and right (monocular)
            gps_data, extraction_method = extract_gps_from_stereo_videos(
                args.input_video, args.input_video, method='auto'
            )
            
            if gps_data:
                logger.info(f"✅ Extracted {len(gps_data)} GPS points using {extraction_method}")
                
                # Log GPS data range for verification
                if len(gps_data) > 0:
                    start_gps = gps_data[0]
                    end_gps = gps_data[-1]
                    logger.info(f"🗺️ GPS range: ({start_gps.latitude:.6f}, {start_gps.longitude:.6f}) to "
                               f"({end_gps.latitude:.6f}, {end_gps.longitude:.6f})")
                    logger.info(f"⏱️ Time range: {start_gps.timestamp:.1f}s to {end_gps.timestamp:.1f}s")
            else:
                logger.warning("⚠️ No GPS data found in video metadata")
                logger.warning("   Processing will continue without GPS synchronization")
                
        except ImportError as e:
            logger.error(f"❌ GPS extraction dependencies missing: {e}")
            logger.error("   Install required packages: pip install beautifulsoup4 lxml")
            gps_data = None
        except Exception as e:
            logger.error(f"❌ GPS extraction failed: {e}")
            gps_data = None
        
        # Initialize simplified tracker
        tracker = SimplifiedLightPostTracker(
            config=config,
            model_path=args.model,
            show_realtime=show_realtime,
            display_size=(1280, 720)
        )
        
        # Show helpful tips
        if show_realtime:
            logger.info("🖥️  Real-time visualization controls:")
            logger.info("   - Press 'q' to quit")
            logger.info("   - Press 'p' to pause/resume") 
            logger.info("   - Press 's' to save screenshot")
            logger.info("   - Only GPS-synchronized frames will be displayed")
        
        # Show processing info
        if gps_data:
            effective_frames = len(gps_data)
            logger.info(f"📍 GPS-synchronized processing: {effective_frames} frames to process")
            logger.info("   → Frames will be processed at GPS frequency (every 6th frame)")
        else:
            logger.warning("⚠️ No GPS data: Processing all frames")
        
        # Process video
        start_time = time.time()
        
        results = tracker.process_video(
            video_path=args.input_video,
            gps_data=gps_data,
            output_path=args.output_video,
            save_results=not args.no_save
        )
        
        processing_time = time.time() - start_time
        
        # Print results
        logger.info("🎉 PROCESSING COMPLETE!")
        logger.info(f"⏱️  Total processing time: {processing_time:.1f} seconds")
        
        # Processing statistics
        logger.info("📊 PROCESSING STATISTICS:")
        logger.info(f"   Total frames in video: {results['total_frames']}")
        logger.info(f"   Processed frames: {results['processed_frames']}")
        logger.info(f"   Skipped (GPS sync): {results['skipped_frames_gps']}")
        logger.info(f"   Skipped (static car): {results['skipped_frames_static']}")
        logger.info(f"   Processing efficiency: {results['avg_fps']:.1f} FPS")
        
        # Track consolidation statistics
        track_stats = results['track_manager_stats']
        logger.info("🔧 TRACK CONSOLIDATION STATISTICS:")
        logger.info(f"   Active tracks: {track_stats['active_tracks']}")
        logger.info(f"   Total consolidations: {track_stats['total_consolidations']}")
        logger.info(f"   Total reappearances: {track_stats['total_reappearances']}")

        # Only log if the key exists
        if 'tracks_created' in track_stats:
            logger.info(f"   Tracks created: {track_stats['tracks_created']}")
        elif 'total_tracks_created' in track_stats:
            logger.info(f"   Total tracks created: {track_stats['total_tracks_created']}")
        
        # Output statistics
        output_stats = results['output_summary']
        logger.info("📄 OUTPUT STATISTICS:")
        logger.info(f"   Unique tracks: {output_stats['unique_tracks']}")
        logger.info(f"   Total detections: {output_stats['total_detections']}")
        logger.info(f"   Avg detections/frame: {output_stats['avg_detections_per_frame']:.1f}")
        logger.info(f"   Frames with GPS: {output_stats['frames_with_gps']}")
        
        # Class distribution
        if output_stats['class_distribution']:
            logger.info("🏷️  CLASS DISTRIBUTION:")
            for class_name, count in output_stats['class_distribution'].items():
                logger.info(f"   {class_name}: {count} detections")
        
        # Output files
        if not args.no_save:
            logger.info("📁 OUTPUT FILES:")
            
            if 'json_output' in results:
                json_path = results['json_output']
                if Path(json_path).exists():
                    file_size = Path(json_path).stat().st_size / (1024 * 1024)
                    logger.info(f"   📄 JSON data: {json_path} ({file_size:.1f} MB)")
                    
                    # Custom JSON output path
                    if args.json_output:
                        custom_path = Path(args.json_output)
                        Path(json_path).rename(custom_path)
                        logger.info(f"   📄 Moved to: {custom_path}")
            
            if 'csv_output' in results:
                csv_path = results['csv_output']
                if Path(csv_path).exists():
                    file_size = Path(csv_path).stat().st_size / 1024
                    logger.info(f"   📍 GPS CSV: {csv_path} ({file_size:.1f} KB)")
                    
                    # Custom CSV output path
                    if args.csv_output:
                        custom_path = Path(args.csv_output)
                        Path(csv_path).rename(custom_path)
                        logger.info(f"   📍 Moved to: {custom_path}")
            
            if args.output_video and Path(args.output_video).exists():
                file_size = Path(args.output_video).stat().st_size / (1024 * 1024)
                logger.info(f"   🎬 Visualization video: {args.output_video} ({file_size:.1f} MB)")
        
        # Success summary
        logger.info("🏁 SUCCESS SUMMARY:")
        logger.info(f"   ✅ {results['processed_frames']} frames processed")
        logger.info(f"   ✅ {track_stats['total_consolidations']} ID consolidations applied")
        logger.info(f"   ✅ {track_stats['total_reappearances']} track reappearances handled")
        logger.info(f"   ✅ {output_stats['unique_tracks']} unique objects tracked")
        
        if gps_data:
            logger.info(f"   ✅ GPS data synchronized for {output_stats['frames_with_gps']} frames")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ Processing interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())