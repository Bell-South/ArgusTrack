import argparse
import logging
from pathlib import Path
from typing import Optional
import time
import numpy as np

from argus_track.config import TrackerConfig
from argus_track import (
    YOLOv11Detector,
    __version__
)
from argus_track.trackers.lightpost_tracker import EnhancedLightPostTracker
from argus_track.utils import setup_logging, load_gps_data

def main():
    """Fixed main function with GPS extraction"""
    parser = argparse.ArgumentParser(
        description=f"Argus Track: Light Post Tracking System v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Real-time visualization
                python main.py input.mp4 --detector yolov11 --model model.pt --show-realtime
                
                # With GPS data
                python main.py input.mp4 --detector yolov11 --model model.pt --gps gps.csv
            """
    )
    
    # Basic arguments
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--model', type=str, help='Path to detection model file')
    parser.add_argument('--output', type=str, help='Path for output video')
    parser.add_argument('--target-classes', nargs="*", default=None, 
                        help="Target class names")
    
    # GPS arguments (removed)
    
    # Visualization
    parser.add_argument('--show-realtime', action='store_true',
                       help='Show real-time visualization')
    
    # Tracking parameters (optional overrides)
    parser.add_argument('--track-thresh', type=float, default=None,
                       help='Detection confidence threshold')
    parser.add_argument('--match-thresh', type=float, default=None,
                       help='IoU threshold for matching')
    
    # Logging
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input_video).exists():
        print(f"❌ Error: Input video not found: {args.input_video}")
        return 1
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_file=args.log_file, level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Argus Track v{__version__}")
    logger.info(f"📹 Input video: {args.input_video}")
    
    try:
        # Create configuration
        config = TrackerConfig.create_ultralytics_optimized()
        logger.info("📷 Using standard configuration")

        logger.info("🗺️ Extracting GPS data from video metadata...")
        gps_data = None
        
        try:
            from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos
            
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
                logger.warning("   Make sure your GoPro video has GPS metadata")
                
        except ImportError as e:
            logger.error(f"❌ GPS extraction dependencies missing: {e}")
            logger.error("   Install required packages: pip install beautifulsoup4 lxml")
            gps_data = None
        except Exception as e:
            logger.error(f"❌ GPS extraction failed: {e}")
            gps_data = None
        
        # ===== END GPS EXTRACTION =====
        
        # Initialize tracker with YOLOv11 model path
        tracker = EnhancedLightPostTracker(
            config=config,
            model_path=args.model,
            show_realtime=args.show_realtime,
            display_size=(1280, 720)
        )
        
        # Show helpful tips
        if args.show_realtime:
            logger.info("🖥️  Real-time visualization controls:")
            logger.info("   - Press 'q' to quit")
            logger.info("   - Press 'p' to pause/resume") 
            logger.info("   - Press 's' to save screenshot")

        # Show GPS status before processing
        if gps_data:
            logger.info(f"📍 GPS-enabled processing: {len(gps_data)} GPS points available")
            logger.info("   → Objects will be geolocated using GPS + depth estimation")
        else:
            logger.warning("⚠️ GPS-disabled processing: No GPS data available")
            logger.warning("   → Objects will be tracked but NOT geolocated")

        # Process video with GPS data
        start_time = time.time()
        
        tracks = tracker.process_video(
            video_path=args.input_video,
            gps_data=gps_data,  # ← THIS IS THE KEY FIX!
            output_path=args.output,
            save_results=not args.no_save,
            resolution_scale=1.0
        )
        
        processing_time = time.time() - start_time
        
        # Print results
        logger.info("🎉 PROCESSING COMPLETE!")
        logger.info(f"⏱️  Processing time: {processing_time:.1f} seconds")
        
        # Get statistics
        if hasattr(tracker, 'get_enhanced_tracking_statistics'):
            stats = tracker.get_enhanced_tracking_statistics()
        else:
            stats = tracker.get_track_statistics()
        
        logger.info("📊 TRACKING STATISTICS:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.3f}")
            else:
                logger.info(f"   {key}: {value}")
        
        # Check for geolocated objects
        if hasattr(tracker, 'track_locations') and tracker.track_locations:
            logger.info("📍 GEOLOCATED OBJECTS:")
            for track_id, location in tracker.track_locations.items():
                logger.info(
                    f"   Track {track_id}: ({location.latitude:.6f}, {location.longitude:.6f}) "
                    f"accuracy: {location.accuracy:.1f}m, reliability: {location.reliability:.2f}"
                )
        
        # Check for GPS-based locations (NEW)
        elif hasattr(tracker, 'track_gps_locations') and tracker.track_gps_locations:
            logger.info("📍 GPS-BASED OBJECT LOCATIONS:")
            total_locations = 0
            for track_id, locations in tracker.track_gps_locations.items():
                if len(locations) >= 3:  # Only show tracks with multiple detections
                    avg_lat = sum(loc['latitude'] for loc in locations) / len(locations)
                    avg_lon = sum(loc['longitude'] for loc in locations) / len(locations)
                    avg_depth = sum(loc['depth'] for loc in locations) / len(locations)
                    confidence = sum(loc['confidence'] for loc in locations) / len(locations)
                    
                    logger.info(
                        f"   Track {track_id}: ({avg_lat:.6f}, {avg_lon:.6f}) "
                        f"depth: {avg_depth:.1f}m, confidence: {confidence:.2f}, "
                        f"detections: {len(locations)}"
                    )
                    total_locations += 1
            
            if total_locations > 0:
                logger.info(f"🎯 SUCCESS: {total_locations} objects with GPS coordinates!")
            else:
                logger.info("📍 No objects with sufficient GPS tracking")
        else:
            logger.info("📍 No static objects found for geolocation")
            if not gps_data:
                logger.info("   💡 Tip: Provide GPS data to enable geolocation")
        
        # Output file summary
        video_stem = Path(args.input_video).stem
        logger.info("📁 OUTPUT FILES:")
        
        output_files = [
            (f"{video_stem}.json", "Tracking results"),
            (f"{video_stem}.geojson", "Geolocation data"),
            (f"{video_stem}.csv", "GPS data"),
        ]
        
        if args.output:
            output_files.append((args.output, "Visualization video"))
        
        for filename, description in output_files:
            if Path(filename).exists():
                file_size = Path(filename).stat().st_size / (1024 * 1024)
                logger.info(f"   📄 {description}: {filename} ({file_size:.1f} MB)")
        
        # Final success message
        total_tracks = len(tracks) if tracks else 0
        geolocated_count = len(tracker.track_locations) if hasattr(tracker, 'track_locations') else 0
        
        # Also check GPS-based locations
        if hasattr(tracker, 'track_gps_locations'):
            gps_based_count = len([t for t, locs in tracker.track_gps_locations.items() if len(locs) >= 3])
            geolocated_count = max(geolocated_count, gps_based_count)
        
        logger.info(f"🏁 SUCCESS: {total_tracks} tracks processed, {geolocated_count} objects geolocated")
        
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