# argus_track/main.py

import argparse
import logging
import time
from pathlib import Path

from argus_track import __version__
from argus_track.config import TrackerConfig
from argus_track.trackers.unique_tracker import UnifiedLightPostTracker
from argus_track.utils import setup_logging


def main():
    """Main function for unified tracking with enhanced GPS heading calculation"""
    parser = argparse.ArgumentParser(
        description=f"Argus Track: Unified Light Post Tracking System v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic tracking with enhanced GPS and real-time display
    python main.py input.mp4 --model model.pt --show-realtime
    
    # Batch processing with enhanced GPS calculation
    python main.py input.mp4 --model model.pt --no-realtime
        """,
    )

    # Basic arguments
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to YOLO model file"
    )

    # Output arguments
    parser.add_argument(
        "--output-video", type=str, help="Path for output visualization video"
    )
    parser.add_argument("--json-output", type=str, help="Path for JSON output file")
    parser.add_argument("--csv-output", type=str, help="Path for CSV output file")

    # Processing arguments
    parser.add_argument(
        "--show-realtime",
        action="store_true",
        default=False,
        help="Show real-time visualization (default: True)",
    )
    parser.add_argument(
        "--no-realtime", action="store_true", help="Disable real-time visualization"
    )

    # Tracking parameters
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.20,
        help="Detection confidence threshold (default: 0.20)",
    )
    parser.add_argument(
        "--gps-interval",
        type=int,
        default=6,
        help="GPS frame interval (default: 6 - every 6th frame)",
    )

    # Static car detection
    parser.add_argument(
        "--disable-static-car", action="store_true", help="Disable static car detection"
    )

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    parser.add_argument("--no-save", action="store_true", help="Do not save results")

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

    logger.info(f"🚀 Argus Track v{__version__} - Enhanced GPS Tracking")
    logger.info(f"📹 Input video: {args.input_video}")
    logger.info(f"🤖 Model: {args.model}")
    logger.info(f"📺 Real-time display: {show_realtime}")

    try:
        # Create simplified configuration
        config = TrackerConfig.create_for_unified_tracker()

        # Apply command line overrides
        config.detection_conf = args.detection_conf
        config.gps_frame_interval = args.gps_interval

        if args.disable_static_car:
            config.enable_static_car_detection = False
            logger.info("🔧 Static car detection disabled")

        logger.info("📷 Configuration created")
        logger.info(f"   Detection confidence: {config.detection_conf}")
        logger.info(f"   GPS frame interval: {config.gps_frame_interval}")
        logger.info(f"   Track memory age: {config.max_track_memory_age}")

        # ENHANCED: Extract GPS data with fallback heading calculation
        logger.info("🗺️ Extracting GPS data with enhanced heading calculation...")
        gps_data = None

        try:
            # Import enhanced GPS extraction
            from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos

            gps_data, extraction_method = extract_gps_from_stereo_videos(
                args.input_video, args.input_video, method="auto"
            )

            if gps_data:
                logger.info(
                    f"✅ Extracted {len(gps_data)} GPS points using {extraction_method}"
                )
                logger.info("🧭 Enhanced heading calculation applied when GPS metadata missing")
                
                # Log heading source statistics
                first_few_points = gps_data[:5]
                logger.info("📍 Sample GPS points with headings:")
                for i, gps_point in enumerate(first_few_points):
                    logger.info(
                        f"   Point {i}: Lat={gps_point.latitude:.6f}, "
                        f"Lon={gps_point.longitude:.6f}, Heading={gps_point.heading:.1f}°"
                    )
            else:
                logger.warning("⚠️ No GPS data found in video metadata")

        except ImportError:
            logger.warning(
                "⚠️ Enhanced GPS extraction not available - trying basic extraction"
            )
            # Fallback to basic GPS extraction
            try:
                from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos

                gps_data, extraction_method = extract_gps_from_stereo_videos(
                    args.input_video, args.input_video, method="auto"
                )

                if gps_data:
                    logger.info(
                        f"✅ Extracted {len(gps_data)} GPS points using {extraction_method} (basic)"
                    )
                else:
                    logger.warning("⚠️ No GPS data found in video metadata")

            except ImportError:
                logger.warning(
                    "⚠️ GPS extraction not available - processing without GPS context"
                )
                gps_data = None
        except Exception as e:
            logger.error(f"❌ Enhanced GPS extraction failed: {e}")
            logger.info("🔄 Falling back to basic GPS extraction...")
            
            try:
                from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos

                gps_data, extraction_method = extract_gps_from_stereo_videos(
                    args.input_video, args.input_video, method="auto"
                )

                if gps_data:
                    logger.info(
                        f"✅ Extracted {len(gps_data)} GPS points using {extraction_method} (fallback)"
                    )
                else:
                    logger.warning("⚠️ No GPS data found in video metadata")
            except Exception as e2:
                logger.error(f"❌ All GPS extraction methods failed: {e2}")
                gps_data = None

        # Initialize unified tracker
        tracker = UnifiedLightPostTracker(
            config=config,
            model_path=args.model,
            show_realtime=show_realtime,
            display_size=(1280, 720),
        )

        # Show helpful tips
        if show_realtime:
            logger.info("🖥️  Real-time visualization controls:")
            logger.info("   - Press 'q' to quit")
            logger.info("   - Press 'p' to pause/resume")
            logger.info("   - Press 's' to save screenshot")

        # Process video
        start_time = time.time()

        results = tracker.process_video(
            video_path=args.input_video,
            gps_data=gps_data,
            output_path=args.output_video,
            save_results=not args.no_save,
        )

        processing_time = time.time() - start_time

        # Print results
        logger.info("🎉 PROCESSING COMPLETE!")
        logger.info(f"⏱️  Total processing time: {processing_time:.1f} seconds")
        logger.info(f"📊 Processing efficiency: {results['avg_fps']:.1f} FPS")

        # Track statistics
        track_stats = results["track_manager_stats"]
        logger.info("🔧 TRACKING STATISTICS:")
        logger.info(f"   Active tracks: {track_stats['active_tracks']}")
        logger.info(f"   Total tracks created: {track_stats['total_tracks_created']}")
        logger.info(
            f"   Resurrections prevented: {track_stats['resurrections_prevented']}"
        )
        logger.info(f"   Vehicle distance: {track_stats['distance_moved']:.1f}m")

        # Enhanced GPS heading statistics
        if gps_data:
            logger.info("🧭 GPS HEADING STATISTICS:")
            logger.info(f"   Total GPS points: {len(gps_data)}")
            logger.info(f"   GPS data available for tracking")
            
            # Show heading range
            headings = [gps.heading for gps in gps_data if gps.heading != 0.0]
            if headings:
                logger.info(f"   Heading range: {min(headings):.1f}° to {max(headings):.1f}°")
                
                # Check for heading calculation vs metadata
                if hasattr(gps_data[0], 'heading_source'):
                    logger.info("   Heading sources in GPS data available")

        # Frame processing statistics
        logger.info("📅 FRAME PROCESSING STATISTICS:")
        logger.info(f"   Frame interval: {config.gps_frame_interval}")
        logger.info(f"   Expected frame naming: 0, {config.gps_frame_interval}, {config.gps_frame_interval*2}, ...")
        logger.info(f"   Processed frames: {results['processed_frames']}")
        logger.info(f"   Skipped (GPS sync): {results['skipped_frames_gps']}")
        logger.info(f"   Skipped (Static car): {results['skipped_frames_static']}")

        # Calculate efficiency
        total_expected = results['total_frames'] // config.gps_frame_interval
        efficiency = (1 - results['processed_frames'] / total_expected) * 100 if total_expected > 0 else 0
        logger.info(f"   Processing efficiency: {efficiency:.1f}% reduction in frames")

        # Output files
        if not args.no_save and "json_output" in results:
            logger.info("📄 OUTPUT FILES:")
            logger.info(f"   JSON: {results['json_output']}")
            logger.info(f"   CSV: {results['csv_output']}")
            
            # Validate output frame naming
            logger.info("✅ FRAME NAMING VALIDATION:")
            try:
                import json
                with open(results['json_output'], 'r') as f:
                    output_data = json.load(f)
                
                frame_keys = list(output_data.get('frames', {}).keys())
                sample_frames = frame_keys[:5]
                logger.info(f"   Sample JSON frame keys: {sample_frames}")
                
                # Extract frame numbers and validate
                frame_numbers = []
                for key in frame_keys:
                    if key.startswith('frame_'):
                        try:
                            frame_num = int(key.split('_')[1])
                            frame_numbers.append(frame_num)
                        except (IndexError, ValueError):
                            pass
                
                if frame_numbers:
                    sorted_frames = sorted(frame_numbers)
                    intervals_valid = all(
                        frame % config.gps_frame_interval == 0 
                        for frame in sorted_frames[:10]  # Check first 10
                    )
                    logger.info(f"   Frame intervals valid: {intervals_valid}")
                    logger.info(f"   Sample frame numbers: {sorted_frames[:5]}")
                else:
                    logger.warning("   Could not extract frame numbers from JSON")
                    
            except Exception as e:
                logger.warning(f"   Could not validate output frame naming: {e}")

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