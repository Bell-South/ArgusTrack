import argparse
import logging
import time
from pathlib import Path


from argus_track import __version__
from argus_track.config import TrackerConfig
from argus_track.trackers.unique_tracker import UnifiedLightPostTracker
from argus_track.utils import setup_logging


def main():
    """Main function for unified tracking"""
    parser = argparse.ArgumentParser(
        description=f"Argus Track: Unified Light Post Tracking System v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic tracking with GPS movement context
    python main.py input.mp4 --model model.pt --show-realtime
    
    # Batch processing without visualization  
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
        default=True,
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

    logger.info(f"🚀 Argus Track v{__version__} - Unified Tracking")
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

        # Extract GPS data from video (simplified)
        logger.info("🗺️ Extracting GPS data from video metadata...")
        gps_data = None

        try:
            # Try to import GPS extraction
            from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos

            gps_data, extraction_method = extract_gps_from_stereo_videos(
                args.input_video, args.input_video, method="auto"
            )

            if gps_data:
                logger.info(
                    f"✅ Extracted {len(gps_data)} GPS points using {extraction_method}"
                )
            else:
                logger.warning("⚠️ No GPS data found in video metadata")

        except ImportError:
            logger.warning(
                "⚠️ GPS extraction not available - processing without GPS context"
            )
            gps_data = None
        except Exception as e:
            logger.error(f"❌ GPS extraction failed: {e}")
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

        # Output files
        if not args.no_save and "json_output" in results:
            logger.info(f"📄 JSON output: {results['json_output']}")
            logger.info(f"📍 CSV output: {results['csv_output']}")

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
