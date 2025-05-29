# examples/stereo_tracking_example.py (NEW FILE)

"""
Stereo Light Post Tracking Example

This example demonstrates the complete stereo tracking pipeline:
1. Load stereo camera calibration
2. Initialize YOLOv11 detector
3. Process stereo video with GPS data
4. Export results in multiple formats
"""

import argparse
import logging
from pathlib import Path

from argus_track import (
    TrackerConfig,
    StereoCalibrationConfig,
    EnhancedStereoLightPostTracker,
    YOLOv11Detector
)
from argus_track.utils.io import load_gps_data
from argus_track.stereo import StereoCalibrationManager


def main():
    """Run stereo light post tracking example"""
    parser = argparse.ArgumentParser(description="Stereo Light Post Tracking Example")
    
    # Required arguments
    parser.add_argument("left_video", type=str, help="Path to left camera video")
    parser.add_argument("right_video", type=str, help="Path to right camera video")
    parser.add_argument("calibration", type=str, help="Path to stereo calibration (.pkl)")
    
    # Optional arguments
    parser.add_argument("--gps", type=str, help="Path to GPS data CSV")
    parser.add_argument("--model", type=str, default="yolov11n.pt", 
                       help="Path to YOLOv11 model")
    parser.add_argument("--config", type=str, help="Path to configuration YAML")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Stereo Light Post Tracking System ===")
    
    # Load configuration
    if args.config:
        config = TrackerConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Use default stereo configuration
        config = TrackerConfig(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=50,
            stereo_mode=True,
            stereo_match_threshold=0.7,
            gps_frame_interval=6,
            static_threshold=2.0,
            min_static_frames=5
        )
        logger.info("Using default stereo configuration")
    
    # Load stereo calibration
    try:
        stereo_calibration = StereoCalibrationConfig.from_pickle(args.calibration)
        logger.info(f"Loaded stereo calibration from {args.calibration}")
        
        # Validate calibration
        calib_manager = StereoCalibrationManager(stereo_calibration)
        is_valid, errors = calib_manager.validate_calibration()
        
        if not is_valid:
            logger.warning(f"Calibration validation issues: {errors}")
        else:
            logger.info("Calibration validation passed")
            
        # Print calibration summary
        summary = calib_manager.get_calibration_summary()
        logger.info(f"Calibration summary: {summary}")
        
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}")
        # Create sample calibration for testing
        logger.info("Creating sample calibration for testing...")
        calib_manager = StereoCalibrationManager()
        stereo_calibration = calib_manager.create_sample_calibration(
            image_width=1920,
            image_height=1080,
            baseline=0.12  # 12cm baseline for GoPro setup
        )
    
    # Initialize YOLOv11 detector
    try:
        detector = YOLOv11Detector(
            model_path=args.model,
            target_classes=[
                "traffic light", "stop sign", "pole"
            ],
            confidence_threshold=0.5,
            device="auto"
        )
        logger.info(f"Initialized YOLOv11 detector with model: {args.model}")
        
        # Print model info
        model_info = detector.get_model_info()
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Failed to initialize YOLOv11 detector: {e}")
        logger.info("Falling back to mock detector for testing...")
        from argus_track.detectors import MockDetector
        detector = MockDetector(target_classes=["light_post", "pole"])
    
    # Load GPS data
    gps_data = None
    if args.gps and Path(args.gps).exists():
        try:
            gps_data = load_gps_data(args.gps)
            logger.info(f"Loaded {len(gps_data)} GPS data points")
        except Exception as e:
            logger.error(f"Failed to load GPS data: {e}")
            gps_data = None
    else:
        logger.warning("No GPS data provided - proceeding without geolocation")
    
    # Initialize stereo tracker
    try:
        tracker = EnhancedStereoLightPostTracker(
            config=config,
            detector=detector,
            stereo_calibration=stereo_calibration
        )
        logger.info("Initialized stereo light post tracker")
        
    except Exception as e:
        logger.error(f"Failed to initialize tracker: {e}")
        return 1
    
    # Set output paths
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "stereo_tracking_output.mp4"
    else:
        output_video = Path(args.left_video).parent / "stereo_tracking_output.mp4"
    
    # Process stereo video
    logger.info("Starting stereo video processing...")
    logger.info(f"Left video: {args.left_video}")
    logger.info(f"Right video: {args.right_video}")
    logger.info(f"Output video: {output_video}")
    
    try:
        stereo_tracks = tracker.process_stereo_video(
            left_video_path=args.left_video,
            right_video_path=args.right_video,
            gps_data=gps_data,
            output_path=str(output_video),
            save_results=True
        )
        
        # Print results
        logger.info("=== Tracking Results ===")
        stats = tracker.get_tracking_statistics()
        
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # Print individual track information
        logger.info("=== Track Details ===")
        for track_id, stereo_track in stereo_tracks.items():
            logger.info(f"Track {track_id}:")
            logger.info(f"  - Detections: {len(stereo_track.stereo_detections)}")
            logger.info(f"  - Static 3D: {stereo_track.is_static_3d}")
            logger.info(f"  - Average depth: {stereo_track.average_depth:.2f}m")
            logger.info(f"  - Depth consistency: {stereo_track.depth_consistency:.2f}")
            
            if stereo_track.estimated_location:
                loc = stereo_track.estimated_location
                logger.info(f"  - Location: ({loc.latitude:.6f}, {loc.longitude:.6f})")
                logger.info(f"  - Reliability: {loc.reliability:.2f}")
                logger.info(f"  - Accuracy: {loc.accuracy:.2f}m")
        
        # Print file outputs
        video_path = Path(args.left_video)
        logger.info("=== Output Files ===")
        logger.info(f"Visualization video: {output_video}")
        logger.info(f"Results JSON: {video_path.with_suffix('.json')}")
        logger.info(f"Locations GeoJSON: {video_path.with_suffix('.geojson')}")
        
        logger.info("=== Processing Complete ===")
        return 0
        
    except Exception as e:
        logger.error(f"Error during stereo processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample stereo calibration...")
    
    # Create sample calibration
    calib_manager = StereoCalibrationManager()
    calibration = calib_manager.create_sample_calibration()
    
    # Save calibration
    calib_path = Path("sample_data/stereo_calibration.pkl")
    calib_path.parent.mkdir(exist_ok=True)
    calib_manager.calibration = calibration
    calib_manager.save_calibration(str(calib_path))
    
    print(f"Saved sample calibration to: {calib_path}")
    
    # Create sample GPS data
    gps_path = Path("sample_data/gps_data.csv")
    with open(gps_path, 'w') as f:
        f.write("timestamp,latitude,longitude,altitude,heading,accuracy\n")
        for i in range(100):
            timestamp = 1000.0 + i * 0.1  # 10Hz GPS
            lat = 40.7128 + i * 0.00001   # Small movement
            lon = -74.0060 + i * 0.00001
            f.write(f"{timestamp},{lat},{lon},10.0,0.0,1.0\n")
    
    print(f"Created sample GPS data: {gps_path}")
    
    print("\nTo test the system:")
    print("1. Record stereo videos with GoPro or similar setup")
    print("2. Calibrate your stereo camera using OpenCV")  
    print("3. Extract GPS data from video metadata")
    print("4. Run: python stereo_tracking_example.py left.mp4 right.mp4 calibration.pkl --gps gps.csv")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_data()
    else:
        exit(main())