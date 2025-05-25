# examples/complete_stereo_gps_example.py (NEW FILE)

"""
Complete Stereo Tracking Example with Automatic GPS Extraction
=============================================================

This example demonstrates the complete enhanced pipeline:
1. Automatic GPS extraction from GoPro videos
2. Stereo camera calibration loading
3. YOLOv11 object detection
4. Stereo matching and 3D triangulation
5. Multi-object tracking with ByteTrack
6. GPS synchronization and geolocation
7. Results export in multiple formats

Usage:
    python complete_stereo_gps_example.py left_camera.mp4 right_camera.mp4 stereo_calibration.pkl your_model.pt
"""

import argparse
import logging
import sys
from pathlib import Path

from argus_track import (
    TrackerConfig,
    StereoCalibrationConfig,
    YOLOv11Detector
)
from argus_track.trackers.stereo_lightpost_tracker import EnhancedStereoLightPostTracker
from argus_track.utils.gps_extraction import extract_gps_from_stereo_videos, save_gps_to_csv
from argus_track.stereo import StereoCalibrationManager


def main():
    """Complete stereo tracking example with GPS extraction"""
    parser = argparse.ArgumentParser(description="Complete Stereo Tracking with GPS Extraction")
    
    # Required arguments
    parser.add_argument("left_video", help="Path to left camera video")
    parser.add_argument("right_video", help="Path to right camera video")
    parser.add_argument("calibration", help="Path to stereo calibration (.pkl)")
    parser.add_argument("model", help="Path to YOLOv11 model (.pt)")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--gps-method", default="auto", 
                       choices=["auto", "exiftool", "gopro_api"],
                       help="GPS extraction method")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("🚀 Starting Complete Stereo Tracking with GPS Extraction")
    logger.info(f"Left video: {args.left_video}")
    logger.info(f"Right video: {args.right_video}")
    logger.info(f"Calibration: {args.calibration}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Validate inputs
    logger.info("📋 Step 1: Validating inputs...")
    
    for path, name in [(args.left_video, "Left video"), 
                       (args.right_video, "Right video"),
                       (args.calibration, "Calibration"),
                       (args.model, "Model")]:
        if not Path(path).exists():
            logger.error(f"❌ {name} not found: {path}")
            return 1
        logger.info(f"✅ {name} found: {path}")
    
    # Step 2: Load and validate calibration
    logger.info("🔧 Step 2: Loading stereo calibration...")
    
    try:
        stereo_calibration = StereoCalibrationConfig.from_pickle(args.calibration)
        
        # Validate calibration
        calib_manager = StereoCalibrationManager(stereo_calibration)
        is_valid, errors = calib_manager.validate_calibration()
        
        if is_valid:
            logger.info("✅ Calibration validation passed")
            summary = calib_manager.get_calibration_summary()
            for key, value in summary.items():
                logger.info(f"   {key}: {value}")
        else:
            logger.warning(f"⚠️  Calibration validation issues: {errors}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load calibration: {e}")
        return 1
    
    # Step 3: Initialize detector
    logger.info("🎯 Step 3: Initializing YOLOv11 detector...")
    
    try:
        detector = YOLOv11Detector(
            model_path=args.model,
            target_classes=[
                'traffic light', 'stop sign', 'pole', 
                'light_post', 'street_light'  # Add your fine-tuned classes
            ],
            confidence_threshold=0.4,
            device='auto'
        )
        
        model_info = detector.get_model_info()
        logger.info(f"✅ Detector initialized: {model_info['device']}")
        logger.info(f"   Target classes: {model_info['target_classes']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        return 1
    
    # Step 4: Configure tracker
    logger.info("⚙️  Step 4: Configuring tracker...")
    
    config = TrackerConfig(
        track_thresh=0.4,           # Lower for fine-tuned models
        match_thresh=0.8,
        track_buffer=50,
        stereo_mode=True,
        stereo_match_threshold=0.6,
        gps_frame_interval=6,       # 60fps -> 10fps GPS sync
        static_threshold=2.0,
        min_static_frames=8
    )
    
    logger.info("✅ Tracker configuration ready")
    
    # Step 5: Initialize enhanced stereo tracker
    logger.info("🔄 Step 5: Initializing enhanced stereo tracker...")
    
    try:
        tracker = EnhancedStereoLightPostTracker(
            config=config,
            detector=detector,
            stereo_calibration=stereo_calibration
        )
        logger.info("✅ Enhanced stereo tracker initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize tracker: {e}")
        return 1
    
    # Step 6: Process videos with automatic GPS extraction
    logger.info("🎬 Step 6: Processing stereo videos with GPS extraction...")
    
    try:
        # Set output paths
        output_video = output_dir / "stereo_tracking_result.mp4"
        
        # Process with automatic GPS extraction
        tracks = tracker.process_stereo_video_with_auto_gps(
            left_video_path=args.left_video,
            right_video_path=args.right_video,
            output_path=str(output_video),
            save_results=True,
            gps_extraction_method=args.gps_method,
            save_extracted_gps=True
        )
        
        logger.info(f"✅ Processing complete! Found {len(tracks)} stereo tracks")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Step 7: Analyze results
    logger.info("📊 Step 7: Analyzing results...")
    
    # Get comprehensive statistics
    stats = tracker.get_enhanced_tracking_statistics()
    
    logger.info("=== TRACKING RESULTS ===")
    logger.info(f"📹 Total stereo tracks: {stats['total_stereo_tracks']}")
    logger.info(f"🏗️  Static tracks: {stats['static_tracks']}")
    logger.info(f"📍 Estimated locations: {stats['estimated_locations']}")
    logger.info(f"🛰️  GPS extraction method: {stats['gps_extraction_method']}")
    logger.info(f"📡 GPS points used: {stats['gps_points_used']}")
    logger.info(f"📏 Average depth: {stats['avg_depth']:.1f}m")
    logger.info(f"🎯 Average accuracy: {stats['accuracy_achieved']:.1f}m")
    logger.info(f"✅ Average reliability: {stats['avg_reliability']:.2f}")
    
    # Step 8: Display individual track results
    logger.info("🎯 Step 8: Individual track results...")
    
    if tracker.estimated_locations:
        logger.info("=== INDIVIDUAL TRACK LOCATIONS ===")
        
        accurate_tracks = 0
        for track_id, location in tracker.estimated_locations.items():
            accuracy_status = "🎯" if location.accuracy <= 2.0 else "✅" if location.accuracy <= 5.0 else "⚠️"
            
            logger.info(
                f"{accuracy_status} Track {track_id}: "
                f"({location.latitude:.6f}, {location.longitude:.6f}) "
                f"accuracy: {location.accuracy:.1f}m, "
                f"reliability: {location.reliability:.2f}"
            )
            
            if location.accuracy <= 2.0:
                accurate_tracks += 1
        
        # Summary statistics
        total_locations = len(tracker.estimated_locations)
        accuracy_rate = (accurate_tracks / total_locations) * 100 if total_locations > 0 else 0
        
        logger.info(f"📈 Accuracy Summary: {accurate_tracks}/{total_locations} tracks with sub-2m accuracy ({accuracy_rate:.1f}%)")
        
        if stats['accuracy_achieved'] <= 2.0:
            logger.info("🏆 TARGET ACHIEVED: Average accuracy ≤ 2 meters!")
        elif stats['accuracy_achieved'] <= 5.0:
            logger.info("✅ Good performance: Average accuracy ≤ 5 meters")
        else:
            logger.info("⚠️  Consider recalibration: Average accuracy > 5 meters")
    
    # Step 9: Output file summary
    logger.info("📁 Step 9: Output files...")
    
    video_path = Path(args.left_video)
    output_files = [
        ("Tracking video", output_video),
        ("Results JSON", video_path.with_suffix('.json')),
        ("Locations GeoJSON", video_path.with_suffix('.geojson')),
        ("GPS data CSV", video_path.with_suffix('.csv'))
    ]
    
    logger.info("=== OUTPUT FILES ===")
    for description, path in output_files:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"📄 {description}: {path} ({size_mb:.1f} MB)")
        else:
            logger.info(f"❌ {description}: {path} (not found)")
    
    # Step 10: Usage suggestions
    logger.info("💡 Step 10: Next steps...")
    
    logger.info("=== USAGE SUGGESTIONS ===")
    logger.info("1. 🗺️  View locations in GIS software:")
    logger.info(f"   - Open {video_path.with_suffix('.geojson')} in QGIS, ArcGIS, or geojson.io")
    
    logger.info("2. 📊 Analyze data:")
    logger.info(f"   - Import {video_path.with_suffix('.json')} for detailed analysis")
    logger.info(f"   - Use {video_path.with_suffix('.csv')} for GPS data processing")
    
    logger.info("3. 🎬 Review tracking:")
    logger.info(f"   - Watch {output_video} to verify tracking quality")
    
    if stats['accuracy_achieved'] > 2.0:
        logger.info("4. 🔧 Improve accuracy:")
        logger.info("   - Recalibrate stereo cameras with more images")
        logger.info("   - Ensure GPS accuracy < 3m during recording")
        logger.info("   - Check camera synchronization")
    
    logger.info("🎉 Complete stereo tracking with GPS extraction finished successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())