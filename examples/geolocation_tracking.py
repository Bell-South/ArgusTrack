# examples/geolocation_tracking.py

"""Example demonstrating light post tracking with geolocation"""

import argparse
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from argus_track import (
    TrackerConfig,
    LightPostTracker,
    MockDetector,
    YOLODetector
)
from argus_track.utils.io import load_gps_data, export_to_geojson
from argus_track.utils.gps_utils import GeoLocation


def main():
    """Run light post tracking with geolocation"""
    parser = argparse.ArgumentParser(description="Light Post Tracking with Geolocation")
    
    # Required arguments
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("gps_path", type=str, help="Path to GPS data CSV")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default=None, help="Path for output video")
    parser.add_argument("--geojson", type=str, default=None, help="Path for GeoJSON output")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--detector", type=str, choices=["yolo", "mock"], default="mock",
                       help="Detector type to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    if args.config:
        from argus_track.utils.io import load_config_from_file
        config_dict = load_config_from_file(args.config)
        config = TrackerConfig(**config_dict)
    else:
        config = TrackerConfig(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=50,
            static_threshold=2.0,
            min_static_frames=5
        )
    
    # Initialize detector
    if args.detector == "yolo":
        try:
            detector = YOLODetector(
                model_path="models/yolov4.weights",
                config_path="models/yolov4.cfg",
                target_classes=["light_post", "street_light", "pole"]
            )
        except Exception as e:
            logging.error(f"Failed to initialize YOLO detector: {e}")
            logging.info("Falling back to mock detector")
            detector = MockDetector(target_classes=["light_post"])
    else:
        detector = MockDetector(target_classes=["light_post"])
    
    # Load GPS data
    try:
        gps_data = load_gps_data(args.gps_path)
        logging.info(f"Loaded {len(gps_data)} GPS data points")
    except Exception as e:
        logging.error(f"Failed to load GPS data: {e}")
        return 1
    
    # Initialize tracker
    tracker = LightPostTracker(config, detector)
    
    # Process video
    logging.info(f"Processing video: {args.video_path}")
    try:
        tracks = tracker.process_video(
            video_path=args.video_path,
            gps_data=gps_data,
            output_path=args.output,
            save_results=True
        )
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return 1
    
    # Analyze static objects and locations
    static_objects = tracker.analyze_static_objects()
    static_count = sum(1 for is_static in static_objects.values() if is_static)
    logging.info(f"Identified {static_count} static objects")
    
    # Get location estimates
    locations = tracker.get_static_locations()
    logging.info(f"Estimated locations for {len(locations)} static objects")
    
    # Export locations to GeoJSON if requested
    if args.geojson:
        geojson_path = args.geojson
    else:
        geojson_path = Path(args.video_path).with_suffix('.geojson')
    
    tracker.export_locations_to_geojson(geojson_path)
    logging.info(f"Exported locations to GeoJSON: {geojson_path}")
    
    # Print location results
    print("\nEstimated Light Post Locations:")
    print("------------------------------")
    for track_id, location in locations.items():
        print(f"Track {track_id}: "
              f"({location.latitude:.6f}, {location.longitude:.6f}) "
              f"Reliability: {location.reliability:.2f}")
    
    # Display stats
    stats = tracker.get_track_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return 0


if __name__ == "__main__":
    exit(main())