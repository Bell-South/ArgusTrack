"""Video tracking with GPS integration example"""

from pathlib import Path
from argus_track import (
    TrackerConfig,
    EnhancedLightPostTracker,
    YOLODetector
)
from argus_track.utils import load_gps_data


def main():
    """Process video with GPS data for geolocation"""
    
    # Setup paths
    video_path = 'input/street_recording.mp4'
    gps_path = 'input/gps_data.csv'
    output_path = 'output/tracked_with_gps.mp4'
    
    # Check if files exist
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        return
    
    # Load configuration from YAML file
    try:
        config = TrackerConfig.from_yaml('config/tracker_config.yaml')
    except FileNotFoundError:
        print("Config file not found, using defaults")
        config = TrackerConfig(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=50,
            static_threshold=2.0,
            min_static_frames=5
        )
    
    # Initialize YOLO detector
    try:
        detector = YOLODetector(
            model_path='models/yolov4.weights',
            config_path='models/yolov4.cfg',
            target_classes=['light_post', 'street_light', 'pole']
        )
        print("Using YOLO detector")
    except Exception as e:
        print(f"Failed to load YOLO: {e}")
        print("Falling back to mock detector")
        from bytetrack_lightpost import MockDetector
        detector = MockDetector(target_classes=['light_post'])
    
    # Load GPS data
    gps_data = None
    if Path(gps_path).exists():
        gps_data = load_gps_data(gps_path)
        print(f"Loaded {len(gps_data)} GPS data points")
    else:
        print("No GPS data found, proceeding without geolocation")
    
    # Create tracker
    tracker = EnhancedLightPostTracker(config, detector)
    
    # Process video
    print(f"Processing video: {video_path}")
    tracks = tracker.process_video(
        video_path=video_path,
        gps_data=gps_data,
        output_path=output_path,
        save_results=True
    )
    
    # Analyze results
    print(f"\nTracking Results:")
    print(f"Total tracks: {len(tracks)}")
    
    # Get statistics
    stats = tracker.get_track_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Analyze static objects
    static_analysis = tracker.analyze_static_objects()
    static_tracks = [tid for tid, is_static in static_analysis.items() if is_static]
    
    print(f"\nStatic Light Posts:")
    for track_id in static_tracks:
        track = tracker.tracker.tracks[track_id]
        print(f"Track {track_id}: {len(track.detections)} detections")
        
        # If GPS data available, show estimated position
        if gps_data and track_id in tracker.gps_tracks:
            positions = tracker.estimate_3d_positions(track_id)
            if positions:
                last_pos = positions[-1]
                print(f"  Location: ({last_pos['x']:.6f}, {last_pos['y']:.6f})")
    
    print(f"\nOutput saved to: {output_path}")
    print(f"Results saved to: {Path(video_path).with_suffix('.json')}")


if __name__ == "__main__":
    main()