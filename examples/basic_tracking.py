"""Basic tracking example"""

from argus_track import (
    TrackerConfig,
    EnhancedLightPostTracker,
    MockDetector
)


def main():
    """Run basic tracking on a video file"""
    
    # Create configuration
    config = TrackerConfig(
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=50,
        min_box_area=100.0
    )
    
    # Initialize mock detector for testing
    detector = MockDetector(target_classes=['light_post', 'street_light'])
    
    # Create tracker
    tracker = EnhancedLightPostTracker(config, detector)
    
    # Process video
    tracks = tracker.process_video(
        video_path='input/test_video.mp4',
        output_path='output/tracked_video.mp4',
        save_results=True
    )
    
    # Print statistics
    stats = tracker.get_track_statistics()
    print(f"Tracked {stats['total_tracks']} objects")
    print(f"Active tracks: {stats['active_tracks']}")
    print(f"Static objects: {stats['static_objects']}")
    
    # Analyze static objects
    static_analysis = tracker.analyze_static_objects()
    static_count = sum(1 for is_static in static_analysis.values() if is_static)
    print(f"Identified {static_count} static light posts")


if __name__ == "__main__":
    main()