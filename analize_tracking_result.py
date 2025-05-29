#!/usr/bin/env python3
"""
Analyze Current Tracking Results
===============================
Let's see what's happening with the tracking and why objects aren't being geolocated
"""

import json
import numpy as np
from pathlib import Path

def analyze_tracking_results():
    """Analyze the current tracking results to understand the issue"""
    
    json_file = "../fellowship_of_the_frame/data/Videos/Camino_8/FI/GX018691.json"
    
    print("🔍 ANALYZING TRACKING RESULTS")
    print("=" * 50)
    
    if not Path(json_file).exists():
        print(f"❌ JSON file not found: {json_file}")
        return
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print("✅ Successfully loaded tracking data")
        
        # Analyze metadata
        metadata = data.get('metadata', {})
        tracks = data.get('tracks', {})
        
        print(f"\n📊 BASIC STATS:")
        print(f"   Total frames processed: {metadata.get('total_frames', 'unknown')}")
        print(f"   Total tracks: {len(tracks)}")
        print(f"   GPS data available: {metadata.get('has_gps_data', False)}")
        print(f"   GPS points used: {metadata.get('gps_points_used', 0)}")
        
        # Analyze each track
        print(f"\n🎯 TRACK ANALYSIS:")
        print("=" * 30)
        
        track_stats = []
        
        for track_id, track_history in tracks.items():
            if not track_history:
                continue
            
            # Calculate track statistics
            detection_count = len(track_history)
            
            # Get positions over time
            positions = []
            scores = []
            states = []
            
            for detection in track_history:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    positions.append([center_x, center_y])
                
                scores.append(detection.get('score', 0))
                states.append(detection.get('state', 'unknown'))
            
            # Calculate movement
            movement_std = 0
            if len(positions) > 1:
                positions_array = np.array(positions)
                movement_std = np.mean(np.std(positions_array, axis=0))
            
            # Track quality metrics
            max_hits = max([d.get('hits', 0) for d in track_history])
            avg_score = np.mean(scores) if scores else 0
            final_state = track_history[-1].get('state', 'unknown') if track_history else 'unknown'
            
            # Determine why it might not be static
            is_long_enough = detection_count >= 15  # min_static_frames from config
            is_stable = movement_std < 8.0  # static_threshold from config
            has_enough_hits = max_hits >= 5
            
            track_stat = {
                'track_id': int(track_id),
                'detections': detection_count,
                'movement_std': movement_std,
                'avg_score': avg_score,
                'max_hits': max_hits,
                'final_state': final_state,
                'is_long_enough': is_long_enough,
                'is_stable': is_stable,
                'has_enough_hits': has_enough_hits,
                'should_be_static': is_long_enough and is_stable and has_enough_hits
            }
            
            track_stats.append(track_stat)
        
        # Sort by detection count (best tracks first)
        track_stats.sort(key=lambda x: x['detections'], reverse=True)
        
        print(f"📋 TOP TRACKS:")
        for i, track in enumerate(track_stats[:5]):
            print(f"\n   Track {track['track_id']}:")
            print(f"     Detections: {track['detections']}")
            print(f"     Movement std: {track['movement_std']:.1f} px")
            print(f"     Avg score: {track['avg_score']:.3f}")
            print(f"     Max hits: {track['max_hits']}")
            print(f"     Final state: {track['final_state']}")
            print(f"     Quality checks:")
            print(f"       ✅ Long enough (≥15): {track['is_long_enough']}")
            print(f"       ✅ Stable (≤8px): {track['is_stable']}")
            print(f"       ✅ Well tracked (≥5 hits): {track['has_enough_hits']}")
            print(f"     🎯 Should be geolocated: {track['should_be_static']}")
        
        # Summary analysis
        long_tracks = [t for t in track_stats if t['is_long_enough']]
        stable_tracks = [t for t in track_stats if t['is_stable']]
        should_be_static = [t for t in track_stats if t['should_be_static']]
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total tracks: {len(track_stats)}")
        print(f"   Long enough tracks (≥15 detections): {len(long_tracks)}")
        print(f"   Stable tracks (≤8px movement): {len(stable_tracks)}")
        print(f"   Should be geolocated: {len(should_be_static)}")
        
        if len(should_be_static) == 0:
            print(f"\n❌ ISSUE IDENTIFIED:")
            
            if len(long_tracks) == 0:
                print(f"   📏 Tracks too short: Need ≥15 detections, max found: {max([t['detections'] for t in track_stats]) if track_stats else 0}")
                print(f"   💡 Solution: Run longer or reduce min_static_frames")
            
            if len(stable_tracks) == 0:
                print(f"   📍 Tracks too unstable: Need ≤8px movement")
                min_movement = min([t['movement_std'] for t in track_stats]) if track_stats else 0
                print(f"   📊 Most stable track: {min_movement:.1f}px movement")
                print(f"   💡 Solution: Increase static_threshold or check for camera shake")
            
            not_enough_hits = [t for t in track_stats if not t['has_enough_hits']]
            if len(not_enough_hits) > 0:
                print(f"   🎯 Some tracks don't have enough hits (≥5)")
                print(f"   💡 Solution: Reduce min_hits requirement")
        
        else:
            print(f"   ✅ Found {len(should_be_static)} tracks that should be geolocated!")
            print(f"   🤔 These tracks meet all criteria but weren't geolocated")
            print(f"   💡 Check GPS synchronization or geolocation algorithm")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if len(track_stats) > 0:
            avg_detections = np.mean([t['detections'] for t in track_stats])
            avg_movement = np.mean([t['movement_std'] for t in track_stats])
            
            if avg_detections < 15:
                print(f"   🔧 Reduce min_static_frames: Current=15, Suggested={int(avg_detections * 0.8)}")
            
            if avg_movement > 8:
                print(f"   🔧 Increase static_threshold: Current=8px, Suggested={int(avg_movement * 1.2)}px")
            
            # Check if interrupted too early
            if metadata.get('user_quit', False):
                print(f"   ⏱️  Processing was interrupted - try running longer")
                print(f"   📊 Only processed 143 frames, try processing more frames")
        
        return track_stats
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return []

if __name__ == "__main__":
    analyze_tracking_results()