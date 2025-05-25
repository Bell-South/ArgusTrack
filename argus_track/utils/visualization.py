"""Visualization utilities"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from ..core import Track, Detection


# Color palette for different track states
TRACK_COLORS = {
    'tentative': (255, 255, 0),    # Yellow
    'confirmed': (0, 255, 0),      # Green  
    'lost': (0, 0, 255),          # Red
    'removed': (128, 128, 128)     # Gray
}


def draw_tracks(frame: np.ndarray, tracks: List[Track], 
                show_trajectory: bool = True,
                show_id: bool = True,
                show_state: bool = True) -> np.ndarray:
    """
    Draw tracks on frame
    
    Args:
        frame: Input frame
        tracks: List of tracks to draw
        show_trajectory: Whether to show track trajectories
        show_id: Whether to show track IDs
        show_state: Whether to show track states
        
    Returns:
        Frame with track visualizations
    """
    vis_frame = frame.copy()
    
    for track in tracks:
        # Get color based on state
        color = TRACK_COLORS.get(track.state, (255, 255, 255))
        
        # Draw bounding box
        x1, y1, x2, y2 = track.to_tlbr().astype(int)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track information
        if show_id or show_state:
            label_parts = []
            if show_id:
                label_parts.append(f"ID: {track.track_id}")
            if show_state:
                label_parts.append(f"[{track.state}]")
            
            label = " ".join(label_parts)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(vis_frame, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Draw text
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw trajectory for confirmed tracks
        if show_trajectory and track.state == 'confirmed' and len(track.detections) > 1:
            points = []
            for det in track.detections[-10:]:  # Last 10 detections
                center = det.center
                points.append(center.astype(int))
            
            points = np.array(points)
            cv2.polylines(vis_frame, [points], False, color, 2)
            
            # Draw points
            for point in points:
                cv2.circle(vis_frame, tuple(point), 3, color, -1)
    
    return vis_frame


def create_track_overlay(frame: np.ndarray, tracks: List[Track],
                        alpha: float = 0.3) -> np.ndarray:
    """
    Create semi-transparent overlay with track information
    
    Args:
        frame: Input frame
        tracks: List of tracks
        alpha: Transparency level (0-1)
        
    Returns:
        Frame with overlay
    """
    overlay = np.zeros_like(frame)
    
    for track in tracks:
        if track.state != 'confirmed':
            continue
            
        # Create mask for track region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = track.to_tlbr().astype(int)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Apply color overlay
        color = TRACK_COLORS[track.state]
        overlay[mask > 0] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    return result


def plot_track_statistics(tracks: Dict[int, Track], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot tracking statistics
    
    Args:
        tracks: Dictionary of all tracks
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Track lengths
    track_lengths = [track.age for track in tracks.values()]
    axes[0, 0].hist(track_lengths, bins=20, edgecolor='black')
    axes[0, 0].set_title('Track Length Distribution')
    axes[0, 0].set_xlabel('Track Length (frames)')
    axes[0, 0].set_ylabel('Count')
    
    # Track states
    state_counts = {}
    for track in tracks.values():
        state = track.state
        state_counts[state] = state_counts.get(state, 0) + 1
    
    axes[0, 1].bar(state_counts.keys(), state_counts.values())
    axes[0, 1].set_title('Track State Distribution')
    axes[0, 1].set_xlabel('State')
    axes[0, 1].set_ylabel('Count')
    
    # Hits distribution
    hits_counts = [track.hits for track in tracks.values()]
    axes[1, 0].hist(hits_counts, bins=20, edgecolor='black')
    axes[1, 0].set_title('Track Hits Distribution')
    axes[1, 0].set_xlabel('Number of Hits')
    axes[1, 0].set_ylabel('Count')
    
    # Time since update for lost tracks
    lost_times = [track.time_since_update for track in tracks.values() 
                  if track.state == 'lost']
    if lost_times:
        axes[1, 1].hist(lost_times, bins=15, edgecolor='black')
        axes[1, 1].set_title('Time Since Update (Lost Tracks)')
        axes[1, 1].set_xlabel('Frames Since Update')
        axes[1, 1].set_ylabel('Count')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Lost Tracks', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def draw_detection(frame: np.ndarray, detection: Detection, 
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
    """
    Draw single detection on frame
    
    Args:
        frame: Input frame
        detection: Detection to draw
        color: Color for drawing
        thickness: Line thickness
        
    Returns:
        Frame with detection drawn
    """
    result = frame.copy()
    x1, y1, x2, y2 = detection.tlbr.astype(int)
    
    # Draw bounding box
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    
    # Draw score
    label = f"{detection.score:.2f}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    
    cv2.rectangle(result,
                 (x1, y1 - label_size[1] - 8),
                 (x1 + label_size[0], y1),
                 color, -1)
    
    cv2.putText(result, label, (x1, y1 - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result


def create_tracking_summary(tracks: Dict[int, Track],
                           frame_width: int = 1920,
                           frame_height: int = 1080) -> np.ndarray:
    """
    Create summary visualization of all tracks
    
    Args:
        tracks: Dictionary of all tracks
        frame_width: Width of output frame
        frame_height: Height of output frame
        
    Returns:
        Summary visualization frame
    """
    # Create blank canvas
    canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    
    # Draw track trajectories
    for track in tracks.values():
        if len(track.detections) < 2:
            continue
            
        # Get trajectory points
        points = np.array([det.center for det in track.detections])
        
        # Scale to fit canvas
        points[:, 0] = points[:, 0] / points[:, 0].max() * (frame_width - 100) + 50
        points[:, 1] = points[:, 1] / points[:, 1].max() * (frame_height - 100) + 50
        
        # Choose color based on track state
        color = TRACK_COLORS.get(track.state, (0, 0, 0))
        
        # Draw trajectory
        for i in range(1, len(points)):
            cv2.line(canvas, 
                    tuple(points[i-1].astype(int)),
                    tuple(points[i].astype(int)),
                    color, 2)
        
        # Draw track ID at end
        cv2.putText(canvas, f"ID: {track.track_id}", 
                   tuple(points[-1].astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add legend
    y_offset = 30
    for state, color in TRACK_COLORS.items():
        cv2.rectangle(canvas, (20, y_offset), (40, y_offset + 20), color, -1)
        cv2.putText(canvas, state, (50, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 30
    
    return canvas