"""Enhanced visualization utilities with real-time display"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core import Detection, Track

# Color palette for different track states
TRACK_COLORS = {
    "tentative": (255, 255, 0),  # Yellow
    "confirmed": (0, 255, 0),  # Green
    "lost": (0, 0, 255),  # Red
    "removed": (128, 128, 128),  # Gray
}

# Class-specific colors
CLASS_COLORS = {
    "Led-150": (255, 0, 0),  # Red
    "Led-240": (0, 0, 255),  # Blue
    "light_post": (0, 255, 0),  # Green
    "street_light": (255, 165, 0),  # Orange
    "pole": (128, 0, 128),  # Purple
}


class RealTimeVisualizer:
    """Real-time visualization during tracking"""

    def __init__(
        self,
        window_name: str = "Argus Track - Real-time Detection",
        display_size: Tuple[int, int] = (1280, 720),
        show_info_panel: bool = True,
    ):
        """
        Initialize real-time visualizer

        Args:
            window_name: Name of the display window
            display_size: Size of the display window (width, height)
            show_info_panel: Whether to show information panel
        """
        self.window_name = window_name
        self.display_size = display_size
        self.show_info_panel = show_info_panel

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.RealTimeVisualizer")

        # Statistics tracking
        self.frame_count = 0
        self.detection_history = []
        self.fps_history = []
        self.last_time = time.time()

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, display_size[0], display_size[1])

        # Create default blank frame for error cases
        self.blank_frame = np.zeros(
            (display_size[1], display_size[0], 3), dtype=np.uint8
        )
        cv2.putText(
            self.blank_frame,
            "No frame data available",
            (display_size[0] // 4, display_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        self.logger.info(f"🖥️  Real-time visualization window opened: {window_name}")
        self.logger.info("   Press 'q' to quit, 'p' to pause, 's' to save screenshot")

    def _add_info_panel(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        gps_data: Optional[Dict] = None,
        frame_info: Optional[Dict] = None,
    ) -> np.ndarray:
        """Enhanced information panel with motion prediction and visual feature info"""

        if frame is None or not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
            return self.blank_frame.copy()

        try:
            # Create larger panel for enhanced info
            panel_height = 200  # Increased height
            panel_width = 400  # Increased width

            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (frame.shape[1] - panel_width - 10, 10),
                (frame.shape[1] - 10, panel_height + 10),
                (0, 0, 0),
                -1,
            )

            # Blend with original frame
            result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Safety checks
            if detections is None:
                detections = []
            if tracks is None:
                tracks = []
            if frame_info is None:
                frame_info = {}

            # Enhanced info lines
            y_offset = 35
            text_color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            small_font_scale = 0.4

            # === BASIC INFO ===
            info_lines = [
                f"Frame: {frame_info.get('frame_idx', self.frame_count)}",
                f"Detections: {len(detections)}",
                f"Active Tracks: {len([t for t in tracks if getattr(t, 'state', None) in ['tentative', 'confirmed']])}",
            ]

            # === MOTION PREDICTION INFO ===
            if frame_info.get("motion_prediction_enabled", False):
                motion_detected = (
                    "YES" if frame_info.get("camera_motion_detected", False) else "NO"
                )
                pred_accuracy = frame_info.get("avg_prediction_accuracy", 0)
                info_lines.extend(
                    [
                        f"Motion Pred: {motion_detected} ({pred_accuracy:.2f})",
                    ]
                )

            # === VISUAL FEATURES INFO ===
            if frame_info.get("visual_features_enabled", False):
                tracks_with_features = frame_info.get("tracks_with_features", 0)
                appearance_stability = frame_info.get("avg_appearance_stability", 0)
                info_lines.extend(
                    [
                        f"Visual Features: {tracks_with_features} tracks",
                        f"Appearance Stability: {appearance_stability:.2f}",
                    ]
                )

            # === DETECTION TYPE BREAKDOWN ===
            motion_compensated = len(
                [d for d in detections if getattr(d, "motion_compensated", False)]
            )
            prediction_matches = len(
                [d for d in detections if getattr(d, "prediction_match", False)]
            )
            reappearances = len(
                [d for d in detections if getattr(d, "reappearance_match", False)]
            )

            if motion_compensated > 0 or prediction_matches > 0 or reappearances > 0:
                info_lines.append(
                    f"Enhanced: MC:{motion_compensated} P:{prediction_matches} R:{reappearances}"
                )

            # === GPS INFO ===
            if gps_data:
                gps_data.get("vehicle_speed_ms", 0)
                speed_kmh = gps_data.get("vehicle_speed_kmh", 0)
                info_lines.extend(
                    [
                        f"GPS: {gps_data.get('latitude', 0):.5f}",
                        f"     {gps_data.get('longitude', 0):.5f}",
                        f"Speed: {speed_kmh:.1f} km/h",
                        f"Heading: {gps_data.get('heading', 0):.1f}°",
                    ]
                )

            # === TRACK CONSOLIDATION INFO ===
            consolidations = frame_info.get("track_consolidations", 0)
            reappearances_total = frame_info.get("track_reappearances", 0)
            if consolidations > 0 or reappearances_total > 0:
                info_lines.append(f"Consolidations: {consolidations}")
                info_lines.append(f"Reappearances: {reappearances_total}")

            # === PERFORMANCE INFO ===
            processed_frames = frame_info.get("processed_frames", 0)
            total_frames = frame_info.get("total_frames", 1)
            efficiency = (
                (processed_frames / total_frames * 100) if total_frames > 0 else 0
            )
            info_lines.append(f"Efficiency: {efficiency:.1f}%")

            # Render text lines
            for i, line in enumerate(info_lines):
                y_pos = y_offset + i * 18

                # Use smaller font for detailed info
                current_font_scale = small_font_scale if i > 6 else font_scale

                # Color coding for different types of info
                if "Motion Pred:" in line:
                    color = (0, 255, 255)  # Cyan for motion
                elif "Visual Features:" in line:
                    color = (255, 0, 255)  # Magenta for visual
                elif "Enhanced:" in line:
                    color = (0, 255, 0)  # Green for enhanced detections
                elif "GPS:" in line or "Speed:" in line:
                    color = (255, 255, 0)  # Yellow for GPS
                elif "Consolidations:" in line or "Reappearances:" in line:
                    color = (255, 165, 0)  # Orange for consolidation
                else:
                    color = text_color  # White for basic info

                cv2.putText(
                    result,
                    line,
                    (frame.shape[1] - panel_width + 5, y_pos),
                    font,
                    current_font_scale,
                    color,
                    1,
                )

            # === DETECTION QUALITY INDICATORS ===
            # Show quality indicators at the bottom of panel
            if detections:
                best_detection = max(detections, key=lambda d: getattr(d, "score", 0))
                best_score = getattr(best_detection, "score", 0)

                # Show best detection score with color coding
                score_color = (
                    (0, 255, 0)
                    if best_score > 0.7
                    else (0, 255, 255) if best_score > 0.5 else (0, 0, 255)
                )
                cv2.putText(
                    result,
                    f"Best: {best_score:.3f}",
                    (frame.shape[1] - panel_width + 5, panel_height - 10),
                    font,
                    font_scale,
                    score_color,
                    2,
                )

            return result

        except Exception as e:
            self.logger.error(f"Error adding enhanced info panel: {e}")
            return frame

    def _draw_tracks(
        self, frame: np.ndarray, tracks: List[Track], scale_x: float, scale_y: float
    ) -> np.ndarray:
        """Enhanced track drawing with motion and visual feature indicators"""

        if frame is None or len(frame.shape) != 3:
            return self.blank_frame.copy()

        if tracks is None:
            tracks = []

        try:
            result = frame.copy()
            for track in tracks:
                # Determine track color based on enhanced attributes
                if getattr(track, "motion_compensated", False):
                    color = (0, 255, 255)  # Cyan for motion compensated
                elif getattr(track, "prediction_match", False):
                    color = (255, 0, 255)  # Magenta for prediction match
                elif getattr(track, "reappearance_match", False):
                    color = (0, 165, 255)  # Orange for reappearance
                else:
                    # Use standard colors based on state
                    color = TRACK_COLORS.get(
                        getattr(track, "state", "confirmed"), (255, 255, 255)
                    )

                # Get bounding box
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = bbox
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                # Draw bounding box with enhanced thickness for special tracks
                thickness = (
                    4
                    if getattr(track, "motion_compensated", False)
                    else 3 if track.state == "confirmed" else 2
                )
                cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

                # Enhanced track info
                track_info_parts = [f"ID:{track.track_id}"]

                if hasattr(track, "hits"):
                    track_info_parts.append(f"H:{track.hits}")

                # Add enhancement indicators
                if getattr(track, "motion_compensated", False):
                    track_info_parts.append("MC")
                if getattr(track, "prediction_match", False):
                    match_score = getattr(track, "match_score", 0)
                    track_info_parts.append(f"P:{match_score:.2f}")
                if getattr(track, "reappearance_match", False):
                    track_info_parts.append("R")

                if track.state == "confirmed":
                    track_info_parts.append("✓")

                track_info = " ".join(track_info_parts)

                # Create enhanced text background
                text_size = cv2.getTextSize(
                    track_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )[0]
                bg_color = color
                cv2.rectangle(
                    result,
                    (x1, y1 - text_size[1] - 8),
                    (x1 + text_size[0] + 4, y1),
                    bg_color,
                    -1,
                )

                # Draw text with contrasting color
                text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                cv2.putText(
                    result,
                    track_info,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    2,
                )

                # Draw enhanced trajectory for confirmed tracks
                if (
                    track.state == "confirmed"
                    and len(getattr(track, "detections", [])) > 1
                    and hasattr(track, "detections")
                ):
                    self._draw_trajectory_enhanced(
                        result, track, scale_x, scale_y, color
                    )

            return result

        except Exception as e:
            self.logger.error(f"Error drawing enhanced tracks: {e}")
            return frame

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        track: Track,
        scale_x: float,
        scale_y: float,
        color: Tuple[int, int, int],
    ):
        """Draw enhanced trajectory with motion prediction indicators"""
        try:
            recent_detections = track.detections[-min(10, len(track.detections)) :]

            if len(recent_detections) < 2:
                return

            points = []
            for detection in recent_detections:
                center = detection.center
                scaled_center = (int(center[0] * scale_x), int(center[1] * scale_y))
                points.append(scaled_center)

            # Draw trajectory lines with varying thickness
            for i in range(1, len(points)):
                # Thicker line for more recent trajectory
                thickness = max(1, 3 - i // 3)
                cv2.line(frame, points[i - 1], points[i], color, thickness)

            # Draw trajectory points with size indicating recency
            for i, point in enumerate(points):
                radius = 4 if i == len(points) - 1 else max(2, 3 - i // 3)
                cv2.circle(frame, point, radius, color, -1)

                # Add prediction indicator for the latest point
                if i == len(points) - 1 and getattr(track, "prediction_match", False):
                    # Draw prediction indicator
                    cv2.circle(frame, point, radius + 3, (255, 255, 255), 1)

        except Exception as e:
            self.logger.error(f"Error drawing enhanced trajectory: {e}")

    def visualize_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        gps_data: Optional[Dict] = None,
        frame_info: Optional[Dict] = None,
    ) -> bool:
        """
        Visualize a single frame with detections and tracks

        Args:
            frame: Input frame
            detections: Raw detections for this frame
            tracks: Active tracks
            gps_data: Optional GPS data
            frame_info: Optional frame information

        Returns:
            False if user wants to quit, True otherwise
        """
        self.frame_count += 1

        # Input validation - use blank frame if input is invalid
        if frame is None or not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
            self.logger.warning(
                f"Invalid frame received (type: {type(frame)}, frame_count: {self.frame_count})"
            )
            vis_frame = self.blank_frame.copy()
        else:
            # Create visualization with error handling
            try:
                vis_frame = self._create_visualization(
                    frame, detections, tracks, gps_data, frame_info
                )
                if vis_frame is None:
                    self.logger.warning("Visualization failed, using blank frame")
                    vis_frame = self.blank_frame.copy()
            except Exception as e:
                self.logger.error(f"Visualization error: {e}")
                vis_frame = self.blank_frame.copy()
                cv2.putText(
                    vis_frame,
                    f"Visualization error: {str(e)[:50]}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / max(0.001, current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time

        # Keep only recent FPS values
        if len(self.fps_history) > 30:
            self.fps_history = self.fps_history[-30:]

        # Add FPS overlay
        avg_fps = np.mean(self.fps_history)
        cv2.putText(
            vis_frame,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Display frame
        cv2.imshow(self.window_name, vis_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False  # Quit
        elif key == ord("p"):
            # Pause - wait for another key press
            cv2.putText(
                vis_frame,
                "PAUSED - Press any key to continue",
                (vis_frame.shape[1] // 4, vis_frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow(self.window_name, vis_frame)
            self.logger.info("⏸️  Paused - Press any key to continue...")
            cv2.waitKey(0)
        elif key == ord("s"):
            # Save screenshot
            screenshot_name = f"argus_track_screenshot_{self.frame_count:06d}.jpg"
            cv2.imwrite(screenshot_name, vis_frame)
            self.logger.info(f"📸 Screenshot saved: {screenshot_name}")

        return True  # Continue

    def _create_visualization(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        gps_data: Optional[Dict] = None,
        frame_info: Optional[Dict] = None,
    ) -> np.ndarray:
        """Create comprehensive visualization frame"""
        # Safety check for frame
        if frame is None or not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
            return self.blank_frame.copy()

        # Resize frame to display size if needed
        try:
            vis_frame = self._resize_frame(frame)
        except Exception as e:
            self.logger.error(f"Error resizing frame: {e}")
            return self.blank_frame.copy()

        # Calculate scale factors for coordinate adjustment
        scale_x = vis_frame.shape[1] / max(1, frame.shape[1])
        scale_y = vis_frame.shape[0] / max(1, frame.shape[0])

        # Draw raw detections first (lighter overlay)
        vis_frame = self._draw_detections(vis_frame, detections, scale_x, scale_y)

        # Draw tracks (more prominent)
        vis_frame = self._draw_tracks(vis_frame, tracks, scale_x, scale_y)

        # Add information panels
        if self.show_info_panel:
            vis_frame = self._add_info_panel(
                vis_frame, detections, tracks, gps_data, frame_info
            )

        return vis_frame

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to display size - DEFENSIVE"""
        # Handle None or invalid frame
        if frame is None:
            return self.blank_frame.copy()

        if len(frame.shape) != 3:
            return self.blank_frame.copy()

        if frame.shape[:2] == (self.display_size[1], self.display_size[0]):
            return frame.copy()

        try:
            return cv2.resize(frame, self.display_size)
        except Exception as e:
            self.logger.error(f"Error resizing frame: {e}")
            return self.blank_frame.copy()

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        """Draw raw detections with semi-transparent overlay"""
        if frame is None or len(frame.shape) != 3:
            return self.blank_frame.copy()

        # Safety check - if frame is valid but detections is None
        if detections is None:
            detections = []

        # Create overlay for semi-transparency
        try:
            overlay = frame.copy()
            for detection in detections:
                bbox = detection.bbox
                x1, y1, x2, y2 = bbox
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
                conf_text = f"{detection.score:.2f}"
                cv2.putText(
                    overlay,
                    conf_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

            # Blend overlay with original frame
            result = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            return result
        except Exception as e:
            self.logger.error(f"Error drawing detections: {e}")
            return frame  # Return original frame if drawing fails

    def close(self):
        """Close the visualization window"""
        try:
            cv2.destroyWindow(self.window_name)
            self.logger.info(f"🖥️  Closed visualization window")

            # Print final statistics
            if self.fps_history:
                avg_fps = np.mean(self.fps_history)
                self.logger.info(f"📊 Average FPS: {avg_fps:.1f}")
                self.logger.info(f"📊 Total frames processed: {self.frame_count}")
        except Exception as e:
            self.logger.error(f"Error closing visualization window: {e}")


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    show_trajectory: bool = True,
    show_id: bool = True,
    show_state: bool = True,
) -> np.ndarray:
    """
    Draw tracks on frame (existing function - now enhanced with error handling)

    Args:
        frame: Input frame
        tracks: List of tracks to draw
        show_trajectory: Whether to show track trajectories
        show_id: Whether to show track IDs
        show_state: Whether to show track states

    Returns:
        Frame with track visualizations
    """
    # Handle None inputs
    if frame is None:
        # Create blank frame
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            blank_frame,
            "No frame data available",
            (400, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        return blank_frame

    if tracks is None:
        tracks = []

    try:
        vis_frame = frame.copy()

        for track in tracks:
            # Get color based on state
            color = TRACK_COLORS.get(track.state, (255, 255, 255))

            # Draw bounding box
            x1, y1, x2, y2 = track.to_tlbr().astype(int)
            thickness = 3 if track.state == "confirmed" else 2
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)

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
                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    vis_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Draw trajectory for confirmed tracks
            if (
                show_trajectory
                and track.state == "confirmed"
                and len(track.detections) > 1
            ):
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
    except Exception as e:
        # In case of error, log and return original frame
        logging.error(f"Error in draw_tracks: {e}")
        return frame


def create_track_overlay(
    frame: np.ndarray, tracks: List[Track], alpha: float = 0.3
) -> np.ndarray:
    """
    Create semi-transparent overlay with track information

    Args:
        frame: Input frame
        tracks: List of tracks
        alpha: Transparency level (0-1)

    Returns:
        Frame with overlay
    """
    # Handle None inputs
    if frame is None:
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return blank_frame

    if tracks is None or len(tracks) == 0:
        return frame.copy()

    try:
        overlay = np.zeros_like(frame)

        for track in tracks:
            if track.state != "confirmed":
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
    except Exception as e:
        logging.error(f"Error in create_track_overlay: {e}")
        return frame.copy()  # Return original frame if error
