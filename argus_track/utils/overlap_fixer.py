# Create new file: argus_track/utils/overlap_fixer.py

"""
Simple Overlap Fixer for Ultralytics Tracking Issues
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np


class OverlapFixer:
    """
    Fixes Ultralytics tracking issues:
    1. Removes overlapping bounding boxes in same frame
    2. Consolidates track IDs based on GPS proximity
    """

    def __init__(self, overlap_threshold: float = 0.5, distance_threshold: float = 3.0):
        """
        Initialize overlap fixer

        Args:
            overlap_threshold: IoU threshold for detecting overlaps (0.5 = 50% overlap)
            distance_threshold: GPS distance threshold for same object (meters)
        """
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.logger = logging.getLogger(f"{__name__}.OverlapFixer")

        self.persistent_id_mapping: Dict[int, int] = {}  # ultralytics_id -> stable_id
        self.stable_id_counter = 1
        self.id_position_history: Dict[int, List[np.ndarray]] = (
            {}
        )  # stable_id -> recent_positions

        # Track ID consolidation
        self.id_mapping = {}  # original_id -> consolidated_id
        self.next_consolidated_id = 1
        self.track_positions = {}  # track_id -> recent GPS positions
        self.fixed_count = 0
        self.overlap_count = 0

    def fix_ultralytics_results(
        self, ultralytics_result, current_gps: Optional["GPSData"], frame_id: int
    ) -> List[Dict]:
        """
        Fix Ultralytics tracking results

        Args:
            ultralytics_result: Single result from model.track()[0]
            current_gps: Current GPS data
            frame_id: Current frame number

        Returns:
            List of fixed detection dictionaries
        """
        # Extract raw detections
        raw_detections = self._extract_detections(ultralytics_result, frame_id)

        if not raw_detections:
            return []

        # Step 1: Remove overlapping bounding boxes
        non_overlapping = self._remove_overlapping_boxes(raw_detections, frame_id)

        # Step 2: Consolidate track IDs
        consolidated = self._consolidate_track_ids(non_overlapping, current_gps)

        return consolidated

    def _extract_detections(self, result, frame_id: int) -> List[Dict]:
        """Extract detections from Ultralytics result"""
        detections = []

        if (
            not result.boxes
            or not hasattr(result.boxes, "id")
            or result.boxes.id is None
        ):
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, score, cls_id, track_id in zip(boxes, scores, classes, track_ids):
            detections.append(
                {
                    "bbox": box,
                    "score": float(score),
                    "class_id": int(cls_id),
                    "track_id": int(track_id),
                    "frame": frame_id,
                }
            )

        return detections

    def _remove_overlapping_boxes(
        self, detections: List[Dict], frame_id: int
    ) -> List[Dict]:
        """Remove overlapping bounding boxes in same frame"""
        if len(detections) <= 1:
            return detections

        # Calculate IoU for all pairs
        keep_indices = set(range(len(detections)))
        overlaps_removed = 0

        for i in range(len(detections)):
            if i not in keep_indices:
                continue

            for j in range(i + 1, len(detections)):
                if j not in keep_indices:
                    continue

                # Calculate IoU
                iou = self._calculate_iou(detections[i]["bbox"], detections[j]["bbox"])

                if iou > self.overlap_threshold:
                    # Keep the detection with higher confidence
                    if detections[i]["score"] >= detections[j]["score"]:
                        keep_indices.discard(j)
                        overlaps_removed += 1
                        self.logger.debug(
                            f"Frame {frame_id}: Removed overlapping track {detections[j]['track_id']} "
                            f"(IoU {iou:.2f} with track {detections[i]['track_id']})"
                        )
                    else:
                        keep_indices.discard(i)
                        overlaps_removed += 1
                        self.logger.debug(
                            f"Frame {frame_id}: Removed overlapping track {detections[i]['track_id']} "
                            f"(IoU {iou:.2f} with track {detections[j]['track_id']})"
                        )
                        break

        if overlaps_removed > 0:
            self.overlap_count += overlaps_removed
            self.logger.info(
                f"Frame {frame_id}: Removed {overlaps_removed} overlapping boxes"
            )

        return [detections[i] for i in sorted(keep_indices)]

    def _consolidate_track_ids(
        self, detections: List[Dict], current_gps: Optional["GPSData"]
    ) -> List[Dict]:
        """Enhanced consolidation with persistent memory"""

        for detection in detections:
            original_id = detection["track_id"]

            # Get stable consolidated ID
            consolidated_id = self._get_consolidated_id(detection, current_gps)

            # Update detection
            detection["original_track_id"] = original_id
            detection["track_id"] = consolidated_id

            if original_id != consolidated_id:
                self.fixed_count += 1

        return detections

    def _consolidate_track_ids_bkp(
        self, detections: List[Dict], current_gps: Optional["GPSData"]
    ) -> List[Dict]:
        """Consolidate track IDs to prevent multiple IDs for same object"""

        for detection in detections:
            original_id = detection["track_id"]

            # Get consolidated ID
            consolidated_id = self._get_consolidated_id(detection, current_gps)

            # Update detection
            detection["original_track_id"] = original_id
            detection["track_id"] = consolidated_id

            if original_id != consolidated_id:
                self.fixed_count += 1
                self.logger.info(
                    f"Frame {detection['frame']}: Consolidated track {original_id} → {consolidated_id}"
                )

        return detections

    def _get_consolidated_id(
        self, detection: Dict, current_gps: Optional["GPSData"]
    ) -> int:
        """Enhanced consolidation with persistent memory"""
        original_id = detection["track_id"]
        detection_center = np.array(
            [
                (detection["bbox"][0] + detection["bbox"][2]) / 2,
                (detection["bbox"][1] + detection["bbox"][3]) / 2,
            ]
        )

        # Check if we've seen this ultralytics ID before
        if original_id in self.persistent_id_mapping:
            stable_id = self.persistent_id_mapping[original_id]
            self.logger.info(f"Reusing known mapping: {original_id} -> {stable_id}")

            # Update position history
            if stable_id not in self.id_position_history:
                self.id_position_history[stable_id] = []
            self.id_position_history[stable_id].append(detection_center)

            # Keep only recent positions
            if len(self.id_position_history[stable_id]) > 10:
                self.id_position_history[stable_id] = self.id_position_history[
                    stable_id
                ][-10:]

            return stable_id

        # Check if this detection is close to any existing stable track
        best_match_id = None
        best_distance = float("inf")

        for stable_id, positions in self.id_position_history.items():
            if not positions:
                continue

            # Check distance to most recent positions
            recent_positions = positions[-3:]  # Last 3 positions
            for pos in recent_positions:
                distance = np.linalg.norm(detection_center - pos)

                # Use spatial proximity to identify same object
                if distance < 50.0 and distance < best_distance:  # 50px tolerance
                    best_match_id = stable_id
                    best_distance = distance

        if best_match_id is not None:
            # Map to existing stable ID
            self.persistent_id_mapping[original_id] = best_match_id
            self.id_position_history[best_match_id].append(detection_center)
            self.logger.info(
                f"Spatial match: {original_id} -> {best_match_id} (distance: {best_distance:.1f}px)"
            )
            return best_match_id

        # Create new stable ID
        new_stable_id = self.stable_id_counter
        self.stable_id_counter += 1

        self.persistent_id_mapping[original_id] = new_stable_id
        self.id_position_history[new_stable_id] = [detection_center]

        self.logger.info(f"New stable ID: {original_id} -> {new_stable_id}")
        return new_stable_id

    def _get_consolidated_id_bkp(
        self, detection: Dict, current_gps: Optional["GPSData"]
    ) -> int:
        """Get consolidated track ID for detection"""
        original_id = detection["track_id"]

        # If we've seen this original ID before, return its mapping
        if original_id in self.id_mapping:
            return self.id_mapping[original_id]

        # Check if this detection is close to any existing tracks (if we have GPS)
        if current_gps:
            detection_gps = self._estimate_detection_gps(detection, current_gps)

            if detection_gps:
                # Find existing tracks within distance threshold
                for existing_id, positions in self.track_positions.items():
                    if not positions:
                        continue

                    # Check distance to most recent position
                    recent_pos = positions[-1]
                    distance = self._gps_distance(
                        detection_gps[0],
                        detection_gps[1],
                        recent_pos["lat"],
                        recent_pos["lon"],
                    )

                    if distance <= self.distance_threshold:
                        # Merge with existing track
                        self.id_mapping[original_id] = existing_id

                        # Add position to existing track
                        self.track_positions[existing_id].append(
                            {
                                "lat": detection_gps[0],
                                "lon": detection_gps[1],
                                "frame": detection["frame"],
                            }
                        )

                        self.logger.info(
                            f"Merged track {original_id} into {existing_id} (distance: {distance:.1f}m)"
                        )
                        return existing_id

        # This is a genuinely new track
        new_consolidated_id = self.next_consolidated_id
        self.id_mapping[original_id] = new_consolidated_id
        self.next_consolidated_id += 1

        # Initialize position tracking
        if current_gps:
            detection_gps = self._estimate_detection_gps(detection, current_gps)
            if detection_gps:
                self.track_positions[new_consolidated_id] = [
                    {
                        "lat": detection_gps[0],
                        "lon": detection_gps[1],
                        "frame": detection["frame"],
                    }
                ]

        return new_consolidated_id

    def _estimate_detection_gps(
        self, detection: Dict, gps: "GPSData"
    ) -> Optional[Tuple[float, float]]:
        """Estimate GPS coordinates for detection (simplified version)"""
        try:
            bbox = detection["bbox"]
            bbox_height = bbox[3] - bbox[1]

            if bbox_height <= 0:
                return None

            # Simple depth estimation
            focal_length = 1400
            lightpost_height = 4.0
            estimated_depth = (lightpost_height * focal_length) / bbox_height

            # GPS calculation (simplified)
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            image_width = 1920  # Assume standard width

            pixels_from_center = bbox_center_x - (image_width / 2)
            degrees_per_pixel = 60.0 / image_width
            bearing_offset = pixels_from_center * degrees_per_pixel
            object_bearing = gps.heading + bearing_offset

            import math

            lat_offset = (
                estimated_depth * math.cos(math.radians(object_bearing))
            ) / 111000
            lon_offset = (estimated_depth * math.sin(math.radians(object_bearing))) / (
                111000 * math.cos(math.radians(gps.latitude))
            )

            object_lat = gps.latitude + lat_offset
            object_lon = gps.longitude + lon_offset

            return (object_lat, object_lon)

        except Exception:
            return None

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _gps_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between GPS points in meters"""
        R = 6378137.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def get_statistics(self) -> Dict:
        """Get overlap fixing statistics"""
        return {
            "overlaps_removed": self.overlap_count,
            "ids_consolidated": self.fixed_count,
            "unique_tracks": len(set(self.id_mapping.values())),
            "original_tracks": len(self.id_mapping),
        }
