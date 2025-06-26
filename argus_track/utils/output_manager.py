# argus_track/utils/output_manager.py

"""
Output Manager - FIXED frame naming convention for proper 6-frame intervals
Enhanced with GPS coordinates in JSON export
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import Detection, GPSData


@dataclass
class FrameData:
    """Data for a single processed frame - ENHANCED with GPS"""

    frame_id: int
    timestamp: float
    detections: List[Detection]
    gps_data: Optional[GPSData] = None

class OutputManager:
    """
    Enhanced Output Manager with FIXED frame naming and GPS coordinates
    Frame naming now follows proper 6-frame intervals: frame_0, frame_6, frame_12...
    """

    def __init__(self, video_path: str, class_names: List[str], frame_interval: int = 6):
        """
        Initialize output manager

        Args:
            video_path: Path to input video (for output naming)
            class_names: List of class names from YOLO model
            frame_interval: Frame processing interval (default: 6)
        """
        self.video_path = Path(video_path)
        self.class_names = class_names
        self.frame_interval = frame_interval  # NEW: Store frame interval
        self.logger = logging.getLogger(f"{__name__}.OutputManager")

        # Storage for output data
        self.frame_data: Dict[int, FrameData] = {}
        self.processing_stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "frames_with_gps": 0,
            "unique_track_ids": set(),
            "class_distribution": {},
            "processing_time": 0.0,
            "frame_interval": frame_interval,  # NEW: Track frame interval
        }

        self.logger.info(f"Output Manager initialized for video: {video_path}")
        self.logger.info(f"Available classes: {class_names}")
        self.logger.info(f"Frame interval: {frame_interval} (every {frame_interval} frames)")

    def add_frame_data(
        self,
        frame_id: int,
        timestamp: float,
        detections: List[Detection],
        gps_data: Optional[GPSData] = None,
    ):
        """
        Add data for a processed frame - FIXED to validate frame naming convention

        Args:
            frame_id: Frame identifier (must be multiple of frame_interval)
            timestamp: Timestamp in original video
            detections: List of detections for this frame
            gps_data: GPS data for this frame (if available)
        """
        # VALIDATION: Ensure frame_id follows our naming convention
        if frame_id % self.frame_interval != 0:
            self.logger.warning(
                f"Frame {frame_id} does not follow {self.frame_interval}-frame interval convention. "
                f"Expected multiples of {self.frame_interval}."
            )
        
        self.frame_data[frame_id] = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections.copy() if detections else [],
            gps_data=gps_data,
        )

        # Update statistics
        self.processing_stats["total_frames_processed"] += 1
        self.processing_stats["total_detections"] += (
            len(detections) if detections else 0
        )
        
        if gps_data is not None:
            self.processing_stats["frames_with_gps"] += 1

        for detection in detections or []:
            self.processing_stats["unique_track_ids"].add(detection.track_id)

            # Update class distribution
            class_name = self._get_class_name(detection.class_id)
            if class_name not in self.processing_stats["class_distribution"]:
                self.processing_stats["class_distribution"][class_name] = 0
            self.processing_stats["class_distribution"][class_name] += 1

    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        Export frame data to JSON file - FIXED with proper frame naming

        Args:
            output_path: Custom output path (optional)

        Returns:
            Path to exported JSON file
        """
        if output_path is None:
            output_path = self.video_path.with_suffix(".json")

        # Validate frame naming convention before export
        validation_result = self._validate_frame_naming()
        if not validation_result["valid"]:
            self.logger.warning("Frame naming validation failed:")
            for issue in validation_result["issues"]:
                self.logger.warning(f"  - {issue}")

        # Prepare JSON data with enhanced frame naming info
        json_data = {
            "metadata": {
                "video_file": str(self.video_path),
                "total_frames": self.processing_stats["total_frames_processed"],
                "total_detections": self.processing_stats["total_detections"],
                "unique_tracks": len(self.processing_stats["unique_track_ids"]),
                "frames_with_gps": self.processing_stats["frames_with_gps"],
                "gps_coverage_percent": (
                    self.processing_stats["frames_with_gps"] / 
                    max(1, self.processing_stats["total_frames_processed"]) * 100
                ),
                "class_names": self.class_names,
                "class_distribution": self.processing_stats["class_distribution"],
                "processing_method": "unified_tracking_with_gps_coordinates",
                # NEW: Frame interval information
                "frame_interval": self.frame_interval,
                "frame_naming_convention": f"frame_N where N is multiple of {self.frame_interval}",
                "frame_naming_valid": validation_result["valid"],
            },
            "frames": {},
        }

        # Add frame data with FIXED naming convention
        for frame_id, data in sorted(self.frame_data.items()):
            # FIXED: Ensure frame key follows proper naming
            frame_key = f"frame_{frame_id}"
            
            # Validate that frame_id is correct
            if frame_id % self.frame_interval != 0:
                self.logger.error(f"Invalid frame ID {frame_id} - not multiple of {self.frame_interval}")
                continue
            
            # Build frame data structure
            frame_json = {
                "timestamp": data.timestamp,
                "frame_id": frame_id,  # NEW: Explicit frame_id for validation
                "detections": [
                    {
                        "track_id": det.track_id,
                        "class": self._get_class_name(det.class_id),
                        "class_id": det.class_id,
                        "bbox": [
                            float(det.bbox[0]),  # x1
                            float(det.bbox[1]),  # y1
                            float(det.bbox[2]),  # x2
                            float(det.bbox[3]),  # y2
                        ],
                        "confidence": float(det.score),
                    }
                    for det in data.detections
                ],
            }
            
            # Add GPS coordinates if available
            if data.gps_data is not None:
                frame_json["gps"] = {
                    "latitude": float(data.gps_data.latitude),
                    "longitude": float(data.gps_data.longitude),
                    "altitude": float(data.gps_data.altitude),
                    "heading": float(data.gps_data.heading),
                    "accuracy": float(data.gps_data.accuracy),
                }
            else:
                frame_json["gps"] = None
            
            json_data["frames"][frame_key] = frame_json

        # Write JSON file
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Enhanced logging with frame naming validation
        self.logger.info(f"📄 Exported frame data to JSON: {output_path}")
        self.logger.info(f"   Frames: {len(self.frame_data)}")
        self.logger.info(f"   Frame interval: {self.frame_interval}")
        self.logger.info(f"   Frame naming valid: {validation_result['valid']}")
        
        # Show sample frame names for verification
        sample_frames = sorted(list(self.frame_data.keys()))[:5]
        self.logger.info(f"   Sample frame IDs: {sample_frames}")
        
        self.logger.info(
            f"   Total detections: {self.processing_stats['total_detections']}"
        )
        self.logger.info(
            f"   Unique tracks: {len(self.processing_stats['unique_track_ids'])}"
        )
        
        # GPS coverage logging
        gps_coverage = (
            self.processing_stats["frames_with_gps"] / 
            max(1, self.processing_stats["total_frames_processed"]) * 100
        )
        self.logger.info(
            f"   GPS coverage: {gps_coverage:.1f}% "
            f"({self.processing_stats['frames_with_gps']}/{self.processing_stats['total_frames_processed']} frames)"
        )

        return str(output_path)

    def _validate_frame_naming(self) -> Dict[str, Any]:
        """
        Validate that all frames follow the proper naming convention
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        frame_ids = list(self.frame_data.keys())
        
        if not frame_ids:
            return {"valid": True, "issues": []}
        
        # Check if all frame IDs are multiples of frame_interval
        invalid_frames = [fid for fid in frame_ids if fid % self.frame_interval != 0]
        if invalid_frames:
            issues.append(
                f"Found {len(invalid_frames)} frames not following {self.frame_interval}-frame interval: "
                f"{invalid_frames[:10]}{'...' if len(invalid_frames) > 10 else ''}"
            )
        
        # Check for expected sequence
        sorted_frames = sorted(frame_ids)
        expected_frames = list(range(0, max(sorted_frames) + 1, self.frame_interval))
        missing_frames = set(expected_frames) - set(frame_ids)
        
        if missing_frames:
            issues.append(
                f"Missing expected frames: {sorted(list(missing_frames))[:10]}"
                f"{'...' if len(missing_frames) > 10 else ''}"
            )
        
        # Check for consistent intervals
        if len(sorted_frames) > 1:
            intervals = [sorted_frames[i] - sorted_frames[i-1] for i in range(1, len(sorted_frames))]
            non_standard_intervals = [interval for interval in intervals if interval % self.frame_interval != 0]
            
            if non_standard_intervals:
                issues.append(
                    f"Found non-standard intervals: {set(non_standard_intervals)}"
                )
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_frames": len(frame_ids),
            "valid_frames": len([fid for fid in frame_ids if fid % self.frame_interval == 0]),
            "frame_interval": self.frame_interval
        }

    def export_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export GPS data to CSV file - UNCHANGED (already working correctly)

        Args:
            output_path: Custom output path (optional)

        Returns:
            Path to exported CSV file
        """
        if output_path is None:
            output_path = self.video_path.with_suffix(".csv")

        # Collect GPS data with frame IDs
        gps_rows = []
        for frame_id, data in sorted(self.frame_data.items()):
            if data.gps_data is not None:
                gps_rows.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": data.timestamp,
                        "latitude": data.gps_data.latitude,
                        "longitude": data.gps_data.longitude,
                        "altitude": data.gps_data.altitude,
                        "heading": data.gps_data.heading,
                        "accuracy": data.gps_data.accuracy,
                    }
                )

        # Write CSV file
        if gps_rows:
            with open(output_path, "w", newline="") as f:
                fieldnames = [
                    "frame_id",
                    "timestamp",
                    "latitude",
                    "longitude",
                    "altitude",
                    "heading",
                    "accuracy",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(gps_rows)

            self.logger.info(f"📍 Exported GPS data to CSV: {output_path}")
            self.logger.info(f"   GPS entries: {len(gps_rows)}")
            
            # Verify frame naming in CSV
            sample_frame_ids = [row["frame_id"] for row in gps_rows[:5]]
            self.logger.info(f"   Sample frame IDs in CSV: {sample_frame_ids}")
        else:
            # Create empty CSV with headers
            with open(output_path, "w", newline="") as f:
                fieldnames = [
                    "frame_id",
                    "timestamp",
                    "latitude",
                    "longitude",
                    "altitude",
                    "heading",
                    "accuracy",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            self.logger.warning(
                f"⚠️  No GPS data available - created empty CSV: {output_path}"
            )

        return str(output_path)

    def export_both(
        self, json_path: Optional[str] = None, csv_path: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Export both JSON and CSV files

        Args:
            json_path: Custom JSON output path (optional)
            csv_path: Custom CSV output path (optional)

        Returns:
            Tuple of (json_path, csv_path)
        """
        json_output = self.export_json(json_path)
        csv_output = self.export_csv(csv_path)

        return json_output, csv_output

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        else:
            return f"unknown_class_{class_id}"

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics - ENHANCED with frame validation"""
        validation = self._validate_frame_naming()
        
        return {
            "frames_processed": self.processing_stats["total_frames_processed"],
            "total_detections": self.processing_stats["total_detections"],
            "unique_tracks": len(self.processing_stats["unique_track_ids"]),
            "avg_detections_per_frame": (
                self.processing_stats["total_detections"]
                / max(1, self.processing_stats["total_frames_processed"])
            ),
            "class_distribution": dict(self.processing_stats["class_distribution"]),
            "frames_with_gps": self.processing_stats["frames_with_gps"],
            "gps_coverage_percent": (
                self.processing_stats["frames_with_gps"] / 
                max(1, self.processing_stats["total_frames_processed"]) * 100
            ),
            # NEW: Frame naming validation
            "frame_interval": self.frame_interval,
            "frame_naming_valid": validation["valid"],
            "frame_naming_issues": validation["issues"]
        }

    def print_summary(self):
        """Print processing summary to console - ENHANCED with frame naming info"""
        summary = self.get_processing_summary()

        print("\n" + "=" * 50)
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"📹 Video: {self.video_path.name}")
        print(f"🎬 Frames processed: {summary['frames_processed']}")
        print(f"📅 Frame interval: {summary['frame_interval']} (every {summary['frame_interval']} frames)")
        print(f"✅ Frame naming valid: {summary['frame_naming_valid']}")
        
        if not summary['frame_naming_valid']:
            print("⚠️  Frame naming issues:")
            for issue in summary['frame_naming_issues']:
                print(f"   - {issue}")
        
        print(f"🎯 Total detections: {summary['total_detections']}")
        print(f"🏷️  Unique tracks: {summary['unique_tracks']}")
        print(f"📊 Avg detections/frame: {summary['avg_detections_per_frame']:.1f}")
        print(f"📍 Frames with GPS: {summary['frames_with_gps']} ({summary['gps_coverage_percent']:.1f}%)")

        if summary["class_distribution"]:
            print("\n🏷️  Class Distribution:")
            for class_name, count in summary["class_distribution"].items():
                print(f"   {class_name}: {count}")

        print("=" * 50)