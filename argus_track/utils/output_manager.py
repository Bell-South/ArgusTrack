# argus_track/utils/output_manager.py (NEW FILE)

"""
Output Manager - Handles JSON and CSV export for simplified tracking
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
    """Data for a single processed frame"""

    frame_id: int
    timestamp: float
    detections: List[Detection]
    gps_data: Optional[GPSData] = None


class OutputManager:
    """
    Manages output file generation for simplified tracking

    Outputs:
    1. JSON file: Frame-by-frame detection data
    2. CSV file: GPS data synchronized with frame IDs
    """

    def __init__(self, video_path: str, class_names: List[str]):
        """
        Initialize output manager

        Args:
            video_path: Path to input video (for output naming)
            class_names: List of class names from YOLO model
        """
        self.video_path = Path(video_path)
        self.class_names = class_names
        self.logger = logging.getLogger(f"{__name__}.OutputManager")

        # Storage for output data
        self.frame_data: Dict[int, FrameData] = {}
        self.processing_stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "unique_track_ids": set(),
            "class_distribution": {},
            "processing_time": 0.0,
        }

        self.logger.info(f"Output Manager initialized for video: {video_path}")
        self.logger.info(f"Available classes: {class_names}")

    def add_frame_data(
        self,
        frame_id: int,
        timestamp: float,
        detections: List[Detection],
        gps_data: Optional[GPSData] = None,
    ):
        """
        Add data for a processed frame

        Args:
            frame_id: Frame identifier
            timestamp: Timestamp in original video
            detections: List of detections for this frame
            gps_data: GPS data for this frame (if available)
        """
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

        for detection in detections or []:
            self.processing_stats["unique_track_ids"].add(detection.track_id)

            # Update class distribution
            class_name = self._get_class_name(detection.class_id)
            if class_name not in self.processing_stats["class_distribution"]:
                self.processing_stats["class_distribution"][class_name] = 0
            self.processing_stats["class_distribution"][class_name] += 1

    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        Export frame data to JSON file

        Args:
            output_path: Custom output path (optional)

        Returns:
            Path to exported JSON file
        """
        if output_path is None:
            output_path = self.video_path.with_suffix(".json")

        # Prepare JSON data
        json_data = {
            "metadata": {
                "video_file": str(self.video_path),
                "total_frames": self.processing_stats["total_frames_processed"],
                "total_detections": self.processing_stats["total_detections"],
                "unique_tracks": len(self.processing_stats["unique_track_ids"]),
                "class_names": self.class_names,
                "class_distribution": self.processing_stats["class_distribution"],
                "processing_method": "simplified_tracking_with_consolidation",
            },
            "frames": {},
        }

        # Add frame data
        for frame_id, data in sorted(self.frame_data.items()):
            frame_key = f"frame_{frame_id}"
            json_data["frames"][frame_key] = {
                "timestamp": data.timestamp,
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

        # Write JSON file
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

        self.logger.info(f"📄 Exported frame data to JSON: {output_path}")
        self.logger.info(f"   Frames: {len(self.frame_data)}")
        self.logger.info(
            f"   Total detections: {self.processing_stats['total_detections']}"
        )
        self.logger.info(
            f"   Unique tracks: {len(self.processing_stats['unique_track_ids'])}"
        )

        return str(output_path)

    def export_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export GPS data to CSV file

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
        """Get processing summary statistics"""
        return {
            "frames_processed": self.processing_stats["total_frames_processed"],
            "total_detections": self.processing_stats["total_detections"],
            "unique_tracks": len(self.processing_stats["unique_track_ids"]),
            "avg_detections_per_frame": (
                self.processing_stats["total_detections"]
                / max(1, self.processing_stats["total_frames_processed"])
            ),
            "class_distribution": dict(self.processing_stats["class_distribution"]),
            "frames_with_gps": len(
                [data for data in self.frame_data.values() if data.gps_data is not None]
            ),
        }

    def print_summary(self):
        """Print processing summary to console"""
        summary = self.get_processing_summary()

        print("\n" + "=" * 50)
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"📹 Video: {self.video_path.name}")
        print(f"🎬 Frames processed: {summary['frames_processed']}")
        print(f"🎯 Total detections: {summary['total_detections']}")
        print(f"🏷️  Unique tracks: {summary['unique_tracks']}")
        print(f"📊 Avg detections/frame: {summary['avg_detections_per_frame']:.1f}")
        print(f"📍 Frames with GPS: {summary['frames_with_gps']}")

        if summary["class_distribution"]:
            print("\n🏷️  Class Distribution:")
            for class_name, count in summary["class_distribution"].items():
                print(f"   {class_name}: {count}")

        print("=" * 50)

    def create_sample_output(self) -> Dict[str, Any]:
        """Create sample output structure for documentation"""
        return {
            "sample_json_structure": {
                "metadata": {
                    "video_file": "example_video.mp4",
                    "total_frames": 150,
                    "total_detections": 45,
                    "unique_tracks": 8,
                    "class_names": ["LED", "traffic_light", "street_lamp"],
                    "class_distribution": {"LED": 30, "traffic_light": 15},
                },
                "frames": {
                    "frame_6": {
                        "timestamp": 0.1,
                        "detections": [
                            {
                                "track_id": 1,
                                "class": "LED",
                                "class_id": 0,
                                "bbox": [100.5, 200.3, 150.8, 280.1],
                                "confidence": 0.85,
                            }
                        ],
                    }
                },
            },
            "sample_csv_structure": [
                {
                    "frame_id": 6,
                    "timestamp": 0.1,
                    "latitude": -34.758432,
                    "longitude": -58.635219,
                    "altitude": 25.4,
                    "heading": 45.2,
                    "accuracy": 1.0,
                }
            ],
        }
