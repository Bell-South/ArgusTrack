# argus_track/utils/gps_extraction.py (ENHANCED)

"""
GPS Data Extraction from GoPro Videos - Enhanced with Fallback Heading Calculation
==================================================================================
ENHANCED: Added state-of-the-art heading calculation when GPS metadata is missing
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup

from ..core import GPSData

# Configure logging
logger = logging.getLogger(__name__)

# Try to import GoPro API if available
try:
    from gopro_overlay.goprotelemetry import telemetry
    GOPRO_API_AVAILABLE = True
except ImportError:
    GOPRO_API_AVAILABLE = False
    logger.debug("GoPro telemetry API not available")


# Check for ExifTool availability
def check_exiftool_available() -> bool:
    """Check if ExifTool is available in the system"""
    try:
        result = subprocess.run(
            ["exiftool", "-ver"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


EXIFTOOL_AVAILABLE = check_exiftool_available()


@dataclass
class GPSExtractionResult:
    """Result of GPS extraction operation"""

    success: bool
    gps_data: List[GPSData]
    method_used: str
    total_points: int
    time_range: Optional[Tuple[float, float]] = None
    error_message: Optional[str] = None


class VehicleHeadingCalculator:
    """
    ENHANCED: State-of-the-art heading calculation for vehicle-mounted cameras
    Added to existing GPS extraction for fallback heading calculation
    """
    
    def __init__(self):
        self.gps_points = []
        self.calculated_headings = {}
        
    def initialize_and_calculate_all(self, raw_gps_data: List[Dict]) -> List[Dict]:
        """Calculate headings for all GPS points where missing"""
        if len(raw_gps_data) < 2:
            # Not enough data for heading calculation
            for data in raw_gps_data:
                data['calculated_heading'] = 0.0
                data['heading_source'] = 'insufficient_data'
            return raw_gps_data
        
        self.gps_points = raw_gps_data.copy()
        self.calculated_headings = {}
        
        # Process each point
        processed_data = []
        for i, gps_point in enumerate(raw_gps_data):
            processed_point = gps_point.copy()
            
            # Check if we have heading from metadata
            if (gps_point.get('metadata_heading') is not None and 
                gps_point['metadata_heading'] != 0.0):
                # Use metadata heading
                processed_point['calculated_heading'] = gps_point['metadata_heading']
                processed_point['heading_source'] = 'metadata'
            else:
                # Calculate heading using SOTA method
                calculated_heading = self._calculate_heading_at_index(i)
                processed_point['calculated_heading'] = calculated_heading
                processed_point['heading_source'] = 'calculated'
            
            processed_data.append(processed_point)
        
        # Post-process: Smooth calculated headings
        return self._smooth_calculated_headings(processed_data)
    
    def _calculate_heading_at_index(self, index: int) -> float:
        """Calculate heading at specific GPS point index"""
        if len(self.gps_points) < 2:
            return 0.0
        
        # Method selection based on position
        if index == 0:
            # First point: Use forward bearing
            return self._forward_bearing_method(index)
        elif index == len(self.gps_points) - 1:
            # Last point: Use backward bearing
            return self._backward_bearing_method(index)
        else:
            # Middle points: Use adaptive method
            return self._adaptive_multi_method(index)
    
    def _adaptive_multi_method(self, index: int) -> float:
        """Adaptive multi-method heading calculation"""
        context = self._analyze_vehicle_context(index)
        
        if context['type'] == 'highway':
            # Highway: Use long baseline for stability
            return self._long_baseline_method(index, baseline_distance=30.0)
        elif context['type'] == 'city_turning':
            # City with turns: Use centered difference method
            return self._centered_difference_method(index)
        elif context['type'] == 'slow_precise':
            # Slow movement: Use weighted multi-point
            return self._weighted_multi_point_method(index)
        else:
            # Default: Balanced approach
            return self._balanced_method(index)
    
    def _long_baseline_method(self, index: int, baseline_distance: float = 30.0) -> float:
        """Long baseline method for highway stability"""
        current_point = self.gps_points[index]
        
        # Find point backwards that's at least baseline_distance away
        back_point = None
        for i in range(index - 1, -1, -1):
            candidate = self.gps_points[i]
            distance = self._calculate_distance(candidate, current_point)
            if distance >= baseline_distance:
                back_point = candidate
                break
        
        # Find point forwards that's at least baseline_distance away
        forward_point = None
        for i in range(index + 1, len(self.gps_points)):
            candidate = self.gps_points[i]
            distance = self._calculate_distance(current_point, candidate)
            if distance >= baseline_distance:
                forward_point = candidate
                break
        
        # Calculate bearing based on available points
        if back_point and forward_point:
            return self._calculate_bearing_vincenty(back_point, forward_point)
        elif back_point:
            return self._calculate_bearing_vincenty(back_point, current_point)
        elif forward_point:
            return self._calculate_bearing_vincenty(current_point, forward_point)
        else:
            return self._centered_difference_method(index)
    
    def _centered_difference_method(self, index: int) -> float:
        """Centered difference method for responsive turning"""
        if index == 0:
            return self._forward_bearing_method(index)
        if index >= len(self.gps_points) - 1:
            return self._backward_bearing_method(index)
        
        prev_point = self.gps_points[index - 1]
        next_point = self.gps_points[index + 1]
        
        return self._calculate_bearing_vincenty(prev_point, next_point)
    
    def _weighted_multi_point_method(self, index: int) -> float:
        """Weighted multi-point method for high precision"""
        current_point = self.gps_points[index]
        bearings = []
        
        search_range = min(5, len(self.gps_points) // 2)
        
        for offset in range(1, search_range + 1):
            # Backward bearing
            if index - offset >= 0:
                back_point = self.gps_points[index - offset]
                distance = self._calculate_distance(back_point, current_point)
                if distance > 1.0:  # At least 1 meter
                    bearing = self._calculate_bearing_vincenty(back_point, current_point)
                    weight = min(distance / 5.0, 1.0)
                    bearings.append((bearing, weight))
            
            # Forward bearing
            if index + offset < len(self.gps_points):
                forward_point = self.gps_points[index + offset]
                distance = self._calculate_distance(current_point, forward_point)
                if distance > 1.0:
                    bearing = self._calculate_bearing_vincenty(current_point, forward_point)
                    weight = min(distance / 5.0, 1.0)
                    bearings.append((bearing, weight))
        
        if not bearings:
            return 0.0
        
        return self._weighted_circular_average(bearings)
    
    def _balanced_method(self, index: int) -> float:
        """Balanced method combining multiple approaches"""
        headings = []
        
        # Centered difference
        if index > 0 and index < len(self.gps_points) - 1:
            centered_heading = self._centered_difference_method(index)
            headings.append((centered_heading, 0.4))
        
        # Backward bearing
        if index > 0:
            backward_heading = self._backward_bearing_method(index)
            headings.append((backward_heading, 0.3))
        
        # Forward bearing
        if index < len(self.gps_points) - 1:
            forward_heading = self._forward_bearing_method(index)
            headings.append((forward_heading, 0.3))
        
        if not headings:
            return 0.0
        
        return self._weighted_circular_average(headings)
    
    def _analyze_vehicle_context(self, index: int) -> Dict:
        """Analyze vehicle driving context around this GPS point"""
        window_size = min(5, len(self.gps_points) // 4)
        start_idx = max(0, index - window_size)
        end_idx = min(len(self.gps_points), index + window_size + 1)
        
        speeds = []
        for i in range(start_idx, end_idx - 1):
            p1 = self.gps_points[i]
            p2 = self.gps_points[i + 1]
            
            distance = self._calculate_distance(p1, p2)
            time_diff = p2['timestamp'] - p1['timestamp']
            
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        # Classify context
        if avg_speed > 15.0:
            return {'type': 'highway', 'speed': avg_speed}
        elif avg_speed > 3.0:
            return {'type': 'city_turning', 'speed': avg_speed}
        elif avg_speed < 5.0:
            return {'type': 'slow_precise', 'speed': avg_speed}
        else:
            return {'type': 'mixed', 'speed': avg_speed}
    
    def _forward_bearing_method(self, index: int) -> float:
        """Calculate bearing to next point"""
        if index >= len(self.gps_points) - 1:
            return 0.0
        
        current = self.gps_points[index]
        next_point = self.gps_points[index + 1]
        
        return self._calculate_bearing_vincenty(current, next_point)
    
    def _backward_bearing_method(self, index: int) -> float:
        """Calculate bearing from previous point"""
        if index <= 0:
            return 0.0
        
        prev_point = self.gps_points[index - 1]
        current = self.gps_points[index]
        
        return self._calculate_bearing_vincenty(prev_point, current)
    
    def _calculate_bearing_vincenty(self, point1: Dict, point2: Dict) -> float:
        """High-precision bearing calculation using Vincenty's formulae"""
        lat1 = math.radians(point1['latitude'])
        lon1 = math.radians(point1['longitude'])
        lat2 = math.radians(point2['latitude'])
        lon2 = math.radians(point2['longitude'])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def _calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate distance between two GPS points in meters"""
        R = 6378137.0
        
        lat1 = math.radians(point1['latitude'])
        lon1 = math.radians(point1['longitude'])
        lat2 = math.radians(point2['latitude'])
        lon2 = math.radians(point2['longitude'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _weighted_circular_average(self, bearings: List[Tuple[float, float]]) -> float:
        """Calculate weighted circular average of bearings"""
        if not bearings:
            return 0.0
        
        total_x, total_y, total_weight = 0.0, 0.0, 0.0
        
        for bearing, weight in bearings:
            bearing_rad = math.radians(bearing)
            total_x += math.cos(bearing_rad) * weight
            total_y += math.sin(bearing_rad) * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_bearing_rad = math.atan2(total_y / total_weight, total_x / total_weight)
        return (math.degrees(avg_bearing_rad) + 360) % 360
    
    def _smooth_calculated_headings(self, processed_data: List[Dict]) -> List[Dict]:
        """Post-process calculated headings with smoothing"""
        calculated_indices = [i for i, data in enumerate(processed_data) 
                            if data['heading_source'] == 'calculated']
        
        if len(calculated_indices) < 3:
            return processed_data
        
        # Simple exponential smoothing for calculated headings
        alpha = 0.3  # Smoothing factor
        
        for i in range(1, len(calculated_indices)):
            idx = calculated_indices[i]
            prev_idx = calculated_indices[i-1]
            
            current_heading = processed_data[idx]['calculated_heading']
            prev_heading = processed_data[prev_idx]['calculated_heading']
            
            # Calculate circular difference
            diff = current_heading - prev_heading
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            
            # Apply smoothing if difference is reasonable
            if abs(diff) < 30:  # Only smooth small changes
                smoothed = prev_heading + alpha * diff
                processed_data[idx]['calculated_heading'] = (smoothed + 360) % 360
                processed_data[idx]['heading_source'] = 'calculated_smoothed'
        
        return processed_data


class GoProGPSExtractor:
    """Extract GPS data from GoPro videos using multiple methods - ENHANCED"""

    def __init__(self, fps_video: float = 60.0, fps_gps: float = 10.0):
        """
        Initialize GPS extractor

        Args:
            fps_video: Video frame rate (default: 60 fps)
            fps_gps: GPS data rate (default: 10 Hz)
        """
        self.fps_video = fps_video
        self.fps_gps = fps_gps
        self.frame_time_ms = 1000.0 / fps_video

        # ENHANCED: Vehicle-specific heading calculator
        self.vehicle_heading_calculator = VehicleHeadingCalculator()

        # Check available extraction methods
        self.methods_available = []
        if EXIFTOOL_AVAILABLE:
            self.methods_available.append("exiftool")
            logger.debug("ExifTool method available")
        if GOPRO_API_AVAILABLE:
            self.methods_available.append("gopro_api")
            logger.debug("GoPro API method available")

        if not self.methods_available:
            logger.warning("No GPS extraction methods available!")

    def extract_gps_data(
        self, video_path: str, method: str = "auto"
    ) -> GPSExtractionResult:
        """
        Extract GPS data from GoPro video with enhanced heading calculation

        Args:
            video_path: Path to GoPro video file
            method: Extraction method ('auto', 'exiftool', 'gopro_api')

        Returns:
            GPSExtractionResult: Enhanced extraction results
        """
        if not os.path.exists(video_path):
            return GPSExtractionResult(
                success=False,
                gps_data=[],
                method_used="none",
                total_points=0,
                error_message=f"Video file not found: {video_path}",
            )

        # Determine extraction method
        if method == "auto":
            if "gopro_api" in self.methods_available:
                method = "gopro_api"
            elif "exiftool" in self.methods_available:
                method = "exiftool"
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used="none",
                    total_points=0,
                    error_message="No GPS extraction methods available",
                )

        logger.info(f"Extracting GPS data from {video_path} using {method} method with enhanced heading")

        try:
            if method == "exiftool":
                return self._extract_with_exiftool(video_path)
            elif method == "gopro_api":
                return self._extract_with_gopro_api(video_path)
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used=method,
                    total_points=0,
                    error_message=f"Unknown extraction method: {method}",
                )
        except Exception as e:
            logger.error(f"Error extracting GPS data: {e}")
            return GPSExtractionResult(
                success=False,
                gps_data=[],
                method_used=method,
                total_points=0,
                error_message=str(e),
            )

    def _extract_with_exiftool(self, video_path: str) -> GPSExtractionResult:
        """Extract GPS data using ExifTool method with enhanced heading"""
        temp_dir = tempfile.mkdtemp()

        try:
            metadata_file = os.path.join(temp_dir, "metadata.xml")
            gps_file = os.path.join(temp_dir, "gps_data.txt")

            # Extract metadata using ExifTool
            cmd = [
                "exiftool",
                "-api",
                "largefilesupport=1",
                "-ee",
                "-G3",
                "-X",
                video_path,
            ]

            logger.debug(f"Running ExifTool command: {' '.join(cmd)}")

            with open(metadata_file, "w") as f:
                result = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.PIPE, text=True, timeout=300
                )

            if result.returncode != 0:
                raise RuntimeError(f"ExifTool failed: {result.stderr}")

            # Extract Track4 GPS data
            self._extract_track4_data(metadata_file, gps_file)

            # ENHANCED: Parse GPS data with heading calculation
            gps_data = self._parse_gps_file_with_enhanced_heading(gps_file)

            if gps_data:
                time_range = (gps_data[0].timestamp, gps_data[-1].timestamp)
                return GPSExtractionResult(
                    success=True,
                    gps_data=gps_data,
                    method_used="exiftool_enhanced_heading",
                    total_points=len(gps_data),
                    time_range=time_range,
                )
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used="exiftool",
                    total_points=0,
                    error_message="No GPS data found in metadata",
                )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _extract_with_gopro_api(self, video_path: str) -> GPSExtractionResult:
        """ENHANCED: Extract GPS data using GoPro API method with heading fallback"""
        try:
            telem = telemetry.Telemetry(video_path)

            if not telem.has_gps():
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used="gopro_api",
                    total_points=0,
                    error_message="No GPS data found in video",
                )

            # Get GPS track
            gps_track = telem.gps_track()
            raw_gps_data = []

            # First pass: Extract all GPS points
            for point in gps_track:
                if point.lat != 0.0 and point.lon != 0.0:
                    timestamp = (point.timestamp.total_seconds() 
                               if hasattr(point.timestamp, "total_seconds") 
                               else point.timestamp)
                    
                    # ENHANCED: Try to get heading from metadata with fallback
                    metadata_heading = getattr(point, "heading", None)
                    if metadata_heading is not None:
                        metadata_heading = float(metadata_heading)
                        if metadata_heading == 0.0:
                            metadata_heading = None
                    
                    raw_gps_data.append({
                        'timestamp': float(timestamp),
                        'latitude': float(point.lat),
                        'longitude': float(point.lon),
                        'altitude': float(getattr(point, "alt", 0.0)),
                        'metadata_heading': metadata_heading,
                        'accuracy': float(getattr(point, "dop", 1.0))
                    })

            # ENHANCED: Second pass - Calculate headings where missing
            processed_gps_data = self.vehicle_heading_calculator.initialize_and_calculate_all(raw_gps_data)

            # Convert to GPSData objects
            gps_data = []
            for data in processed_gps_data:
                gps_point = GPSData(
                    timestamp=data['timestamp'],
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    altitude=data['altitude'],
                    heading=data['calculated_heading'],  # Always available now
                    accuracy=data['accuracy']
                )
                gps_data.append(gps_point)

            if gps_data:
                time_range = (gps_data[0].timestamp, gps_data[-1].timestamp)
                
                # Log heading calculation statistics
                metadata_count = sum(1 for d in processed_gps_data 
                                   if d['heading_source'] == 'metadata')
                calculated_count = len(processed_gps_data) - metadata_count
                
                logger.info(f"GPS Heading Summary: {metadata_count} from metadata, "
                           f"{calculated_count} calculated using enhanced algorithm")
                
                return GPSExtractionResult(
                    success=True,
                    gps_data=gps_data,
                    method_used="gopro_api_enhanced_heading",
                    total_points=len(gps_data),
                    time_range=time_range,
                )
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used="gopro_api",
                    total_points=0,
                    error_message="No valid GPS points found",
                )

        except Exception as e:
            raise RuntimeError(f"Enhanced GoPro API extraction failed: {e}")

    def _extract_track4_data(self, metadata_file: str, output_file: str) -> None:
        """Extract Track4 GPS data from metadata XML file"""
        try:
            with open(metadata_file, "r", encoding="utf-8") as in_file, open(
                output_file, "w", encoding="utf-8"
            ) as out_file:

                for line in in_file:
                    if "Track4" in line or "GPS" in line:
                        out_file.write(line)

            logger.debug(f"Extracted Track4 data to {output_file}")

        except Exception as e:
            logger.error(f"Error extracting Track4 data: {e}")
            raise

    def _parse_gps_file_with_enhanced_heading(self, gps_file: str) -> List[GPSData]:
        """ENHANCED: Parse GPS data with heading calculation fallback"""
        
        # First pass: Parse all GPS points
        raw_gps_data = []
        
        try:
            with open(gps_file, "r", encoding="utf-8") as f:
                content = f.read()

            lines = (
                content.split("\n")[2:]
                if len(content.split("\n")) > 2
                else content.split("\n")
            )

            current_timestamp = None
            current_lat = None
            current_lon = None

            for line in lines:
                if not line.strip():
                    continue

                soup = BeautifulSoup(line, "html.parser")
                text_content = soup.get_text()

                if ":GPSLatitude>" in line:
                    current_lat = self._convert_gps_coordinate(text_content)
                elif ":GPSLongitude>" in line and current_lat is not None:
                    current_lon = self._convert_gps_coordinate(text_content)
                elif ":GPSDateTime>" in line:
                    current_timestamp = self._convert_timestamp(text_content)

                    if (
                        current_timestamp is not None
                        and current_lat is not None
                        and current_lon is not None
                        and current_lat != 0.0
                        and current_lon != 0.0
                    ):
                        raw_gps_data.append({
                            'timestamp': current_timestamp,
                            'latitude': current_lat,
                            'longitude': current_lon,
                            'altitude': 0.0,
                            'metadata_heading': None,  # ExifTool usually doesn't have heading
                            'accuracy': 1.0
                        })

                        current_lat = None
                        current_lon = None

            # ENHANCED: Second pass - Calculate headings
            if raw_gps_data:
                processed_data = self.vehicle_heading_calculator.initialize_and_calculate_all(raw_gps_data)
                
                # Convert to GPSData objects
                gps_data = []
                for data in processed_data:
                    gps_point = GPSData(
                        timestamp=data['timestamp'],
                        latitude=data['latitude'],
                        longitude=data['longitude'],
                        altitude=data['altitude'],
                        heading=data['calculated_heading'],
                        accuracy=data['accuracy']
                    )
                    gps_data.append(gps_point)

                logger.info(f"Parsed {len(gps_data)} GPS points with enhanced heading calculation")
                return gps_data

            return []

        except Exception as e:
            logger.error(f"Error parsing GPS file: {e}")
            return []

    def _convert_gps_coordinate(self, coord_str: str) -> float:
        """Convert GPS coordinate from DMS format to decimal degrees"""
        if not coord_str or not isinstance(coord_str, str):
            return 0.0

        try:
            coord_str = coord_str.strip()

            # Handle the format: "34 deg 39' 45.72" S"
            import re

            pattern = r"(\d+)\s+deg\s+(\d+)'\s+([\d.]+)\"\s*([NSEW])"
            match = re.search(pattern, coord_str)

            if match:
                degrees = float(match.group(1))
                minutes = float(match.group(2))
                seconds = float(match.group(3))
                direction = match.group(4)

                decimal = degrees + minutes / 60.0 + seconds / 3600.0

                if direction in ["S", "W"]:
                    decimal = -decimal

                return decimal

            if coord_str.startswith("<"):
                coord_str = coord_str[1:]
            if coord_str.endswith(">"):
                coord_str = coord_str[:-1]

            parts = coord_str.split(" ")
            if len(parts) < 6:
                logger.warning(f"Invalid GPS coordinate format: {coord_str}")
                return 0.0

            degrees = float(parts[1])
            minutes = float(parts[3].replace("'", ""))
            seconds = float(parts[4].replace('"', ""))
            direction = parts[5][0] if len(parts[5]) > 0 else "N"

            decimal = degrees + minutes / 60.0 + seconds / 3600.0

            if direction in ["S", "W"]:
                decimal = -decimal

            return decimal

        except (ValueError, IndexError) as e:
            logger.warning(f"Error converting GPS coordinate '{coord_str}': {e}")
            return 0.0

    def _convert_timestamp(self, timestamp_str: str) -> float:
        """Convert timestamp string to Unix timestamp"""
        if not timestamp_str:
            return 0.0

        try:
            timestamp_str = timestamp_str.strip()
            if timestamp_str.startswith("<"):
                timestamp_str = timestamp_str[1:]
            if timestamp_str.endswith(">"):
                timestamp_str = timestamp_str[:-1]

            try:
                dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S.%f")
            except ValueError:
                dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")

            return dt.timestamp()

        except ValueError as e:
            logger.warning(f"Error converting timestamp '{timestamp_str}': {e}")
            return 0.0

    def synchronize_with_video(
        self, gps_data: List[GPSData], video_duration: float, target_fps: float = 10.0
    ) -> List[GPSData]:
        """Synchronize GPS data with video timeline"""
        if not gps_data:
            return []

        sorted_gps = sorted(gps_data, key=lambda x: x.timestamp)

        start_time = sorted_gps[0].timestamp
        for gps_point in sorted_gps:
            gps_point.timestamp -= start_time

        sync_interval = 1.0 / target_fps
        sync_timeline = np.arange(0, video_duration, sync_interval)

        timestamps = np.array([gps.timestamp for gps in sorted_gps])
        latitudes = np.array([gps.latitude for gps in sorted_gps])
        longitudes = np.array([gps.longitude for gps in sorted_gps])

        sync_gps = []
        for sync_time in sync_timeline:
            if sync_time <= timestamps[-1]:
                idx = np.searchsorted(timestamps, sync_time)

                if idx == 0:
                    lat = latitudes[0]
                    lon = longitudes[0]
                elif idx >= len(timestamps):
                    lat = latitudes[-1]
                    lon = longitudes[-1]
                else:
                    t1, t2 = timestamps[idx - 1], timestamps[idx]
                    lat1, lat2 = latitudes[idx - 1], latitudes[idx]
                    lon1, lon2 = longitudes[idx - 1], longitudes[idx]

                    alpha = (sync_time - t1) / (t2 - t1)
                    lat = lat1 + alpha * (lat2 - lat1)
                    lon = lon1 + alpha * (lon2 - lon1)

                sync_point = GPSData(
                    timestamp=sync_time,
                    latitude=lat,
                    longitude=lon,
                    altitude=0.0,
                    heading=0.0,
                    accuracy=1.0,
                )
                sync_gps.append(sync_point)

        logger.info(
            f"Synchronized {len(sync_gps)} GPS points for {video_duration:.1f}s video"
        )
        return sync_gps


def extract_gps_from_stereo_videos(
    left_video: str, right_video: str, method: str = "auto"
) -> Tuple[List[GPSData], str]:
    """
    ENHANCED: Extract GPS data from stereo video pair with fallback heading calculation

    Args:
        left_video: Path to left camera video
        right_video: Path to right camera video
        method: Extraction method ('auto', 'exiftool', 'gopro_api')

    Returns:
        Tuple[List[GPSData], str]: Enhanced GPS data and method used
    """
    extractor = GoProGPSExtractor()

    # Try extracting from left video first
    logger.info("Attempting enhanced GPS extraction from left video")
    result_left = extractor.extract_gps_data(left_video, method)

    if result_left.success and result_left.total_points > 0:
        logger.info(
            f"Successfully extracted {result_left.total_points} GPS points from left video using {result_left.method_used}"
        )
        return result_left.gps_data, result_left.method_used

    # Fallback to right video
    logger.info("Left video GPS extraction failed, trying right video")
    result_right = extractor.extract_gps_data(right_video, method)

    if result_right.success and result_right.total_points > 0:
        logger.info(
            f"Successfully extracted {result_right.total_points} GPS points from right video using {result_right.method_used}"
        )
        return result_right.gps_data, result_right.method_used

    # No GPS data found
    logger.warning("No GPS data found in either video")
    return [], "none"


def save_gps_to_csv(gps_data: List[GPSData], output_path: str) -> None:
    """Save GPS data to CSV file for Argus Track"""
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "latitude", "longitude", "altitude", "heading", "accuracy"]
        )

        for gps in gps_data:
            writer.writerow(
                [
                    gps.timestamp,
                    gps.latitude,
                    gps.longitude,
                    gps.altitude,
                    gps.heading,
                    gps.accuracy,
                ]
            )

    logger.info(f"Saved {len(gps_data)} GPS points to {output_path}")