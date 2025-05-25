# argus_track/utils/gps_extraction.py (NEW FILE)

"""
GPS Data Extraction from GoPro Videos
=====================================
Integrated GPS extraction functionality for Argus Track stereo processing.
Supports both ExifTool and GoPro API methods for extracting GPS metadata.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from bs4 import BeautifulSoup
import tempfile
import shutil
from dataclasses import dataclass

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
        result = subprocess.run(['exiftool', '-ver'], 
                               capture_output=True, text=True, timeout=10)
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


class GoProGPSExtractor:
    """Extract GPS data from GoPro videos using multiple methods"""
    
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
        
        # Check available extraction methods
        self.methods_available = []
        if EXIFTOOL_AVAILABLE:
            self.methods_available.append('exiftool')
            logger.debug("ExifTool method available")
        if GOPRO_API_AVAILABLE:
            self.methods_available.append('gopro_api')
            logger.debug("GoPro API method available")
        
        if not self.methods_available:
            logger.warning("No GPS extraction methods available!")
    
    def extract_gps_data(self, video_path: str, 
                        method: str = 'auto') -> GPSExtractionResult:
        """
        Extract GPS data from GoPro video
        
        Args:
            video_path: Path to GoPro video file
            method: Extraction method ('auto', 'exiftool', 'gopro_api')
            
        Returns:
            GPSExtractionResult: Extraction results
        """
        if not os.path.exists(video_path):
            return GPSExtractionResult(
                success=False,
                gps_data=[],
                method_used='none',
                total_points=0,
                error_message=f"Video file not found: {video_path}"
            )
        
        # Determine extraction method
        if method == 'auto':
            # Prefer GoPro API for better accuracy, fallback to ExifTool
            if 'gopro_api' in self.methods_available:
                method = 'gopro_api'
            elif 'exiftool' in self.methods_available:
                method = 'exiftool'
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used='none',
                    total_points=0,
                    error_message="No GPS extraction methods available"
                )
        
        logger.info(f"Extracting GPS data from {video_path} using {method} method")
        
        try:
            if method == 'exiftool':
                return self._extract_with_exiftool(video_path)
            elif method == 'gopro_api':
                return self._extract_with_gopro_api(video_path)
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used=method,
                    total_points=0,
                    error_message=f"Unknown extraction method: {method}"
                )
        except Exception as e:
            logger.error(f"Error extracting GPS data: {e}")
            return GPSExtractionResult(
                success=False,
                gps_data=[],
                method_used=method,
                total_points=0,
                error_message=str(e)
            )
    
    def _extract_with_exiftool(self, video_path: str) -> GPSExtractionResult:
        """Extract GPS data using ExifTool method"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            metadata_file = os.path.join(temp_dir, 'metadata.xml')
            gps_file = os.path.join(temp_dir, 'gps_data.txt')
            
            # Extract metadata using ExifTool
            cmd = [
                'exiftool',
                '-api', 'largefilesupport=1',
                '-ee',  # Extract embedded data
                '-G3',  # Show group names
                '-X',   # XML format
                video_path
            ]
            
            logger.debug(f"Running ExifTool command: {' '.join(cmd)}")
            
            with open(metadata_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                      text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"ExifTool failed: {result.stderr}")
            
            # Extract Track4 GPS data
            self._extract_track4_data(metadata_file, gps_file)
            
            # Parse GPS data
            gps_data = self._parse_gps_file(gps_file)
            
            if gps_data:
                time_range = (gps_data[0].timestamp, gps_data[-1].timestamp)
                return GPSExtractionResult(
                    success=True,
                    gps_data=gps_data,
                    method_used='exiftool',
                    total_points=len(gps_data),
                    time_range=time_range
                )
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used='exiftool',
                    total_points=0,
                    error_message="No GPS data found in metadata"
                )
                
        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _extract_with_gopro_api(self, video_path: str) -> GPSExtractionResult:
        """Extract GPS data using GoPro API method"""
        try:
            # Extract telemetry data
            telem = telemetry.Telemetry(video_path)
            
            if not telem.has_gps():
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used='gopro_api',
                    total_points=0,
                    error_message="No GPS data found in video"
                )
            
            # Get GPS track
            gps_track = telem.gps_track()
            gps_data = []
            
            for point in gps_track:
                if point.lat != 0.0 and point.lon != 0.0:
                    # Convert timestamp to seconds
                    timestamp = point.timestamp.total_seconds() if hasattr(point.timestamp, 'total_seconds') else point.timestamp
                    
                    gps_point = GPSData(
                        timestamp=float(timestamp),
                        latitude=float(point.lat),
                        longitude=float(point.lon),
                        altitude=float(getattr(point, 'alt', 0.0)),
                        heading=float(getattr(point, 'heading', 0.0)),
                        accuracy=float(getattr(point, 'dop', 1.0))
                    )
                    gps_data.append(gps_point)
            
            if gps_data:
                time_range = (gps_data[0].timestamp, gps_data[-1].timestamp)
                return GPSExtractionResult(
                    success=True,
                    gps_data=gps_data,
                    method_used='gopro_api',
                    total_points=len(gps_data),
                    time_range=time_range
                )
            else:
                return GPSExtractionResult(
                    success=False,
                    gps_data=[],
                    method_used='gopro_api',
                    total_points=0,
                    error_message="No valid GPS points found"
                )
                
        except Exception as e:
            raise RuntimeError(f"GoPro API extraction failed: {e}")
    
    def _extract_track4_data(self, metadata_file: str, output_file: str) -> None:
        """Extract Track4 GPS data from metadata XML file"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as in_file, \
                 open(output_file, 'w', encoding='utf-8') as out_file:
                
                for line in in_file:
                    # Look for Track4 GPS data
                    if 'Track4' in line or 'GPS' in line:
                        out_file.write(line)
                        
            logger.debug(f"Extracted Track4 data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error extracting Track4 data: {e}")
            raise
    
    def _parse_gps_file(self, gps_file: str) -> List[GPSData]:
        """Parse GPS data from extracted Track4 file"""
        gps_data = []
        
        try:
            with open(gps_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Skip first two lines if they exist
            lines = content.split('\n')[2:] if len(content.split('\n')) > 2 else content.split('\n')
            
            current_timestamp = None
            current_lat = None
            current_lon = None
            
            for line in lines:
                if not line.strip():
                    continue
                    
                # Parse XML-like content
                soup = BeautifulSoup(line, "html.parser")
                text_content = soup.get_text()
                
                # Look for GPS tags
                if ':GPSLatitude>' in line:
                    current_lat = self._convert_gps_coordinate(text_content)
                elif ':GPSLongitude>' in line and current_lat is not None:
                    current_lon = self._convert_gps_coordinate(text_content)
                elif ':GPSDateTime>' in line:
                    current_timestamp = self._convert_timestamp(text_content)
                    
                    # If we have complete GPS data, save it
                    if (current_timestamp is not None and 
                        current_lat is not None and 
                        current_lon is not None and
                        current_lat != 0.0 and current_lon != 0.0):
                        
                        gps_point = GPSData(
                            timestamp=current_timestamp,
                            latitude=current_lat,
                            longitude=current_lon,
                            altitude=0.0,
                            heading=0.0,
                            accuracy=1.0
                        )
                        gps_data.append(gps_point)
                        
                        # Reset for next point
                        current_lat = None
                        current_lon = None
            
            logger.info(f"Parsed {len(gps_data)} GPS points from file")
            return gps_data
            
        except Exception as e:
            logger.error(f"Error parsing GPS file: {e}")
            return []
    
    def _convert_gps_coordinate(self, coord_str: str) -> float:
        """Convert GPS coordinate from DMS format to decimal degrees"""
        if not coord_str or not isinstance(coord_str, str):
            return 0.0
            
        try:
            # Clean the string
            coord_str = coord_str.strip()
            if coord_str.startswith('<'):
                coord_str = coord_str[1:]
            if coord_str.endswith('>'):
                coord_str = coord_str[:-1]
                
            # Parse DMS format: "deg min' sec" N/S/E/W"
            parts = coord_str.split(' ')
            if len(parts) < 6:
                logger.warning(f"Invalid GPS coordinate format: {coord_str}")
                return 0.0
            
            degrees = float(parts[1])
            minutes = float(parts[3].replace("'", ""))
            seconds = float(parts[4].replace('"', ""))
            direction = parts[5][0] if len(parts[5]) > 0 else 'N'
            
            # Convert to decimal degrees
            decimal = degrees + minutes/60.0 + seconds/3600.0
            
            # Apply sign based on direction
            if direction in ['S', 'W']:
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
            # Clean timestamp string
            timestamp_str = timestamp_str.strip()
            if timestamp_str.startswith('<'):
                timestamp_str = timestamp_str[1:]
            if timestamp_str.endswith('>'):
                timestamp_str = timestamp_str[:-1]
            
            # Parse timestamp formats
            try:
                # Try with microseconds
                dt = datetime.strptime(timestamp_str, '%Y:%m:%d %H:%M:%S.%f')
            except ValueError:
                # Try without microseconds
                dt = datetime.strptime(timestamp_str, '%Y:%m:%d %H:%M:%S')
            
            return dt.timestamp()
            
        except ValueError as e:
            logger.warning(f"Error converting timestamp '{timestamp_str}': {e}")
            return 0.0
    
    def synchronize_with_video(self, gps_data: List[GPSData], 
                              video_duration: float,
                              target_fps: float = 10.0) -> List[GPSData]:
        """
        Synchronize GPS data with video timeline
        
        Args:
            gps_data: Raw GPS data
            video_duration: Video duration in seconds
            target_fps: Target GPS sampling rate
            
        Returns:
            List[GPSData]: Synchronized GPS data
        """
        if not gps_data:
            return []
        
        # Sort GPS data by timestamp
        sorted_gps = sorted(gps_data, key=lambda x: x.timestamp)
        
        # Normalize timestamps to start from 0
        start_time = sorted_gps[0].timestamp
        for gps_point in sorted_gps:
            gps_point.timestamp -= start_time
        
        # Create synchronized timeline
        sync_interval = 1.0 / target_fps
        sync_timeline = np.arange(0, video_duration, sync_interval)
        
        # Interpolate GPS data to match timeline
        timestamps = np.array([gps.timestamp for gps in sorted_gps])
        latitudes = np.array([gps.latitude for gps in sorted_gps])
        longitudes = np.array([gps.longitude for gps in sorted_gps])
        
        # Interpolate
        sync_gps = []
        for sync_time in sync_timeline:
            if sync_time <= timestamps[-1]:
                # Find closest GPS points for interpolation
                idx = np.searchsorted(timestamps, sync_time)
                
                if idx == 0:
                    # Use first point
                    lat = latitudes[0]
                    lon = longitudes[0]
                elif idx >= len(timestamps):
                    # Use last point
                    lat = latitudes[-1]
                    lon = longitudes[-1]
                else:
                    # Linear interpolation
                    t1, t2 = timestamps[idx-1], timestamps[idx]
                    lat1, lat2 = latitudes[idx-1], latitudes[idx]
                    lon1, lon2 = longitudes[idx-1], longitudes[idx]
                    
                    alpha = (sync_time - t1) / (t2 - t1)
                    lat = lat1 + alpha * (lat2 - lat1)
                    lon = lon1 + alpha * (lon2 - lon1)
                
                sync_point = GPSData(
                    timestamp=sync_time,
                    latitude=lat,
                    longitude=lon,
                    altitude=0.0,
                    heading=0.0,
                    accuracy=1.0
                )
                sync_gps.append(sync_point)
        
        logger.info(f"Synchronized {len(sync_gps)} GPS points for {video_duration:.1f}s video")
        return sync_gps


def extract_gps_from_stereo_videos(left_video: str, 
                                  right_video: str,
                                  method: str = 'auto') -> Tuple[List[GPSData], str]:
    """
    Extract GPS data from stereo video pair
    
    Args:
        left_video: Path to left camera video
        right_video: Path to right camera video  
        method: Extraction method ('auto', 'exiftool', 'gopro_api')
        
    Returns:
        Tuple[List[GPSData], str]: GPS data and method used
    """
    extractor = GoProGPSExtractor()
    
    # Try extracting from left video first
    logger.info("Attempting GPS extraction from left video")
    result_left = extractor.extract_gps_data(left_video, method)
    
    if result_left.success and result_left.total_points > 0:
        logger.info(f"Successfully extracted {result_left.total_points} GPS points from left video")
        return result_left.gps_data, result_left.method_used
    
    # Fallback to right video
    logger.info("Left video GPS extraction failed, trying right video")
    result_right = extractor.extract_gps_data(right_video, method)
    
    if result_right.success and result_right.total_points > 0:
        logger.info(f"Successfully extracted {result_right.total_points} GPS points from right video")
        return result_right.gps_data, result_right.method_used
    
    # No GPS data found
    logger.warning("No GPS data found in either video")
    return [], 'none'


def save_gps_to_csv(gps_data: List[GPSData], output_path: str) -> None:
    """
    Save GPS data to CSV file for Argus Track
    
    Args:
        gps_data: GPS data to save
        output_path: Path to output CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'latitude', 'longitude', 'altitude', 'heading', 'accuracy'])
        
        for gps in gps_data:
            writer.writerow([
                gps.timestamp,
                gps.latitude,
                gps.longitude,
                gps.altitude,
                gps.heading,
                gps.accuracy
            ])
    
    logger.info(f"Saved {len(gps_data)} GPS points to {output_path}")