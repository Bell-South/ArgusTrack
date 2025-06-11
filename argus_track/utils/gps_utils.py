# argus_track/utils/gps_utils.py

"""Enhanced GPS utilities for tracking"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pyproj
from scipy.interpolate import interp1d

from ..core import GPSData


class GPSInterpolator:
    """Interpolate GPS data between frames"""

    def __init__(self, gps_data: List[GPSData]):
        """
        Initialize GPS interpolator

        Args:
            gps_data: List of GPS data points
        """
        self.gps_data = sorted(gps_data, key=lambda x: x.timestamp)
        self.timestamps = np.array([gps.timestamp for gps in self.gps_data])

        # Create interpolation functions
        self.lat_interp = interp1d(
            self.timestamps,
            [gps.latitude for gps in self.gps_data],
            kind="linear",
            fill_value="extrapolate",
        )
        self.lon_interp = interp1d(
            self.timestamps,
            [gps.longitude for gps in self.gps_data],
            kind="linear",
            fill_value="extrapolate",
        )
        self.heading_interp = interp1d(
            self.timestamps,
            [gps.heading for gps in self.gps_data],
            kind="linear",
            fill_value="extrapolate",
        )

    def interpolate(self, timestamp: float) -> GPSData:
        """
        Interpolate GPS data for a specific timestamp

        Args:
            timestamp: Target timestamp

        Returns:
            Interpolated GPS data
        """
        return GPSData(
            timestamp=timestamp,
            latitude=float(self.lat_interp(timestamp)),
            longitude=float(self.lon_interp(timestamp)),
            altitude=0.0,  # We're not focusing on altitude
            heading=float(self.heading_interp(timestamp)),
            accuracy=1.0,  # Interpolated accuracy
        )

    def get_range(self) -> Tuple[float, float]:
        """Get timestamp range of GPS data"""
        return self.timestamps[0], self.timestamps[-1]


class CoordinateTransformer:
    """Transform between GPS coordinates and local coordinate systems"""

    def __init__(self, reference_lat: float, reference_lon: float):
        """
        Initialize transformer with reference point

        Args:
            reference_lat: Reference latitude
            reference_lon: Reference longitude
        """
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon

        # Setup projections
        self.wgs84 = pyproj.CRS("EPSG:4326")  # GPS coordinates
        self.utm = pyproj.CRS(f"EPSG:{self._get_utm_zone()}")
        self.transformer = pyproj.Transformer.from_crs(
            self.wgs84, self.utm, always_xy=True
        )
        self.inverse_transformer = pyproj.Transformer.from_crs(
            self.utm, self.wgs84, always_xy=True
        )

        # Calculate reference point in UTM
        self.ref_x, self.ref_y = self.transformer.transform(
            reference_lon, reference_lat
        )

    def _get_utm_zone(self) -> int:
        """Get UTM zone for reference point"""
        zone = int((self.reference_lon + 180) / 6) + 1
        if self.reference_lat >= 0:
            return 32600 + zone  # Northern hemisphere
        else:
            return 32700 + zone  # Southern hemisphere

    def gps_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS coordinates to local coordinate system

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            (x, y) in meters from reference point
        """
        utm_x, utm_y = self.transformer.transform(lon, lat)
        return utm_x - self.ref_x, utm_y - self.ref_y

    def local_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert local coordinates to GPS

        Args:
            x: X coordinate in meters from reference
            y: Y coordinate in meters from reference

        Returns:
            (latitude, longitude)
        """
        utm_x = x + self.ref_x
        utm_y = y + self.ref_y
        lon, lat = self.inverse_transformer.transform(utm_x, utm_y)
        return lat, lon

    def distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS points

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Distance in meters
        """
        x1, y1 = self.gps_to_local(lat1, lon1)
        x2, y2 = self.gps_to_local(lat2, lon2)
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


@dataclass
class GeoLocation:
    """Represents a geographic location with reliability information"""

    latitude: float
    longitude: float
    accuracy: float = 1.0  # Accuracy in meters
    reliability: float = 1.0  # Value between 0 and 1
    timestamp: Optional[float] = None


def sync_gps_with_frames(
    gps_data: List[GPSData], video_fps: float, start_timestamp: Optional[float] = None
) -> List[GPSData]:
    """
    Synchronize GPS data with video frames

    Args:
        gps_data: List of GPS data points
        video_fps: Video frame rate
        start_timestamp: Optional start timestamp

    Returns:
        List of GPS data aligned with frames
    """
    if not gps_data:
        return []

    # Sort GPS data by timestamp
    gps_data = sorted(gps_data, key=lambda x: x.timestamp)

    # Determine start timestamp
    if start_timestamp is None:
        start_timestamp = gps_data[0].timestamp

    # Create interpolator
    interpolator = GPSInterpolator(gps_data)

    # Generate frame-aligned GPS data
    frame_gps = []
    frame_duration = 1.0 / video_fps

    timestamp = start_timestamp
    while timestamp <= gps_data[-1].timestamp:
        frame_gps.append(interpolator.interpolate(timestamp))
        timestamp += frame_duration

    return frame_gps


def compute_average_location(locations: List[GPSData]) -> GeoLocation:
    """
    Compute the average location from multiple GPS points

    Args:
        locations: List of GPS data points

    Returns:
        Average location with reliability score
    """
    if not locations:
        return GeoLocation(0.0, 0.0, 0.0, 0.0)

    # Simple weighted average based on accuracy
    weights = np.array([1.0 / max(loc.accuracy, 0.1) for loc in locations])
    weights = weights / np.sum(weights)  # Normalize

    avg_lat = np.sum([loc.latitude * w for loc, w in zip(locations, weights)])
    avg_lon = np.sum([loc.longitude * w for loc, w in zip(locations, weights)])

    # Calculate reliability based on consistency of points
    if len(locations) > 1:
        # Create transformer using the first point as reference
        transformer = CoordinateTransformer(
            locations[0].latitude, locations[0].longitude
        )

        # Calculate standard deviation in meters
        distances = []
        for loc in locations:
            dist = transformer.distance(loc.latitude, loc.longitude, avg_lat, avg_lon)
            distances.append(dist)

        std_dev = np.std(distances)
        reliability = 1.0 / (
            1.0 + std_dev / 10.0
        )  # Decreases with higher standard deviation
        reliability = min(1.0, max(0.1, reliability))  # Clamp between 0.1 and 1.0
    else:
        reliability = 0.5  # Only one point, medium reliability

    # Average accuracy is the weighted average of individual accuracies
    avg_accuracy = np.sum([loc.accuracy * w for loc, w in zip(locations, weights)])

    # Use the latest timestamp
    latest_timestamp = max([loc.timestamp for loc in locations])

    return GeoLocation(
        latitude=avg_lat,
        longitude=avg_lon,
        accuracy=avg_accuracy,
        reliability=reliability,
        timestamp=latest_timestamp,
    )


def filter_gps_outliers(
    locations: List[GPSData], threshold_meters: float = 30.0
) -> List[GPSData]:
    """
    Filter outliers from GPS data using DBSCAN clustering

    Args:
        locations: List of GPS data points
        threshold_meters: Distance threshold for outlier detection

    Returns:
        Filtered list of GPS data points
    """
    if len(locations) <= 2:
        return locations

    from sklearn.cluster import DBSCAN

    # Create transformer using the first point as reference
    transformer = CoordinateTransformer(locations[0].latitude, locations[0].longitude)

    # Convert to local coordinates
    local_points = []
    for loc in locations:
        x, y = transformer.gps_to_local(loc.latitude, loc.longitude)
        local_points.append([x, y])

    # Cluster points
    clustering = DBSCAN(eps=threshold_meters, min_samples=1).fit(local_points)

    # Find the largest cluster
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]

    # Keep only points from the largest cluster
    filtered_locations = [
        loc for i, loc in enumerate(locations) if labels[i] == largest_cluster
    ]

    return filtered_locations
