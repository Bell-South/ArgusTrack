# Add this to argus_track/utils/kalman_gps_filter.py - NEW FILE

"""
Kalman Filter for GPS Location Deduplication
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter


@dataclass 
class GPSMeasurement:
    """Single GPS measurement for a LED"""
    latitude: float
    longitude: float
    detection_count: int
    confidence: float
    distance_m: float
    track_id: int
    

class LEDLocationKalmanFilter:
    """
    Kalman filter for estimating true LED position from multiple GPS measurements
    
    State: [latitude, longitude, lat_velocity, lon_velocity]
    Measurements: [latitude, longitude]
    """
    
    def __init__(self, initial_lat: float, initial_lon: float):
        """
        Initialize Kalman filter for LED location estimation
        
        Args:
            initial_lat: Initial latitude estimate
            initial_lon: Initial longitude estimate
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        dt = 1.0  # Time step (not critical for static objects)
        self.kf.F = np.array([
            [1, 0, dt, 0],   # lat = lat + lat_vel*dt
            [0, 1, 0, dt],   # lon = lon + lon_vel*dt  
            [0, 0, 1, 0],    # lat_vel = lat_vel
            [0, 0, 0, 1]     # lon_vel = lon_vel
        ])
        
        # Measurement matrix (we measure lat, lon directly)
        self.kf.H = np.array([
            [1, 0, 0, 0],    # measure latitude
            [0, 1, 0, 0]     # measure longitude
        ])
        
        # Initial state [lat, lon, lat_vel, lon_vel]
        self.kf.x = np.array([initial_lat, initial_lon, 0, 0])
        
        # Initial uncertainty (higher for velocities since LEDs are static)
        self.kf.P = np.diag([1e-6, 1e-6, 1e-8, 1e-8])  # Very low uncertainty for static objects
        
        # Process noise (very low for static LEDs)
        self.kf.Q = np.diag([1e-8, 1e-8, 1e-10, 1e-10])
        
        # Measurement noise (will be updated based on measurement quality)
        self.kf.R = np.diag([1e-6, 1e-6])  # Base measurement noise
        
        self.measurements: List[GPSMeasurement] = []
        self.logger = logging.getLogger(f"{__name__}.LEDLocationKalmanFilter")
    
    def add_measurement(self, measurement: GPSMeasurement):
        """
        Add a GPS measurement and update Kalman filter
        
        Args:
            measurement: GPS measurement for this LED
        """
        self.measurements.append(measurement)
        
        # Calculate measurement noise based on quality
        measurement_noise = self._calculate_measurement_noise(measurement)
        
        # Update measurement noise matrix
        self.kf.R = np.diag([measurement_noise, measurement_noise])
        
        # Kalman predict step
        self.kf.predict()
        
        # Kalman update step
        z = np.array([measurement.latitude, measurement.longitude])
        self.kf.update(z)
        
        self.logger.debug(f"Added measurement for track {measurement.track_id}: "
                         f"({measurement.latitude:.6f}, {measurement.longitude:.6f}), "
                         f"noise: {measurement_noise:.2e}")
    
    def _calculate_measurement_noise(self, measurement: GPSMeasurement) -> float:
        """
        Calculate measurement noise based on measurement quality
        
        Lower noise = more trusted measurement
        """
        # Base noise level
        base_noise = 1e-6
        
        # Factors that increase noise (less reliable):
        # 1. Lower detection count
        detection_factor = 1.0 / max(1, measurement.detection_count)
        
        # 2. Lower confidence
        confidence_factor = 1.0 / max(0.1, measurement.confidence)
        
        # 3. Greater distance from estimated center (outlier detection)
        if len(self.measurements) > 0:
            # Distance from current filter estimate
            current_lat, current_lon = self.kf.x[0], self.kf.x[1]
            distance_to_estimate = self._gps_distance(
                current_lat, current_lon,
                measurement.latitude, measurement.longitude
            )
            # Penalize measurements far from current estimate
            distance_factor = 1.0 + distance_to_estimate * 1000  # Convert to penalty
        else:
            distance_factor = 1.0
        
        # Combined noise
        total_noise = base_noise * detection_factor * confidence_factor * distance_factor
        
        # Cap the noise (don't completely ignore any measurement)
        return min(total_noise, 1e-4)
    
    def get_estimated_location(self) -> Tuple[float, float, float]:
        """
        Get final estimated location with uncertainty
        
        Returns:
            (latitude, longitude, uncertainty_meters)
        """
        if len(self.measurements) == 0:
            return 0.0, 0.0, float('inf')
        
        lat_estimate = float(self.kf.x[0])
        lon_estimate = float(self.kf.x[1])
        
        # Calculate uncertainty in meters
        lat_variance = float(self.kf.P[0, 0])
        lon_variance = float(self.kf.P[1, 1])
        
        # Convert variance to standard deviation in meters
        lat_std_m = np.sqrt(lat_variance) * 111000  # Rough conversion
        lon_std_m = np.sqrt(lon_variance) * 111000 * np.cos(np.radians(lat_estimate))
        uncertainty_m = np.sqrt(lat_std_m**2 + lon_std_m**2)
        
        return lat_estimate, lon_estimate, uncertainty_m
    
    def _gps_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between GPS points in meters"""
        R = 6378137.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


class KalmanGPSDeduplicator:
    """
    GPS deduplication using Kalman filters for optimal location estimation
    """
    
    def __init__(self, merge_distance_m: float = 3.0):
        """
        Initialize Kalman GPS deduplicator
        
        Args:
            merge_distance_m: Distance threshold for clustering LEDs
        """
        self.merge_distance_m = merge_distance_m
        self.logger = logging.getLogger(f"{__name__}.KalmanGPSDeduplicator")
    
    def deduplicate_locations(self, features: List[dict]) -> List[dict]:
        """
        Deduplicate GPS locations using Kalman filtering
        
        Args:
            features: List of GeoJSON features
            
        Returns:
            Deduplicated features with Kalman-estimated locations
        """
        if len(features) <= 1:
            return features
        
        self.logger.info(f"🔄 Starting Kalman GPS deduplication with {len(features)} locations")
        self.logger.info(f"   Merge threshold: {self.merge_distance_m} meters")
        
        # Step 1: Cluster nearby features
        clusters = self._cluster_features(features)
        
        # Step 2: Apply Kalman filtering to each cluster
        deduplicated_features = []
        for cluster in clusters:
            if len(cluster) == 1:
                # Single feature, no filtering needed
                deduplicated_features.append(cluster[0])
            else:
                # Multiple features, apply Kalman filtering
                filtered_feature = self._apply_kalman_filtering(cluster)
                deduplicated_features.append(filtered_feature)
        
        removed_count = len(features) - len(deduplicated_features)
        self.logger.info(f"✅ Kalman GPS deduplication complete:")
        self.logger.info(f"   Original locations: {len(features)}")
        self.logger.info(f"   Clusters found: {len(clusters)}")
        self.logger.info(f"   Final locations: {len(deduplicated_features)}")
        self.logger.info(f"   Duplicates removed: {removed_count}")
        
        return deduplicated_features
    
    def _cluster_features(self, features: List[dict]) -> List[List[dict]]:
        """
        Cluster features by proximity using single-linkage clustering
        """
        clusters = []
        used_indices = set()
        
        for i, feature in enumerate(features):
            if i in used_indices:
                continue
            
            # Start new cluster
            cluster = [feature]
            cluster_indices = {i}
            
            lat1 = feature['geometry']['coordinates'][1]
            lon1 = feature['geometry']['coordinates'][0]
            
            # Find all features within merge distance
            for j, other_feature in enumerate(features):
                if j == i or j in used_indices:
                    continue
                
                lat2 = other_feature['geometry']['coordinates'][1]
                lon2 = other_feature['geometry']['coordinates'][0]
                
                distance = self._gps_distance(lat1, lon1, lat2, lon2)
                
                if distance <= self.merge_distance_m:
                    cluster.append(other_feature)
                    cluster_indices.add(j)
                    
                    self.logger.debug(f"   Clustering tracks {feature['properties']['track_id']} "
                                    f"and {other_feature['properties']['track_id']} "
                                    f"(distance: {distance:.1f}m)")
            
            used_indices.update(cluster_indices)
            clusters.append(cluster)
        
        return clusters
    
    def _apply_kalman_filtering(self, cluster: List[dict]) -> dict:
        """
        Apply Kalman filtering to a cluster of features
        """
        # Calculate initial estimate (centroid)
        initial_lat = np.mean([f['geometry']['coordinates'][1] for f in cluster])
        initial_lon = np.mean([f['geometry']['coordinates'][0] for f in cluster])
        
        # Create Kalman filter
        kalman_filter = LEDLocationKalmanFilter(initial_lat, initial_lon)
        
        # Add all measurements to filter
        for feature in cluster:
            measurement = GPSMeasurement(
                latitude=feature['geometry']['coordinates'][1],
                longitude=feature['geometry']['coordinates'][0],
                detection_count=feature['properties']['detection_count'],
                confidence=feature['properties']['confidence'],
                distance_m=feature['properties']['estimated_distance_m'],
                track_id=feature['properties']['track_id']
            )
            kalman_filter.add_measurement(measurement)
        
        # Get final filtered estimate
        final_lat, final_lon, uncertainty_m = kalman_filter.get_estimated_location()
        
        # Create merged feature
        total_detections = sum(f['properties']['detection_count'] for f in cluster)
        avg_confidence = np.mean([f['properties']['confidence'] for f in cluster])
        avg_distance = np.mean([f['properties']['estimated_distance_m'] for f in cluster])
        
        # Use track with most detections as primary
        primary_track = max(cluster, key=lambda f: f['properties']['detection_count'])
        merged_track_ids = [f['properties']['track_id'] for f in cluster]
        
        merged_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(final_lon), float(final_lat)]
            },
            "properties": {
                "track_id": primary_track['properties']['track_id'],
                "merged_tracks": merged_track_ids,
                "confidence": round(float(avg_confidence), 3),
                "estimated_distance_m": round(float(avg_distance), 1),
                "detection_count": int(total_detections),
                "class_id": primary_track['properties']['class_id'],
                "processing_method": "kalman_filtered_gps_deduplication",
                "merge_cluster_size": len(cluster)
            }
        }
        
        self.logger.debug(f"   Kalman filtered cluster of {len(cluster)} tracks → "
                         f"Final location: ({final_lat:.6f}, {final_lon:.6f})")
        
        return merged_feature
    
    def _gps_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between GPS points in meters"""
        R = 6378137.0
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


# Convenience function
def create_kalman_gps_deduplicator(merge_distance_m: float = 3.0) -> KalmanGPSDeduplicator:
    """
    Create a Kalman GPS deduplicator
    
    Args:
        merge_distance_m: Distance threshold for merging duplicates
        
    Returns:
        Configured Kalman GPS deduplicator
    """
    return KalmanGPSDeduplicator(merge_distance_m)