# argus_track/utils/io.py

"""I/O utilities for loading and saving tracking data"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import csv

import numpy as np

from ..core import GPSData, Track


def setup_logging(log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> None:
    """
    Setup logging configuration
    
    Args:
        log_file: Optional log file path
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_tracking_results(tracks: Dict[int, List[Dict]], 
                         output_path: Path,
                         metadata: Optional[Dict[str, Any]] = None,
                         gps_tracks: Optional[Dict[int, List[GPSData]]] = None,
                         track_locations: Optional[Dict[int, Dict]] = None) -> None:
    """
    Save tracking results to JSON file
    
    Args:
        tracks: Dictionary of track histories
        output_path: Path for output file
        metadata: Optional metadata to include
        gps_tracks: Optional GPS data for tracks
        track_locations: Optional estimated locations for tracks
    """
    results = {
        'metadata': metadata or {},
        'tracks': tracks
    }
    
    # Add GPS data if provided
    if gps_tracks:
        results['gps_tracks'] = {
            track_id: [gps.to_dict() for gps in gps_list]
            for track_id, gps_list in gps_tracks.items()
        }
        
    # Add track locations if provided
    if track_locations:
        results['track_locations'] = track_locations
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Saved tracking results to {output_path}")


def load_tracking_results(input_path: Path) -> Dict[str, Any]:
    """
    Load tracking results from JSON file
    
    Args:
        input_path: Path to input file
        
    Returns:
        Dictionary with tracking results
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    logging.info(f"Loaded tracking results from {input_path}")
    return results


def load_gps_data(gps_file: str) -> List[GPSData]:
    """
    Load GPS data from file
    
    Args:
        gps_file: Path to GPS data file (CSV format)
        
    Returns:
        List of GPS data points
    """
    gps_data = []
    
    with open(gps_file, 'r') as f:
        reader = csv.reader(f)
        # Skip header if exists
        header = next(reader, None)
        
        try:
            for row in reader:
                if len(row) >= 5:
                    gps_data.append(GPSData(
                        timestamp=float(row[0]),
                        latitude=float(row[1]),
                        longitude=float(row[2]),
                        altitude=float(row[3]) if len(row) > 3 else 0.0,
                        heading=float(row[4]) if len(row) > 4 else 0.0,
                        accuracy=float(row[5]) if len(row) > 5 else 1.0
                    ))
        except ValueError as e:
            logging.error(f"Error parsing GPS data: {e}")
            logging.error(f"Problematic row: {row}")
            raise
    
    logging.info(f"Loaded {len(gps_data)} GPS data points from {gps_file}")
    return gps_data


def save_gps_data(gps_data: List[GPSData], output_path: str) -> None:
    """
    Save GPS data to CSV file
    
    Args:
        gps_data: List of GPS data points
        output_path: Path for output file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['timestamp', 'latitude', 'longitude', 
                        'altitude', 'heading', 'accuracy'])
        
        # Write data
        for gps in gps_data:
            writer.writerow([
                gps.timestamp,
                gps.latitude,
                gps.longitude,
                gps.altitude,
                gps.heading,
                gps.accuracy
            ])
    
    logging.info(f"Saved {len(gps_data)} GPS data points to {output_path}")


def export_locations_to_csv(track_locations: Dict[int, Dict],
                           output_path: str) -> None:
    """
    Export estimated track locations to CSV
    
    Args:
        track_locations: Dictionary of track locations
        output_path: Output CSV path
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'latitude', 'longitude', 
                         'accuracy', 'reliability', 'timestamp'])
        
        for track_id, location in track_locations.items():
            writer.writerow([
                track_id,
                location['latitude'],
                location['longitude'],
                location.get('accuracy', 1.0),
                location.get('reliability', 1.0),
                location.get('timestamp', '')
            ])
    
    logging.info(f"Exported {len(track_locations)} locations to CSV: {output_path}")


def export_tracks_to_csv(tracks: Dict[int, Track], 
                        output_path: str) -> None:
    """
    Export track data to CSV format
    
    Args:
        tracks: Dictionary of tracks
        output_path: Path for output CSV file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['track_id', 'frame', 'x1', 'y1', 'x2', 'y2', 
                        'state', 'hits', 'age'])
        
        # Write track data
        for track_id, track in tracks.items():
            for detection in track.detections:
                x1, y1, x2, y2 = detection.tlbr
                writer.writerow([
                    track_id,
                    detection.frame_id,
                    x1, y1, x2, y2,
                    track.state,
                    track.hits,
                    track.age
                ])
    
    logging.info(f"Exported tracks to CSV: {output_path}")


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
# argus_track/utils/io.py (continued)

    if path.suffix == '.yaml' or path.suffix == '.yml':
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    logging.info(f"Loaded configuration from {config_path}")
    return config


def export_to_geojson(track_locations: Dict[int, Dict], 
                     output_path: str,
                     properties: Optional[Dict[int, Dict]] = None) -> None:
    """
    Export track locations to GeoJSON format
    
    Args:
        track_locations: Dictionary of track locations
        output_path: Path for output GeoJSON file
        properties: Optional additional properties for each feature
    """
    features = []
    
    for track_id, location in track_locations.items():
        # Create basic properties
        feature_props = {
            'track_id': track_id,
            'accuracy': location.get('accuracy', 1.0),
            'reliability': location.get('reliability', 1.0)
        }
        
        # Add additional properties if provided
        if properties and track_id in properties:
            feature_props.update(properties[track_id])
            
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [location['longitude'], location['latitude']]
            },
            'properties': feature_props
        }
        
        features.append(feature)
    
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    logging.info(f"Exported {len(features)} locations to GeoJSON: {output_path}")