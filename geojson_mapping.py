#!/usr/bin/env python3
"""
GeoJSON Visualization Script for Argus Track Results
===================================================

This script creates interactive maps and visualizations from the geolocated
objects produced by Argus Track. It supports multiple output formats:
- Interactive HTML maps with Folium
- Static plots with Matplotlib
- Data analysis and statistics
"""

import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Try to import optional dependencies
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("⚠️  Folium not available. Install with: pip install folium")

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️  GeoPandas not available. Install with: pip install geopandas")

# Set up plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class GeoJSONVisualizer:
    """Visualize GeoJSON results from Argus Track"""
    
    def __init__(self, geojson_path: str):
        """
        Initialize visualizer with GeoJSON file
        
        Args:
            geojson_path: Path to GeoJSON file
        """
        self.geojson_path = Path(geojson_path)
        self.logger = logging.getLogger(__name__)
        
        # Load and parse GeoJSON
        self.data = self._load_geojson()
        self.df = self._geojson_to_dataframe()
        
        print(f"📊 Loaded {len(self.df)} geolocated objects from {geojson_path}")
    
    def _load_geojson(self) -> dict:
        """Load GeoJSON file"""
        if not self.geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {self.geojson_path}")
        
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        if data.get('type') != 'FeatureCollection':
            raise ValueError("Invalid GeoJSON: Expected FeatureCollection")
        
        return data
    
    def _geojson_to_dataframe(self) -> pd.DataFrame:
        """Convert GeoJSON features to pandas DataFrame"""
        features = self.data.get('features', [])
        
        if not features:
            print("⚠️  No features found in GeoJSON")
            return pd.DataFrame()
        
        # Extract data from features
        rows = []
        for feature in features:
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            
            if geometry.get('type') == 'Point':
                coords = geometry.get('coordinates', [0, 0])
                row = {
                    'longitude': coords[0],
                    'latitude': coords[1],
                    **properties
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Clean up data types
        numeric_cols = ['confidence', 'reliability', 'accuracy_meters', 
                       'estimated_distance_m', 'first_frame', 'last_frame', 
                       'duration_frames', 'detection_count']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def print_summary(self):
        """Print summary statistics"""
        if self.df.empty:
            print("❌ No data to summarize")
            return
        
        print("\n📊 DETECTION SUMMARY")
        print("=" * 50)
        
        # Basic counts
        total_objects = len(self.df)
        classes = self.df['class_name'].value_counts() if 'class_name' in self.df.columns else {}
        
        print(f"Total Objects: {total_objects}")
        
        if classes.any():
            print("\nClass Distribution:")
            for class_name, count in classes.items():
                print(f"  {class_name}: {count}")
        
        # Quality metrics
        if 'confidence' in self.df.columns:
            print(f"\nConfidence Stats:")
            print(f"  Mean: {self.df['confidence'].mean():.3f}")
            print(f"  Range: {self.df['confidence'].min():.3f} - {self.df['confidence'].max():.3f}")
        
        if 'reliability' in self.df.columns:
            print(f"\nReliability Stats:")
            print(f"  Mean: {self.df['reliability'].mean():.3f}")
            print(f"  Range: {self.df['reliability'].min():.3f} - {self.df['reliability'].max():.3f}")
        
        if 'accuracy_meters' in self.df.columns:
            print(f"\nAccuracy Stats:")
            print(f"  Mean: {self.df['accuracy_meters'].mean():.1f}m")
            print(f"  Range: {self.df['accuracy_meters'].min():.1f}m - {self.df['accuracy_meters'].max():.1f}m")
        
        # Geographic bounds
        print(f"\nGeographic Bounds:")
        print(f"  Latitude: {self.df['latitude'].min():.6f} to {self.df['latitude'].max():.6f}")
        print(f"  Longitude: {self.df['longitude'].min():.6f} to {self.df['longitude'].max():.6f}")
        
        # Calculate area covered
        lat_range = self.df['latitude'].max() - self.df['latitude'].min()
        lon_range = self.df['longitude'].max() - self.df['longitude'].min()
        
        # Convert to approximate meters (rough calculation)
        lat_meters = lat_range * 111000  # 1 degree lat ≈ 111km
        lon_meters = lon_range * 111000 * np.cos(np.radians(self.df['latitude'].mean()))
        
        print(f"  Area covered: ~{lat_meters:.0f}m × {lon_meters:.0f}m")
    
    def create_interactive_map(self, output_path: str = "argus_track_map.html"):
        """Create interactive map with Folium"""
        if not FOLIUM_AVAILABLE:
            print("❌ Folium not available. Cannot create interactive map.")
            return None
        
        if self.df.empty:
            print("❌ No data to map")
            return None
        
        # Calculate map center
        center_lat = self.df['latitude'].mean()
        center_lon = self.df['longitude'].mean()
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='OpenStreetMap'
        )
        
        # Add satellite view option
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Color mapping for different classes
        class_colors = {
            'Led-150': 'red',
            'Led-240': 'blue',
            'light_post': 'green',
            'street_light': 'orange',
            'pole': 'purple'
        }
        
        # Add markers for each detection
        for idx, row in self.df.iterrows():
            # Determine marker color
            class_name = row.get('class_name', 'unknown')
            color = class_colors.get(class_name, 'gray')
            
            # Create popup content
            popup_content = f"""
            <b>{class_name}</b><br>
            Track ID: {row.get('track_id', 'N/A')}<br>
            Confidence: {row.get('confidence', 0):.3f}<br>
            Reliability: {row.get('reliability', 0):.3f}<br>
            Accuracy: {row.get('accuracy_meters', 0):.1f}m<br>
            Distance: {row.get('estimated_distance_m', 0):.1f}m<br>
            Frames: {row.get('first_frame', 0)} - {row.get('last_frame', 0)}<br>
            Coordinates: {row['latitude']:.6f}, {row['longitude']:.6f}
            """
            
            # Size marker based on confidence
            confidence = row.get('confidence', 0.5)
            radius = 5 + confidence * 10  # 5-15 pixel radius
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=folium.Popup(popup_content, max_width=300),
                color='black',
                weight=2,
                fillColor=color,
                fillOpacity=0.7,
                tooltip=f"{class_name} (ID: {row.get('track_id', '?')})"
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 10px">
        <p><b>Argus Track Results</b></p>
        '''
        
        for class_name, color in class_colors.items():
            if class_name in self.df['class_name'].values:
                count = (self.df['class_name'] == class_name).sum()
                legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {class_name} ({count})</p>'
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_path)
        print(f"🗺️  Interactive map saved to: {output_path}")
        
        return m
    
    def create_static_plots(self, output_dir: str = "plots"):
        """Create static plots and analysis"""
        if self.df.empty:
            print("❌ No data to plot")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set up the plotting grid
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Map view
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(
            self.df['longitude'], self.df['latitude'],
            c=self.df.get('confidence', 0.5),
            s=self.df.get('reliability', 0.5) * 100,
            alpha=0.7,
            cmap='viridis'
        )
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Object Locations\n(Color: Confidence, Size: Reliability)')
        plt.colorbar(scatter, ax=ax1, label='Confidence')
        
        # Plot 2: Class distribution
        if 'class_name' in self.df.columns:
            ax2 = plt.subplot(2, 3, 2)
            class_counts = self.df['class_name'].value_counts()
            bars = ax2.bar(class_counts.index, class_counts.values)
            ax2.set_title('Detection Class Distribution')
            ax2.set_ylabel('Count')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Plot 3: Confidence distribution
        if 'confidence' in self.df.columns:
            ax3 = plt.subplot(2, 3, 3)
            ax3.hist(self.df['confidence'], bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(self.df['confidence'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.df["confidence"].mean():.3f}')
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('Count')
            ax3.set_title('Confidence Distribution')
            ax3.legend()
        
        # Plot 4: Reliability vs Accuracy
        if 'reliability' in self.df.columns and 'accuracy_meters' in self.df.columns:
            ax4 = plt.subplot(2, 3, 4)
            ax4.scatter(self.df['reliability'], self.df['accuracy_meters'], alpha=0.7)
            ax4.set_xlabel('Reliability')
            ax4.set_ylabel('Accuracy (meters)')
            ax4.set_title('Reliability vs Accuracy')
            
            # Add trend line
            z = np.polyfit(self.df['reliability'], self.df['accuracy_meters'], 1)
            p = np.poly1d(z)
            ax4.plot(self.df['reliability'], p(self.df['reliability']), "r--", alpha=0.8)
        
        # Plot 5: Distance distribution
        if 'estimated_distance_m' in self.df.columns:
            ax5 = plt.subplot(2, 3, 5)
            ax5.hist(self.df['estimated_distance_m'], bins=20, alpha=0.7, edgecolor='black')
            ax5.axvline(self.df['estimated_distance_m'].mean(), color='red', linestyle='--',
                       label=f'Mean: {self.df["estimated_distance_m"].mean():.1f}m')
            ax5.set_xlabel('Estimated Distance (meters)')
            ax5.set_ylabel('Count')
            ax5.set_title('Distance Distribution')
            ax5.legend()
        
        # Plot 6: Frame duration
        if 'duration_frames' in self.df.columns:
            ax6 = plt.subplot(2, 3, 6)
            ax6.hist(self.df['duration_frames'], bins=20, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Duration (frames)')
            ax6.set_ylabel('Count')
            ax6.set_title('Track Duration Distribution')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = output_dir / "argus_track_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Static plots saved to: {plot_path}")
        
        plt.show()
    
    def export_data(self, output_dir: str = "exports"):
        """Export data in various formats"""
        if self.df.empty:
            print("❌ No data to export")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Export CSV
        csv_path = output_dir / "argus_track_detections.csv"
        self.df.to_csv(csv_path, index=False)
        print(f"📄 CSV exported to: {csv_path}")
        
        # Export detailed report
        report_path = output_dir / "argus_track_report.txt"
        with open(report_path, 'w') as f:
            f.write("ARGUS TRACK DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Source: {self.geojson_path}\n\n")
            
            # Summary stats
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Objects Detected: {len(self.df)}\n")
            
            if 'class_name' in self.df.columns:
                f.write("\nClass Distribution:\n")
                for class_name, count in self.df['class_name'].value_counts().items():
                    f.write(f"  {class_name}: {count}\n")
            
            if 'confidence' in self.df.columns:
                f.write(f"\nConfidence: {self.df['confidence'].mean():.3f} ± {self.df['confidence'].std():.3f}\n")
            
            if 'accuracy_meters' in self.df.columns:
                f.write(f"Accuracy: {self.df['accuracy_meters'].mean():.1f} ± {self.df['accuracy_meters'].std():.1f} meters\n")
            
            # Individual detections
            f.write(f"\nDETAILED DETECTIONS\n")
            f.write("-" * 30 + "\n")
            
            for idx, row in self.df.iterrows():
                f.write(f"\nTrack {row.get('track_id', idx)}:\n")
                f.write(f"  Class: {row.get('class_name', 'Unknown')}\n")
                f.write(f"  Location: {row['latitude']:.6f}, {row['longitude']:.6f}\n")
                f.write(f"  Confidence: {row.get('confidence', 0):.3f}\n")
                f.write(f"  Reliability: {row.get('reliability', 0):.3f}\n")
                f.write(f"  Accuracy: {row.get('accuracy_meters', 0):.1f}m\n")
                f.write(f"  Distance: {row.get('estimated_distance_m', 0):.1f}m\n")
        
        print(f"📋 Report exported to: {report_path}")
        
        # Export KML for Google Earth (if geopandas available)
        if GEOPANDAS_AVAILABLE:
            try:
                import fiona
                gdf = gpd.GeoDataFrame(
                    self.df,
                    geometry=gpd.points_from_xy(self.df.longitude, self.df.latitude),
                    crs='EPSG:4326'
                )
                
                kml_path = output_dir / "argus_track_detections.kml"
                gdf.to_file(kml_path, driver='KML')
                print(f"🌍 KML exported to: {kml_path}")
                
            except ImportError:
                print("⚠️  KML export requires fiona: pip install fiona")

def main():
    parser = argparse.ArgumentParser(description="Visualize Argus Track GeoJSON results")
    parser.add_argument('geojson_file', help='Path to GeoJSON file')
    parser.add_argument('--output-dir', default='visualization_output', 
                       help='Output directory for visualizations')
    parser.add_argument('--map-only', action='store_true', 
                       help='Only create interactive map')
    parser.add_argument('--plots-only', action='store_true', 
                       help='Only create static plots')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive map creation')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip static plots creation')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize visualizer
        viz = GeoJSONVisualizer(args.geojson_file)
        
        # Print summary
        viz.print_summary()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create visualizations based on arguments
        if args.map_only:
            if FOLIUM_AVAILABLE:
                viz.create_interactive_map(str(output_dir / "interactive_map.html"))
            else:
                print("❌ Cannot create map: Folium not available")
        
        elif args.plots_only:
            viz.create_static_plots(str(output_dir / "plots"))
        
        else:
            # Create all visualizations
            if not args.no_interactive and FOLIUM_AVAILABLE:
                viz.create_interactive_map(str(output_dir / "interactive_map.html"))
            
            if not args.no_plots:
                viz.create_static_plots(str(output_dir / "plots"))
            
            # Export data
            viz.export_data(str(output_dir / "exports"))
        
        print(f"\n✅ Visualization complete! Check output in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())