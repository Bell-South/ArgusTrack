import pandas as pd
import folium

# Load CSV
df = pd.read_csv("GX018691.csv")

# Get unique (latitude, longitude) pairs
unique_points = df[['latitude', 'longitude']].drop_duplicates()

# Center map on the average point
center_lat = unique_points['latitude'].mean()
center_long = unique_points['longitude'].mean()

# Create folium map
m = folium.Map(location=[center_lat, center_long], zoom_start=14)

# Add each point to map
for _, row in unique_points.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7
    ).add_to(m)

# Save map to HTML file
m.save("map.html")

print("Map saved as map.html")
