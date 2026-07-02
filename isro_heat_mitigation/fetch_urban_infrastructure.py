import osmnx as ox
import argparse

def get_urban_data(lat, lon, buffer_meters):
    print(f"Fetching building footprints within {buffer_meters}m of ({lat}, {lon})...")
    print("This might take a minute or two depending on building density...")
    
    # Query by point and radius instead of a massive city boundary string
    buildings = ox.features.features_from_point(
        (lat, lon), 
        tags={"building": True}, 
        dist=buffer_meters
    )
    
    print(f"Successfully downloaded {len(buildings)} buildings.")
    
    # Project to the local UTM CRS to accurately calculate metric area
    buildings_proj = buildings.to_crs(buildings.estimate_utm_crs())
    
    # Calculate the square meter area of each building footprint
    buildings_proj['building_area_sqm'] = buildings_proj.geometry.area
    
    # Save the output to a GeoPackage file
    buildings_proj[['geometry', 'building_area_sqm']].to_file("buildings.gpkg", driver="GPKG")
    print("Saved building geometries to buildings.gpkg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Urban Vector Data from OSM")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--buffer", type=int, default=5000, help="Buffer radius in meters")
    
    args = parser.parse_args()
    get_urban_data(args.lat, args.lon, args.buffer)