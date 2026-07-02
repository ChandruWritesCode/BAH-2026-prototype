import ee
import argparse

def initialize_gee(project_id):
    # Initialize the library using your Google Cloud Project ID
    ee.Initialize(project=project_id)

def get_satellite_data(lat, lon, buffer_meters):
    roi = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)
    
    # 1. Fetch Landsat 8 for Land Surface Temperature (LST)
    # Using Collection 2 Tier 1 L2 (Surface Reflectance & Surface Temperature)
    landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(roi) \
        .filterDate('2026-03-23', '2026-03-29') \
        .filterMetadata('CLOUD_COVER', 'less_than', 10) \
        .median()
        
    # Scale factor formula to convert Landsat 8 LST band (ST_B10) from Kelvin to Celsius
    lst_celsius = landsat.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
    
    # 2. Fetch Sentinel-2 for NDVI (Green Spaces)
    sentinel = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate('2026-03-23', '2026-03-29') \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10) \
        .median()
        
    # Calculate NDVI using the Near-Infrared (B8) and Red (B4) bands
    ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    print(f"Satellite queries constructed successfully for coordinates ({lat}, {lon}).")
    # Note: In Phase 2, we will export these pixel arrays into a flattened pandas DataFrame.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch GEE Satellite Rasters")
    parser.add_argument("--project", required=True, help="Google Cloud Project ID")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--buffer", type=int, default=5000, help="Buffer radius in meters")
    
    args = parser.parse_args()
    initialize_gee(args.project)
    get_satellite_data(args.lat, args.lon, args.buffer)