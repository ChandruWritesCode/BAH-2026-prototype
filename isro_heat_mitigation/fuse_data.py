import geopandas as gpd
import pandas as pd
import json

def fuse_datasets():
    print("Loading satellite pixel data...")
    df = pd.read_csv("bengaluru_heat_dataset.csv")
    
    print("Extracting coordinates from Earth Engine format...")
    # GEE stores the geometry as a JSON string in a column named '.geo'
    # We will crack open the JSON to extract the exact longitude and latitude
    def extract_coords(geo_str):
        try:
            geom = json.loads(geo_str)
            return geom['coordinates'][0], geom['coordinates'][1]
        except:
            return None, None
            
    # Apply the extraction and create the missing columns
    df[['longitude', 'latitude']] = df.apply(
        lambda row: pd.Series(extract_coords(row['.geo'])), axis=1
    )
    
    # Drop any rows where coordinate extraction failed (just in case)
    df = df.dropna(subset=['longitude', 'latitude'])

    print("Converting to spatial GeoDataFrame...")
    gdf_points = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude), 
        crs="EPSG:4326" # Standard GPS coordinate system
    )
    
    print("Loading building footprints (this takes a few seconds)...")
    buildings = gpd.read_file("buildings.gpkg")
    
    print("Aligning Coordinate Reference Systems (CRS)...")
    # Reproject satellite points to match the metric projection of the buildings
    gdf_points = gdf_points.to_crs(buildings.crs)
    
    print("Performing Spatial Join to find built-up pixels...")
    joined = gpd.sjoin(gdf_points, buildings, how="left", predicate="intersects")
    
    # Create a binary feature: 1 if the pixel is on a building, 0 if it is not
    joined['is_built_up'] = joined['index_right'].notnull().astype(int)
    
    print("Cleaning up final dataset...")
    final_df = joined.drop_duplicates(subset=['longitude', 'latitude'])
    
    # Drop the geometric columns and spatial indexes, keeping only ML features
    cols_to_drop = ['geometry', 'index_right', 'building_area_sqm', '.geo']
    final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns])
    
    # Save the final dataset ready for XGBoost
    output_filename = "ml_training_data.csv"
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nSuccess! Fused dataset saved as '{output_filename}'.")
    print(final_df[['longitude', 'latitude', 'LST_Celsius', 'NDVI', 'Albedo', 'is_built_up']].head())

if __name__ == "__main__":
    fuse_datasets()