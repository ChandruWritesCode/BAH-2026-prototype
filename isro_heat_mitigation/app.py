import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Bharatiya Antariksh Hackathon 2026", layout="wide")
st.title("🛰️ Urban Heat Mitigation & AI Scenario Simulator")
st.markdown("Team CodeCosmos | **Target:** Bengaluru Core (5km Radius)")

# --- Load Assets ---
# We use Streamlit caching so the data only loads once, keeping the app fast
@st.cache_data
def load_data():
    # Update this line to include the folder path
    return pd.read_csv("isro_heat_mitigation/ml_training_data.csv")

@st.cache_resource
def load_model():
    # Update this path to include the folder name
    return joblib.load("isro_heat_mitigation/heat_stress_xgboost_model.joblib")

df_baseline = load_data()
model = load_model()

# --- Sidebar Controls (Interventions) ---
st.sidebar.header("Cooling Interventions")
st.sidebar.markdown("Adjust parameters to simulate urban cooling strategies.")

# Sliders map directly to the math we discussed
tree_expansion = st.sidebar.slider("Tree Cover Expansion (NDVI Boost)", min_value=0.0, max_value=0.5, value=0.0, step=0.05)
cool_roof_albedo = st.sidebar.slider("Apply Cool Roofs (Target Albedo)", min_value=0.15, max_value=0.80, value=0.15, step=0.05)

# --- Simulation Engine ---
# Create a copy of the data to manipulate
df_simulated = df_baseline.copy()

# Apply Interventions to built-up areas
built_up_mask = df_simulated['is_built_up'] == 1

# 1. Increase NDVI (Green Spaces)
df_simulated.loc[built_up_mask, 'NDVI'] = np.clip(df_simulated.loc[built_up_mask, 'NDVI'] + tree_expansion, -1.0, 1.0)

# 2. Increase Albedo (Cool Roofs)
# Only upgrade the albedo if the slider is higher than the current building albedo
df_simulated.loc[built_up_mask, 'Albedo'] = np.maximum(df_simulated.loc[built_up_mask, 'Albedo'], cool_roof_albedo)

# --- Run AI Predictions ---
features = ['NDVI', 'Albedo', 'is_built_up']

# Predict baseline and simulated temperatures
baseline_temp = model.predict(df_baseline[features])
simulated_temp = model.predict(df_simulated[features])

df_simulated['Baseline_LST'] = baseline_temp
df_simulated['Simulated_LST'] = simulated_temp
df_simulated['Temp_Reduction'] = df_simulated['Baseline_LST'] - df_simulated['Simulated_LST']

# --- Dashboard Layout ---
col1, col2, col3 = st.columns(3)

avg_baseline = df_simulated['Baseline_LST'].mean()
avg_simulated = df_simulated['Simulated_LST'].mean()
max_reduction = df_simulated['Temp_Reduction'].max()

col1.metric("Current Avg Temp", f"{avg_baseline:.2f} °C")
col2.metric("Simulated Avg Temp", f"{avg_simulated:.2f} °C", f"{-abs(avg_baseline - avg_simulated):.2f} °C")
col3.metric("Max Local Temp Drop", f"{max_reduction:.2f} °C", "Hotspot Relief")

st.divider()

# --- Heat Map Visualization ---
st.subheader("Interactive Heat Stress Map")
st.markdown("Displaying a sample of the urban grid. Red indicates severe heat hotspots.")

# Rendering 88k points will crash the browser, so we randomly sample 8,000 points for the visual map
df_map_sample = df_simulated.sample(n=8000, random_state=42)

# Initialize Folium Map centered on Bengaluru
m = folium.Map(location=[12.9716, 77.5946], zoom_start=13, tiles="CartoDB dark_matter")

# Create HeatMap data array: [lat, lon, weight (temperature)]
heat_data = [[row['latitude'], row['longitude'], row['Simulated_LST']] for index, row in df_map_sample.iterrows()]

# Add HeatMap layer
HeatMap(heat_data, radius=15, blur=10, max_zoom=1, gradient={0.4: 'blue', 0.65: 'yellow', 1: 'red'}).add_to(m)

# Render map in Streamlit
st_folium(m, width=1200, height=500)