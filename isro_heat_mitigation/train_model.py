import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

def train_xgboost():
    print("Loading fused ML dataset...")
    try:
        df = pd.read_csv("ml_training_data.csv")
    except FileNotFoundError:
        print("Error: ml_training_data.csv not found. Did you run fuse_data.py first?")
        return

    print(f"Dataset loaded with {len(df)} rows.")

    # Drop any rows with missing data to prevent math errors
    df = df.dropna()

    # Define our mathematical features (X) and what we want to predict (y)
    features = ['NDVI', 'Albedo', 'is_built_up']
    target = 'LST_Celsius'

    X = df[features]
    y = df[target]

    print("Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Initializing XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    print("Training the model... (This might take a few seconds)")
    model.fit(X_train, y_train)

    print("Evaluating model performance on unseen test data...")
    predictions = model.predict(X_test)
    
    # Calculate how far off our predictions are on average using the new function
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- Model Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")
    print(f"R-squared (R2) Score: {r2:.2f}")
    
    # Save the trained model to disk so our Streamlit dashboard can use it instantly
    model_filename = "heat_stress_xgboost_model.joblib"
    joblib.dump(model, model_filename)
    print(f"\nSuccess! AI Engine saved to disk as '{model_filename}'")

if __name__ == "__main__":
    train_xgboost()