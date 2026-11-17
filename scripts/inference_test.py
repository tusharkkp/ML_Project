"""
Standalone Inference Test Script
Verifies that models/price_prediction_model.joblib and models/model_metadata.json
can be loaded independently and used for real-time inference without retraining.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = Path("models")
model_path = MODELS_DIR / "price_prediction_model.joblib"
metadata_path = MODELS_DIR / "model_metadata.json"

assert model_path.exists(), f"Model file not found at {model_path}"
assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"

# 1. Load Metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

print("=== LOADED MODEL METADATA ===")
print(f"Model Name        : {meta['model_name']}")
print(f"Commodity & Market: {meta['commodity']} ({meta['market']}, {meta['state']})")
print(f"Target Variable   : {meta['target_variable']} ({meta['target_definition']})")
print(f"Features Expected : {len(meta['feature_names'])} features")
print(f"Test RMSE         : Rs. {meta['test_evaluation_metrics']['RMSE']}")
print(f"Test R2           : {meta['test_evaluation_metrics']['R2']}")

# 2. Load Model Pipeline
model = joblib.load(model_path)
print("\nModel pipeline loaded successfully from joblib!")

# 3. Simulate an Unseen Input Vector (e.g. Current Market Observation)
# Scenario: Tomato at Pune APMC Mandi on a Thursday in July
sample_input = {
    'price_lag_1': 1400.0,            # Today's modal price: Rs. 14.00/kg
    'price_lag_3': 1250.0,            # 3 sessions ago: Rs. 12.50/kg
    'price_lag_7': 1100.0,            # 7 sessions ago: Rs. 11.00/kg
    'rolling_mean_7': 1280.0,         # 7-session moving average
    'rolling_std_7': 120.0,           # Local standard deviation
    'rolling_min_7': 1100.0,          # Weekly low
    'rolling_max_7': 1400.0,          # Weekly high
    'price_momentum_7': 300.0,        # Upward momentum (+Rs. 300/Qtl)
    'daily_spread': 400.0,            # Max - Min intra-day auction spread
    'arrival_tonnes': 145.0,          # Today's arrival volume
    'arrival_lag_1': 160.0,           # Yesterday's arrival volume
    'rolling_arrival_mean_7': 155.0,  # 7-day average arrival
    'month': 7,                       # July (Peak monsoon month)
    'day_of_week': 3,                 # Thursday
    'is_monsoon_season': 1            # Monsoon indicator: True
}

# Convert to DataFrame ensuring exact feature order from metadata
feature_order = meta['feature_names']
input_df = pd.DataFrame([sample_input])[feature_order]

# Predict
predicted_price = float(model.predict(input_df)[0])

print("\n=== INFERENCE TEST EXECUTION ===")
print("Input Market Observation Vector:")
for k, v in sample_input.items():
    print(f"  {k:<24}: {v}")

print("-" * 50)
print(f"Observed Price Today (t)         : Rs. {sample_input['price_lag_1']:.2f} / Quintal (Rs. {sample_input['price_lag_1']/100:.2f} / Kg)")
print(f"PREDICTED NEXT-SESSION PRICE (t+1): Rs. {predicted_price:.2f} / Quintal (Rs. {predicted_price/100:.2f} / Kg)")
print(f"Predicted Daily Change            : Rs. {predicted_price - sample_input['price_lag_1']:+.2f} / Quintal")
print("-" * 50)
print("INFERENCE TEST PASSED: Model is fully standalone and deployment-ready!")
