"""
Inference Engine for Food Commodity Price Prediction
Loads serialized scikit-learn models and metadata without retraining.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib

class CommodityPricePredictor:
    def __init__(self, models_dir: Path = None):
        if models_dir is None:
            # Look in models or ../models
            p1 = Path("models")
            p2 = Path("../models")
            self.models_dir = p1 if p1.exists() else p2
        else:
            self.models_dir = Path(models_dir)

        self.model_path = self.models_dir / "price_prediction_model.joblib"
        self.metadata_path = self.models_dir / "model_metadata.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {self.metadata_path}")

        # Load metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load model pipeline
        self.model = joblib.load(self.model_path)
        self.feature_names = self.metadata["feature_names"]

    def predict(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes single-observation wholesale price prediction.
        Returns detailed prediction dictionary in Rs/Quintal and Rs/Kg.
        """
        # Ensure input features contain all required columns
        missing = [f for f in self.feature_names if f not in feature_dict]
        if missing:
            raise ValueError(f"Missing required input features: {missing}")

        input_df = pd.DataFrame([feature_dict])[self.feature_names]
        pred_val = float(self.model.predict(input_df)[0])

        current_price = float(feature_dict.get("price_lag_1", pred_val))
        price_diff = pred_val - current_price
        pct_change = (price_diff / current_price) * 100 if current_price > 0 else 0.0

        return {
            "status": "success",
            "commodity": self.metadata.get("commodity", "Tomato"),
            "market": self.metadata.get("market", "Pune APMC"),
            "observed_price_rs_per_qtl": round(current_price, 2),
            "predicted_next_price_rs_per_qtl": round(pred_val, 2),
            "observed_price_rs_per_kg": round(current_price / 100.0, 2),
            "predicted_next_price_rs_per_kg": round(pred_val / 100.0, 2),
            "predicted_change_rs_per_qtl": round(price_diff, 2),
            "predicted_change_percentage": round(pct_change, 2),
            "prediction_horizon": self.metadata.get("prediction_horizon", "t+1 session"),
            "model_version": self.metadata.get("dataset_version", "v1.0.0"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """Executes batch inference on a DataFrame."""
        input_features = df[self.feature_names]
        return list(self.model.predict(input_features))

    def get_metadata(self) -> Dict[str, Any]:
        """Returns loaded model metadata."""
        return self.metadata

if __name__ == "__main__":
    predictor = CommodityPricePredictor()
    test_sample = {
        'price_lag_1': 1200.0,
        'price_lag_3': 1100.0,
        'price_lag_7': 1000.0,
        'rolling_mean_7': 1150.0,
        'rolling_std_7': 80.0,
        'rolling_min_7': 1000.0,
        'rolling_max_7': 1200.0,
        'price_momentum_7': 200.0,
        'daily_spread': 300.0,
        'arrival_tonnes': 160.0,
        'arrival_lag_1': 170.0,
        'rolling_arrival_mean_7': 165.0,
        'month': 7,
        'day_of_week': 2,
        'is_monsoon_season': 1
    }
    result = predictor.predict(test_sample)
    print("Inference Test Output:", result)
