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
    def __init__(self, models_dir: Path = None, commodity: str = None):
        if models_dir is None:
            # Look in models or ../models
            p1 = Path("models")
            p2 = Path("../models")
            self.models_dir = p1 if p1.exists() else p2
        else:
            self.models_dir = Path(models_dir)

        # Allow selecting commodity-specific model if available
        if commodity:
            safe_comm = commodity.lower().replace(' ', '_')
            comm_model = self.models_dir / f"price_prediction_model_{safe_comm}.joblib"
            comm_meta = self.models_dir / f"model_metadata_{safe_comm}.json"
            if comm_model.exists() and comm_meta.exists():
                self.model_path = comm_model
                self.metadata_path = comm_meta
            else:
                self.model_path = self.models_dir / "price_prediction_model.joblib"
                self.metadata_path = self.models_dir / "model_metadata.json"
        else:
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


class UnifiedCommodityPredictor:
    """
    Inference Engine for the Unified Multi-Commodity Food Price Forecasting Model.
    Accepts ANY food commodity and predicts wholesale (Rs/Quintal) and retail (Rs/Kg) prices.
    Uses ColumnTransformer with OneHotEncoder (handle_unknown='ignore') and StandardScaler.
    """
    DEFAULT_CATEGORIES = {
        'tomato': ('Vegetables', 1),
        'onion': ('Vegetables', 1),
        'potato': ('Vegetables', 1),
        'brinjal': ('Vegetables', 1),
        'cabbage': ('Vegetables', 1),
        'cauliflower': ('Vegetables', 1),
        'sugar': ('Sugars & Sweets', 0),
        'gur': ('Sugars & Sweets', 0),
        'jaggery': ('Sugars & Sweets', 0),
        'moong dal': ('Pulses', 0),
        'green gram dal (moong dal)': ('Pulses', 0),
        'tur dal': ('Pulses', 0),
        'arhar dal': ('Pulses', 0),
        'urad dal': ('Pulses', 0),
        'gram dal': ('Pulses', 0),
        'wheat': ('Cereals', 0),
        'rice': ('Cereals', 0),
        'apple': ('Fruits', 1),
        'banana': ('Fruits', 1)
    }

    def __init__(self, models_dir: Path = None):
        if models_dir is None:
            p1 = Path("models")
            p2 = Path("../models")
            self.models_dir = p1 if p1.exists() else p2
        else:
            self.models_dir = Path(models_dir)

        self.model_path = self.models_dir / "unified_food_price_model.joblib"
        self.metadata_path = self.models_dir / "unified_model_metadata.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Unified model not found at {self.model_path}. Train it first using scripts/train_unified_model.py")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Unified model metadata not found at {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = joblib.load(self.model_path)
        self.feature_names = self.metadata["all_features"]

    def infer_category_and_perishability(self, commodity: str):
        """Infers commodity category and perishability if not explicitly provided."""
        comm_clean = commodity.lower().strip()
        if comm_clean in self.DEFAULT_CATEGORIES:
            return self.DEFAULT_CATEGORIES[comm_clean]
        
        # Heuristics for unknown commodities
        if any(w in comm_clean for w in ['dal', 'gram', 'pulse', 'lentil', 'bean']):
            return ('Pulses', 0)
        elif any(w in comm_clean for w in ['sweet', 'sugar', 'jaggery', 'honey']):
            return ('Sugars & Sweets', 0)
        elif any(w in comm_clean for w in ['rice', 'wheat', 'atta', 'flour', 'cereal', 'maize', 'barley', 'bajra', 'jowar']):
            return ('Cereals', 0)
        elif any(w in comm_clean for w in ['oil', 'mustard', 'sunflower', 'groundnut']):
            return ('Edible Oils', 0)
        elif any(w in comm_clean for w in ['fruit', 'mango', 'banana', 'apple', 'orange', 'grape']):
            return ('Fruits', 1)
        else:
            # Default to Vegetables (perishable)
            return ('Vegetables', 1)

    def predict(
        self,
        commodity: str,
        current_price: float,
        arrivals: float = 100.0,
        month: int = None,
        day_of_week: int = None,
        category: str = None,
        is_perishable: int = None,
        detailed_features: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Executes prediction for ANY food commodity.
        Accepts either explicit detailed_features or constructs them from price & market context.
        """
        now = datetime.now()
        month = month if month is not None else now.month
        day_of_week = day_of_week if day_of_week is not None else now.weekday()
        is_monsoon = 1 if month in [6, 7, 8, 9] else 0

        inferred_cat, inferred_perish = self.infer_category_and_perishability(commodity)
        cat = category if category is not None else inferred_cat
        perish = is_perishable if is_perishable is not None else inferred_perish

        if detailed_features:
            feature_dict = detailed_features.copy()
            feature_dict['commodity_name'] = commodity
            feature_dict['category'] = cat
            feature_dict['is_perishable'] = perish
            if 'price_lag_1' not in feature_dict:
                feature_dict['price_lag_1'] = current_price
        else:
            # Construct standard realistic feature vector around current_price
            feature_dict = {
                'commodity_name': commodity,
                'category': cat,
                'is_perishable': perish,
                'price_lag_1': float(current_price),
                'price_lag_3': float(current_price * 0.99),
                'price_lag_7': float(current_price * 0.98),
                'rolling_mean_7': float(current_price * 0.99),
                'rolling_std_7': float(current_price * 0.05),
                'rolling_min_7': float(current_price * 0.94),
                'rolling_max_7': float(current_price * 1.04),
                'price_momentum_7': float(current_price * 0.02),
                'daily_spread': float(current_price * 0.15),
                'arrival_tonnes': float(arrivals),
                'arrival_lag_1': float(arrivals * 1.02),
                'rolling_arrival_mean_7': float(arrivals),
                'month': int(month),
                'day_of_week': int(day_of_week),
                'is_monsoon_season': int(is_monsoon)
            }

        input_df = pd.DataFrame([feature_dict])[self.feature_names]
        pred_val = float(self.model.predict(input_df)[0])

        price_diff = pred_val - current_price
        pct_change = (price_diff / current_price) * 100 if current_price > 0 else 0.0

        return {
            "status": "success",
            "model_type": "Unified Cross-Commodity Regression",
            "commodity": commodity,
            "category": cat,
            "is_perishable": bool(perish),
            "observed_price_rs_per_qtl": round(float(current_price), 2),
            "predicted_next_price_rs_per_qtl": round(pred_val, 2),
            "observed_price_rs_per_kg": round(float(current_price) / 100.0, 2),
            "predicted_next_price_rs_per_kg": round(pred_val / 100.0, 2),
            "predicted_change_rs_per_qtl": round(price_diff, 2),
            "predicted_change_percentage": round(pct_change, 2),
            "prediction_horizon": self.metadata.get("prediction_horizon", "t+1 trading session"),
            "model_version": self.metadata.get("dataset_version", "v2.0.0-unified"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """Executes batch inference on a multi-commodity DataFrame."""
        input_features = df[self.feature_names]
        return list(self.model.predict(input_features))

    def get_metadata(self) -> Dict[str, Any]:
        """Returns loaded model metadata."""
        return self.metadata


if __name__ == "__main__":
    print("Testing CommodityPricePredictor (Dedicated Crop Model)...")
    p1 = CommodityPricePredictor()
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
    res1 = p1.predict(test_sample)
    print("Dedicated Model Result:", res1)

    print("\nTesting UnifiedCommodityPredictor (Universal Model for Any Food)...")
    p_uni = UnifiedCommodityPredictor()
    for comm, price in [("Tomato", 1400.0), ("Onion", 1250.0), ("Sugar", 3600.0), ("Moong Dal", 7400.0), ("Potato", 1800.0)]:
        res = p_uni.predict(commodity=comm, current_price=price)
        print(f"  {comm:<10} ({res['category']:<15}): Current=Rs.{res['observed_price_rs_per_kg']:.2f}/kg -> Predicted=Rs.{res['predicted_next_price_rs_per_kg']:.2f}/kg ({res['predicted_change_percentage']:+.2f}%)")
