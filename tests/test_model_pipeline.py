"""
Unit Tests for Machine Learning Forecasting Pipelines
Course: Third-Year B.Tech AIML Practical Examination (Rubric Component D, E, G)
"""

import pytest
from pathlib import Path
from src.inference import UnifiedCommodityPredictor


def test_unified_predictor_loading():
    models_dir = Path("models")
    if not (models_dir / "unified_food_price_model.joblib").exists():
        pytest.skip("Unified model artifact not found.")

    predictor = UnifiedCommodityPredictor(models_dir=models_dir)
    assert predictor.model is not None
    assert "Moong Dal" in predictor.metadata["supported_commodities"]
    assert "Tomato" in predictor.metadata["supported_commodities"]


def test_unified_predictor_inference():
    models_dir = Path("models")
    if not (models_dir / "unified_food_price_model.joblib").exists():
        pytest.skip("Unified model artifact not found.")

    predictor = UnifiedCommodityPredictor(models_dir=models_dir)
    res = predictor.predict(
        commodity="Tomato",
        current_price=1500.0,
        arrivals=120.0,
        month=8,
        day_of_week=2
    )

    assert "predicted_next_price_rs_per_qtl" in res
    assert "predicted_next_price_rs_per_kg" in res
    assert res["predicted_next_price_rs_per_qtl"] > 0
    assert res["predicted_next_price_rs_per_kg"] > 0
