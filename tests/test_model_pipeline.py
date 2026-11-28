"""
Tests for Specialist & Unified Inference, Unsupported Commodity Guards, and Early-Warning Classification.
"""

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    CommodityPricePredictor,
    UnifiedCommodityPredictor,
    UnsupportedCommodityError,
)
from src.live_data import get_historical_latest_features


def test_unified_model_supported_commodities_and_forecast():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    assert set(predictor.get_supported_commodities()) == {"Moong Dal", "Onion", "Sugar", "Tomato"}

    hist = get_historical_latest_features("Tomato", "Pune")
    res = predictor.predict_from_features(hist["features"])

    assert res["commodity"] == "Tomato"
    assert res["market"] == "Pune"
    assert res["is_validated_commodity"] is True
    assert res["is_validated_pair"] is True
    assert res["predicted_next_session_rs_per_qtl"] > 0
    assert res["wholesale_equiv_rs_per_kg"] == round(res["predicted_next_session_rs_per_qtl"] / 100.0, 2)
    assert res["retail_estimation_supported"] is False
    assert "next observed" in res["prediction_horizon"].lower() and "trading session" in res["prediction_horizon"].lower()

    ew = res["early_warning_classifier"]
    assert ew["available"] is True
    assert ew["predicted_regime"] in {"Normal", "Surge", "Crash"}
    assert abs(sum(ew["class_probabilities"].values()) - 1.0) < 0.01


def test_unsupported_commodity_rejected_unless_experimental():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    with pytest.raises(UnsupportedCommodityError):
        predictor.predict(commodity="Potato", market="Pune", modal_price=1200.0, allow_manual_approx=True)

    exp_res = predictor.predict(
        commodity="Potato",
        market="Pune",
        modal_price=1200.0,
        allow_experimental=True,
        allow_manual_approx=True,
    )
    assert exp_res["is_validated_commodity"] is False
    assert "EXPERIMENTAL UNVALIDATED COMMODITY" in (exp_res["validation_warning"] or "")


def test_strict_mode_refuses_synthetic_lag_fabrication():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    with pytest.raises(ValueError, match="Refusing to manufacture missing lag/rolling features"):
        predictor.predict(commodity="Tomato", market="Pune", modal_price=1400.0, allow_manual_approx=False)


def test_specialist_predictors_and_missing_crop_guard():
    tomato_p = CommodityPricePredictor(models_dir=ROOT / "models", commodity="Tomato")
    onion_p = CommodityPricePredictor(models_dir=ROOT / "models", commodity="Onion")

    hist_t = get_historical_latest_features("Tomato", "Pune")
    hist_o = get_historical_latest_features("Onion", "Pune")
    res_t = tomato_p.predict_from_features(hist_t["features"])
    res_o = onion_p.predict_from_features(hist_o["features"])

    assert res_t["predicted_next_session_rs_per_qtl"] > 0
    assert res_o["predicted_next_session_rs_per_qtl"] > 0

    with pytest.raises(UnsupportedCommodityError):
        CommodityPricePredictor(models_dir=ROOT / "models", commodity="Potato")
