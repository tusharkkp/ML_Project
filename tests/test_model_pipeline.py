"""
Phase 11B: Mode A vs Mode B Inference, Guards, Schema Validation & Live Completeness Tests.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import FEATURE_SCHEMA_VERSION, compute_price_momentum_7_pct
from src.inference import (
    CommodityPricePredictor,
    FeatureSchemaMismatchError,
    UnifiedCommodityPredictor,
    UnsupportedCommodityError,
    UnsupportedMarketPairError,
)
from src.live_data import get_historical_latest_features


def test_mode_a_validated_forecast_and_momentum_parity():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    hist = get_historical_latest_features("Tomato", "Pune")
    feats = hist["features"]

    expected_mom = compute_price_momentum_7_pct(feats["modal_price_rs_per_qtl"], feats["price_lag_7"])
    assert abs(float(feats["price_momentum_7"]) - expected_mom) < 1e-6

    res = predictor.predict_validated(feats)
    assert res["is_validated_forecast"] is True
    assert "Mode A" in res["inference_mode"]
    assert res["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert res["experimental_disclaimer"] is None
    assert res["wholesale_equiv_rs_per_kg"] == round(res["predicted_next_session_rs_per_qtl"] / 100.0, 2)
    assert res["retail_estimation_supported"] is False
    assert res["early_warning_classifier"]["available"] is True


def test_mode_b_experimental_manual_excludes_t_from_rolling_and_discloses():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    res = predictor.predict_experimental_manual(
        commodity="Tomato",
        market="Pune",
        modal_price=2000.0,
        price_lag_1=1400.0,
        price_lag_3=1300.0,
        price_lag_7=1000.0,
        arrival_tonnes=150.0,
        arrival_lag_1=140.0,
        rolling_arrival_mean_7=145.0,
    )
    assert res["is_validated_forecast"] is False
    assert "Mode B" in res["inference_mode"]
    assert "EXPERIMENTAL EXPLORATION ONLY" in (res["experimental_disclaimer"] or "")
    assert any("excludes t" in a for a in res["approximations_used"])


def test_unsupported_commodity_and_unsupported_pair_rejection():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    hist = get_historical_latest_features("Tomato", "Pune")

    # 1. Unsupported commodity
    with pytest.raises(UnsupportedCommodityError):
        predictor.predict(commodity="Potato", market="Pune", modal_price=1200.0, allow_manual_approx=True)

    # 2. Supported commodity at untrained market pair fails Mode A
    bad_pair = dict(hist["features"])
    bad_pair["market_name"] = "Mumbai"
    with pytest.raises(UnsupportedMarketPairError):
        predictor.predict_validated(bad_pair)

    # 3. Missing required arrivals in Mode A fails
    no_arr = dict(hist["features"])
    no_arr["arrival_tonnes"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predictor.predict_validated(no_arr)


def test_schema_mismatch_and_invalid_numeric_guards():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    hist = get_historical_latest_features("Onion", "Lasalgaon")
    base = dict(hist["features"])

    # Missing column
    bad_missing = dict(base)
    bad_missing.pop("rolling_mean_7")
    with pytest.raises(FeatureSchemaMismatchError, match="Missing required features"):
        predictor.predict_validated(bad_missing)

    # Mismatched schema version
    bad_ver = dict(base)
    bad_ver["feature_schema_version"] = "0.0.1-stale"
    with pytest.raises(FeatureSchemaMismatchError, match="schema version mismatch"):
        predictor.predict_validated(bad_ver)

    # Negative/Non-finite price
    bad_price = dict(base)
    bad_price["modal_price_rs_per_qtl"] = -100.0
    with pytest.raises(ValueError, match="strictly positive"):
        predictor.predict_validated(bad_price)

    bad_inf = dict(base)
    bad_inf["price_momentum_7"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        predictor.predict_validated(bad_inf)


def test_specialist_predictors_and_guards():
    t_pred = CommodityPricePredictor(models_dir=ROOT / "models", commodity="Tomato")
    o_pred = CommodityPricePredictor(models_dir=ROOT / "models", commodity="Onion")

    res_t = t_pred.predict_validated(get_historical_latest_features("Tomato", "Pune")["features"])
    res_o = o_pred.predict_validated(get_historical_latest_features("Onion", "Pune")["features"])
    assert res_t["is_validated_forecast"] is True and res_o["is_validated_forecast"] is True

    with pytest.raises(UnsupportedCommodityError):
        CommodityPricePredictor(models_dir=ROOT / "models", commodity="Wheat")
