"""
Verification Smoke Script for Leak-Free Specialist & Unified Inference (v3.0.0-leakfree).

Tests:
1. Specialist CommodityPricePredictor (16 features with current modal_price_rs_per_qtl).
2. UnifiedCommodityPredictor on validated commodities & markets + EarlyWarningClassifier.
3. Explicit rejection (UnsupportedCommodityError) on unseen commodity ("Potato").
4. Retail ML unavailability notice and observed DCA reference.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    CommodityPricePredictor,
    UnifiedCommodityPredictor,
    UnsupportedCommodityError,
)
from src.live_data import get_historical_latest_features


def run_smoke_verification():
    print("=" * 80)
    print("INFERENCE VERIFICATION TEST (v3.0.0-leakfree)")
    print("=" * 80)

    # 1. Specialist Model Test (Tomato @ Pune)
    spec_meta_path = PROJECT_ROOT / "models" / "model_metadata_tomato.json"
    with open(spec_meta_path, "r", encoding="utf-8") as f:
        spec_meta = json.load(f)

    assert "target_definition" in spec_meta or "prediction_horizon" in spec_meta
    print(f"Specialist Metadata Loaded : {spec_meta.get('model_name')} | Horizon: {spec_meta.get('prediction_horizon')}")

    tomato_predictor = CommodityPricePredictor(models_dir=PROJECT_ROOT / "models", commodity="Tomato")
    sample_specialist_features = {
        "modal_price_rs_per_qtl": 1400.0,
        "price_lag_1": 1350.0,
        "price_lag_3": 1280.0,
        "price_lag_7": 1150.0,
        "rolling_mean_7": 1275.0,
        "rolling_std_7": 82.4,
        "rolling_min_7": 1150.0,
        "rolling_max_7": 1400.0,
        "price_momentum_7": ((1400.0 - 1150.0) / 1150.0) * 100.0,
        "daily_spread": 250.0,
        "arrival_tonnes": 118.0,
        "arrival_lag_1": 125.0,
        "rolling_arrival_mean_7": 122.0,
        "month": 11,
        "day_of_week": 2,
        "is_monsoon_season": 0,
    }
    res_spec = tomato_predictor.predict_from_features(sample_specialist_features)
    print(
        f"[Specialist] Current={res_spec['current_modal_price_rs_per_qtl']} Rs/Qtl -> "
        f"Next Trading Session={res_spec['predicted_next_session_rs_per_qtl']} Rs/Qtl "
        f"(Wholesale Equiv={res_spec['wholesale_equiv_rs_per_kg']} Rs/Kg)"
    )

    # 2. Historical Replay + Unified Predictor + Early Warning Classifier
    uni = UnifiedCommodityPredictor(models_dir=PROJECT_ROOT / "models")
    print(f"\nUnified Supported Commodities : {uni.get_supported_commodities()}")
    print(f"Unified Supported Pairs       : {uni.supported_pairs}")

    hist = get_historical_latest_features("Onion", "Lasalgaon")
    res_uni = uni.predict_from_features(hist["features"])
    ew = res_uni["early_warning_classifier"]
    print(
        f"\n[Unified Replay Onion @ Lasalgaon ({hist['observation_date']})]\n"
        f"  Current Modal Price         : Rs. {res_uni['current_modal_price_rs_per_qtl']}/Qtl\n"
        f"  Predicted Next Session      : Rs. {res_uni['predicted_next_session_rs_per_qtl']}/Qtl "
        f"(Equiv Rs. {res_uni['wholesale_equiv_rs_per_kg']}/Kg)\n"
        f"  Actual Archived Next Session: Rs. {hist['actual_next_price_rs_per_qtl']}/Qtl\n"
        f"  Early-Warning Regime        : {ew['predicted_regime']} "
        f"(Probs={ew['class_probabilities']}, Reliable={ew['is_reliable_early_warning']})\n"
        f"  Retail ML Status            : {res_uni['retail_estimation_status']}"
    )

    # 3. Unsupported Commodity Guard Verification
    print("\n[Unsupported Commodity Guard Check: Potato]")
    try:
        uni.predict(commodity="Potato", market="Pune", modal_price=1200.0, allow_manual_approx=True)
        raise AssertionError("Expected UnsupportedCommodityError for Potato was NOT raised!")
    except UnsupportedCommodityError as exc:
        print(f"  Properly blocked Potato with UnsupportedCommodityError:\n  {exc}")

    print("\nALL INFERENCE VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    run_smoke_verification()
