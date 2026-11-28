"""
Verification Smoke Script for Mode A vs Mode B Parity & Validation Guards (v3.1.0-parity).

Tests:
1. Canonical `price_momentum_7` (%) parity between historical dataset features and inference helper.
2. Mode A Validated Forecasting on Tomato @ Pune specialist & Onion @ Lasalgaon unified.
3. Rejection of unsupported commodity (`Potato`) and unsupported pair (`Tomato @ Mumbai`) in Mode A.
4. Explicit Mode B Experimental Manual Exploration disclaimer & non-leaking past-lag stats.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import compute_price_momentum_7_pct
from src.inference import (
    CommodityPricePredictor,
    UnifiedCommodityPredictor,
    UnsupportedCommodityError,
    UnsupportedMarketPairError,
)
from src.live_data import get_historical_latest_features


def run_smoke_verification():
    print("=" * 80)
    print("INFERENCE & FEATURE PARITY VERIFICATION TEST (v3.1.0-parity)")
    print("=" * 80)

    spec_meta_path = PROJECT_ROOT / "models" / "model_metadata_tomato.json"
    with open(spec_meta_path, "r", encoding="utf-8") as f:
        spec_meta = json.load(f)
    print(f"Specialist Metadata : {spec_meta.get('model_name')} | Schema={spec_meta.get('feature_schema_version')}")

    # 1. Exact Training-Serving Parity on Real Historical Sequence
    hist_tomato = get_historical_latest_features("Tomato", "Pune")
    row = hist_tomato["features"]
    expected_mom = compute_price_momentum_7_pct(row["modal_price_rs_per_qtl"], row["price_lag_7"])
    assert abs(float(row["price_momentum_7"]) - expected_mom) < 1e-6, "Momentum parity mismatch!"
    print(f"Parity Check PASS   : stored price_momentum_7={row['price_momentum_7']:.4f}% == helper={expected_mom:.4f}%")

    tomato_predictor = CommodityPricePredictor(models_dir=PROJECT_ROOT / "models", commodity="Tomato")
    res_spec = tomato_predictor.predict_validated(row)
    assert res_spec["is_validated_forecast"] is True
    print(f"[Specialist Mode A] Tomato @ Pune -> {res_spec['predicted_next_session_rs_per_qtl']} Rs/Qtl")

    # 2. Unified Mode A + Classifier
    uni = UnifiedCommodityPredictor(models_dir=PROJECT_ROOT / "models")
    hist_onion = get_historical_latest_features("Onion", "Lasalgaon")
    res_uni = uni.predict_validated(hist_onion["features"])
    ew = res_uni["early_warning_classifier"]
    print(
        f"[Unified Mode A]    Onion @ Lasalgaon ({hist_onion['observation_date']}) -> "
        f"{res_uni['predicted_next_session_rs_per_qtl']} Rs/Qtl | Regime={ew['predicted_regime']}"
    )

    # 3. Mode B Experimental Manual Check
    res_man = uni.predict_experimental_manual(
        commodity="Tomato",
        market="Pune",
        modal_price=1400.0,
        price_lag_1=1350.0,
        price_lag_3=1280.0,
        price_lag_7=1200.0,
        arrival_tonnes=120.0,
        arrival_lag_1=115.0,
        rolling_arrival_mean_7=125.0,
    )
    assert res_man["is_validated_forecast"] is False
    assert res_man["experimental_disclaimer"] is not None
    print(f"[Mode B Manual]     Forecast={res_man['predicted_next_session_rs_per_qtl']} | Validated={res_man['is_validated_forecast']}")

    # 4. Guards
    try:
        uni.predict(commodity="Potato", market="Pune", modal_price=1200.0, allow_manual_approx=True)
        raise AssertionError("Potato should fail!")
    except UnsupportedCommodityError:
        print("Guard PASS          : Potato blocked cleanly.")

    try:
        bad_pair = dict(hist_tomato["features"])
        bad_pair["market_name"] = "Mumbai"
        uni.predict_validated(bad_pair)
        raise AssertionError("(Tomato, Mumbai) should fail in Mode A!")
    except UnsupportedMarketPairError:
        print("Guard PASS          : Unsupported pair (Tomato @ Mumbai) blocked in Mode A.")

    print("\nALL PHASE 2 VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    run_smoke_verification()
