"""
CLI Verification Runner for Unified Multi-Commodity Predictor (v3.0.0-leakfree).

Usage:
  py -3.12 scripts/test_unified_model.py --quick
  py -3.12 scripts/test_unified_model.py --historical --commodity Onion --market Lasalgaon
  py -3.12 scripts/test_unified_model.py --eval-summary
  py -3.12 scripts/test_unified_model.py --custom --commodity Sugar --market Mumbai --price 3300 --arrivals 150 --month 11
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import UnifiedCommodityPredictor, UnsupportedCommodityError
from src.live_data import get_historical_latest_features


def run_quick_test(predictor: UnifiedCommodityPredictor):
    print("\n--- QUICK UNIFIED MULTI-COMMODITY CHECK + UNSUPPORTED CROP GUARD ---")
    for commodity, market in predictor.supported_pairs:
        hist = get_historical_latest_features(commodity, market)
        res = predictor.predict_from_features(hist["features"], feature_mode="historical_archive_replay")
        ew = res["early_warning_classifier"]
        print(
            f"  {commodity:<10} @ {market:<10} ({hist['observation_date']}) | "
            f"Curr={res['current_modal_price_rs_per_qtl']:>7.1f} -> Pred={res['predicted_next_session_rs_per_qtl']:>7.1f} | "
            f"Actual t+1={hist['actual_next_price_rs_per_qtl']:>7.1f} | "
            f"Regime={ew.get('predicted_regime')} (Reliable={ew.get('is_reliable_early_warning')})"
        )

    print("\nVerifying unsupported commodity rejection (Potato):")
    try:
        predictor.predict(commodity="Potato", market="Pune", modal_price=1200.0, allow_manual_approx=True)
        raise AssertionError("Potato should have been rejected!")
    except UnsupportedCommodityError as exc:
        print(f"  PASS -> Rejected unseen crop cleanly: {str(exc)[:140]}...")


def run_historical_test(predictor: UnifiedCommodityPredictor, commodity: str, market: str):
    print(f"\n--- HISTORICAL ARCHIVE BACKTEST: {commodity} @ {market} ---")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "model_ready_unified_commodities_clean.csv")
    sub = df[(df["commodity_name"] == commodity) & (df["market_name"] == market) & (df["split_tag"] == "Test")].tail(5)
    if sub.empty:
        print(f"No untouched Test rows for {commodity} @ {market}.")
        return
    for _, row in sub.iterrows():
        feats = row.to_dict()
        res = predictor.predict_from_features(feats, feature_mode="historical_test_replay")
        actual = float(row["target_price_t_plus_1"])
        print(
            f"  {row['observation_date']} -> {row.get('target_date_t_plus_1', 't+1')} | "
            f"Curr={row['modal_price_rs_per_qtl']:.1f} | Pred={res['predicted_next_session_rs_per_qtl']:.1f} | "
            f"Actual={actual:.1f} | Err={res['predicted_next_session_rs_per_qtl'] - actual:+.1f}"
        )


def run_eval_summary(predictor: UnifiedCommodityPredictor):
    print("\n--- UNIFIED MODEL BENCHMARK SUMMARY (POOLED, MACRO, PER-COMMODITY & LOCO) ---")
    m = predictor.metadata
    ov = m.get("overall_test_metrics", {})
    bm = m.get("persistence_baseline_test_metrics", {})
    macro = m.get("macro_average_test_metrics", {})

    print(f"Champion (Val-Selected): {m.get('champion_algorithm')} | Params={m.get('champion_hyperparameters')}")
    print(f"Cutoffs                : {m.get('split_cutoffs')}")
    print(f"Pooled Test            : MAE={ov.get('mae_rs_per_qtl')} | RMSE={ov.get('rmse_rs_per_qtl')} (Base={bm.get('RMSE')}) | R2={ov.get('r2_score')} | MAPE={ov.get('mape_pct')}%")
    print(f"Macro-Average Test     : MAE={macro.get('macro_avg_mae')} | RMSE={macro.get('macro_avg_rmse')} (Base={macro.get('baseline_macro_avg_rmse')}) | R2={macro.get('macro_avg_r2')}")

    print("\nPer-Commodity Untouched Test Breakdown:")
    for comm, cm in m.get("per_commodity_metrics", {}).items():
        base_rmse = cm.get("persistence_baseline", {}).get("rmse")
        imp = cm.get("improvement_over_baseline", {}).get("beats_baseline_rmse")
        print(
            f"  {comm:<10} | N={cm.get('sample_count'):>3} | RMSE={cm.get('test_rmse_rs_per_qtl'):>6.2f} "
            f"(Base={base_rmse:>6.2f}, Beats={imp}) | R2={cm.get('test_r2'):.4f} | MAPE={cm.get('test_mape_pct'):>5.2f}%"
        )

    print("\nLeave-One-Commodity-Out (Unseen Commodity Check):")
    for comm, lm in m.get("leave_one_commodity_out_experiment", {}).items():
        print(
            f"  Held-out {comm:<10} | Model RMSE={lm['model_metrics']['RMSE']:>6.2f} "
            f"vs Base RMSE={lm['persistence_baseline']['RMSE']:>6.2f} | Beats={lm['beats_baseline_rmse']}"
        )


def run_custom_test(predictor: UnifiedCommodityPredictor, commodity: str, market: str, price: float, arrivals: float, month: int):
    print(f"\n--- CUSTOM MANUAL SCENARIO: {commodity} @ {market} ---")
    res = predictor.predict(
        commodity=commodity,
        market=market,
        modal_price=price,
        price_lag_1=price * 0.98,
        arrival_tonnes=arrivals,
        month=month,
        allow_manual_approx=True,
    )
    print(f"Validated Pair    : {res['is_validated_pair']} | Warning={res['validation_warning']}")
    print(f"Predicted t+1     : Rs. {res['predicted_next_session_rs_per_qtl']:.2f}/Qtl (Equiv Rs. {res['wholesale_equiv_rs_per_kg']}/Kg)")
    print(f"Alert             : {res['inflation_alert_badge']}")


def main():
    parser = argparse.ArgumentParser(description="CLI verifier for Unified Commodity Predictor.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--eval-summary", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--commodity", type=str, default="Onion")
    parser.add_argument("--market", type=str, default="Lasalgaon")
    parser.add_argument("--price", type=float, default=1200.0)
    parser.add_argument("--arrivals", type=float, default=100.0)
    parser.add_argument("--month", type=int, default=11)
    args = parser.parse_args()

    predictor = UnifiedCommodityPredictor(models_dir=PROJECT_ROOT / "models")
    if args.historical:
        run_historical_test(predictor, args.commodity, args.market)
    elif args.eval_summary:
        run_eval_summary(predictor)
    elif args.custom:
        run_custom_test(predictor, args.commodity, args.market, args.price, args.arrivals, args.month)
    else:
        run_quick_test(predictor)


if __name__ == "__main__":
    main()
