"""
CLI Verification Runner for Specialist Commodity Models (Tomato / Onion @ Pune).

Usage:
  py -3.12 scripts/test_model.py --commodity Tomato --quick
  py -3.12 scripts/test_model.py --commodity Tomato --historical
  py -3.12 scripts/test_model.py --commodity Onion --eval-summary
  py -3.12 scripts/test_model.py --commodity Tomato --custom --price 1600 --arrivals 110 --month 7
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import CommodityPricePredictor
from src.live_data import get_historical_latest_features


def run_quick_test(predictor: CommodityPricePredictor):
    print("\n--- QUICK SPECIALIST SANITY TEST ---")
    hist = get_historical_latest_features(predictor.commodity, predictor.market)
    res = predictor.predict_from_features(hist["features"], feature_mode="historical_archive_replay")
    print(f"Commodity / Market        : {res['commodity']} @ {res['market']}")
    print(f"Observation Date          : {hist['observation_date']} (Actual t+1: Rs. {hist['actual_next_price_rs_per_qtl']}/Qtl)")
    print(f"Predicted Next Session    : Rs. {res['predicted_next_session_rs_per_qtl']}/Qtl")
    print(f"Wholesale Equiv (Rs/Kg)   : Rs. {res['wholesale_equiv_rs_per_kg']}/Kg (unit conversion only)")
    print(f"Retail ML Estimation      : {res['retail_estimation_status']}")
    print(f"Alert Status              : {res['inflation_alert_badge']}")


def run_historical_test(predictor: CommodityPricePredictor):
    print("\n--- HISTORICAL ARCHIVE BACKTEST (LAST 5 UNTOUCHED TEST SESSIONS) ---")
    safe_comm = predictor.commodity.lower().replace(" ", "_")
    safe_mkt = predictor.market.lower().replace(" ", "_")
    path = PROJECT_ROOT / "data" / "processed" / f"model_ready_{safe_mkt}_{safe_comm}.csv"
    df = pd.read_csv(path)
    test_df = df[df["split_tag"] == "Test"].tail(5)

    for _, row in test_df.iterrows():
        feats = {col: float(row[col]) for col in predictor.feature_names}
        res = predictor.predict_from_features(feats, feature_mode="historical_test_replay")
        actual = float(row["target_price_t_plus_1"])
        err = res["predicted_next_session_rs_per_qtl"] - actual
        print(
            f"  {row['observation_date']} -> Target {row.get('target_date_t_plus_1', 't+1')} | "
            f"Current={row['modal_price_rs_per_qtl']:.1f} | Pred={res['predicted_next_session_rs_per_qtl']:.1f} | "
            f"Actual={actual:.1f} | Error={err:+.1f}"
        )


def run_eval_summary(predictor: CommodityPricePredictor):
    print("\n--- SPECIALIST MODEL EVALUATION SUMMARY ---")
    m = predictor.metadata
    tm = m.get("test_evaluation_metrics", {})
    bm = m.get("persistence_baseline_test_metrics", {})
    imp = m.get("improvement_over_baseline_test", {})
    print(f"Champion (Selected on Val) : {m.get('model_name')}")
    print(f"Selection Protocol         : {m.get('selection_protocol')}")
    print(f"Prediction Horizon         : {m.get('prediction_horizon')}")
    print(f"Splits (Tr / Val / Te)     : {m.get('training_samples')} / {m.get('validation_samples')} / {m.get('test_samples')}")
    print(f"Untouched Test Metrics     : MAE={tm.get('MAE')} | RMSE={tm.get('RMSE')} | R2={tm.get('R2')} | MAPE={tm.get('MAPE (%)')}%")
    print(f"Persistence Baseline Test  : MAE={bm.get('MAE')} | RMSE={bm.get('RMSE')} | R2={bm.get('R2')}")
    print(f"RMSE Improvement vs Base   : {imp.get('rmse_reduction_rs_per_qtl')} Rs/Qtl (Beats={imp.get('beats_baseline_rmse')})")


def run_custom_test(predictor: CommodityPricePredictor, price: float, arrivals: float, month: int):
    print("\n--- CUSTOM MANUAL SCENARIO (APPROXIMATED LAGS OPT-IN) ---")
    res = predictor.predict(
        modal_price=price,
        price_lag_1=price * 0.97,
        arrival_tonnes=arrivals,
        month=month,
        allow_manual_approx=True,
    )
    print(f"Input Current Price : Rs. {price:.2f}/Qtl | Mode: {res['feature_mode']}")
    print(f"Predicted Next Sess : Rs. {res['predicted_next_session_rs_per_qtl']:.2f}/Qtl ({res['predicted_change_pct']:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="CLI verifier for Specialist Models.")
    parser.add_argument("--commodity", type=str, default="Tomato")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--eval-summary", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--price", type=float, default=1600.0)
    parser.add_argument("--arrivals", type=float, default=110.0)
    parser.add_argument("--month", type=int, default=7)
    args = parser.parse_args()

    predictor = CommodityPricePredictor(models_dir=PROJECT_ROOT / "models", commodity=args.commodity)
    if args.historical:
        run_historical_test(predictor)
    elif args.eval_summary:
        run_eval_summary(predictor)
    elif args.custom:
        run_custom_test(predictor, args.price, args.arrivals, args.month)
    else:
        run_quick_test(predictor)


if __name__ == "__main__":
    main()
