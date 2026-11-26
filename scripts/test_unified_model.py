"""
Unified Multi-Commodity Model Testing & Inference CLI
AI-Based Food Inflation Prediction and Household Affordability Monitoring System

Tests and evaluates the single global cross-commodity regression model across any food commodity.

Usage:
  1. Quick sanity test across all benchmark food categories:
     py -3.12 scripts/test_unified_model.py

  2. Test against real unseen historical records:
     py -3.12 scripts/test_unified_model.py --historical --samples 10
     py -3.12 scripts/test_unified_model.py --historical --commodity Tomato --samples 5

  3. Test with custom food commodity & price:
     py -3.12 scripts/test_unified_model.py --custom --commodity "Moong Dal" --price 7200
     py -3.12 scripts/test_unified_model.py --custom --commodity "Potato" --price 1800 --arrivals 200

  4. Full unseen test-set evaluation summary:
     py -3.12 scripts/test_unified_model.py --eval-summary
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import UnifiedCommodityPredictor

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_unified_commodities.csv"


def run_quick_test(predictor: UnifiedCommodityPredictor):
    print("=" * 85)
    print("MODE 1: MULTI-COMMODITY QUICK SANITY TEST (Single Universal Model)")
    print("=" * 85)
    print(f"Model: {predictor.metadata.get('champion_algorithm', 'Ridge Regression')}")
    print(f"Trained across: {', '.join(predictor.metadata.get('supported_commodities', []))}")
    print("-" * 85)

    test_basket = [
        {"commodity": "Tomato", "price": 1400.0, "arrivals": 150.0, "note": "Perishable Vegetable"},
        {"commodity": "Onion", "price": 1250.0, "arrivals": 900.0, "note": "Semi-perishable Vegetable"},
        {"commodity": "Sugar", "price": 3600.0, "arrivals": 50.0, "note": "Non-perishable Sweet"},
        {"commodity": "Moong Dal", "price": 7400.0, "arrivals": 30.0, "note": "Staple Pulse"},
        {"commodity": "Potato", "price": 1800.0, "arrivals": 250.0, "note": "Generalization Test (Unseen Crop)"}
    ]

    print(f"{'Commodity':<12} | {'Category':<15} | {'Today Price':<14} | {'Predicted (t+1)':<15} | {'Shift':<10} | {'Shift %':<8}")
    print("-" * 85)

    for item in test_basket:
        res = predictor.predict(
            commodity=item["commodity"],
            current_price=item["price"],
            arrivals=item["arrivals"]
        )
        comm_display = f"{res['commodity']}"
        cat_display = f"{res['category']}"
        today_str = f"Rs.{res['observed_price_rs_per_kg']:.2f}/kg"
        pred_str = f"Rs.{res['predicted_next_price_rs_per_kg']:.2f}/kg"
        diff_str = f"Rs.{res['predicted_change_rs_per_qtl']:+.1f}/q"
        pct_str = f"{res['predicted_change_percentage']:+.2f}%"

        print(f"{comm_display:<12} | {cat_display:<15} | {today_str:<14} | {pred_str:<15} | {diff_str:<10} | {pct_str:<8}")

    print("=" * 85)
    print("All commodity predictions executed by a SINGLE unified model pipeline.\n")


def run_historical_test(predictor: UnifiedCommodityPredictor, commodity: str = None, n_samples: int = 10):
    print("=" * 88)
    target_desc = f"'{commodity}'" if commodity else "All Commodities"
    print(f"MODE 2: HISTORICAL TEST ON UNSEEN DATA ({target_desc} - Unified Model)")
    print("=" * 88)

    if not DATA_PATH.exists():
        print(f"Error: Unified dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    test_df = df[df['split_tag'] == 'Test'].copy()

    if commodity:
        test_df = test_df[test_df['commodity_name'].str.lower() == commodity.lower()].copy()
        if len(test_df) == 0:
            print(f"No test records found for commodity '{commodity}'. Available: {df['commodity_name'].unique()}")
            return

    step = max(1, len(test_df) // n_samples)
    sampled = test_df.iloc[::step].head(n_samples)

    preds = predictor.predict_batch(sampled)

    print(f"{'Date':<11} | {'Commodity':<11} | {'Today (t)':<12} | {'Actual (t+1)':<13} | {'Predicted':<12} | {'Abs Error':<10} | {'Error %':<7}")
    print("-" * 88)

    errors = []
    for idx, (_, row) in enumerate(sampled.iterrows()):
        actual = row['target_price_t_plus_1']
        pred = preds[idx]
        today = row['price_lag_1']
        comm = row['commodity_name']
        date_str = str(row['observation_date'])

        err = abs(actual - pred)
        err_pct = (err / actual) * 100 if actual > 0 else 0
        errors.append(err)

        print(f"{date_str:<11} | {comm:<11} | Rs.{today:<8.0f}/q | Rs.{actual:<9.1f}/q | Rs.{pred:<8.1f}/q | Rs.{err:<6.1f}/q | {err_pct:<5.1f}%")

    print("-" * 88)
    mean_err = np.mean(errors)
    print(f"Sample Average Absolute Error: Rs. {mean_err:.2f}/Quintal (Rs. {mean_err/100:.2f}/Kg)")
    print("=" * 88 + "\n")


def run_eval_summary(predictor: UnifiedCommodityPredictor):
    print("=" * 85)
    print("MODE 3: UNSEEN TEST SET BENCHMARK EVALUATION (Unified Multi-Commodity Model)")
    print("=" * 85)

    meta = predictor.get_metadata()
    overall = meta.get("overall_test_metrics", {})

    print(f"Model Architecture : {meta.get('model_architecture')}")
    print(f"Champion Algorithm : {meta.get('champion_algorithm')}")
    print(f"Total Test Records : {meta.get('total_test_records')} trading sessions")
    print("-" * 85)
    print(f"  Overall Mean Absolute Error (MAE)     : Rs. {overall.get('mae_rs_per_qtl', 0):.2f} / Quintal (Rs. {overall.get('mae_rs_per_qtl', 0)/100:.2f} / Kg)")
    print(f"  Overall Root Mean Squared Error (RMSE): Rs. {overall.get('rmse_rs_per_qtl', 0):.2f} / Quintal")
    print(f"  Overall Coefficient of Det. (R2 Score): {overall.get('r2_score', 0):.4f} ({overall.get('r2_score', 0)*100:.2f}% Variance Explained)")
    print(f"  Overall Mean Abs Percentage Err (MAPE): {overall.get('mape_pct', 0):.2f}%")
    print("-" * 85)

    print("Per-Commodity Breakdown (Champion Model Performance):")
    print(f"{'Commodity':<12} | {'Category':<16} | {'Mean Price':<14} | {'Test MAE':<14} | {'Test RMSE':<12} | {'R2':<7} | {'MAPE':<6}")
    print("-" * 85)
    for comm, metrics in meta.get("per_commodity_metrics", {}).items():
        cat = metrics.get('category', 'Food')
        mean_p = metrics.get('mean_price_rs_per_qtl', 0)
        mae = metrics.get('test_mae_rs_per_qtl', 0)
        rmse = metrics.get('test_rmse_rs_per_qtl', 0)
        r2 = metrics.get('test_r2', 0)
        mape = metrics.get('test_mape_pct', 0)
        print(f"{comm:<12} | {cat:<16} | Rs.{mean_p:<10.0f}/q | Rs.{mae:<6.2f} (Rs.{mae/100:.2f}/k) | Rs.{rmse:<8.2f}/q | {r2:<6.4f} | {mape:<5.1f}%")

    print("-" * 85)
    print("Candidate Algorithms Benchmarked:")
    for m in meta.get("all_models_benchmarked", []):
        marker = " <-- [CHAMPION]" if m["Model"] == meta["champion_algorithm"] else ""
        print(f"  {m['Model']:<30}: RMSE = Rs.{m['RMSE']:<7.2f} | R2 = {m['R2']:<6.4f} | MAPE = {m['MAPE']:<5.2f}%{marker}")
    print("=" * 85 + "\n")


def run_custom_test(predictor: UnifiedCommodityPredictor, commodity: str, price: float, arrivals: float, month: int, day_of_week: int, category: str):
    print("=" * 75)
    print("MODE 4: CUSTOM FOOD COMMODITY INFERENCE (Unified Model)")
    print("=" * 75)

    res = predictor.predict(
        commodity=commodity,
        current_price=price,
        arrivals=arrivals,
        month=month,
        day_of_week=day_of_week,
        category=category
    )

    print(f"Input Commodity     : {res['commodity']} ({res['category']})")
    print(f"Perishable Crop     : {'Yes' if res['is_perishable'] else 'No'}")
    print(f"Current Price (t)   : Rs. {res['observed_price_rs_per_qtl']:.2f}/Quintal (Rs. {res['observed_price_rs_per_kg']:.2f}/Kg)")
    print(f"Arrival Volume      : {arrivals:.1f} Tonnes")
    print(f"Prediction Horizon  : {res['prediction_horizon']}")
    print("-" * 75)
    print(f"PREDICTED PRICE(t+1): Rs. {res['predicted_next_price_rs_per_qtl']:.2f}/Quintal (Rs. {res['predicted_next_price_rs_per_kg']:.2f}/Kg)")
    print(f"Predicted Shift     : Rs. {res['predicted_change_rs_per_qtl']:+.2f}/Quintal ({res['predicted_change_percentage']:+.2f}%)")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Commodity Food Inflation Testing & Inference CLI.")
    parser.add_argument("--commodity", type=str, default=None, help="Commodity name (e.g. Tomato, Onion, Sugar, Moong Dal, Potato).")
    parser.add_argument("--quick", action="store_true", help="Run multi-commodity quick sanity test.")
    parser.add_argument("--historical", action="store_true", help="Test model on real unseen historical test records.")
    parser.add_argument("--samples", type=int, default=10, help="Number of historical samples to display (default: 10).")
    parser.add_argument("--eval-summary", action="store_true", help="Display full unseen test set performance metrics.")
    parser.add_argument("--custom", action="store_true", help="Test with custom commodity and price.")
    parser.add_argument("--price", type=float, default=1500.0, help="Current wholesale price in Rs/Quintal (default: 1500.0).")
    parser.add_argument("--arrivals", type=float, default=100.0, help="Arrival volume in Metric Tonnes (default: 100.0).")
    parser.add_argument("--category", type=str, default=None, help="Food category (e.g. Vegetables, Pulses, Cereals).")
    parser.add_argument("--month", type=int, default=None, help="Month 1-12.")
    parser.add_argument("--day", type=int, default=None, help="Day of week 0-6.")

    args = parser.parse_args()

    try:
        predictor = UnifiedCommodityPredictor()
    except Exception as e:
        print(f"Failed to load UnifiedCommodityPredictor: {e}")
        sys.exit(1)

    has_specific_flag = args.quick or args.historical or args.eval_summary or args.custom

    if not has_specific_flag:
        run_quick_test(predictor)
        run_historical_test(predictor, commodity=args.commodity, n_samples=8)
        print("Tip: Run 'py -3.12 scripts/test_unified_model.py --help' to see all modes (--historical, --custom, --eval-summary).")
    else:
        if args.quick:
            run_quick_test(predictor)
        if args.historical:
            run_historical_test(predictor, commodity=args.commodity, n_samples=args.samples)
        if args.eval_summary:
            run_eval_summary(predictor)
        if args.custom:
            run_custom_test(
                predictor,
                commodity=args.commodity or "Tomato",
                price=args.price,
                arrivals=args.arrivals,
                month=args.month,
                day_of_week=args.day,
                category=args.category
            )


if __name__ == "__main__":
    main()
