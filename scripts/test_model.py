"""
Universal Model Testing & Inference CLI
AI-Based Food Inflation Prediction and Household Affordability Monitoring System

Usage:
  1. Quick sanity test:
     py -3.12 scripts/test_model.py

  2. Test against real unseen historical market records:
     py -3.12 scripts/test_model.py --historical --samples 10

  3. Test with custom values:
     py -3.12 scripts/test_model.py --custom --price 1500 --arrivals 160 --month 7

  4. Run overall test-set evaluation summary:
     py -3.12 scripts/test_model.py --eval-summary
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import CommodityPricePredictor


def run_quick_test(predictor: CommodityPricePredictor):
    print("=" * 70)
    print("MODE 1: QUICK SANITY TEST (Simulated Monsoon Trading Session)")
    print("=" * 70)

    sample_market_day = {
        'price_lag_1': 1400.0,            # Today's price: Rs. 14.00/kg
        'price_lag_3': 1250.0,            # 3 days ago: Rs. 12.50/kg
        'price_lag_7': 1100.0,            # 7 days ago: Rs. 11.00/kg
        'rolling_mean_7': 1280.0,         # 7-day average
        'rolling_std_7': 120.0,           # Volatility
        'rolling_min_7': 1100.0,          # Weekly low
        'rolling_max_7': 1400.0,          # Weekly high
        'price_momentum_7': 300.0,        # Trend (+Rs. 300/Qtl)
        'daily_spread': 400.0,            # Intraday auction spread
        'arrival_tonnes': 145.0,          # Today's arrival volume
        'arrival_lag_1': 160.0,           # Yesterday's arrivals
        'rolling_arrival_mean_7': 155.0,  # Weekly average arrivals
        'month': 7,                       # July (Peak Monsoon)
        'day_of_week': 3,                 # Thursday
        'is_monsoon_season': 1            # Monsoon: Yes
    }

    res = predictor.predict(sample_market_day)

    print(f"Commodity & Market        : {res['commodity']} ({res['market']})")
    print(f"Current Observed Price (t): Rs. {res['observed_price_rs_per_qtl']:.2f}/Quintal (Rs. {res['observed_price_rs_per_kg']:.2f}/Kg)")
    print(f"Arrival Volume Today      : {sample_market_day['arrival_tonnes']} Metric Tonnes")
    print(f"Forecast Horizon          : {res['prediction_horizon']}")
    print("-" * 70)
    print(f"PREDICTED NEXT PRICE (t+1): Rs. {res['predicted_next_price_rs_per_qtl']:.2f}/Quintal (Rs. {res['predicted_next_price_rs_per_kg']:.2f}/Kg)")
    print(f"Predicted Daily Change    : Rs. {res['predicted_change_rs_per_qtl']:+.2f}/Quintal ({res['predicted_change_percentage']:+.2f}%)")
    print("=" * 70)
    print("Result: Quick test passed successfully!\n")


def run_historical_test(predictor: CommodityPricePredictor, commodity: str = "Tomato", market: str = "Pune", n_samples: int = 10):
    print("=" * 80)
    print(f"MODE 2: HISTORICAL TEST ON UNSEEN DATA (Evaluating {n_samples} Real APMC Sessions for {commodity} @ {market})")
    print("=" * 80)

    safe_comm = commodity.lower().replace(' ', '_')
    safe_market = market.lower().replace(' ', '_')
    data_file = PROJECT_ROOT / "data" / "processed" / f"model_ready_{safe_market}_{safe_comm}.csv"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "data" / "processed" / "model_ready_pune_tomato.csv"

    if not data_file.exists():
        print(f"Error: Dataset not found at {data_file}")
        return

    df = pd.read_csv(data_file)
    test_df = df[df['split_tag'] == 'Test'].copy()

    if len(test_df) == 0:
        print("Error: No test partition found in dataset.")
        return

    # Select samples evenly spaced across test set
    step = max(1, len(test_df) // n_samples)
    sampled = test_df.iloc[::step].head(n_samples)

    features = predictor.feature_names
    predictions = predictor.predict_batch(sampled)

    print(f"{'Date':<12} | {'Today (t)':<10} | {'Arrivals':<9} | {'Actual (t+1)':<12} | {'Predicted':<11} | {'Abs Err':<9} | {'Error %':<7}")
    print("-" * 80)

    errors = []
    for idx, (_, row) in enumerate(sampled.iterrows()):
        actual = row['target_price_t_plus_1']
        pred = predictions[idx]
        today = row['price_lag_1']
        arrival = row['arrival_tonnes']
        date_str = str(row['observation_date'])

        err = abs(actual - pred)
        err_pct = (err / actual) * 100 if actual > 0 else 0
        errors.append(err)

        print(f"{date_str:<12} | Rs.{today:<6.0f} | {arrival:<7.0f}t | Rs.{actual:<8.2f} | Rs.{pred:<7.2f} | Rs.{err:<5.2f} | {err_pct:<5.1f}%")

    print("-" * 80)
    print(f"Sample Average Absolute Error: Rs. {np.mean(errors):.2f}/Quintal (Rs. {np.mean(errors)/100:.2f}/Kg)")
    print("=" * 80 + "\n")


def run_eval_summary(predictor: CommodityPricePredictor, commodity: str = "Tomato", market: str = "Pune"):
    print("=" * 70)
    print(f"MODE 3: FULL UNSEEN TEST SET BENCHMARK EVALUATION ({commodity} @ {market})")
    print("=" * 70)

    safe_comm = commodity.lower().replace(' ', '_')
    safe_market = market.lower().replace(' ', '_')
    data_file = PROJECT_ROOT / "data" / "processed" / f"model_ready_{safe_market}_{safe_comm}.csv"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "data" / "processed" / "model_ready_pune_tomato.csv"

    df = pd.read_csv(data_file)
    test_df = df[df['split_tag'] == 'Test'].copy()

    y_true = test_df['target_price_t_plus_1'].values
    y_pred = np.array(predictor.predict_batch(test_df))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    meta = predictor.get_metadata()

    print(f"Total Test Set Records Evaluated: {len(test_df)} trading sessions")
    print(f"Temporal Period Covered         : {test_df['observation_date'].min()} to {test_df['observation_date'].max()}")
    print("-" * 70)
    print(f"  Mean Absolute Error (MAE)     : Rs. {mae:.2f} / Quintal")
    print(f"  Root Mean Squared Error (RMSE): Rs. {rmse:.2f} / Quintal (Rs. {rmse/100:.2f}/Kg)")
    print(f"  Coefficient of Det. (R2 Score): {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"  Mean Abs Percentage Err (MAPE): {mape:.2f}%")
    print("-" * 70)
    print("Evaluated Model Algorithms Comparison:")
    for m in meta.get("all_models_evaluated", []):
        marker = " <-- [WINNER]" if m["Model"] == meta["model_name"] else ""
        print(f"  {m['Model']:<30}: RMSE = Rs. {m['RMSE']:<7.2f} | R2 = {m['R2']:<6.4f}{marker}")
    print("=" * 70 + "\n")


def run_custom_test(predictor: CommodityPricePredictor, price: float, arrivals: float, month: int, day_of_week: int):
    print("=" * 70)
    print("MODE 4: CUSTOM MARKET OBSERVATION TEST")
    print("=" * 70)

    # Approximate plausible rolling features around the user's custom price
    is_monsoon = 1 if month in [6, 7, 8, 9] else 0
    custom_vector = {
        'price_lag_1': price,
        'price_lag_3': price * 0.98,
        'price_lag_7': price * 0.95,
        'rolling_mean_7': price * 0.97,
        'rolling_std_7': price * 0.08,
        'rolling_min_7': price * 0.90,
        'rolling_max_7': price * 1.05,
        'price_momentum_7': price * 0.05,
        'daily_spread': price * 0.25,
        'arrival_tonnes': arrivals,
        'arrival_lag_1': arrivals * 1.05,
        'rolling_arrival_mean_7': arrivals,
        'month': month,
        'day_of_week': day_of_week,
        'is_monsoon_season': is_monsoon
    }

    res = predictor.predict(custom_vector)
    print(f"Input Today's Price : Rs. {price:.2f} / Quintal (Rs. {price/100:.2f} / Kg)")
    print(f"Input Arrival Volume: {arrivals:.1f} Tonnes")
    print(f"Input Month / Day   : Month {month}, Day of Week {day_of_week} (Monsoon: {'Yes' if is_monsoon else 'No'})")
    print("-" * 70)
    print(f"PREDICTED NEXT PRICE: Rs. {res['predicted_next_price_rs_per_qtl']:.2f} / Quintal (Rs. {res['predicted_next_price_rs_per_kg']:.2f} / Kg)")
    print(f"Expected Price Shift: Rs. {res['predicted_change_rs_per_qtl']:+.2f} / Quintal ({res['predicted_change_percentage']:+.2f}%)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test and run inference on the trained Commodity Price Model.")
    parser.add_argument("--commodity", type=str, default="Tomato", help="Commodity to test (e.g. Tomato, Onion).")
    parser.add_argument("--market", type=str, default="Pune", help="Market name (e.g. Pune, Lasalgaon).")
    parser.add_argument("--quick", action="store_true", help="Run quick sanity check on simulated market day.")
    parser.add_argument("--historical", action="store_true", help="Test model on real unseen historical test records.")
    parser.add_argument("--samples", type=int, default=10, help="Number of historical samples to display (default: 10).")
    parser.add_argument("--eval-summary", action="store_true", help="Display full unseen test set performance metrics.")
    parser.add_argument("--custom", action="store_true", help="Test with custom price and arrival values.")
    parser.add_argument("--price", type=float, default=1500.0, help="Custom wholesale price in Rs/Quintal (default: 1500.0).")
    parser.add_argument("--arrivals", type=float, default=150.0, help="Custom arrival volume in Metric Tonnes (default: 150.0).")
    parser.add_argument("--month", type=int, default=7, help="Month 1-12 (default: 7).")
    parser.add_argument("--day", type=int, default=3, help="Day of week 0-6 (0=Mon, default: 3=Thu).")

    args = parser.parse_args()

    # Load predictor with commodity
    try:
        predictor = CommodityPricePredictor(commodity=args.commodity)
    except Exception as e:
        print(f"Failed to load CommodityPricePredictor: {e}")
        sys.exit(1)

    # Route based on flags
    has_specific_flag = args.quick or args.historical or args.eval_summary or args.custom

    if not has_specific_flag:
        # Default run: run quick test followed by historical test sample
        run_quick_test(predictor)
        run_historical_test(predictor, commodity=args.commodity, market=args.market, n_samples=5)
        print("Tip: Run 'py -3.12 scripts/test_model.py --help' to see all testing modes (historical, custom, eval-summary).")
    else:
        if args.quick:
            run_quick_test(predictor)
        if args.historical:
            run_historical_test(predictor, commodity=args.commodity, market=args.market, n_samples=args.samples)
        if args.eval_summary:
            run_eval_summary(predictor, commodity=args.commodity, market=args.market)
        if args.custom:
            run_custom_test(predictor, price=args.price, arrivals=args.arrivals, month=args.month, day_of_week=args.day)


if __name__ == "__main__":
    main()
