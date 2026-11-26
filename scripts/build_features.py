"""
Multi-Commodity Feature Engineering Pipeline
Generates model-ready datasets for ANY commodity from wholesale_prices.csv + market_arrivals.csv.

Usage:
  py -3.12 scripts/build_features.py --commodity Tomato --market Pune
  py -3.12 scripts/build_features.py --commodity Onion --market Pune
  py -3.12 scripts/build_features.py --all  (builds for all viable commodity-market pairs)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_wholesale_series(commodity: str, market_keyword: str) -> pd.DataFrame:
    """Load and filter wholesale price series for a given commodity and market."""
    ws = pd.read_csv(PROCESSED_DIR / "wholesale_prices.csv")
    # Filter commodity (exact match)
    mask = ws['commodity_name'].str.lower() == commodity.lower()
    sub = ws[mask].copy()
    if len(sub) == 0:
        raise ValueError(f"No wholesale records found for commodity '{commodity}'")
    
    # Filter market
    market_mask = sub['market_name'].str.contains(market_keyword, case=False, na=False)
    sub = sub[market_mask].copy()
    if len(sub) == 0:
        raise ValueError(f"No records for '{commodity}' in market '{market_keyword}'")
    
    sub['observation_date'] = pd.to_datetime(sub['observation_date'])
    sub = sub.sort_values('observation_date').reset_index(drop=True)
    
    # Aggregate multiple sub-markets (e.g. Pune, Pune(Pimpri)) to daily median
    daily = sub.groupby('observation_date').agg({
        'min_price_rs_per_qtl': 'min',
        'max_price_rs_per_qtl': 'max',
        'modal_price_rs_per_qtl': 'median',
        'commodity_name': 'first',
        'state': 'first',
        'district': 'first',
        'market_name': 'first',
        'variety': 'first',
        'price_unit': 'first',
        'price_type': 'first',
        'data_source': 'first'
    }).reset_index()
    
    return daily


def load_arrivals(commodity: str, market_keyword: str) -> pd.DataFrame:
    """Load market arrivals data if available."""
    arrivals_path = PROCESSED_DIR / "market_arrivals.csv"
    if not arrivals_path.exists():
        return pd.DataFrame()
    
    ma = pd.read_csv(arrivals_path)
    mask = ma['commodity_name'].str.lower() == commodity.lower()
    sub = ma[mask].copy()
    
    market_mask = sub['market_name'].str.contains(market_keyword, case=False, na=False)
    sub = sub[market_mask].copy()
    
    if len(sub) == 0:
        return pd.DataFrame()
    
    sub['observation_date'] = pd.to_datetime(sub['observation_date'])
    daily = sub.groupby('observation_date').agg({
        'arrival_quantity_tonnes': 'sum'
    }).reset_index()
    
    return daily


def engineer_features(df: pd.DataFrame, has_arrivals: bool = True) -> pd.DataFrame:
    """Engineer time-series features from price data. All lookback only (zero future leakage)."""
    out = df.copy()
    price_col = 'modal_price_rs_per_qtl'
    
    # Price lag features (backward only)
    out['price_lag_1'] = out[price_col].shift(1)
    out['price_lag_3'] = out[price_col].shift(3)
    out['price_lag_7'] = out[price_col].shift(7)
    
    # Rolling window statistics (backward only)
    out['rolling_mean_7'] = out[price_col].shift(1).rolling(window=7, min_periods=4).mean()
    out['rolling_std_7'] = out[price_col].shift(1).rolling(window=7, min_periods=4).std()
    out['rolling_min_7'] = out[price_col].shift(1).rolling(window=7, min_periods=4).min()
    out['rolling_max_7'] = out[price_col].shift(1).rolling(window=7, min_periods=4).max()
    
    # Price momentum and volatility indicators
    out['price_momentum_7'] = out[price_col] - out[price_col].shift(7)
    out['daily_spread'] = out['max_price_rs_per_qtl'] - out['min_price_rs_per_qtl']
    
    # Supply-side features (arrivals)
    if has_arrivals and 'arrival_quantity_tonnes' in out.columns:
        out['arrival_tonnes'] = out['arrival_quantity_tonnes']
        out['arrival_lag_1'] = out['arrival_tonnes'].shift(1)
        out['rolling_arrival_mean_7'] = out['arrival_tonnes'].shift(1).rolling(window=7, min_periods=4).mean()
    else:
        out['arrival_tonnes'] = 0.0
        out['arrival_lag_1'] = 0.0
        out['rolling_arrival_mean_7'] = 0.0
    
    # Calendar features
    out['month'] = out['observation_date'].dt.month
    out['day_of_week'] = out['observation_date'].dt.dayofweek
    out['day_of_year'] = out['observation_date'].dt.dayofyear
    out['is_monsoon_season'] = out['month'].isin([6, 7, 8, 9]).astype(int)
    
    # Target variables (future prices — only used as labels, never as features)
    out['target_price_t_plus_1'] = out[price_col].shift(-1)
    out['target_price_t_plus_7'] = out[price_col].shift(-7)
    
    # Classification target: price spike > 10% in next session
    out['price_change_pct'] = ((out['target_price_t_plus_1'] - out[price_col]) / out[price_col]) * 100
    out['is_price_spike'] = (out['price_change_pct'].abs() > 10).astype(int)
    out['spike_direction'] = 'Normal'
    out.loc[out['price_change_pct'] > 10, 'spike_direction'] = 'Surge'
    out.loc[out['price_change_pct'] < -10, 'spike_direction'] = 'Crash'
    
    return out


def add_split_tags(df: pd.DataFrame, train_frac=0.70, val_frac=0.15) -> pd.DataFrame:
    """Add chronological train/validation/test split tags."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    df['split_tag'] = 'Test'
    df.iloc[:train_end, df.columns.get_loc('split_tag')] = 'Train'
    df.iloc[train_end:val_end, df.columns.get_loc('split_tag')] = 'Validation'
    
    return df


def build_model_ready_dataset(commodity: str, market_keyword: str) -> pd.DataFrame:
    """Full pipeline: load → merge arrivals → engineer features → split → clean."""
    print(f"\n{'='*60}")
    print(f"Building features for: {commodity} @ {market_keyword}")
    print(f"{'='*60}")
    
    # Load wholesale prices
    prices = load_wholesale_series(commodity, market_keyword)
    print(f"  Wholesale price records: {len(prices)}")
    print(f"  Date range: {prices['observation_date'].min()} to {prices['observation_date'].max()}")
    
    # Load arrivals
    arrivals = load_arrivals(commodity, market_keyword)
    has_arrivals = len(arrivals) > 0
    if has_arrivals:
        prices = prices.merge(arrivals, on='observation_date', how='left')
        prices['arrival_quantity_tonnes'] = prices['arrival_quantity_tonnes'].fillna(0)
        print(f"  Arrivals merged: {(prices['arrival_quantity_tonnes'] > 0).sum()} days with arrival data")
    else:
        print(f"  WARNING: No arrival data available for {commodity} @ {market_keyword}")
    
    # Engineer features
    featured = engineer_features(prices, has_arrivals)
    
    # Drop rows with NaN in key feature/target columns
    feature_cols = [
        'price_lag_1', 'price_lag_3', 'price_lag_7',
        'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
        'price_momentum_7', 'daily_spread', 'arrival_tonnes', 'arrival_lag_1',
        'rolling_arrival_mean_7', 'month', 'day_of_week', 'is_monsoon_season'
    ]
    
    # Must have target
    featured = featured.dropna(subset=['target_price_t_plus_1'] + feature_cols)
    featured = featured.reset_index(drop=True)
    
    # Add chronological split tags
    featured = add_split_tags(featured)
    
    print(f"  Model-ready records: {len(featured)}")
    train_n = (featured['split_tag'] == 'Train').sum()
    val_n = (featured['split_tag'] == 'Validation').sum()
    test_n = (featured['split_tag'] == 'Test').sum()
    print(f"  Split: Train={train_n}, Validation={val_n}, Test={test_n}")
    
    return featured


def save_dataset(df: pd.DataFrame, commodity: str, market: str):
    """Save model-ready dataset to processed directory."""
    safe_commodity = commodity.lower().replace(' ', '_').replace('(', '').replace(')', '')
    safe_market = market.lower().replace(' ', '_')
    filename = f"model_ready_{safe_market}_{safe_commodity}.csv"
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} rows)")
    return path


# Viable commodity-market pairs with sufficient historical data
VIABLE_PAIRS = [
    ("Tomato", "Pune"),
    ("Onion", "Pune"),
    ("Onion", "Lasalgaon"),
]


def build_unified_dataset() -> Path:
    """Build unified cross-commodity dataset spanning vegetables, pulses, and sweets."""
    print(f"\n{'='*60}")
    print("Building Unified Multi-Commodity Dataset")
    print(f"{'='*60}")

    dfs = []

    # 1. Tomato Pune
    tom_path = PROCESSED_DIR / "model_ready_pune_tomato.csv"
    if not tom_path.exists():
        df_tom = build_model_ready_dataset("Tomato", "Pune")
        save_dataset(df_tom, "Tomato", "Pune")
    else:
        df_tom = pd.read_csv(tom_path)
    df_tom["commodity_name"] = "Tomato"
    df_tom["category"] = "Vegetables"
    df_tom["is_perishable"] = 1
    dfs.append(df_tom)
    print(f"  Loaded Tomato: {len(df_tom)} rows")

    # 2. Onion Pune
    oni_path = PROCESSED_DIR / "model_ready_pune_onion.csv"
    if not oni_path.exists():
        df_oni = build_model_ready_dataset("Onion", "Pune")
        save_dataset(df_oni, "Onion", "Pune")
    else:
        df_oni = pd.read_csv(oni_path)
    df_oni["commodity_name"] = "Onion"
    df_oni["category"] = "Vegetables"
    df_oni["is_perishable"] = 1
    dfs.append(df_oni)
    print(f"  Loaded Onion (Pune): {len(df_oni)} rows")

    # 3. Onion Lasalgaon
    las_path = PROCESSED_DIR / "model_ready_lasalgaon_onion.csv"
    if not las_path.exists():
        df_las = build_model_ready_dataset("Onion", "Lasalgaon")
        save_dataset(df_las, "Onion", "Lasalgaon")
    else:
        df_las = pd.read_csv(las_path)
    df_las["commodity_name"] = "Onion"
    df_las["category"] = "Vegetables"
    df_las["is_perishable"] = 1
    dfs.append(df_las)
    print(f"  Loaded Onion (Lasalgaon): {len(df_las)} rows")

    # 4. Sugar & Moong Dal from wholesale_prices.csv
    ws_path = PROCESSED_DIR / "wholesale_prices.csv"
    if ws_path.exists():
        ws = pd.read_csv(ws_path)
        ws['observation_date'] = pd.to_datetime(ws['observation_date'])

        # Sugar
        sub_sug = ws[ws['commodity_name'] == 'Sugar'].sort_values('observation_date')
        if len(sub_sug) > 100:
            daily_sug = sub_sug.groupby('observation_date').agg({
                'min_price_rs_per_qtl': 'min',
                'max_price_rs_per_qtl': 'max',
                'modal_price_rs_per_qtl': 'median',
                'commodity_name': 'first',
                'market_name': 'first'
            }).reset_index()
            feat_sug = engineer_features(daily_sug, has_arrivals=False)
            feat_sug = feat_sug.dropna(subset=['target_price_t_plus_1', 'price_lag_7']).reset_index(drop=True)
            feat_sug = add_split_tags(feat_sug)
            feat_sug['commodity_name'] = 'Sugar'
            feat_sug['category'] = 'Sugars & Sweets'
            feat_sug['is_perishable'] = 0
            dfs.append(feat_sug)
            print(f"  Loaded Sugar: {len(feat_sug)} rows")

        # Moong Dal
        sub_moong = ws[ws['commodity_name'] == 'Green Gram Dal (Moong Dal)'].sort_values('observation_date')
        if len(sub_moong) > 100:
            daily_moong = sub_moong.groupby('observation_date').agg({
                'min_price_rs_per_qtl': 'min',
                'max_price_rs_per_qtl': 'max',
                'modal_price_rs_per_qtl': 'median',
                'commodity_name': 'first',
                'market_name': 'first'
            }).reset_index()
            feat_moong = engineer_features(daily_moong, has_arrivals=False)
            feat_moong = feat_moong.dropna(subset=['target_price_t_plus_1', 'price_lag_7']).reset_index(drop=True)
            feat_moong = add_split_tags(feat_moong)
            feat_moong['commodity_name'] = 'Moong Dal'
            feat_moong['category'] = 'Pulses'
            feat_moong['is_perishable'] = 0
            dfs.append(feat_moong)
            print(f"  Loaded Moong Dal: {len(feat_moong)} rows")

    unified = pd.concat(dfs, ignore_index=True)
    out_csv = PROCESSED_DIR / "model_ready_unified_commodities.csv"
    unified.to_csv(out_csv, index=False)
    print(f"  Saved unified dataset: {out_csv} ({len(unified)} total rows)")
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Multi-commodity feature engineering pipeline.")
    parser.add_argument("--commodity", type=str, help="Commodity name (e.g., Tomato, Onion)")
    parser.add_argument("--market", type=str, help="Market keyword (e.g., Pune, Lasalgaon)")
    parser.add_argument("--all", action="store_true", help="Build all viable commodity-market pairs")
    parser.add_argument("--unified", action="store_true", help="Build unified multi-commodity pooled dataset")
    args = parser.parse_args()
    
    datasets_built = []
    
    if args.unified:
        out_csv = build_unified_dataset()
        print(f"Unified multi-commodity dataset ready at: {out_csv}")
        return

    if args.all:
        for commodity, market in VIABLE_PAIRS:
            try:
                df = build_model_ready_dataset(commodity, market)
                if len(df) >= 100:
                    path = save_dataset(df, commodity, market)
                    datasets_built.append((commodity, market, len(df), str(path)))
                else:
                    print(f"  SKIPPED: Only {len(df)} records (minimum 100 required)")
            except Exception as e:
                print(f"  ERROR for {commodity} @ {market}: {e}")
        # Also build unified when --all is specified
        build_unified_dataset()
    elif args.commodity and args.market:
        df = build_model_ready_dataset(args.commodity, args.market)
        path = save_dataset(df, args.commodity, args.market)
        datasets_built.append((args.commodity, args.market, len(df), str(path)))
    else:
        # Default: build Tomato Pune (backward compatible) and Onion Pune
        for commodity, market in [("Tomato", "Pune"), ("Onion", "Pune")]:
            try:
                df = build_model_ready_dataset(commodity, market)
                path = save_dataset(df, commodity, market)
                datasets_built.append((commodity, market, len(df), str(path)))
            except Exception as e:
                print(f"  ERROR for {commodity} @ {market}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Feature Engineering Complete: {len(datasets_built)} datasets built")
    for c, m, n, p in datasets_built:
        print(f"  {c} @ {m}: {n} rows → {p}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
