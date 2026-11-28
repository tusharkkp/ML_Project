"""
Canonical Leak-Free Time-Series Feature Engineering & Chronological Splitting Engine.

Shared by both offline training pipelines (scripts/build_features.py) and live/historical
inference workflows (src/live_data.py, src/inference.py) to guarantee zero training-serving skew
and zero future data leakage.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


SPECIALIST_NUMERIC_FEATURES: List[str] = [
    "modal_price_rs_per_qtl",
    "price_lag_1",
    "price_lag_3",
    "price_lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_min_7",
    "rolling_max_7",
    "price_momentum_7",
    "daily_spread",
    "arrival_tonnes",
    "arrival_lag_1",
    "rolling_arrival_mean_7",
    "month",
    "day_of_week",
    "is_monsoon_season",
]

UNIFIED_CATEGORICAL_FEATURES: List[str] = [
    "commodity_name",
    "category",
    "market_name",
]

UNIFIED_NUMERIC_FEATURES: List[str] = SPECIALIST_NUMERIC_FEATURES + ["is_perishable"]
UNIFIED_ALL_FEATURES: List[str] = UNIFIED_CATEGORICAL_FEATURES + UNIFIED_NUMERIC_FEATURES

TARGET_REG_COL = "target_price_t_plus_1"
TARGET_CLF_SPIKE_COL = "is_price_spike"
TARGET_CLF_DIR_COL = "spike_direction"

# Exact canonical single-market historical series with sufficient longitudinal density
CANONICAL_SERIES_SPECS: List[Dict[str, object]] = [
    {
        "raw_commodity": "Tomato",
        "commodity_name": "Tomato",
        "category": "Vegetables",
        "is_perishable": 1,
        "market_name": "Pune",
    },
    {
        "raw_commodity": "Onion",
        "commodity_name": "Onion",
        "category": "Vegetables",
        "is_perishable": 1,
        "market_name": "Pune",
    },
    {
        "raw_commodity": "Onion",
        "commodity_name": "Onion",
        "category": "Vegetables",
        "is_perishable": 1,
        "market_name": "Lasalgaon",
    },
    {
        "raw_commodity": "Green Gram Dal (Moong Dal)",
        "commodity_name": "Moong Dal",
        "category": "Pulses",
        "is_perishable": 0,
        "market_name": "Mumbai",
    },
    {
        "raw_commodity": "Sugar",
        "commodity_name": "Sugar",
        "category": "Sugars & Sweets",
        "is_perishable": 0,
        "market_name": "Mumbai",
    },
]

COMMODITY_CATEGORY_MAP: Dict[str, Tuple[str, int]] = {
    "Tomato": ("Vegetables", 1),
    "Onion": ("Vegetables", 1),
    "Moong Dal": ("Pulses", 0),
    "Sugar": ("Sugars & Sweets", 0),
}


def canonicalize_commodity_name(raw_name: str) -> str:
    """Maps source-specific commodity labels to canonical names without altering unknown items."""
    text = str(raw_name).strip()
    if text.lower() in {"green gram dal (moong dal)", "moong dal"}:
        return "Moong Dal"
    return text


def engineer_single_series_features(
    df: pd.DataFrame,
    has_arrivals: bool = True,
    max_target_gap_days: Optional[int] = 45,
    max_lookback_gap_days: Optional[int] = 60,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Compute backward-looking features and forward targets for a SINGLE (commodity, market) series.

    Guarantees:
    - Sorts strictly by ascending `observation_date`.
    - Requires unique `observation_date` (caller must aggregate/deduplicate explicitly before calling).
    - `modal_price_rs_per_qtl` is P(t); `price_lag_1` is strictly P(t-1), never P(t).
    - Rolling statistics use `.shift(1).rolling(7, min_periods=4)` over [t-7..t-1] only.
    - Explicit `target_date_t_plus_1` and session gap diagnostics (`lookback_7_gap_days`,
      `target_gap_days`) allow split-boundary purging and multi-year gap detection.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["observation_date"] = pd.to_datetime(out["observation_date"])
    if out["observation_date"].duplicated().any():
        dups = int(out["observation_date"].duplicated().sum())
        raise ValueError(
            f"Duplicate observation_date entries detected ({dups}) in single-series input. "
            "Deduplicate or aggregate explicitly before feature engineering."
        )

    out = out.sort_values("observation_date").reset_index(drop=True)
    price_col = "modal_price_rs_per_qtl"

    # Backward-only autoregressive price lags
    out["price_lag_1"] = out[price_col].shift(1)
    out["price_lag_3"] = out[price_col].shift(3)
    out["price_lag_7"] = out[price_col].shift(7)

    # Strictly backward rolling price statistics over sessions [t-7 .. t-1]
    shifted_price = out[price_col].shift(1)
    out["rolling_mean_7"] = shifted_price.rolling(window=7, min_periods=4).mean()
    out["rolling_std_7"] = shifted_price.rolling(window=7, min_periods=4).std()
    out["rolling_min_7"] = shifted_price.rolling(window=7, min_periods=4).min()
    out["rolling_max_7"] = shifted_price.rolling(window=7, min_periods=4).max()

    # Contemporaneous observable momentum & intra-day auction spread (available at close of t)
    out["price_momentum_7"] = out[price_col] - out[price_col].shift(7)
    out["daily_spread"] = out["max_price_rs_per_qtl"] - out["min_price_rs_per_qtl"]

    # Supply-side features (physical arrivals)
    if has_arrivals and "arrival_quantity_tonnes" in out.columns:
        out["arrival_tonnes"] = out["arrival_quantity_tonnes"].fillna(0.0)
        out["arrival_lag_1"] = out["arrival_tonnes"].shift(1)
        out["rolling_arrival_mean_7"] = (
            out["arrival_tonnes"].shift(1).rolling(window=7, min_periods=4).mean()
        )
    else:
        out["arrival_tonnes"] = 0.0
        out["arrival_lag_1"] = 0.0
        out["rolling_arrival_mean_7"] = 0.0

    # Calendar / monsoon attributes
    out["month"] = out["observation_date"].dt.month
    out["day_of_week"] = out["observation_date"].dt.dayofweek
    out["day_of_year"] = out["observation_date"].dt.dayofyear
    out["is_monsoon_season"] = out["month"].isin([6, 7, 8, 9]).astype(int)

    # Explicit target dates and trading-session gaps
    out["target_date_t_plus_1"] = out["observation_date"].shift(-1)
    out["target_date_t_plus_7"] = out["observation_date"].shift(-7)
    out["lookback_7_gap_days"] = (
        out["observation_date"] - out["observation_date"].shift(7)
    ).dt.days
    out["target_gap_days"] = (
        out["target_date_t_plus_1"] - out["observation_date"]
    ).dt.days

    # Target prices & regime shift labels (never used as model features)
    out["target_price_t_plus_1"] = out[price_col].shift(-1)
    out["target_price_t_plus_7"] = out[price_col].shift(-7)

    out["price_change_pct"] = (
        (out["target_price_t_plus_1"] - out[price_col]) / out[price_col]
    ) * 100.0
    out["is_price_spike"] = (out["price_change_pct"].abs() > 10.0).astype(int)
    out["spike_direction"] = "Normal"
    out.loc[out["price_change_pct"] > 10.0, "spike_direction"] = "Surge"
    out.loc[out["price_change_pct"] < -10.0, "spike_direction"] = "Crash"

    if drop_incomplete:
        required_cols = [
            "modal_price_rs_per_qtl",
            "price_lag_1",
            "price_lag_3",
            "price_lag_7",
            "rolling_mean_7",
            "rolling_std_7",
            "rolling_min_7",
            "rolling_max_7",
            "price_momentum_7",
            "daily_spread",
            "arrival_tonnes",
            "arrival_lag_1",
            "rolling_arrival_mean_7",
            "target_price_t_plus_1",
            "target_date_t_plus_1",
        ]
        out = out.dropna(subset=required_cols).copy()
        if max_target_gap_days is not None:
            out = out[out["target_gap_days"] <= max_target_gap_days].copy()
        if max_lookback_gap_days is not None:
            out = out[out["lookback_7_gap_days"] <= max_lookback_gap_days].copy()

    return out.reset_index(drop=True)


def engineer_multi_series_features(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("commodity_name", "market_name"),
    arrival_col: str = "arrival_quantity_tonnes",
    max_target_gap_days: Optional[int] = 45,
    max_lookback_gap_days: Optional[int] = 60,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Compute lag, rolling, and target features independently per group in `group_cols`.
    Prevents any cross-commodity or cross-market lag leakage.
    """
    frames: List[pd.DataFrame] = []
    for _, group_df in df.groupby(list(group_cols), sort=True):
        has_arr = arrival_col in group_df.columns and group_df[arrival_col].fillna(0).sum() > 0
        feat = engineer_single_series_features(
            group_df,
            has_arrivals=has_arr,
            max_target_gap_days=max_target_gap_days,
            max_lookback_gap_days=max_lookback_gap_days,
            drop_incomplete=drop_incomplete,
        )
        frames.append(feat)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def split_chronologically_with_purging(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    target_date_col: str = "target_date_t_plus_1",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Partition rows chronologically using shared global cutoff dates across all series and purge
    observations whose target observation date crosses into the subsequent partition.
    """
    if df.empty:
        return df.copy(), {}

    work = df.copy()
    work["observation_date"] = pd.to_datetime(work["observation_date"])
    work[target_date_col] = pd.to_datetime(work[target_date_col])
    work = work.sort_values(["observation_date", "commodity_name", "market_name"]).reset_index(drop=True)

    sorted_obs = work["observation_date"].sort_values().reset_index(drop=True)
    n = len(sorted_obs)
    val_start_date = sorted_obs.iloc[min(n - 2, max(1, int(n * train_frac)))]
    test_start_date = sorted_obs.iloc[min(n - 1, max(2, int(n * (train_frac + val_frac))))]

    # Ensure strictly distinct boundary dates even if ties exist
    if test_start_date <= val_start_date:
        later_dates = sorted_obs[sorted_obs > val_start_date]
        if not later_dates.empty:
            test_start_date = later_dates.iloc[0]

    work["split_tag"] = "Test"
    work.loc[work["observation_date"] < val_start_date, "split_tag"] = "Train"
    work.loc[
        (work["observation_date"] >= val_start_date)
        & (work["observation_date"] < test_start_date),
        "split_tag",
    ] = "Validation"

    # Purge observations whose target_date leaks across the split boundary
    purge_train = (work["split_tag"] == "Train") & (work[target_date_col] >= val_start_date)
    purge_val = (work["split_tag"] == "Validation") & (work[target_date_col] >= test_start_date)
    purged_count = int((purge_train | purge_val).sum())
    work = work[~(purge_train | purge_val)].copy().reset_index(drop=True)

    # ISO string formatting for clean CSV persistence
    for dcol in ["observation_date", "target_date_t_plus_1", "target_date_t_plus_7"]:
        if dcol in work.columns:
            work[dcol] = pd.to_datetime(work[dcol], errors="coerce").dt.strftime("%Y-%m-%d")

    train_sub = work[work["split_tag"] == "Train"]
    val_sub = work[work["split_tag"] == "Validation"]
    test_sub = work[work["split_tag"] == "Test"]

    meta = {
        "train_dates": f"{train_sub['observation_date'].min()} to {train_sub['observation_date'].max()}",
        "validation_dates": f"{val_sub['observation_date'].min()} to {val_sub['observation_date'].max()}",
        "test_dates": f"{test_sub['observation_date'].min()} to {test_sub['observation_date'].max()}",
        "purged_boundary_rows": purged_count,
    }
    return work, meta
