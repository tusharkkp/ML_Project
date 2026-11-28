"""
Canonical Leak-Free Time-Series Feature Engineering, Normalization & Purged Splitting Engine.

Schema Version: 3.1.0-parity
Shared identically across:
- Offline training dataset preparation (scripts/build_features.py)
- Live & historical feature construction (src/live_data.py)
- Production & exploratory inference (src/inference.py, src/schemas.py)

Critical Invariants:
1. Zero future leakage: P(t) = `modal_price_rs_per_qtl`; previous-window rolling stats
   operate strictly over [t-7 .. t-1] via `.shift(1).rolling(...)` (excluding t).
2. Unified momentum definition: `price_momentum_7` is ALWAYS percentage change (%) over
   the last 7 sessions: `((P(t) - P(t-7)) / P(t-7)) * 100.0` via `compute_price_momentum_7_pct()`.
3. Exact canonical market matching: normalized alias matching only (never arbitrary substring
   checks that merge submarkets such as Pune(Pimpri) or Lasalgaon(Niphad)).
4. Honest missingness: unavailable arrival measurements stay NaN rather than silent 0.0.
5. Date-grouped target-purged cross-validation via `DatePurgedWalkForwardCV`.
"""

import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FEATURE_SCHEMA_VERSION = "3.1.0-parity"
DATASET_VERSION = "v3.1.0-parity"

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

FEATURE_UNITS: Dict[str, str] = {
    "modal_price_rs_per_qtl": "Rs/Qtl",
    "price_lag_1": "Rs/Qtl",
    "price_lag_3": "Rs/Qtl",
    "price_lag_7": "Rs/Qtl",
    "rolling_mean_7": "Rs/Qtl",
    "rolling_std_7": "Rs/Qtl",
    "rolling_min_7": "Rs/Qtl",
    "rolling_max_7": "Rs/Qtl",
    "price_momentum_7": "%",
    "daily_spread": "Rs/Qtl",
    "arrival_tonnes": "Tonnes",
    "arrival_lag_1": "Tonnes",
    "rolling_arrival_mean_7": "Tonnes",
    "month": "Integer (1-12)",
    "day_of_week": "Integer (0-6)",
    "is_monsoon_season": "Binary {0,1}",
    "is_perishable": "Binary {0,1}",
}

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

# Exact normalized market alias table (prevents submarket leakage like Pune(Pimpri) or Lasalgaon(Niphad))
CANONICAL_MARKET_ALIASES: Dict[str, set] = {
    "pune": {"pune", "apmc pune", "pune apmc"},
    "lasalgaon": {"lasalgaon", "apmc lasalgaon", "lasalgaon apmc"},
    "mumbai": {"mumbai", "apmc mumbai", "mumbai apmc"},
}

CANONICAL_MARKET_PROPER_NAMES: Dict[str, str] = {
    "pune": "Pune",
    "lasalgaon": "Lasalgaon",
    "mumbai": "Mumbai",
}


def canonicalize_commodity_name(raw_name: str) -> str:
    """Map source-specific commodity labels to canonical names without altering unknown items."""
    text = re.sub(r"\s+", " ", str(raw_name or "").strip())
    if text.lower() in {"green gram dal (moong dal)", "moong dal"}:
        return "Moong Dal"
    return text


def normalize_market_token(raw_market: str) -> str:
    """Normalize whitespace, case, and APMC punctuation without dropping submarket qualifiers."""
    text = str(raw_market or "").strip().lower()
    text = text.replace("a.p.m.c.", "apmc").replace("a.p.m.c", "apmc")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\(\s*", "(", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    return text.strip()


def canonicalize_market_name(raw_market: str) -> str:
    """Resolve exact canonical market name when it matches curated exact aliases; preserve unknown/submarket tokens."""
    norm = normalize_market_token(raw_market)
    for canon_lower, aliases in CANONICAL_MARKET_ALIASES.items():
        if norm in aliases:
            return CANONICAL_MARKET_PROPER_NAMES[canon_lower]
    return re.sub(r"\s+", " ", str(raw_market or "").strip())


def match_canonical_market(raw_market: str, expected_market: str) -> Optional[str]:
    """
    Match `raw_market` strictly via exact normalized alias equality against `expected_market`.
    Returns canonical market name on match, else None (never performs substring matching).
    """
    norm_raw = normalize_market_token(raw_market)
    norm_exp = normalize_market_token(expected_market)
    if not norm_raw or not norm_exp:
        return None

    exp_canon = canonicalize_market_name(expected_market)
    exp_key = exp_canon.lower()
    allowed = CANONICAL_MARKET_ALIASES.get(exp_key, {norm_exp})
    if norm_raw in allowed:
        return exp_canon
    return None


def compute_price_momentum_7_pct(current_price: Any, price_lag_7: Any) -> Any:
    """
    Canonical 7-session price momentum (%) shared identically by training and inference:
        ((P(t) - P(t-7)) / P(t-7)) * 100.0
    Supports scalars and pandas Series / numpy arrays; returns NaN where P(t-7) <= 0.
    """
    if isinstance(current_price, pd.Series) or isinstance(price_lag_7, pd.Series):
        cur_s = pd.to_numeric(current_price, errors="coerce")
        lag_s = pd.to_numeric(price_lag_7, errors="coerce")
        valid = lag_s > 0
        out = pd.Series(np.nan, index=cur_s.index, dtype=float)
        out.loc[valid] = ((cur_s.loc[valid] - lag_s.loc[valid]) / lag_s.loc[valid]) * 100.0
        return out
    cur_f = float(current_price)
    lag_f = float(price_lag_7)
    if not (np.isfinite(cur_f) and np.isfinite(lag_f) and lag_f > 0):
        return float("nan")
    return ((cur_f - lag_f) / lag_f) * 100.0


def compute_previous_window_price_stats(prior_prices: Sequence[float]) -> Dict[str, float]:
    """
    Compute rolling mean, std (ddof=1), min, and max from strictly prior session prices
    (used identically for rolling checks and experimental manual exploration).
    """
    arr = np.asarray([float(x) for x in prior_prices], dtype=float)
    if len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("prior_prices must contain at least one finite value.")
    return {
        "rolling_mean_7": float(np.mean(arr)),
        "rolling_std_7": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "rolling_min_7": float(np.min(arr)),
        "rolling_max_7": float(np.max(arr)),
    }


def engineer_single_series_features(
    df: pd.DataFrame,
    has_arrivals: bool = True,
    max_target_gap_days: Optional[int] = 45,
    max_lookback_gap_days: Optional[int] = 60,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Compute backward-looking features and forward targets for a SINGLE exact (commodity, market) series.

    Guarantees:
    - Sorts strictly by ascending `observation_date` and rejects duplicate dates.
    - `modal_price_rs_per_qtl` is P(t); `price_lag_1` is strictly P(t-1), never P(t).
    - Rolling price statistics use `.shift(1).rolling(7, min_periods=4)` strictly over [t-7 .. t-1].
    - `price_momentum_7` uses `compute_price_momentum_7_pct()` in % identically to serving.
    - Unavailable arrival measurements remain NaN (never silently zero-filled).
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
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out["min_price_rs_per_qtl"] = pd.to_numeric(out["min_price_rs_per_qtl"], errors="coerce")
    out["max_price_rs_per_qtl"] = pd.to_numeric(out["max_price_rs_per_qtl"], errors="coerce")

    # Backward-only autoregressive price lags
    out["price_lag_1"] = out[price_col].shift(1)
    out["price_lag_3"] = out[price_col].shift(3)
    out["price_lag_7"] = out[price_col].shift(7)

    # Strictly backward rolling price statistics over previous 7 active sessions [t-7 .. t-1]
    shifted_price = out[price_col].shift(1)
    out["rolling_mean_7"] = shifted_price.rolling(window=7, min_periods=4).mean()
    out["rolling_std_7"] = shifted_price.rolling(window=7, min_periods=4).std()
    out["rolling_min_7"] = shifted_price.rolling(window=7, min_periods=4).min()
    out["rolling_max_7"] = shifted_price.rolling(window=7, min_periods=4).max()

    # Canonical 7-session percentage momentum (%) + intraday spread (Rs/Qtl)
    out["price_momentum_7"] = compute_price_momentum_7_pct(out[price_col], out["price_lag_7"])
    out["daily_spread"] = out["max_price_rs_per_qtl"] - out["min_price_rs_per_qtl"]

    # Supply-side features: preserve missingness (NaN remains NaN, genuine 0 remains 0)
    if has_arrivals and "arrival_quantity_tonnes" in out.columns:
        out["arrival_tonnes"] = pd.to_numeric(out["arrival_quantity_tonnes"], errors="coerce")
        out["arrival_lag_1"] = out["arrival_tonnes"].shift(1)
        out["rolling_arrival_mean_7"] = (
            out["arrival_tonnes"].shift(1).rolling(window=7, min_periods=4).mean()
        )
    else:
        out["arrival_tonnes"] = np.nan
        out["arrival_lag_1"] = np.nan
        out["rolling_arrival_mean_7"] = np.nan

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
        has_arr = arrival_col in group_df.columns and group_df[arrival_col].notna().any()
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
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

    unique_dates = pd.Series(work["observation_date"].sort_values().unique())
    n_dates = len(unique_dates)
    if n_dates < 4:
        raise ValueError(f"Need at least 4 unique dates to create Train/Validation/Test splits, got {n_dates}.")

    val_idx = min(n_dates - 2, max(1, int(n_dates * train_frac)))
    test_idx = min(n_dates - 1, max(val_idx + 1, int(n_dates * (train_frac + val_frac))))
    val_start_date = unique_dates.iloc[val_idx]
    test_start_date = unique_dates.iloc[test_idx]

    work["split_tag"] = "Test"
    work.loc[work["observation_date"] < val_start_date, "split_tag"] = "Train"
    work.loc[
        (work["observation_date"] >= val_start_date)
        & (work["observation_date"] < test_start_date),
        "split_tag",
    ] = "Validation"

    purge_train = (work["split_tag"] == "Train") & (work[target_date_col] >= val_start_date)
    purge_val = (work["split_tag"] == "Validation") & (work[target_date_col] >= test_start_date)
    purged_count = int((purge_train | purge_val).sum())
    work = work[~(purge_train | purge_val)].copy().reset_index(drop=True)

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
        "schema_version": FEATURE_SCHEMA_VERSION,
    }
    return work, meta


class DatePurgedWalkForwardCV:
    """
    Reusable date-grouped, target-boundary-purged walk-forward cross-validation splitter.

    Guarantees per fold:
    1. Works strictly on unique sorted observation dates (all commodity-market rows sharing
       the same observation date stay together in the exact same fold partition).
    2. `max(obs_date[train_idx]) < min(obs_date[val_idx])` strictly.
    3. Any training sample whose `target_date >= min(obs_date[val_idx])` is purged.
    """

    def __init__(
        self,
        observation_dates: Sequence[Any],
        target_dates: Sequence[Any],
        n_splits: int = 5,
    ):
        self.obs_dates = pd.to_datetime(pd.Series(list(observation_dates))).reset_index(drop=True)
        self.tgt_dates = pd.to_datetime(pd.Series(list(target_dates))).reset_index(drop=True)
        if len(self.obs_dates) != len(self.tgt_dates):
            raise ValueError("observation_dates and target_dates must have equal length.")
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        self.n_splits = int(n_splits)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return len(list(self.split(X, y, groups)))

    def split(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if X is not None and len(X) != len(self.obs_dates):
            raise ValueError(
                f"DatePurgedWalkForwardCV expected {len(self.obs_dates)} rows, received {len(X)}."
            )

        unique_dates = np.sort(self.obs_dates.dropna().unique())
        n_dates = len(unique_dates)
        effective_splits = min(self.n_splits, max(1, n_dates // 2 - 1))
        if effective_splits < 1:
            raise ValueError("Not enough unique observation dates for walk-forward CV.")

        fold_blocks = np.array_split(unique_dates, effective_splits + 1)
        for fold_idx in range(1, len(fold_blocks)):
            val_dates = fold_blocks[fold_idx]
            if len(val_dates) == 0:
                continue
            val_start = pd.Timestamp(val_dates[0])
            val_end = pd.Timestamp(val_dates[-1])

            train_mask = (
                (self.obs_dates < val_start)
                & self.tgt_dates.notna()
                & (self.tgt_dates < val_start)
            )
            val_mask = (self.obs_dates >= val_start) & (self.obs_dates <= val_end)

            train_idx = np.flatnonzero(train_mask.to_numpy())
            val_idx = np.flatnonzero(val_mask.to_numpy())
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
