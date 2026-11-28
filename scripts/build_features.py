"""
Leak-Free Multi-Commodity Feature Engineering Pipeline.

Builds model-ready datasets for exact canonical (commodity, market) pairs from
data/processed/wholesale_prices.csv and data/processed/market_arrivals.csv.

Guarantees:
- Exact market matching by default (no silent combining of Pune & Pune(Pimpri) or
  Lasalgaon & Lasalgaon(Niphad)).
- Separate lag/rolling calculation per (commodity, market) series.
- Shared chronological cutoff dates with target-date boundary purging across
  Train -> Validation -> Test partitions.

Usage:
  py -3.12 scripts/build_features.py --all
  py -3.12 scripts/build_features.py --unified
  py -3.12 scripts/build_features.py --commodity Tomato --market Pune
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    CANONICAL_SERIES_SPECS,
    COMMODITY_CATEGORY_MAP,
    canonicalize_commodity_name,
    engineer_single_series_features,
    split_chronologically_with_purging,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_wholesale_series(
    commodity: str,
    market: str,
    allow_submarket_aggregation: bool = False,
) -> pd.DataFrame:
    """
    Load wholesale price time series for an exact (commodity, market) pair.

    Args:
        commodity: Commodity name (e.g. 'Tomato', 'Onion', 'Moong Dal', 'Sugar').
        market: Exact APMC market name (e.g. 'Pune', 'Lasalgaon', 'Mumbai').
        allow_submarket_aggregation: Must remain False unless explicitly opting into
            documented submarket aggregation (never silent).
    """
    ws = pd.read_csv(PROCESSED_DIR / "wholesale_prices.csv")
    ws["canonical_commodity"] = ws["commodity_name"].apply(canonicalize_commodity_name)

    target_canon = canonicalize_commodity_name(commodity).lower()
    comm_mask = (
        (ws["canonical_commodity"].str.lower() == target_canon)
        | (ws["commodity_name"].str.lower() == commodity.strip().lower())
    )
    sub = ws[comm_mask].copy()
    if sub.empty:
        raise ValueError(f"No wholesale records found for commodity '{commodity}'.")

    if allow_submarket_aggregation:
        mkt_mask = sub["market_name"].str.contains(market, case=False, na=False)
        sub = sub[mkt_mask].copy()
        explicit_market_label = f"{market}_Submarkets_Aggregated"
    else:
        mkt_mask = sub["market_name"].str.strip().str.lower() == market.strip().lower()
        sub = sub[mkt_mask].copy()
        explicit_market_label = market.strip()

    if sub.empty:
        available = sorted(ws[comm_mask]["market_name"].dropna().unique().tolist())
        raise ValueError(
            f"No exact records for '{commodity}' in market '{market}'. "
            f"Sample available markets: {available[:10]}"
        )

    sub["observation_date"] = pd.to_datetime(sub["observation_date"])
    sub = sub.sort_values("observation_date").reset_index(drop=True)

    # Aggregate within the exact market if multiple varieties reported on same date
    daily = (
        sub.groupby("observation_date", as_index=False)
        .agg(
            {
                "min_price_rs_per_qtl": "min",
                "max_price_rs_per_qtl": "max",
                "modal_price_rs_per_qtl": "median",
                "state": "first",
                "district": "first",
                "price_unit": "first",
                "price_type": "first",
                "data_source": "first",
                "is_anomalous_flag": "max",
            }
        )
    )
    daily["commodity_name"] = canonicalize_commodity_name(commodity)
    daily["market_name"] = explicit_market_label
    daily["is_anomalous_flag"] = daily["is_anomalous_flag"].astype(bool)
    return daily


def load_arrivals(
    commodity: str,
    market: str,
    allow_submarket_aggregation: bool = False,
) -> pd.DataFrame:
    """Load exact market-level arrivals series if present."""
    arrivals_path = PROCESSED_DIR / "market_arrivals.csv"
    if not arrivals_path.exists():
        return pd.DataFrame()

    ma = pd.read_csv(arrivals_path)
    ma["canonical_commodity"] = ma["commodity_name"].apply(canonicalize_commodity_name)
    target_canon = canonicalize_commodity_name(commodity).lower()
    comm_mask = (
        (ma["canonical_commodity"].str.lower() == target_canon)
        | (ma["commodity_name"].str.lower() == commodity.strip().lower())
    )
    sub = ma[comm_mask].copy()
    if sub.empty:
        return pd.DataFrame()

    if allow_submarket_aggregation:
        mkt_mask = sub["market_name"].str.contains(market, case=False, na=False)
    else:
        mkt_mask = sub["market_name"].str.strip().str.lower() == market.strip().lower()
    sub = sub[mkt_mask].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["observation_date"] = pd.to_datetime(sub["observation_date"])
    return (
        sub.groupby("observation_date", as_index=False)["arrival_quantity_tonnes"].sum()
    )


def build_model_ready_dataset(
    commodity: str,
    market: str,
    category: Optional[str] = None,
    is_perishable: Optional[int] = None,
    allow_submarket_aggregation: bool = False,
) -> pd.DataFrame:
    """Build leak-free feature dataset for one canonical (commodity, market) series."""
    prices = load_wholesale_series(
        commodity, market, allow_submarket_aggregation=allow_submarket_aggregation
    )
    arrivals = load_arrivals(
        commodity, market, allow_submarket_aggregation=allow_submarket_aggregation
    )

    has_arrivals = not arrivals.empty
    if has_arrivals:
        prices = prices.merge(arrivals, on="observation_date", how="left")
        prices["arrival_quantity_tonnes"] = prices["arrival_quantity_tonnes"].fillna(0.0)
    else:
        prices["arrival_quantity_tonnes"] = 0.0

    # Restrict strictly to primary continuous historical window where dense daily data exists
    # (prevents single isolated 2026-11-17 API snapshot from creating an 8-year session gap)
    featured = engineer_single_series_features(
        prices,
        has_arrivals=has_arrivals,
        max_target_gap_days=45,
        max_lookback_gap_days=60,
        drop_incomplete=True,
    )

    canon = canonicalize_commodity_name(commodity)
    default_cat, default_perish = COMMODITY_CATEGORY_MAP.get(canon, ("Other", 1))
    featured["commodity_name"] = canon
    featured["category"] = category if category is not None else default_cat
    featured["is_perishable"] = int(is_perishable if is_perishable is not None else default_perish)

    split_df, _ = split_chronologically_with_purging(featured, train_frac=0.70, val_frac=0.15)
    return split_df


def save_dataset(df: pd.DataFrame, commodity: str, market: str) -> Path:
    """Persist per-series feature store CSV."""
    safe_comm = commodity.lower().replace(" ", "_").replace("(", "").replace(")", "")
    safe_mkt = market.lower().replace(" ", "_")
    path = PROCESSED_DIR / f"model_ready_{safe_mkt}_{safe_comm}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}: {len(df)} rows | splits={df['split_tag'].value_counts().to_dict()}")
    return path


def build_unified_dataset() -> Tuple[Path, Path, Dict[str, str]]:
    """
    Build the canonical unified multi-commodity dataset across all 5 validated
    exact single-market series using shared chronological cutoffs and boundary purging.
    """
    print("=" * 72)
    print("BUILDING LEAK-FREE UNIFIED MULTI-COMMODITY FEATURE STORE")
    print("=" * 72)

    series_frames: List[pd.DataFrame] = []
    for spec in CANONICAL_SERIES_SPECS:
        df_series = build_model_ready_dataset(
            commodity=str(spec["commodity_name"]),
            market=str(spec["market_name"]),
            category=str(spec["category"]),
            is_perishable=int(spec["is_perishable"]),
            allow_submarket_aggregation=False,
        )
        # Save exact per-series dataset as well so specialist & unified stay 100% consistent
        save_dataset(df_series, str(spec["commodity_name"]), str(spec["market_name"]))
        series_frames.append(df_series)

    pooled = pd.concat(series_frames, ignore_index=True)

    # Drop any old independent split tags and apply SHARED chronological cutoffs + boundary purging
    unified, split_meta = split_chronologically_with_purging(
        pooled.drop(columns=["split_tag"], errors="ignore"),
        train_frac=0.70,
        val_frac=0.15,
    )

    clean_path = PROCESSED_DIR / "model_ready_unified_commodities_clean.csv"
    legacy_path = PROCESSED_DIR / "model_ready_unified_commodities.csv"
    unified.to_csv(clean_path, index=False)
    unified.to_csv(legacy_path, index=False)

    print(f"\nUnified Feature Store Saved: {clean_path.name} (and synchronized {legacy_path.name})")
    print(f"  Total Rows (after boundary purge): {len(unified)}")
    print(f"  Purged Cross-Boundary Rows       : {split_meta.get('purged_boundary_rows', 0)}")
    print(f"  Train Cutoffs      : {split_meta['train_dates']}")
    print(f"  Validation Cutoffs : {split_meta['validation_dates']}")
    print(f"  Test Cutoffs       : {split_meta['test_dates']}")
    print(f"  Split Counts       : {unified['split_tag'].value_counts().to_dict()}")
    print("=" * 72)
    return clean_path, legacy_path, split_meta


def main():
    parser = argparse.ArgumentParser(description="Leak-free multi-commodity feature builder.")
    parser.add_argument("--commodity", type=str, help="Canonical commodity name (e.g. Tomato)")
    parser.add_argument("--market", type=str, help="Exact market name (e.g. Pune)")
    parser.add_argument("--all", action="store_true", help="Build all canonical series & unified store")
    parser.add_argument("--unified", action="store_true", help="Build unified feature store")
    args = parser.parse_args()

    if args.all or args.unified:
        build_unified_dataset()
    elif args.commodity and args.market:
        df = build_model_ready_dataset(args.commodity, args.market)
        save_dataset(df, args.commodity, args.market)
    else:
        build_unified_dataset()


if __name__ == "__main__":
    main()
