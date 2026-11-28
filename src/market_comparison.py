"""
Market & City Historical Price Comparison Engine (Phase 3).

Uses processed AGMARKNET datasets (`wholesale_prices.csv` and `market_arrivals.csv`)
with exact market matching, explicit variety aggregation on duplicate same-day records,
overlapping-date direct price comparisons, and honest missing-observation reporting.
Never synthesizes artificial prices or claims historical comparison is ML forecasting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from src.features import canonicalize_commodity_name

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _load_wholesale_df(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    base = Path(processed_dir) if processed_dir else PROCESSED_DIR
    ws_path = base / "wholesale_prices.csv"
    if not ws_path.exists():
        raise FileNotFoundError(f"Wholesale dataset not found at {ws_path}")
    df = pd.read_csv(ws_path)
    df["canonical_commodity"] = df["commodity_name"].apply(canonicalize_commodity_name)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["market_name"] = df["market_name"].astype(str).str.strip()
    return df


def _load_arrivals_df(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    base = Path(processed_dir) if processed_dir else PROCESSED_DIR
    arr_path = base / "market_arrivals.csv"
    if not arr_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(arr_path)
    df["canonical_commodity"] = df["commodity_name"].apply(canonicalize_commodity_name)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["market_name"] = df["market_name"].astype(str).str.strip()
    return df


def get_available_commodities(
    min_markets: int = 2,
    min_rows_per_market: int = 30,
    processed_dir: Optional[Path] = None,
) -> List[str]:
    """Return canonical commodities having at least `min_markets` exact markets with >= `min_rows_per_market` rows."""
    ws = _load_wholesale_df(processed_dir)
    counts = ws.groupby(["canonical_commodity", "market_name"]).size().reset_index(name="cnt")
    viable = counts[counts["cnt"] >= min_rows_per_market]
    mkt_per_comm = viable.groupby("canonical_commodity")["market_name"].nunique()
    eligible = sorted(mkt_per_comm[mkt_per_comm >= min_markets].index.tolist())
    # Prioritize signature commodities first
    priority = [c for c in ["Tomato", "Onion", "Moong Dal", "Sugar", "Potato"] if c in eligible]
    rest = [c for c in eligible if c not in priority]
    return priority + rest


def get_markets_for_commodity(
    commodity: str,
    min_rows: int = 10,
    processed_dir: Optional[Path] = None,
) -> List[str]:
    """Return exact sorted `market_name` values for a commodity with >= `min_rows` records."""
    ws = _load_wholesale_df(processed_dir)
    canon = canonicalize_commodity_name(commodity).lower()
    sub = ws[ws["canonical_commodity"].str.lower() == canon]
    vc = sub["market_name"].value_counts()
    markets = vc[vc >= min_rows].index.tolist()
    # Sort alphabetically while placing major benchmarks cleanly at top if present
    preferred = [m for m in ["Pune", "Lasalgaon", "Mumbai", "Nagpur", "Nashik"] if m in markets]
    others = sorted([m for m in markets if m not in preferred])
    return preferred + others


def compare_markets(
    commodity: str,
    markets: Sequence[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    processed_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compare historical wholesale prices & arrivals across 2+ exact markets for one commodity.

    Calculates direct inter-market spreads ONLY on overlapping trading dates without imputation.
    """
    if not markets or len(markets) < 2:
        raise ValueError("Select at least two distinct markets for comparison.")

    unique_markets = list(dict.fromkeys([m.strip() for m in markets if m and m.strip()]))
    if len(unique_markets) < 2:
        raise ValueError("Selected markets must contain at least two distinct names.")

    ws = _load_wholesale_df(processed_dir)
    arr = _load_arrivals_df(processed_dir)
    canon = canonicalize_commodity_name(commodity)

    ws_c = ws[ws["canonical_commodity"].str.lower() == canon.lower()].copy()
    if ws_c.empty:
        raise ValueError(f"No wholesale records found for commodity '{commodity}'.")

    if start_date is not None:
        sd = pd.to_datetime(start_date)
        ws_c = ws_c[ws_c["observation_date"] >= sd]
        if not arr.empty:
            arr = arr[arr["observation_date"] >= sd]
    if end_date is not None:
        ed = pd.to_datetime(end_date)
        ws_c = ws_c[ws_c["observation_date"] <= ed]
        if not arr.empty:
            arr = arr[arr["observation_date"] <= ed]

    series_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    arrivals_frames: List[pd.DataFrame] = []
    all_sources = set()

    for mkt in unique_markets:
        sub = ws_c[ws_c["market_name"] == mkt].copy()
        if sub.empty:
            summary_rows.append(
                {
                    "Market": mkt,
                    "District": "N/A",
                    "Trading_Sessions": 0,
                    "Raw_Records": 0,
                    "First_Date": None,
                    "Last_Date": None,
                    "Missing_MonSat_Sessions": None,
                    "Coverage_Pct": 0.0,
                    "Mean_Modal_Rs_Qtl": None,
                    "Min_Modal_Rs_Qtl": None,
                    "Max_Modal_Rs_Qtl": None,
                    "Std_Dev_Rs_Qtl": None,
                    "CV_Volatility_Pct": None,
                    "Arrival_Sessions": 0,
                    "Mean_Arrival_Tonnes": None,
                    "Data_Sources": "None in range",
                }
            )
            continue

        all_sources.update(sub["data_source"].dropna().unique().tolist())
        raw_count = int(len(sub))
        district = str(sub["district"].iloc[0])

        # Explicit daily variety aggregation strictly within this single exact market
        daily = (
            sub.groupby("observation_date", as_index=False)
            .agg(
                min_price_rs_per_qtl=("min_price_rs_per_qtl", "min"),
                max_price_rs_per_qtl=("max_price_rs_per_qtl", "max"),
                modal_price_rs_per_qtl=("modal_price_rs_per_qtl", "median"),
                data_source=("data_source", "first"),
            )
            .sort_values("observation_date")
        )
        daily["market_name"] = mkt
        daily["commodity_name"] = canon

        # Merge arrivals strictly for same (commodity, exact market, date)
        arr_cnt = 0
        mean_arr = None
        if not arr.empty:
            sub_arr = arr[
                (arr["canonical_commodity"].str.lower() == canon.lower())
                & (arr["market_name"] == mkt)
            ]
            if not sub_arr.empty:
                daily_arr = sub_arr.groupby("observation_date", as_index=False)[
                    "arrival_quantity_tonnes"
                ].sum()
                daily = daily.merge(daily_arr, on="observation_date", how="left")
                arr_cnt = int(daily["arrival_quantity_tonnes"].notna().sum())
                if arr_cnt > 0:
                    mean_arr = round(float(daily["arrival_quantity_tonnes"].mean()), 2)
                    arr_sub = daily[["observation_date", "market_name", "arrival_quantity_tonnes"]].dropna()
                    arrivals_frames.append(arr_sub)

        dmin = daily["observation_date"].min()
        dmax = daily["observation_date"].max()
        # Benchmark calendar Mon-Sat trading days in observed span
        expected_bdays = int((pd.date_range(dmin, dmax, freq="D").dayofweek < 6).sum())
        sessions = int(len(daily))
        missing_est = max(0, expected_bdays - sessions)
        cov_pct = round((sessions / expected_bdays) * 100.0, 1) if expected_bdays > 0 else 0.0

        prices = daily["modal_price_rs_per_qtl"].dropna()
        mean_p = float(prices.mean())
        std_p = float(prices.std()) if len(prices) > 1 else 0.0
        cv_p = (std_p / mean_p * 100.0) if mean_p > 0 else 0.0

        summary_rows.append(
            {
                "Market": mkt,
                "District": district,
                "Trading_Sessions": sessions,
                "Raw_Records": raw_count,
                "First_Date": dmin.strftime("%Y-%m-%d"),
                "Last_Date": dmax.strftime("%Y-%m-%d"),
                "Missing_MonSat_Sessions": missing_est,
                "Coverage_Pct": cov_pct,
                "Mean_Modal_Rs_Qtl": round(mean_p, 2),
                "Min_Modal_Rs_Qtl": round(float(prices.min()), 2),
                "Max_Modal_Rs_Qtl": round(float(prices.max()), 2),
                "Std_Dev_Rs_Qtl": round(std_p, 2),
                "CV_Volatility_Pct": round(cv_p, 2),
                "Arrival_Sessions": arr_cnt,
                "Mean_Arrival_Tonnes": mean_arr,
                "Data_Sources": "; ".join(sorted(sub["data_source"].dropna().unique().tolist())),
            }
        )
        series_frames.append(daily)

    combined_ts = pd.concat(series_frames, ignore_index=True) if series_frames else pd.DataFrame()
    combined_arr = pd.concat(arrivals_frames, ignore_index=True) if arrivals_frames else pd.DataFrame()

    # Pairwise comparison STRICTLY on exact overlapping observation dates
    overlap_rows: List[Dict[str, Any]] = []
    if len(series_frames) >= 2:
        for i in range(len(series_frames)):
            for j in range(i + 1, len(series_frames)):
                df_a = series_frames[i]
                df_b = series_frames[j]
                mkt_a = str(df_a["market_name"].iloc[0])
                mkt_b = str(df_b["market_name"].iloc[0])
                merged = pd.merge(
                    df_a[["observation_date", "modal_price_rs_per_qtl"]],
                    df_b[["observation_date", "modal_price_rs_per_qtl"]],
                    on="observation_date",
                    suffixes=("_a", "_b"),
                    how="inner",
                )
                n_ov = int(len(merged))
                if n_ov > 0:
                    diff = merged["modal_price_rs_per_qtl_a"] - merged["modal_price_rs_per_qtl_b"]
                    corr = (
                        round(float(merged["modal_price_rs_per_qtl_a"].corr(merged["modal_price_rs_per_qtl_b"])), 4)
                        if n_ov >= 3 and merged["modal_price_rs_per_qtl_a"].std() > 0 and merged["modal_price_rs_per_qtl_b"].std() > 0
                        else None
                    )
                    overlap_rows.append(
                        {
                            "Market_A": mkt_a,
                            "Market_B": mkt_b,
                            "Overlapping_Dates": n_ov,
                            "Overlap_Span": f"{merged['observation_date'].min().strftime('%Y-%m-%d')} to {merged['observation_date'].max().strftime('%Y-%m-%d')}",
                            "Mean_Diff_A_Minus_B_Rs_Qtl": round(float(diff.mean()), 2),
                            "Mean_Abs_Spread_Rs_Qtl": round(float(diff.abs().mean()), 2),
                            "Pearson_Correlation": corr,
                        }
                    )
                else:
                    overlap_rows.append(
                        {
                            "Market_A": mkt_a,
                            "Market_B": mkt_b,
                            "Overlapping_Dates": 0,
                            "Overlap_Span": "No overlapping trading dates",
                            "Mean_Diff_A_Minus_B_Rs_Qtl": None,
                            "Mean_Abs_Spread_Rs_Qtl": None,
                            "Pearson_Correlation": None,
                        }
                    )

    return {
        "commodity": canon,
        "markets_requested": unique_markets,
        "summary_df": pd.DataFrame(summary_rows),
        "timeseries_df": combined_ts,
        "arrivals_df": combined_arr,
        "overlapping_comparison_df": pd.DataFrame(overlap_rows),
        "sources": sorted(all_sources),
        "last_available_date": (
            combined_ts["observation_date"].max().strftime("%Y-%m-%d")
            if not combined_ts.empty
            else None
        ),
        "disclaimer": (
            "Descriptive historical APMC comparison in Rs/Quintal. Direct differences and correlations "
            "are computed strictly on overlapping dates without artificial price interpolation. "
            "Historical comparison is NOT multi-city ML price forecasting."
        ),
    }
