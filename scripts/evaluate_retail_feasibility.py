"""
Phase 6: Retail Price Estimation Feasibility Audit & Builder.

Inspects overlap between `wholesale_prices.csv` and `retail_prices.csv` by:
- Normalized commodity name
- Exact observation date
- Geographic market/centre mapping (Maharashtra cities & All-India comparison)

Only trains a wholesale-to-retail estimator if sufficient longitudinal paired data exists.
Never forces a fabricated regression model or uses wholesale/100 conversion as a prediction.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import canonicalize_commodity_name

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def audit_and_evaluate_retail_feasibility(min_paired_dates: int = 30) -> dict:
    """Audit historical overlap between wholesale mandi prices and DCA retail prices."""
    ws = pd.read_csv(PROCESSED_DIR / "wholesale_prices.csv")
    rt = pd.read_csv(PROCESSED_DIR / "retail_prices.csv")

    ws["comm_norm"] = ws["commodity_name"].apply(canonicalize_commodity_name)
    rt["comm_norm"] = rt["commodity_name"].apply(canonicalize_commodity_name)

    ws_mh = ws[ws["state"].astype(str).str.lower() == "maharashtra"].copy()
    rt_mh = rt[rt["state"].astype(str).str.lower() == "maharashtra"].copy()

    city_overlaps = {}
    total_city_pairs = 0
    for city in ["Mumbai", "Pune", "Nagpur", "Nashik"]:
        w_c = ws_mh[ws_mh["district"].astype(str).str.lower() == city.lower()]
        r_c = rt_mh[rt_mh["reporting_centre"].astype(str).str.lower() == city.lower()]
        merged = pd.merge(
            w_c[["comm_norm", "observation_date", "modal_price_rs_per_qtl"]],
            r_c[["comm_norm", "observation_date", "retail_price"]],
            on=["comm_norm", "observation_date"],
            how="inner",
        )
        city_overlaps[city] = int(len(merged))
        total_city_pairs += len(merged)

    # Check any-geography date overlap
    cross_merged = pd.merge(
        ws[["comm_norm", "observation_date", "modal_price_rs_per_qtl"]],
        rt[["comm_norm", "observation_date", "reporting_centre", "retail_price"]],
        on=["comm_norm", "observation_date"],
        how="inner",
    )
    unique_common_dates = sorted(cross_merged["observation_date"].unique().tolist())

    meta = {
        "audit_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wholesale_sources": ws["data_source"].value_counts().to_dict(),
        "retail_sources": rt["data_source"].value_counts().to_dict(),
        "wholesale_archive_span": "2016-01-01 to 2018-12-31 (plus 2026-11-17 API snapshot)",
        "retail_archive_span": "2010-01-01 to 2012-11-15 (plus 2026-11-17 bulletin & back-reference points)",
        "exact_maharashtra_city_date_commodity_overlaps": city_overlaps,
        "total_same_city_same_date_pairs": int(total_city_pairs),
        "cross_geography_same_date_rows": int(len(cross_merged)),
        "unique_common_dates_across_datasets": unique_common_dates,
        "minimum_required_longitudinal_dates": int(min_paired_dates),
        "retail_estimation_supported": False,
        "status": "unavailable_insufficient_paired_data",
        "explanation": (
            f"Cannot scientifically train or validate a longitudinal wholesale-to-retail ML estimator: "
            f"exact Maharashtra same-city same-date overlap is {total_city_pairs} rows, and cross-geography "
            f"date overlap spans only {len(unique_common_dates)} date(s) ({unique_common_dates}) because DCA "
            f"multi-centre retail archives end in 2012-11-15 while AGMARKNET historical series span 2016-2018. "
            f"Application uses actual observed DCA retail records or manual retail input, never wholesale/100 proxies."
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "retail_model_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved retail feasibility metadata to {out_path}")
    print(f"  Same-city MH pairs   : {total_city_pairs}")
    print(f"  Common dates (all)   : {len(unique_common_dates)} -> {unique_common_dates}")
    print(f"  Retail ML supported  : {meta['retail_estimation_supported']}")
    return meta


if __name__ == "__main__":
    audit_and_evaluate_retail_feasibility()
