"""
Official Data Acquisition, Cache Management & Live-to-Feature Parity Module.

Version: 3.1.0-parity (Phases 3, 4, 5)
Integrates:
1. data.gov.in AGMARKNET API (`resource/9ef84268-d588-465a-a308-a864a43d0070`)
2. DCA Daily Retail Bulletin (`fcainfoweb.nic.in`) + processed `retail_prices.csv`
3. Canonical local history (`wholesale_prices.csv`, `market_arrivals.csv`) via exact
   `src/features.engineer_single_series_features()` parity.

Scientific Guarantees:
- Exact normalized alias equality for markets (never merges submarkets like Pune(Pimpri)
  or Lasalgaon(Niphad) into exact Pune / Lasalgaon via substring matching).
- Rejects ambiguous market payloads (mixed canonical targets / conflicting districts).
- Preserves unavailable API arrival telemetry as NaN/None (never fabricates 0.0) and
  withholds automated Mode A inference whenever genuine arrivals or continuous 7-day
  lags are unavailable.
- Never fabricates fallback retail defaults (returns None/"Unavailable" when absent).
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import urllib3

from src.features import (
    FEATURE_SCHEMA_VERSION,
    SPECIALIST_NUMERIC_FEATURES,
    canonicalize_commodity_name,
    canonicalize_market_name,
    engineer_single_series_features,
    match_canonical_market,
    normalize_market_token,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_PATH = PROJECT_ROOT / "data" / "metadata" / "live_fetch_cache.json"

DATA_GOV_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
DCA_URL = "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx"

COMMODITY_API_ALIASES: Dict[str, List[str]] = {
    "Tomato": ["Tomato"],
    "Onion": ["Onion"],
    "Sugar": ["Sugar"],
    "Moong Dal": ["Green Gram Dal (Moong Dal)", "Moong Dal"],
}


def _read_cache() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _write_cache(cache_data: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)


def _parse_api_date(raw_val: Any) -> Optional[pd.Timestamp]:
    if raw_val is None or str(raw_val).strip() == "":
        return None
    text = str(raw_val).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _extract_genuine_arrival_tonnes(record: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Extract arrival volume strictly when explicit known field & known unit exist.
    Never sets missing arrivals to 0.0; unknown units remain NaN.
    """
    tonnes_keys = ("arrival_quantity_tonnes", "arrivals_tonnes", "arrival_tonnes")
    qtl_keys = ("arrival_quantity_quintal", "arrivals_quintal", "arrival_quintals")

    for k in tonnes_keys:
        if k in record and record[k] not in (None, "", "--", "NA", "null"):
            try:
                val = float(record[k])
                if math.isfinite(val) and val >= 0.0:
                    return val, "Tonnes"
            except (TypeError, ValueError):
                return float("nan"), None

    for k in qtl_keys:
        if k in record and record[k] not in (None, "", "--", "NA", "null"):
            try:
                val = float(record[k])
                if math.isfinite(val) and val >= 0.0:
                    return val / 10.0, "Tonnes (converted from explicit Quintal field)"
            except (TypeError, ValueError):
                return float("nan"), None

    return float("nan"), None


def parse_and_filter_api_records(
    records: List[Dict[str, Any]],
    target_commodity: str,
    target_market: str,
) -> Dict[str, Any]:
    """
    Deterministically filter and validate raw `data.gov.in` records for exact commodity and market.

    Enforces:
    - Exact normalized market alias matching (`match_canonical_market` — NO substring match).
    - Rejection of delimited multi-market strings (`/` or `,` inside market name) and
      conflicting district tags for the same market/date.
    - Preservation of genuine `NaN` arrivals when unavailable in payload.
    - Variety-level aggregation strictly within `(canonical_commodity, canonical_market, date)`.
    """
    canon_comm = canonicalize_commodity_name(target_commodity)
    canon_mkt = canonicalize_market_name(target_market)
    allowed_comms = {canon_comm.lower()} | {c.lower() for c in COMMODITY_API_ALIASES.get(canon_comm, [])}

    accepted_rows: List[Dict[str, Any]] = []
    rejected_submarkets: List[str] = []
    invalid_records_skipped = 0

    for r in records:
        r_comm_raw = str(r.get("commodity", "")).strip()
        r_comm_canon = canonicalize_commodity_name(r_comm_raw).lower()
        if r_comm_raw.lower() not in allowed_comms and r_comm_canon not in allowed_comms:
            continue

        raw_mkt = str(r.get("market", "")).strip()
        # Reject composite market identifiers explicitly
        if any(sep in raw_mkt for sep in ("/", "&", ";")):
            rejected_submarkets.append(raw_mkt)
            continue

        matched_canon = match_canonical_market(raw_mkt, canon_mkt)
        if matched_canon is None:
            norm_tok = normalize_market_token(raw_mkt)
            if canon_mkt.lower() in norm_tok:
                rejected_submarkets.append(raw_mkt)
            continue

        dt = _parse_api_date(r.get("arrival_date", ""))
        try:
            p_mod = float(r.get("modal_price", ""))
            p_min = float(r.get("min_price", p_mod) if r.get("min_price") not in (None, "") else p_mod)
            p_max = float(r.get("max_price", p_mod) if r.get("max_price") not in (None, "") else p_mod)
        except (TypeError, ValueError):
            invalid_records_skipped += 1
            continue

        if dt is None or not math.isfinite(p_mod) or p_mod <= 0:
            invalid_records_skipped += 1
            continue
        if not math.isfinite(p_min) or p_min <= 0:
            p_min = p_mod
        if not math.isfinite(p_max) or p_max <= 0:
            p_max = p_mod
        if p_min > p_max:
            p_min, p_max = p_max, p_min

        arr_val, arr_unit = _extract_genuine_arrival_tonnes(r)

        accepted_rows.append(
            {
                "observation_date": dt,
                "canonical_commodity": canon_comm,
                "canonical_market": matched_canon,
                "raw_market": raw_mkt,
                "district": str(r.get("district", "")).strip(),
                "state": str(r.get("state", "")).strip(),
                "variety": str(r.get("variety", "Other")).strip(),
                "min_price_rs_per_qtl": p_min,
                "max_price_rs_per_qtl": p_max,
                "modal_price_rs_per_qtl": p_mod,
                "arrival_quantity_tonnes": arr_val,
                "arrival_unit": arr_unit,
            }
        )

    if not accepted_rows:
        return {
            "status": "no_matching_records",
            "daily_df": pd.DataFrame(),
            "accepted_rows": [],
            "rejected_submarkets": sorted(set(rejected_submarkets)),
            "invalid_records_skipped": invalid_records_skipped,
        }

    df_acc = pd.DataFrame(accepted_rows)

    # Check ambiguity: contradictory non-empty districts mapped under the same canonical market on a single date
    ambiguous_districts = sorted({d.lower() for d in df_acc["district"] if d})
    if len(ambiguous_districts) > 1:
        return {
            "status": "ambiguous_market_match",
            "daily_df": pd.DataFrame(),
            "accepted_rows": accepted_rows,
            "rejected_submarkets": sorted(set(rejected_submarkets)),
            "ambiguous_districts": sorted({d for d in df_acc["district"] if d}),
            "invalid_records_skipped": invalid_records_skipped,
        }

    # Log contributing raw API records transparently
    for rec in accepted_rows:
        logger.info(
            "Accepted API record: commodity=%s raw_market=%r canonical=%s date=%s variety=%s modal=%.2f arrivals=%s",
            canon_comm,
            rec["raw_market"],
            rec["canonical_market"],
            rec["observation_date"].strftime("%Y-%m-%d"),
            rec["variety"],
            rec["modal_price_rs_per_qtl"],
            rec["arrival_quantity_tonnes"],
        )

    # Aggregate varieties ONLY within the single exact canonical market & observation date
    # `sum(min_count=1)` guarantees genuine all-NaN arrival dates stay NaN (NEVER 0.0!)
    daily_df = (
        df_acc.groupby("observation_date", as_index=False)
        .agg(
            min_price_rs_per_qtl=("min_price_rs_per_qtl", "min"),
            max_price_rs_per_qtl=("max_price_rs_per_qtl", "max"),
            modal_price_rs_per_qtl=("modal_price_rs_per_qtl", "median"),
            arrival_quantity_tonnes=("arrival_quantity_tonnes", lambda s: s.sum(min_count=1)),
            raw_market=("raw_market", lambda s: "; ".join(sorted(set(s)))),
            canonical_market=("canonical_market", "first"),
            varieties_count=("variety", "nunique"),
        )
        .sort_values("observation_date")
        .reset_index(drop=True)
    )

    arr_units = [u for u in df_acc["arrival_unit"].dropna().unique().tolist() if u]
    return {
        "status": "ok",
        "daily_df": daily_df,
        "accepted_rows": accepted_rows,
        "rejected_submarkets": sorted(set(rejected_submarkets)),
        "raw_api_markets": sorted(set(df_acc["raw_market"].tolist())),
        "arrival_units_detected": arr_units,
        "invalid_records_skipped": invalid_records_skipped,
    }


def _load_local_exact_history(
    commodity: str, market: str, max_obs_date: Optional[str] = "2019-12-31"
) -> pd.DataFrame:
    """Load local processed wholesale + genuine arrivals series for an exact (commodity, market)."""
    ws = pd.read_csv(PROCESSED_DIR / "wholesale_prices.csv")
    ws["canonical_commodity"] = ws["commodity_name"].apply(canonicalize_commodity_name)
    canon = canonicalize_commodity_name(commodity).lower()
    canon_mkt = canonicalize_market_name(market)

    sub = ws[
        (ws["canonical_commodity"].str.lower() == canon)
        & (ws["market_name"].apply(lambda m: match_canonical_market(str(m), canon_mkt) is not None))
    ].copy()
    sub["observation_date"] = pd.to_datetime(sub["observation_date"])
    if max_obs_date:
        sub = sub[sub["observation_date"] <= pd.to_datetime(max_obs_date)].copy()

    if sub.empty:
        return pd.DataFrame()

    daily = (
        sub.groupby("observation_date", as_index=False)
        .agg(
            {
                "min_price_rs_per_qtl": "min",
                "max_price_rs_per_qtl": "max",
                "modal_price_rs_per_qtl": "median",
                "data_source": "first",
            }
        )
        .sort_values("observation_date")
    )

    arr_path = PROCESSED_DIR / "market_arrivals.csv"
    if arr_path.exists():
        ma = pd.read_csv(arr_path)
        ma["canonical_commodity"] = ma["commodity_name"].apply(canonicalize_commodity_name)
        sub_arr = ma[
            (ma["canonical_commodity"].str.lower() == canon)
            & (ma["market_name"].apply(lambda m: match_canonical_market(str(m), canon_mkt) is not None))
        ].copy()
        if not sub_arr.empty:
            sub_arr["observation_date"] = pd.to_datetime(sub_arr["observation_date"])
            d_arr = sub_arr.groupby("observation_date", as_index=False)["arrival_quantity_tonnes"].sum(min_count=1)
            daily = daily.merge(d_arr, on="observation_date", how="left")
            return daily

    daily["arrival_quantity_tonnes"] = float("nan")
    return daily


def get_historical_latest_features(commodity: str, market: str) -> Dict[str, Any]:
    """
    Retrieve the latest complete real-history observation & engineered features from the
    cleaned model-ready dataset (or wholesale_prices pipeline) via canonical `src.features`.
    """
    clean_path = PROCESSED_DIR / "model_ready_unified_commodities_clean.csv"
    canon = canonicalize_commodity_name(commodity)
    canon_mkt = canonicalize_market_name(market)
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        sub = df[
            (df["commodity_name"].str.lower() == canon.lower())
            & (df["market_name"].str.lower() == canon_mkt.lower())
        ].copy()
        if not sub.empty:
            sub = sub.sort_values("observation_date")
            row = sub.iloc[-1].to_dict()
            return {
                "status": "historical_archive_complete",
                "is_live": False,
                "freshness_label": "Historical Validated Archive (Not Live)",
                "source": row.get("data_source", "AGMARKNET Historical Archive"),
                "observation_date": str(row["observation_date"]),
                "target_date_t_plus_1": str(row.get("target_date_t_plus_1")),
                "actual_next_price_rs_per_qtl": float(row["target_price_t_plus_1"]),
                "arrivals_available": pd.notna(row.get("arrival_tonnes")),
                "arrival_observation_date": str(row["observation_date"]) if pd.notna(row.get("arrival_tonnes")) else None,
                "lag_features_ready": True,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "features": row,
            }

    hist = _load_local_exact_history(canon, canon_mkt, max_obs_date=None)
    if hist.empty:
        raise ValueError(f"No historical series found for {canon} @ {canon_mkt}.")
    has_arr = bool(hist["arrival_quantity_tonnes"].notna().any())
    feat_df = engineer_single_series_features(hist, has_arrivals=has_arr, drop_incomplete=True)
    if feat_df.empty:
        raise ValueError(f"Insufficient historical sessions with genuine arrivals for {canon} @ {canon_mkt}.")
    feat_df["commodity_name"] = canon
    feat_df["market_name"] = canon_mkt
    row = feat_df.iloc[-1].to_dict()
    obs_str = pd.Timestamp(row["observation_date"]).strftime("%Y-%m-%d")
    return {
        "status": "historical_archive_complete",
        "is_live": False,
        "freshness_label": "Historical Processed Archive (Not Live)",
        "source": str(row.get("data_source", "AGMARKNET")),
        "observation_date": obs_str,
        "arrivals_available": pd.notna(row.get("arrival_tonnes")),
        "arrival_observation_date": obs_str if pd.notna(row.get("arrival_tonnes")) else None,
        "lag_features_ready": True,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": row,
    }


def fetch_latest_market_data(
    commodity: str,
    market: str,
    force_refresh: bool = False,
    cache_ttl_seconds: int = 900,
    timeout_seconds: int = 10,
) -> Dict[str, Any]:
    """
    Fetch latest official Maharashtra mandi observation via `data.gov.in` API
    for exact (commodity, market), preserve genuine missing arrivals as NaN,
    and withhold Mode A inference unless all continuous lags + arrivals are present.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    canon = canonicalize_commodity_name(commodity)
    canon_mkt = canonicalize_market_name(market)
    cache_key = f"ws::{canon}::{canon_mkt}"
    cache = _read_cache()
    now_utc = datetime.now(timezone.utc)

    if not force_refresh and cache_key in cache:
        entry = cache[cache_key]
        cached_at = datetime.fromisoformat(entry["fetched_at_utc"])
        age_sec = (now_utc - cached_at).total_seconds()
        if age_sec <= cache_ttl_seconds:
            out = dict(entry)
            out["from_cache"] = True
            out["is_live"] = False
            out["cache_age_minutes"] = round(age_sec / 60.0, 1)
            out["freshness_label"] = (
                f"Cached API Response ({out['cache_age_minutes']} min old — not fetched live this second)"
            )
            return out

    if not api_key or api_key == "your_api_key_here":
        return {
            "status": "credentials_missing",
            "is_live": False,
            "from_cache": False,
            "commodity": canon,
            "market": canon_mkt,
            "arrivals_available": False,
            "arrival_observation_date": None,
            "lag_features_ready": False,
            "freshness_label": "Official API Key Missing",
            "message": "DATA_GOV_IN_API_KEY is not configured. Use Historical Replay Mode A or Manual Mode B.",
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    all_raw_records: List[Dict[str, Any]] = []
    api_error: Optional[str] = None
    error_code: Optional[str] = None

    try:
        for comm_q in COMMODITY_API_ALIASES.get(canon, [canon]):
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": "250",
                "filters[state]": "Maharashtra",
                "filters[commodity]": comm_q,
            }
            resp = requests.get(DATA_GOV_URL, params=params, timeout=timeout_seconds)
            if resp.status_code == 200:
                payload = resp.json()
                all_raw_records.extend(payload.get("records", []))
            elif resp.status_code in (401, 403):
                error_code = f"http_{resp.status_code}"
                api_error = f"Authentication error HTTP {resp.status_code} from data.gov.in"
                break
            elif resp.status_code == 429:
                error_code = "http_429"
                api_error = "Rate limit exceeded (HTTP 429) on data.gov.in API"
                break
            else:
                error_code = f"http_{resp.status_code}"
                api_error = f"HTTP {resp.status_code} returned by data.gov.in API"
    except requests.Timeout:
        error_code = "timeout"
        api_error = f"Request timed out after {timeout_seconds}s waiting for api.data.gov.in"
    except Exception as exc:
        error_code = "network_error"
        api_error = f"Network/API error querying data.gov.in: {exc}"

    if api_error:
        if cache_key in cache:
            stale = dict(cache[cache_key])
            cached_at = datetime.fromisoformat(stale["fetched_at_utc"])
            age_mins = round((now_utc - cached_at).total_seconds() / 60.0, 1)
            stale["is_live"] = False
            stale["from_cache"] = True
            stale["cache_age_minutes"] = age_mins
            stale["error_code"] = error_code
            stale["freshness_label"] = (
                f"STALE CACHED DATA ({age_mins} min old; live refresh failed: {api_error})"
            )
            stale["api_error"] = api_error
            return stale
        return {
            "status": "api_error",
            "error_code": error_code,
            "is_live": False,
            "from_cache": False,
            "commodity": canon,
            "market": canon_mkt,
            "arrivals_available": False,
            "arrival_observation_date": None,
            "lag_features_ready": False,
            "freshness_label": f"Live Fetch Failed ({error_code})",
            "message": f"{api_error}. Use Historical Archive Replay (Mode A) or Manual Exploration (Mode B).",
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    parsed = parse_and_filter_api_records(all_raw_records, canon, canon_mkt)
    if parsed["status"] == "ambiguous_market_match":
        return {
            "status": "ambiguous_market_match",
            "is_live": False,
            "from_cache": False,
            "commodity": canon,
            "market": canon_mkt,
            "arrivals_available": False,
            "arrival_observation_date": None,
            "lag_features_ready": False,
            "ambiguous_districts": parsed.get("ambiguous_districts", []),
            "rejected_submarkets": parsed.get("rejected_submarkets", []),
            "freshness_label": "Ambiguous API Market Match Rejected",
            "message": (
                f"API returned conflicting district identifiers {parsed.get('ambiguous_districts')} "
                f"for '{canon_mkt}'. Ambiguous payloads are rejected to prevent silent market mixing."
            ),
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    live_df: pd.DataFrame = parsed["daily_df"]
    if live_df.empty:
        return {
            "status": (
                "invalid_api_payload"
                if parsed.get("invalid_records_skipped", 0) > 0
                else "no_matching_live_record"
            ),
            "is_live": False,
            "from_cache": False,
            "commodity": canon,
            "market": canon_mkt,
            "arrivals_available": False,
            "arrival_observation_date": None,
            "rejected_submarkets": parsed.get("rejected_submarkets", []),
            "invalid_records_skipped": parsed.get("invalid_records_skipped", 0),
            "lag_features_ready": False,
            "freshness_label": "No Exact Valid Live Observation Available",
            "message": (
                f"0 valid exact-alias records matched {canon} @ {canon_mkt} "
                f"(rejected submarkets ignored: {parsed.get('rejected_submarkets', [])}, "
                f"invalid rows skipped: {parsed.get('invalid_records_skipped', 0)})."
            ),
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    latest_row = live_df.iloc[-1]
    latest_dt = pd.Timestamp(latest_row["observation_date"])
    obs_date_str = latest_dt.strftime("%Y-%m-%d")

    arr_today = latest_row["arrival_quantity_tonnes"]
    arrivals_avail = bool(pd.notna(arr_today) and math.isfinite(float(arr_today)))
    arr_val_clean = round(float(arr_today), 2) if arrivals_avail else None
    arr_obs_date = obs_date_str if arrivals_avail else None

    # Construct genuine Mode A features ONLY when continuous <=60d lookback AND all 16 features
    # (including physical arrivals) exist with zero fabrication.
    hist_df = _load_local_exact_history(canon, canon_mkt, max_obs_date=None)
    lag_features_ready = False
    feature_row = None
    missing_reasons: List[str] = []

    if not arrivals_avail:
        missing_reasons.append(
            "data.gov.in resource 9ef84268-d588-465a-a308-a864a43d0070 omits daily arrival volume "
            "(unavailable arrivals are kept NaN/None, never fabricated as 0.0)"
        )

    if not hist_df.empty:
        hist_clean = hist_df[hist_df["observation_date"] < latest_dt].copy()
        if not hist_clean.empty:
            last_hist_dt = pd.Timestamp(hist_clean["observation_date"].max())
            gap_days = int((latest_dt - last_hist_dt).days)
            if gap_days > 60:
                missing_reasons.append(
                    f"{gap_days:,}-day archive gap between historical series ending {last_hist_dt.strftime('%Y-%m-%d')} "
                    f"and API date {obs_date_str}"
                )
            elif len(hist_clean) >= 7:
                # Combined continuous history using exact same schema & feature builder
                hist_sub = hist_clean[["observation_date", "min_price_rs_per_qtl", "max_price_rs_per_qtl", "modal_price_rs_per_qtl", "arrival_quantity_tonnes"]]
                live_sub = live_df[["observation_date", "min_price_rs_per_qtl", "max_price_rs_per_qtl", "modal_price_rs_per_qtl", "arrival_quantity_tonnes"]]
                combined = pd.concat([hist_sub, live_sub], ignore_index=True).sort_values("observation_date")
                feat_df = engineer_single_series_features(
                    combined,
                    has_arrivals=True,
                    max_target_gap_days=None,
                    max_lookback_gap_days=60,
                    drop_incomplete=False,
                )
                last_feat = feat_df.iloc[-1]
                missing_cols = [c for c in SPECIALIST_NUMERIC_FEATURES if pd.isna(last_feat.get(c))]
                if not missing_cols:
                    lag_features_ready = True
                    row_dict = last_feat.to_dict()
                    row_dict["commodity_name"] = canon
                    row_dict["market_name"] = canon_mkt
                    feature_row = row_dict
                else:
                    missing_reasons.append(f"Incomplete required features: {missing_cols}")
        else:
            missing_reasons.append("No prior continuous sessions preceding live date.")
    else:
        missing_reasons.append("No historical series found for exact (commodity, market).")

    msg = (
        "Live market observation and complete 16-feature vector verified without imputation."
        if lag_features_ready
        else "Automated Mode A forecast withheld: " + "; ".join(missing_reasons)
    )

    result = {
        "status": "live_fetched",
        "is_live": True,
        "from_cache": False,
        "commodity": canon,
        "market": canon_mkt,
        "matched_api_market": str(latest_row["raw_market"]),
        "rejected_submarkets": parsed.get("rejected_submarkets", []),
        "varieties_aggregated": int(latest_row["varieties_count"]),
        "source": "data.gov.in AGMARKNET API (resource 9ef84268-d588-465a-a308-a864a43d0070)",
        "observation_date": obs_date_str,
        "observed_price_rs_per_qtl": round(float(latest_row["modal_price_rs_per_qtl"]), 2),
        "min_price_rs_per_qtl": round(float(latest_row["min_price_rs_per_qtl"]), 2),
        "max_price_rs_per_qtl": round(float(latest_row["max_price_rs_per_qtl"]), 2),
        "unit": "Rs/Quintal",
        "arrivals_available": arrivals_avail,
        "arrival_quantity_tonnes": arr_val_clean,
        "arrival_observation_date": arr_obs_date,
        "arrival_unit": parsed["arrival_units_detected"][0] if parsed.get("arrival_units_detected") else None,
        "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "fetched_at_utc": now_utc.isoformat(),
        "freshness_label": f"Live Official API Fetch ({now_utc.strftime('%Y-%m-%d %H:%M UTC')})",
        "lag_features_ready": lag_features_ready,
        "features": feature_row,
        "message": msg,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    cache[cache_key] = result
    _write_cache(cache)
    return result


def get_observed_retail_price(
    commodity: str,
    centre: str = "All-India Average",
    try_live_dca: bool = False,
    timeout_seconds: int = 8,
) -> Dict[str, Any]:
    """
    Retrieve actual observed DCA retail price (Rs/Kg).
    Returns None / status='unavailable' cleanly if absent — NEVER returns hardcoded fallback prices.
    """
    canon = canonicalize_commodity_name(commodity)
    rt_path = PROCESSED_DIR / "retail_prices.csv"
    live_err = None

    if try_live_dca:
        try:
            r = requests.get(
                DCA_URL,
                headers={"User-Agent": "Mozilla/5.0 Academic Research"},
                timeout=timeout_seconds,
                verify=False,
            )
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                tables = soup.find_all("table")
                if len(tables) >= 2:
                    rows = tables[1].find_all("tr")
                    date_row = [c.get_text(strip=True) for c in rows[2].find_all(["th", "td"])]
                    obs_date = date_row[0] if date_row else datetime.now().strftime("%Y-%m-%d")
                    for row in rows[3:]:
                        cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                        if cols and canonicalize_commodity_name(cols[0]).lower() == canon.lower():
                            val = pd.to_numeric(cols[2].replace("--", ""), errors="coerce") if len(cols) > 2 else None
                            if pd.notna(val) and float(val) > 0:
                                return {
                                    "status": "live_dca_observed",
                                    "is_live": True,
                                    "is_stale_archive": False,
                                    "commodity": canon,
                                    "retail_price_rs_per_kg": round(float(val), 2),
                                    "observation_date": str(obs_date),
                                    "centre": "All-India Average",
                                    "source": "DCA Daily Bulletin (fcainfoweb.nic.in)",
                                    "provenance": "Actual Observed DCA Live",
                                }
        except Exception as exc:
            live_err = str(exc)

    if not rt_path.exists():
        return {
            "status": "unavailable",
            "is_live": False,
            "is_stale_archive": True,
            "commodity": canon,
            "retail_price_rs_per_kg": None,
            "observation_date": None,
            "centre": None,
            "source": None,
            "provenance": "Unavailable",
        }

    rt = pd.read_csv(rt_path)
    rt["canonical_commodity"] = rt["commodity_name"].apply(canonicalize_commodity_name)
    sub = rt[rt["canonical_commodity"].str.lower() == canon.lower()].copy()
    sub = sub[pd.to_numeric(sub["retail_price"], errors="coerce") > 0].copy()
    if sub.empty:
        return {
            "status": "unavailable",
            "is_live": False,
            "is_stale_archive": True,
            "commodity": canon,
            "retail_price_rs_per_kg": None,
            "observation_date": None,
            "centre": None,
            "source": None,
            "provenance": "Unavailable",
            "live_attempt_error": live_err,
        }

    sub_centre = sub[sub["reporting_centre"].str.strip().str.lower() == centre.strip().lower()]
    pick_df = sub_centre if not sub_centre.empty else sub
    pick_df = pick_df.sort_values("observation_date")
    latest = pick_df.iloc[-1]
    obs_date_str = str(latest["observation_date"])

    return {
        "status": "archived_dca_observed",
        "is_live": False,
        "is_stale_archive": True,
        "commodity": canon,
        "retail_price_rs_per_kg": round(float(latest["retail_price"]), 2),
        "observation_date": obs_date_str,
        "centre": str(latest["reporting_centre"]),
        "source": str(latest["data_source"]),
        "provenance": f"Archived DCA Observed ({latest['reporting_centre']}, {obs_date_str})",
        "live_attempt_error": live_err,
    }
