"""
Official Data Acquisition, Cache Management & Live-to-Feature Parity Module (Phase 5 & Phase 6).

Integrates:
1. data.gov.in AGMARKNET API (`resource/9ef84268-d588-465a-a308-a864a43d0070`) using `DATA_GOV_IN_API_KEY`
2. DCA Daily Retail Bulletin (`fcainfoweb.nic.in`) + processed `retail_prices.csv`
3. Canonical local processed history (`wholesale_prices.csv`, `market_arrivals.csv`) fed through
   the EXACT production `engineer_single_series_features()` function from `src/features.py`.

Strict Scientific Honesty Rules:
- Never labels stale cached records or archival data as live.
- Never manufactures synthetic lag history across missing multi-month/year gaps.
- Explicit source, observation date, fetch timestamp, and staleness tracking.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import urllib3

from src.features import canonicalize_commodity_name, engineer_single_series_features

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_PATH = PROJECT_ROOT / "data" / "metadata" / "live_fetch_cache.json"

DATA_GOV_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
DCA_URL = "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx"

# Mapping canonical (commodity, market) to known data.gov.in equivalents
MARKET_API_ALIASES: Dict[str, List[str]] = {
    "Pune": ["Pune", "Apmc Pune", "Pune(Gultekdi)"],
    "Lasalgaon": ["Lasalgaon", "Apmc Lasalgaon"],
    "Mumbai": ["Mumbai", "Apmc Mumbai", "Vashi"],
}

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


def _parse_api_date(raw_val: str) -> Optional[pd.Timestamp]:
    if not raw_val:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(str(raw_val).strip(), fmt))
        except ValueError:
            continue
    parsed = pd.to_datetime(raw_val, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _load_local_exact_history(
    commodity: str, market: str, max_obs_date: Optional[str] = "2019-12-31"
) -> pd.DataFrame:
    """Load local processed wholesale + arrivals series for an exact (commodity, market)."""
    ws = pd.read_csv(PROCESSED_DIR / "wholesale_prices.csv")
    ws["canonical_commodity"] = ws["commodity_name"].apply(canonicalize_commodity_name)
    canon = canonicalize_commodity_name(commodity).lower()

    sub = ws[
        (ws["canonical_commodity"].str.lower() == canon)
        & (ws["market_name"].str.strip().str.lower() == market.strip().lower())
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
            & (ma["market_name"].str.strip().str.lower() == market.strip().lower())
        ].copy()
        if not sub_arr.empty:
            sub_arr["observation_date"] = pd.to_datetime(sub_arr["observation_date"])
            d_arr = sub_arr.groupby("observation_date", as_index=False)["arrival_quantity_tonnes"].sum()
            daily = daily.merge(d_arr, on="observation_date", how="left")
            daily["arrival_quantity_tonnes"] = daily["arrival_quantity_tonnes"].fillna(0.0)
            return daily

    daily["arrival_quantity_tonnes"] = 0.0
    return daily


def get_historical_latest_features(commodity: str, market: str) -> Dict[str, Any]:
    """
    Retrieve the latest complete real-history observation & engineered features from the
    cleaned model-ready dataset (or wholesale_prices pipeline) without fabrication.
    """
    clean_path = PROCESSED_DIR / "model_ready_unified_commodities_clean.csv"
    canon = canonicalize_commodity_name(commodity)
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        sub = df[
            (df["commodity_name"].str.lower() == canon.lower())
            & (df["market_name"].str.lower() == market.strip().lower())
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
                "lag_features_ready": True,
                "features": row,
            }

    hist = _load_local_exact_history(canon, market, max_obs_date=None)
    if hist.empty:
        raise ValueError(f"No historical series found for {canon} @ {market}.")
    has_arr = bool(hist["arrival_quantity_tonnes"].sum() > 0)
    feat_df = engineer_single_series_features(hist, has_arrivals=has_arr, drop_incomplete=True)
    if feat_df.empty:
        raise ValueError(f"Insufficient historical sessions for {canon} @ {market} lag computation.")
    feat_df["commodity_name"] = canon
    feat_df["market_name"] = market
    row = feat_df.iloc[-1].to_dict()
    return {
        "status": "historical_archive_complete",
        "is_live": False,
        "freshness_label": "Historical Processed Archive (Not Live)",
        "source": str(row.get("data_source", "AGMARKNET")),
        "observation_date": pd.Timestamp(row["observation_date"]).strftime("%Y-%m-%d"),
        "lag_features_ready": True,
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
    Fetch the latest official Maharashtra mandi observation via `data.gov.in` API
    for the requested (commodity, market) pair and evaluate whether real continuous
    lag features can be constructed without fabricating history.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    canon = canonicalize_commodity_name(commodity)
    cache_key = f"ws::{canon}::{market}"
    cache = _read_cache()
    now_utc = datetime.now(timezone.utc)

    if not force_refresh and cache_key in cache:
        entry = cache[cache_key]
        cached_at = datetime.fromisoformat(entry["fetched_at_utc"])
        age_sec = (now_utc - cached_at).total_seconds()
        if age_sec <= cache_ttl_seconds:
            out = dict(entry)
            out["from_cache"] = True
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
            "market": market,
            "lag_features_ready": False,
            "freshness_label": "Official API Key Missing",
            "message": (
                "DATA_GOV_IN_API_KEY is not configured. Using offline archives or manual entry fallback."
            ),
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    comm_aliases = [c.lower() for c in COMMODITY_API_ALIASES.get(canon, [canon])]
    mkt_aliases = [m.lower() for m in MARKET_API_ALIASES.get(market, [market])]

    matched_records: List[Dict[str, Any]] = []
    api_error: Optional[str] = None

    try:
        for comm_q in COMMODITY_API_ALIASES.get(canon, [canon]):
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": "200",
                "filters[state]": "Maharashtra",
                "filters[commodity]": comm_q,
            }
            resp = requests.get(DATA_GOV_URL, params=params, timeout=timeout_seconds)
            if resp.status_code == 200:
                payload = resp.json()
                for r in payload.get("records", []):
                    r_comm = canonicalize_commodity_name(r.get("commodity", "")).lower()
                    r_mkt = str(r.get("market", "")).strip().lower()
                    if r_comm == canon.lower() or r_comm in comm_aliases:
                        if r_mkt in mkt_aliases or market.lower() in r_mkt:
                            matched_records.append(r)
            elif resp.status_code in (401, 403):
                api_error = f"Authentication error HTTP {resp.status_code} from data.gov.in"
                break
            elif resp.status_code == 429:
                api_error = "Rate limit exceeded (HTTP 429) on data.gov.in API"
                break
            else:
                api_error = f"HTTP {resp.status_code} returned by data.gov.in API"
    except requests.Timeout:
        api_error = f"Request timed out after {timeout_seconds}s waiting for api.data.gov.in"
    except Exception as exc:
        api_error = f"Network/API error querying data.gov.in: {exc}"

    if api_error:
        # Check if stale cache exists — label clearly as stale cache, NEVER live!
        if cache_key in cache:
            stale = dict(cache[cache_key])
            cached_at = datetime.fromisoformat(stale["fetched_at_utc"])
            age_mins = round((now_utc - cached_at).total_seconds() / 60.0, 1)
            stale["is_live"] = False
            stale["from_cache"] = True
            stale["cache_age_minutes"] = age_mins
            stale["freshness_label"] = (
                f"STALE CACHED DATA ({age_mins} min old; live refresh failed: {api_error})"
            )
            stale["api_error"] = api_error
            return stale
        return {
            "status": "api_error",
            "is_live": False,
            "from_cache": False,
            "commodity": canon,
            "market": market,
            "lag_features_ready": False,
            "freshness_label": "Live Fetch Failed / Timed Out",
            "message": f"{api_error}. Please use Historical Replay Mode or Manual Input fallback.",
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    if not matched_records:
        # Also inspect stored data.gov.in 2026-11-17 snapshot transparently without mislabeling as live
        return {
            "status": "no_matching_live_record",
            "is_live": False,
            "from_cache": False,
            "commodity": canon,
            "market": market,
            "lag_features_ready": False,
            "freshness_label": "No Live Observation Available for Today",
            "message": (
                f"data.gov.in returned 0 current Maharashtra auction records matching {canon} @ {market}. "
                "Mandis may be closed or have not reported yet today. Use Historical Replay or Manual mode."
            ),
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    # Parse, validate numeric prices, and pick most recent observation date
    valid_rows = []
    for r in matched_records:
        dt = _parse_api_date(r.get("arrival_date", ""))
        try:
            p_min = float(r.get("min_price", 0))
            p_max = float(r.get("max_price", 0))
            p_mod = float(r.get("modal_price", 0))
        except (TypeError, ValueError):
            continue
        if dt is None or p_mod <= 0:
            continue
        if p_min > p_max and p_max > 0:
            p_min, p_max = p_max, p_min
        valid_rows.append(
            {
                "observation_date": dt,
                "min_price_rs_per_qtl": p_min if p_min > 0 else p_mod,
                "max_price_rs_per_qtl": p_max if p_max > 0 else p_mod,
                "modal_price_rs_per_qtl": p_mod,
                "raw_market": str(r.get("market", market)),
                "variety": str(r.get("variety", "Other")),
            }
        )

    if not valid_rows:
        return {
            "status": "invalid_api_payload",
            "is_live": False,
            "lag_features_ready": False,
            "commodity": canon,
            "market": market,
            "freshness_label": "Invalid Numeric Values in API Response",
            "message": "Fetched API rows lacked valid positive modal prices or parseable dates.",
            "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    live_df = (
        pd.DataFrame(valid_rows)
        .groupby("observation_date", as_index=False)
        .agg(
            {
                "min_price_rs_per_qtl": "min",
                "max_price_rs_per_qtl": "max",
                "modal_price_rs_per_qtl": "median",
                "raw_market": "first",
            }
        )
        .sort_values("observation_date")
    )
    latest_row = live_df.iloc[-1]
    latest_dt = pd.Timestamp(latest_row["observation_date"])

    # Combine with local history ONLY if continuous without multi-month/year gap
    hist_df = _load_local_exact_history(canon, market, max_obs_date=None)
    lag_features_ready = False
    feature_row = None
    gap_note = ""

    if not hist_df.empty:
        hist_clean = hist_df[hist_df["observation_date"] < latest_dt].copy()
        if not hist_clean.empty:
            last_hist_dt = pd.Timestamp(hist_clean["observation_date"].max())
            gap_days = int((latest_dt - last_hist_dt).days)
            if gap_days <= 30:
                combined = pd.concat(
                    [
                        hist_clean[["observation_date", "min_price_rs_per_qtl", "max_price_rs_per_qtl", "modal_price_rs_per_qtl", "arrival_quantity_tonnes"]],
                        live_df.assign(arrival_quantity_tonnes=0.0)[
                            ["observation_date", "min_price_rs_per_qtl", "max_price_rs_per_qtl", "modal_price_rs_per_qtl", "arrival_quantity_tonnes"]
                        ],
                    ],
                    ignore_index=True,
                )
                feat_df = engineer_single_series_features(
                    combined,
                    has_arrivals=bool(combined["arrival_quantity_tonnes"].sum() > 0),
                    max_target_gap_days=None,
                    max_lookback_gap_days=60,
                    drop_incomplete=False,
                )
                last_feat = feat_df.iloc[-1]
                needed = ["modal_price_rs_per_qtl", "price_lag_1", "price_lag_3", "price_lag_7", "rolling_mean_7", "rolling_std_7"]
                if all(pd.notna(last_feat[k]) for k in needed):
                    lag_features_ready = True
                    feature_row = last_feat.to_dict()
            else:
                gap_note = (
                    f"Latest live observation ({latest_dt.strftime('%Y-%m-%d')}) is separated by {gap_days:,} days "
                    f"from the dense historical series ending {last_hist_dt.strftime('%Y-%m-%d')}. "
                    "Per strict zero-fabrication rules, 7-day rolling & lag features cannot be synthesized across a multi-year gap."
                )

    result = {
        "status": "live_fetched",
        "is_live": True,
        "from_cache": False,
        "commodity": canon,
        "market": market,
        "matched_api_market": str(latest_row["raw_market"]),
        "source": "data.gov.in AGMARKNET API (resource 9ef84268-d588-465a-a308-a864a43d0070)",
        "observation_date": latest_dt.strftime("%Y-%m-%d"),
        "observed_price_rs_per_qtl": round(float(latest_row["modal_price_rs_per_qtl"]), 2),
        "min_price_rs_per_qtl": round(float(latest_row["min_price_rs_per_qtl"]), 2),
        "max_price_rs_per_qtl": round(float(latest_row["max_price_rs_per_qtl"]), 2),
        "unit": "Rs/Quintal",
        "fetched_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "fetched_at_utc": now_utc.isoformat(),
        "freshness_label": f"Live Official API Fetch ({now_utc.strftime('%Y-%m-%d %H:%M UTC')})",
        "lag_features_ready": lag_features_ready,
        "features": feature_row,
        "message": gap_note or "Live market observation fetched successfully.",
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
    Retrieve the latest actual observed DCA retail price (Rs/Kg) from processed
    `retail_prices.csv` or live DCA bulletin when requested. Never converts wholesale/100.
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
                                    "commodity": canon,
                                    "retail_price_rs_per_kg": round(float(val), 2),
                                    "observation_date": str(obs_date),
                                    "centre": "All-India Average",
                                    "source": "DCA Daily Bulletin (fcainfoweb.nic.in)",
                                }
        except Exception as exc:
            live_err = str(exc)

    if not rt_path.exists():
        return {"status": "unavailable", "is_live": False, "commodity": canon, "retail_price_rs_per_kg": None}

    rt = pd.read_csv(rt_path)
    rt["canonical_commodity"] = rt["commodity_name"].apply(canonicalize_commodity_name)
    sub = rt[rt["canonical_commodity"].str.lower() == canon.lower()].copy()
    if sub.empty:
        return {"status": "unavailable", "is_live": False, "commodity": canon, "retail_price_rs_per_kg": None}

    # Prefer requested reporting centre if recent, else latest overall DCA observation
    sub_centre = sub[sub["reporting_centre"].str.lower() == centre.lower()]
    pick_df = sub_centre if not sub_centre.empty else sub
    pick_df = pick_df.sort_values("observation_date")
    latest = pick_df.iloc[-1]

    return {
        "status": "archived_dca_observed",
        "is_live": False,
        "commodity": canon,
        "retail_price_rs_per_kg": round(float(latest["retail_price"]), 2),
        "observation_date": str(latest["observation_date"]),
        "centre": str(latest["reporting_centre"]),
        "source": str(latest["data_source"]),
        "live_attempt_error": live_err,
    }
