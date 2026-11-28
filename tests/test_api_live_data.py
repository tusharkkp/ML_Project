"""
Phase 11C: Live API Exact Market Matching, Missing-Arrival Protection & Error Handling Tests.
"""

import math
from datetime import datetime, timezone
from pathlib import Path
import sys
import requests
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.live_data as live_data


class _DummyResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_exact_api_market_matching_rejects_submarkets_and_preserves_varieties():
    """Verify Pune excludes Pimpri/Hadapsar, Lasalgaon excludes Niphad, whitespace/case normalized."""
    records = [
        {"state": "Maharashtra", "district": "Pune", "market": "  APMC Pune ", "commodity": "Tomato", "variety": "Local", "arrival_date": "17/11/2026", "min_price": 1200, "max_price": 1800, "modal_price": 1500},
        {"state": "Maharashtra", "district": "Pune", "market": "pune apmc", "commodity": "Tomato", "variety": "Hybrid", "arrival_date": "17/11/2026", "min_price": 1400, "max_price": 2000, "modal_price": 1700},
        {"state": "Maharashtra", "district": "Pune", "market": "Pune(Pimpri)", "commodity": "Tomato", "variety": "Local", "arrival_date": "17/11/2026", "min_price": 5000, "max_price": 6000, "modal_price": 5500},
        {"state": "Maharashtra", "district": "Pune", "market": "Pune (Hadapsar)", "commodity": "Tomato", "variety": "Local", "arrival_date": "17/11/2026", "min_price": 7000, "max_price": 8000, "modal_price": 7500},
        {"state": "Maharashtra", "district": "Nashik", "market": "Lasalgaon(Niphad) ", "commodity": "Onion", "variety": "Red", "arrival_date": "17/11/2026", "min_price": 900, "max_price": 1200, "modal_price": 1050},
        {"state": "Maharashtra", "district": "Nashik", "market": "Lasalgaon", "commodity": "Onion", "variety": "Red", "arrival_date": "17/11/2026", "min_price": 1100, "max_price": 1500, "modal_price": 1300},
    ]

    pune_res = live_data.parse_and_filter_api_records(records, "Tomato", "Pune")
    assert pune_res["status"] == "ok"
    assert len(pune_res["daily_df"]) == 1
    assert pune_res["daily_df"].iloc[0]["modal_price_rs_per_qtl"] == 1600.0  # median(1500, 1700), Pimpri/Hadapsar excluded
    assert pune_res["daily_df"].iloc[0]["varieties_count"] == 2
    assert set(pune_res["rejected_submarkets"]) == {"Pune(Pimpri)", "Pune (Hadapsar)"}
    assert math.isnan(pune_res["daily_df"].iloc[0]["arrival_quantity_tonnes"])

    las_res = live_data.parse_and_filter_api_records(records, "Onion", "Lasalgaon")
    assert las_res["status"] == "ok"
    assert las_res["daily_df"].iloc[0]["modal_price_rs_per_qtl"] == 1300.0
    assert "Lasalgaon(Niphad)" in las_res["rejected_submarkets"]


def test_ambiguous_api_market_payload_is_rejected():
    """Conflicting districts under same canonical alias OR composite delimited markets are rejected."""
    records = [
        {"district": "Pune", "market": "APMC Pune", "commodity": "Tomato", "arrival_date": "17/11/2026", "min_price": 1200, "max_price": 1500, "modal_price": 1400},
        {"district": "Solapur", "market": "Pune", "commodity": "Tomato", "arrival_date": "17/11/2026", "min_price": 1200, "max_price": 1500, "modal_price": 1300},
    ]
    res = live_data.parse_and_filter_api_records(records, "Tomato", "Pune")
    assert res["status"] == "ambiguous_market_match"

    comp_rec = [
        {"district": "Pune", "market": "Pune / Pimpri", "commodity": "Tomato", "arrival_date": "17/11/2026", "min_price": 1200, "max_price": 1500, "modal_price": 1400}
    ]
    res2 = live_data.parse_and_filter_api_records(comp_rec, "Tomato", "Pune")
    assert res2["status"] == "no_matching_records"


def test_invalid_dates_prices_and_arrivals_handling():
    records = [
        {"district": "Pune", "market": "Pune", "commodity": "Tomato", "arrival_date": "bad-date", "min_price": 100, "max_price": 200, "modal_price": 150},
        {"district": "Pune", "market": "Pune", "commodity": "Tomato", "arrival_date": "17/11/2026", "min_price": -50, "max_price": 200, "modal_price": -10},
        {"district": "Pune", "market": "Pune", "commodity": "Tomato", "arrival_date": "17/11/2026", "min_price": 1200, "max_price": 1600, "modal_price": 1400, "arrivals_quintal": "500"},
    ]
    res = live_data.parse_and_filter_api_records(records, "Tomato", "Pune")
    assert res["invalid_records_skipped"] == 2
    assert res["status"] == "ok"
    # 500 quintals / 10 = 50.0 Tonnes
    assert math.isclose(res["daily_df"].iloc[0]["arrival_quantity_tonnes"], 50.0)


def test_fetch_api_auth_rate_limit_timeout_cache_and_missing_arrivals(monkeypatch, tmp_path):
    cache_file = tmp_path / "live_fetch_cache.json"
    monkeypatch.setattr(live_data, "CACHE_PATH", cache_file)
    monkeypatch.setattr(live_data, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("DATA_GOV_IN_API_KEY", "test_key")

    # 1. Auth HTTP 401
    monkeypatch.setattr(live_data.requests, "get", lambda *a, **kw: _DummyResp(401, {}))
    r401 = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=True)
    assert r401["status"] == "api_error" and r401["error_code"] == "http_401"

    # 2. Rate Limit HTTP 429
    monkeypatch.setattr(live_data.requests, "get", lambda *a, **kw: _DummyResp(429, {}))
    r429 = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=True)
    assert r429["status"] == "api_error" and r429["error_code"] == "http_429"

    # 3. Live response WITHOUT arrivals field -> preserves unavailable (None, NOT 0.0) & withholds Mode A
    monkeypatch.setattr(
        live_data.requests,
        "get",
        lambda *a, **kw: _DummyResp(
            200,
            {
                "records": [
                    {
                        "state": "Maharashtra",
                        "district": "Pune",
                        "market": "APMC Pune ",
                        "commodity": "Tomato",
                        "variety": "Local",
                        "arrival_date": "28/11/2026",
                        "min_price": "1800",
                        "max_price": "2400",
                        "modal_price": "2100",
                    },
                    {
                        "state": "Maharashtra",
                        "district": "Pune",
                        "market": "Pune(Pimpri)",
                        "commodity": "Tomato",
                        "variety": "Local",
                        "arrival_date": "28/11/2026",
                        "min_price": "9999",
                        "max_price": "9999",
                        "modal_price": "9999",
                    },
                ]
            },
        ),
    )
    rlive = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=True)
    assert rlive["is_live"] is True
    assert rlive["observed_price_rs_per_qtl"] == 2100.0
    assert rlive["arrivals_available"] is False
    assert rlive["arrival_quantity_tonnes"] is None
    assert rlive["lag_features_ready"] is False
    assert "Pune(Pimpri)" in rlive["rejected_submarkets"]

    # 4. Fresh cache hit marks from_cache=True and is_live=False
    rcache = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=False)
    assert rcache["from_cache"] is True
    assert rcache["is_live"] is False
    assert "Cached API Response" in rcache["freshness_label"]

    # 5. Timeout falls back cleanly to STALE CACHED DATA
    def _raise_to(*a, **kw):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(live_data.requests, "get", _raise_to)
    rstale = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=True)
    assert rstale["from_cache"] is True
    assert rstale["is_live"] is False
    assert rstale["error_code"] == "timeout"
    assert "STALE CACHED DATA" in rstale["freshness_label"]
