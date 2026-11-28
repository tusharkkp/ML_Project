"""
End-to-End Integration Tests: Mocked Live API + Fallback + Forecast + Affordability + UI (Phase 5 & 10).
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import requests

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.live_data as live_data
from src.affordability import calculate_household_affordability
from src.inference import UnifiedCommodityPredictor
from src.schemas import CommodityPricePredictionRequest, HouseholdAffordabilityRequest


class _DummyResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_live_fetch_gap_protection_cache_and_timeout_fallback(monkeypatch, tmp_path):
    """Test live API fetch -> multi-year gap protection -> cache -> timeout stale fallback."""
    cache_file = tmp_path / "live_fetch_cache.json"
    monkeypatch.setattr(live_data, "CACHE_PATH", cache_file)
    monkeypatch.setattr(live_data, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("DATA_GOV_IN_API_KEY", "test_dummy_key")

    def fake_get_ok(*args, **kwargs):
        return _DummyResp(
            200,
            {
                "records": [
                    {
                        "state": "Maharashtra",
                        "district": "Pune",
                        "market": "Pune",
                        "commodity": "Tomato",
                        "variety": "Local",
                        "arrival_date": "28/11/2026",
                        "min_price": "1800",
                        "max_price": "2400",
                        "modal_price": "2100",
                    }
                ]
            },
        )

    monkeypatch.setattr(live_data.requests, "get", fake_get_ok)

    # 1. Live fetch succeeds and explicitly refuses synthetic lag creation across multi-year gap
    res_live = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=True)
    assert res_live["status"] == "live_fetched"
    assert res_live["is_live"] is True
    assert res_live["from_cache"] is False
    assert res_live["observed_price_rs_per_qtl"] == 2100.0
    assert res_live["lag_features_ready"] is False
    assert "cannot be synthesized" in res_live["message"].lower()

    # 2. Cached call returns cached flag clearly (not marked fresh live)
    res_cached = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=False)
    assert res_cached["from_cache"] is True
    assert "Cached API Response" in res_cached["freshness_label"]

    # 3. Simulated network timeout falls back to stale cache labeled STALE (is_live=False)
    def fake_get_timeout(*args, **kwargs):
        raise requests.Timeout("Simulated timeout")

    monkeypatch.setattr(live_data.requests, "get", fake_get_timeout)
    res_stale = live_data.fetch_latest_market_data("Tomato", "Pune", force_refresh=True)
    assert res_stale["is_live"] is False
    assert res_stale["from_cache"] is True
    assert "STALE CACHED DATA" in res_stale["freshness_label"]


def test_end_to_end_replay_forecast_affordability_and_schemas():
    """Test schema contracts, historical replay forecast, retail reference, and affordability integration."""
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    hist = live_data.get_historical_latest_features("Tomato", "Pune")

    # Validate incoming features with Pydantic request contract
    req = CommodityPricePredictionRequest(**hist["features"])
    assert req.modal_price_rs_per_qtl > 0

    forecast = predictor.predict_from_features(req.model_dump())
    ret_ref = forecast["observed_retail_reference"]
    base_retail_kg = float(ret_ref.get("retail_price_rs_per_kg") or 40.0)

    # Integrate forecast % change into household scenario simulation
    proj_retail = round(base_retail_kg * (1.0 + forecast["predicted_change_pct"] / 100.0), 2)
    aff_req = HouseholdAffordabilityRequest(
        monthly_income_rs=18000.0,
        sector="Urban",
        basket_items=[
            {
                "commodity_name": "Tomato",
                "monthly_quantity_kg": 12.0,
                "current_retail_price_per_kg": base_retail_kg,
                "predicted_retail_price_per_kg": proj_retail,
            }
        ],
    )
    aff_res = calculate_household_affordability(
        monthly_income=aff_req.monthly_income_rs,
        basket_items=[i.model_dump() for i in aff_req.basket_items],
        sector=aff_req.sector,
    )
    assert aff_res["has_complete_predictions"] is True
    assert aff_res["current_food_basket_cost_rs"] > 0


def test_gradio_and_streamlit_app_smoke():
    """Verify Gradio callbacks and Streamlit AppTest execute cleanly without KeyError/ImportError."""
    import gradio_app

    pred_txt = gradio_app.gradio_predict("Onion", "Lasalgaon", "Historical Archive Replay", 0, 0, 0, 0, 0, 11, False)
    assert "Onion @ Lasalgaon" in pred_txt
    assert "Early Warning (ML)" in pred_txt

    blocked_txt = gradio_app.gradio_predict("Potato", "Pune", "Manual Entry", 1200, 1180, 1150, 1100, 100, 11, False)
    assert "UNSUPPORTED COMMODITY BLOCKED" in blocked_txt

    aff_txt = gradio_app.gradio_affordability(15000.0, "Urban", 10.0, 12.0, 4.0, 3.0)
    assert "Food Budget Share" in aff_txt

    cmp_txt = gradio_app.gradio_market_compare("Onion")
    assert "Lasalgaon" in cmp_txt and "Overlapping Dates Comparison" in cmp_txt

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
    at.run()
    assert not at.exception
