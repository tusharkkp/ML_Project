"""
Phase 11D: Retail Honesty (No Rs.45 Fallback), Complete-Basket Rules & Affordability Tests.
"""

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.live_data as live_data
from src.affordability import calculate_household_affordability, classify_affordability


def test_missing_retail_never_returns_45_or_fabricated_price(tmp_path, monkeypatch):
    # Empty retail CSV
    fake_proc = tmp_path / "processed"
    fake_proc.mkdir(parents=True)
    (fake_proc / "retail_prices.csv").write_text(
        "observation_date,commodity_name,reporting_centre,retail_price,data_source\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(live_data, "PROCESSED_DIR", fake_proc)

    obs = live_data.get_observed_retail_price("Tomato", try_live_dca=False)
    assert obs["status"] == "unavailable"
    assert obs["retail_price_rs_per_kg"] is None

    # Unknown crop on real CSV also stays unavailable
    monkeypatch.setattr(live_data, "PROCESSED_DIR", ROOT / "data" / "processed")
    obs_unk = live_data.get_observed_retail_price("Dragonfruit", try_live_dca=False)
    assert obs_unk["status"] == "unavailable"
    assert obs_unk["retail_price_rs_per_kg"] is None


def test_affordability_provenance_and_partial_basket_rejection():
    res = calculate_household_affordability(
        monthly_income=10000.0,
        basket_items=[
            {
                "commodity_name": "Tomato",
                "monthly_quantity_kg": 10.0,
                "current_retail_price_per_kg": 40.0,
                "current_price_source": "User Supplied",
                "predicted_retail_price_per_kg": 48.0,
                "prediction_source": "Hypothetical Sensitivity Scenario",
            },
            {
                "commodity_name": "Onion",
                "monthly_quantity_kg": 10.0,
                "current_retail_price_per_kg": 30.0,
                "current_price_source": "Archived DCA Observed (Mumbai, 2012-11-15)",
                "predicted_retail_price_per_kg": None,
            },
        ],
    )
    assert res["has_complete_predictions"] is False
    assert res["prediction_status"] == "incomplete_partial_predictions"
    assert res["predicted_food_basket_cost_rs"] is None
    assert res["item_breakdown"][0]["current_price_source"] == "User Supplied"
    assert "Archived DCA Observed" in res["item_breakdown"][1]["current_price_source"]


def test_gradio_affordability_never_fakes_future_equals_current():
    import gradio_app

    txt = gradio_app.gradio_affordability(15000.0, "Urban", 10.0, 12.0, 4.0, 3.0)
    assert "Retail ML Preds    : unavailable" in txt
    assert "Hypothetical Scenario Output" not in txt

    # With explicit hypothetical shock, explicitly labels as Hypothetical Scenario (NOT ML Prediction)
    txt_scen = gradio_app.gradio_affordability(15000.0, "Urban", 10.0, 12.0, 4.0, 3.0, hypothetical_shock_pct=10.0)
    assert "Hypothetical Scenario Output (NOT an ML Prediction)" in txt_scen


def test_affordability_rejects_invalid_inputs_and_tiers():
    assert "Low Vulnerability" in classify_affordability(25.0)
    assert "Critical Vulnerability" in classify_affordability(65.0)
    valid = [{"commodity_name": "Tomato", "monthly_quantity_kg": 5.0, "current_retail_price_per_kg": 40.0}]
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_household_affordability(0.0, valid)
    with pytest.raises(ValueError, match="non-negative finite"):
        calculate_household_affordability(15000.0, [{"commodity_name": "Tomato", "monthly_quantity_kg": float("nan"), "current_retail_price_per_kg": 40.0}])
