"""
Phase 11F: End-to-End Integration Tests (Schemas, Streamlit Multi-Page AppTest, Gradio & HF Space).
"""

import importlib.util
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.affordability import calculate_household_affordability
from src.inference import UnifiedCommodityPredictor
from src.live_data import get_historical_latest_features, get_observed_retail_price
from src.schemas import CommodityPricePredictionRequest, HouseholdAffordabilityRequest


def test_end_to_end_replay_schema_and_affordability():
    predictor = UnifiedCommodityPredictor(models_dir=ROOT / "models")
    hist = get_historical_latest_features("Tomato", "Pune")
    req = CommodityPricePredictionRequest(**hist["features"])
    forecast = predictor.predict_validated(req.model_dump())

    ret = get_observed_retail_price("Tomato")
    assert ret["retail_price_rs_per_kg"] is not None

    aff_req = HouseholdAffordabilityRequest(
        monthly_income_rs=18000.0,
        sector="Urban",
        basket_items=[
            {
                "commodity_name": "Tomato",
                "monthly_quantity_kg": 12.0,
                "current_retail_price_per_kg": float(ret["retail_price_rs_per_kg"]),
                "predicted_retail_price_per_kg": None,
                "price_source": str(ret["provenance"]),
            }
        ],
    )
    res = calculate_household_affordability(
        monthly_income=aff_req.monthly_income_rs,
        basket_items=[i.model_dump() for i in aff_req.basket_items],
        sector=aff_req.sector,
    )
    assert res["prediction_status"] == "unavailable"
    assert forecast["is_validated_forecast"] is True


def test_gradio_and_huggingface_entrypoint_parity():
    import gradio_app

    # Load HF deployment/huggingface/app.py cleanly as distinct module
    hf_spec = importlib.util.spec_from_file_location("hf_app", str(ROOT / "deployment" / "huggingface" / "app.py"))
    hf_mod = importlib.util.module_from_spec(hf_spec)
    hf_spec.loader.exec_module(hf_mod)

    for mod in (gradio_app, hf_mod):
        p_mode_a = mod.gradio_predict("Onion", "Lasalgaon", "Mode A: Historical Archive Replay", 0, 0, 0, 0, 0, 0, 0, 11, False)
        assert "Mode A: Validated Forecasting" in p_mode_a and "Onion @ Lasalgaon" in p_mode_a

        p_mode_b = mod.gradio_predict("Tomato", "Pune", "Mode B: Experimental Manual Exploration", 1400, 1350, 1280, 1200, 120, 115, 125, 11, False)
        assert "Mode B: Experimental Manual Exploration" in p_mode_b and "EXPERIMENTAL DISCLAIMER" in p_mode_b

        p_block = mod.gradio_predict("Potato", "Pune", "Mode B: Experimental Manual Exploration", 1200, 1180, 1150, 1100, 100, 100, 100, 11, False)
        assert "MODE A VALIDATION BLOCK" in p_block

        cmp_out = mod.gradio_market_compare("Onion")
        assert "Lasalgaon" in cmp_out

        aff_out = mod.gradio_affordability(15000.0, "Urban", 12.0, 15.0, 4.0, 3.0)
        assert "Current Basket Cost" in aff_out


def test_streamlit_all_pages_smoke():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=25)
    at.run()
    assert not at.exception

    # Navigate across Page 2 (Market Comparison) & Page 3 (Affordability) & Page 4 (Benchmarks)
    at.sidebar.radio[0].set_value("2. Market Comparison").run()
    assert not at.exception

    at.sidebar.radio[0].set_value("3. Household Affordability").run()
    assert not at.exception

    at.sidebar.radio[0].set_value("4. Model Benchmarks & Classification").run()
    assert not at.exception
