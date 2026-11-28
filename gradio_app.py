"""
Gradio Dashboard — Hugging Face Spaces & Standalone Web UI (v3.0.0-leakfree).

Synchronized with leak-free `src/inference.py`, `src/live_data.py`, `src/market_comparison.py`,
and `src/affordability.py`.
"""

import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.affordability import calculate_household_affordability
from src.inference import UnifiedCommodityPredictor, UnsupportedCommodityError
from src.live_data import (
    fetch_latest_market_data,
    get_historical_latest_features,
    get_observed_retail_price,
)
from src.market_comparison import compare_markets, get_markets_for_commodity

MODELS_DIR = PROJECT_ROOT / "models"
unified = UnifiedCommodityPredictor(models_dir=MODELS_DIR)


def gradio_predict(
    commodity: str,
    market: str,
    input_mode: str,
    modal_price: float,
    price_lag_1: float,
    price_lag_3: float,
    price_lag_7: float,
    arrival_tonnes: float,
    month: int,
    allow_experimental: bool,
):
    try:
        if input_mode == "Historical Archive Replay":
            hist = get_historical_latest_features(commodity, market)
            res = unified.predict_from_features(
                hist["features"],
                allow_experimental=allow_experimental,
                feature_mode="historical_archive_replay",
            )
            obs_info = f"Observation: {hist['observation_date']} (Actual t+1: Rs. {hist['actual_next_price_rs_per_qtl']}/Qtl)"
        elif input_mode == "Live API Fetch":
            live = fetch_latest_market_data(commodity, market)
            if not live.get("lag_features_ready"):
                return (
                    f"Live Fetch Status: {live.get('freshness_label')}\n"
                    f"Message: {live.get('message')}\n"
                    "Per zero-fabrication rules, lag inference is withheld across multi-year archive gaps."
                )
            res = unified.predict_from_features(
                live["features"],
                allow_experimental=allow_experimental,
                feature_mode="live_api_continuous",
            )
            obs_info = f"Live Date: {live.get('observation_date')}"
        else:
            vals = [modal_price, price_lag_1, price_lag_3, price_lag_7]
            res = unified.predict(
                commodity=commodity,
                market=market,
                modal_price=modal_price,
                price_lag_1=price_lag_1,
                price_lag_3=price_lag_3,
                price_lag_7=price_lag_7,
                rolling_mean_7=sum(vals) / 4.0,
                rolling_std_7=float(pd.Series(vals).std()),
                rolling_min_7=min(vals),
                rolling_max_7=max(vals),
                arrival_tonnes=arrival_tonnes,
                arrival_lag_1=arrival_tonnes,
                rolling_arrival_mean_7=arrival_tonnes,
                daily_spread=150.0,
                month=int(month),
                day_of_week=2,
                allow_experimental=allow_experimental,
                allow_manual_approx=False,
            )
            obs_info = "Manual user-entered lags"

        ew = res["early_warning_classifier"]
        obs_ret = res["observed_retail_reference"]
        return (
            f"Commodity / Market : {res['commodity']} @ {res['market']}\n"
            f"Source / Horizon   : {obs_info} | {res['prediction_horizon']}\n"
            f"Current Modal      : Rs. {res['current_modal_price_rs_per_qtl']}/Qtl (Equiv Rs. {res['current_wholesale_equiv_rs_per_kg']}/Kg)\n"
            f"Predicted Next t+1 : Rs. {res['predicted_next_session_rs_per_qtl']}/Qtl ({res['predicted_change_pct']:+.2f}%)\n"
            f"Wholesale Equiv/Kg : Rs. {res['wholesale_equiv_rs_per_kg']}/Kg (unit conversion only, NOT retail prediction)\n"
            f"Observed DCA Retail: Rs. {obs_ret.get('retail_price_rs_per_kg')}/Kg ({obs_ret.get('centre')}, {obs_ret.get('observation_date')})\n"
            f"Retail ML Status   : {res['retail_estimation_status']}\n"
            f"Early Warning (ML) : {ew.get('predicted_regime')} | Probs={ew.get('class_probabilities')} | Reliable={ew.get('is_reliable_early_warning')}\n"
            f"Warning / Notes    : {res.get('validation_warning') or 'None'}"
        )
    except UnsupportedCommodityError as exc:
        return f"UNSUPPORTED COMMODITY BLOCKED:\n{exc}"
    except Exception as exc:
        return f"Error: {exc}"


def gradio_affordability(income: float, sector: str, tomato_kg: float, onion_kg: float, dal_kg: float, sugar_kg: float):
    try:
        items = []
        for comm, qty in [("Tomato", tomato_kg), ("Onion", onion_kg), ("Moong Dal", dal_kg), ("Sugar", sugar_kg)]:
            obs = get_observed_retail_price(comm)
            price = float(obs.get("retail_price_rs_per_kg") or 45.0)
            items.append(
                {
                    "commodity_name": comm,
                    "monthly_quantity_kg": qty,
                    "current_retail_price_per_kg": price,
                    "predicted_retail_price_per_kg": price,
                }
            )
        res = calculate_household_affordability(monthly_income=income, basket_items=items, sector=sector)
        return (
            f"Status: {res['affordability_status']}\n"
            f"Current Basket Cost: Rs. {res['current_food_basket_cost_rs']:.2f}\n"
            f"Food Budget Share  : {res['current_food_budget_share_pct']:.2f}%\n"
            f"Remaining Income   : Rs. {res['discretionary_income_remaining_rs']:.2f}\n"
            f"Disclaimer: {res['methodology_disclaimer']}"
        )
    except Exception as exc:
        return f"Affordability Error: {exc}"


def gradio_market_compare(commodity: str):
    try:
        mkts = get_markets_for_commodity(commodity, min_rows=10)[:4]
        comp = compare_markets(commodity, mkts)
        return comp["summary_df"].to_string(index=False) + "\n\nOverlapping Dates Comparison:\n" + comp["overlapping_comparison_df"].to_string(index=False)
    except Exception as exc:
        return f"Comparison Error: {exc}"


with gr.Blocks(title="Food Inflation & Household Affordability Monitor") as demo:
    gr.Markdown("# AI Food Inflation Prediction & Household Affordability Monitor (`v3.0.0-leakfree`)")
    with gr.Tab("Price Forecasting"):
        c_in = gr.Textbox(value="Tomato", label="Commodity (Validated: Tomato, Onion, Moong Dal, Sugar)")
        m_in = gr.Dropdown(["Pune", "Mumbai", "Lasalgaon"], value="Pune", label="Market")
        mode_in = gr.Radio(["Historical Archive Replay", "Live API Fetch", "Manual Entry"], value="Historical Archive Replay", label="Input Mode")
        with gr.Row():
            mp = gr.Number(value=1300, label="Current Modal (Rs/Qtl)")
            l1 = gr.Number(value=1250, label="Lag 1")
            l3 = gr.Number(value=1180, label="Lag 3")
            l7 = gr.Number(value=1100, label="Lag 7")
        with gr.Row():
            arr = gr.Number(value=120, label="Arrivals (Tonnes)")
            mo = gr.Slider(1, 12, value=11, step=1, label="Month")
            exp = gr.Checkbox(value=False, label="Allow Experimental Unseen Crop")
        btn = gr.Button("Predict Next Active Trading Session")
        out = gr.Textbox(lines=10, label="Forecast Output")
        btn.click(gradio_predict, [c_in, m_in, mode_in, mp, l1, l3, l7, arr, mo, exp], out)

    with gr.Tab("Market Comparison"):
        mc = gr.Dropdown(["Onion", "Tomato", "Sugar", "Moong Dal"], value="Onion", label="Commodity")
        mbtn = gr.Button("Compare Exact APMC Markets")
        mout = gr.Textbox(lines=14, label="Cross-Market Historical Comparison")
        mbtn.click(gradio_market_compare, [mc], mout)

    with gr.Tab("Household Affordability"):
        inc = gr.Number(value=15000, label="Monthly Income (Rs.)")
        sec = gr.Radio(["Urban", "Rural"], value="Urban", label="Sector")
        with gr.Row():
            tk = gr.Number(value=12, label="Tomato (Kg/mo)")
            ok = gr.Number(value=15, label="Onion (Kg/mo)")
            dk = gr.Number(value=4, label="Moong Dal (Kg/mo)")
            sk = gr.Number(value=3, label="Sugar (Kg/mo)")
        abtn = gr.Button("Assess Affordability")
        aout = gr.Textbox(lines=8, label="Affordability Output")
        abtn.click(gradio_affordability, [inc, sec, tk, ok, dk, sk], aout)

if __name__ == "__main__":
    demo.launch()
