"""
Gradio Dashboard for Hugging Face Spaces Deployment (v3.1.0-parity).

Resolves PROJECT_ROOT dynamically so the file operates identically inside
`deployment/huggingface/app.py` and at `/` in Hugging Face Spaces.
Synchronized verbatim with `gradio_app.py`.
"""

import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE if (_HERE / "models").exists() else _HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.affordability import calculate_household_affordability
from src.inference import (
    UnifiedCommodityPredictor,
    UnsupportedCommodityError,
    UnsupportedMarketPairError,
)
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
    arrival_lag_1: float,
    rolling_arrival_mean_7: float,
    month: int,
    allow_experimental: bool,
):
    try:
        if input_mode == "Mode A: Historical Archive Replay":
            hist = get_historical_latest_features(commodity, market)
            res = unified.predict_validated(hist["features"])
            obs_info = (
                f"Observation: {hist['observation_date']} (Actual t+1: Rs. {hist['actual_next_price_rs_per_qtl']}/Qtl) | "
                f"Arrivals Available: {hist.get('arrivals_available')} ({hist.get('arrival_observation_date')})"
            )
        elif input_mode == "Mode A: Live API Fetch":
            live = fetch_latest_market_data(commodity, market)
            arr_txt = f"{live['arrival_quantity_tonnes']} Tonnes" if live.get("arrivals_available") else "Unavailable (Not Zero)"
            if not live.get("lag_features_ready"):
                return (
                    f"Live Fetch Status    : {live.get('freshness_label')}\n"
                    f"Matched API Market   : {live.get('matched_api_market', 'N/A')} (Rejected submarkets: {live.get('rejected_submarkets', [])})\n"
                    f"Price Observation    : {live.get('observation_date', 'N/A')} | Modal Rs. {live.get('observed_price_rs_per_qtl', 'N/A')}/Qtl\n"
                    f"Arrivals Telemetry   : {arr_txt}\n"
                    f"Mode A Withheld      : {live.get('message')}\n"
                    "Switch to Historical Archive Replay (Mode A) or Experimental Manual Exploration (Mode B)."
                )
            res = unified.predict_validated(live["features"])
            obs_info = f"Live Date: {live.get('observation_date')} | Arrivals: {arr_txt}"
        else:
            res = unified.predict_experimental_manual(
                commodity=commodity,
                market=market,
                modal_price=modal_price,
                price_lag_1=price_lag_1,
                price_lag_3=price_lag_3,
                price_lag_7=price_lag_7,
                arrival_tonnes=arrival_tonnes,
                arrival_lag_1=arrival_lag_1,
                rolling_arrival_mean_7=rolling_arrival_mean_7,
                daily_spread=150.0,
                month=int(month),
                day_of_week=2,
                allow_experimental_commodity=allow_experimental,
            )
            obs_info = "Manual Mode B exploration (past-lag subset approximation, excludes P(t))"

        ew = res["early_warning_classifier"]
        obs_ret = res["observed_retail_reference"]
        if obs_ret.get("retail_price_rs_per_kg") is not None:
            ret_txt = f"Rs. {obs_ret['retail_price_rs_per_kg']}/Kg ({obs_ret['provenance']})"
        else:
            ret_txt = "Unavailable (No official DCA archive record; never fabricated)"

        lines = [
            f"Inference Mode     : {res['inference_mode']} (Validated={res['is_validated_forecast']}, Schema={res['feature_schema_version']})",
            f"Commodity / Market : {res['commodity']} @ {res['market']}",
            f"Source / Horizon   : {obs_info} | {res['prediction_horizon']}",
            f"Current Modal      : Rs. {res['current_modal_price_rs_per_qtl']}/Qtl (Equiv Rs. {res['current_wholesale_equiv_rs_per_kg']}/Kg)",
            f"Predicted Next t+1 : Rs. {res['predicted_next_session_rs_per_qtl']}/Qtl ({res['predicted_change_pct']:+.2f}%)",
            f"Wholesale Equiv/Kg : Rs. {res['wholesale_equiv_rs_per_kg']}/Kg (unit conversion only — NOT retail prediction)",
            f"Observed DCA Retail: {ret_txt}",
            f"Retail ML Status   : {res['retail_estimation_status']}",
            f"Early Warning (ML) : {ew.get('predicted_regime')} | Probs={ew.get('class_probabilities')} | Reliable={ew.get('is_reliable_early_warning')}",
        ]
        if res.get("experimental_disclaimer"):
            lines.append(f"EXPERIMENTAL DISCLAIMER: {res['experimental_disclaimer']}")
        if res.get("approximations_used"):
            lines.append(f"Approximations Used: {'; '.join(res['approximations_used'])}")
        if res.get("validation_warning"):
            lines.append(f"Warnings           : {res['validation_warning']}")
        return "\n".join(lines)
    except (UnsupportedCommodityError, UnsupportedMarketPairError) as exc:
        return f"MODE A VALIDATION BLOCK:\n{exc}"
    except Exception as exc:
        return f"Error: {exc}"


def gradio_retail_summary() -> str:
    lines = ["Observed Official DCA Retail Reference Prices (No Rs.45 Fallbacks):"]
    for comm in ("Tomato", "Onion", "Moong Dal", "Sugar"):
        obs = get_observed_retail_price(comm)
        if obs.get("retail_price_rs_per_kg") is not None:
            lines.append(
                f"  - {comm}: Rs. {obs['retail_price_rs_per_kg']}/Kg | "
                f"{obs.get('centre')} | Date={obs.get('observation_date')} | Source={obs.get('source')}"
            )
        else:
            lines.append(f"  - {comm}: Unavailable (User Supplied price required)")
    return "\n".join(lines)


def gradio_affordability(
    income: float,
    sector: str,
    tomato_kg: float,
    onion_kg: float,
    dal_kg: float,
    sugar_kg: float,
    tomato_price_override: Optional[float] = None,
    onion_price_override: Optional[float] = None,
    dal_price_override: Optional[float] = None,
    sugar_price_override: Optional[float] = None,
    hypothetical_shock_pct: Optional[float] = None,
):
    try:
        overrides = {
            "Tomato": tomato_price_override,
            "Onion": onion_price_override,
            "Moong Dal": dal_price_override,
            "Sugar": sugar_price_override,
        }
        qtys = {"Tomato": tomato_kg, "Onion": onion_kg, "Moong Dal": dal_kg, "Sugar": sugar_kg}
        items = []
        missing_prices = []

        for comm in ("Tomato", "Onion", "Moong Dal", "Sugar"):
            qty = float(qtys[comm])
            ovr = overrides[comm]
            obs = get_observed_retail_price(comm)
            if ovr is not None and float(ovr) > 0:
                price = float(ovr)
                src = "User Supplied"
            elif obs.get("retail_price_rs_per_kg") is not None:
                price = float(obs["retail_price_rs_per_kg"])
                src = str(obs.get("provenance", "Archived DCA Observed"))
            else:
                missing_prices.append(comm)
                price = 0.0
                src = "Unavailable"

            pred_p: Optional[float] = None
            pred_src = "Unavailable"
            if hypothetical_shock_pct is not None and float(hypothetical_shock_pct) != 0.0 and price > 0:
                pred_p = round(price * (1.0 + float(hypothetical_shock_pct) / 100.0), 2)
                pred_src = "Hypothetical Sensitivity Scenario"

            items.append(
                {
                    "commodity_name": comm,
                    "monthly_quantity_kg": qty,
                    "current_retail_price_per_kg": price,
                    "current_price_source": src,
                    "predicted_retail_price_per_kg": pred_p,
                    "prediction_source": pred_src,
                }
            )

        if missing_prices:
            return (
                f"CANNOT COMPUTE COMPLETE BASKET: DCA retail price is Unavailable for {missing_prices}. "
                "Provide explicit User Supplied prices (Rs/Kg > 0); silent Rs.45 fallbacks are prohibited."
            )

        res = calculate_household_affordability(monthly_income=income, basket_items=items, sector=sector)
        lines = [
            f"Current Status     : {res['affordability_status']}",
            f"Current Basket Cost: Rs. {res['current_food_basket_cost_rs']:.2f}",
            f"Food Budget Share  : {res['current_food_budget_share_pct']:.2f}%",
            f"Remaining Income   : Rs. {res['discretionary_income_remaining_rs']:.2f}",
            f"Retail ML Preds    : {res['prediction_status']} (retail ML unavailable; current prices never copied silently as future predictions)",
        ]
        if res["has_complete_predictions"]:
            lines.extend(
                [
                    "Hypothetical Scenario Output (NOT an ML Prediction):",
                    f"  Scenario Cost    : Rs. {res['predicted_food_basket_cost_rs']:.2f} ({res['monthly_inflation_impact_rs']:+.2f} Rs)",
                    f"  Scenario Share   : {res['predicted_food_budget_share_pct']:.2f}% ({res['budget_share_change_pct_points']:+.2f} pp)",
                ]
            )
        lines.append("Item Provenance:")
        for b in res["item_breakdown"]:
            lines.append(f"  - {b['commodity_name']}: Rs. {b['current_retail_price_per_kg']}/Kg [{b['current_price_source']}]")
        lines.append(f"Disclaimer: {res['methodology_disclaimer']}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Affordability Error: {exc}"


def gradio_market_compare(commodity: str):
    try:
        mkts = get_markets_for_commodity(commodity, min_rows=10)[:4]
        comp = compare_markets(commodity, mkts)
        return (
            comp["summary_df"].to_string(index=False)
            + "\n\nOverlapping Dates Comparison:\n"
            + comp["overlapping_comparison_df"].to_string(index=False)
        )
    except Exception as exc:
        return f"Comparison Error: {exc}"


with gr.Blocks(title="Food Inflation & Household Affordability Monitor") as demo:
    gr.Markdown("# AI Food Inflation Prediction & Household Affordability Monitor (`v3.1.0-parity`)")
    with gr.Tab("Price Forecasting"):
        c_in = gr.Textbox(value="Tomato", label="Commodity (Validated: Tomato, Onion, Moong Dal, Sugar)")
        m_in = gr.Dropdown(["Pune", "Mumbai", "Lasalgaon"], value="Pune", label="Exact Canonical Market")
        mode_in = gr.Radio(
            [
                "Mode A: Historical Archive Replay",
                "Mode A: Live API Fetch",
                "Mode B: Experimental Manual Exploration",
            ],
            value="Mode A: Historical Archive Replay",
            label="Inference Mode",
        )
        with gr.Row():
            mp = gr.Number(value=1400, label="Current Modal P(t) (Rs/Qtl)")
            l1 = gr.Number(value=1350, label="Lag 1 P(t-1)")
            l3 = gr.Number(value=1280, label="Lag 3 P(t-3)")
            l7 = gr.Number(value=1200, label="Lag 7 P(t-7)")
        with gr.Row():
            arr = gr.Number(value=120, label="Arrivals Today (Tonnes)")
            arr1 = gr.Number(value=115, label="Arrivals t-1 (Tonnes)")
            arr_m = gr.Number(value=125, label="7d Prior Arrival Mean (Tonnes)")
            mo = gr.Slider(1, 12, value=11, step=1, label="Month")
        exp = gr.Checkbox(value=False, label="Allow Mode B Experimental Unseen Commodity")
        btn = gr.Button("Run Forecast")
        out = gr.Textbox(lines=12, label="Forecast Output")
        btn.click(gradio_predict, [c_in, m_in, mode_in, mp, l1, l3, l7, arr, arr1, arr_m, mo, exp], out)

    with gr.Tab("Market Comparison"):
        mc = gr.Dropdown(["Onion", "Tomato", "Sugar", "Moong Dal"], value="Onion", label="Commodity")
        mbtn = gr.Button("Compare Exact APMC Markets")
        mout = gr.Textbox(lines=14, label="Cross-Market Historical Comparison")
        mbtn.click(gradio_market_compare, [mc], mout)

    with gr.Tab("Household Affordability"):
        gr.Markdown(gradio_retail_summary())
        inc = gr.Number(value=15000, label="Monthly Income (Rs.)")
        sec = gr.Radio(["Urban", "Rural"], value="Urban", label="Sector")
        with gr.Row():
            tk = gr.Number(value=12, label="Tomato (Kg/mo)")
            ok = gr.Number(value=15, label="Onion (Kg/mo)")
            dk = gr.Number(value=4, label="Moong Dal (Kg/mo)")
            sk = gr.Number(value=3, label="Sugar (Kg/mo)")
        with gr.Row():
            tp = gr.Number(value=None, label="User Retail Tomato Rs/Kg (blank=use DCA if available)")
            op = gr.Number(value=None, label="User Retail Onion Rs/Kg (blank=use DCA if available)")
            dp = gr.Number(value=None, label="User Retail Moong Dal Rs/Kg (blank=use DCA if available)")
            sp = gr.Number(value=None, label="User Retail Sugar Rs/Kg (blank=use DCA if available)")
        shock = gr.Number(value=None, label="Optional Hypothetical Scenario Retail Shock % (NOT ML prediction)")
        abtn = gr.Button("Assess Affordability")
        aout = gr.Textbox(lines=12, label="Affordability Output")
        abtn.click(gradio_affordability, [inc, sec, tk, ok, dk, sk, tp, op, dp, sp, shock], aout)

if __name__ == "__main__":
    demo.launch()
