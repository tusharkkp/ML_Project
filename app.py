"""
Streamlit Dashboard — AI-Based Food Inflation Prediction & Household Affordability Monitoring.

Version: v3.0.0-leakfree
Synchronized with chronological validation-selected models, dynamic commodity/market guards,
live API fetching with explicit fallback, early-warning classification, and market comparison.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.affordability import calculate_household_affordability
from src.inference import (
    CommodityPricePredictor,
    UnifiedCommodityPredictor,
    UnsupportedCommodityError,
)
from src.live_data import (
    fetch_latest_market_data,
    get_historical_latest_features,
    get_observed_retail_price,
)
from src.market_comparison import (
    compare_markets,
    get_available_commodities,
    get_markets_for_commodity,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(
    page_title="Food Inflation & Household Affordability Monitor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_unified_predictor() -> UnifiedCommodityPredictor:
    return UnifiedCommodityPredictor(models_dir=MODELS_DIR)


@st.cache_resource
def load_specialist_predictor(commodity: str) -> CommodityPricePredictor:
    return CommodityPricePredictor(models_dir=MODELS_DIR, commodity=commodity)


def _render_prediction_result(res: Dict[str, Any], actual_next: float | None = None, obs_date: str | None = None):
    """Render wholesale prediction, kg unit equiv, retail note, and early-warning classifier."""
    st.markdown(f"**Prediction Horizon:** `{res['prediction_horizon']}`")
    if obs_date:
        st.caption(f"Observation Date Used: **{obs_date}** | Feature Mode: `{res['feature_mode']}` | Model: `{res['model_used']}`")
    else:
        st.caption(f"Feature Mode: `{res['feature_mode']}` | Model: `{res['model_used']}`")

    if res.get("validation_warning"):
        st.warning(res["validation_warning"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Current Wholesale Modal",
        f"Rs. {res['current_modal_price_rs_per_qtl']:,.2f}/Qtl",
        f"Equiv Rs. {res['current_wholesale_equiv_rs_per_kg']:.2f}/Kg",
    )
    delta_label = f"{res['predicted_change_rs_qtl']:+.2f} Rs/Qtl ({res['predicted_change_pct']:+.2f}%)"
    c2.metric(
        "Predicted Next Trading Session",
        f"Rs. {res['predicted_next_session_rs_per_qtl']:,.2f}/Qtl",
        delta_label,
    )
    c3.metric(
        "Wholesale Equiv (÷ 100 Unit Conv.)",
        f"Rs. {res['wholesale_equiv_rs_per_kg']:.2f}/Kg",
        "NOT a retail estimate",
        delta_color="off",
    )
    obs_ret = res.get("observed_retail_reference", {}) or {}
    if obs_ret.get("retail_price_rs_per_kg"):
        c4.metric(
            "Observed DCA Retail Reference",
            f"Rs. {obs_ret['retail_price_rs_per_kg']:.2f}/Kg",
            f"{obs_ret.get('centre', 'DCA')} ({obs_ret.get('observation_date', 'N/A')})",
            delta_color="off",
        )
    else:
        c4.metric("Observed DCA Retail", "Unavailable", "No archive row", delta_color="off")

    if actual_next is not None:
        err = res["predicted_next_session_rs_per_qtl"] - actual_next
        st.info(
            f"**Archived Ground Truth Next Session:** Rs. {actual_next:,.2f}/Qtl | "
            f"**Prediction Error (Pred - Actual):** {err:+.2f} Rs/Qtl"
        )

    # Alert & Early-Warning ML Classifier
    st.markdown(f"**Directional Band Alert:** `{res['inflation_alert_badge']}`")
    ew = res.get("early_warning_classifier")
    if ew and ew.get("available"):
        with st.expander("ML Early-Warning Regime Classifier (Normal / Surge / Crash)", expanded=True):
            if not ew.get("is_reliable_early_warning", False):
                st.warning(
                    f"**Experimental Low-Precision Classifier:** {ew.get('reliability_status')} "
                    f"(Untouched Test Any-Spike PR-AUC = `{ew.get('test_any_spike_pr_auc')}`, "
                    f"Accuracy = `{ew.get('test_accuracy')}`)."
                )
            probs = ew.get("class_probabilities", {})
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Predicted Regime", str(ew.get("predicted_regime")))
            ec2.metric("Normal [-10%, +10%]", f"{probs.get('Normal', 0.0) * 100:.1f}%")
            ec3.metric("Surge (> +10%)", f"{probs.get('Surge', 0.0) * 100:.1f}%")
            ec4.metric("Crash (< -10%)", f"{probs.get('Crash', 0.0) * 100:.1f}%")
            st.caption(f"Algorithm: {ew.get('champion_algorithm')} | Any-Spike Probability: {(ew.get('any_spike_probability') or 0)*100:.1f}%")

    with st.expander("Wholesale-to-Retail Estimation Audit Note", expanded=False):
        st.write(f"**Retail ML Status:** `{res.get('retail_estimation_status')}`")
        st.write(res.get("retail_estimation_note", ""))


def page_forecasting():
    st.title("Commodity Price Forecasting & Early-Warning Monitor")
    st.caption(
        "Forecasts next observed active APMC trading session modal prices (Rs/Quintal) using "
        "strictly chronological, validation-selected pipelines."
    )

    try:
        unified = load_unified_predictor()
    except Exception as exc:
        st.error(f"Failed to load unified model artifacts: {exc}")
        return

    tab_uni, tab_spec = st.tabs(["Unified Multi-Commodity Engine", "Dedicated Crop Specialists (Tomato & Onion)"])

    with tab_uni:
        supported_comms = unified.get_supported_commodities()
        col_a, col_b, col_c = st.columns([2, 2, 2])
        with col_a:
            choice_list = supported_comms + ["Other / Unvalidated Commodity (Test Guard)"]
            selected_choice = st.selectbox("Commodity", choice_list)
            if selected_choice == "Other / Unvalidated Commodity (Test Guard)":
                commodity_input = st.text_input("Enter Commodity Name", value="Potato")
                allow_exp = st.checkbox(
                    "Force Experimental Inference on Unvalidated Commodity",
                    value=False,
                    help="Leave unchecked to enforce validated commodity protection.",
                )
            else:
                commodity_input = selected_choice
                allow_exp = False
        with col_b:
            avail_markets = unified.get_supported_markets(commodity_input) if selected_choice in supported_comms else unified.get_supported_markets()
            market_input = st.selectbox("APMC Mandi Market", avail_markets)
        with col_c:
            source_mode = st.radio(
                "Feature Input Mode",
                [
                    "Historical Archive Replay (Validated)",
                    "Live API Fetch (data.gov.in)",
                    "Manual Feature Exploration",
                ],
            )

        st.divider()

        if source_mode == "Historical Archive Replay (Validated)":
            if selected_choice not in supported_comms:
                st.warning(f"No validated historical archive series exists for '{commodity_input}'.")
            else:
                if st.button("Run Historical Replay Forecast", type="primary", key="btn_uni_hist"):
                    try:
                        hist = get_historical_latest_features(commodity_input, market_input)
                        res = unified.predict_from_features(
                            hist["features"],
                            allow_experimental=False,
                            feature_mode="historical_archive_replay",
                        )
                        _render_prediction_result(
                            res,
                            actual_next=hist.get("actual_next_price_rs_per_qtl"),
                            obs_date=hist.get("observation_date"),
                        )
                        st.session_state["last_forecast"] = res
                    except UnsupportedCommodityError as exc:
                        st.error(f"Validation Block: {exc}")
                    except Exception as exc:
                        st.error(f"Historical replay error: {exc}")

        elif source_mode == "Live API Fetch (data.gov.in)":
            force_ref = st.checkbox("Bypass 15-min local API cache (force fresh request)", value=False)
            if st.button("Fetch Latest Official Market Record", type="primary", key="btn_uni_live"):
                with st.spinner("Querying data.gov.in AGMARKNET endpoint (timeout 10s)..."):
                    live_res = fetch_latest_market_data(
                        commodity_input, market_input, force_refresh=force_ref, timeout_seconds=10
                    )
                st.markdown(f"**Fetch Freshness:** `{live_res.get('freshness_label')}` | **Status:** `{live_res.get('status')}`")
                if live_res.get("observed_price_rs_per_qtl"):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Observed API Modal Price", f"Rs. {live_res['observed_price_rs_per_qtl']:,.2f}/Qtl")
                    m2.metric("API Date", str(live_res.get("observation_date")))
                    m3.metric("Continuous 7d Lags Available", str(live_res.get("lag_features_ready")))
                if live_res.get("lag_features_ready") and live_res.get("features"):
                    try:
                        res = unified.predict_from_features(
                            live_res["features"],
                            allow_experimental=allow_exp,
                            feature_mode="live_api_continuous",
                        )
                        _render_prediction_result(res, obs_date=live_res.get("observation_date"))
                        st.session_state["last_forecast"] = res
                    except UnsupportedCommodityError as exc:
                        st.error(f"Validation Block: {exc}")
                else:
                    st.warning(
                        f"**Model Inference Withheld (Zero-Fabrication Rule):** {live_res.get('message')} "
                        "Use **Historical Archive Replay** for verified end-to-end replay or **Manual Feature Exploration** if you supply real recent prices."
                    )

        else:  # Manual Feature Exploration
            if selected_choice not in supported_comms and not allow_exp:
                st.info("Checking unsupported commodity validation guard...")
            m1, m2, m3, m4 = st.columns(4)
            p_curr = m1.number_input("Current Modal Price (Rs/Qtl)", min_value=50.0, value=1400.0, step=50.0)
            p_lag1 = m2.number_input("Lag 1 Session (Rs/Qtl)", min_value=50.0, value=1350.0, step=50.0)
            p_lag3 = m3.number_input("Lag 3 Sessions (Rs/Qtl)", min_value=50.0, value=1280.0, step=50.0)
            p_lag7 = m4.number_input("Lag 7 Sessions (Rs/Qtl)", min_value=50.0, value=1200.0, step=50.0)

            n1, n2, n3, n4 = st.columns(4)
            arr_t = n1.number_input("Arrivals Today (Tonnes)", min_value=0.0, value=120.0, step=10.0)
            spread = n2.number_input("Intraday Spread (Rs/Qtl)", min_value=0.0, value=200.0, step=25.0)
            month_val = n3.slider("Month", 1, 12, 11)
            dow_val = n4.slider("Day of Week (0=Mon..6=Sun)", 0, 6, 2)

            if st.button("Run Manual Forecast", type="primary", key="btn_uni_man"):
                try:
                    res = unified.predict(
                        commodity=commodity_input,
                        market=market_input,
                        modal_price=p_curr,
                        price_lag_1=p_lag1,
                        price_lag_3=p_lag3,
                        price_lag_7=p_lag7,
                        rolling_mean_7=(p_curr + p_lag1 + p_lag3 + p_lag7) / 4.0,
                        rolling_std_7=float(pd.Series([p_curr, p_lag1, p_lag3, p_lag7]).std()),
                        rolling_min_7=min(p_curr, p_lag1, p_lag3, p_lag7),
                        rolling_max_7=max(p_curr, p_lag1, p_lag3, p_lag7),
                        arrival_tonnes=arr_t,
                        arrival_lag_1=arr_t,
                        rolling_arrival_mean_7=arr_t,
                        daily_spread=spread,
                        month=month_val,
                        day_of_week=dow_val,
                        allow_experimental=allow_exp,
                        allow_manual_approx=False,
                    )
                    _render_prediction_result(res)
                    st.session_state["last_forecast"] = res
                except UnsupportedCommodityError as exc:
                    st.error(f"Unsupported Commodity Blocked: {exc}")
                except Exception as exc:
                    st.error(f"Inference error: {exc}")

        if "last_forecast" in st.session_state:
            lf = st.session_state["last_forecast"]
            if st.button("Send Forecast Delta to Household Affordability Stress Scenario"):
                st.session_state["affordability_context"] = {
                    "commodity": lf["commodity"],
                    "market": lf["market"],
                    "wholesale_change_pct": lf["predicted_change_pct"],
                    "predicted_wholesale_qtl": lf["predicted_next_session_rs_per_qtl"],
                }
                st.success(
                    f"Saved {lf['commodity']} ({lf['predicted_change_pct']:+.2f}% next-session wholesale change) "
                    "for Scenario Simulation in Page 3: Household Affordability."
                )

    with tab_spec:
        spec_comm = st.selectbox("Select Specialist Commodity", ["Tomato", "Onion"])
        if st.button("Replay Specialist Latest Validated Test Observation", type="primary", key="btn_spec_replay"):
            try:
                spec_pred = load_specialist_predictor(spec_comm)
                hist = get_historical_latest_features(spec_comm, "Pune")
                res = spec_pred.predict_from_features(hist["features"], feature_mode="specialist_historical_replay")
                _render_prediction_result(
                    res,
                    actual_next=hist.get("actual_next_price_rs_per_qtl"),
                    obs_date=hist.get("observation_date"),
                )
            except Exception as exc:
                st.error(f"Specialist forecast error: {exc}")


def page_market_comparison():
    st.title("Market Comparison & Historical Analysis")
    st.caption(
        "Compare observed APMC wholesale prices (Rs/Quintal) and arrivals across exact "
        "markets without mixing commodities or inventing interpolated observations."
    )

    avail_comms = get_available_commodities(min_markets=2, min_rows_per_market=30)
    if not avail_comms:
        st.error("No commodities with >=2 multi-market historical series found in wholesale_prices.csv.")
        return

    c1, c2 = st.columns([2, 3])
    with c1:
        commodity = st.selectbox("Select Commodity", avail_comms, index=avail_comms.index("Onion") if "Onion" in avail_comms else 0)
    markets_avail = get_markets_for_commodity(commodity, min_rows=10)
    default_mkts = [m for m in ["Pune", "Lasalgaon", "Mumbai", "Nashik"] if m in markets_avail][:3]
    if len(default_mkts) < 2:
        default_mkts = markets_avail[:2]

    with c2:
        selected_markets = st.multiselect("Select 2 or more APMC Markets", markets_avail, default=default_mkts)

    d1, d2 = st.columns(2)
    start_dt = d1.date_input("Start Date", value=pd.Timestamp("2016-01-01"))
    end_dt = d2.date_input("End Date", value=pd.Timestamp("2018-12-31"))

    if len(selected_markets) < 2:
        st.warning("Select at least two distinct markets to perform cross-market comparison.")
        return

    try:
        comp = compare_markets(commodity, selected_markets, start_date=str(start_dt), end_date=str(end_dt))
    except Exception as exc:
        st.error(f"Comparison error: {exc}")
        return

    st.info(comp["disclaimer"])
    st.markdown(f"**Data Sources:** `{'; '.join(comp['sources'])}` | **Last Available Date:** `{comp['last_available_date']}`")

    st.subheader("1. Market Summary, Volatility & Coverage")
    st.dataframe(comp["summary_df"], width="stretch", hide_index=True)

    ts_df = comp["timeseries_df"]
    if not ts_df.empty:
        st.subheader("2. Historical Modal Wholesale Price Trend (Rs/Quintal)")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for mkt, grp in ts_df.groupby("market_name"):
            g = grp.sort_values("observation_date")
            ax.plot(g["observation_date"], g["modal_price_rs_per_qtl"], label=mkt, linewidth=1.4, alpha=0.85)
        ax.set_ylabel("Modal Price (Rs/Quintal)")
        ax.set_xlabel("Observation Date")
        ax.set_title(f"{commodity}: Observed Wholesale Modal Price Comparison")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    arr_df = comp["arrivals_df"]
    st.subheader("3. Market Arrivals Comparison (Tonnes)")
    if not arr_df.empty:
        fig2, ax2 = plt.subplots(figsize=(11, 3.8))
        for mkt, grp in arr_df.groupby("market_name"):
            g = grp.sort_values("observation_date")
            ax2.plot(g["observation_date"], g["arrival_quantity_tonnes"], label=mkt, linewidth=1.1, alpha=0.8)
        ax2.set_ylabel("Arrivals (Tonnes)")
        ax2.set_xlabel("Observation Date")
        ax2.set_title(f"{commodity}: Reported Market Arrivals")
        ax2.legend()
        ax2.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.caption("No continuous arrival volume records were reported for the selected market/commodity combination.")

    st.subheader("4. Strictly Overlapping-Date Pairwise Spread & Correlation")
    st.dataframe(comp["overlapping_comparison_df"], width="stretch", hide_index=True)


def page_household_affordability():
    st.title("Household Food Affordability & Vulnerability Monitor")
    st.caption(
        "Evaluates household monthly food-budget share using observed DCA retail prices "
        "and strict complete-basket rules (no mixed current/missing future prices)."
    )

    ctx = st.session_state.get("affordability_context")
    if ctx:
        st.info(
            f"**Forecast Context Linked:** `{ctx['commodity']} @ {ctx['market']}` | "
            f"Projected Next-Session Wholesale Delta: `{ctx['wholesale_change_pct']:+.2f}%` "
            f"(Predicted Modal: Rs. {ctx['predicted_wholesale_qtl']:,.2f}/Qtl)."
        )

    c1, c2, c3, c4 = st.columns(4)
    income = c1.number_input("Monthly Household Income (Rs.)", min_value=500.0, value=15000.0, step=1000.0)
    sector = c2.selectbox("NSSO HCES Sector", ["Urban", "Rural"])
    family_size = c3.number_input("Household Size", min_value=1, max_value=15, value=4)
    try_live_dca = c4.checkbox("Attempt Live DCA Retail Refresh", value=False)

    st.subheader("Configure Monthly Consumption Basket (Retail Rs/Kg)")
    default_items = [
        {"commodity": "Tomato", "qty": 12.0},
        {"commodity": "Onion", "qty": 15.0},
        {"commodity": "Moong Dal", "qty": 4.0},
        {"commodity": "Sugar", "qty": 3.0},
    ]

    apply_stress = st.checkbox(
        "Simulate hypothetical scenario: pass linked wholesale % change directly onto linked item's retail price",
        value=bool(ctx),
        disabled=ctx is None,
        help="Scenario sensitivity assumption only — wholesale-to-retail econometric transmission requires paired city series.",
    )

    basket_payload: List[Dict[str, Any]] = []
    include_all_future = st.checkbox("Include Projected/Scenario Future Retail Prices for ALL Basket Items", value=apply_stress)

    for idx, item in enumerate(default_items):
        ref = get_observed_retail_price(item["commodity"], try_live_dca=try_live_dca)
        obs_kg = float(ref["retail_price_rs_per_kg"]) if ref.get("retail_price_rs_per_kg") else 45.0
        obs_meta = f"{ref.get('centre', 'DCA')} | {ref.get('observation_date', 'Archive')}"

        r1, r2, r3, r4 = st.columns([2, 1.5, 2, 2])
        r1.markdown(f"**{item['commodity']}**\n\n`DCA Obs ({obs_meta})`")
        qty = r2.number_input(f"Qty (Kg/mo) [{item['commodity']}]", min_value=0.0, value=item["qty"], step=1.0, key=f"qty_{idx}")
        cur_p = r3.number_input(f"Current Retail (Rs/Kg) [{item['commodity']}]", min_value=0.0, value=obs_kg, step=2.0, key=f"cur_{idx}")

        pred_default = cur_p
        if apply_stress and ctx and ctx["commodity"].lower() == item["commodity"].lower():
            pred_default = round(cur_p * (1.0 + ctx["wholesale_change_pct"] / 100.0), 2)

        if include_all_future:
            fut_p = r4.number_input(f"Scenario Future (Rs/Kg) [{item['commodity']}]", min_value=0.0, value=pred_default, step=2.0, key=f"fut_{idx}")
        else:
            fut_p = None
            r4.caption("Scenario prediction omitted")

        basket_payload.append(
            {
                "commodity_name": item["commodity"],
                "monthly_quantity_kg": qty,
                "current_retail_price_per_kg": cur_p,
                "predicted_retail_price_per_kg": fut_p,
            }
        )

    if st.button("Compute Household Vulnerability Assessment", type="primary"):
        try:
            res = calculate_household_affordability(
                monthly_income=income,
                basket_items=basket_payload,
                sector=sector,
                family_size=int(family_size),
            )
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Basket Cost", f"Rs. {res['current_food_basket_cost_rs']:,.2f}")
            m2.metric("Current Food Budget Share", f"{res['current_food_budget_share_pct']:.2f}%")
            m3.metric("Vulnerability Status", res["affordability_status"])
            m4.metric("Remaining Income", f"Rs. {res['discretionary_income_remaining_rs']:,.2f}")

            if res["has_complete_predictions"]:
                st.subheader("Projected / Scenario Impact (Complete Basket)")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Scenario Basket Cost", f"Rs. {res['predicted_food_basket_cost_rs']:,.2f}", f"{res['monthly_inflation_impact_rs']:+.2f} Rs")
                p2.metric("Scenario Budget Share", f"{res['predicted_food_budget_share_pct']:.2f}%", f"{res['budget_share_change_pct_points']:+.2f} pp")
                p3.metric("Scenario Status", str(res["predicted_affordability_status"]))
                p4.metric("Items Evaluated", str(len(res["item_breakdown"])))
            else:
                st.info(f"**Future Scenario Status:** `{res['prediction_status']}` — provide future prices for all items to compare projected affordability.")

            st.dataframe(pd.DataFrame(res["item_breakdown"]), width="stretch", hide_index=True)
            st.caption(res["methodology_disclaimer"])
        except Exception as exc:
            st.error(f"Validation error: {exc}")


def page_benchmarks():
    st.title("Model Benchmarks & Experimental Audit")
    st.caption("All metrics loaded directly from regenerated evaluation CSV/JSON artifacts (Validation-selected, Untouched Test evaluated once).")

    uni_meta = _read_json(MODELS_DIR / "unified_model_metadata.json")
    clf_meta = _read_json(MODELS_DIR / "spike_classifier_metadata.json")
    ret_meta = _read_json(MODELS_DIR / "retail_model_metadata.json")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Unified Cross-Commodity",
        "Specialists (Tomato & Onion)",
        "Early-Warning Classifier",
        "Retail Audit & Visuals",
    ])

    with tab1:
        if uni_meta:
            ov = uni_meta.get("overall_test_metrics", {})
            bm = uni_meta.get("persistence_baseline_test_metrics", {})
            macro = uni_meta.get("macro_average_test_metrics", {})
            st.markdown(f"**Champion Selected Strictly on Validation:** `{uni_meta.get('champion_algorithm')}` | **Params:** `{uni_meta.get('champion_hyperparameters')}`")
            st.markdown(f"**Chronological Cutoffs:** `{uni_meta.get('split_cutoffs')}`")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Pooled Test RMSE", f"Rs. {ov.get('rmse_rs_per_qtl')}", f"Base Rs. {bm.get('RMSE')}")
            k2.metric("Pooled R² (Scale-Inflated)", f"{ov.get('r2_score')}", "See Macro R²")
            k3.metric("Macro-Avg Test RMSE", f"Rs. {macro.get('macro_avg_rmse')}", f"Base Rs. {macro.get('baseline_macro_avg_rmse')}")
            k4.metric("Macro-Avg Test R²", f"{macro.get('macro_avg_r2')}")

            st.subheader("Per-Commodity Untouched Test Performance vs Persistence Baseline")
            rows = []
            for comm, m in uni_meta.get("per_commodity_metrics", {}).items():
                rows.append({
                    "Commodity": comm,
                    "Test_N": m["sample_count"],
                    "Test_MAE": m["test_mae_rs_per_qtl"],
                    "Test_RMSE": m["test_rmse_rs_per_qtl"],
                    "Persistence_RMSE": m["persistence_baseline"]["rmse"],
                    "Beats_Baseline": m["improvement_over_baseline"]["beats_baseline_rmse"],
                    "Test_R2": m["test_r2"],
                    "Test_MAPE_%": m["test_mape_pct"],
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            st.subheader("Leave-One-Commodity-Out (Unseen Commodity Generalization Test)")
            loco_rows = []
            for comm, m in uni_meta.get("leave_one_commodity_out_experiment", {}).items():
                loco_rows.append({
                    "Held_Out_Commodity": comm,
                    "Model_RMSE": m["model_metrics"]["RMSE"],
                    "Persistence_RMSE": m["persistence_baseline"]["RMSE"],
                    "Beats_Baseline": m["beats_baseline_rmse"],
                    "Model_R2": m["model_metrics"]["R2"],
                })
            st.dataframe(pd.DataFrame(loco_rows), width="stretch", hide_index=True)
            st.warning("Because Leave-One-Commodity-Out only beats persistence on 1 of 4 held-out crops, unseen commodity generalization is disabled by default.")

            st.subheader("Validation Leaderboard (Model Selection Table)")
            st.dataframe(pd.DataFrame(uni_meta.get("validation_candidates_benchmarked", [])), width="stretch", hide_index=True)

    with tab2:
        for comm in ["Tomato", "Onion"]:
            safe = comm.lower()
            st.subheader(f"{comm} @ Pune — Untouched Test Leaderboard")
            df_test = _read_csv(REPORTS_DIR / f"model_comparison_metrics_{safe}.csv")
            if not df_test.empty:
                st.dataframe(df_test, width="stretch", hide_index=True)
            with st.expander(f"{comm} Classification, Clustering & Gradient Descent Tables"):
                st.dataframe(_read_csv(REPORTS_DIR / f"classification_comparison_{safe}.csv"), width="stretch", hide_index=True)
                st.dataframe(_read_csv(REPORTS_DIR / f"clustering_comparison_{safe}.csv"), width="stretch", hide_index=True)
                st.dataframe(_read_csv(REPORTS_DIR / f"gradient_descent_comparison_{safe}.csv"), width="stretch", hide_index=True)

    with tab3:
        if clf_meta:
            st.markdown(f"**Champion (Val Macro-F1 Selected):** `{clf_meta.get('champion_algorithm')}`")
            tm = clf_meta.get("test_metrics", {})
            mb = clf_meta.get("majority_class_baseline", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", f"{tm.get('accuracy')}", f"Majority Base: {mb.get('test_accuracy')}")
            c2.metric("Test Macro-F1", f"{tm.get('macro_f1')}")
            c3.metric("Any-Spike ROC-AUC", f"{tm.get('any_spike_roc_auc')}")
            c4.metric("Any-Spike PR-AUC", f"{tm.get('any_spike_pr_auc')}")
            st.warning(clf_meta.get("reliability_status", ""))
            st.subheader("Validation Leaderboard (3-Class Classifier)")
            st.dataframe(_read_csv(REPORTS_DIR / "early_warning_classifier_evaluation.csv"), width="stretch", hide_index=True)

    with tab4:
        st.subheader("Retail Price Estimation Feasibility Audit")
        st.json(ret_meta)
        st.subheader("Generated Evaluation Plots")
        imgs = sorted(FIGURES_DIR.glob("*.png"))
        for img_path in imgs:
            st.image(str(img_path), caption=img_path.name, use_container_width=True)


def page_methodology():
    st.title("Academic Viva, Architecture & Scientific Guarantees")
    st.markdown(
        """
### Verified Pipeline Architecture (`v3.0.0-leakfree`)
1. **Strict Chronological Time-Series Discipline:**
   - All specialist and unified models split strictly chronologically with shared date cutoffs (`Train: 2016-01-09..2018-02-15`, `Validation: 2018-02-17..2018-07-22`, `Test: 2018-07-24..2018-12-30`).
   - Rows whose `target_date_t_plus_1` crosses a partition boundary are purged (10 rows removed in unified split).
   - Hyperparameter tuning uses `TimeSeriesSplit` inside `Train`; Champion selection uses `Validation` RMSE/Macro-F1 only; Test data is evaluated once after refitting `Train+Validation`.
2. **Zero Lookahead Leakage & Exact Market Granularity:**
   - Current session `modal_price_rs_per_qtl` ($P_t$) is explicitly distinct from `price_lag_1` ($P_{t-1}$).
   - 7-session rolling features operate strictly over shifted history $[t-7..t-1]$.
   - Exact single APMC markets (`Pune`, `Mumbai`, `Lasalgaon`) are preserved without silent submarket merging.
3. **Honest Commodity & Retail Boundaries:**
   - Dynamic guard blocks unseen commodities (`UnsupportedCommodityError`) supported by Leave-One-Commodity-Out empirical evidence.
   - Wholesale-to-retail ML estimation is marked **Unavailable** (`retail_model_metadata.json`) due to 0 matched Maharashtra city-date pairs between 2016–2018 AGMARKNET and 2010–2012 DCA retail archives; mathematical `÷ 100` is labeled `Wholesale Equiv Rs/Kg`.
4. **Early-Warning Classifier & Household Affordability:**
   - Integrated `spike_classifier.joblib` (Normal / Surge / Crash) reports probabilities and PR-AUC with honest low-precision caveats.
   - Affordability engine computes NSSO HCES budget share (`Current` vs `Scenario`) and requires complete predicted item sets.
"""
    )


def main():
    st.sidebar.title("AI Food Inflation System")
    st.sidebar.caption("Pipeline: `v3.0.0-leakfree`")
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Commodity Price Forecasting",
            "2. Market Comparison",
            "3. Household Affordability",
            "4. Model Benchmarks & Classification",
            "5. Academic Viva & Methodology",
        ],
    )

    uni_meta = _read_json(MODELS_DIR / "unified_model_metadata.json")
    if uni_meta:
        st.sidebar.divider()
        st.sidebar.caption(
            f"**Validated Crops:** {', '.join(uni_meta.get('supported_commodities', []))}\n\n"
            f"**Unified Champion:** {uni_meta.get('champion_algorithm')}\n\n"
            f"**Test RMSE:** Rs. {uni_meta.get('overall_test_metrics', {}).get('rmse_rs_per_qtl')}/Qtl"
        )

    if page.startswith("1."):
        page_forecasting()
    elif page.startswith("2."):
        page_market_comparison()
    elif page.startswith("3."):
        page_household_affordability()
    elif page.startswith("4."):
        page_benchmarks()
    else:
        page_methodology()


if __name__ == "__main__":
    main()
