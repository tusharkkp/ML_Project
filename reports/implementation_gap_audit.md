# Initial Implementation & Methodology Gap Audit

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Audit Date:** 2026-11-27  
**Scope:** Full-stack inspection of data pipelines (`scripts/`), inference & affordability modules (`src/`), saved artifacts (`models/`, `data/processed/`, `reports/`), Streamlit & Gradio UI (`app.py`, `gradio_app.py`), tests (`tests/`), and documentation (`README.md`).

---

## 1. Executive Summary of Audit Findings

Inspection of the current repository revealed a **critical synchronization mismatch** and **eight substantive methodological / engineering gaps** across the Python source code:

1. **Repository Sync Mismatch (Broken Execution):**
   - `models/` (`unified_food_price_model.joblib`, `unified_model_metadata.json`, `model_metadata_*.json`, `spike_classifier.joblib`) and `data/processed/model_ready_unified_commodities_clean.csv` reflect an updated `v3.0.0` feature schema (requiring `market_name` and `modal_price_rs_per_qtl`), while `scripts/` and `src/` currently on disk (`commit bb6e86b`) follow the older `v2.0.0` schema.
   - Running tests immediately confirms breakage: `tests/test_model_pipeline.py::test_unified_predictor_inference` fails with `KeyError: "['market_name', 'modal_price_rs_per_qtl'] not in index"`, and `scripts/inference_test.py` fails with `KeyError: 'target_definition'`.

---

## 2. Phase-by-Phase Technical Gap Inventory

### Phase 1: ML Correctness, Leakage & Methodology Gaps
1. **Test-Set Leakage in Model Selection:**
   - [`scripts/train_and_evaluate.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_and_evaluate.py) selects `winning_model` by comparing `test_res_rf['RMSE']` against `test_res_lr['RMSE']` directly on `X_test`.
   - [`scripts/train_unified_model.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_unified_model.py) evaluates every candidate pipeline directly on `X_test` (`results_df.sort_values('RMSE')`) and selects the champion using test RMSE without ever evaluating `X_val`.
   - [`scripts/train_all_models.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_all_models.py) fits directly on combined `Train+Val` without validation-based model selection and picks the best regression model by test `RMSE`.
2. **Unified Chronological Split Overlap & Boundary Leakage:**
   - [`scripts/build_features.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/build_features.py) applies `add_split_tags()` independently per commodity-market file before concatenating into `model_ready_unified_commodities.csv` (`01-01-2017` mixed date string formatting), causing uneven split dates across commodities and lacking `target_date_t_plus_1` boundary checks across `Train → Validation → Test`.
3. **Silent Submarket Mixing & Inconsistent Lag Semantics:**
   - `scripts/build_features.py` uses `.str.contains(market_keyword)` (silently grouping `Pune`, `Pune(Hadapsar)`, `Pune(Pimpri)`, `Pune(Khadiki)` and labeling the result `market_name: 'first'`), and similarly `Lasalgaon` with `Lasalgaon(Niphad)`/`Lasalgaon(Vinchur)`.
   - `scripts/preprocess_and_prepare_data.py` defines `price_lag_1` as `P(t)`, whereas `scripts/build_features.py` defines `price_lag_1` as `P(t-1)` while training/inference consumers (`src/inference.py`, `app.py`) conflate `price_lag_1` with today's current price.
4. **Incomplete Benchmark Reporting:**
   - Unified benchmarks lack persistence baseline metrics, improvement over baseline, MSE, macro-average metrics, and validation leaderboards; pooled $R^2=0.9935$ obscures per-commodity performance (where earlier Tomato/Onion/Sugar $R^2$ varied dramatically).

### Phase 2: Dishonest / Unsupported Commodity Coverage
- `README.md`, `app.py`, and `src/inference.py` advertise **"Predicts Any Food"** (`Potato`, `Rice`, `Wheat`, custom commodities) relying on `OneHotEncoder(handle_unknown='ignore')`.
- The unified model is trained on only **4 commodities** (`Tomato`, `Onion`, `Moong Dal`, `Sugar`) across 5 market series (`Pune`, `Lasalgaon`, `Mumbai`). Unseen-crop generalization was never empirically verified via Leave-One-Commodity-Out testing, and unsupported crops silently return unvalidated numbers while fabricating synthetic lag ratios (`0.99 * price`, `0.98 * price`).

### Phase 3: Missing Market / City Comparison Module
- Despite README claims and 1,108 mandis in `wholesale_prices.csv` + arrivals in `market_arrivals.csv`, [`app.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/app.py) has **no Market Comparison page or workflow**.

### Phase 4: Inaccurate Forecasting Horizon Labels
- `app.py`, `gradio_app.py`, and `README.md` label `target_price_t_plus_1` as `"Tomorrow's Price"` / `"Predicted Tomorrow"`. Agricultural APMCs close on Sundays and holidays, so `t+1` strictly represents the **next observed active trading session**, not a guaranteed next calendar day.

### Phase 5: Disconnected Live Data Acquisition & Synthetic Inference Features
- `DATA_GOV_IN_API_KEY` is configured in `.env`, and `scripts/api/data_gov_client.py` / `scripts/scraping/relevant_scrapers.py` exist, yet `app.py` inference runs purely on manually typed numbers and internally fabricates lag history (`current_price_qtl * 0.98`, `* 0.95`, `* 1.05`). No "Fetch Latest Market Data" integration exists with timeout/error/staleness handling.

### Phase 6: Wholesale ÷ 100 Masquerading as Retail Price Prediction
- `app.py` and `src/inference.py` report `predicted_next_price_rs_per_kg = pred_val / 100.0` as consumer retail price.
- **Data Overlap Audit:** Inspecting `wholesale_prices.csv` (`2016-01-01..2018-12-31` historical + `2026-11-17` API snapshot) against `retail_prices.csv` (`2010-01-01..2012-11-15` DCA historical + `2026-11-17` bulletin snapshot) yields **0 exact (commodity, same-date, same-Maharashtra-city) pairs**, and only **1 common calendar date (`2026-11-17`)** cross-nationally. Thus, sufficient longitudinal paired data for wholesale-to-retail ML estimation does not exist and must be explicitly marked unavailable rather than faked.

### Phase 7: Early-Warning Classification Disconnected from App
- `app.py` computes surge/crash alerts using fixed `> +10%` / `< -10%` arithmetic rules on regression outputs rather than invoking an actual classifier (`Normal`, `Surge`, `Crash`). Classification threshold selection, PR-AUC, majority-class baseline, confusion matrices, and an explicit low-reliability disclosure must be verified and wired into inference.

### Phase 8: Household Affordability Disconnected & Partial-Prediction Risk
- `src/affordability.py` computes deterministic basket costs, but:
  1. It is not connected to wholesale/retail predictions in `app.py`.
  2. It sets `has_predictions = True` if even **one** basket item has a predicted price while summing `0` for unpredicted items—silently undercounting a partial predicted basket as complete!
  3. It lacks negative price/quantity/NaN validation and explicit NSSO HCES cross-sectional caveats.

### Phase 9, 10 & 11: UI, Test Coverage & Documentation Drift
- `app.py` navigation lacks Market Comparison, contains fabricated precision claims ("30 / 30 Marks (100% Implemented)", "29/29 Algorithms"), and mixes hardcoded benchmark numbers instead of reading reproducible artifact outputs.
- Existing `tests/` has only 5 tests (1 failing due to schema drift) and lacks unit/integration tests for feature leakage, validation-only selection, market comparison, classification, partial affordability baskets, and UI pipeline wiring.
- `README.md`, reports, and metadata contain stale metrics and unsupported claims requiring complete reconciliation once Phase 1–10 fixes complete.
