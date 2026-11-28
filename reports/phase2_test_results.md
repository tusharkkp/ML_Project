# Phase 2 Verification & Test Execution Results (`v3.1.0-parity`)

**Execution Timestamp:** 2026-11-28 12:09 IST  
**Python Environment:** Python `3.12.10`, `pytest 9.1.1`  

---

## 1. Automated Test Suite Summary (`py -3.12 -m pytest tests/ -v`)

**Outcome:** **`27 passed in 24.76s`** (`Exit Code 0`, zero skipped/relaxed assertions)

| Test Module | Category (Phase 11) | Cases | Result | Verified Scope |
| :--- | :--- | :---: | :---: | :--- |
| [tests/test_features.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_features.py) | A. Feature Contract & E. Purged CV | 6 | **6/6 PASSED** | Canonical `%` momentum helper == batch feature calculation; `rolling_*_7` excludes current `t`; missing arrivals stay `NaN` (`drop_incomplete=True` drops gaps); exact alias normalization (`Pune(Pimpri)`/`Lasalgaon(Niphad)` rejected); zero cross-commodity bleed; `DatePurgedWalkForwardCV` atomic same-date grouping & target purging |
| [tests/test_model_pipeline.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_model_pipeline.py) | B. Inference & Guards | 5 | **5/5 PASSED** | Mode A `predict_validated` parity; Mode B `predict_experimental_manual` disclaimer & `P(t)` exclusion; `UnsupportedCommodityError` (`Potato`, `Wheat`); `UnsupportedMarketPairError` (`Tomato @ Mumbai`); missing arrival rejection; `FeatureSchemaMismatchError`, invalid bounds, non-finite rejection; Tomato & Onion specialists |
| [tests/test_api_live_data.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_api_live_data.py) | C. API & Missing Arrivals | 4 | **4/4 PASSED** | `APMC Pune` vs `Pune(Pimpri)` / `Pune(Hadapsar)` / `Lasalgaon(Niphad)` exact matching; intra-market variety aggregation; contradictory district & composite delimiter rejection; quintal-to-tonnes explicit conversion; HTTP 401/429 & `Timeout` stale cache fallback; missing arrivals kept `None` & Mode A withheld |
| [tests/test_affordability_engine.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_affordability_engine.py) | D. Retail & Affordability | 4 | **4/4 PASSED** | Missing DCA retail stays `None`/`unavailable` (no Rs.45 fabrication); item price provenance (`User Supplied` vs `Archived DCA Observed`); partial basket blocks fake totals; Gradio affordability keeps predictions `unavailable` unless hypothetical scenario explicitly enabled; invalid income/`NaN` rejection |
| [tests/test_market_comparison.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_market_comparison.py) | Market Comparison | 2 | **2/2 PASSED** | Exact-market comparison for `Onion` across `Pune` vs `Lasalgaon`; strict inner-join overlapping date counts & no interpolation |
| [tests/test_model_training.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_model_training.py) | E. Selection, LOCO & Unsupervised | 3 | **3/3 PASSED** | Unified selected strictly by `Val_Macro_RMSE`; independent LOCO `selected_algorithm` recorded per holdout; specialists verified (`test_cmp` has strictly Champion vs Persistence); Train-only clustering & cluster profiles CSV verified; retail `0` MH city-date overlap verified |
| [tests/test_integration.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/tests/test_integration.py) | F. Application E2E Integration | 3 | **3/3 PASSED** | Replay -> Pydantic contract -> Mode A forecast -> affordability schema; Gradio & `deployment/huggingface/app.py` Mode A/B/Market Compare/Affordability callbacks; Streamlit headless `AppTest` rendering Pages 1–4 with zero exceptions |

---

## 2. CLI Verification Smoke Commands

- `py -3.12 scripts/inference_test.py`: **PASSED (Exit 0)** — verified 30.0000% momentum parity, Mode A Specialist & Unified replay, Mode B disclaimer, and `Potato` + `(Tomato @ Mumbai)` Mode A rejections.
- `py -3.12 scripts/test_model.py --commodity Tomato --eval-summary`: **PASSED (Exit 0)** — Champion `Lasso Regression (L1)` (`MAE=75.89, RMSE=103.68` vs Baseline `113.58`, `R²=0.7954`).
- `py -3.12 scripts/test_unified_model.py --eval-summary`: **PASSED (Exit 0)** — Champion `Multiple Linear Regression` (`Val_Macro_RMSE` selected), Untouched Test `Macro_RMSE=77.34` (Base `79.10`, `Macro_R²=0.8586`), `Pooled_RMSE=93.68` (Base `98.16`), and independent LOCO holdouts (`1/4` beating persistence).
