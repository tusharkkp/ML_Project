# Phase 2 Implementation Report: Deep ML Parity, Reliability & Methodology Upgrades (`v3.1.0-parity`)

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Repository:** `https://github.com/tusharkkp/ML_Project` (`branch: main`, base commit `cf1df49`)  
**Completion Date:** 2026-11-28  
**Related Deliverables:**
- Initial Audit: [reports/phase2_deep_audit.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/phase2_deep_audit.md)
- Canonical Feature Contract: [reports/feature_contract.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/feature_contract.md)
- Test Execution Record: [reports/phase2_test_results.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/phase2_test_results.md)

---

## 1. Defects Resolved & Architectural Changes

| Defect ID | Criticality | Root Cause Fixed | Modified / Created Files |
| :--- | :---: | :--- | :--- |
| **D1 (Phase 1)** | CRITICAL | Unified `price_momentum_7` (`%`) across training and inference via shared `compute_price_momentum_7_pct()`, corrected `rolling_*_7` schema docstring (excludes current session $t$). | `src/features.py`, `src/inference.py`, `src/schemas.py`, `reports/feature_contract.md` |
| **D2 (Phase 2)** | CRITICAL | Separated **Mode A: Validated Forecasting** from **Mode B: Experimental Manual Exploration**; manual exploration now excludes $P(t)$ from rolling window summaries and displays prominent experimental disclaimers. | `src/inference.py`, `app.py`, `gradio_app.py`, `deployment/huggingface/app.py` |
| **D3 (Phase 3)** | CRITICAL | Replaced substring matching `market.lower() in r_mkt` with normalized exact alias matching (`normalize_market_token`, `match_canonical_market`), composite delimiter rejection, contradictory-district ambiguity detection, and intra-market variety logging. | `src/features.py`, `src/live_data.py`, `scripts/build_features.py` |
| **D4 (Phase 4)** | CRITICAL | Verified `data.gov.in` resource `9ef84268-...` omits daily arrival quantities; stopped silent `0.0` imputation, preserved missing arrivals as `NaN/None`, handled unit conversions strictly for explicit known units, and withheld Mode A automation whenever arrivals or continuous 7-day history are unavailable. | `src/live_data.py`, `src/features.py`, `scripts/build_features.py`, `app.py`, `gradio_app.py` |
| **D5 (Phase 5)** | HIGH | Removed all hardcoded `Rs.45.0/Kg` retail defaults cleanly; displays `"Unavailable"` when absent and preserves exact centre/date/source provenance (`Actual Observed DCA` vs `User Supplied`). | `src/live_data.py`, `app.py`, `gradio_app.py`, `deployment/huggingface/app.py` |
| **D6 (Phase 6)** | HIGH | Stopped Gradio/HF UI from copying current retail price into `predicted_retail_price_per_kg`; synchronized Streamlit/Gradio/HF affordability workflows with complete-basket validation and explicit hypothetical scenario framing. | `src/affordability.py`, `app.py`, `gradio_app.py`, `deployment/huggingface/app.py` |
| **D7 (Phase 7)** | HIGH | Implemented reusable `DatePurgedWalkForwardCV` keeping multi-commodity same-date rows atomic per fold and purging any training observation whose `target_date` enters or crosses a validation fold. | `src/features.py`, `scripts/train_unified_model.py`, `scripts/train_and_evaluate.py`, `scripts/train_all_models.py` |
| **D8 (Phase 8)** | HIGH | Rewrote `run_leave_one_commodity_out()` to run full independent hyperparameter CV + validation selection (`Val_Macro_RMSE`) strictly excluding the held-out commodity. | `scripts/train_unified_model.py` |
| **D9 (Phase 9)** | MEDIUM | Restricted scaling, `k` selection, K-Means, Ward Hierarchical, DBSCAN, and PCA strictly to the `Train` partition; added out-of-sample limitation notes and Train cluster profiles (`Mean_Modal_Price`, `CV_Volatility_Pct`, `Price_Spike_Rate_Pct`). | `scripts/train_all_models.py` |
| **D10 (Phase 10)** | HIGH | Standardized unified model selection strictly on `Val_Macro_RMSE` (recording `Val_RMSE` secondary), restricted Test evaluation strictly to the single Validation-selected Champion + Baseline, evaluated GD optimizers strictly on Validation, and reconciled metadata & reports. | `scripts/train_unified_model.py`, `scripts/train_all_models.py`, `README.md`, `reports/model_evaluation.md` |

---

## 2. Models Retrained & Verification of Before vs. After Metrics

All datasets (`data/processed/model_ready_*.csv`) and all serialized models (`models/*.joblib`, `models/*.json`) were rebuilt and retrained cleanly under `feature_schema_version = "3.1.0-parity"`.

| Pipeline / Model | Baseline (`v3.0.0-leakfree`) | Rebuilt (`v3.1.0-parity`, Purged Date CV & `%` Momentum) | Key Verification Note |
| :--- | :--- | :--- | :--- |
| **Unified Multi-Commodity** | Champion `Ridge (alpha=1.0)` (Pooled `Val_RMSE=83.22`) -> Test `Macro_RMSE=77.03`, `Macro_R²=0.8603`, `Pooled_RMSE=93.39` | Champion **`Multiple Linear Regression`** (`Val_Macro_RMSE=77.15`, `Val_RMSE=84.00`) -> Test **`Macro_RMSE=77.34`** (Base `79.10`), **`Macro_R²=0.8586`**, `Pooled_RMSE=93.68` | Selected strictly by documented `Val_Macro_RMSE`; Test evaluated ONCE for Champion vs Baseline. |
| **Tomato @ Pune Specialist** | `Lasso (alpha=10.0)` -> Test `RMSE=103.61`, `R²=0.7957` (vs Base `113.58`) | **`Lasso (alpha=10.0)`** (`Val_RMSE=117.15`) -> Test **`RMSE=103.68`**, **`R²=0.7954`** (vs Base `113.58`) | Beats Persistence by `+9.90 Rs/Qtl` under identical serving/training % momentum. |
| **Onion @ Pune Specialist** | `Lasso (alpha=5.0)` -> Test `RMSE=121.29`, `R²=0.6661` (vs Base `138.95`) | **`Lasso (alpha=5.0)`** (`Val_RMSE=89.14`) -> Test **`RMSE=121.22`**, **`R²=0.6665`** (vs Base `138.95`) | Beats Persistence by `+17.73 Rs/Qtl`. |
| **Early-Warning 3-Class Classifier** | `Logistic Regression (Balanced, C=10.0)` -> Test `Acc=0.6003`, `PR-AUC=0.4293` | **`Logistic Regression (Balanced, C=1.0)`** (`Val_Macro_F1=0.5506`) -> Test `Acc=0.5884`, `Any-Spike PR-AUC=0.4671`, `ROC-AUC=0.8200` | Honestly flagged `is_reliable_early_warning=False` (Surge Recall `83.8%`, Precision `23.6%`). |
| **Independent LOCO Holdouts** | Used leaked 4-crop hyperparameter champion; `1/4` (`Tomato`) beat baseline | Fully independent per-fold CV & selection excluding held-out crop; still **`1/4`** (`Tomato` `106.00` vs `114.48`) beats baseline | Proves mathematically why unseen commodities remain blocked in Mode A (`UnsupportedCommodityError`). |

---

## 3. Remaining Known Limitations

1. **Historical vs. Live Gap & API Arrival Omission:** Dense daily archival data ends `2018-12-31`, while `data.gov.in` resource `9ef84268-d588-465a-a308-a864a43d0070` returns only prices (`min`, `max`, `modal`) without arrival tonnage. Automated Mode A live forecasts are therefore withheld until a continuous multi-day price + arrival source is integrated (users can replay historical validated records or test Mode B experimental scenarios).
2. **Retail Longitudinal Gap:** Zero matched Maharashtra same-city same-date pairs remain between wholesale (`2016–2018`) and archived DCA retail (`2010–2012`); retail price estimation stays marked `Unavailable`.
3. **Shock Classification Precision:** `Any-Spike PR-AUC = 0.4671` reflects fundamental unpredictability of >10% single-session agricultural price shocks using price/arrival lags alone without exogenous weather/policy telemetry.

---

## 4. Exact Reproduction Commands

```bash
py -3.12 scripts/build_features.py --all
py -3.12 scripts/train_unified_model.py
py -3.12 scripts/train_and_evaluate.py
py -3.12 scripts/train_all_models.py --commodity Onion --market Pune
py -3.12 scripts/train_all_models.py --commodity Tomato --market Pune
py -3.12 scripts/train_all_models.py --train-global-classifier
py -3.12 scripts/evaluate_retail_feasibility.py
py -3.12 scripts/inference_test.py
py -3.12 scripts/test_model.py --commodity Tomato --eval-summary
py -3.12 scripts/test_unified_model.py --eval-summary
py -3.12 -m pytest tests/ -v
```
