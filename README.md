# AI-Based Food Inflation Prediction and Household Affordability Monitoring System (`v3.0.0-leakfree`)

An end-to-end, scientifically validated machine learning pipeline for forecasting short-term agricultural wholesale price movements across verified APMC mandi series in Maharashtra, classifying early-warning price shock regimes, comparing multi-market historical behavior, and assessing household monthly food-budget vulnerability under NSSO HCES 2022–23 expenditure benchmarks.

---

## Validated Capabilities & Honest Scope Boundaries

- **Validated Commodity Coverage (No Unvalidated Extrapolation):**
  - The Unified Cross-Commodity model is trained and validated strictly on **4 commodities across 5 exact single-market series**:
    - `Tomato` (`Pune`)
    - `Onion` (`Pune`, `Lasalgaon`)
    - `Moong Dal` (`Mumbai`, canonicalized from `Green Gram Dal (Moong Dal)`)
    - `Sugar` (`Mumbai`)
  - **Leave-One-Commodity-Out (LOCO) Audit:** Empirical holdout evaluation proved unseen-commodity generalization fails naive persistence baselines on 3 of 4 held-out crops (`Moong Dal`, `Onion`, `Sugar`), beating persistence only on `Tomato`. Consequently, [src/inference.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/inference.py) enforces `UnsupportedCommodityError` unless `allow_experimental=True` is explicitly opted in.
- **Prediction Horizon Precision:**
  - Models forecast **the next observed active APMC trading session ($t+1$)** in Rs/Quintal — not a guaranteed next calendar day when markets close on Sundays or holidays.
- **Wholesale vs. Retail Boundary (`models/retail_model_metadata.json`):**
  - Cross-dataset temporal auditing confirmed **0 matched Maharashtra same-city same-date pairs** (1 cross-geography overlap date: `2026-11-17`) between `wholesale_prices.csv` (2016–2018 dense archive) and `retail_prices.csv` (2010–2012 DCA archive).
  - Wholesale-to-retail ML estimation is therefore marked **Unavailable (`unavailable_insufficient_paired_data`)**. Dividing wholesale Rs/Qtl by 100 yields **Wholesale Equiv (Rs/Kg)** strictly as a unit conversion — never a retail price estimate. Household affordability uses observed DCA retail prices or explicit user-entered retail rates.

---

## Leak-Free Machine Learning Methodology

All pipelines follow strict chronological hygiene ([src/features.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/features.py), [scripts/train_unified_model.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_unified_model.py), [scripts/train_all_models.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_all_models.py)):

1. **Exact Single-Market Time-Series Isolation:** Lags ($P_{t-1}, P_{t-3}, P_{t-7}$) and strictly backward shifted rolling stats ($[t-7 \dots t-1]$) are engineered independently per `(commodity_name, market_name)` without silent submarket merging (`allow_submarket_aggregation=False`). Current session `modal_price_rs_per_qtl` ($P_t$) is strictly separated from `price_lag_1` ($P_{t-1}$).
2. **Shared Chronological Cutoffs & Boundary Purging:** Unified data (`data/processed/model_ready_unified_commodities_clean.csv`, **3,998 rows** after **10 boundary-crossing rows purged**) splits cleanly across all commodities simultaneously:
   - `Train`: 2,800 rows (`2016-01-09` to `2018-02-15`)
   - `Validation`: 595 rows (`2018-02-17` to `2018-07-22`)
   - `Test`: 603 rows (`2018-07-24` to `2018-12-30`)
3. **Validation-Only Model Selection:** Hyperparameters tuned via 5-fold `TimeSeriesSplit` inside `Train`; Champion selected strictly on `Validation` RMSE / Macro-F1; refitted once on `Train + Validation`; evaluated once on untouched `Test`.

---

## Verified Benchmark Results (Untouched Test Set)

### 1. Unified Multi-Commodity Regression (`Ridge Regression (L2)`, $\alpha = 1.0$)
- **Pooled Test (`N=603`):** `MAE = Rs. 55.27` | `RMSE = Rs. 93.39` (Persistence Baseline: `Rs. 97.99`, improvement `+4.60`) | Pooled $R^2 = 0.9986$ | `MAPE = 6.69%`
- **Scale-Corrected Macro-Average:** `Macro_MAE = Rs. 50.42` | `Macro_RMSE = Rs. 77.03` (Baseline: `Rs. 78.95`, improvement `+1.92`) | **Macro $R^2 = 0.8603$** | `Macro_MAPE = 5.86%`

| Commodity | Test N | Test MAE | Test RMSE | Persistence RMSE | Beats Persistence? | Test $R^2$ | Test MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Moong Dal** | 123 | Rs. 20.65 | Rs. 45.25 | Rs. 42.29 | No (-2.96) | 0.9847 | 0.29% |
| **Onion** | 228 | Rs. 75.72 | **Rs. 122.33** | Rs. 128.41 | **Yes (+6.08)** | 0.8490 | 10.04% |
| **Sugar** | 121 | Rs. 27.18 | Rs. 35.59 | Rs. 31.52 | No (-4.07) | 0.8170 | 0.85% |
| **Tomato** | 131 | Rs. 78.11 | **Rs. 104.93** | Rs. 113.58 | **Yes (+8.65)** | 0.7905 | 12.25% |

### 2. Dedicated Single-Commodity Specialists (`Lasso Regression (L1)`)
- **Tomato @ Pune (`N=131`):** `MAE = Rs. 75.95` | **`RMSE = Rs. 103.61`** (vs Persistence `Rs. 113.58` -> **+9.97 Rs/Qtl improvement**) | **$R^2 = 0.7957$** | `MAPE = 11.99%`
- **Onion @ Pune (`N=130`):** `MAE = Rs. 76.88` | **`RMSE = Rs. 121.29`** (vs Persistence `Rs. 138.95` -> **+17.66 Rs/Qtl improvement**) | **$R^2 = 0.6661$** | `MAPE = 8.96%`

### 3. Global 3-Class Early-Warning Spike Classifier (`Normal` / `Surge > +10%` / `Crash < -10%`)
- **Champion (Validation Macro-F1 Selected):** `Logistic Regression (Balanced, C=10.0)`
- **Untouched Test (`N=603`, Majority Baseline `Normal` = `0.7811`):** `Accuracy = 0.6003` | `Macro-F1 = 0.4946` | `Any-Spike ROC-AUC = 0.8117` | `Any-Spike PR-AUC = 0.4293` (Surge Recall `82.35%`, Surge Precision `23.53%`).
- **Reliability Status:** Tagged `is_reliable_early_warning = False` and presented strictly as an experimental risk indicator alongside regression forecasts.

*(Full candidate leaderboards across all 22 syllabus algorithms — Gradient Descent variants, Binary Classification, Clustering, PCA, and LOCO tables — are documented in [reports/model_evaluation.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/model_evaluation.md).)*

---

## Repository Structure

```text
ML_Project/
├── app.py                              # Streamlit Dashboard (5 modules: Forecast, Market Compare, Affordability, Benchmarks, Viva)
├── gradio_app.py                       # Gradio Web UI (Synchronized with v3.0.0-leakfree)
├── deployment/huggingface/app.py       # Hugging Face Spaces deployment entrypoint
├── src/
│   ├── features.py                     # Canonical leak-free lag/rolling features & purged chronological splitter
│   ├── inference.py                    # Unified/Specialist predictors, UnsupportedCommodityError, EarlyWarningClassifier
│   ├── live_data.py                    # data.gov.in + DCA API client, TTL cache, multi-year gap protection
│   ├── market_comparison.py            # Cross-market historical analyzer (strict exact markets & inner-join overlaps)
│   ├── affordability.py                # NSSO HCES budget share engine (strict complete-basket validation)
│   ├── metrics.py                      # Regression, classification, baseline, and macro-average metrics
│   └── schemas.py                      # Pydantic v2 input/output contracts
├── scripts/
│   ├── build_features.py               # Deterministic rebuild of all model-ready CSVs
│   ├── train_unified_model.py          # Unified Ridge trainer + LOCO generalization benchmark
│   ├── train_and_evaluate.py           # Specialist Pune Tomato CV & validation benchmark
│   ├── train_all_models.py             # All 22 syllabus algorithms + global 3-class spike classifier
│   ├── evaluate_retail_feasibility.py  # Longitudinal wholesale-vs-retail overlap auditor
│   ├── inference_test.py               # End-to-end verification script
│   ├── test_model.py                   # Specialist CLI tester
│   └── test_unified_model.py           # Unified CLI tester
├── data/processed/                     # Cleaned wholesale, retail, arrivals & leak-free model-ready CSVs
├── models/                             # Saved `.joblib` pipelines & JSON metadata audits
├── reports/                            # Evaluation CSVs, figures, gap audit & status reports
└── tests/                              # 6 automated pytest modules (19 unit & integration tests)
```

---

## Setup & Execution

### 1. Environment & Dependencies
```bash
pip install -r requirements.txt
# Optional: configure live data.gov.in API access in .env:
# DATA_GOV_IN_API_KEY=your_api_key_here
```

### 2. Run Interactive Applications
```bash
# Streamlit Dashboard (Recommended locally)
py -3.12 -m streamlit run app.py

# Gradio Interface
py -3.12 gradio_app.py
```

### 3. Reproduce Training & Verification
```bash
# Rebuild leak-free feature datasets
py -3.12 scripts/build_features.py --all

# Retrain unified model, Tomato/Onion specialists, global spike classifier & retail audit
py -3.12 scripts/train_unified_model.py
py -3.12 scripts/train_all_models.py --commodity Onion --market Pune
py -3.12 scripts/train_all_models.py --commodity Tomato --market Pune
py -3.12 scripts/train_all_models.py --train-global-classifier
py -3.12 scripts/evaluate_retail_feasibility.py

# Run complete unit & integration test suite (19 tests)
py -3.12 -m pytest tests/ -v
```
