# AI-Based Food Inflation Prediction and Household Affordability Monitoring System (`v3.1.0-parity`)

An end-to-end, scientifically verified machine learning pipeline for forecasting short-term agricultural wholesale price movements across verified APMC mandi series in Maharashtra, classifying early-warning price shock regimes, comparing multi-market historical series, and assessing household food-budget vulnerability under NSSO HCES 2022–23 expenditure benchmarks.

---

## Validated Capabilities & Scope Boundaries

- **100% Training–Inference Feature Parity ([reports/feature_contract.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/feature_contract.md)):**
  - `price_momentum_7` uses one shared helper (`src/features.compute_price_momentum_7_pct()`) across offline dataset creation, historical replay, live ingestion, and exploration (`((P_t - P_{t-7}) / P_{t-7}) * 100.0` in **`%`**).
  - `rolling_mean_7`, `rolling_std_7`, `rolling_min_7`, and `rolling_max_7` compute strictly over the previous 7 active trading sessions (`[t-7 .. t-1]`, excluding current session `t`).
  - Current session physical arrivals (`arrival_tonnes`, `arrival_lag_1`, `rolling_arrival_mean_7`) preserve missing telemetry as `NaN` — unavailable API arrivals are NEVER silently set to `0.0`.
- **Two Explicit Inference Modes ([src/inference.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/inference.py)):**
  - **Mode A (`predict_validated`):** Requires genuine continuous APMC series, all 7-session rolling stats, physical arrivals, validated commodity (`['Moong Dal', 'Onion', 'Sugar', 'Tomato']`), and validated commodity-market pair (`UnsupportedCommodityError` / `UnsupportedMarketPairError` otherwise).
  - **Mode B (`predict_experimental_manual`):** Explicit exploratory UI mode; excludes current price $P_t$ from prior-lag rolling summaries and attaches `EXPERIMENTAL EXPLORATION ONLY` disclaimers.
- **Exact Live Market Alias Resolution ([src/live_data.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/live_data.py)):**
  - Matches normalized exact aliases (`APMC Pune` $\to$ `Pune`; `Lasalgaon`) without substring matching, preventing submarkets like `Pune(Pimpri)`, `Pune(Hadapsar)`, or `Lasalgaon(Niphad)` from mixing into single-market models. Rejects ambiguous district mappings.
- **Retail Integrity & Household Affordability ([src/affordability.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/affordability.py)):**
  - Wholesale-to-retail ML estimation stays **`Unavailable`** (`models/retail_model_metadata.json`: 0 matched MH city-date pairs). `price / 100` is strictly labeled **Wholesale Equiv (Rs/Kg)**.
  - Hardcoded Rs.45/Kg retail fallbacks removed across Streamlit, Gradio, and Hugging Face UIs; missing retail data displays `"Unavailable"` and requires explicit `"User Supplied"` input. Future scenarios require complete baskets and never silently copy current prices as future predictions.

---

## Verified Benchmark Results (`v3.1.0-parity`, `DatePurgedWalkForwardCV`)

### 1. Unified Multi-Commodity Regression (`Multiple Linear Regression`, Primary Metric: `Val_Macro_RMSE`)
- **Dataset (`model_ready_unified_commodities_clean.csv`):** `3,998` boundary-purged rows (`10` rows purged; `Train = 2,787`, `Validation = 606`, `Test = 605`).
- **Validation Selection:** `Multiple Linear Regression` (`Val_Macro_RMSE = Rs. 77.15/Qtl`, `Val_RMSE = Rs. 84.00/Qtl`) outperformed Ridge ($\alpha=0.1$: `Val_Macro_RMSE = 77.19`), Lasso (`78.38`), ElasticNet (`81.49`), and tree/kernel models.
- **Untouched Test (`605` rows, evaluated ONCE for Champion vs Persistence):**
  - **Macro-Average Test (Primary):** `Macro_MAE = Rs. 50.52` | **`Macro_RMSE = Rs. 77.34`** (Baseline `79.10`, `+1.76` improvement) | **`Macro R² = 0.8586`** | `Macro_MAPE = 5.85%`
  - **Pooled Test (Secondary):** `MAE = Rs. 55.31` | **`RMSE = Rs. 93.68`** (Baseline `98.16`, `+4.48` improvement) | `Pooled R² = 0.9986` | `MAPE = 6.68%`

| Commodity | Test N | Test MAE | Test RMSE | Persistence RMSE | Beats Persistence? | Test $R^2$ | Test MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Moong Dal** | 123 | Rs. 21.20 | Rs. 45.45 | Rs. 42.29 | No (-3.16) | 0.9846 | 0.30% |
| **Onion** | 229 | Rs. 75.00 | **Rs. 122.04** | Rs. 128.12 | **Yes (+6.08)** | 0.8491 | 9.96% |
| **Sugar** | 121 | Rs. 27.05 | Rs. 35.56 | Rs. 31.52 | No (-4.04) | 0.8172 | 0.84% |
| **Tomato** | 132 | Rs. 78.81 | **Rs. 106.31** | Rs. 114.48 | **Yes (+8.17)** | 0.7836 | 12.29% |

- **Independent LOCO Generalization Check:** Each held-out commodity was excluded completely from tuning and validation selection; unseen-crop extrapolation failed persistence baselines on `3/4` crops (`Moong Dal`, `Onion`, `Sugar`), beating persistence only on `Tomato` (`106.00` vs `114.48`).

### 2. Dedicated Single-Commodity Specialists (`Lasso Regression (L1)`)
- **Tomato @ Pune (`Train=609, Val=130, Test=131`):** `Lasso (alpha=10.0)` (`Val_RMSE = 117.15`) -> Untouched Test `MAE = Rs. 75.89` | **`RMSE = Rs. 103.68`** (vs Persistence `113.58`, **`+9.90 Rs/Qtl`**) | **`R² = 0.7954`** | `MAPE = 11.96%`
- **Onion @ Pune (`Train=601, Val=128, Test=130`):** `Lasso (alpha=5.0)` (`Val_RMSE = 89.14`) -> Untouched Test `MAE = Rs. 76.57` | **`RMSE = Rs. 121.22`** (vs Persistence `138.95`, **`+17.73 Rs/Qtl`**) | **`R² = 0.6665`** | `MAPE = 8.91%`

### 3. Global 3-Class Early-Warning Spike Classifier (`models/spike_classifier.joblib`)
- **Validation Champion (`Val_Macro_F1`):** `Logistic Regression (Balanced, C=1.0)` (`Val_Macro_F1 = 0.5506`)
- **Untouched Test (`605` rows, Majority `Normal = 0.7802`):** `Accuracy = 0.5884` | `Macro-F1 = 0.4825` | `Any-Spike ROC-AUC = 0.8200` | `Any-Spike PR-AUC = 0.4671` (Surge Recall `83.82%`, Surge Precision `23.55%`). Flagged `is_reliable_early_warning = False` honestly due to low rare-event precision.

*(Complete validation leaderboards, train-only clustering profiles, Phase 2 deep audit, and parity tests are documented under [reports/](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/).)*

---

## Setup, Reproducibility & Verification

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Rebuild canonical v3.1.0-parity feature datasets
py -3.12 scripts/build_features.py --all

# 2. Retrain all pipelines (Unified + Independent LOCO, Tomato/Onion Specialists, Global Classifier, Retail Audit)
py -3.12 scripts/train_unified_model.py
py -3.12 scripts/train_and_evaluate.py
py -3.12 scripts/train_all_models.py --commodity Onion --market Pune
py -3.12 scripts/train_all_models.py --commodity Tomato --market Pune
py -3.12 scripts/train_all_models.py --train-global-classifier
py -3.12 scripts/evaluate_retail_feasibility.py

# 3. Verify inference parity & full test suite
py -3.12 scripts/inference_test.py
py -3.12 -m pytest tests/ -v

# 4. Run interactive dashboards
py -3.12 -m streamlit run app.py
py -3.12 gradio_app.py
```
