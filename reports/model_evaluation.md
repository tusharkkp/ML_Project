# Machine Learning Model Evaluation & Comparative Analysis (`v3.0.0-leakfree`)

**Methodology Guarantee:** All models below were trained under a strict leak-free protocol:
1. **Feature Isolation:** Lags ($P_{t-1}, P_{t-3}, P_{t-7}$) and 7-session shifted rolling windows ($[t-7 \dots t-1]$) are computed independently within exact `(commodity_name, market_name)` series. Current session `modal_price_rs_per_qtl` ($P_t$) is strictly distinct from `price_lag_1` ($P_{t-1}$).
2. **Chronological Splitting & Purging:** Shared chronological cutoffs across all series with boundary purging wherever `target_date_t_plus_1` crosses into Validation/Test.
3. **Validation-Only Model Selection:** Hyperparameters tuned via 5-fold `TimeSeriesSplit` inside `Train`; Champion chosen strictly on `Validation` metrics (`RMSE` for regression, `Macro-F1` for classification); refitted once on `Train + Validation` and evaluated once on untouched `Test`.
4. **Target Definition:** Next observed active APMC trading session modal wholesale price (`Rs/Quintal`, $t+1$ — not guaranteed next calendar day).

---

## 1. Unified Multi-Commodity Regression Pipeline (`Ridge Regression (L2)`)

- **Dataset:** `data/processed/model_ready_unified_commodities_clean.csv` (**3,998 rows** after **10 cross-boundary rows purged**)
- **Shared Cutoffs:** Train `2,800` (`2016-01-09..2018-02-15`) | Validation `595` (`2018-02-17..2018-07-22`) | Test `603` (`2018-07-24..2018-12-30`)
- **Validation Champion:** **Ridge Regression (L2)** (`alpha=1.0`, `Val_RMSE = Rs. 83.22/Qtl`, `Val_Macro_RMSE = Rs. 76.64/Qtl`, `Val_R² = 0.9987`)

### A. Overall Pooled vs. Scale-Corrected Macro-Average Test Performance (`N = 603`)

| Aggregation View | Model MAE | Model RMSE | Persistence Baseline RMSE | Model $R^2$ | Model MAPE | Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pooled Global Test** | Rs. 55.27 | **Rs. 93.39** | Rs. 97.99 | 0.9986 | 6.69% | Pooled $R^2$ is inflated by cross-crop price scale differences (Tomato ~700 vs Moong Dal ~7,034) |
| **Macro-Average Test** | Rs. 50.42 | **Rs. 77.03** | Rs. 78.95 | **0.8603** | 5.86% | Honest unweighted average across all 4 validated commodities |

### B. Per-Commodity Test Breakdown vs. Naive Persistence ($P_{t+1} = P_t$)

| Commodity | Category | Test N | Mean Price | Test MAE | Test RMSE | Persistence RMSE | Beats Persistence? | Test $R^2$ | Test MAPE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Moong Dal** | Pulses | 123 | Rs. 7,034.15 | Rs. 20.65 | Rs. 45.25 | Rs. 42.29 | No (-2.96) | 0.9847 | 0.29% |
| **Onion** | Vegetables | 228 | Rs. 867.50 | Rs. 75.72 | **Rs. 122.33** | Rs. 128.41 | **Yes (+6.08)** | 0.8490 | 10.04% |
| **Sugar** | Sugars & Sweets | 121 | Rs. 3,205.61 | Rs. 27.18 | Rs. 35.59 | Rs. 31.52 | No (-4.07) | 0.8170 | 0.85% |
| **Tomato** | Vegetables | 131 | Rs. 706.87 | Rs. 78.11 | **Rs. 104.93** | Rs. 113.58 | **Yes (+8.65)** | 0.7905 | 12.25% |

*Key Scientific Insight:* Pooled Ridge improves volatility forecasting on perishable crops (**Onion +6.08**, **Tomato +8.65** Rs/Qtl RMSE over persistence), whereas sticky/low-variance staples (**Moong Dal**, **Sugar**) are slightly harder for a shared regularized hyperplane to beat persistence on MAE/RMSE — confirming why reporting pooled $R^2$ alone would be misleading.

### C. Leave-One-Commodity-Out (LOCO) Unseen Commodity Test

| Held-Out Commodity | Model RMSE (Rs/Qtl) | Persistence RMSE (Rs/Qtl) | Model $R^2$ | Beats Persistence? |
| :--- | :---: | :---: | :---: | :---: |
| Moong Dal | 109.36 | 42.29 | 0.9107 | No |
| Onion | 135.63 | 128.41 | 0.8143 | No |
| Sugar | 68.88 | 31.52 | 0.3143 | No |
| Tomato | 104.08 | 113.58 | 0.7938 | Yes |

**Decision:** Because held-out generalization fails persistence on **3 out of 4** commodities, `UnifiedCommodityPredictor` explicitly restricts inference to `['Moong Dal', 'Onion', 'Sugar', 'Tomato']` via `UnsupportedCommodityError`.

---

## 2. Dedicated Specialist Models (`Tomato @ Pune` & `Onion @ Pune`)

### A. Tomato @ Pune (`N_train=609, N_val=130, N_test=131`)
*(From [reports/model_comparison_metrics_tomato.csv](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/model_comparison_metrics_tomato.csv))*

- **Validation Champion:** **Lasso Regression ($L_1$, `alpha=10.0`)** (`Val_MAE=76.90`, `Val_RMSE=118.40`, `Val_R²=0.8027`)

| Model | Test MAE | Test MSE | Test RMSE | Test $R^2$ | Test MAPE (%) | Val Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Persistence Baseline ($P_t$) | 69.47 | 12,900.76 | 113.58 | 0.7545 | 10.56% | Baseline |
| **Lasso Regression (L1)** | **75.95** | **10,736.07** | **103.61** | **0.7957** | **11.99%** | **Yes (Champion)** |
| Voting Regressor | 76.14 | 10,917.51 | 104.49 | 0.7922 | 11.71% | No |
| Multiple Linear Regression | 77.00 | 10,999.09 | 104.88 | 0.7907 | 11.77% | No |
| Ridge Regression (L2) | 83.49 | 11,938.77 | 109.26 | 0.7728 | 13.09% | No |
| Stacking Regressor (Chronological OOF) | 86.21 | 12,715.85 | 112.76 | 0.7580 | 12.22% | No |
| Simple Linear Regression | 79.31 | 12,923.16 | 113.68 | 0.7541 | 12.50% | No |
| Bagging Regressor | 83.47 | 13,502.75 | 116.20 | 0.7430 | 12.57% | No |
| Random Forest Regressor | 84.13 | 13,519.56 | 116.27 | 0.7427 | 12.59% | No |
| Gradient Boosting Regressor | 87.18 | 15,352.79 | 123.91 | 0.7078 | 13.71% | No |
| AdaBoost Regressor | 93.30 | 17,068.08 | 130.64 | 0.6752 | 13.79% | No |
| Polynomial Regression (deg=2) | 113.17 | 19,386.30 | 139.23 | 0.6311 | 18.23% | No |
| SVR (RBF Kernel) | 107.78 | 19,487.75 | 139.60 | 0.6291 | 16.65% | No |
| Decision Tree Regressor | 101.27 | 19,505.33 | 139.66 | 0.6288 | 15.51% | No |
| KNN Regressor | 174.14 | 48,931.66 | 221.21 | 0.0688 | 26.33% | No |

- **Optimization Benchmarks (Tomato):** Mini-batch GD (`RMSE=104.82, R²=0.7909`), Batch GD (`RMSE=105.03, R²=0.7901`), SGD (`RMSE=105.07, R²=0.7899`).
- **Binary Spike Classifier (`is_price_spike`, Tomato):** Val-selected **Bagging Classifier** (`Val_Threshold=0.35`): Test `Acc=0.4580, Prec=0.4488, Rec=0.9828, F1=0.6162, ROC-AUC=0.5704, PR-AUC=0.5162` (vs Majority Baseline `Acc=0.4427, F1=0.6138`).
- **Clustering & PCA (Tomato):** K-Means (`k=2`, Silhouette `0.5119`, DB `0.9196`), Hierarchical Ward (`k=2`, Silhouette `0.5100`, DB `0.9266`), DBSCAN (`4 clusters`, Silhouette `0.4808`, DB `1.0169`).

### B. Onion @ Pune (`N_train=601, N_val=128, N_test=130`)
*(From [reports/model_comparison_metrics_onion.csv](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/model_comparison_metrics_onion.csv))*

- **Validation Champion:** **Lasso Regression ($L_1$, `alpha=5.0`)** (`Val_RMSE=87.98, Val_MAE=64.90, Val_R²=0.8159`)
- **Untouched Test:** **MAE=76.88, RMSE=121.29, R²=0.6661, MAPE=8.96%** vs Naive Persistence Baseline (`MAE=74.62, RMSE=138.95, R²=0.5618, MAPE=8.32%` -> **+17.66 Rs/Qtl RMSE improvement**).

---

## 3. Global Early-Warning 3-Class Classifier (`Normal / Surge / Crash`)

- **Artifact:** `models/spike_classifier.joblib` & `models/spike_classifier_metadata.json`
- **Definition:** `Surge` (> +10% next-session jump) | `Crash` (< -10% drop) | `Normal` ([-10%, +10%])
- **Validation Champion (Macro-F1 Selected):** **Logistic Regression (Balanced, `C=10.0`)** (`Val_Macro_F1=0.5377`)
- **Untouched Test Evaluation (`N=603`, Majority `Normal=78.11%`):**
  - **Accuracy:** `0.6003` (vs Majority Baseline `0.7811`)
  - **Macro F1:** `0.4946` | **Any-Spike ROC-AUC:** `0.8117` | **Any-Spike PR-AUC:** `0.4293`
  - **Per-Class:** `Normal` (Prec=0.989, Rec=0.590), `Surge` (Prec=0.235, Rec=0.824), `Crash` (Prec=0.333, Rec=0.438)
  - **Honest Reliability Flag (`is_reliable_early_warning = False`):** Class-balanced linear boundaries achieve strong shock recall (82.4% on Surges), but low precision (23.5–33.3%) due to unrecorded exogenous shocks (rainfall/export bans). Displayed strictly as an experimental risk indicator alongside continuous regression alerts.

---

## 4. Wholesale-to-Retail Audit Finding (`models/retail_model_metadata.json`)

- **Exact Maharashtra Same-City Same-Date Pairs:** **0** (`Mumbai=0, Pune=0, Nagpur=0, Nashik=0`)
- **Common Dates Across All Records:** **1** (`2026-11-17`) because DCA historical archives end `2012-11-15` while AGMARKNET archives span `2016–2018`.
- **Conclusion:** Retail regression is marked **`unavailable_insufficient_paired_data`**. All Rs/Kg outputs derived via `price_qtl / 100` are strictly labeled **Wholesale Equiv (Rs/Kg)**, while the Household Affordability Monitor operates directly on actual observed DCA retail references or user-entered retail prices.
