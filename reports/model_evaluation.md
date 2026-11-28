# Machine Learning Model Evaluation & Comparative Analysis (`v3.1.0-parity`)

**Methodology Protocol:**
1. **Training–Inference Parity:** Canonical percentage momentum (`compute_price_momentum_7_pct`), previous-7-session rolling stats strictly excluding `t` (`[t-7..t-1]`), genuine physical arrivals (never imputed from missing as `0.0`), and exact canonical market names (`allow_submarket_aggregation=False`).
2. **Date-Grouped Purged CV (`DatePurgedWalkForwardCV`):** All tuning folds split unique `observation_date` blocks so multi-commodity same-date rows stay atomic and training targets reaching validation windows are purged.
3. **Single Untouched Test Rule:** Candidate ranking & optimization benchmarks reside strictly in Validation leaderboards; ONLY the single Validation-selected Champion and Baseline evaluate on untouched `Test`.
4. **Train-Only Unsupervised Fitting:** Scaling, optimal `k` selection, K-Means, Ward Hierarchical, DBSCAN, and PCA fit strictly inside `Train`.

---

## 1. Unified Multi-Commodity Regression (`Multiple Linear Regression`)

- **Primary Selection Metric:** `Val_Macro_RMSE` (unweighted macro-average across commodities, avoiding Rs.7,000 Moong Dal / Rs.3,200 Sugar scale domination). Secondary: pooled `Val_RMSE`.
- **Dataset:** `data/processed/model_ready_unified_commodities_clean.csv` (**3,998 rows** after **10 boundary rows purged**)
- **Unique-Date Cutoffs:** `Train = 2,787` (`2016-01-09..2018-02-12`) | `Validation = 606` (`2018-02-14..2018-07-20`) | `Test = 605` (`2018-07-23..2018-12-30`)

### A. Validation Leaderboard (`reports/unified_validation_leaderboard.csv`)

| Model | Val_Macro_RMSE | Val_Macro_R2 | Val_RMSE (Pooled) | Val_R2 (Pooled) | Beats Macro Base |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** *(Champion)* | **77.15** | **0.8280** | **84.00** | 0.9987 | Yes |
| Ridge Regression (L2, `alpha=0.1`) | 77.19 | 0.8278 | 84.01 | 0.9987 | Yes |
| Lasso Regression (L1, `alpha=10.0`) | 78.38 | 0.8078 | 84.31 | 0.9987 | Yes |
| ElasticNet Regression | 81.49 | 0.8094 | 87.61 | 0.9985 | No |
| Random Forest Regressor | 87.54 | 0.7942 | 92.69 | 0.9984 | No |
| Gradient Boosting Regressor | 94.43 | 0.7025 | 96.14 | 0.9983 | No |
| Support Vector Regressor | 101.28 | 0.6925 | 119.02 | 0.9973 | No |
| Decision Tree Regressor | 127.75 | 0.4793 | 127.32 | 0.9969 | No |

### B. Untouched Test Evaluation (`N = 605` — Champion vs Persistence Only)

- **Macro-Average Test (Primary):** `Macro_MAE = Rs. 50.52` | **`Macro_RMSE = Rs. 77.34`** (Baseline `Rs. 79.10`, **+1.76 Rs/Qtl**) | **`Macro R² = 0.8586`** | `Macro_MAPE = 5.85%`
- **Pooled Test (Secondary):** `MAE = Rs. 55.31` | **`RMSE = Rs. 93.68`** (Baseline `Rs. 98.16`, **+4.48 Rs/Qtl**) | `Pooled R² = 0.9986` | `MAPE = 6.68%`

| Commodity | Test N | Test MAE | Test RMSE | Persistence RMSE | Beats Persistence | Test $R^2$ | Test MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Moong Dal** | 123 | 21.20 | 45.45 | 42.29 | No (-3.16) | 0.9846 | 0.30% |
| **Onion** | 229 | 75.00 | **122.04** | 128.12 | **Yes (+6.08)** | 0.8491 | 9.96% |
| **Sugar** | 121 | 27.05 | 35.56 | 31.52 | No (-4.04) | 0.8172 | 0.84% |
| **Tomato** | 132 | 78.81 | **106.31** | 114.48 | **Yes (+8.17)** | 0.7836 | 12.29% |

### C. Independent Leave-One-Commodity-Out (LOCO) Evaluation (Zero Held-Out Leakage)

| Held-Out Commodity | Independent Val Champion | Model RMSE | Persistence RMSE | Model $R^2$ | Beats Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Moong Dal | Lasso Regression (`alpha=10.0`) | 125.98 | 42.29 | 0.8815 | No |
| Onion | Multiple Linear Regression | 132.16 | 128.12 | 0.8231 | No |
| Sugar | Multiple Linear Regression | 68.24 | 31.52 | 0.3269 | No |
| Tomato | Lasso Regression (`alpha=10.0`) | 106.00 | 114.48 | 0.7848 | Yes |

**Decision:** Unseen-commodity extrapolation beats persistence on only `1/4` held-out crops; `UnifiedCommodityPredictor` therefore restricts Mode A forecasting strictly to validated commodities and pairs.

---

## 2. Specialist Pipelines (`Tomato @ Pune` & `Onion @ Pune`)

- **Tomato @ Pune (`Train=609, Val=130, Test=131`):**
  - **Validation Leaderboard (`reports/validation_leaderboard_tomato.csv`):** 14 regression candidates ranked on Validation (`Val_RMSE`); **Champion:** `Lasso Regression (L1, alpha=10.0)` (`Val_MAE=75.39, Val_RMSE=117.15, Val_R²=0.8068`).
  - **Untouched Test (`reports/model_comparison_metrics_tomato.csv` — 2 rows only):** `Lasso` **`MAE=75.89, RMSE=103.68, R²=0.7954, MAPE=11.96%`** vs Persistence **`MAE=69.47, RMSE=113.58, R²=0.7545`** (**+9.90 Rs/Qtl improvement**).
  - **Gradient Descent (Validation Benchmarked Only):** Batch GD (`Val_RMSE=120.39, R²=0.7960`), Mini-batch GD (`Val_RMSE=120.63, R²=0.7952`), SGD (`Val_RMSE=121.28, R²=0.7930`).
  - **Binary Classification (Val Selected):** `KNN (k=5, threshold=0.25)` -> Untouched Test `Acc=0.5115, F1=0.5616` vs Majority Baseline `Acc=0.4427, F1=0.6138`.
  - **Train-Only Unsupervised Clustering & Profiles (`reports/clustering_profiles_tomato.csv`):** K-Means `k=2` (`Train_Silhouette=0.4675, DB=0.9180`; Ward `0.4461`; DBSCAN `0.2991`): Cluster 0 (`474` Train rows, Mean `Rs.868.40/Qtl`, CV `49.09%`, Arrival `174.71t`, Spike rate `52.53%`), Cluster 1 (`135` Train rows, Mean `Rs.3,239.26/Qtl`, CV `32.43%`, Arrival `125.27t`, Spike rate `51.11%`). *(Ward and DBSCAN explicitly marked non-inductive / Train-only.)*

- **Onion @ Pune (`Train=601, Val=128, Test=130`):**
  - **Validation Champion:** `Lasso Regression (L1, alpha=5.0)` (`Val_RMSE=89.14, Val_R²=0.8110`).
  - **Untouched Test (`reports/model_comparison_metrics_onion.csv`):** `Lasso` **`MAE=76.57, RMSE=121.22, R²=0.6665, MAPE=8.91%`** vs Persistence **`MAE=74.62, RMSE=138.95, R²=0.5618`** (**+17.73 Rs/Qtl improvement**).

---

## 3. Global 3-Class Early-Warning Spike Classifier & Retail Audit

- **Spike Classifier (`spike_classifier_metadata.json`):** `Logistic Regression (Balanced, C=1.0)` selected on Validation (`Val_Macro_F1=0.5506`); Untouched Test (`605` rows): `Accuracy=0.5884` (Majority Base `0.7802`), `Macro-F1=0.4825`, `Any-Spike ROC-AUC=0.8200`, `Any-Spike PR-AUC=0.4671`, Surge Recall `83.82%`, Surge Precision `23.55%` -> tagged `is_reliable_early_warning=False`.
- **Retail Audit (`retail_model_metadata.json`):** `total_same_city_same_date_pairs = 0`, `retail_estimation_supported = False`.
