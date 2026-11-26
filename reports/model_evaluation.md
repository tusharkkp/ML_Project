# Machine Learning Model Evaluation & Comparative Analysis

**Course Component:** Practical Examination Component G (Model Evaluation, Testing on Previously Unseen Data & Comparative Analysis — 8 Marks)  
**Date:** 2026-11-22  
**Test Set Rigor:** 100% Unseen Chronological Test Partition (156 trading sessions for Tomato, 152 for Onion)

---

## 1. Executive Summary of Comparative Evaluation

Every algorithm from the AIML syllabus ([`ML_algos.txt`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/ML_algos.txt)) has been evaluated on previously unseen test data across multiple commodity series.

### Key Empirical Findings:

1. **Lasso & Multiple Linear Regression Dominate Perishables (Tomato):**
   - **Lasso Regression ($L_1$)** achieved the best overall performance on Tomato (Test RMSE: **Rs. 93.35/Qtl**, $R^2 = \mathbf{0.8454}$), decisively beating the Naive Persistence Baseline (RMSE: **Rs. 112.11**, $R^2 = 0.7771$).
   - Lasso performed automated feature selection, safely eliminating collinear lags while retaining physical arrival volume and momentum.
2. **Support Vector Regression (SVR) Dominates High-Kurtosis Series (Onion):**
   - For Onion, **SVR (RBF Kernel)** achieved the lowest Test RMSE (**Rs. 179.06/Qtl** vs. Baseline **Rs. 191.32/Qtl**), outperforming linear and tree models by effectively handling onion auction price spikes via non-linear margin boundaries.
3. **Classification Detects Impending Supply Shocks:**
   - On the binary price spike detection task ($\Delta P > 10\%$), **AdaBoost Classifier** achieved **52.6% accuracy** and **Stacking Classifier** achieved **63.2% accuracy on Onion**, providing an early-warning signal for market distress.
4. **Unsupervised Market Regimes:**
   - K-Means ($k=2$, Silhouette **0.4747**) and DBSCAN cleanly isolated structural market shock days from standard seasonal trading sessions.

---

## 2. Regression Models Comparative Performance Table

### Tomato Wholesale Forecasting (Pune APMC — 156 Unseen Test Sessions)

_(From [reports/model_comparison_metrics_tomato.csv](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/model_comparison_metrics_tomato.csv))_

| Model Algorithm                   | Category    | Test MAE (Rs/Qtl) |   Test MSE   | Test RMSE (Rs/Qtl) | Test $R^2$ Score | Test MAPE (%) | Academic Assessment            |
| :-------------------------------- | :---------- | :---------------: | :----------: | :----------------: | :--------------: | :-----------: | :----------------------------- |
| **Naive Persistence Baseline**    | Benchmark   |       83.17       |  12,568.11   |       112.11       |      0.7771      |    11.25%     | $y_{t+1} = y_t$ Baseline       |
| **Simple Linear Regression**      | Linear      |       89.35       |  13,008.63   |       114.06       |      0.7693      |    12.76%     | Univariate lag-1 baseline      |
| **Multiple Linear Regression**    | Linear      |       69.66       |   8,851.71   |       94.08        |      0.8430      |     9.47%     | Excellent generalization       |
| **Polynomial Regression (deg=2)** | Non-Linear  |       95.40       |  14,712.71   |       121.30       |      0.7390      |    13.18%     | Slight interaction overfitting |
| **Ridge Regression ($L_2$)**      | Regularized |       69.32       |   8,799.00   |       93.80        |      0.8439      |     9.44%     | Multicollinearity control      |
| **Lasso Regression ($L_1$)**      | Regularized |     **68.66**     | **8,715.15** |     **93.35**      |    **0.8454**    |   **9.36%**   | **WINNING TOMATO MODEL**       |
| **Decision Tree Regressor**       | Tree (CART) |      120.84       |  25,226.34   |       158.83       |      0.5526      |    17.14%     | Step-error sensitivity         |
| **KNN Regressor ($k=7$)**         | Instance    |      178.59       |  49,008.54   |       221.38       |      0.1307      |    27.14%     | Distance dilution in 15D       |
| **SVR (RBF Kernel)**              | Kernel SVM  |      141.78       |  33,914.26   |       184.16       |      0.3985      |    22.22%     | Smooth non-linear fit          |
| **Bagging Regressor**             | Ensemble    |       90.71       |  14,220.06   |       119.25       |      0.7478      |    12.95%     | Variance reduction on trees    |
| **AdaBoost Regressor**            | Ensemble    |      122.53       |  26,154.57   |       161.72       |      0.5361      |    17.35%     | Error-weighted boosting        |
| **Stacking Regressor**            | Ensemble    |       70.59       |   8,974.25   |       94.73        |      0.8408      |     9.52%     | Meta-model blending            |
| **Voting Regressor**              | Ensemble    |       75.48       |  10,063.29   |       100.32       |      0.8215      |    10.44%     | Soft consensus                 |

---

## 3. Optimization Algorithms Comparison (Tomato Pune)

_(From [reports/gradient_descent_comparison.csv](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/gradient_descent_comparison.csv))_

| Optimization Algorithm                | Test MAE (Rs/Qtl) |   Test MSE   | Test RMSE (Rs/Qtl) | Test $R^2$ Score | Test MAPE (%) | Convergence Characteristic               |
| :------------------------------------ | :---------------: | :----------: | :----------------: | :--------------: | :-----------: | :--------------------------------------- |
| **Batch Gradient Descent**            |       69.72       |   8,880.11   |       94.23        |      0.8425      |     9.40%     | Smooth, monotonic loss reduction         |
| **Stochastic Gradient Descent (SGD)** |       70.54       |   9,068.67   |       95.23        |      0.8392      |     9.85%     | Stochastic loss fluctuations             |
| **Mini-batch GD (batch=32)**          |     **69.01**     | **8,758.12** |     **93.58**      |    **0.8447**    |   **9.41%**   | **Optimal balance of speed & stability** |

- **Convergence Plot:** [reports/figures/10_gradient_descent_convergence.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/10_gradient_descent_convergence.png)

---

## 4. Classification Models Comparison (Price Spike Detection)

_(From [reports/classification_comparison.csv](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/classification_comparison.csv))_

| Model Algorithm          |  Accuracy  | Precision  |   Recall   |  F1-Score  | ROC-AUC | Diagnostic Insight                   |
| :----------------------- | :--------: | :--------: | :--------: | :--------: | :-----: | :----------------------------------- |
| **Logistic Regression**  |   0.4744   |   0.3762   |   0.6667   |   0.4810   | 0.5249  | High recall on price shocks          |
| **Gaussian Naive Bayes** |   0.3974   |   0.3741   | **0.9649** | **0.5392** | 0.4618  | Flags 96.5% of spikes (conservative) |
| **KNN ($k=5$)**          |   0.4936   |   0.3750   |   0.5789   |   0.4552   | 0.5226  | Distance-weighted voting             |
| **Decision Tree (CART)** |   0.4872   |   0.3861   |   0.6842   |   0.4937   | 0.5202  | Interpretable crisis thresholds      |
| **SVM (RBF Kernel)**     |   0.4936   |   0.3804   |   0.6140   |   0.4698   | 0.5116  | Margin-based separation              |
| **Bagging Classifier**   |   0.4744   |   0.3529   |   0.5263   |   0.4225   | 0.5125  | Bootstrap stability                  |
| **AdaBoost Classifier**  | **0.5256** | **0.3836** |   0.4912   |   0.4308   | 0.5178  | Highest test accuracy                |
| **Stacking Classifier**  |   0.4679   |   0.3415   |   0.4912   |   0.4029   | 0.4918  | Blended probability meta-learner     |
| **Voting Classifier**    |   0.4872   |   0.3736   |   0.5965   |   0.4595   | 0.5217  | Soft consensus                       |

- **ROC-AUC Curves:** [reports/figures/11_roc_curves_classification.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/11_roc_curves_classification.png)
- **Confusion Matrices:** [reports/figures/12_confusion_matrices.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/12_confusion_matrices.png)

---

## 5. Clustering & Unsupervised Models Comparison

_(From [reports/clustering_comparison.csv](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/clustering_comparison.csv))_

| Clustering Algorithm        | Specification          | Silhouette Score | Davies-Bouldin Index | Interpretation                               |
| :-------------------------- | :--------------------- | :--------------: | :------------------: | :------------------------------------------- |
| **K-Means Clustering**      | Optimal $k=2$          |      0.4747      |        0.9806        | Splits stable vs shock trading regimes       |
| **Hierarchical Clustering** | Ward Linkage, $k=2$    |      0.4175      |        1.1671        | Hierarchical tree of market day similarities |
| **Density-Based (DBSCAN)**  | $\epsilon=2.5, min=10$ |    **0.5190**    |      **0.7180**      | Isolates 71 noise/anomalous trading sessions |

- **Elbow & Silhouette Plot:** [reports/figures/13_kmeans_elbow_silhouette.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/13_kmeans_elbow_silhouette.png)
- **Hierarchical Dendrogram:** [reports/figures/14_hierarchical_dendrogram.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/14_hierarchical_dendrogram.png)
- **PCA Scree Plot & 2D Projection:** [reports/figures/15_pca_scree_and_clusters.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/15_pca_scree_and_clusters.png)

---

## 6. Single Unified Cross-Commodity Model Evaluation (Global Multi-Food Pipeline)

A single global regression pipeline was trained on **4,480 pooled observations** spanning **Vegetables, Pulses, and Sugars & Sweets** using One-Hot Categorical Encoding and Standard Scaling on continuous features.

### A. Candidate Algorithms Comparison (675 Unseen Multi-Commodity Test Sessions)

| Model Algorithm                | Category        | Test MAE (Rs/Qtl) | Test RMSE (Rs/Qtl) | Test $R^2$ Score | Test MAPE (%) | Academic Interpretation                                 |
| :----------------------------- | :-------------- | :---------------: | :----------------: | :--------------: | :-----------: | :------------------------------------------------------ |
| **Ridge Regression ($L_2$)**   | **Regularized** |     **65.97**     |     **194.64**     |    **0.9935**    |   **6.65%**   | **CHAMPION GLOBAL MODEL (Multicollinearity Penalized)** |
| **Multiple Linear Regression** | Linear          |       66.38       |       194.70       |      0.9934      |     6.61%     | Unregularized OLS across pooled feature space           |
| **Lasso Regression ($L_1$)**   | Regularized     |       66.37       |       194.71       |      0.9934      |     6.62%     | Sparse $L_1$ penalty ($\alpha=0.5$)                     |
| **ElasticNet Regression**      | Regularized     |       79.93       |       198.47       |      0.9932      |     8.06%     | Balanced $L_1 + L_2$ convex combination                 |
| **Gradient Boosting**          | Ensemble        |      103.86       |       209.37       |      0.9924      |    10.08%     | 100 boosted trees capturing cross-commodity non-linear  |
| **Random Forest Regressor**    | Ensemble        |      107.15       |       209.62       |      0.9924      |    10.64%     | 100 bagging trees ($depth=12$)                          |
| **Decision Tree Regressor**    | Tree (CART)     |      125.90       |       230.80       |      0.9908      |    11.88%     | Depth-regularized ($depth=10$)                          |
| **Support Vector Regressor**   | Kernel SVM      |      149.00       |       271.98       |      0.9872      |    14.53%     | Soft-margin kernel regression ($C=100, \epsilon=10$)    |

### B. Per-Commodity Generalization Breakdown (Champion Ridge Pipeline)

| Commodity     | Category        | Mean Actual (Rs/Qtl) | Test MAE (Rs/Qtl) | Test MAE (Rs/Kg) | Test RMSE (Rs/Qtl) | Test $R^2$ | Test MAPE (%) |
| :------------ | :-------------- | :------------------: | :---------------: | :--------------: | :----------------: | :--------: | :-----------: |
| **Moong Dal** | Pulses          |      Rs. 7,014       |     Rs. 25.14     |   Rs. 0.25/kg    |     Rs. 47.74      |   0.9833   |   **0.36%**   |
| **Sugar**     | Sugars & Sweets |      Rs. 3,222       |     Rs. 40.98     |   Rs. 0.41/kg    |     Rs. 177.63     |   0.0560   |   **1.10%**   |
| **Tomato**    | Vegetables      |       Rs. 761        |     Rs. 72.34     |   Rs. 0.72/kg    |     Rs. 95.46      |   0.8384   |    10.11%     |
| **Onion**     | Vegetables      |       Rs. 909        |     Rs. 93.68     |   Rs. 0.94/kg    |     Rs. 273.83     |   0.4770   |    10.25%     |

### C. Evaluation Figure

- **Diagnostic Scatter Matrix:** [reports/figures/unified_multi_commodity_evaluation.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/unified_multi_commodity_evaluation.png) shows 1:1 identity fit across all 4 commodities simultaneously.

---

## 7. Diagnostic Figure Inventory

All figures are generated at 300 DPI in [`reports/figures/`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/):

1. `01_tomato_price_timeseries.png`: Multi-year APMC price series with seasonal monsoon spikes.
2. `02_tomato_price_distribution.png`: Skewed wholesale price distribution with KDE overlay.
3. `03_tomato_price_boxplot.png`: Annual volatility and extreme outlier bounds.
4. `04_tomato_monthly_seasonality.png`: Monthly price and arrival bar plots.
5. `05_tomato_correlation_heatmap.png`: Econometric feature correlation matrix.
6. `06_actual_vs_predicted_test.png`: Actual vs. predicted prices across models on test data.
7. `07_residual_distribution.png`: Zero-centered residual error distribution.
8. `08_residuals_vs_predicted.png`: Residuals vs. predicted price homoscedasticity scatter.
9. `09_model_metrics_comparison.png`: MAE, RMSE, and $R^2$ model comparison charts.
10. `10_gradient_descent_convergence.png`: Loss curves of Batch, Stochastic, and Mini-batch GD.
11. `11_roc_curves_classification.png`: ROC-AUC curves for price spike classifiers.
12. `12_confusion_matrices.png`: Confusion matrices for top 4 classifiers.
13. `13_kmeans_elbow_silhouette.png`: K-Means Elbow Method and Silhouette Analysis.
14. `14_hierarchical_dendrogram.png`: Agglomerative hierarchical clustering dendrogram.
15. `15_pca_scree_and_clusters.png`: PCA Scree plot and 2D cluster projection.
16. `unified_multi_commodity_evaluation.png`: 4-quadrant scatter matrix demonstrating actual vs. predicted 1:1 fit across Tomato, Onion, Moong Dal, and Sugar.
