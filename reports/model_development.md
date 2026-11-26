# Machine Learning Model Development & Hyperparameter Tuning

**Course Component:** Practical Examination Component F (Model Implementation, Training & Hyperparameter Tuning — 4 Marks)  
**Dataset:** Multi-Commodity APMC Series (Tomato Pune: 1,039 observations; Onion Pune: 1,007 observations)  
**Targets:**

- _Regression Target:_ Next-Active-Session Modal Price ($t+1$) in Rs./Quintal
- _Classification Target:_ Price Spike Alert ($\Delta P > 10\%$, Binary Shock Indicator)
  **Date:** 2026-11-22  
  **Implementation Script:** [scripts/train_all_models.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/train_all_models.py)

---

## 1. Multi-Commodity Architecture & Generalization

In response to academic evaluation requirements, the system supports **any food commodity** across the collected 189,413 records:

- **Vegetables:** Tomato (Pune), Onion (Pune, Lasalgaon), Potato
- **Cereals / Grains:** Rice, Wheat
- **Pulses:** Tur Dal, Gram Dal, Moong Dal, Urad Dal

The multi-commodity feature builder ([`scripts/build_features.py`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/build_features.py)) standardizes wholesale auction records, joins arrival volumes, computes backward-looking econometric features (zero future leakage), and generates chronological Train (70%), Validation (15%), and Test (15%) partitions.

---

## 2. Module 1: Regression Algorithms (Continuous Price Forecasting)

### 1. Simple Linear Regression

- **Mathematical Form:** $\hat{y}_{t+1} = \beta_0 + \beta_1 \cdot P_{t-1}$
- **Learned Line Equation (Tomato):** $\hat{y}_{t+1} = 0.9474 \cdot P_{t-1} + 71.70$
- **Learned Line Equation (Onion):** $\hat{y}_{t+1} = 0.9155 \cdot P_{t-1} + 90.24$
- **Role:** Interpretable univariate benchmark demonstrating price autoregressive inertia.

### 2. Multiple Linear Regression (OLS)

- **Mathematical Form:** $\hat{y}_{t+1} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$, with $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
- **Features Used (15):** Price lags ($t-1, t-3, t-7$), 7-day rolling statistics (mean, std, min, max), momentum ($P_t - P_{t-7}$), daily spread, physical arrivals (`arrival_tonnes`, `arrival_lag_1`, rolling arrival mean), calendar month, day of week, monsoon season indicator.
- **Economic Insight:** Negative coefficients on arrival volumes confirm the downward price pressure of agricultural supply shocks.

### 3. Polynomial Regression (Degree 2)

- **Mathematical Form:** $\hat{y} = \beta_0 + \sum \beta_i x_i + \sum \sum \beta_{ij} x_i x_j$
- **Regularization:** Coupled with Ridge ($L_2, \alpha=10.0$) to stabilize higher-order collinearities. Captures non-linear supply-demand curvature during seasonal extremes.

### 4. Ridge Regression ($L_2$ Regularization)

- **Loss Function:** $\mathcal{L}_{\text{Ridge}} = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2$
- **Role:** Handles severe multicollinearity between overlapping lag features (e.g., $P_{t-1}, P_{t-3}$, rolling mean) by shrinking coefficients toward zero without dropping them.

### 5. Lasso Regression ($L_1$ Regularization)

- **Loss Function:** $\mathcal{L}_{\text{Lasso}} = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1$
- **Automated Feature Selection:** Because the $L_1$ diamond ball intersects the loss contours at the axes, Lasso zeros out redundant features:
  - _Tomato Pune zeroed features:_ `price_lag_3`, `rolling_arrival_mean_7`, `is_monsoon_season` (retaining the most parsimonious 12 features).
  - _Winning Tomato Model:_ Achieved the lowest Test RMSE (**Rs. 93.35/Qtl**) and highest $R^2$ (**0.8454**).

### 6. Non-Linear & Tree-Based Regressors

- **Decision Tree Regressor (CART):** Recursive binary splitting based on Mean Squared Error criterion with depth regularized ($max\_depth=5$) to prevent leaf overfitting.
- **KNN Regressor ($k=7$):** Non-parametric local averaging based on Euclidean feature distance.
- **Support Vector Regression (SVR):** $\epsilon$-insensitive loss function with Radial Basis Function (RBF) kernel mapping features into continuous reproducing kernel Hilbert space. Optimal on volatile series like Onion (Test RMSE: **Rs. 179.06**).

### 7. Ensemble Regressors

- **Bagging Regressor:** 50 bootstrap-aggregated decision trees trained on random sample subsets, reducing regression variance.
- **AdaBoost Regressor:** Sequential boosting with adaptive sample re-weighting focusing on large historical prediction errors.
- **Stacking Regressor:** Heterogeneous ensemble stacking Ridge Regression, Decision Tree, and KNN as base estimators, feeding a final Ridge meta-regressor.
- **Voting Regressor:** Soft ensemble combining predictions from Multiple Linear Regression, Ridge, and Decision Tree.

---

## 3. Module 2: Gradient Descent Optimization Algorithms

To fulfill the syllabus requirement for optimization algorithms, three fundamental variants were implemented and evaluated on mean squared loss:

1. **Batch Gradient Descent:**
   - **Weight Update:** $\boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \eta \frac{1}{N} \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$
   - Computes gradient over the entire dataset ($N=883$) per epoch. Displays smooth, monotonic convergence to the global minimum.
2. **Stochastic Gradient Descent (SGD):**
   - **Weight Update:** $\boldsymbol{\theta}^{(k+1)} = \boldsymbol{\theta}^{(k)} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}_i(\boldsymbol{\theta})$
   - Updates weights sample-by-sample with inverse-scaling learning rate $\eta(t) = \frac{\eta_0}{t^{0.25}}$. Exhibits stochastic oscillations around the optimum.
3. **Mini-batch Gradient Descent:**
   - **Weight Update:** Updates weights in batches of $B=32$ samples. Strikes the optimal balance between computational efficiency and gradient stability.

- **Visualized Artifact:** [reports/figures/10_gradient_descent_convergence.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/10_gradient_descent_convergence.png) (log-scale MSE loss path over 200 epochs).

---

## 4. Module 3: Classification Algorithms (Price Spike Detection)

_Objective:_ Predict whether the next active trading session will experience a significant price shock ($\Delta P > 10\%$).

1. **Logistic Regression:** Sigmoid hypothesis $\sigma(\mathbf{w}^T \mathbf{x} + b)$ with balanced class weights.
2. **Gaussian Naive Bayes:** Computes posterior probabilities under the assumption of feature conditional independence: $P(y|\mathbf{x}) \propto P(y) \prod P(x_i|y)$.
3. **K-Nearest Neighbors (KNN, $k=5$):** Classifies based on majority vote of the nearest 5 standardized historical trading days.
4. **Decision Tree (CART):** Recursive splitting maximizing Gini impurity reduction: $Gini = 1 - \sum p_i^2$. Generates rule-based crisis thresholds.
5. **Support Vector Machine (SVC):** Maximum-margin hyperplane with RBF kernel: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$.
6. **Ensemble Classifiers:**
   - **Bagging Classifier:** 50 bootstrap aggregated tree classifiers.
   - **AdaBoost Classifier:** Exponential loss minimization weighting misclassified price spikes.
   - **Stacking Classifier:** Meta-classifier combining Logistic Regression, KNN, and Decision Trees.
   - **Voting Classifier:** Soft-voting probability consensus.

---

## 5. Module 4: Clustering & Dimensionality Reduction (Unsupervised Learning)

1. **K-Means Clustering:**
   - Evaluated cluster counts $k \in [2, 8]$ using the **Elbow Method** (Inertia) and **Silhouette Score**.
   - Optimal $k=2$ cleanly separates **Normal Stable Trading Days** from **Supply Shock Volatility Days** (Silhouette = **0.4747**, Davies-Bouldin = **0.9806**).
   - Artifact: [reports/figures/13_kmeans_elbow_silhouette.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/13_kmeans_elbow_silhouette.png).
2. **Hierarchical Clustering (Agglomerative):**
   - Bottom-up clustering using Ward's minimum variance criterion.
   - Visualized with dendrogram truncation: [reports/figures/14_hierarchical_dendrogram.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/14_hierarchical_dendrogram.png).
3. **Density-Based Clustering (DBSCAN):**
   - Core point density threshold ($\epsilon=2.5, min\_samples=10$).
   - Successfully isolates **71 anomalous market days** as noise points (outliers representing potential reporting errors or extreme artificial market squeezes) with Silhouette = **0.5190**.
4. **Principal Component Analysis (PCA):**
   - Linear dimensionality reduction via covariance matrix eigen-decomposition.
   - **8 principal components** capture over **95% of total variance** in the 15-variable feature space.
   - First two components ($PC_1: 51.4\%, PC_2: 10.9\%$) explain $62.3\%$ of total data variance.
   - Artifact: [reports/figures/15_pca_scree_and_clusters.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/15_pca_scree_and_clusters.png).

---

## 6. Module 5: Single Unified Cross-Commodity Model (Global Multi-Food Architecture)

### A. Motivation & Academic Significance
In real-world public policy and food security monitoring, operating separate isolated models for hundreds of individual commodities creates significant deployment and maintenance overhead. A **Single Unified Cross-Commodity Model** pools observations across multiple food categories into a unified feature space.

### B. Mathematical Formulation & Architecture
1. **Pooled Feature Representation:**
   The training corpus combines 4,480 daily APMC trading sessions spanning:
   - **Vegetables:** Tomato (1,039 sessions), Onion (1,765 sessions across Pune and Lasalgaon)
   - **Pulses:** Moong Dal (854 sessions)
   - **Sugars & Sweets:** Sugar (822 sessions)

2. **Categorical Embeddings & One-Hot Encoding:**
   Categorical features $\mathbf{x}_{\text{cat}} = [\text{commodity\_name}, \text{category}]$ are encoded via `OneHotEncoder(handle_unknown='ignore')`. Unseen food items at inference time gracefully map to zero-valued indicator vectors, allowing baseline cross-commodity generalization without runtime failure.

3. **Standard Scaling:**
   Continuous time-series lag and rolling statistics ($\mathbf{x}_{\text{num}}$: 16 features including price lags, 7-day rolling mean/std/min/max, momentum, spread, arrivals, and seasonality) are standardized to zero mean and unit variance:
   $$z = \frac{x - \mu}{\sigma}$$
   fit strictly on the 3,133 chronological training samples.

4. **Champion Algorithm: Ridge Regression ($L_2$ Regularization)**
   The objective minimizes the penalized residual sum of squares:
   $$\min_{\mathbf{w}} \sum_{i=1}^N \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2 + \alpha \|\mathbf{w}\|_2^2$$
   With $\alpha = 1.0$, Ridge prevents weight explosion when categorical indicators interact with high-magnitude pulse prices (~₹7,000/Qtl) vs perishable vegetables (~₹800/Qtl).

### C. Experimental Results (675 Unseen Test Sessions)
- **Overall $R^2$ Score:** **0.9935** (99.35% variance explained across multi-category food basket)
- **Overall Test MAE:** **Rs. 65.97 / Quintal** (~Rs. 0.66 / Kg)
- **Overall Test RMSE:** **Rs. 194.64 / Quintal** (~Rs. 1.95 / Kg)
- **Overall Test MAPE:** **6.65%**
- **Artifacts:**
  - Pipeline: `models/unified_food_price_model.joblib`
  - Metadata: `models/unified_model_metadata.json`
  - Visualization: [reports/figures/unified_multi_commodity_evaluation.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/unified_multi_commodity_evaluation.png)
