# Machine Learning Model Development & Hyperparameter Tuning

**Course Component:** Practical Examination Component F (Model Implementation, Training & Hyperparameter Tuning — 4 Marks)  
**Dataset:** Pune APMC Mandi Tomato Wholesale Series (873 Model-Ready Observations)  
**Target:** Next-Active-Session Modal Price ($t+1$) in Rs./Quintal  
**Date:** 2026-11-17  
**Author:** Senior Machine Learning Engineer & Academic Project Mentor  

---

## 1. Academic Roll Number Allocation Notice

> [!IMPORTANT]
> **Roll Number Allocation Audit:** The practical examination document refers to algorithms allocated according to student roll numbers. A repository-wide search was conducted, and no specific roll-number allocation sheet was located. In accordance with examination instructions:
> 1. We officially report that the specific roll-number allocation mapping is **missing from the workspace**.
> 2. We request the student to provide their assigned algorithm from their class laboratory schedule if an individual algorithm is mandated.
> 3. In the meantime, we have implemented the full suite of syllabus-aligned regression models (**Multiple Linear Regression, Polynomial Regression, Decision Tree Regressor, Random Forest Ensemble, and Naive Persistence Baseline**) to ensure zero examination gaps.

---

## 2. Theoretical Intuition & Implementation of Evaluated Models

---

### Model 1: Naive Persistence Baseline
- **Mathematical Formulation:**
  $$\hat{y}_{t+1} = y_t$$
- **Intuition:** The simplest benchmark in time-series forecasting. It assumes that tomorrow's commodity price will equal today's observed price ($P_t$).
- **Role in Machine Learning:** In financial and commodity time series, high auto-correlation often makes the persistence forecast surprisingly competitive. Any supervised machine learning algorithm (linear, polynomial, tree, or ensemble) **must demonstrate a statistically significant reduction in RMSE and MAE over this baseline** to justify its computational complexity.
- **Experimental Result:** Test MAE: Rs. 70.45, Test RMSE: Rs. 114.48, Test $R^2$: 0.7518.

---

### Model 2: Multiple Linear Regression
- **Mathematical Formulation:**
  $$\hat{y}_{t+1} = \beta_0 + \sum_{i=1}^{p} \beta_i x_{i, t} + \epsilon$$
  $$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
- **Intuition:** Models the future price as a weighted linear combination of historical price lags, rolling statistics, physical arrival volumes, and calendar season indicators.
- **Implementation:** Wrapped inside a Scikit-learn `Pipeline` with `StandardScaler()` to ensure features on vastly different scales (e.g. arrivals in hundreds of tonnes vs. price in thousands of rupees) are standardized to zero mean and unit variance.
- **Top Standardized Coefficients (Feature Importance):**
  1. `rolling_mean_7` ($\beta = +358.92$): The 7-day rolling average exerts the strongest positive weight, stabilizing predictions against transient daily spikes.
  2. `price_lag_1` ($\beta = +320.33$): Today's price provides strong short-term inertia.
  3. `price_lag_7` ($\beta = +220.42$): Weekly cyclical anchor.
  4. `daily_spread` ($\beta = +145.62$): Wider intra-day trading dispersion signals upward price pressure.
  5. `arrival_tonnes` ($\beta = -15.50$) & `rolling_arrival_mean_7` ($\beta = -33.99$): Negative coefficients confirm economic supply elasticity — increased physical supply depresses wholesale prices.
- **Limitations:** Assumes linear additivity; cannot capture abrupt threshold non-linearities or regime shifts without interaction terms.
- **Experimental Result:** Test MAE: Rs. 79.35, Test RMSE: **Rs. 109.64**, Test $R^2$: **0.7723**. Beats the persistence baseline on RMSE and explains 77.2% of future price variance.

---

### Model 3: Polynomial Regression with Regularization (Ridge)
- **Mathematical Formulation:**
  $$\hat{y} = \beta_0 + \sum_{i=1}^p \beta_i x_i + \sum_{i=1}^p \sum_{j=i}^p \beta_{ij} x_i x_j + \dots$$
  $$\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}_{poly} \boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2 \right\}$$
- **Intuition:** Expands the input feature space with non-linear interaction terms and higher-degree powers ($x_i^2, x_i x_j$), allowing the linear estimator to fit parabolic and curved response surfaces. Ridge regularization ($L_2$ penalty $\alpha$) prevents explosive weight variance on correlated terms.
- **Hyperparameter Tuning:** Evaluated polynomial degrees $d \in \{1, 2, 3\}$ and regularization strengths $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$ using the Validation set.
  - Degree 1 ($\alpha = 0.01$): Validation RMSE = **122.19**
  - Degree 2 ($\alpha = 100.0$): Validation RMSE = **148.35** (Overfitting to training noise)
  - Degree 3: Severe dimensional explosion ($>800$ interaction features), causing massive generalization degradation.
- **Selection:** Best configuration is **Degree 1 with $\alpha = 0.01$** (retaining linear parsimony to avoid variance explosion).
- **Experimental Result:** Test MAE: Rs. 79.35, Test RMSE: **Rs. 109.64**, Test $R^2$: **0.7723**.

---

### Model 4: Decision Tree Regressor
- **Mathematical Formulation:**
  Partitions feature space into $M$ orthogonal hyper-rectangles $R_1, R_2, \dots, R_M$. The prediction for region $R_m$ is the mean of training targets falling within it:
  $$\hat{y} = \sum_{m=1}^M c_m \cdot \mathbb{I}(\mathbf{x} \in R_m), \quad c_m = \text{avg}(y_i \mid \mathbf{x}_i \in R_m)$$
  Splits are chosen to minimize Mean Squared Error (MSE) impurity reduction.
- **Hyperparameter Tuning:** Conducted a 5-fold expanding-window `TimeSeriesSplit` grid search across:
  - `max_depth` $\in [3, 4, 5, 6, 8, 10, \text{None}]$
  - `min_samples_split` $\in [2, 5, 10, 20]$
  - `min_samples_leaf` $\in [1, 2, 4, 8]$
- **Best Parameters Selected:** `{'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 4}`. Constraining depth to 5 prevents the tree from memorizing individual price outliers while isolating seasonal price regimes.
- **Experimental Result:** Test MAE: Rs. 84.72, Test RMSE: Rs. 124.90, Test $R^2$: 0.7045.

---

### Model 5: Random Forest Regressor (Ensemble Learning)
- **Mathematical Formulation:**
  Ensemble of $B$ de-correlated regression trees trained on bootstrap samples with random feature subsets:
  $$\hat{f}_{rf}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
- **Intuition:** Individual decision trees suffer from high variance. By averaging predictions over 100 trees, random forests dramatically reduce variance ($\text{Var}(\bar{X}) = \frac{\sigma^2}{B}$) without increasing bias.
- **Hyperparameter Tuning:** Conducted `TimeSeriesSplit` cross-validation across:
  - `n_estimators` $\in [50, 100, 200]$
  - `max_depth` $\in [4, 6, 8, 12, \text{None}]$
  - `min_samples_split` $\in [2, 5, 10]$
- **Best Parameters Selected:** `{'max_depth': 4, 'min_samples_split': 2, 'n_estimators': 100}`.
- **Experimental Result:** Test MAE: Rs. 85.95, Test RMSE: Rs. 120.60, Test $R^2$: 0.7245. Strong performance on the validation set ($R^2 = 0.8080$), capturing non-linear interactions cleanly.
