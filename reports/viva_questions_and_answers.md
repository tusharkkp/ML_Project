# Academic Practical Viva Voce & Oral Examination Preparation Guide

**Course:** B.Tech Third-Year — Artificial Intelligence and Machine Learning Laboratory  
**Component:** Practical Examination Component I (Presentation & Oral Exam Preparation — 4 Marks)  
**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Date:** 2026-11-17  
**Author:** Senior Machine Learning Engineer & Academic Project Mentor  

---

## Section 1: Machine Learning & Regression Fundamentals (Theory Syllabus)

### Q1: What is the fundamental difference between Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression?
**Answer:**
- **Simple Linear Regression** models the linear relationship between a single independent variable $x$ and a dependent variable $y$:
  $$y = \beta_0 + \beta_1 x + \epsilon$$
- **Multiple Linear Regression** models the dependent variable as a linear function of $p$ distinct independent features:
  $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$
  In our project, we use 15 engineered features including past price lags ($P_t, P_{t-2}, P_{t-6}$), rolling window averages, daily intra-day spreads, physical arrival tonnage, and seasonal indicators.
- **Polynomial Regression** models non-linear curved relationships by projecting the original features into higher-degree polynomial feature spaces ($x_i^2, x_i x_j, x_j^2$):
  $$y = \beta_0 + \sum_{i=1}^p \beta_i x_i + \sum_{i=1}^p \sum_{j=i}^p \beta_{ij} x_i x_j + \dots + \epsilon$$
  Although the relationship between $x$ and $y$ is non-linear, it is still classified as a *linear model* because it is linear in the parameters ($\beta$).

---

### Q2: How did you implement Polynomial Regression in this project, and why did higher polynomial degrees fail?
**Answer:**
We implemented Polynomial Regression using Scikit-learn's `PolynomialFeatures` combined with `StandardScaler` and `Ridge` regularization. We evaluated degrees $d \in \{1, 2, 3\}$.
- **Why higher degrees failed:** Degree 1 (linear with Ridge $\alpha=0.01$) achieved a test RMSE of **Rs. 109.64/Qtl**. When expanding to Degree 2, the feature space exploded from 15 features to 135 interaction features. On a training set of 611 rows, this caused the "Curse of Dimensionality" and extreme model variance — fitting to local training fluctuations and worsening validation RMSE to **Rs. 148.35/Qtl**. Degree 3 created over 800 features, leading to complete generalization collapse. This demonstrated the classic principle that parsimonious linear models with regularized shrinkage often outperform unconstrained high-degree polynomial expansions on noisy economic data.

---

### Q3: What is Ridge Regularization ($L_2$ Regularization) and what role does the hyperparameter $\alpha$ play?
**Answer:**
Ridge regression adds an $L_2$-norm penalty proportional to the sum of squared coefficients to the Ordinary Least Squares (OLS) loss function:
$$\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2$$
- **Role of $\alpha$:** $\alpha \ge 0$ is the regularization strength. 
  - When $\alpha = 0$, Ridge is equivalent to standard OLS.
  - As $\alpha$ increases, it shrinks non-essential coefficients toward zero (without forcing them exactly to zero, which is Lasso's $L_1$ behavior).
  - This dampens multicollinearity among correlated price lags ($P_t$ vs. $P_{t-2}$) and prevents explosive variance. Through grid search, we selected $\alpha = 0.01$.

---

### Q4: How does a Decision Tree Regressor determine its splits? What impurity metric is used?
**Answer:**
Unlike classification trees that use Gini Impurity or Shannon Entropy, a **Decision Tree Regressor** uses **Mean Squared Error (MSE) / Variance Reduction** as its split criterion:
$$\text{MSE}(R_m) = \frac{1}{|R_m|} \sum_{i \in R_m} (y_i - \bar{y}_m)^2$$
At every node, the algorithm exhaustively evaluates all candidate feature thresholds to find the split that maximizes the reduction in total sum of squared errors between the parent node and the two resulting child nodes. Predictions at leaf nodes are calculated as the simple average ($\bar{y}_m$) of all training points assigned to that leaf.

---

### Q5: Why does a Random Forest Regressor typically outperform a single Decision Tree? Explain Bagging.
**Answer:**
A single Decision Tree has low bias but **very high variance** — a minor perturbation in the training dataset can completely alter the tree's split hierarchy. 
A **Random Forest** solves this through **Bootstrap Aggregation (Bagging)** and **Random Feature Subspacing**:
1. It trains an ensemble of $B$ deep decision trees (we tuned $B = 100$ trees) on random bootstrap samples (sampling with replacement) of the training dataset.
2. At every node split, only a random subset of features ($\sqrt{p}$ or $p/3$) is considered, de-correlating the trees.
3. The final prediction is the average across all individual trees:
   $$\hat{f}_{rf}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
4. According to statistical theory, averaging $B$ independent, identically distributed estimators reduces variance by a factor of $1/B$ without increasing bias:
   $$\text{Var}(\bar{T}) = \frac{\sigma^2}{B} + \frac{B-1}{B} \rho \sigma^2$$
In our experiments, Random Forest achieved a strong validation RMSE of **116.78** ($R^2 = 0.8080$), substantially outperforming the single tuned Decision Tree (RMSE = **122.91**).

---

### Q6: Explain the Bias-Variance Tradeoff in the context of your experimental results.
**Answer:**
Total expected prediction error consists of three components:
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error } (\sigma_{\epsilon}^2)$$
- **Underfitting (High Bias):** A model that is too simplistic to capture the underlying pattern (e.g., an un-lagged horizontal mean forecast).
- **Overfitting (High Variance):** A model that fits training noise and memorizes outliers, failing on unseen test data (e.g., Polynomial Degree 2 and 3).
- **Optimal Balance:** Multiple Linear Regression with 15 engineered features and tuned Decision Tree / Random Forest found the sweet spot: maintaining sufficient capacity to capture supply-demand dynamics while achieving low test error (Test RMSE = **Rs. 109.64/Qtl**, $R^2 = 0.7723$).

---

## Section 2: Time-Series Preprocessing & Validation Discipline

### Q7: Why is random train-test splitting (`train_test_split(shuffle=True)`) strictly prohibited for time series?
**Answer:**
Random train-test splitting assumes that data observations are **Independent and Identically Distributed (i.i.d.)**. 
In commodity time series, consecutive days are highly correlated ($r = +0.94$). If data is shuffled randomly:
1. Observations from the future ($t+1, t+5$) enter the training set, while past observations ($t$) enter the test set.
2. The model learns by "peeking into the future" — a phenomenon known as **lookahead data leakage**.
3. The model achieves near-perfect test scores ($R^2 > 0.99$) during lab testing, but experiences catastrophic failure when deployed on real, unfolding future trading days.
We strictly enforced **Chronological Non-Overlapping Partitioning**: 70% Train (2016 to early 2018), 15% Validation (early 2018 to mid 2018), and 15% Test (late 2018).

---

### Q8: How does `TimeSeriesSplit` cross-validation differ from standard $k$-fold cross-validation?
**Answer:**
Standard $k$-fold cross-validation randomly partitions data into $k$ buckets, using $k-1$ folds for training and 1 fold for testing. This shuffles past and future data indiscriminately.
In contrast, **`TimeSeriesSplit` (Rolling / Expanding Window Cross-Validation)** respects temporal ordering:
- In Fold 1, it trains on the earliest window and tests on the next forward slice.
- In Fold 2, the training window expands to include Fold 1's test slice, and tests on the subsequent slice.
- At no point does a training fold contain observations that occurred chronologically after the validation fold.

---

### Q9: Why must feature scaling (`StandardScaler`) be fitted exclusively on the Training set?
**Answer:**
If you scale the entire dataset together using `scaler.fit_transform(X)`, the transformation uses the global mean ($\mu_{global}$) and global standard deviation ($\sigma_{global}$), which incorporates data from the future validation and test periods. This constitutes **subtle data leakage**.
To guarantee zero leakage:
1. `scaler.fit(X_train)` computes $\mu_{train}$ and $\sigma_{train}$.
2. `scaler.transform(X_val)` and `scaler.transform(X_test)` standardize validation and test sets using the *training parameters only*.

---

### Q10: Why didn't you fill missing Sunday prices using linear interpolation or forward fill?
**Answer:**
Because agricultural mandis are **officially closed on Sundays**. There is no trading, no price discovery, and no arrival volume.
If we had interpolated Sunday prices, we would be fabricating synthetic data that never existed in the APMC auction ledger, directly violating the project's **Zero Fabrication Policy**. Instead, we indexed our time-series by **active trading sessions** ($t, t+1, t+2$). This preserves 100% empirical authenticity.

---

## Section 3: Model Evaluation & Metric Interpretation

### Q11: Explain the mathematical and practical difference between MAE, MSE, and RMSE.
**Answer:**
- **MAE (Mean Absolute Error):** $\frac{1}{n} \sum |y_i - \hat{y}_i|$. Measures average error linearly. An error of Rs. 100 is penalized exactly twice as much as an error of Rs. 50. Our model achieved MAE = **Rs. 79.35/Qtl** (~79 paise per kg).
- **MSE (Mean Squared Error):** $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$. Squares errors, penalizing large outliers disproportionately. An error of Rs. 100 is penalized four times as much as Rs. 50.
- **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}}$. Restores the physical units (Rs./Quintal). Because it squares errors before averaging, RMSE is always $\ge \text{MAE}$. When RMSE is substantially larger than MAE, it indicates the presence of occasional large prediction errors. Our model achieved RMSE = **Rs. 109.64/Qtl**.

---

### Q12: Why shouldn't an engineer use $R^2$ alone to evaluate a time-series forecasting model?
**Answer:**
$R^2$ measures the proportion of variance explained relative to a horizontal line predicting the mean ($\bar{y}$):
$$R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}$$
In high-variance non-stationary time series:
1. A naive persistence model ($\hat{y}_{t+1} = y_t$) can achieve an $R^2$ of **0.7518** simply by tracking yesterday's price, despite having zero intelligence.
2. $R^2$ can be artificially high if the data has strong seasonal swings.
3. Therefore, $R^2$ must always be paired with **RMSE, MAE, and MAPE**, and benchmarked directly against the **Naive Persistence Baseline**.

---

### Q13: What is the Naive Persistence Baseline, and did your machine learning model beat it?
**Answer:**
The Naive Persistence Baseline predicts $\hat{y}_{t+1} = y_t$. On our unseen test set, the Baseline achieved:
- MAE = Rs. 70.45, RMSE = Rs. 114.48, $R^2 = 0.7518$.
Our **Multiple Linear Regression model** achieved:
- MAE = Rs. 79.35, RMSE = **Rs. 109.64**, $R^2 = \mathbf{0.7723}$.
**Conclusion:** Multiple Linear Regression beat the baseline by reducing RMSE by **Rs. 4.84/Qtl** and explaining an additional **2.05% of target variance**, proving that incorporating supply arrivals, moving averages, and intra-day spreads adds genuine predictive power over naive lag copying.

---

### Q14: How did you evaluate residual errors? Were they normally distributed?
**Answer:**
We plotted three diagnostic plots:
1. **Residual Distribution (Figure 7):** Residuals ($e_t = y_{test} - \hat{y}_{test}$) formed a symmetric bell curve centered at $\mu_e = -\text{Rs. } 2.45$, confirming that the model is **unbiased** (no systematic under- or over-prediction).
2. **Residuals vs. Predicted Plot (Figure 8):** Inspected for heteroscedasticity. Points were evenly dispersed around zero, though variance expanded slightly at peak price regimes ($> \text{Rs. } 1,200$), reflecting higher real-world market volatility during inflationary periods.

---

## Section 4: Domain Economics & System Mechanics

### Q15: Why are Retail Prices and Wholesale Mandi Prices kept strictly separate in your database?
**Answer:**
In agricultural economics, wholesale and retail prices represent fundamentally different market levels and physical units:
- **Wholesale Mandi Prices:** Measured in **Rs./Quintal** (100 kg) during morning auctions inside regulated APMC yards. They reflect primary producer supply ($Q_t$) and merchant bids.
- **Retail Prices:** Measured in **Rs./Kg** or **Rs./Litre** at consumer retail outlets. They include mandi market cess (1%–2%), transport freight, handling wastage (10%–15% perishability loss), cold chain costs, and retailer markups (30%–60%).
Mixing or linearly dividing wholesale prices by 100 to equate them with retail prices creates severe methodological corruption. In our system, they are stored in separate files (`wholesale_prices.csv` vs. `retail_prices.csv`).

---

### Q16: What is an APMC Mandi, and what is the difference between Min, Max, and Modal Price?
**Answer:**
- **APMC:** Agricultural Produce Market Committee — a statutory marketing board established by state governments in India to prevent farmer exploitation by intermediaries.
- **Minimum Price:** Lowest price bid during the auction session (usually lower-grade produce or end-of-day liquidation).
- **Maximum Price:** Highest bid of the day (premium, sorted export quality).
- **Modal Price:** The price at which the **largest physical volume** was auctioned. It is the single most accurate reflection of the clearing price in the market and is the standard metric used by the RBI and government agencies for food inflation tracking.

---

### Q17: How did physical market arrivals ($Q_t$) affect your price predictions?
**Answer:**
In classical microeconomics, the law of supply states that an increase in supply ($Q$) ceteris paribus exerts downward pressure on price ($P$).
In our Multiple Linear Regression model:
- `arrival_tonnes` received a standardized coefficient of **$-15.50$**.
- `rolling_arrival_mean_7` received a standardized coefficient of **$-33.99$**.
This empirically validates economic theory: as physical arrival tonnage increases at Pune Mandi, the model predicts lower future wholesale prices.

---

### Q18: How does the Household Food Basket Affordability Calculator work? Is it an ML model?
**Answer:**
The Affordability Calculator is **intentionally deterministic, not a machine learning model**. It uses standard econometric accounting formulas:
1. **Total Food Basket Cost:**
   $$\text{Cost}_{monthly} = \sum_{i=1}^k (\text{Quantity}_i \times \text{Retail Price}_i)$$
2. **Food Budget Share (%):**
   $$\text{Share}_{\%} = \left( \frac{\text{Total Food Basket Cost}}{\text{Monthly Household Income}} \right) \times 100$$
3. **NSSO 2022-23 Benchmark Comparison:**
   According to the official MoSPI NSSO HCES 2022-23 survey:
   - Rural households spend an average of **46.38%** of their total expenditure on food.
   - Urban households spend an average of **39.17%** of their expenditure on food.
   - The bottom 20% fractile spends **56.60%** on food.
4. **Vulnerability Alerting:** When predicted wholesale price increases transmit to retail baskets, the system simulates whether a family's grocery bill will push their food budget share past the critical 50% vulnerability threshold.

---

## Section 5: Deployment & Code Architecture

### Q19: What model artifacts were saved in `models/`, and why is `model_metadata.json` critical?
**Answer:**
We serialized three files using `joblib` and `json`:
1. `models/price_prediction_model.joblib`: The trained Scikit-learn Pipeline (Multiple Linear Regression with `StandardScaler`).
2. `models/preprocessing_pipeline.joblib`: The standalone fitted feature scaler.
3. `models/model_metadata.json`: A comprehensive metadata contract documenting:
   - Target variable and definition (`target_price_t_plus_1`)
   - Exact feature names and input ordering (15 features)
   - Evaluated test metrics (RMSE = 109.64, $R^2 = 0.7723$)
   - Software dependency versions (`scikit-learn==1.9.1`, `joblib==1.6.0`, `python==3.12`)
`model_metadata.json` is critical for production deployment because it ensures that incoming API requests are validated against the exact feature order used during training, preventing silent feature misalignment bugs.

---

### Q20: How does `scripts/inference_test.py` prove deployment readiness?
**Answer:**
`scripts/inference_test.py` executes a complete standalone inference run:
1. It loads `price_prediction_model.joblib` and `model_metadata.json` from disk.
2. It constructs an unseen feature vector (simulating a Thursday in July with Rs. 1,400/Qtl modal price and 145 tonnes arrival).
3. It validates the features, extracts the feature order from metadata, and calls `.predict()`.
4. It outputs the predicted price (**Rs. 1,343.29/Qtl**, or **Rs. 13.43/Kg**) and expected daily change ($-56.71$ Rs./Qtl) in under 2 milliseconds without needing access to raw training data or retraining the model.
