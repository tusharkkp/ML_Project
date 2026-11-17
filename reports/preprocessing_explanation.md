# Data Preprocessing, Feature Engineering & Time-Series Validation

**Course Components:**  
- **Component D (2 Marks):** Data preprocessing, cleaning, and feature engineering.  
- **Component E (2 Marks):** Data splitting, chronological validation, and data leakage prevention.  
**Dataset:** Pune APMC Mandi Tomato Wholesale Series  
**Output Dataset:** `data/processed/model_ready_pune_tomato.csv`  
**Date:** 2026-11-17  
**Author:** Senior Machine Learning Engineer & Academic Project Mentor  

---

## 1. Data Cleaning & Integrity Verification (Component D)

### A. Missing Value Analysis & Zero-Fabrication Policy
- **Audit Findings:** The Pune APMC Mandi Tomato wholesale price and arrival series contained **0 missing values** across all active trading sessions between January 2016 and December 2018.
- **Handling Market Non-Trading Days (Sundays & Mandi Holidays):**
  - Agricultural mandis do not operate on Sundays or gazetted state holidays. 
  - **Ethical Rule Applied:** We strictly reject synthetic interpolation (e.g. linearly interpolating or forward-filling prices on closed Sundays). Filling closed days with artificial prices would manufacture fictitious trading volumes and corrupt volatility metrics.
  - **Solution:** The time-series is indexed by **active auction trading sessions**. Lag features represent the price $k$ trading sessions prior, matching the exact operational workflow of market traders.

### B. Duplicate Detection & De-duplication
- **Audit Findings:** APMC mandi registers occasionally log multiple entries for distinct commercial sub-varieties (e.g. "Hybrid" vs. "Local").
- **Resolution:** Observations were aggregated and deduplicated by calendar date (`observation_date`), ensuring one unambiguous representative modal price and total arrival volume per trading day.

### C. Outlier Investigation: Retaining Genuine Economic Shocks
- **Statistical Findings:** Tukey's upper fence for Pune Tomato is **Rs. 2,250/Quintal**. 93 observations exceed this threshold, reaching a peak of **Rs. 5,700/Quintal** in July 2017.
- **Handling Decision:** These points were **explicitly retained without clipping or trimming**. In agricultural commodity modeling, price surges during monsoonal supply contractions are the defining economic shocks that policy makers and households need early warnings for. Removing them would render the machine learning model useless during real inflationary crises.

---

## 2. Feature Engineering & Mathematical Formulations

To provide machine learning algorithms with historical context, momentum, and supply dynamics, 14 engineered features were created strictly using information available at or before trading close on day $t$:

### A. Autoregressive Lag Features
1. **`price_lag_1` ($P_t$):** The modal price observed today. Captures direct short-term serial persistence ($r = +0.94$).
2. **`price_lag_3` ($P_{t-2}$):** The modal price from 3 trading sessions ago. Captures medium-term weekly momentum.
3. **`price_lag_7` ($P_{t-6}$):** The modal price from approximately 1 calendar week ago. Captures cyclical weekly baseline.

### B. Rolling Window Volatility & Boundary Features
4. **`rolling_mean_7` ($\bar{P}_{7, t}$):** The 7-trading-day moving average:
   $$\bar{P}_{7, t} = \frac{1}{7} \sum_{i=0}^6 P_{t-i}$$
   Filters out high-frequency day-to-day noise to reveal the underlying weekly price trajectory.
5. **`rolling_std_7` ($\sigma_{7, t}$):** The 7-trading-day rolling standard deviation:
   $$\sigma_{7, t} = \sqrt{\frac{1}{6} \sum_{i=0}^6 (P_{t-i} - \bar{P}_{7, t})^2}$$
   Measures local market turbulence. Rapid expansion in $\sigma_{7, t}$ signals an impending price breakout.
6. **`rolling_min_7` & `rolling_max_7`:** The lowest and highest modal price over the past 7 active sessions, defining the immediate support and resistance trading corridor.

### C. Market Momentum & Intra-Day Spread Features
7. **`price_momentum_7` ($\Delta P_{7, t}$):**
   $$\Delta P_{7, t} = P_t - P_{t-6}$$
   Quantifies the absolute directional velocity of price change over the past week.
8. **`daily_spread` ($S_t$):**
   $$S_t = P_{max, t} - P_{min, t}$$
   The difference between the highest and lowest auction bids on date $t$. A widening spread indicates high bidder uncertainty and quality disparity, which typically precedes retail price inflation.

### D. Exogenous Supply-Side Arrival Features
9. **`arrival_tonnes` ($Q_t$):** Physical volume of tomatoes unloaded at Pune Mandi on date $t$.
10. **`arrival_lag_1` ($Q_{t-1}$):** Previous session's arrival volume.
11. **`rolling_arrival_mean_7` ($\bar{Q}_{7, t}$):** 7-session moving average of market arrivals. Provides the model with an indicator of systemic supply depletion or glut.

### E. Calendar & Seasonal Indicators
12. **`month` (1 to 12):** Captures annual agricultural crop cycles.
13. **`day_of_week` (0 to 5):** Captures intra-week market patterns (e.g. Saturday stock-up trading vs. Monday resupply).
14. **`is_monsoon_season`:** Binary indicator ($1$ for June, July, August; $0$ otherwise). Explicitly alerts non-linear models to the high-risk monsoon inflation window.

---

## 3. Data Splitting & Cross-Validation Strategy (Component E)

### A. Why Random Train-Test Splitting Causes Catastrophic Failure in Time Series
In standard tabular machine learning (e.g. customer churn or image classification), data rows are assumed to be Independent and Identically Distributed (i.i.d.). Functions like `train_test_split(shuffle=True)` randomly scramble rows into train and test sets.

In **time-series forecasting**, the i.i.d. assumption is completely violated:
1. **Lookahead Data Leakage:** If day $t+1$ is randomly placed in the training set while day $t$ is in the test set, the model uses future knowledge to predict the past.
2. **Spurious Performance:** The model achieves artificially inflated $R^2 > 0.99$ in testing, but completely collapses when deployed on real future data.

### B. Implemented Solution: Chronological Partitioning
The dataset of 873 model-ready observations was chronologically partitioned into three non-overlapping historical windows:

| Partition | Proportion | Row Count | Calendar Date Window | Methodological Purpose |
|:---|:---:|:---:|:---:|:---|
| **Training Set** | **70.0%** | **611 rows** | **2016-01-08 to 2018-02-18** | Model parameter estimation and coefficient fitting. |
| **Validation Set** | **14.9%** | **130 rows** | **2018-02-19 to 2018-07-23** | Model selection, polynomial degree tuning, tree depth tuning. |
| **Test Set (Unseen)** | **15.1%** | **132 rows** | **2018-07-24 to 2018-12-30** | Final unbiased evaluation of generalization performance. |

```
[================= TRAIN (70%) =================] [=== VAL (15%) ===] [=== TEST (15%) ===]
2016-01-08                                    2018-02-18            2018-07-23         2018-12-30
---> Time Flow (Strictly Forward) ------------------------------------------------------------>
```

### C. Time-Series Cross-Validation (`TimeSeriesSplit`)
For hyperparameter tuning within the training period, standard $k$-fold cross-validation cannot be used because it shuffles folds. Instead, we implement an expanding-window **`TimeSeriesSplit` (5 folds)**:
- **Fold 1:** Train on $[0 : k]$, Validate on $[k : 2k]$
- **Fold 2:** Train on $[0 : 2k]$, Validate on $[2k : 3k]$
- **Fold 3:** Train on $[0 : 3k]$, Validate on $[3k : 4k]$
- **Fold 4:** Train on $[0 : 4k]$, Validate on $[4k : 5k]$
- **Fold 5:** Train on $[0 : 5k]$, Validate on $[5k : 6k]$
At every split, the model trains strictly on the past and validates on the immediate future.

### D. Transformation Scaling Discipline
Feature scaling (`StandardScaler`) is fitted **strictly on the Training partition** ($X_{train}$). The fitted transformation parameters (mean $\mu_{train}$, standard deviation $\sigma_{train}$) are subsequently applied to transform $X_{val}$ and $X_{test}$. No statistics from the validation or test windows ever leak into feature scaling.
