# Descriptive Statistical Analysis of Commodity Prices

**Course Component:** Practical Examination Component B (Statistical Summary & Descriptive Statistics — 2 Marks)  
**Dataset Analyzed:** Pune APMC Mandi Daily Wholesale Series (2016–2018)  
**Date:** 2026-11-17  
**Author:** Senior Machine Learning Engineer & Academic Project Mentor  

---

## 1. Overview of Evaluated Statistical Operations

The practical examination evaluation rubric requires applying and interpreting **at least three distinct descriptive statistical operations**. In this study, we implement **seven statistical measures** across the Pune APMC Mandi daily wholesale time-series, comparing the perishable benchmark (**Tomato**) with the staple benchmark (**Onion**).

All prices are measured in **Indian Rupees per Quintal (Rs./Quintal)**, where $1\text{ Quintal} = 100\text{ Kilograms}$.

---

## 2. Statistical Metrics & Economic Interpretation

### A. Summary Table: Pune APMC Mandi (2016–2018)

| Statistical Operation | Mathematical Formulation | Tomato (Pune Mandi) | Onion (Pune Mandi) | Comparative Economic Insight |
|:---|:---:|---:|---:|:---|
| **Sample Size ($N$)** | $N = \sum_{i=1}^n 1$ | **880 observations** | **869 observations** | Dense 3-year daily trading continuity. |
| **Arithmetic Mean ($\mu$)** | $\frac{1}{N} \sum_{i=1}^N x_i$ | **Rs. 1,186.16** | **Rs. 1,016.92** | Average price level over 3 complete crop cycles. |
| **Median ($Q_2$)** | Value at 50th percentile | **Rs. 800.00** | **Rs. 800.00** | Typical day-to-day trading benchmark. |
| **Standard Deviation ($\sigma$)** | $\sqrt{\frac{\sum (x_i - \mu)^2}{N - 1}}$ | **Rs. 1,035.11** | **Rs. 580.61** | Tomato displays nearly **double** the volatility of Onion. |
| **Variance ($\sigma^2$)** | $\frac{\sum (x_i - \mu)^2}{N - 1}$ | **1,071,459.41** | **337,109.84** | Measures dispersion around the mean price. |
| **Minimum ($P_{min}$)** | $\min(X)$ | **Rs. 300.00** | **Rs. 400.00** | Post-harvest glut floor price (Rs. 3.00/kg). |
| **Maximum ($P_{max}$)** | $\max(X)$ | **Rs. 5,700.00** | **Rs. 3,500.00** | Peak monsoon supply disruption price (Rs. 57.00/kg). |
| **25th Percentile ($Q_1$)** | Value at 25th percentile | **Rs. 500.00** | **Rs. 600.00** | Baseline low-price regime threshold. |
| **75th Percentile ($Q_3$)** | Value at 75th percentile | **Rs. 1,200.00** | **Rs. 1,100.00** | Elevated price regime threshold. |
| **Interquartile Range (IQR)** | $Q_3 - Q_1$ | **Rs. 700.00** | **Rs. 500.00** | Robust measure of typical price spread. |
| **Skewness ($\gamma_1$)** | $\frac{\frac{1}{N}\sum (x_i - \mu)^3}{\sigma^3}$ | **+2.12** | **+1.98** | Highly positive right skew: rapid spikes, slow declines. |
| **Kurtosis ($\gamma_2$)** | $\frac{\frac{1}{N}\sum (x_i - \mu)^4}{\sigma^4} - 3$ | **+4.20** | **+3.85** | Leptokurtic (fat-tailed): high probability of extreme spikes. |

---

## 3. Deep Dive into Individual Statistical Operations

### 1. Arithmetic Mean vs. Median (Divergence of Central Tendency)
- **Mathematical Reality:** For Tomato, the mean ($\mu = \text{Rs. } 1,186.16$) exceeds the median ($Q_2 = \text{Rs. } 800.00$) by **Rs. 386.16 (48.3%)**.
- **Economic Insight:** In symmetric distributions (Gaussian), mean equals median. In agricultural commodities, this massive divergence proves that **Tomato prices do not follow a normal distribution**. Daily prices spend most of the year at low or moderate levels (half of all trading days are at or below Rs. 800/Qtl), but occasional, severe supply crunches pull the arithmetic mean upward. 
- **Significance for Machine Learning:** Models optimized purely on Mean Squared Error (MSE) will disproportionately penalize errors on rare high spikes. This justifies evaluating Mean Absolute Error (MAE) and using tree-based non-linear estimators.

### 2. Standard Deviation and Variance (Volatility Analysis)
- **Tomato Standard Deviation ($\sigma = \text{Rs. } 1,035.11$):**
  - Coefficient of Variation ($CV = \frac{\sigma}{\mu} = \frac{1035.11}{1186.16} = 87.26\%$).
- **Onion Standard Deviation ($\sigma = \text{Rs. } 580.61$):**
  - Coefficient of Variation ($CV = \frac{580.61}{1016.92} = 57.09\%$).
- **Economic Insight:** Tomatoes have a $CV$ of 87.3%, meaning day-to-day volatility is nearly equal to the average price itself. This reflects the extreme perishability of tomatoes: farmers cannot hold harvested tomatoes off the market to wait for better prices, causing violent price elasticity when arrivals fluctuate.

### 3. Quartiles and Interquartile Range (IQR) (Outlier Bounds)
- **Lower Quartile ($Q_1$):** Rs. 500.00 (25% of trading sessions trade below Rs. 5.00/kg).
- **Upper Quartile ($Q_3$):** Rs. 1,200.00 (75% of trading sessions trade below Rs. 12.00/kg).
- **IQR ($Q_3 - Q_1$):** Rs. 700.00.
- **Tukey's Upper Outlier Fence:**
  $$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR} = 1200 + 1.5 \times 700 = \text{Rs. } 2,250.00$$
- **Finding:** Any trading price above **Rs. 2,250.00/Qtl** is mathematically an outlier. In our dataset, 93 observations (10.6% of trading days) fall above Rs. 2,250, reaching up to Rs. 5,700/Qtl.
- **Significance for Machine Learning:** These are **not data entry errors**; they represent genuine food inflation spikes (the infamous July 2017 monsoon tomato surge). Discarding them would blind the forecasting model to the exact phenomena it is intended to predict.

### 4. Skewness and Kurtosis (Higher-Order Moments)
- **Skewness (+2.12):** Substantially greater than 0, reflecting strong positive asymmetry. Prices rise rapidly and dramatically during supply shocks, but descend more gradually as new harvests arrive.
- **Kurtosis (+4.20):** A kurtosis above 3 indicates a **leptokurtic** distribution with fatter tails than a normal distribution. Extreme price events (shocks) happen far more often than Gaussian assumptions would anticipate, confirming that linear models will benefit from non-linear feature transformations and polynomial expansion.
