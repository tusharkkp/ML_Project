# Exploratory Data Analysis & Visualization Findings

**Course Component:** Practical Examination Component C (Data Exploration & Visualization — 2 Marks)  
**Figures Location:** `reports/figures/`  
**Date:** 2026-11-17  
**Author:** Senior Machine Learning Engineer & Academic Project Mentor  

---

## 1. Overview of Visualization Methodology

In accordance with academic requirements, five distinct visualization methods were implemented to uncover temporal dynamics, distributional asymmetry, seasonal cycles, and feature interdependencies.

All plots were generated using Matplotlib and Seaborn with standardized aesthetics, clear legends, physical units (`Rs./Quintal`), and saved in high-resolution PNG format.

---

## 2. In-Depth Interpretation of Visualizations

---

### Figure 1: Historical Time-Series Price Movement & 30-Day Moving Average
- **File:** [01_tomato_price_timeseries.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/01_tomato_price_timeseries.png)
- **What is Plotted:** Daily modal price of Tomato (blue line) alongside a 30-day centered rolling moving average (red line) across 880 active trading sessions at Pune APMC Mandi (January 2016 to December 2018).
- **Analytical Utility:** Reveals multi-year macroeconomic cycles, high-frequency day-to-day noise versus underlying medium-term price trends, and the exact timing of food inflation shocks.
- **Key Empirical Observations:**
  1. **Extreme Seasonality in Mid-Year:** Pronounced price spikes occur reliably in mid-year (June–August) each year, reaching Rs. 3,600/Qtl in June 2016, Rs. 5,700/Qtl in July 2017, and Rs. 2,800/Qtl in July 2018.
  2. **Winter Gluts:** Prices plummet to historical lows (Rs. 300 to Rs. 500/Qtl) every year between December and March during winter rabi harvest arrivals.
  3. **High Serial Autocorrelation:** Day-to-day price shifts exhibit strong momentum, justifying the creation of autoregressive lag features ($t-1, t-3, t-7$).
- **Economic Interpretation:** Tomatoes have a short crop cycle (90–120 days). During the peak monsoon (July), heavy rain damages flowering plants and disrupts road transportation across Maharashtra, choking supply and triggering severe headline food inflation. By winter, harvest from multiple states arrives simultaneously, creating a glut and price collapse.

---

### Figure 2: Price Density Distribution (Histogram with KDE)
- **File:** [02_tomato_price_distribution.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/02_tomato_price_distribution.png)
- **What is Plotted:** Frequency histogram (35 bins) overlaid with a continuous Kernel Density Estimation (KDE) curve, Arithmetic Mean (red dashed line at Rs. 1,186.16), and Median (blue dashed-dot line at Rs. 800.00).
- **Analytical Utility:** Evaluates whether prices adhere to Gaussian assumptions, quantifying distributional skewness and tail fatness.
- **Key Empirical Observations:**
  1. **Right-Skewed Unimodal Shape:** The mode of the distribution is centered tightly between Rs. 500 and Rs. 800/Qtl. Over 50% of all observations fall below Rs. 800.
  2. **Elongated Right Tail:** The distribution extends far to the right, up to Rs. 5,700/Qtl.
  3. **Divergence of Mean and Median:** The mean sits Rs. 386.16 above the median due to the leverage exerted by the upper tail.
- **Economic Interpretation:** Households experience "normal" or manageable wholesale prices (~Rs. 6 to Rs. 10/kg) for the majority of the year, but suffer extreme, sudden budget pressure during episodic tail events. Standard linear regression models will tend to underpredict these extreme spikes unless features capturing momentum and rolling variance are provided.

---

### Figure 3: Boxplot of Price Dispersion & Outlier Fences
- **File:** [03_tomato_price_boxplot.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/03_tomato_price_boxplot.png)
- **What is Plotted:** Horizontal Tukey boxplot depicting the minimum (Rs. 300), First Quartile $Q_1$ (Rs. 500), Median $Q_2$ (Rs. 800), Third Quartile $Q_3$ (Rs. 1,200), and points exceeding the upper fence ($Q_3 + 1.5 \times \text{IQR} = \text{Rs. } 2,250$).
- **Analytical Utility:** Quantifies the interquartile range (IQR = Rs. 700) and visually isolates statistical outliers.
- **Key Empirical Observations:**
  1. **Concentrated Middle 50%:** Half of all trading days occur within a narrow corridor of Rs. 500 to Rs. 1,200/Qtl.
  2. **Outlier Cluster:** 93 observations (10.6% of data) lie beyond the upper fence, forming a continuous cluster of high prices between Rs. 2,300 and Rs. 5,700/Qtl.
  3. **Absence of Low Outliers:** The lower whisker reaches Rs. 300/Qtl; there are zero lower outliers because prices cannot drop below zero.
- **Economic Interpretation:** In financial and economic engineering, high outliers in perishable agricultural commodities are **vital market signals**, not erroneous readings. They reflect acute supply shortages where demand remains relatively inelastic. They must be preserved during model training.

---

### Figure 4: Monthly Seasonality & Variance Analysis
- **File:** [04_tomato_monthly_seasonality.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/04_tomato_monthly_seasonality.png)
- **What is Plotted:** Bar chart of monthly mean modal prices with standard deviation error bars for each calendar month (January to December).
- **Analytical Utility:** Identifies recurring annual calendar cycles for feature engineering (e.g. creating month indicators or cyclical sine/cosine features).
- **Key Empirical Observations:**
  1. **July Peak:** July is unequivocally the most expensive and volatile month ($\mu = \text{Rs. } 2,683/\text{Qtl}, \sigma = \text{Rs. } 1,465/\text{Qtl}$).
  2. **June & August Elevation:** June ($\mu = \text{Rs. } 1,840$) and August ($\mu = \text{Rs. } 1,490$) form the broader high-cost monsoon plateau.
  3. **Winter Depression:** February is the cheapest month ($\mu = \text{Rs. } 574/\text{Qtl}, \sigma = \text{Rs. } 146/\text{Qtl}$), exhibiting the lowest price volatility of the entire year.
- **Economic Interpretation:** Farmers in Maharashtra harvest the rabi crop in late winter, flooding local mandis and collapsing prices. Conversely, the kharif planting season coincides with monsoonal rainfall, creating a seasonal supply trough during July. Incorporating categorical calendar features (Month, Season) will provide strong explanatory power to the regression models.

---

### Figure 5: Feature Correlation Matrix Heatmap
- **File:** [05_tomato_correlation_heatmap.png](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/05_tomato_correlation_heatmap.png)
- **What is Plotted:** Triangular correlation heatmap of the dependent variable (`modal_price_rs_per_qtl`), daily trading bounds (`min_price`, `max_price`), physical arrivals (`arrival_quantity_tonnes`), and autoregressive lag/rolling features ($P_{t-1}, P_{t-3}, P_{t-7}, \text{Rolling Mean}_7$).
- **Analytical Utility:** Quantifies linear relationships, screens for high predictive feature candidates, and audits multicollinearity.
- **Key Empirical Observations:**
  1. **Very High Autoregressive Persistence:**
     - $P_{t-1}$ (Lag 1) correlation with $P_t$: **$r = +0.94$** (extremely strong predictor).
     - $P_{t-3}$ (Lag 3) correlation with $P_t$: **$r = +0.86$**.
     - $P_{t-7}$ (Lag 7) correlation with $P_t$: **$r = +0.76$**.
     - 7-day Rolling Mean correlation with $P_t$: **$r = +0.91$**.
  2. **Intra-Day Bounds Alignment:**
     - `min_price` ($r = +0.95$) and `max_price` ($r = +0.96$) move in near-perfect synchronization with `modal_price`.
  3. **Supply-Price Elasticity ($Q_t$ vs. $P_t$):**
     - Physical arrivals (`arrival_quantity_tonnes`) exhibits a negative correlation ($r = -0.18$) with modal price. As supply arrivals increase, prices soften, validating classical economic supply-demand theory.
- **Economic & Modeling Interpretation:** Food commodity prices exhibit massive short-term inertia: today's price is the strongest linear predictor of tomorrow's price. However, the non-linear shocks captured in Figures 1 and 4 indicate that linear autoregression alone will lag behind sharp inflections, necessitating non-linear models (Polynomial Regression, Decision Trees, Random Forests) and supply-side arrival features.
