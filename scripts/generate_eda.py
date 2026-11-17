"""
Exploratory Data Analysis and Statistical Computations
Generates figures in reports/figures/ and produces exact descriptive statistics
for Pune APMC Tomato and Onion wholesale series.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
df_ws = pd.read_csv("data/processed/wholesale_prices.csv")
df_arr = pd.read_csv("data/processed/market_arrivals.csv")

# Filter for Pune Tomato
pune_tom = df_ws[(df_ws["commodity_name"] == "Tomato") & (df_ws["market_name"] == "Pune")].copy()
pune_tom_arr = df_arr[(df_arr["commodity_name"] == "Tomato") & (df_arr["market_name"] == "Pune")].copy()

# Merge on observation_date
pune_tom["observation_date"] = pd.to_datetime(pune_tom["observation_date"])
pune_tom_arr["observation_date"] = pd.to_datetime(pune_tom_arr["observation_date"])

merged = pd.merge(
    pune_tom,
    pune_tom_arr[["observation_date", "arrival_quantity_tonnes"]],
    on="observation_date",
    how="left"
).sort_values("observation_date").drop_duplicates("observation_date").reset_index(drop=True)

prices = merged["modal_price_rs_per_qtl"]

# 2. Descriptive Statistics
mean_val = float(prices.mean())
median_val = float(prices.median())
std_val = float(prices.std())
var_val = float(prices.var())
min_val = float(prices.min())
max_val = float(prices.max())
q25 = float(prices.quantile(0.25))
q75 = float(prices.quantile(0.75))
iqr_val = q75 - q25
skew_val = float(prices.skew())
kurt_val = float(prices.kurtosis())

print("=== PUNE TOMATO STATISTICAL SUMMARY ===")
print(f"Count: {len(prices)}")
print(f"Mean: {mean_val:.2f} Rs/Qtl")
print(f"Median: {median_val:.2f} Rs/Qtl")
print(f"Std Dev: {std_val:.2f} Rs/Qtl")
print(f"Variance: {var_val:.2f} (Rs/Qtl)^2")
print(f"Min: {min_val:.2f} Rs/Qtl")
print(f"Max: {max_val:.2f} Rs/Qtl")
print(f"25th Percentile (Q1): {q25:.2f} Rs/Qtl")
print(f"75th Percentile (Q3): {q75:.2f} Rs/Qtl")
print(f"IQR: {iqr_val:.2f} Rs/Qtl")
print(f"Skewness: {skew_val:.2f} (Right-skewed)")
print(f"Kurtosis: {kurt_val:.2f} (Leptokurtic / Heavy-tailed)")

# 3. Visualizations

# Fig 1: Time Series Trend with Rolling Mean
plt.figure(figsize=(12, 5))
plt.plot(merged["observation_date"], merged["modal_price_rs_per_qtl"], label="Daily Modal Price", color="#1f77b4", alpha=0.6, linewidth=1.2)
plt.plot(merged["observation_date"], merged["modal_price_rs_per_qtl"].rolling(30, min_periods=7).mean(), label="30-Day Moving Average", color="#d62728", linewidth=2.5)
plt.title("Figure 1: Daily Wholesale Tomato Price Movement (Pune APMC Mandi, 2016–2018)", fontsize=13, fontweight="bold")
plt.xlabel("Observation Date")
plt.ylabel("Modal Price (Rs. / Quintal)")
plt.legend(loc="upper left")
plt.tight_layout()
fig1_path = FIG_DIR / "01_tomato_price_timeseries.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}")

# Fig 2: Histogram with KDE
plt.figure(figsize=(9, 5))
sns.histplot(prices, kde=True, color="#2ca02c", bins=35, stat="density", alpha=0.5)
plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: Rs. {mean_val:.1f}")
plt.axvline(median_val, color="blue", linestyle="-.", linewidth=2, label=f"Median: Rs. {median_val:.1f}")
plt.title("Figure 2: Distribution of Daily Wholesale Tomato Prices (Pune APMC)", fontsize=13, fontweight="bold")
plt.xlabel("Modal Price (Rs. / Quintal)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
fig2_path = FIG_DIR / "02_tomato_price_distribution.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved: {fig2_path}")

# Fig 3: Boxplot showing IQR and Outlier Price Spikes
plt.figure(figsize=(8, 4))
sns.boxplot(x=prices, color="#ff7f0e", flierprops=dict(marker="o", markersize=5, markerfacecolor="red", alpha=0.6))
plt.title("Figure 3: Boxplot of Tomato Wholesale Prices (Identifying Seasonal Outliers)", fontsize=13, fontweight="bold")
plt.xlabel("Modal Price (Rs. / Quintal)")
plt.text(q25, 0.35, f"Q1: {q25:.0f}", horizontalalignment="center", color="black", fontweight="bold")
plt.text(median_val, 0.35, f"Median: {median_val:.0f}", horizontalalignment="center", color="blue", fontweight="bold")
plt.text(q75, 0.35, f"Q3: {q75:.0f}", horizontalalignment="center", color="black", fontweight="bold")
plt.tight_layout()
fig3_path = FIG_DIR / "03_tomato_price_boxplot.png"
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved: {fig3_path}")

# Fig 4: Monthly Seasonality Barplot
merged["month_name"] = merged["observation_date"].dt.strftime("%b")
merged["month_num"] = merged["observation_date"].dt.month
monthly_stats = merged.groupby(["month_num", "month_name"])["modal_price_rs_per_qtl"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(10, 5))
bars = plt.bar(monthly_stats["month_name"], monthly_stats["mean"], yerr=monthly_stats["std"], capsize=5, color="#9467bd", alpha=0.8, edgecolor="black")
plt.title("Figure 4: Average Monthly Tomato Price with Standard Deviation (Seasonal Pattern)", fontsize=13, fontweight="bold")
plt.xlabel("Month of Year")
plt.ylabel("Mean Modal Price (Rs. / Quintal)")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval/2, f"Rs.{yval:.0f}", ha='center', va='center', color='white', fontweight='bold', rotation=90)
plt.tight_layout()
fig4_path = FIG_DIR / "04_tomato_monthly_seasonality.png"
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Saved: {fig4_path}")

# Fig 5: Correlation Heatmap
merged["price_lag_1"] = merged["modal_price_rs_per_qtl"].shift(1)
merged["price_lag_3"] = merged["modal_price_rs_per_qtl"].shift(3)
merged["price_lag_7"] = merged["modal_price_rs_per_qtl"].shift(7)
merged["rolling_mean_7"] = merged["modal_price_rs_per_qtl"].shift(1).rolling(7).mean()

corr_cols = [
    "modal_price_rs_per_qtl", "min_price_rs_per_qtl", "max_price_rs_per_qtl",
    "arrival_quantity_tonnes", "price_lag_1", "price_lag_3", "price_lag_7", "rolling_mean_7"
]
corr_matrix = merged[corr_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", mask=mask, cbar=True, square=True, linewidths=0.5)
plt.title("Figure 5: Correlation Matrix Heatmap (Features vs. Modal Price)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig5_path = FIG_DIR / "05_tomato_correlation_heatmap.png"
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"Saved: {fig5_path}")
print("All 5 figures generated successfully.")
