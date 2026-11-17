"""
Data Preprocessing, Feature Engineering & Time-Series Partitioning
Prepares data/processed/model_ready_pune_tomato.csv
Enforces strict zero-leakage, chronological splitting, and Scikit-learn Pipeline compatibility.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Load datasets
DATA_DIR = Path("data/processed")
df_ws = pd.read_csv(DATA_DIR / "wholesale_prices.csv")
df_arr = pd.read_csv(DATA_DIR / "market_arrivals.csv")

# Filter for Pune Tomato series
pune_tom = df_ws[(df_ws["commodity_name"] == "Tomato") & (df_ws["market_name"] == "Pune")].copy()
pune_arr = df_arr[(df_arr["commodity_name"] == "Tomato") & (df_arr["market_name"] == "Pune")].copy()

pune_tom["observation_date"] = pd.to_datetime(pune_tom["observation_date"])
pune_arr["observation_date"] = pd.to_datetime(pune_arr["observation_date"])

# Merge price and physical arrival quantity
df = pd.merge(
    pune_tom,
    pune_arr[["observation_date", "arrival_quantity_tonnes"]],
    on="observation_date",
    how="left"
).sort_values("observation_date").drop_duplicates("observation_date").reset_index(drop=True)

# 1. Target Formulation: Next Active Trading Session Modal Price (t+1)
# Primary target: price at t+1
df["target_price_t_plus_1"] = df["modal_price_rs_per_qtl"].shift(-1)
# Secondary 7-session ahead target for comparison
df["target_price_t_plus_7"] = df["modal_price_rs_per_qtl"].shift(-7)

# 2. Autoregressive Lag Features (Strictly at t or earlier)
df["price_lag_1"] = df["modal_price_rs_per_qtl"] # Price today at t
df["price_lag_3"] = df["modal_price_rs_per_qtl"].shift(2) # Price at t-2
df["price_lag_7"] = df["modal_price_rs_per_qtl"].shift(6) # Price at t-6

# 3. Rolling Window Features (Calculated over past 7 active trading days up to t)
df["rolling_mean_7"] = df["modal_price_rs_per_qtl"].rolling(7).mean()
df["rolling_std_7"] = df["modal_price_rs_per_qtl"].rolling(7).std().fillna(0)
df["rolling_min_7"] = df["modal_price_rs_per_qtl"].rolling(7).min()
df["rolling_max_7"] = df["modal_price_rs_per_qtl"].rolling(7).max()

# 4. Momentum & Intra-day Market Spread Features
df["price_momentum_7"] = df["price_lag_1"] - df["price_lag_7"]
df["daily_spread"] = df["max_price_rs_per_qtl"] - df["min_price_rs_per_qtl"]

# 5. Supply-Side Exogenous Features
df["arrival_tonnes"] = df["arrival_quantity_tonnes"]
df["arrival_lag_1"] = df["arrival_quantity_tonnes"].shift(1)
df["rolling_arrival_mean_7"] = df["arrival_quantity_tonnes"].rolling(7).mean()

# 6. Calendar & Seasonal Features
df["month"] = df["observation_date"].dt.month
df["day_of_week"] = df["observation_date"].dt.dayofweek
df["day_of_year"] = df["observation_date"].dt.dayofyear
# Monsoon indicator (June, July, August when tomato supply is historically disrupted)
df["is_monsoon_season"] = df["month"].isin([6, 7, 8]).astype(int)

# Drop initial rows where lags/rolling windows are NaN (first 7 rows)
# and final row where target_price_t_plus_1 is NaN (last row)
model_df = df.dropna(subset=[
    "target_price_t_plus_1", "price_lag_1", "price_lag_3", "price_lag_7",
    "rolling_mean_7", "rolling_std_7", "rolling_min_7", "rolling_max_7",
    "arrival_tonnes", "arrival_lag_1", "rolling_arrival_mean_7"
]).copy().reset_index(drop=True)

# 7. Chronological Partitioning: 70% Train, 15% Validation, 15% Test
n_total = len(model_df)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val

model_df["split_tag"] = "Train"
model_df.iloc[n_train:n_train + n_val, model_df.columns.get_loc("split_tag")] = "Validation"
model_df.iloc[n_train + n_val:, model_df.columns.get_loc("split_tag")] = "Test"

# Save Model-Ready Dataset
out_csv = DATA_DIR / "model_ready_pune_tomato.csv"
model_df.to_csv(out_csv, index=False)

print(f"Model-Ready Dataset Created Successfully!")
print(f"File Path: {out_csv}")
print(f"Total Rows: {len(model_df)} (from 880 raw)")
print(f"Features: {list(model_df.columns)}")
print(f"\nChronological Split Breakdown:")
print(f"  Train Set      : {n_train} rows ({n_train/n_total*100:.1f}%) | {model_df[model_df['split_tag']=='Train']['observation_date'].min().date()} to {model_df[model_df['split_tag']=='Train']['observation_date'].max().date()}")
print(f"  Validation Set : {n_val} rows ({n_val/n_total*100:.1f}%) | {model_df[model_df['split_tag']=='Validation']['observation_date'].min().date()} to {model_df[model_df['split_tag']=='Validation']['observation_date'].max().date()}")
print(f"  Test Set       : {n_test} rows ({n_test/n_total*100:.1f}%) | {model_df[model_df['split_tag']=='Test']['observation_date'].min().date()} to {model_df[model_df['split_tag']=='Test']['observation_date'].max().date()}")
