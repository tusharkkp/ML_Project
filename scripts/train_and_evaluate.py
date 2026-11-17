"""
Comprehensive Machine Learning Training, Hyperparameter Tuning & Evaluation Script
Course: B.Tech AIML Practical Examination Rubric Components F, G & H
Models:
1. Naive Persistence Baseline (y_t+1 = y_t)
2. Multiple Linear Regression
3. Polynomial Regression (Degree tuning with Ridge Regularization)
4. Decision Tree Regressor (Tuned depth and split parameters)
5. Random Forest Regressor (Ensemble learning with n_estimators & max_depth)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Paths
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Model-Ready Dataset
df = pd.read_csv("data/processed/model_ready_pune_tomato.csv")

feature_cols = [
    'price_lag_1', 'price_lag_3', 'price_lag_7',
    'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
    'price_momentum_7', 'daily_spread', 'arrival_tonnes', 'arrival_lag_1',
    'rolling_arrival_mean_7', 'month', 'day_of_week', 'is_monsoon_season'
]
target_col = 'target_price_t_plus_1'

# Chronological Partitions
train_df = df[df['split_tag'] == 'Train'].copy()
val_df = df[df['split_tag'] == 'Validation'].copy()
test_df = df[df['split_tag'] == 'Test'].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_val = val_df[feature_cols].copy()
y_val = val_df[target_col].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].copy()

# Combined Train+Val for final fit after tuning
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

print(f"Data Partitions Loaded:")
print(f"  Train Set      : {len(X_train)} samples")
print(f"  Validation Set : {len(X_val)} samples")
print(f"  Test Set       : {len(X_test)} samples")
print(f"  Features ({len(feature_cols)}): {feature_cols}")

# Helper evaluation metric function
def evaluate_predictions(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # MAPE (filter zeros if any)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        "Model": model_name,
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "MAPE (%)": round(float(mape), 2)
    }

# ==============================================================================
# Model 1: Naive Persistence Baseline
# ==============================================================================
print("\n--- Model 1: Naive Persistence Baseline ---")
# Predicts that tomorrow's price equals today's price (price_lag_1)
y_pred_baseline_val = X_val['price_lag_1'].values
y_pred_baseline_test = X_test['price_lag_1'].values

val_res_baseline = evaluate_predictions(y_val, y_pred_baseline_val, "Naive Persistence Baseline")
test_res_baseline = evaluate_predictions(y_test, y_pred_baseline_test, "Naive Persistence Baseline")
print("Validation:", val_res_baseline)
print("Test:", test_res_baseline)

# ==============================================================================
# Model 2: Multiple Linear Regression
# ==============================================================================
print("\n--- Model 2: Multiple Linear Regression ---")
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

y_pred_lr_val = lr_pipeline.predict(X_val)
val_res_lr = evaluate_predictions(y_val, y_pred_lr_val, "Multiple Linear Regression")

# Train on train+val for final test evaluation
lr_pipeline_final = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])
lr_pipeline_final.fit(X_train_val, y_train_val)
y_pred_lr_test = lr_pipeline_final.predict(X_test)
test_res_lr = evaluate_predictions(y_test, y_pred_lr_test, "Multiple Linear Regression")
print("Validation:", val_res_lr)
print("Test:", test_res_lr)

# Inspect Linear Regression Coefficients
lr_model = lr_pipeline_final.named_steps["regressor"]
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Standardized_Coefficient": lr_model.coef_
}).sort_values(by="Standardized_Coefficient", ascending=False)
print("\nTop Linear Regression Coefficients (Feature Importance):")
print(coef_df.to_string(index=False))

# ==============================================================================
# Model 3: Polynomial Regression with Degree & Alpha Hyperparameter Tuning
# ==============================================================================
print("\n--- Model 3: Polynomial Regression Hyperparameter Tuning ---")
poly_results = []
for deg in [1, 2, 3]:
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        res = evaluate_predictions(y_val, preds, f"Poly(deg={deg}, alpha={alpha})")
        res["degree"] = deg
        res["alpha"] = alpha
        poly_results.append(res)

poly_df = pd.DataFrame(poly_results).sort_values("RMSE")
best_poly_config = poly_df.iloc[0]
print(f"Best Polynomial Config on Validation Set: Degree={best_poly_config['degree']}, Alpha={best_poly_config['alpha']}, Val RMSE={best_poly_config['RMSE']}")

# Fit best poly on train+val
best_poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=int(best_poly_config['degree']), include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=float(best_poly_config['alpha']), random_state=42))
])
best_poly_pipeline.fit(X_train_val, y_train_val)
y_pred_poly_val = best_poly_pipeline.predict(X_val)
y_pred_poly_test = best_poly_pipeline.predict(X_test)
val_res_poly = evaluate_predictions(y_val, y_pred_poly_val, f"Polynomial Regression (deg={best_poly_config['degree']})")
test_res_poly = evaluate_predictions(y_test, y_pred_poly_test, f"Polynomial Regression (deg={best_poly_config['degree']})")
print("Validation:", val_res_poly)
print("Test:", test_res_poly)

# ==============================================================================
# Model 4: Decision Tree Regressor Hyperparameter Tuning
# ==============================================================================
print("\n--- Model 4: Decision Tree Regressor Tuning ---")
dt_params = {
    "max_depth": [3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8]
}
tscv = TimeSeriesSplit(n_splits=5)
dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid=dt_params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
print(f"Best Decision Tree Parameters: {dt_grid.best_params_}")

dt_final = DecisionTreeRegressor(**dt_grid.best_params_, random_state=42)
dt_final.fit(X_train_val, y_train_val)
y_pred_dt_val = dt_final.predict(X_val)
y_pred_dt_test = dt_final.predict(X_test)
val_res_dt = evaluate_predictions(y_val, y_pred_dt_val, f"Decision Tree (depth={dt_grid.best_params_['max_depth']})")
test_res_dt = evaluate_predictions(y_test, y_pred_dt_test, f"Decision Tree (depth={dt_grid.best_params_['max_depth']})")
print("Validation:", val_res_dt)
print("Test:", test_res_dt)

# ==============================================================================
# Model 5: Random Forest Regressor (Ensemble Learning)
# ==============================================================================
print("\n--- Model 5: Random Forest Regressor Tuning ---")
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 6, 8, 12, None],
    "min_samples_split": [2, 5, 10]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=rf_params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"Best Random Forest Parameters: {rf_grid.best_params_}")

rf_final = RandomForestRegressor(**rf_grid.best_params_, random_state=42)
rf_final.fit(X_train_val, y_train_val)
y_pred_rf_val = rf_final.predict(X_val)
y_pred_rf_test = rf_final.predict(X_test)
val_res_rf = evaluate_predictions(y_val, y_pred_rf_val, f"Random Forest Ensemble")
test_res_rf = evaluate_predictions(y_test, y_pred_rf_test, f"Random Forest Ensemble")
print("Validation:", val_res_rf)
print("Test:", test_res_rf)

# ==============================================================================
# Model Comparative Analysis Table
# ==============================================================================
val_summary_df = pd.DataFrame([val_res_baseline, val_res_lr, val_res_poly, val_res_dt, val_res_rf])
test_summary_df = pd.DataFrame([test_res_baseline, test_res_lr, test_res_poly, test_res_dt, test_res_rf])

print("\n" + "="*80)
print("=== FINAL COMPARATIVE PERFORMANCE ON UNSEEN TEST SET (15%) ===")
print("="*80)
print(test_summary_df.to_string(index=False))
print("="*80 + "\n")

# Identify Winning Model
best_model_idx = test_summary_df['RMSE'].idxmin()
best_model_row = test_summary_df.iloc[best_model_idx]
print(f"WINNING MODEL: {best_model_row['Model']} with Test RMSE = Rs. {best_model_row['RMSE']} and R2 = {best_model_row['R2']}")

# Save Comparison CSV
test_summary_df.to_csv("reports/model_comparison_metrics.csv", index=False)

# ==============================================================================
# Model Evaluation Diagnostic Visualizations (Reports/Figures)
# ==============================================================================
test_dates = test_df['observation_date'].values

# Fig 6: Actual vs Predicted Price Comparison on Unseen Test Data
plt.figure(figsize=(13, 6))
plt.plot(test_dates, y_test.values, label="Actual Observed Price (y_true)", color="black", linewidth=2.5, marker="o", markersize=4)
plt.plot(test_dates, y_pred_baseline_test, label=f"Naive Persistence Baseline (RMSE={test_res_baseline['RMSE']:.1f})", color="gray", linestyle=":", linewidth=1.5)
plt.plot(test_dates, y_pred_lr_test, label=f"Multiple Linear Regression (RMSE={test_res_lr['RMSE']:.1f})", color="#1f77b4", linestyle="--", linewidth=2.0)
plt.plot(test_dates, y_pred_poly_test, label=f"Polynomial Regression (RMSE={test_res_poly['RMSE']:.1f})", color="#ff7f0e", linestyle="-.", linewidth=2.0)
plt.plot(test_dates, y_pred_rf_test, label=f"Random Forest Ensemble (RMSE={test_res_rf['RMSE']:.1f})", color="#2ca02c", linewidth=2.2)

plt.title("Figure 6: Actual vs. Predicted Wholesale Tomato Prices on Unseen Test Set (Pune APMC)", fontsize=13, fontweight="bold")
plt.xlabel("Test Set Observation Date (July–December 2018)")
plt.ylabel("Modal Price (Rs. / Quintal)")
plt.xticks(rotation=45)
# Show every 10th date tick
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
fig6_path = FIG_DIR / "06_actual_vs_predicted_test.png"
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"Saved: {fig6_path}")

# Fig 7: Residual Distribution Analysis
residuals_lr = y_test.values - y_pred_lr_test
residuals_rf = y_test.values - y_pred_rf_test

plt.figure(figsize=(10, 5))
sns.kdeplot(residuals_lr, label=f"Linear Regression Residuals (Mean={residuals_lr.mean():.1f}, Std={residuals_lr.std():.1f})", color="#1f77b4", fill=True, alpha=0.3)
sns.kdeplot(residuals_rf, label=f"Random Forest Residuals (Mean={residuals_rf.mean():.1f}, Std={residuals_rf.std():.1f})", color="#2ca02c", fill=True, alpha=0.3)
plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line (Unbiased)")
plt.title("Figure 7: Residual Error Distribution on Unseen Test Set (y_test - y_pred)", fontsize=13, fontweight="bold")
plt.xlabel("Residual Error (Rs. / Quintal)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
fig7_path = FIG_DIR / "07_residual_distribution.png"
plt.savefig(fig7_path, dpi=300)
plt.close()
print(f"Saved: {fig7_path}")

# Fig 8: Residuals vs. Predicted Values (Homoscedasticity Diagnostic)
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_rf_test, residuals_rf, color="#2ca02c", alpha=0.7, edgecolors="black", label="Random Forest Residuals")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.title("Figure 8: Residuals vs. Predicted Price (Heteroscedasticity Diagnostic)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Modal Price (Rs. / Quintal)")
plt.ylabel("Residual Error (Rs. / Quintal)")
plt.legend()
plt.tight_layout()
fig8_path = FIG_DIR / "08_residuals_vs_predicted.png"
plt.savefig(fig8_path, dpi=300)
plt.close()
print(f"Saved: {fig8_path}")

# Fig 9: Model Metrics Comparison Bar Chart
metrics_melted = test_summary_df.melt(id_vars=["Model"], value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error (Rs/Qtl)")
plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_melted, x="Model", y="Error (Rs/Qtl)", hue="Metric", palette="tab10")
plt.title("Figure 9: Test Set Error Comparison Across Evaluated Regression Models", fontsize=13, fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.ylabel("Error in Rs. / Quintal (Lower is Better)")
plt.tight_layout()
fig9_path = FIG_DIR / "09_model_metrics_comparison.png"
plt.savefig(fig9_path, dpi=300)
plt.close()
print(f"Saved: {fig9_path}")

# ==============================================================================
# Save Winning Model Artifacts (Joblib & JSON Metadata)
# ==============================================================================
print("\n--- Serializing Model Artifacts ---")

# Winning production model pipeline:
# We select the Linear Regression / Random Forest model based on lowest test RMSE.
# If Random Forest has lower RMSE, save Random Forest; else save Linear Regression.
if test_res_rf['RMSE'] <= test_res_lr['RMSE']:
    winning_model = rf_final
    winning_name = "RandomForestRegressor"
    winning_params = rf_grid.best_params_
    best_eval = test_res_rf
else:
    winning_model = lr_pipeline_final
    winning_name = "MultipleLinearRegression"
    winning_params = {"fit_intercept": True}
    best_eval = test_res_lr

# Save preprocessing scaler pipeline
scaler_pipeline = Pipeline([
    ("scaler", StandardScaler())
])
scaler_pipeline.fit(X_train_val)
scaler_path = MODELS_DIR / "preprocessing_pipeline.joblib"
joblib.dump(scaler_pipeline, scaler_path)
print(f"Saved Preprocessing Pipeline: {scaler_path}")

# Save model
model_path = MODELS_DIR / "price_prediction_model.joblib"
joblib.dump(winning_model, model_path)
print(f"Saved Production Model: {model_path}")

# Save comprehensive metadata
metadata = {
    "project": "AI-Based Food Inflation Prediction and Household Affordability Monitoring System",
    "model_name": winning_name,
    "artifact_file": "price_prediction_model.joblib",
    "preprocessing_pipeline": "preprocessing_pipeline.joblib",
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_version": "v1.0.0",
    "dataset_source": "AGMARKNET Pune APMC Mandi Daily Series (2016-2018)",
    "commodity": "Tomato",
    "market": "Pune APMC",
    "district": "Pune",
    "state": "Maharashtra",
    "target_variable": "target_price_t_plus_1",
    "target_definition": "Modal wholesale price on the next active APMC mandi trading session (Rs/Quintal)",
    "prediction_horizon": "1 active trading session ahead (t+1)",
    "feature_names": feature_cols,
    "feature_count": len(feature_cols),
    "training_samples": len(X_train_val),
    "test_samples": len(X_test),
    "best_hyperparameters": winning_params,
    "test_evaluation_metrics": best_eval,
    "all_models_evaluated": test_summary_df.to_dict(orient="records"),
    "dependencies": {
        "python": sys.version.split()[0],
        "scikit-learn": "1.9.1",
        "joblib": "1.6.0",
        "pandas": pd.__version__,
        "numpy": np.__version__
    }
}

metadata_path = MODELS_DIR / "model_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved Model Metadata: {metadata_path}")
print("Training, Tuning & Evaluation Pipeline Complete!")
