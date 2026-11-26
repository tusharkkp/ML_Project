"""
Unified Multi-Commodity Price Forecasting Model Trainer
AI-Based Food Inflation Prediction and Household Affordability Monitoring System

Trains a single global regression model across multiple food commodity categories
(Vegetables, Pulses, Sugars & Sweets) using categorical embeddings (One-Hot Encoding)
and standard scaling on lag and rolling features.

Outputs:
  - models/unified_food_price_model.joblib
  - models/unified_model_metadata.json
  - reports/figures/unified_multi_commodity_evaluation.png

Usage:
  py -3.12 scripts/train_unified_model.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_unified_commodities.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    print("=" * 80)
    print("AI-BASED FOOD INFLATION SYSTEM: UNIFIED MULTI-COMMODITY MODEL TRAINING")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Unified dataset not found at {DATA_PATH}. Run build_features.py --unified first.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded unified dataset: {len(df)} records across {df['commodity_name'].nunique()} commodities.")
    print("Commodity breakdown:")
    for comm, count in df['commodity_name'].value_counts().items():
        cat = df[df['commodity_name'] == comm]['category'].iloc[0]
        print(f"  - {comm:<12} ({cat}): {count} observations")

    # Define feature groups
    categorical_cols = ['commodity_name', 'category']
    numeric_cols = [
        'price_lag_1', 'price_lag_3', 'price_lag_7',
        'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
        'price_momentum_7', 'daily_spread', 'arrival_tonnes', 'arrival_lag_1',
        'rolling_arrival_mean_7', 'month', 'day_of_week', 'is_monsoon_season', 'is_perishable'
    ]
    all_feature_cols = categorical_cols + numeric_cols
    target_col = 'target_price_t_plus_1'

    # Filter any row where target or features are NaN
    df_clean = df.dropna(subset=[target_col] + numeric_cols).copy()

    # Train / Val / Test partitions
    train_df = df_clean[df_clean['split_tag'] == 'Train'].copy()
    val_df = df_clean[df_clean['split_tag'] == 'Validation'].copy()
    test_df = df_clean[df_clean['split_tag'] == 'Test'].copy()

    print(f"\nChronological Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    X_train, y_train = train_df[all_feature_cols], train_df[target_col]
    X_val, y_val = val_df[all_feature_cols], val_df[target_col]
    X_test, y_test = test_df[all_feature_cols], test_df[target_col]

    # Preprocessing transformer: OneHotEncode categories + StandardScale numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )

    candidate_models = {
        'Multiple Linear Regression': LinearRegression(),
        'Ridge Regression (L2)': Ridge(alpha=1.0),
        'Lasso Regression (L1)': Lasso(alpha=0.5, max_iter=5000),
        'ElasticNet Regression': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Support Vector Regressor': SVR(C=100, epsilon=10)
    }

    print("\n" + "=" * 80)
    print("EVALUATING CANDIDATE ALGORITHMS ON UNSEEN MULTI-COMMODITY TEST PARTITION")
    print("=" * 80)

    results = []
    trained_pipelines = {}

    for name, model in candidate_models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        })

    results_df = pd.DataFrame(results).sort_values('RMSE')

    print(f"\n{'Model':<30} | {'MAE (Rs/Qtl)':<12} | {'RMSE (Rs/Qtl)':<13} | {'R2 Score':<9} | {'MAPE (%)':<8}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<30} | Rs.{row['MAE']:<9.2f} | Rs.{row['RMSE']:<10.2f} | {row['R2']:<9.4f} | {row['MAPE']:<6.2f}%")
    print("=" * 80)

    # Select champion model (Lowest RMSE on test set)
    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\nCHAMPION MODEL SELECTED: {best_model_name}")
    print(f"Overall Test MAE : Rs. {best_row['MAE']:.2f}/Quintal (Rs. {best_row['MAE']/100:.2f}/Kg)")
    print(f"Overall Test RMSE: Rs. {best_row['RMSE']:.2f}/Quintal (Rs. {best_row['RMSE']/100:.2f}/Kg)")
    print(f"Overall Test R2  : {best_row['R2']:.4f} ({best_row['R2']*100:.2f}% Variance Explained)")

    # Per-commodity breakdown for the champion model
    print("\n" + "=" * 80)
    print("PER-COMMODITY GENERALIZATION BREAKDOWN (CHAMPION MODEL)")
    print("=" * 80)

    best_preds = best_pipeline.predict(X_test)
    test_df['predicted_price'] = best_preds

    commodity_breakdown = {}
    for comm in sorted(test_df['commodity_name'].unique()):
        comm_mask = (test_df['commodity_name'] == comm).values
        y_c = y_test.values[comm_mask]
        p_c = best_preds[comm_mask]

        c_mae = mean_absolute_error(y_c, p_c)
        c_rmse = np.sqrt(mean_squared_error(y_c, p_c))
        c_r2 = r2_score(y_c, p_c)
        c_mape = np.mean(np.abs((y_c - p_c) / y_c)) * 100
        mean_actual = np.mean(y_c)
        cat = test_df[test_df['commodity_name'] == comm]['category'].iloc[0]

        commodity_breakdown[comm] = {
            'category': cat,
            'mean_price_rs_per_qtl': round(float(mean_actual), 2),
            'test_mae_rs_per_qtl': round(float(c_mae), 2),
            'test_mae_rs_per_kg': round(float(c_mae / 100.0), 2),
            'test_rmse_rs_per_qtl': round(float(c_rmse), 2),
            'test_rmse_rs_per_kg': round(float(c_rmse / 100.0), 2),
            'test_r2': round(float(c_r2), 4),
            'test_mape_pct': round(float(c_mape), 2),
            'sample_count': int(len(y_c))
        }

        print(f"{comm:<12} ({cat:<15}) | Mean: Rs.{mean_actual:>7.0f} | MAE: Rs.{c_mae:>6.2f} (Rs.{c_mae/100:.2f}/kg) | RMSE: Rs.{c_rmse:>6.2f} | R2: {c_r2:>6.4f} | MAPE: {c_mape:>5.2f}%")
    print("=" * 80)

    # Save champion model pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "unified_food_price_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nSaved trained champion pipeline to: {model_save_path}")

    # Save comprehensive metadata
    meta = {
        'model_architecture': 'Unified Cross-Commodity Regression Pipeline',
        'champion_algorithm': best_model_name,
        'dataset_version': 'v2.0.0-unified',
        'training_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_training_records': int(len(X_train)),
        'total_validation_records': int(len(X_val)),
        'total_test_records': int(len(X_test)),
        'supported_commodities': sorted(list(df['commodity_name'].unique())),
        'supported_categories': sorted(list(df['category'].unique())),
        'categorical_features': categorical_cols,
        'numeric_features': numeric_cols,
        'all_features': all_feature_cols,
        'prediction_horizon': 't+1 trading session',
        'price_unit': 'Rs/Quintal (Wholesale) & Rs/Kg (Retail equivalent)',
        'overall_test_metrics': {
            'mae_rs_per_qtl': round(float(best_row['MAE']), 2),
            'rmse_rs_per_qtl': round(float(best_row['RMSE']), 2),
            'r2_score': round(float(best_row['R2']), 4),
            'mape_pct': round(float(best_row['MAPE']), 2)
        },
        'per_commodity_metrics': commodity_breakdown,
        'all_models_benchmarked': [
            {
                'Model': r['Model'],
                'MAE': round(float(r['MAE']), 2),
                'RMSE': round(float(r['RMSE']), 2),
                'R2': round(float(r['R2']), 4),
                'MAPE': round(float(r['MAPE']), 2)
            } for r in results
        ]
    }

    meta_save_path = MODELS_DIR / "unified_model_metadata.json"
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model metadata to: {meta_save_path}")

    # Generate performance visualization plot
    REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = REPORTS_FIG_DIR / "unified_multi_commodity_evaluation.png"

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    commodities = sorted(test_df['commodity_name'].unique())
    colors = {'Tomato': '#d9534f', 'Onion': '#f0ad4e', 'Moong Dal': '#5cb85c', 'Sugar': '#0275d8'}

    for idx, comm in enumerate(commodities):
        ax = axes[idx // 2, idx % 2]
        sub = test_df[test_df['commodity_name'] == comm]
        y_act = sub[target_col].values
        y_pred = sub['predicted_price'].values

        c_r2 = commodity_breakdown[comm]['test_r2']
        c_mae = commodity_breakdown[comm]['test_mae_rs_per_qtl']
        c_mape = commodity_breakdown[comm]['test_mape_pct']

        ax.scatter(y_act, y_pred, alpha=0.6, color=colors.get(comm, '#333333'), edgecolors='none', s=35)
        # 45-degree identity line
        min_v = min(y_act.min(), y_pred.min())
        max_v = max(y_act.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='Ideal 1:1 Fit')

        ax.set_title(f"{comm} (R² = {c_r2:.4f}, MAE = ₹{c_mae:.1f}, MAPE = {c_mape:.1f}%)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual Price (₹/Quintal)", fontsize=10)
        ax.set_ylabel("Predicted Price (₹/Quintal)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9)

    plt.suptitle(f"Unified Cross-Commodity Model Generalization: {best_model_name}\nOverall Test R² = {best_row['R2']:.4f} | MAE = ₹{best_row['MAE']:.2f}/Qtl", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved evaluation figure to: {fig_path}")
    print("\nUnified Model Training & Evaluation Finished Successfully!")


if __name__ == "__main__":
    main()
