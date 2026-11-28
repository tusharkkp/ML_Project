"""
Leak-Free Specialist Training, TimeSeriesSplit CV & Validation Selection (Pune Tomato).

Fixes earlier test-set leakage by:
1. Tuning hyperparameters inside Train via 5-fold TimeSeriesSplit CV.
2. Ranking and selecting the winning model strictly on Validation RMSE.
3. Refitting the single selected winner on combined Train + Validation.
4. Evaluating ONCE on the untouched Test partition with Persistence Baseline comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import SPECIALIST_NUMERIC_FEATURES, TARGET_REG_COL
from src.metrics import compute_regression_metrics

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_pune_tomato.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run scripts/build_features.py --all first.")

    df = pd.read_csv(DATA_PATH)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    train_df = df[df["split_tag"] == "Train"].copy()
    val_df = df[df["split_tag"] == "Validation"].copy()
    test_df = df[df["split_tag"] == "Test"].copy()
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    X_train, y_train = train_df[SPECIALIST_NUMERIC_FEATURES], train_df[TARGET_REG_COL].values
    X_val, y_val = val_df[SPECIALIST_NUMERIC_FEATURES], val_df[TARGET_REG_COL].values
    X_tv, y_tv = train_val_df[SPECIALIST_NUMERIC_FEATURES], train_val_df[TARGET_REG_COL].values
    X_test, y_test = test_df[SPECIALIST_NUMERIC_FEATURES], test_df[TARGET_REG_COL].values

    val_base = compute_regression_metrics(y_val, val_df["modal_price_rs_per_qtl"].values)
    test_base = compute_regression_metrics(y_test, test_df["modal_price_rs_per_qtl"].values)

    tscv = TimeSeriesSplit(n_splits=5)
    candidates = [
        ("Lasso Regression (L1)", Pipeline([("scaler", StandardScaler()), ("reg", Lasso(max_iter=10000, random_state=42))]), {"reg__alpha": [0.1, 1.0, 5.0, 10.0, 50.0]}),
        ("Multiple Linear Regression", Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]), {}),
        ("Ridge Regression (L2)", Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42))]), {"reg__alpha": [0.1, 1.0, 10.0, 100.0]}),
        ("Polynomial Regression", Pipeline([("scaler", StandardScaler()), ("poly", PolynomialFeatures(include_bias=False)), ("reg", Ridge(random_state=42))]), {"poly__degree": [1, 2], "reg__alpha": [1.0, 10.0, 100.0]}),
        ("Random Forest Ensemble", Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor(random_state=42, n_jobs=1))]), {"reg__n_estimators": [50, 100], "reg__max_depth": [4, 6, 8], "reg__min_samples_leaf": [2, 5]}),
        ("Decision Tree Regressor", Pipeline([("scaler", StandardScaler()), ("reg", DecisionTreeRegressor(random_state=42))]), {"reg__max_depth": [3, 5, 7], "reg__min_samples_split": [5, 10], "reg__min_samples_leaf": [2, 4]}),
    ]

    rows = []
    fitted = {}
    for name, pipe, grid in candidates:
        if grid:
            gs = GridSearchCV(pipe, grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1)
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            cv_rmse = round(float(-gs.best_score_), 2)
            params = gs.best_params_
        else:
            best_pipe = pipe.fit(X_train, y_train)
            cv_rmse = round(float(np.sqrt(mean_squared_error(y_train, best_pipe.predict(X_train)))), 2)
            params = {}

        vp = best_pipe.predict(X_val)
        vm = compute_regression_metrics(y_val, vp)
        fitted[name] = (best_pipe, params)
        rows.append(
            {
                "Model": name,
                "Val_MAE": vm["MAE"],
                "Val_MSE": vm["MSE"],
                "Val_RMSE": vm["RMSE"],
                "Val_R2": vm["R2"],
                "Val_MAPE": vm["MAPE"],
                "CV_Train_RMSE": cv_rmse,
                "Beats_Val_Baseline": bool(vm["RMSE"] < val_base["RMSE"]),
                "Best_Params": json.dumps(params),
            }
        )

    board = pd.DataFrame(rows).sort_values("Val_RMSE").reset_index(drop=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    board_path = REPORTS_DIR / "specialist_tomato_validation_leaderboard.csv"
    board.to_csv(board_path, index=False)

    winner_name = str(board.iloc[0]["Model"])
    winner_pipe, winner_params = fitted[winner_name]
    print("VALIDATION LEADERBOARD (PUNE TOMATO):")
    print(board[["Model", "Val_MAE", "Val_RMSE", "Val_R2", "Beats_Val_Baseline"]].to_string(index=False))
    print(f"\nSelected Strictly on Validation: {winner_name} ({winner_params})")

    # Final refit on Train+Val -> Single evaluation on untouched Test
    from sklearn.base import clone
    final_pipe = clone(winner_pipe).fit(X_tv, y_tv)
    test_preds = final_pipe.predict(X_test)
    tm = compute_regression_metrics(y_test, test_preds)
    print(
        f"Untouched Test Evaluation: MAE=Rs.{tm['MAE']:.2f} | RMSE=Rs.{tm['RMSE']:.2f} "
        f"(Baseline=Rs.{test_base['RMSE']:.2f}) | R2={tm['R2']:.4f} | MAPE={tm['MAPE']:.2f}%"
    )


if __name__ == "__main__":
    main()
