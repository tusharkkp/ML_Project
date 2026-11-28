"""
Leak-Free Unified Multi-Commodity Price Forecasting Model Trainer (v3.1.0-parity).

Strict Experimental Protocol:
1. Load leak-free chronological dataset (`model_ready_unified_commodities_clean.csv`).
2. Step 1 (Training + Date-Grouped Purged CV): Tune candidate hyperparameters using
   `DatePurgedWalkForwardCV` strictly inside `Train` (all commodities on same date grouped
   together; train targets reaching validation folds purged).
3. Step 2 (Validation Selection): Select Champion strictly on **Val_Macro_RMSE** (unweighted
   macro-average RMSE across commodities, preventing high-price Moong Dal/Sugar scale bias),
   recording `Val_RMSE` (pooled) as secondary metric.
4. Step 3 (Final Refit): Refit Validation-selected Champion once on combined `Train + Validation`.
5. Step 4 (Single Untouched Test Evaluation): Evaluate ONCE for the selected Champion + Persistence
   Baseline on untouched Test (never using Test for selection).
6. Independent Leave-One-Commodity-Out (LOCO): For each held-out commodity, re-run full
   candidate tuning & Validation selection EXCLUSIVELY on the remaining 3 commodities.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    DATASET_VERSION,
    FEATURE_SCHEMA_VERSION,
    TARGET_REG_COL,
    UNIFIED_ALL_FEATURES,
    UNIFIED_CATEGORICAL_FEATURES,
    UNIFIED_NUMERIC_FEATURES,
    DatePurgedWalkForwardCV,
)
from src.metrics import compute_regression_metrics

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_unified_commodities_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIG_DIR = REPORTS_DIR / "figures"

PRIMARY_SELECTION_METRIC = "Val_Macro_RMSE"
SECONDARY_SELECTION_METRIC = "Val_RMSE"


def make_preprocessor() -> ColumnTransformer:
    """Create column transformer matching canonical unified schema."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                UNIFIED_CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), UNIFIED_NUMERIC_FEATURES),
        ]
    )


def get_candidate_models() -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Unified candidate models and hyperparameter grids."""
    return [
        ("Multiple Linear Regression", LinearRegression(), {}),
        ("Ridge Regression (L2)", Ridge(random_state=42), {"regressor__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}),
        ("Lasso Regression (L1)", Lasso(max_iter=10000, random_state=42), {"regressor__alpha": [0.1, 0.5, 1.0, 5.0, 10.0]}),
        (
            "ElasticNet Regression",
            ElasticNet(max_iter=10000, random_state=42),
            {"regressor__alpha": [0.01, 0.1, 1.0], "regressor__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        (
            "Decision Tree Regressor",
            DecisionTreeRegressor(random_state=42),
            {"regressor__max_depth": [4, 6, 8, 10], "regressor__min_samples_leaf": [2, 5, 10]},
        ),
        (
            "Random Forest Regressor",
            RandomForestRegressor(random_state=42, n_jobs=1),
            {
                "regressor__n_estimators": [50, 100],
                "regressor__max_depth": [6, 10, 12],
                "regressor__min_samples_leaf": [2, 5],
            },
        ),
        (
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(random_state=42),
            {
                "regressor__n_estimators": [50, 100],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [3, 5],
            },
        ),
        (
            "Support Vector Regressor",
            SVR(kernel="rbf"),
            {"regressor__C": [10.0, 100.0, 500.0], "regressor__epsilon": [5.0, 10.0]},
        ),
    ]


def _prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[UNIFIED_ALL_FEATURES].copy()
    for col in UNIFIED_CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).astype(object)
    return X


def compute_group_macro_metrics(
    df_eval: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute unweighted macro-averages across commodities to prevent high-price bias."""
    comms = sorted(df_eval["commodity_name"].unique())
    maes, mses, rmses, r2s, mapes = [], [], [], [], []
    b_maes, b_rmses = [], []
    y_base = df_eval["modal_price_rs_per_qtl"].to_numpy(dtype=float)

    for comm in comms:
        mask = (df_eval["commodity_name"] == comm).to_numpy()
        m = compute_regression_metrics(y_true[mask], y_pred[mask])
        b = compute_regression_metrics(y_true[mask], y_base[mask])
        maes.append(m["MAE"])
        mses.append(m["MSE"])
        rmses.append(m["RMSE"])
        r2s.append(m["R2"])
        mapes.append(m["MAPE"])
        b_maes.append(b["MAE"])
        b_rmses.append(b["RMSE"])

    macro_rmse = float(np.mean(rmses))
    base_macro_rmse = float(np.mean(b_rmses))
    return {
        "macro_avg_mae": round(float(np.mean(maes)), 2),
        "macro_avg_mse": round(float(np.mean(mses)), 2),
        "macro_avg_rmse": round(macro_rmse, 2),
        "macro_avg_r2": round(float(np.mean(r2s)), 4),
        "macro_avg_mape_pct": round(float(np.mean(mapes)), 2),
        "baseline_macro_avg_mae": round(float(np.mean(b_maes)), 2),
        "baseline_macro_avg_rmse": round(base_macro_rmse, 2),
        "macro_rmse_improvement": round(base_macro_rmse - macro_rmse, 2),
    }


def select_candidate_with_purged_cv(
    train_df: pd.DataFrame, val_df: pd.DataFrame, verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Tune all candidates using DatePurgedWalkForwardCV inside Train, score on Validation."""
    X_train = _prepare_feature_frame(train_df)
    y_train = train_df[TARGET_REG_COL].to_numpy(dtype=float)
    X_val = _prepare_feature_frame(val_df)
    y_val = val_df[TARGET_REG_COL].to_numpy(dtype=float)

    val_baseline = compute_regression_metrics(y_val, val_df["modal_price_rs_per_qtl"].to_numpy(dtype=float))
    val_base_macro = compute_group_macro_metrics(val_df, y_val, val_df["modal_price_rs_per_qtl"].to_numpy(dtype=float))

    purged_cv = DatePurgedWalkForwardCV(
        observation_dates=train_df["observation_date"],
        target_dates=train_df["target_date_t_plus_1"],
        n_splits=5,
    )

    val_records: List[Dict[str, Any]] = []
    fitted_estimators: Dict[str, Any] = {}
    best_params_map: Dict[str, Dict[str, Any]] = {}

    for name, base_est, grid in get_candidate_models():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", base_est)])
        if grid:
            gs = GridSearchCV(
                pipe,
                param_grid=grid,
                cv=purged_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            cv_rmse = round(float(-gs.best_score_), 2)
            raw_params = {k.replace("regressor__", ""): v for k, v in gs.best_params_.items()}
        else:
            best_pipe = pipe.fit(X_train, y_train)
            raw_params = {}
            tr_preds = best_pipe.predict(X_train)
            cv_rmse = round(float(np.sqrt(mean_squared_error(y_train, tr_preds))), 2)

        val_preds = best_pipe.predict(X_val)
        vm = compute_regression_metrics(y_val, val_preds)
        vmacro = compute_group_macro_metrics(val_df, y_val, val_preds)

        fitted_estimators[name] = clone(best_pipe.named_steps["regressor"])
        best_params_map[name] = raw_params

        val_records.append(
            {
                "Model": name,
                "Val_Macro_RMSE": vmacro["macro_avg_rmse"],
                "Val_Macro_MAE": vmacro["macro_avg_mae"],
                "Val_Macro_R2": vmacro["macro_avg_r2"],
                "Val_MAE": vm["MAE"],
                "Val_MSE": vm["MSE"],
                "Val_RMSE": vm["RMSE"],
                "Val_R2": vm["R2"],
                "Val_MAPE": vm["MAPE"],
                "CV_Train_RMSE": cv_rmse,
                "Beats_Val_Macro_Baseline": bool(vmacro["macro_avg_rmse"] < val_base_macro["macro_avg_rmse"]),
                "Beats_Val_Pooled_Baseline": bool(vm["RMSE"] < val_baseline["RMSE"]),
                "Best_Params": json.dumps(raw_params),
            }
        )
        if verbose:
            print(
                f"  {name:<28} | Val_Macro_RMSE=Rs.{vmacro['macro_avg_rmse']:<6.2f} | "
                f"Val_RMSE=Rs.{vm['RMSE']:<7.2f} | Val_Macro_R2={vmacro['macro_avg_r2']:.4f} | Params={raw_params}"
            )

    board = (
        pd.DataFrame(val_records)
        .sort_values([PRIMARY_SELECTION_METRIC, SECONDARY_SELECTION_METRIC], ascending=True)
        .reset_index(drop=True)
    )
    return board, fitted_estimators, best_params_map


def run_leave_one_commodity_out(df_clean: pd.DataFrame) -> Dict[str, Any]:
    """
    Phase 8 Independent Leave-One-Commodity-Out Experiment:
    For each held-out commodity `c`, exclude ALL rows of `c` from training CV and validation
    selection, select the best model & hyperparameters using ONLY the remaining commodities,
    refit on remaining Train+Validation, and evaluate on `c`'s future Test partition.
    """
    loco_results: Dict[str, Any] = {}
    for held_out in sorted(df_clean["commodity_name"].unique()):
        train_rem = df_clean[(df_clean["commodity_name"] != held_out) & (df_clean["split_tag"] == "Train")].copy().reset_index(drop=True)
        val_rem = df_clean[(df_clean["commodity_name"] != held_out) & (df_clean["split_tag"] == "Validation")].copy().reset_index(drop=True)
        tv_rem = pd.concat([train_rem, val_rem], ignore_index=True).sort_values(
            ["observation_date", "commodity_name", "market_name"]
        ).reset_index(drop=True)
        test_held = df_clean[(df_clean["commodity_name"] == held_out) & (df_clean["split_tag"] == "Test")].copy().reset_index(drop=True)

        loco_board, loco_ests, loco_params = select_candidate_with_purged_cv(train_rem, val_rem, verbose=False)
        sel_row = loco_board.iloc[0]
        sel_algo = str(sel_row["Model"])
        sel_est = clone(loco_ests[sel_algo])

        pipe = Pipeline([("preprocessor", make_preprocessor()), ("regressor", sel_est)])
        pipe.fit(_prepare_feature_frame(tv_rem), tv_rem[TARGET_REG_COL].to_numpy(dtype=float))

        y_true = test_held[TARGET_REG_COL].to_numpy(dtype=float)
        y_pred = pipe.predict(_prepare_feature_frame(test_held))
        y_base = test_held["modal_price_rs_per_qtl"].to_numpy(dtype=float)

        m = compute_regression_metrics(y_true, y_pred)
        b = compute_regression_metrics(y_true, y_base)
        loco_results[held_out] = {
            "selected_algorithm": sel_algo,
            "selected_hyperparameters": loco_params[sel_algo],
            "selection_metric": PRIMARY_SELECTION_METRIC,
            "remaining_val_macro_rmse": float(sel_row["Val_Macro_RMSE"]),
            "remaining_val_rmse": float(sel_row["Val_RMSE"]),
            "test_samples": int(len(test_held)),
            "model_metrics": m,
            "persistence_baseline": {
                "MAE": b["MAE"],
                "MSE": b["MSE"],
                "RMSE": b["RMSE"],
                "R2": b["R2"],
                "MAPE": b["MAPE"],
            },
            "beats_baseline_rmse": bool(m["RMSE"] < b["RMSE"]),
        }
    return loco_results


def main():
    print("=" * 92)
    print("UNIFIED MULTI-COMMODITY MODEL TRAINING (v3.1.0-parity — DATE-PURGED CV & VAL_MACRO_RMSE)")
    print("=" * 92)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean unified dataset not found at {DATA_PATH}. Run py -3.12 scripts/build_features.py --all first."
        )

    df = pd.read_csv(DATA_PATH)
    df_clean = df.dropna(subset=[TARGET_REG_COL] + UNIFIED_NUMERIC_FEATURES).copy()
    df_clean["observation_date"] = pd.to_datetime(df_clean["observation_date"])
    df_clean["target_date_t_plus_1"] = pd.to_datetime(df_clean["target_date_t_plus_1"])
    df_clean = df_clean.sort_values(["observation_date", "commodity_name", "market_name"]).reset_index(drop=True)

    train_df = df_clean[df_clean["split_tag"] == "Train"].copy().reset_index(drop=True)
    val_df = df_clean[df_clean["split_tag"] == "Validation"].copy().reset_index(drop=True)
    test_df = df_clean[df_clean["split_tag"] == "Test"].copy().reset_index(drop=True)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(
        ["observation_date", "commodity_name", "market_name"]
    ).reset_index(drop=True)

    print(f"Chronological Partitions (Shared Cutoffs): Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    print(f"  Train Dates      : {train_df['observation_date'].min().date()} to {train_df['observation_date'].max().date()}")
    print(f"  Validation Dates : {val_df['observation_date'].min().date()} to {val_df['observation_date'].max().date()}")
    print(f"  Test Dates       : {test_df['observation_date'].min().date()} to {test_df['observation_date'].max().date()}")

    y_val = val_df[TARGET_REG_COL].to_numpy(dtype=float)
    val_base_macro = compute_group_macro_metrics(val_df, y_val, val_df["modal_price_rs_per_qtl"].to_numpy(dtype=float))
    print(
        f"\nValidation Persistence Baseline: Macro_RMSE=Rs.{val_base_macro['baseline_macro_avg_rmse']:.2f}/Qtl | "
        f"Pooled_RMSE=Rs.{compute_regression_metrics(y_val, val_df['modal_price_rs_per_qtl'].to_numpy(dtype=float))['RMSE']:.2f}/Qtl"
    )

    print("\n" + "-" * 92)
    print(f"STEP 1 & 2: DATE-PURGED WALK-FORWARD CV -> VALIDATION SELECTION BY {PRIMARY_SELECTION_METRIC}")
    print("-" * 92)

    val_df_board, fitted_estimators, best_params_map = select_candidate_with_purged_cv(train_df, val_df, verbose=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    val_df_board.to_csv(REPORTS_DIR / "unified_validation_leaderboard.csv", index=False)

    champion_row = val_df_board.iloc[0]
    champion_name = str(champion_row["Model"])
    champion_params = best_params_map[champion_name]
    champion_estimator = clone(fitted_estimators[champion_name])

    print(
        f"\nCHAMPION SELECTED STRICTLY BY {PRIMARY_SELECTION_METRIC}: {champion_name} "
        f"(Val_Macro_RMSE=Rs.{champion_row['Val_Macro_RMSE']:.2f}, Val_RMSE=Rs.{champion_row['Val_RMSE']:.2f}, Params={champion_params})"
    )

    # STEP 3: Refit selected champion on combined Train + Validation
    champion_pipeline = Pipeline([("preprocessor", make_preprocessor()), ("regressor", champion_estimator)])
    champion_pipeline.fit(_prepare_feature_frame(train_val_df), train_val_df[TARGET_REG_COL].to_numpy(dtype=float))

    # STEP 4: Single evaluation on untouched Test partition
    y_test = test_df[TARGET_REG_COL].to_numpy(dtype=float)
    test_preds = champion_pipeline.predict(_prepare_feature_frame(test_df))
    test_baseline_metrics = compute_regression_metrics(y_test, test_df["modal_price_rs_per_qtl"].to_numpy(dtype=float))
    overall_test = compute_regression_metrics(y_test, test_preds)
    macro_test = compute_group_macro_metrics(test_df, y_test, test_preds)
    test_df["predicted_price"] = test_preds

    print("\n" + "=" * 92)
    print("STEP 4: SINGLE UNTOUCHED TEST SET EVALUATION (CHAMPION VS BASELINE ONLY)")
    print("=" * 92)
    print(
        f"Macro-Average Metrics (Primary): Macro_MAE=Rs.{macro_test['macro_avg_mae']:.2f} | "
        f"Macro_RMSE=Rs.{macro_test['macro_avg_rmse']:.2f} (Base=Rs.{macro_test['baseline_macro_avg_rmse']:.2f}) | "
        f"Macro_R2={macro_test['macro_avg_r2']:.4f} | Macro_MAPE={macro_test['macro_avg_mape_pct']:.2f}%"
    )
    print(
        f"Pooled Test Metrics (Secondary): MAE=Rs.{overall_test['MAE']:.2f} | "
        f"RMSE=Rs.{overall_test['RMSE']:.2f} (Base=Rs.{test_baseline_metrics['RMSE']:.2f}) | "
        f"R2={overall_test['R2']:.4f} | MAPE={overall_test['MAPE']:.2f}%"
    )

    commodity_breakdown: Dict[str, Any] = {}
    per_comm_csv_rows: List[Dict[str, Any]] = []
    print("\nPer-Commodity Untouched Test Breakdown:")
    for comm in sorted(test_df["commodity_name"].unique()):
        sub = test_df[test_df["commodity_name"] == comm]
        y_c = sub[TARGET_REG_COL].to_numpy(dtype=float)
        p_c = sub["predicted_price"].to_numpy(dtype=float)
        b_c = sub["modal_price_rs_per_qtl"].to_numpy(dtype=float)

        cm = compute_regression_metrics(y_c, p_c)
        bm = compute_regression_metrics(y_c, b_c)
        cat = str(sub["category"].iloc[0])
        mean_p = round(float(np.mean(y_c)), 2)

        commodity_breakdown[comm] = {
            "category": cat,
            "mean_price_rs_per_qtl": mean_p,
            "test_mae_rs_per_qtl": cm["MAE"],
            "test_mse_rs_per_qtl": cm["MSE"],
            "test_rmse_rs_per_qtl": cm["RMSE"],
            "test_r2": cm["R2"],
            "test_mape_pct": cm["MAPE"],
            "sample_count": int(len(sub)),
            "persistence_baseline": {"mae": bm["MAE"], "rmse": bm["RMSE"], "r2": bm["R2"], "mape_pct": bm["MAPE"]},
            "improvement_over_baseline": {
                "rmse_reduction_rs_per_qtl": round(bm["RMSE"] - cm["RMSE"], 2),
                "mae_reduction_rs_per_qtl": round(bm["MAE"] - cm["MAE"], 2),
                "beats_baseline_rmse": bool(cm["RMSE"] < bm["RMSE"]),
            },
        }
        per_comm_csv_rows.append(
            {
                "Commodity": comm,
                "Category": cat,
                "Test_Samples": int(len(sub)),
                "Mean_Price": mean_p,
                "MAE": cm["MAE"],
                "MSE": cm["MSE"],
                "RMSE": cm["RMSE"],
                "R2": cm["R2"],
                "MAPE_Pct": cm["MAPE"],
                "Baseline_MAE": bm["MAE"],
                "Baseline_RMSE": bm["RMSE"],
                "Baseline_R2": bm["R2"],
                "RMSE_Improvement": round(bm["RMSE"] - cm["RMSE"], 2),
                "Beats_Baseline": bool(cm["RMSE"] < bm["RMSE"]),
            }
        )
        print(
            f"  {comm:<10} | N={len(sub):<3} | RMSE=Rs.{cm['RMSE']:<6.2f} (Base Rs.{bm['RMSE']:<6.2f}) | "
            f"MAE=Rs.{cm['MAE']:<5.2f} | R2={cm['R2']:.4f} | MAPE={cm['MAPE']:5.2f}%"
        )

    pd.DataFrame(per_comm_csv_rows).to_csv(REPORTS_DIR / "unified_test_per_commodity_metrics.csv", index=False)

    per_series_breakdown: Dict[str, Any] = {}
    for (comm, mkt), sub in test_df.groupby(["commodity_name", "market_name"]):
        key = f"{comm} @ {mkt}"
        y_s = sub[TARGET_REG_COL].to_numpy(dtype=float)
        p_s = sub["predicted_price"].to_numpy(dtype=float)
        b_s = sub["modal_price_rs_per_qtl"].to_numpy(dtype=float)
        sm = compute_regression_metrics(y_s, p_s)
        bm = compute_regression_metrics(y_s, b_s)
        per_series_breakdown[key] = {
            "category": str(sub["category"].iloc[0]),
            "sample_count": int(len(sub)),
            "mean_actual_rs_per_qtl": round(float(np.mean(y_s)), 2),
            "std_actual_rs_per_qtl": round(float(np.std(y_s)), 2),
            "mae": sm["MAE"],
            "mse": sm["MSE"],
            "rmse": sm["RMSE"],
            "r2": sm["R2"],
            "mape_pct": sm["MAPE"],
            "persistence_baseline": {"mae": bm["MAE"], "rmse": bm["RMSE"], "r2": bm["R2"], "mape_pct": bm["MAPE"]},
            "improvement_over_baseline": {
                "rmse_reduction_rs_per_qtl": round(bm["RMSE"] - sm["RMSE"], 2),
                "mae_reduction_rs_per_qtl": round(bm["MAE"] - sm["MAE"], 2),
                "beats_baseline_rmse": bool(sm["RMSE"] < bm["RMSE"]),
            },
        }

    # Independent LOCO Experiment (Phase 8: excludes held-out commodity from CV & validation selection!)
    print("\nRunning Independent Leave-One-Commodity-Out (LOCO) Generalization Check (Zero Held-Out Leakage)...")
    loco_results = run_leave_one_commodity_out(df_clean)
    for comm, lr in loco_results.items():
        print(
            f"  Held-out {comm:<10} | Indep Selected={lr['selected_algorithm']:<26} | "
            f"Model RMSE=Rs.{lr['model_metrics']['RMSE']:<6.2f} vs Base Rs.{lr['persistence_baseline']['RMSE']:<6.2f} | "
            f"Beats={lr['beats_baseline_rmse']}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "unified_food_price_model.joblib"
    joblib.dump(champion_pipeline, model_path)

    supported_pairs = [
        {"commodity": c, "market": m}
        for c, m in sorted(df_clean[["commodity_name", "market_name"]].drop_duplicates().itertuples(index=False, name=None))
    ]

    meta = {
        "model_architecture": "Unified Cross-Commodity Regression Pipeline",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "selection_protocol": (
            f"DatePurgedWalkForwardCV inside Train -> Validation Selection by {PRIMARY_SELECTION_METRIC} -> "
            "Refit Train+Val -> Single Evaluation of Champion vs Baseline on Untouched Test"
        ),
        "primary_selection_metric": PRIMARY_SELECTION_METRIC,
        "secondary_selection_metric": SECONDARY_SELECTION_METRIC,
        "champion_algorithm": champion_name,
        "champion_hyperparameters": champion_params,
        "dataset_file": DATA_PATH.name,
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split_cutoffs": {
            "train_dates": f"{train_df['observation_date'].min().date()} to {train_df['observation_date'].max().date()}",
            "validation_dates": f"{val_df['observation_date'].min().date()} to {val_df['observation_date'].max().date()}",
            "test_dates": f"{test_df['observation_date'].min().date()} to {test_df['observation_date'].max().date()}",
        },
        "total_training_records": int(len(train_df)),
        "total_validation_records": int(len(val_df)),
        "total_test_records": int(len(test_df)),
        "supported_commodities": sorted(df_clean["commodity_name"].unique().tolist()),
        "supported_markets": sorted(df_clean["market_name"].unique().tolist()),
        "supported_commodity_market_pairs": supported_pairs,
        "supported_categories": sorted(df_clean["category"].unique().tolist()),
        "categorical_features": UNIFIED_CATEGORICAL_FEATURES,
        "numeric_features": UNIFIED_NUMERIC_FEATURES,
        "all_features": UNIFIED_ALL_FEATURES,
        "prediction_horizon": "Next observed active trading session (t+1, not guaranteed next calendar day)",
        "price_unit": "Rs/Quintal (Wholesale auction bids only; Rs/Kg division is unit conversion only)",
        "macro_average_test_metrics": macro_test,
        "overall_test_metrics": {
            "mae_rs_per_qtl": overall_test["MAE"],
            "mse_rs_per_qtl": overall_test["MSE"],
            "rmse_rs_per_qtl": overall_test["RMSE"],
            "r2_score": overall_test["R2"],
            "mape_pct": overall_test["MAPE"],
        },
        "persistence_baseline_test_metrics": test_baseline_metrics,
        "improvement_over_baseline_test": {
            "macro_rmse_reduction_rs_per_qtl": macro_test["macro_rmse_improvement"],
            "mae_reduction_rs_per_qtl": round(test_baseline_metrics["MAE"] - overall_test["MAE"], 2),
            "rmse_reduction_rs_per_qtl": round(test_baseline_metrics["RMSE"] - overall_test["RMSE"], 2),
            "beats_baseline_macro_rmse": bool(macro_test["macro_avg_rmse"] < macro_test["baseline_macro_avg_rmse"]),
            "beats_baseline_pooled_rmse": bool(overall_test["RMSE"] < test_baseline_metrics["RMSE"]),
        },
        "per_commodity_metrics": commodity_breakdown,
        "per_series_metrics": per_series_breakdown,
        "leave_one_commodity_out_experiment": loco_results,
        "validation_candidates_benchmarked": val_df_board.to_dict(orient="records"),
    }

    with open(MODELS_DIR / "unified_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved Champion Model & Metadata: {model_path.name}, unified_model_metadata.json")

    REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = REPORTS_FIG_DIR / "unified_multi_commodity_evaluation.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {"Tomato": "#d9534f", "Onion": "#f0ad4e", "Moong Dal": "#5cb85c", "Sugar": "#0275d8"}

    for idx, comm in enumerate(sorted(test_df["commodity_name"].unique())):
        ax = axes[idx // 2, idx % 2]
        sub = test_df[test_df["commodity_name"] == comm]
        y_act = sub[TARGET_REG_COL].to_numpy(dtype=float)
        y_pr = sub["predicted_price"].to_numpy(dtype=float)
        cm = commodity_breakdown[comm]
        ax.scatter(y_act, y_pr, alpha=0.65, color=colors.get(comm, "#333333"), s=32)
        lo, hi = min(y_act.min(), y_pr.min()), max(y_act.max(), y_pr.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, label="Ideal 1:1 Fit")
        ax.set_title(
            f"{comm} (R2={cm['test_r2']:.4f}, RMSE=Rs.{cm['test_rmse_rs_per_qtl']:.1f}, Base=Rs.{cm['persistence_baseline']['rmse']:.1f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Actual Next Session Price (Rs/Qtl)")
        ax.set_ylabel("Predicted Price (Rs/Qtl)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle(
        f"Unified Model Untouched Test Performance: {champion_name} (Selected by {PRIMARY_SELECTION_METRIC})\n"
        f"Macro-Avg RMSE=Rs.{macro_test['macro_avg_rmse']:.2f} (Macro-R2={macro_test['macro_avg_r2']:.4f}) | "
        f"Pooled RMSE=Rs.{overall_test['RMSE']:.2f} (R2={overall_test['R2']:.4f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved evaluation figure: {fig_path}")


if __name__ == "__main__":
    main()
