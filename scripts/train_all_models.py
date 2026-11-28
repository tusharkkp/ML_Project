"""
Leak-Free Comprehensive ML Training & Evaluation Pipeline — All Syllabus Algorithms (v3.1.0-parity).

Strict Experimental Integrity Guarantees:
1. DatePurgedWalkForwardCV inside Train across regression & global classifier hyperparameter search.
2. Validation-only model comparison & Champion selection (`validation_leaderboard_*.csv`).
3. Single Untouched Test Evaluation: evaluates ONLY the Validation-selected Champion vs
   Naive Persistence / Majority Baseline on Test (`model_comparison_metrics_*.csv` &
   `classification_comparison_*.csv`) — zero multi-candidate snooping on Test.
4. Train-Only Unsupervised Analysis (Phase 9): scaling, optimal `k` selection, K-Means,
   Hierarchical Ward, DBSCAN, and PCA fit strictly on Train (Validation out-of-sample projection
   only for KMeans/PCA; untouched Test untouched), plus detailed behavioral Cluster Profiles.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.metrics import (
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    DATASET_VERSION,
    FEATURE_SCHEMA_VERSION,
    SPECIALIST_NUMERIC_FEATURES,
    TARGET_CLF_DIR_COL,
    TARGET_CLF_SPIKE_COL,
    TARGET_REG_COL,
    UNIFIED_ALL_FEATURES,
    UNIFIED_CATEGORICAL_FEATURES,
    UNIFIED_NUMERIC_FEATURES,
    DatePurgedWalkForwardCV,
)
from src.metrics import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
    compute_regression_metrics,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = SPECIALIST_NUMERIC_FEATURES


class ChronologicalStackingRegressor(BaseEstimator, RegressorMixin):
    """Leak-free stacking regressor using DatePurgedWalkForwardCV / forward splits for meta-features."""

    def __init__(self, estimators, final_estimator, obs_dates=None, tgt_dates=None, n_splits: int = 5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.obs_dates = obs_dates
        self.tgt_dates = tgt_dates
        self.n_splits = n_splits

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)
        if self.obs_dates is not None and self.tgt_dates is not None and len(self.obs_dates) == len(X_arr):
            splitter = DatePurgedWalkForwardCV(self.obs_dates, self.tgt_dates, n_splits=self.n_splits)
            splits = list(splitter.split(X_arr))
        else:
            splits = list(TimeSeriesSplit(n_splits=self.n_splits).split(X_arr))

        oof_preds, oof_targets = [], []
        for tr_idx, val_idx in splits:
            fold_cols = []
            for _, est in self.estimators:
                e = clone(est).fit(X_arr[tr_idx], y_arr[tr_idx])
                fold_cols.append(e.predict(X_arr[val_idx]))
            oof_preds.append(np.column_stack(fold_cols))
            oof_targets.append(y_arr[val_idx])

        self.final_estimator_ = clone(self.final_estimator).fit(np.vstack(oof_preds), np.concatenate(oof_targets))
        self.estimators_ = [clone(est).fit(X_arr, y_arr) for _, est in self.estimators]
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        meta_X = np.column_stack([e.predict(X_arr) for e in self.estimators_])
        return self.final_estimator_.predict(meta_X)


def load_series_data(dataset_path: str) -> Dict[str, Any]:
    """Load clean chronological dataset partitions preserving dates for purged CV."""
    df = pd.read_csv(dataset_path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["target_date_t_plus_1"] = pd.to_datetime(df["target_date_t_plus_1"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    train_df = df[df["split_tag"] == "Train"].reset_index(drop=True).copy()
    val_df = df[df["split_tag"] == "Validation"].reset_index(drop=True).copy()
    test_df = df[df["split_tag"] == "Test"].reset_index(drop=True).copy()
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_val_df": train_val_df,
        "X_train": train_df[FEATURE_COLS],
        "y_train_reg": train_df[TARGET_REG_COL].to_numpy(dtype=float),
        "y_train_clf": train_df[TARGET_CLF_SPIKE_COL].to_numpy(dtype=int),
        "X_val": val_df[FEATURE_COLS],
        "y_val_reg": val_df[TARGET_REG_COL].to_numpy(dtype=float),
        "y_val_clf": val_df[TARGET_CLF_SPIKE_COL].to_numpy(dtype=int),
        "X_train_val": train_val_df[FEATURE_COLS],
        "y_train_val_reg": train_val_df[TARGET_REG_COL].to_numpy(dtype=float),
        "y_train_val_clf": train_val_df[TARGET_CLF_SPIKE_COL].to_numpy(dtype=int),
        "X_test": test_df[FEATURE_COLS],
        "y_test_reg": test_df[TARGET_REG_COL].to_numpy(dtype=float),
        "y_test_clf": test_df[TARGET_CLF_SPIKE_COL].to_numpy(dtype=int),
        "test_dates": test_df["observation_date"].dt.strftime("%Y-%m-%d").to_numpy(),
    }


def run_regression_models(data: Dict[str, Any], commodity_name: str):
    """
    Tune candidates on Train with DatePurgedWalkForwardCV, rank ALL 14 on Validation,
    and evaluate ONLY the Validation Champion vs Persistence Baseline once on Test.
    """
    print("\n" + "=" * 84)
    print(f"MODULE 1: LEAK-FREE REGRESSION PIPELINES ({commodity_name} — DATE-PURGED CV)")
    print("=" * 84)

    X_tr, y_tr = data["X_train"], data["y_train_reg"]
    X_val, y_val = data["X_val"], data["y_val_reg"]
    X_tv, y_tv = data["X_train_val"], data["y_train_val_reg"]
    X_te, y_te = data["X_test"], data["y_test_reg"]

    y_val_base = data["val_df"]["modal_price_rs_per_qtl"].to_numpy(dtype=float)
    y_test_base = data["test_df"]["modal_price_rs_per_qtl"].to_numpy(dtype=float)

    purged_cv_train = DatePurgedWalkForwardCV(
        data["train_df"]["observation_date"],
        data["train_df"]["target_date_t_plus_1"],
        n_splits=5,
    )
    p_idx = FEATURE_COLS.index("modal_price_rs_per_qtl")

    from sklearn.compose import ColumnTransformer as CT
    slr_pipe = Pipeline([("select", CT([("p", "passthrough", [p_idx])], remainder="drop")), ("reg", LinearRegression())])

    def make_stacking(obs_d, tgt_d):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ChronologicalStackingRegressor(
                estimators=[
                    ("lasso", Lasso(alpha=1.0, max_iter=5000)),
                    ("ridge", Ridge(alpha=10.0)),
                    ("dt", DecisionTreeRegressor(max_depth=4, random_state=42)),
                ],
                final_estimator=Ridge(alpha=1.0),
                obs_dates=obs_d,
                tgt_dates=tgt_d,
                n_splits=5,
            )),
        ])

    candidates = [
        ("Simple Linear Regression", slr_pipe, {}),
        ("Multiple Linear Regression", Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]), {}),
        (
            "Polynomial Regression (deg=2)",
            Pipeline([("scaler", StandardScaler()), ("poly", PolynomialFeatures(degree=2, include_bias=False)), ("reg", Ridge(random_state=42))]),
            {"reg__alpha": [1.0, 10.0, 100.0]},
        ),
        (
            "Ridge Regression (L2)",
            Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42))]),
            {"reg__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        (
            "Lasso Regression (L1)",
            Pipeline([("scaler", StandardScaler()), ("reg", Lasso(max_iter=10000, random_state=42))]),
            {"reg__alpha": [0.1, 0.5, 1.0, 5.0, 10.0]},
        ),
        (
            "Decision Tree Regressor",
            Pipeline([("scaler", StandardScaler()), ("reg", DecisionTreeRegressor(random_state=42))]),
            {"reg__max_depth": [3, 4, 5, 6], "reg__min_samples_leaf": [2, 5, 10]},
        ),
        ("KNN Regressor", Pipeline([("scaler", StandardScaler()), ("reg", KNeighborsRegressor())]), {"reg__n_neighbors": [5, 7, 11]}),
        ("SVR (RBF Kernel)", Pipeline([("scaler", StandardScaler()), ("reg", SVR(kernel="rbf"))]), {"reg__C": [10.0, 100.0, 500.0], "reg__epsilon": [5.0, 10.0]}),
        ("Bagging Regressor", Pipeline([("scaler", StandardScaler()), ("reg", BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=5), n_estimators=50, random_state=42, n_jobs=1))]), {}),
        ("Random Forest Regressor", Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))]), {"reg__max_depth": [5, 8, 12], "reg__min_samples_leaf": [2, 5]}),
        ("AdaBoost Regressor", Pipeline([("scaler", StandardScaler()), ("reg", AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4), random_state=42))]), {"reg__n_estimators": [50, 100], "reg__learning_rate": [0.05, 0.1]}),
        ("Gradient Boosting Regressor", Pipeline([("scaler", StandardScaler()), ("reg", GradientBoostingRegressor(random_state=42))]), {"reg__n_estimators": [50, 100], "reg__learning_rate": [0.05, 0.1], "reg__max_depth": [3, 4]}),
        ("Stacking Regressor", make_stacking(data["train_df"]["observation_date"], data["train_df"]["target_date_t_plus_1"]), {}),
        (
            "Voting Regressor",
            Pipeline([
                ("scaler", StandardScaler()),
                ("reg", VotingRegressor([
                    ("lasso", Lasso(alpha=1.0, max_iter=5000)),
                    ("ridge", Ridge(alpha=10.0)),
                    ("rf", RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=1)),
                ])),
            ]),
            {},
        ),
    ]

    val_base_m = compute_regression_metrics(y_val, y_val_base)
    val_rows = []
    tuned_pipes: Dict[str, Any] = {}
    val_predictions: Dict[str, np.ndarray] = {"Naive Persistence Baseline (P_t)": y_val_base}

    for name, pipe, grid in candidates:
        if grid:
            gs = GridSearchCV(pipe, grid, cv=purged_cv_train, scoring="neg_root_mean_squared_error", n_jobs=1)
            gs.fit(X_tr, y_tr)
            best_p = gs.best_estimator_
            params = {k.replace("reg__", ""): v for k, v in gs.best_params_.items()}
        else:
            best_p = clone(pipe).fit(X_tr, y_tr)
            params = {}
        tuned_pipes[name] = best_p
        vp = best_p.predict(X_val)
        val_predictions[name] = vp
        vm = compute_regression_metrics(y_val, vp)
        val_rows.append(
            {
                "Model": name,
                "Val_MAE": vm["MAE"],
                "Val_MSE": vm["MSE"],
                "Val_RMSE": vm["RMSE"],
                "Val_R2": vm["R2"],
                "Val_MAPE (%)": vm["MAPE"],
                "Beats_Val_Baseline": bool(vm["RMSE"] < val_base_m["RMSE"]),
                "Best_Params": json.dumps(params),
            }
        )

    val_df = pd.DataFrame(val_rows).sort_values("Val_RMSE").reset_index(drop=True)
    winner_name = str(val_df.iloc[0]["Model"])
    print("\nValidation Leaderboard (All Candidates Evaluated Strictly Here):")
    print(val_df[["Model", "Val_MAE", "Val_RMSE", "Val_R2", "Beats_Val_Baseline"]].head(6).to_string(index=False))
    print(f"--> Validation Champion: {winner_name} (Val_RMSE=Rs.{val_df.iloc[0]['Val_RMSE']:.2f})")

    # Step 3 & 4: Refit ONLY Champion on Train+Val and evaluate ONLY Champion vs Persistence ONCE on Test
    if winner_name == "Stacking Regressor":
        champion_pipe = make_stacking(
            data["train_val_df"]["observation_date"],
            data["train_val_df"]["target_date_t_plus_1"],
        ).fit(X_tv, y_tv)
    else:
        champion_pipe = clone(tuned_pipes[winner_name]).fit(X_tv, y_tv)

    champ_test_preds = champion_pipe.predict(X_te)
    champ_test_m = compute_regression_metrics(y_te, champ_test_preds)
    base_test_m = compute_regression_metrics(y_te, y_test_base)

    test_rows = [
        {
            "Model": "Naive Persistence Baseline (P_t)",
            "MAE": base_test_m["MAE"],
            "MSE": base_test_m["MSE"],
            "RMSE": base_test_m["RMSE"],
            "R2": base_test_m["R2"],
            "MAPE (%)": base_test_m["MAPE"],
            "Selected_By_Validation": False,
        },
        {
            "Model": winner_name,
            "MAE": champ_test_m["MAE"],
            "MSE": champ_test_m["MSE"],
            "RMSE": champ_test_m["RMSE"],
            "R2": champ_test_m["R2"],
            "MAPE (%)": champ_test_m["MAPE"],
            "Selected_By_Validation": True,
        },
    ]

    reg_df = pd.DataFrame(test_rows)
    test_preds = {
        "Naive Persistence Baseline (P_t)": y_test_base,
        winner_name: champ_test_preds,
    }
    return reg_df, val_df, winner_name, champion_pipe, test_preds, val_predictions


def run_gradient_descent(data: Dict[str, Any]):
    """
    Benchmark Batch GD, SGD, and Mini-batch GD strictly with Train-fitted scaler evaluated
    on Validation (keeping Test untouched for Champion evaluation only).
    """
    print("\n" + "=" * 84)
    print("MODULE 2: GRADIENT DESCENT OPTIMIZATION VARIANTS (VALIDATION BENCHMARK)")
    print("=" * 84)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(data["X_train"].to_numpy(dtype=float))
    y_tr = data["y_train_reg"]
    X_val = scaler.transform(data["X_val"].to_numpy(dtype=float))
    y_val = data["y_val_reg"]

    results = []
    losses = {}

    bgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=0.001, learning_rate="constant", eta0=0.001, max_iter=1, warm_start=True, random_state=42)
    bgd_l = []
    for _ in range(300):
        bgd.partial_fit(X_tr, y_tr)
        bgd_l.append(float(np.mean((y_tr - bgd.predict(X_tr)) ** 2)))
    m_bgd = compute_regression_metrics(y_val, bgd.predict(X_val))
    results.append({"Model": "Batch Gradient Descent", "Evaluated_Split": "Validation", "Val_MAE": m_bgd["MAE"], "Val_MSE": m_bgd["MSE"], "Val_RMSE": m_bgd["RMSE"], "Val_R2": m_bgd["R2"], "Val_MAPE (%)": m_bgd["MAPE"]})
    losses["Batch GD"] = bgd_l

    sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=0.001, learning_rate="invscaling", eta0=0.01, power_t=0.25, max_iter=1, warm_start=True, random_state=42)
    sgd_l = []
    for ep in range(300):
        idx = np.random.RandomState(ep).permutation(len(X_tr))
        sgd.partial_fit(X_tr[idx], y_tr[idx])
        sgd_l.append(float(np.mean((y_tr - sgd.predict(X_tr)) ** 2)))
    m_sgd = compute_regression_metrics(y_val, sgd.predict(X_val))
    results.append({"Model": "Stochastic Gradient Descent", "Evaluated_Split": "Validation", "Val_MAE": m_sgd["MAE"], "Val_MSE": m_sgd["MSE"], "Val_RMSE": m_sgd["RMSE"], "Val_R2": m_sgd["R2"], "Val_MAPE (%)": m_sgd["MAPE"]})
    losses["SGD"] = sgd_l

    mbgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=0.001, learning_rate="invscaling", eta0=0.01, power_t=0.25, max_iter=1, warm_start=True, random_state=42)
    mb_l = []
    for ep in range(300):
        idx = np.random.RandomState(ep).permutation(len(X_tr))
        for start in range(0, len(X_tr), 32):
            b = idx[start : start + 32]
            mbgd.partial_fit(X_tr[b], y_tr[b])
        mb_l.append(float(np.mean((y_tr - mbgd.predict(X_tr)) ** 2)))
    m_mb = compute_regression_metrics(y_val, mbgd.predict(X_val))
    results.append({"Model": "Mini-batch GD (batch=32)", "Evaluated_Split": "Validation", "Val_MAE": m_mb["MAE"], "Val_MSE": m_mb["MSE"], "Val_RMSE": m_mb["RMSE"], "Val_R2": m_mb["R2"], "Val_MAPE (%)": m_mb["MAPE"]})
    losses["Mini-batch GD"] = mb_l

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, vals in losses.items():
        ax.plot(vals[:200], label=k, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE (Log Scale)")
    ax.set_title("Gradient Descent Optimization Convergence (Train Fit / Val Benchmarked)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_gradient_descent_convergence.png", dpi=300)
    plt.close()

    return pd.DataFrame(results)


def run_classification_models(data: Dict[str, Any]):
    """
    Tune thresholds & rank ALL 10 specialist classifiers strictly on Validation;
    evaluate ONLY the Validation Champion vs Majority Baseline once on untouched Test.
    """
    print("\n" + "=" * 84)
    print("MODULE 3: BINARY PRICE SPIKE CLASSIFIERS (VAL RANKING -> SINGLE TEST EVAL)")
    print("=" * 84)

    X_tr, y_tr = data["X_train"], data["y_train_clf"]
    X_val, y_val = data["X_val"], data["y_val_clf"]
    X_tv, y_tv = data["X_train_val"], data["y_train_val_clf"]
    X_te, y_te = data["X_test"], data["y_test_clf"]

    maj_cls = int(np.bincount(y_tr).argmax())
    maj_preds_test = np.full_like(y_te, maj_cls)
    base_test = compute_binary_classification_metrics(y_te, maj_preds_test, None, f"Majority Baseline (class={maj_cls})")
    base_test["Selected_By_Val"] = False
    base_test["Val_Threshold"] = None

    def make_pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    candidates = [
        ("Logistic Regression", make_pipe(LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))),
        ("Gaussian Naive Bayes", make_pipe(GaussianNB())),
        ("KNN (k=5)", make_pipe(KNeighborsClassifier(n_neighbors=5))),
        ("Decision Tree (CART)", make_pipe(DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"))),
        ("SVM (RBF Kernel)", make_pipe(SVC(kernel="rbf", C=5.0, probability=True, random_state=42, class_weight="balanced"))),
        ("Bagging Classifier", make_pipe(BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=42, n_jobs=1))),
        ("AdaBoost Classifier", make_pipe(AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.1, random_state=42))),
        ("Gradient Boosting Classifier", make_pipe(GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42))),
        ("Stacking Classifier", make_pipe(StackingClassifier(estimators=[("lr", LogisticRegression(max_iter=1000, class_weight="balanced")), ("knn", KNeighborsClassifier(n_neighbors=5)), ("dt", DecisionTreeClassifier(max_depth=4, class_weight="balanced"))], final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=1))),
        ("Voting Classifier", make_pipe(VotingClassifier(estimators=[("lr", LogisticRegression(max_iter=1000, class_weight="balanced")), ("knn", KNeighborsClassifier(n_neighbors=5)), ("svm", SVC(kernel="rbf", C=5.0, probability=True, class_weight="balanced"))], voting="soft", n_jobs=1))),
    ]

    val_rows = []
    best_thresholds: Dict[str, float] = {}
    roc_val_data = {}

    for name, pipe in candidates:
        fitted = clone(pipe).fit(X_tr, y_tr)
        v_prob = fitted.predict_proba(X_val)[:, 1]
        best_th, best_f1 = 0.5, -1.0
        for th in np.arange(0.25, 0.76, 0.05):
            f = f1_score(y_val, (v_prob >= th).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_th = round(float(th), 2)
        best_thresholds[name] = best_th
        vm = compute_binary_classification_metrics(y_val, (v_prob >= best_th).astype(int), v_prob, name)
        vm["Val_Threshold"] = best_th
        val_rows.append(vm)
        if len(np.unique(y_val)) > 1:
            fpr, tpr, _ = roc_curve(y_val, v_prob)
            roc_val_data[name] = (fpr, tpr, vm["ROC-AUC"])

    val_board = pd.DataFrame(val_rows).sort_values(["F1-Score", "PR-AUC"], ascending=False).reset_index(drop=True)
    winner_name = str(val_board.iloc[0]["Model"])
    winner_th = best_thresholds[winner_name]

    winner_pipe = dict(candidates)[winner_name]
    final_p = clone(winner_pipe).fit(X_tv, y_tv)
    t_prob = final_p.predict_proba(X_te)[:, 1]
    t_pred = (t_prob >= winner_th).astype(int)
    champ_test = compute_binary_classification_metrics(y_te, t_pred, t_prob, winner_name)
    champ_test["Selected_By_Val"] = True
    champ_test["Val_Threshold"] = winner_th

    test_df = pd.DataFrame([base_test, champ_test])

    if roc_val_data:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for name, (fpr, tpr, auc_v) in roc_val_data.items():
            ax.plot(fpr, tpr, label=f"{name} (Val AUC={auc_v:.3f})", linewidth=1.4)
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Validation ROC Curves (Candidate Selection)", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "11_roc_curves_classification.png", dpi=300)
        plt.close()

    cm_champ = confusion_matrix(y_te, t_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm_champ, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Normal", "Spike"], yticklabels=["Normal", "Spike"])
    ax.set_title(f"Untouched Test Confusion Matrix: {winner_name}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_confusion_matrices.png", dpi=300)
    plt.close()

    return test_df, val_board


def run_clustering_pca(data: Dict[str, Any], commodity_name: str):
    """
    Phase 9 Train-Only Unsupervised Evaluation:
    - Scaling, optimal `k` selection, K-Means, Ward Hierarchical, DBSCAN & PCA fit STRICTLY on Train.
    - Test set is NEVER used for scaling, cluster count selection, or unsupervised fitting.
    - Reports explicit cluster behavioral profiles and out-of-sample generalization caveats.
    """
    print("\n" + "=" * 84)
    print(f"MODULE 4: TRAIN-ONLY CLUSTERING, PROFILES & PCA ({commodity_name})")
    print("=" * 84)

    train_df = data["train_df"].copy()
    val_df = data["val_df"].copy()

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(data["X_train"].to_numpy(dtype=float))
    X_val_scaled = scaler.transform(data["X_val"].to_numpy(dtype=float))

    inertias, sils = [], []
    k_range = list(range(2, 9))
    for k in k_range:
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_tr_scaled)
        lbl = km_k.labels_
        inertias.append(float(km_k.inertia_))
        sils.append(float(silhouette_score(X_tr_scaled, lbl)))

    best_k = k_range[int(np.argmax(sils))]
    km_model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_tr_scaled)
    km_tr_lbl = km_model.labels_
    km_val_lbl = km_model.predict(X_val_scaled)

    km_val_sil = (
        round(float(silhouette_score(X_val_scaled, km_val_lbl)), 4)
        if len(set(km_val_lbl)) > 1
        else None
    )

    agg_lbl = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X_tr_scaled)
    db_lbl = DBSCAN(eps=2.5, min_samples=10).fit_predict(X_tr_scaled)
    n_db = len(set(db_lbl)) - (1 if -1 in db_lbl else 0)
    db_mask = db_lbl != -1
    db_sil = silhouette_score(X_tr_scaled[db_mask], db_lbl[db_mask]) if n_db > 1 and db_mask.sum() > 1 else 0.0
    db_db = davies_bouldin_score(X_tr_scaled[db_mask], db_lbl[db_mask]) if n_db > 1 and db_mask.sum() > 1 else 0.0

    cluster_df = pd.DataFrame([
        {
            "Model": f"K-Means (k={best_k})",
            "Fitted_Split": "Train Only",
            "Out_Of_Sample_Supported": True,
            "Train_Silhouette": round(float(silhouette_score(X_tr_scaled, km_tr_lbl)), 4),
            "Train_Davies_Bouldin": round(float(davies_bouldin_score(X_tr_scaled, km_tr_lbl)), 4),
            "Val_Out_Of_Sample_Silhouette": km_val_sil,
            "Limitation_Note": "Selected on Train only; high silhouette alone does not prove stable forecasting regimes.",
        },
        {
            "Model": f"Hierarchical (Ward, k={best_k})",
            "Fitted_Split": "Train Only (Non-Inductive)",
            "Out_Of_Sample_Supported": False,
            "Train_Silhouette": round(float(silhouette_score(X_tr_scaled, agg_lbl)), 4),
            "Train_Davies_Bouldin": round(float(davies_bouldin_score(X_tr_scaled, agg_lbl)), 4),
            "Val_Out_Of_Sample_Silhouette": None,
            "Limitation_Note": "Agglomerative clustering lacks native out-of-sample `.predict()`; descriptive Train analysis only.",
        },
        {
            "Model": f"DBSCAN (Train clusters={n_db})",
            "Fitted_Split": "Train Only (Transductive)",
            "Out_Of_Sample_Supported": False,
            "Train_Silhouette": round(float(db_sil), 4),
            "Train_Davies_Bouldin": round(float(db_db), 4),
            "Val_Out_Of_Sample_Silhouette": None,
            "Limitation_Note": "Transductive density clustering restricted strictly to Train without unseen-sample extrapolation.",
        },
    ])

    # Cluster behavioral profiles (Train partition, K-Means clusters)
    profile_rows = []
    train_df["kmeans_cluster"] = km_tr_lbl
    total_tr = len(train_df)
    for cid, grp in train_df.groupby("kmeans_cluster"):
        mean_p = float(grp["modal_price_rs_per_qtl"].mean())
        std_p = float(grp["modal_price_rs_per_qtl"].std()) if len(grp) > 1 else 0.0
        cv_pct = (std_p / mean_p * 100.0) if mean_p > 0 else 0.0
        profile_rows.append(
            {
                "Cluster_ID": int(cid),
                "Train_Count": int(len(grp)),
                "Share_Pct": round(len(grp) / total_tr * 100.0, 1),
                "Mean_Modal_Price_Rs_Qtl": round(mean_p, 2),
                "Price_Std_Dev_Rs_Qtl": round(std_p, 2),
                "CV_Volatility_Pct": round(cv_pct, 2),
                "Mean_Arrival_Tonnes": round(float(grp["arrival_tonnes"].mean()), 2),
                "Price_Spike_Rate_Pct": round(float(grp["is_price_spike"].mean() * 100.0), 2),
            }
        )
    profile_df = pd.DataFrame(profile_rows).sort_values("Cluster_ID").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(k_range, inertias, "bo-")
    ax1.axvline(best_k, color="red", linestyle="--")
    ax1.set_title(f"Train-Only K-Means Elbow ({commodity_name})")
    ax2.plot(k_range, sils, "gs-")
    ax2.axvline(best_k, color="red", linestyle="--")
    ax2.set_title("Train-Only Silhouette Analysis")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_kmeans_elbow_silhouette.png", dpi=300)
    plt.close()

    sub_idx = np.random.RandomState(42).choice(len(X_tr_scaled), size=min(200, len(X_tr_scaled)), replace=False)
    Z = linkage(X_tr_scaled[sub_idx], method="ward")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, truncate_mode="level", p=5, ax=ax)
    ax.set_title(f"Train-Only Hierarchical Dendrogram ({commodity_name})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_hierarchical_dendrogram.png", dpi=300)
    plt.close()

    pca = PCA(n_components=min(15, X_tr_scaled.shape[1])).fit(X_tr_scaled)
    X_pca_tr = pca.transform(X_tr_scaled)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, alpha=0.7, label="Individual")
    ax1.plot(range(1, len(cum) + 1), cum, "ro-", label="Cumulative")
    ax1.axhline(0.95, color="green", linestyle="--", label="95% Threshold")
    ax1.set_title("Train-Only PCA Scree Plot")
    ax1.legend()
    sc = ax2.scatter(X_pca_tr[:, 0], X_pca_tr[:, 1], c=km_tr_lbl, cmap="viridis", s=14, alpha=0.65)
    ax2.set_title("Train PCA 2D Cluster Projection")
    plt.colorbar(sc, ax=ax2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "15_pca_scree_and_clusters.png", dpi=300)
    plt.close()

    return cluster_df, profile_df


def train_global_early_warning_classifier():
    """
    Phase 7: Train global 3-class (Normal / Surge / Crash) early-warning classifier
    using DatePurgedWalkForwardCV on Train, Validation selection by Macro-F1,
    refit Train+Val, and single evaluation on untouched Test.
    """
    print("\n" + "=" * 88)
    print("PHASE 7: GLOBAL MULTI-COMMODITY EARLY-WARNING SPIKE CLASSIFIER (DATE-PURGED CV)")
    print("=" * 88)

    data_path = PROCESSED_DIR / "model_ready_unified_commodities_clean.csv"
    df = pd.read_csv(data_path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["target_date_t_plus_1"] = pd.to_datetime(df["target_date_t_plus_1"])
    df = df.sort_values(["observation_date", "commodity_name", "market_name"]).reset_index(drop=True)

    train_df = df[df["split_tag"] == "Train"].reset_index(drop=True).copy()
    val_df = df[df["split_tag"] == "Validation"].reset_index(drop=True).copy()
    test_df = df[df["split_tag"] == "Test"].reset_index(drop=True).copy()
    tv_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(
        ["observation_date", "commodity_name", "market_name"]
    ).reset_index(drop=True)

    X_tr = train_df[UNIFIED_ALL_FEATURES].copy()
    X_val = val_df[UNIFIED_ALL_FEATURES].copy()
    X_tv = tv_df[UNIFIED_ALL_FEATURES].copy()
    X_te = test_df[UNIFIED_ALL_FEATURES].copy()
    for frame in (X_tr, X_val, X_tv, X_te):
        for col in UNIFIED_CATEGORICAL_FEATURES:
            frame[col] = frame[col].astype(str).astype(object)

    y_tr = train_df[TARGET_CLF_DIR_COL].astype(str).to_numpy(dtype=object)
    y_val = val_df[TARGET_CLF_DIR_COL].astype(str).to_numpy(dtype=object)
    y_tv = tv_df[TARGET_CLF_DIR_COL].astype(str).to_numpy(dtype=object)
    y_te = test_df[TARGET_CLF_DIR_COL].astype(str).to_numpy(dtype=object)

    classes = ["Normal", "Surge", "Crash"]

    def make_pre():
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), UNIFIED_CATEGORICAL_FEATURES),
            ("num", StandardScaler(), UNIFIED_NUMERIC_FEATURES),
        ])

    candidates = [
        ("Random Forest (Balanced)", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1), {"clf__n_estimators": [50, 100], "clf__max_depth": [8, 12], "clf__min_samples_leaf": [5, 10]}),
        ("Logistic Regression (Balanced)", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42), {"clf__C": [0.1, 1.0, 10.0]}),
        ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=42), {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3]}),
        ("Decision Tree (Balanced)", DecisionTreeClassifier(class_weight="balanced", random_state=42), {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [5, 10]}),
        ("AdaBoost Classifier", AdaBoostClassifier(random_state=42), {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.05, 0.1]}),
        ("Gaussian Naive Bayes", GaussianNB(), {}),
    ]

    purged_cv = DatePurgedWalkForwardCV(
        train_df["observation_date"],
        train_df["target_date_t_plus_1"],
        n_splits=5,
    )
    val_rows = []
    tuned: Dict[str, Any] = {}
    tuned_params: Dict[str, Dict[str, Any]] = {}

    for name, est, grid in candidates:
        pipe = Pipeline([("preprocessor", make_pre()), ("clf", est)])
        if grid:
            gs = GridSearchCV(pipe, grid, cv=purged_cv, scoring="f1_macro", n_jobs=1)
            gs.fit(X_tr, y_tr)
            best_p = gs.best_estimator_
            params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        else:
            best_p = pipe.fit(X_tr, y_tr)
            params = {}

        vp = best_p.predict(X_val)
        v_proba = best_p.predict_proba(X_val)
        clf_classes = list(best_p.named_steps["clf"].classes_)
        reorder = [clf_classes.index(c) for c in classes]
        v_proba_aligned = v_proba[:, reorder]
        vm = compute_multiclass_classification_metrics(y_val, vp, v_proba_aligned, classes, name)
        tuned[name] = clone(best_p.named_steps["clf"])
        tuned_params[name] = params
        val_rows.append(
            {
                "Model": name,
                "accuracy": vm["accuracy"],
                "macro_precision": vm["macro_precision"],
                "macro_recall": vm["macro_recall"],
                "macro_f1": vm["macro_f1"],
                "weighted_f1": vm["weighted_f1"],
                "roc_auc_ovr_macro": vm["roc_auc_ovr_macro"],
                "pr_auc_macro": vm["pr_auc_macro"],
                "any_spike_roc_auc": vm["any_spike_roc_auc"],
                "any_spike_pr_auc": vm["any_spike_pr_auc"],
                "Best_Params": json.dumps(params),
            }
        )
        print(f"  {name:<30} | Val_Macro_F1={vm['macro_f1']:.4f} | AnySpike_PR_AUC={vm['any_spike_pr_auc']:.4f}")

    val_board = pd.DataFrame(val_rows).sort_values(["macro_f1", "pr_auc_macro"], ascending=False).reset_index(drop=True)
    val_board.to_csv(REPORTS_DIR / "early_warning_classifier_evaluation.csv", index=False)

    champ_name = str(val_board.iloc[0]["Model"])
    champ_params = tuned_params[champ_name]
    champ_pipe = Pipeline([("preprocessor", make_pre()), ("clf", clone(tuned[champ_name]))])
    champ_pipe.fit(X_tv, y_tv)

    t_pred = champ_pipe.predict(X_te)
    t_proba = champ_pipe.predict_proba(X_te)
    clf_classes = list(champ_pipe.named_steps["clf"].classes_)
    reorder = [clf_classes.index(c) for c in classes]
    t_proba_aligned = t_proba[:, reorder]
    tm = compute_multiclass_classification_metrics(y_te, t_pred, t_proba_aligned, classes, champ_name)

    val_maj = float((val_df[TARGET_CLF_DIR_COL] == "Normal").mean())
    test_maj = float((test_df[TARGET_CLF_DIR_COL] == "Normal").mean())

    surge_prec = tm["per_class_report"]["Surge"]["precision"]
    crash_prec = tm["per_class_report"]["Crash"]["precision"]
    is_reliable = bool(tm["any_spike_pr_auc"] >= 0.65 and surge_prec >= 0.50 and crash_prec >= 0.50)

    status_note = (
        "Validated Early-Warning Classifier"
        if is_reliable
        else "Experimental / Low Reliability Classifier: Predicting exact single-session >10% surges/crashes from historical price & arrival lags alone yields weak precision due to unrecorded exogenous weather/policy shocks. Present strictly as experimental alongside regression alerts."
    )

    joblib.dump(champ_pipe, MODELS_DIR / "spike_classifier.joblib")
    meta = {
        "model_architecture": "Multi-Commodity Price Shock Classifier (3-Class: Normal / Surge / Crash)",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "selection_protocol": "DatePurgedWalkForwardCV on Train -> Validation-Only Selection -> Refit Train+Val -> Single Test Eval",
        "primary_selection_metric": "macro_f1",
        "champion_algorithm": champ_name,
        "champion_hyperparameters": champ_params,
        "classes": classes,
        "spike_definition": "Surge: next-session change > +10% | Crash: next-session change < -10% | Normal: within [-10%, +10%]",
        "features": UNIFIED_ALL_FEATURES,
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_file": "model_ready_unified_commodities_clean.csv",
        "class_distribution": {
            "Train": train_df[TARGET_CLF_DIR_COL].value_counts().to_dict(),
            "Validation": val_df[TARGET_CLF_DIR_COL].value_counts().to_dict(),
            "Test": test_df[TARGET_CLF_DIR_COL].value_counts().to_dict(),
        },
        "majority_class_baseline": {"class": "Normal", "val_accuracy": round(val_maj, 4), "test_accuracy": round(test_maj, 4)},
        "test_metrics": {
            "accuracy": tm["accuracy"],
            "macro_precision": tm["macro_precision"],
            "macro_recall": tm["macro_recall"],
            "macro_f1": tm["macro_f1"],
            "weighted_f1": tm["weighted_f1"],
            "roc_auc_ovr_macro": tm["roc_auc_ovr_macro"],
            "pr_auc_macro": tm["pr_auc_macro"],
            "any_spike_roc_auc": tm["any_spike_roc_auc"],
            "any_spike_pr_auc": tm["any_spike_pr_auc"],
        },
        "per_class_test_report": tm["per_class_report"],
        "confusion_matrix_labels": classes,
        "confusion_matrix": tm["confusion_matrix"],
        "is_reliable_early_warning": is_reliable,
        "reliability_status": status_note,
        "validation_leaderboard": val_board.to_dict(orient="records"),
    }
    with open(MODELS_DIR / "spike_classifier_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Saved spike_classifier.joblib ({champ_name}) | Test Acc={tm['accuracy']:.4f} "
        f"(Maj Base={test_maj:.4f}), Any-Spike PR-AUC={tm['any_spike_pr_auc']:.4f}, Reliable={is_reliable}"
    )


def plot_regression_comparison(reg_df, val_df, test_preds, val_preds, data, commodity_name, market_name):
    test_dates = data["test_dates"]
    y_te = data["y_test_reg"]
    best_name = str(reg_df[reg_df["Selected_By_Validation"]].iloc[0]["Model"])
    base_name = "Naive Persistence Baseline (P_t)"

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(test_dates, y_te, label="Actual Next Session", color="black", linewidth=2.2)
    ax.plot(test_dates, test_preds[best_name], "--", label=f"Champion: {best_name} (RMSE={reg_df.loc[reg_df['Model']==best_name, 'RMSE'].iloc[0]})", color="#1f77b4", linewidth=1.7)
    ax.plot(test_dates, test_preds[base_name], ":", label=f"Persistence Baseline (RMSE={reg_df.loc[reg_df['Model']==base_name, 'RMSE'].iloc[0]})", color="#d62728", linewidth=1.3)
    ax.set_title(f"Untouched Test: Actual vs Validation-Selected Champion ({commodity_name} @ {market_name})", fontweight="bold")
    ax.set_xlabel("Observation Date")
    ax.set_ylabel("Modal Price (Rs/Quintal)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.xticks(rotation=35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_actual_vs_predicted_test.png", dpi=300)
    plt.close()

    # Plot Validation leaderboard across all 14 candidates cleanly (without touching Test)
    val_s = val_df.sort_values("Val_RMSE")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.barh(val_s["Model"], val_s["Val_RMSE"], color="steelblue", alpha=0.85, label="Val RMSE")
    ax1.barh(val_s["Model"], val_s["Val_MAE"], color="orange", alpha=0.75, label="Val MAE")
    ax1.invert_yaxis()
    ax1.set_title("Validation Error Comparison (All Candidates, Rs/Qtl)")
    ax1.legend()
    ax2.barh(val_s["Model"], val_s["Val_R2"], color="seagreen", alpha=0.85)
    ax2.invert_yaxis()
    ax2.set_title("Validation R2 Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_model_metrics_comparison.png", dpi=300)
    plt.close()

    resid = y_te - test_preds[best_name]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.kdeplot(resid, fill=True, ax=ax, label=f"{best_name} (mean={resid.mean():.1f}, std={resid.std():.1f})")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"Untouched Test Residual Distribution ({commodity_name})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_residual_distribution.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(test_preds[best_name], resid, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"Residuals vs Predicted ({commodity_name})", fontweight="bold")
    ax.set_xlabel("Predicted Price (Rs/Qtl)")
    ax.set_ylabel("Residual (Rs/Qtl)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_residuals_vs_predicted.png", dpi=300)
    plt.close()


def save_specialist_outputs(data, reg_df, val_df, winner_name, winner_pipe, gd_df, clf_test_df, clf_val_df, cluster_df, profile_df, commodity_name, market_name):
    safe_comm = commodity_name.lower().replace(" ", "_")

    model_comm = MODELS_DIR / f"price_prediction_model_{safe_comm}.joblib"
    joblib.dump(winner_pipe, model_comm)
    joblib.dump(winner_pipe, MODELS_DIR / "price_prediction_model.joblib")

    if hasattr(winner_pipe, "named_steps") and "scaler" in winner_pipe.named_steps:
        joblib.dump(winner_pipe.named_steps["scaler"], MODELS_DIR / f"preprocessing_pipeline_{safe_comm}.joblib")

    best_row = reg_df[reg_df["Model"] == winner_name].iloc[0].to_dict()
    base_row = reg_df[reg_df["Model"] == "Naive Persistence Baseline (P_t)"].iloc[0].to_dict()

    metadata = {
        "project": "AI-Based Food Inflation Prediction and Household Affordability Monitoring System",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "selection_protocol": "DatePurgedWalkForwardCV inside Train -> Validation Selection by Val_RMSE -> Refit Train+Val -> Single Evaluation of Champion vs Baseline on Untouched Test",
        "primary_selection_metric": "Val_RMSE",
        "model_name": winner_name,
        "artifact_file": model_comm.name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_source": f"AGMARKNET {market_name} APMC Mandi Daily Series",
        "commodity": commodity_name,
        "market": market_name,
        "district": market_name,
        "state": "Maharashtra",
        "target_variable": TARGET_REG_COL,
        "target_definition": "Modal wholesale price on the next observed active APMC trading session (Rs/Quintal)",
        "prediction_horizon": "Next observed active trading session (t+1, not guaranteed next calendar day)",
        "feature_names": FEATURE_COLS,
        "feature_count": len(FEATURE_COLS),
        "training_samples": len(data["X_train"]),
        "validation_samples": len(data["X_val"]),
        "train_val_samples": len(data["X_train_val"]),
        "test_samples": len(data["X_test"]),
        "test_evaluation_metrics": best_row,
        "persistence_baseline_test_metrics": base_row,
        "improvement_over_baseline_test": {
            "rmse_reduction_rs_per_qtl": round(base_row["RMSE"] - best_row["RMSE"], 2),
            "mae_reduction_rs_per_qtl": round(base_row["MAE"] - best_row["MAE"], 2),
            "beats_baseline_rmse": bool(best_row["RMSE"] < base_row["RMSE"]),
        },
        "validation_leaderboard": val_df.to_dict(orient="records"),
    }

    for meta_file in [MODELS_DIR / "model_metadata.json", MODELS_DIR / f"model_metadata_{safe_comm}.json"]:
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    reg_df.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)
    reg_df.to_csv(REPORTS_DIR / f"model_comparison_metrics_{safe_comm}.csv", index=False)
    val_df.to_csv(REPORTS_DIR / f"validation_leaderboard_{safe_comm}.csv", index=False)
    clf_test_df.to_csv(REPORTS_DIR / "classification_comparison.csv", index=False)
    clf_test_df.to_csv(REPORTS_DIR / f"classification_comparison_{safe_comm}.csv", index=False)
    clf_val_df.to_csv(REPORTS_DIR / f"classification_validation_leaderboard_{safe_comm}.csv", index=False)
    cluster_df.to_csv(REPORTS_DIR / "clustering_comparison.csv", index=False)
    cluster_df.to_csv(REPORTS_DIR / f"clustering_comparison_{safe_comm}.csv", index=False)
    profile_df.to_csv(REPORTS_DIR / "clustering_profiles.csv", index=False)
    profile_df.to_csv(REPORTS_DIR / f"clustering_profiles_{safe_comm}.csv", index=False)
    gd_df.to_csv(REPORTS_DIR / "gradient_descent_comparison.csv", index=False)
    gd_df.to_csv(REPORTS_DIR / f"gradient_descent_comparison_{safe_comm}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Train leak-free syllabus ML models.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--commodity", type=str, default="Tomato")
    parser.add_argument("--market", type=str, default="Pune")
    parser.add_argument("--train-global-classifier", action="store_true")
    args = parser.parse_args()

    if args.train_global_classifier:
        train_global_early_warning_classifier()
        return

    safe_comm = args.commodity.lower().replace(" ", "_")
    safe_mkt = args.market.lower().replace(" ", "_")
    ds_path = args.dataset or str(PROCESSED_DIR / f"model_ready_{safe_mkt}_{safe_comm}.csv")
    if not Path(ds_path).exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}. Run py -3.12 scripts/build_features.py --all")

    data = load_series_data(ds_path)
    reg_df, val_df, winner_name, winner_pipe, test_preds, val_preds = run_regression_models(data, args.commodity)
    gd_df = run_gradient_descent(data)
    clf_test_df, clf_val_df = run_classification_models(data)
    cluster_df, profile_df = run_clustering_pca(data, args.commodity)
    plot_regression_comparison(reg_df, val_df, test_preds, val_preds, data, args.commodity, args.market)
    save_specialist_outputs(data, reg_df, val_df, winner_name, winner_pipe, gd_df, clf_test_df, clf_val_df, cluster_df, profile_df, args.commodity, args.market)
    print(f"\nCompleted leak-free pipeline for {args.commodity} @ {args.market}. Champion (Validation-selected): {winner_name}")


if __name__ == "__main__":
    main()
