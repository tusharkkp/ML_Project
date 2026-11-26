"""
Comprehensive ML Training & Evaluation Pipeline — All Syllabus Algorithms
==========================================================================
Course: B.Tech Third-Year AIML Practical Examination (30 Marks)
Covers: Components F (Training & Tuning), G (Evaluation & Comparative Analysis), H (Deployment)

Implements ALL algorithms from ML_algos.txt:

REGRESSION (Module 1):
  1. Simple Linear Regression
  2. Multiple Linear Regression
  3. Polynomial Regression (degree 2)
  4. Ridge Regression (L2)
  5. Lasso Regression (L1)

OPTIMIZATION (Module 2):
  6. Batch Gradient Descent (via SGDRegressor with large batch)
  7. Stochastic Gradient Descent (SGDRegressor)
  8. Mini-batch Gradient Descent (via partial_fit)

CLASSIFICATION (Module 3 — Price Spike Detection):
  9. Logistic Regression
  10. Gaussian Naive Bayes
  11. K-Nearest Neighbors (KNN)
  12. Decision Tree (CART)
  13. Support Vector Machine (SVM)
  14. Bagging Classifier
  15. AdaBoost Classifier
  16. Stacking Classifier
  17. Voting Classifier

CLUSTERING & UNSUPERVISED (Module 4):
  18. K-Means Clustering
  19. Hierarchical Clustering (Agglomerative)
  20. Density-Based Clustering (DBSCAN)
  21. PCA (Principal Component Analysis)

Usage:
  py -3.12 scripts/train_all_models.py                          # Tomato Pune (default)
  py -3.12 scripts/train_all_models.py --commodity Onion --market Pune
  py -3.12 scripts/train_all_models.py --dataset data/processed/model_ready_pune_onion.csv
"""

import os
import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso,
    SGDRegressor, LogisticRegression, SGDClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    BaggingRegressor, BaggingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
    GradientBoostingRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    silhouette_score, davies_bouldin_score
)
import joblib

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"


# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================
FEATURE_COLS = [
    'price_lag_1', 'price_lag_3', 'price_lag_7',
    'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
    'price_momentum_7', 'daily_spread', 'arrival_tonnes', 'arrival_lag_1',
    'rolling_arrival_mean_7', 'month', 'day_of_week', 'is_monsoon_season'
]

TARGET_REG = 'target_price_t_plus_1'
TARGET_CLF = 'is_price_spike'


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Compute regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {
        "Model": model_name, "MAE": round(mae, 2), "MSE": round(mse, 2),
        "RMSE": round(rmse, 2), "R2": round(r2, 4), "MAPE (%)": round(mape, 2)
    }


def evaluate_classification(y_true, y_pred, y_proba, model_name="Model"):
    """Compute classification evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0
    except:
        auc = 0.0
    return {
        "Model": model_name, "Accuracy": round(acc, 4),
        "Precision": round(prec, 4), "Recall": round(rec, 4),
        "F1-Score": round(f1, 4), "ROC-AUC": round(auc, 4)
    }


def load_data(dataset_path: str):
    """Load model-ready dataset and split into train/val/test."""
    df = pd.read_csv(dataset_path)
    
    train_df = df[df['split_tag'] == 'Train'].copy()
    val_df = df[df['split_tag'] == 'Validation'].copy()
    test_df = df[df['split_tag'] == 'Test'].copy()
    
    X_train = train_df[FEATURE_COLS].values
    y_train_reg = train_df[TARGET_REG].values
    y_train_clf = train_df[TARGET_CLF].values
    
    X_val = val_df[FEATURE_COLS].values
    y_val_reg = val_df[TARGET_REG].values
    y_val_clf = val_df[TARGET_CLF].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test_reg = test_df[TARGET_REG].values
    y_test_clf = test_df[TARGET_CLF].values
    
    # Combined train+val for final model fitting
    X_train_val = np.vstack([X_train, X_val])
    y_train_val_reg = np.concatenate([y_train_reg, y_val_reg])
    y_train_val_clf = np.concatenate([y_train_clf, y_val_clf])
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    X_train_val_s = scaler.fit_transform(X_train_val)
    X_test_s_final = scaler.transform(X_test)
    
    return {
        'df': df, 'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_s': X_train_s, 'X_val_s': X_val_s, 'X_test_s': X_test_s,
        'X_train_val': X_train_val, 'X_train_val_s': X_train_val_s,
        'X_test_s_final': X_test_s_final,
        'y_train_reg': y_train_reg, 'y_val_reg': y_val_reg, 'y_test_reg': y_test_reg,
        'y_train_val_reg': y_train_val_reg,
        'y_train_clf': y_train_clf, 'y_val_clf': y_val_clf, 'y_test_clf': y_test_clf,
        'y_train_val_clf': y_train_val_clf,
        'scaler': scaler,
        'test_dates': test_df['observation_date'].values
    }


# ==============================================================================
# MODULE 1: REGRESSION MODELS
# ==============================================================================
def run_regression_models(data: dict):
    """Train and evaluate all syllabus regression algorithms."""
    print("\n" + "=" * 80)
    print("MODULE 1: REGRESSION MODELS (Continuous Price Forecasting)")
    print("=" * 80)
    
    X_tr = data['X_train_s']
    y_tr = data['y_train_reg']
    X_tv = data['X_train_val_s']
    y_tv = data['y_train_val_reg']
    X_te = data['X_test_s_final']
    y_te = data['y_test_reg']
    
    results = []
    models_dict = {}
    predictions = {}
    
    # 0. Naive Persistence Baseline
    print("\n--- Naive Persistence Baseline ---")
    y_pred_baseline = data['X_test'][:, FEATURE_COLS.index('price_lag_1')]
    res = evaluate_regression(y_te, y_pred_baseline, "Naive Persistence Baseline")
    results.append(res)
    predictions['Baseline'] = y_pred_baseline
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # 1. Simple Linear Regression (univariate: price_lag_1 only)
    print("\n--- 1. Simple Linear Regression ---")
    slr = LinearRegression()
    idx_lag1 = FEATURE_COLS.index('price_lag_1')
    # Use UNSCALED data for Simple LR (interpretable equation)
    X_tv_raw = data['X_train_val']
    X_te_raw = data['X_test']
    slr.fit(X_tv_raw[:, idx_lag1:idx_lag1+1], y_tv)
    y_pred_slr = slr.predict(X_te_raw[:, idx_lag1:idx_lag1+1])
    res = evaluate_regression(y_te, y_pred_slr, "Simple Linear Regression")
    results.append(res)
    models_dict['SLR'] = slr
    predictions['SLR'] = y_pred_slr
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    print(f"  Equation: price_t+1 = {slr.coef_[0]:.4f} * price_lag_1 + {slr.intercept_:.4f}")
    
    # 2. Multiple Linear Regression
    print("\n--- 2. Multiple Linear Regression ---")
    mlr = LinearRegression()
    mlr.fit(X_tv, y_tv)
    y_pred_mlr = mlr.predict(X_te)
    res = evaluate_regression(y_te, y_pred_mlr, "Multiple Linear Regression")
    results.append(res)
    models_dict['MLR'] = mlr
    predictions['MLR'] = y_pred_mlr
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # 3. Polynomial Regression (degree 2)
    print("\n--- 3. Polynomial Regression (Degree 2) ---")
    poly_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
        ('ridge', Ridge(alpha=10.0, random_state=42))
    ])
    poly_pipe.fit(X_tv, y_tv)
    y_pred_poly = poly_pipe.predict(X_te)
    res = evaluate_regression(y_te, y_pred_poly, "Polynomial Regression (deg=2)")
    results.append(res)
    models_dict['Poly'] = poly_pipe
    predictions['Poly'] = y_pred_poly
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # 4. Ridge Regression (L2 Regularization)
    print("\n--- 4. Ridge Regression (L2) ---")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_tv, y_tv)
    y_pred_ridge = ridge.predict(X_te)
    res = evaluate_regression(y_te, y_pred_ridge, "Ridge Regression (L2)")
    results.append(res)
    models_dict['Ridge'] = ridge
    predictions['Ridge'] = y_pred_ridge
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # 5. Lasso Regression (L1 Regularization)
    print("\n--- 5. Lasso Regression (L1) ---")
    lasso = Lasso(alpha=1.0, random_state=42, max_iter=5000)
    lasso.fit(X_tv, y_tv)
    y_pred_lasso = lasso.predict(X_te)
    res = evaluate_regression(y_te, y_pred_lasso, "Lasso Regression (L1)")
    results.append(res)
    models_dict['Lasso'] = lasso
    # Print which features Lasso zeroed out
    zero_feats = [f for f, c in zip(FEATURE_COLS, lasso.coef_) if abs(c) < 1e-6]
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    print(f"  Lasso zeroed features (automated selection): {zero_feats if zero_feats else 'None — all features retained'}")
    predictions['Lasso'] = y_pred_lasso
    
    # 6. Decision Tree Regressor (CART)
    print("\n--- 6. Decision Tree Regressor (CART) ---")
    dt_reg = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    dt_reg.fit(X_tv, y_tv)
    y_pred_dt = dt_reg.predict(X_te)
    res = evaluate_regression(y_te, y_pred_dt, "Decision Tree Regressor")
    results.append(res)
    models_dict['DT'] = dt_reg
    predictions['DT'] = y_pred_dt
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # 7. KNN Regressor
    print("\n--- 7. KNN Regressor ---")
    knn_reg = KNeighborsRegressor(n_neighbors=7)
    knn_reg.fit(X_tv, y_tv)
    y_pred_knn = knn_reg.predict(X_te)
    res = evaluate_regression(y_te, y_pred_knn, "KNN Regressor (k=7)")
    results.append(res)
    models_dict['KNN_Reg'] = knn_reg
    predictions['KNN'] = y_pred_knn
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # 8. SVR (Support Vector Regressor)
    print("\n--- 8. SVR (Support Vector Regression) ---")
    svr = SVR(kernel='rbf', C=100.0, epsilon=10.0)
    svr.fit(X_tv, y_tv)
    y_pred_svr = svr.predict(X_te)
    res = evaluate_regression(y_te, y_pred_svr, "SVR (RBF Kernel)")
    results.append(res)
    models_dict['SVR'] = svr
    predictions['SVR'] = y_pred_svr
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # --- ENSEMBLE REGRESSION ---
    print("\n--- 9. Bagging Regressor ---")
    bag_reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=6),
        n_estimators=50, random_state=42, n_jobs=-1
    )
    bag_reg.fit(X_tv, y_tv)
    y_pred_bag = bag_reg.predict(X_te)
    res = evaluate_regression(y_te, y_pred_bag, "Bagging Regressor")
    results.append(res)
    models_dict['Bagging'] = bag_reg
    predictions['Bagging'] = y_pred_bag
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    print("\n--- 10. AdaBoost Regressor ---")
    ada_reg = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=100, learning_rate=0.1, random_state=42
    )
    ada_reg.fit(X_tv, y_tv)
    y_pred_ada = ada_reg.predict(X_te)
    res = evaluate_regression(y_te, y_pred_ada, "AdaBoost Regressor")
    results.append(res)
    models_dict['AdaBoost'] = ada_reg
    predictions['AdaBoost'] = y_pred_ada
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    print("\n--- 11. Stacking Regressor ---")
    stack_reg = StackingRegressor(
        estimators=[
            ('ridge', Ridge(alpha=1.0)),
            ('dt', DecisionTreeRegressor(max_depth=5)),
            ('knn', KNeighborsRegressor(n_neighbors=7))
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1
    )
    stack_reg.fit(X_tv, y_tv)
    y_pred_stack = stack_reg.predict(X_te)
    res = evaluate_regression(y_te, y_pred_stack, "Stacking Regressor")
    results.append(res)
    models_dict['Stacking'] = stack_reg
    predictions['Stacking'] = y_pred_stack
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    print("\n--- 12. Voting Regressor ---")
    vote_reg = VotingRegressor(
        estimators=[
            ('mlr', LinearRegression()),
            ('ridge', Ridge(alpha=1.0)),
            ('dt', DecisionTreeRegressor(max_depth=5))
        ], n_jobs=-1
    )
    vote_reg.fit(X_tv, y_tv)
    y_pred_vote = vote_reg.predict(X_te)
    res = evaluate_regression(y_te, y_pred_vote, "Voting Regressor")
    results.append(res)
    models_dict['Voting'] = vote_reg
    predictions['Voting'] = y_pred_vote
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    reg_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("REGRESSION MODEL COMPARISON (Unseen Test Set)")
    print("=" * 80)
    print(reg_df.to_string(index=False))
    
    return reg_df, models_dict, predictions


# ==============================================================================
# MODULE 2: GRADIENT DESCENT OPTIMIZATION VARIANTS
# ==============================================================================
def run_gradient_descent(data: dict):
    """Demonstrate Batch, Stochastic, and Mini-batch Gradient Descent convergence."""
    print("\n" + "=" * 80)
    print("MODULE 2: GRADIENT DESCENT OPTIMIZATION ALGORITHMS")
    print("=" * 80)
    
    X_tv = data['X_train_val_s']
    y_tv = data['y_train_val_reg']
    X_te = data['X_test_s_final']
    y_te = data['y_test_reg']
    
    results = []
    convergence_data = {}
    
    # 1. Batch Gradient Descent (Full dataset per epoch)
    print("\n--- Batch Gradient Descent ---")
    bgd = SGDRegressor(
        loss='squared_error', penalty='l2', alpha=0.0001,
        learning_rate='constant', eta0=0.001,
        max_iter=1, warm_start=True, random_state=42
    )
    bgd_losses = []
    for epoch in range(500):
        bgd.partial_fit(X_tv, y_tv)
        pred = bgd.predict(X_tv)
        loss = np.mean((y_tv - pred) ** 2)
        bgd_losses.append(loss)
    y_pred_bgd = bgd.predict(X_te)
    res = evaluate_regression(y_te, y_pred_bgd, "Batch Gradient Descent")
    results.append(res)
    convergence_data['Batch GD'] = bgd_losses
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}, Final Loss={bgd_losses[-1]:.2f}")
    
    # 2. Stochastic Gradient Descent (1 sample per update)
    print("\n--- Stochastic Gradient Descent ---")
    sgd = SGDRegressor(
        loss='squared_error', penalty='l2', alpha=0.0001,
        learning_rate='invscaling', eta0=0.01, power_t=0.25,
        max_iter=500, random_state=42
    )
    sgd.fit(X_tv, y_tv)
    y_pred_sgd = sgd.predict(X_te)
    res = evaluate_regression(y_te, y_pred_sgd, "Stochastic Gradient Descent")
    results.append(res)
    # Track loss per epoch for SGD
    sgd_losses = []
    sgd2 = SGDRegressor(
        loss='squared_error', penalty='l2', alpha=0.0001,
        learning_rate='invscaling', eta0=0.01, power_t=0.25,
        max_iter=1, warm_start=True, random_state=42
    )
    for epoch in range(500):
        sgd2.partial_fit(X_tv, y_tv)
        pred = sgd2.predict(X_tv)
        loss = np.mean((y_tv - pred) ** 2)
        sgd_losses.append(loss)
    convergence_data['SGD'] = sgd_losses
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    # 3. Mini-batch Gradient Descent (batch_size=32)
    print("\n--- Mini-batch Gradient Descent (batch_size=32) ---")
    mbgd = SGDRegressor(
        loss='squared_error', penalty='l2', alpha=0.0001,
        learning_rate='invscaling', eta0=0.01, power_t=0.25,
        max_iter=1, warm_start=True, random_state=42
    )
    batch_size = 32
    mbgd_losses = []
    for epoch in range(500):
        indices = np.random.RandomState(epoch).permutation(len(X_tv))
        for start in range(0, len(X_tv), batch_size):
            end = min(start + batch_size, len(X_tv))
            batch_idx = indices[start:end]
            mbgd.partial_fit(X_tv[batch_idx], y_tv[batch_idx])
        pred = mbgd.predict(X_tv)
        loss = np.mean((y_tv - pred) ** 2)
        mbgd_losses.append(loss)
    y_pred_mbgd = mbgd.predict(X_te)
    res = evaluate_regression(y_te, y_pred_mbgd, "Mini-batch GD (batch=32)")
    results.append(res)
    convergence_data['Mini-batch GD'] = mbgd_losses
    print(f"  Test: RMSE={res['RMSE']}, R2={res['R2']}")
    
    gd_df = pd.DataFrame(results)
    print("\n--- Gradient Descent Comparison ---")
    print(gd_df.to_string(index=False))
    
    # Plot convergence curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, losses in convergence_data.items():
        ax.plot(losses[:200], label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (Training Loss)")
    ax.set_title("Gradient Descent Convergence Comparison (First 200 Epochs)", fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    gd_fig_path = FIG_DIR / "10_gradient_descent_convergence.png"
    plt.savefig(gd_fig_path, dpi=300)
    plt.close()
    print(f"  Saved: {gd_fig_path}")
    
    return gd_df, convergence_data


# ==============================================================================
# MODULE 3: CLASSIFICATION MODELS (Price Spike Detection)
# ==============================================================================
def run_classification_models(data: dict):
    """Train and evaluate all syllabus classification algorithms on price spike prediction."""
    print("\n" + "=" * 80)
    print("MODULE 3: CLASSIFICATION MODELS (Food Price Spike Detection)")
    print("=" * 80)
    
    X_tv = data['X_train_val_s']
    y_tv = data['y_train_val_clf']
    X_te = data['X_test_s_final']
    y_te = data['y_test_clf']
    
    # Check class balance
    unique, counts = np.unique(y_tv, return_counts=True)
    print(f"  Training class distribution: {dict(zip(unique, counts))}")
    unique_te, counts_te = np.unique(y_te, return_counts=True)
    print(f"  Test class distribution: {dict(zip(unique_te, counts_te))}")
    
    results = []
    roc_data = {}
    cm_data = {}
    
    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree (CART)", DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')),
        ("SVM (RBF Kernel)", SVC(kernel='rbf', C=10.0, probability=True, random_state=42, class_weight='balanced')),
        ("Bagging Classifier", BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=4),
            n_estimators=50, random_state=42, n_jobs=-1
        )),
        ("AdaBoost Classifier", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100, learning_rate=0.1, random_state=42
        )),
        ("Stacking Classifier", StackingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('dt', DecisionTreeClassifier(max_depth=4, class_weight='balanced'))
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3, n_jobs=-1
        )),
        ("Voting Classifier", VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('svm', SVC(kernel='rbf', C=10.0, probability=True, class_weight='balanced'))
            ],
            voting='soft', n_jobs=-1
        )),
    ]
    
    for name, clf in classifiers:
        print(f"\n--- {name} ---")
        clf.fit(X_tv, y_tv)
        y_pred = clf.predict(X_te)
        
        try:
            y_proba = clf.predict_proba(X_te)[:, 1]
        except:
            y_proba = None
        
        res = evaluate_classification(y_te, y_pred, y_proba, name)
        results.append(res)
        cm_data[name] = confusion_matrix(y_te, y_pred)
        
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            roc_data[name] = (fpr, tpr, res['ROC-AUC'])
        
        print(f"  Accuracy={res['Accuracy']}, F1={res['F1-Score']}, ROC-AUC={res['ROC-AUC']}")
    
    clf_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("CLASSIFICATION MODEL COMPARISON (Price Spike Detection on Unseen Test Set)")
    print("=" * 80)
    print(clf_df.to_string(index=False))
    
    # ROC Curve Plot
    if roc_data:
        fig, ax = plt.subplots(figsize=(9, 7))
        for name, (fpr, tpr, auc_val) in roc_data.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves: Price Spike Classification Models", fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        roc_path = FIG_DIR / "11_roc_curves_classification.png"
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"  Saved: {roc_path}")
    
    # Confusion Matrix Heatmaps (top 4 models)
    top_models = clf_df.nlargest(4, 'F1-Score')['Model'].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, name in enumerate(top_models):
        ax = axes[idx // 2][idx % 2]
        cm = cm_data[name]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Spike'], yticklabels=['Normal', 'Spike'])
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices: Top 4 Classifiers (Price Spike Detection)", fontweight='bold', y=1.01)
    plt.tight_layout()
    cm_path = FIG_DIR / "12_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cm_path}")
    
    return clf_df, cm_data, roc_data


# ==============================================================================
# MODULE 4: CLUSTERING & PCA (Unsupervised Learning)
# ==============================================================================
def run_clustering_pca(data: dict, commodity_name: str):
    """Perform K-Means, Hierarchical, DBSCAN clustering and PCA on market trading days."""
    print("\n" + "=" * 80)
    print("MODULE 4: CLUSTERING & DIMENSIONALITY REDUCTION (Unsupervised Learning)")
    print("=" * 80)
    
    # Use all data (train+val+test) for unsupervised learning
    X_all = np.vstack([data['X_train_val_s'], data['X_test_s_final']])
    
    results = []
    
    # 1. K-Means Clustering
    print("\n--- K-Means Clustering ---")
    # Elbow Method: test k=2..8
    inertias = []
    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_all)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_all, labels)
        sil_scores.append(sil)
    
    best_k = range(2, 9)[np.argmax(sil_scores)]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_all)
    km_sil = silhouette_score(X_all, km_labels)
    km_db = davies_bouldin_score(X_all, km_labels)
    results.append({"Model": f"K-Means (k={best_k})", "Silhouette": round(km_sil, 4), "Davies-Bouldin": round(km_db, 4)})
    print(f"  Best k={best_k}, Silhouette={km_sil:.4f}, Davies-Bouldin={km_db:.4f}")
    
    # Elbow + Silhouette Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(range(2, 9), inertias, 'bo-', linewidth=2)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.set_title("K-Means Elbow Method", fontweight='bold')
    ax1.axvline(x=best_k, color='red', linestyle='--', label=f'Optimal k={best_k}')
    ax1.legend()
    
    ax2.plot(range(2, 9), sil_scores, 'gs-', linewidth=2)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis", fontweight='bold')
    ax2.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    ax2.legend()
    plt.tight_layout()
    elbow_path = FIG_DIR / "13_kmeans_elbow_silhouette.png"
    plt.savefig(elbow_path, dpi=300)
    plt.close()
    print(f"  Saved: {elbow_path}")
    
    # 2. Hierarchical Clustering (Agglomerative)
    print("\n--- Hierarchical Clustering (Agglomerative) ---")
    agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    agg_labels = agg.fit_predict(X_all)
    agg_sil = silhouette_score(X_all, agg_labels)
    agg_db = davies_bouldin_score(X_all, agg_labels)
    results.append({"Model": f"Hierarchical (Ward, k={best_k})", "Silhouette": round(agg_sil, 4), "Davies-Bouldin": round(agg_db, 4)})
    print(f"  Silhouette={agg_sil:.4f}, Davies-Bouldin={agg_db:.4f}")
    
    # Dendrogram (on subsample for readability)
    from scipy.cluster.hierarchy import dendrogram, linkage
    subsample_idx = np.random.RandomState(42).choice(len(X_all), size=min(200, len(X_all)), replace=False)
    X_sub = X_all[subsample_idx]
    Z = linkage(X_sub, method='ward')
    
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode='level', p=5, ax=ax, color_threshold=Z[-best_k+1, 2])
    ax.set_title(f"Hierarchical Clustering Dendrogram ({commodity_name} Market Days)", fontweight='bold')
    ax.set_xlabel("Sample Index (Truncated)")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    dendro_path = FIG_DIR / "14_hierarchical_dendrogram.png"
    plt.savefig(dendro_path, dpi=300)
    plt.close()
    print(f"  Saved: {dendro_path}")
    
    # 3. DBSCAN
    print("\n--- Density-Based Clustering (DBSCAN) ---")
    dbscan = DBSCAN(eps=2.5, min_samples=10)
    db_labels = dbscan.fit_predict(X_all)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    if n_clusters_db > 1:
        # Exclude noise points for silhouette
        mask = db_labels != -1
        db_sil = silhouette_score(X_all[mask], db_labels[mask]) if mask.sum() > 1 else 0
        db_db_score = davies_bouldin_score(X_all[mask], db_labels[mask]) if mask.sum() > 1 else 0
    else:
        db_sil = 0
        db_db_score = 0
    results.append({"Model": f"DBSCAN (clusters={n_clusters_db})", "Silhouette": round(db_sil, 4), "Davies-Bouldin": round(db_db_score, 4)})
    print(f"  Clusters found: {n_clusters_db}, Noise points: {n_noise}/{len(X_all)}")
    print(f"  Silhouette={db_sil:.4f}, Davies-Bouldin={db_db_score:.4f}")
    
    # 4. PCA (Principal Component Analysis)
    print("\n--- PCA (Principal Component Analysis) ---")
    pca = PCA(n_components=min(15, X_all.shape[1]))
    X_pca = pca.fit_transform(X_all)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cum_var >= 0.95) + 1
    print(f"  Components for 95% variance: {n_components_95} (out of {X_all.shape[1]})")
    print(f"  Explained variance ratios: {[round(v, 4) for v in pca.explained_variance_ratio_[:5]]}")
    
    # Scree Plot + 2D PCA Scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Scree plot
    ax1.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_,
            alpha=0.7, color='steelblue', label='Individual')
    ax1.plot(range(1, len(cum_var)+1), cum_var, 'ro-', linewidth=2, label='Cumulative')
    ax1.axhline(y=0.95, color='green', linestyle='--', label='95% Threshold')
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA Scree Plot (Explained Variance)", fontweight='bold')
    ax1.legend()
    
    # 2D scatter (colored by K-Means clusters)
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='viridis',
                          alpha=0.6, s=15, edgecolors='k', linewidth=0.3)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax2.set_title(f"PCA 2D Projection (K-Means k={best_k} Clusters)", fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    plt.tight_layout()
    pca_path = FIG_DIR / "15_pca_scree_and_clusters.png"
    plt.savefig(pca_path, dpi=300)
    plt.close()
    print(f"  Saved: {pca_path}")
    
    cluster_df = pd.DataFrame(results)
    print("\n--- Clustering Comparison ---")
    print(cluster_df.to_string(index=False))
    
    return cluster_df, pca, kmeans


# ==============================================================================
# VISUALIZATION: Regression model comparison
# ==============================================================================
def plot_regression_comparison(reg_df, predictions, data, commodity_name, market_name):
    """Generate comprehensive regression comparison visualizations."""
    test_dates = data['test_dates']
    y_te = data['y_test_reg']
    
    # Figure: Actual vs Predicted (top 5 models)
    top5 = reg_df.nsmallest(5, 'RMSE')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_dates, y_te, label="Actual Observed Price", color="black", linewidth=2.5, marker="o", markersize=3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (_, row) in enumerate(top5.iterrows()):
        model_key = None
        for k, v in predictions.items():
            if row['Model'].startswith(k) or k in row['Model']:
                model_key = k
                break
        # Fallback: try matching
        if model_key is None:
            for k in predictions:
                if k.lower() in row['Model'].lower() or row['Model'].lower().startswith(k.lower()):
                    model_key = k
                    break
        if model_key and model_key in predictions:
            ax.plot(test_dates, predictions[model_key],
                    label=f"{row['Model']} (RMSE={row['RMSE']})",
                    color=colors[idx % len(colors)], linestyle='--', linewidth=1.5)
    
    ax.set_title(f"Actual vs. Predicted Wholesale Prices ({commodity_name} @ {market_name} APMC)", fontweight='bold')
    ax.set_xlabel("Test Set Date")
    ax.set_ylabel("Modal Price (Rs./Quintal)")
    ax.legend(loc='best', fontsize=8)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.tight_layout()
    fig_path = FIG_DIR / "06_actual_vs_predicted_test.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")
    
    # Figure: Regression model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE / RMSE comparison
    reg_sorted = reg_df.sort_values('RMSE')
    x_pos = range(len(reg_sorted))
    axes[0].barh(reg_sorted['Model'], reg_sorted['RMSE'], color='steelblue', alpha=0.8, label='RMSE')
    axes[0].barh(reg_sorted['Model'], reg_sorted['MAE'], color='orange', alpha=0.7, label='MAE')
    axes[0].set_xlabel("Error (Rs/Quintal) — Lower is Better")
    axes[0].set_title("Regression Model Error Comparison", fontweight='bold')
    axes[0].legend()
    axes[0].invert_yaxis()
    
    # R2 comparison
    axes[1].barh(reg_sorted['Model'], reg_sorted['R2'], color='seagreen', alpha=0.8)
    axes[1].set_xlabel("R² Score — Higher is Better")
    axes[1].set_title("R² Score Comparison", fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    bar_path = FIG_DIR / "09_model_metrics_comparison.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}")
    
    # Figure: Residual distribution for top 2 models
    top2 = reg_df.nsmallest(2, 'RMSE')
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in top2.iterrows():
        for k, preds in predictions.items():
            if k in row['Model'] or row['Model'].startswith(k):
                residuals = y_te - preds
                sns.kdeplot(residuals, label=f"{row['Model']} (std={residuals.std():.1f})",
                            fill=True, alpha=0.3, ax=ax)
                break
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    ax.set_title("Residual Distribution (Top Models on Test Set)", fontweight='bold')
    ax.set_xlabel("Residual Error (Rs/Quintal)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    res_path = FIG_DIR / "07_residual_distribution.png"
    plt.savefig(res_path, dpi=300)
    plt.close()
    print(f"Saved: {res_path}")


# ==============================================================================
# SAVE MODEL ARTIFACTS
# ==============================================================================
def save_winning_model(data, reg_df, models_dict, commodity_name, market_name):
    """Serialize the best performing regression model."""
    best_row = reg_df.loc[reg_df[reg_df['Model'] != 'Naive Persistence Baseline']['RMSE'].idxmin()]
    best_name = best_row['Model']
    
    # Find model key
    best_model = None
    for k, m in models_dict.items():
        if k in best_name or best_name.startswith(k):
            best_model = m
            break
    if best_model is None:
        best_model = list(models_dict.values())[0]
        best_name = list(models_dict.keys())[0]
    
    # Create an end-to-end production pipeline with scaler so inference handles raw features
    if not isinstance(best_model, Pipeline):
        production_pipeline = Pipeline([
            ("scaler", data['scaler']),
            ("regressor", best_model)
        ])
    else:
        production_pipeline = best_model

    # Save model (both general and commodity-specific)
    safe_comm = commodity_name.lower().replace(' ', '_')
    model_path = MODELS_DIR / "price_prediction_model.joblib"
    model_comm_path = MODELS_DIR / f"price_prediction_model_{safe_comm}.joblib"
    joblib.dump(production_pipeline, model_path)
    joblib.dump(production_pipeline, model_comm_path)
    print(f"Saved winning model pipeline: {model_path} and {model_comm_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    scaler_comm_path = MODELS_DIR / f"preprocessing_pipeline_{safe_comm}.joblib"
    joblib.dump(data['scaler'], scaler_path)
    joblib.dump(data['scaler'], scaler_comm_path)
    print(f"Saved scaler: {scaler_path} and {scaler_comm_path}")
    
    # Save metadata
    metadata = {
        "project": "AI-Based Food Inflation Prediction and Household Affordability Monitoring System",
        "model_name": best_name,
        "artifact_file": "price_prediction_model.joblib",
        "preprocessing_pipeline": "preprocessing_pipeline.joblib",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": "v2.0.0",
        "dataset_source": f"AGMARKNET {market_name} APMC Mandi Daily Series",
        "commodity": commodity_name,
        "market": f"{market_name} APMC",
        "district": market_name,
        "state": "Maharashtra",
        "target_variable": TARGET_REG,
        "target_definition": "Modal wholesale price on the next active APMC mandi trading session (Rs/Quintal)",
        "prediction_horizon": "1 active trading session ahead (t+1)",
        "feature_names": FEATURE_COLS,
        "feature_count": len(FEATURE_COLS),
        "training_samples": len(data['X_train_val']),
        "test_samples": len(data['X_test']),
        "test_evaluation_metrics": best_row.to_dict(),
        "all_models_evaluated": reg_df.to_dict(orient='records'),
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_comm_path = MODELS_DIR / f"model_metadata_{safe_comm}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(metadata_comm_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata: {metadata_path} and {metadata_comm_path}")
    
    return best_name


# ==============================================================================
# SAVE COMPREHENSIVE REPORT
# ==============================================================================
def save_reports(reg_df, gd_df, clf_df, cluster_df, commodity_name, market_name):
    """Save all results to CSV and markdown reports."""
    safe_comm = commodity_name.lower().replace(' ', '_')
    
    # Combined regression comparison
    reg_df.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)
    reg_df.to_csv(REPORTS_DIR / f"model_comparison_metrics_{safe_comm}.csv", index=False)
    
    # Classification comparison
    clf_df.to_csv(REPORTS_DIR / "classification_comparison.csv", index=False)
    clf_df.to_csv(REPORTS_DIR / f"classification_comparison_{safe_comm}.csv", index=False)
    
    # Clustering comparison
    cluster_df.to_csv(REPORTS_DIR / "clustering_comparison.csv", index=False)
    cluster_df.to_csv(REPORTS_DIR / f"clustering_comparison_{safe_comm}.csv", index=False)
    
    # Gradient descent comparison
    gd_df.to_csv(REPORTS_DIR / "gradient_descent_comparison.csv", index=False)
    gd_df.to_csv(REPORTS_DIR / f"gradient_descent_comparison_{safe_comm}.csv", index=False)
    
    print(f"\nAll comparison CSVs saved to {REPORTS_DIR}/ (both standard and {safe_comm}-specific)")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train ALL syllabus ML algorithms.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to model-ready CSV.")
    parser.add_argument("--commodity", type=str, default="Tomato", help="Commodity name.")
    parser.add_argument("--market", type=str, default="Pune", help="Market name.")
    args = parser.parse_args()
    
    # Determine dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        safe_commodity = args.commodity.lower().replace(' ', '_')
        safe_market = args.market.lower().replace(' ', '_')
        dataset_path = str(PROCESSED_DIR / f"model_ready_{safe_market}_{safe_commodity}.csv")
    
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run: py -3.12 scripts/build_features.py --all")
        sys.exit(1)
    
    print("=" * 80)
    print(f"COMPREHENSIVE ML PIPELINE: {args.commodity} @ {args.market}")
    print(f"Dataset: {dataset_path}")
    print("=" * 80)
    
    # Load data
    data = load_data(dataset_path)
    print(f"Train: {len(data['X_train'])} | Val: {len(data['X_val'])} | Test: {len(data['X_test'])}")
    
    # Module 1: Regression
    reg_df, models_dict, predictions = run_regression_models(data)
    
    # Module 2: Gradient Descent
    gd_df, convergence_data = run_gradient_descent(data)
    
    # Module 3: Classification
    clf_df, cm_data, roc_data = run_classification_models(data)
    
    # Module 4: Clustering & PCA
    cluster_df, pca, kmeans = run_clustering_pca(data, args.commodity)
    
    # Visualizations
    plot_regression_comparison(reg_df, predictions, data, args.commodity, args.market)
    
    # Save winning model
    best_name = save_winning_model(data, reg_df, models_dict, args.commodity, args.market)
    
    # Save reports
    save_reports(reg_df, gd_df, clf_df, cluster_df, args.commodity, args.market)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Commodity          : {args.commodity}")
    print(f"Market             : {args.market}")
    print(f"Regression Models  : {len(reg_df)} trained and evaluated")
    print(f"Best Regression    : {best_name}")
    print(f"GD Variants        : {len(gd_df)} optimization algorithms demonstrated")
    print(f"Classifiers        : {len(clf_df)} trained and evaluated")
    print(f"Clustering Methods : {len(cluster_df)} algorithms applied")
    print(f"PCA Components     : {pca.n_components_} extracted")
    print(f"Figures Generated  : reports/figures/")
    print(f"Reports Saved      : reports/")
    print("=" * 80)


if __name__ == "__main__":
    main()
