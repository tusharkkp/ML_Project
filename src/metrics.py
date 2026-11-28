"""
Standardized Evaluation Metrics for Regression and Classification Workflows.
Shared across all training pipelines to guarantee consistent MAE, MSE, RMSE, R2, MAPE,
baseline comparisons, macro-averages, ROC-AUC, PR-AUC, and confusion matrices.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    round_digits: int = 2,
) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, R2, and MAPE (%) for continuous predictions."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 0 else 0.0

    nz = yt > 0
    mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100.0) if np.any(nz) else 0.0

    return {
        "MAE": round(mae, round_digits),
        "MSE": round(mse, round_digits),
        "RMSE": round(rmse, round_digits),
        "R2": round(r2, 4),
        "MAPE": round(mape, round_digits),
    }


def evaluate_regression_with_baseline(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_persist: Sequence[float],
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate a regression model alongside Persistence Baseline P(t) and RMSE improvement."""
    m = compute_regression_metrics(y_true, y_pred)
    b = compute_regression_metrics(y_true, y_persist)
    rmse_imp = round(float(b["RMSE"] - m["RMSE"]), 2)
    mae_imp = round(float(b["MAE"] - m["MAE"]), 2)

    return {
        "Model": model_name,
        "MAE": m["MAE"],
        "MSE": m["MSE"],
        "RMSE": m["RMSE"],
        "R2": m["R2"],
        "MAPE (%)": m["MAPE"],
        "Baseline_MAE": b["MAE"],
        "Baseline_RMSE": b["RMSE"],
        "Baseline_R2": b["R2"],
        "RMSE_Improvement": rmse_imp,
        "MAE_Improvement": mae_imp,
        "Beats_Baseline": bool(m["RMSE"] < b["RMSE"]),
    }


def compute_binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[Sequence[float]] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Compute binary Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(yt, yp))
    prec = float(precision_score(yt, yp, zero_division=0))
    rec = float(recall_score(yt, yp, zero_division=0))
    f1 = float(f1_score(yt, yp, zero_division=0))

    roc_auc = 0.5
    pr_auc = float(np.mean(yt)) if len(yt) > 0 else 0.0
    if y_proba is not None and len(np.unique(yt)) > 1:
        ypr = np.asarray(y_proba, dtype=float)
        try:
            roc_auc = float(roc_auc_score(yt, ypr))
        except Exception:
            pass
        try:
            pr_auc = float(average_precision_score(yt, ypr))
        except Exception:
            pass

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
    }


def compute_multiclass_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: np.ndarray,
    classes: Sequence[str],
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive multi-class early-warning metrics (Normal / Surge / Crash),
    plus binary Any-Spike (Surge + Crash vs Normal) ROC-AUC and PR-AUC.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    class_list = list(classes)

    acc = float(accuracy_score(yt, yp))
    macro_prec = float(precision_score(yt, yp, average="macro", zero_division=0))
    macro_rec = float(recall_score(yt, yp, average="macro", zero_division=0))
    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(yt, yp, average="weighted", zero_division=0))

    # Align probability columns cleanly with `classes`
    roc_ovr = 0.5
    pr_macro = 0.0
    any_spike_roc = 0.5
    any_spike_pr = 0.0

    if y_proba is not None and y_proba.shape[1] == len(class_list):
        try:
            y_bin = np.column_stack([(yt == c).astype(int) for c in class_list])
            if np.all(y_bin.sum(axis=0) > 0):
                roc_ovr = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
                pr_macro = float(average_precision_score(y_bin, y_proba, average="macro"))
        except Exception:
            pass

        try:
            y_any_spike = np.isin(yt, ["Surge", "Crash"]).astype(int)
            spike_cols = [i for i, c in enumerate(class_list) if c in ("Surge", "Crash")]
            if spike_cols and len(np.unique(y_any_spike)) > 1:
                p_any_spike = y_proba[:, spike_cols].sum(axis=1)
                any_spike_roc = float(roc_auc_score(y_any_spike, p_any_spike))
                any_spike_pr = float(average_precision_score(y_any_spike, p_any_spike))
        except Exception:
            pass

    cm = confusion_matrix(yt, yp, labels=class_list).tolist()
    report = classification_report(yt, yp, labels=class_list, output_dict=True, zero_division=0)

    return {
        "Model": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_ovr_macro": round(roc_ovr, 4),
        "pr_auc_macro": round(pr_macro, 4),
        "any_spike_roc_auc": round(any_spike_roc, 4),
        "any_spike_pr_auc": round(any_spike_pr, 4),
        "confusion_matrix": cm,
        "per_class_report": report,
    }
