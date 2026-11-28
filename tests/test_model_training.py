"""
Phase 11E: Model Training Selection Consistency, Independent LOCO & Train-Only Clustering Tests.
"""

import json
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def test_unified_model_selected_by_documented_primary_metric_and_independent_loco():
    with open(MODELS_DIR / "unified_model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    primary = meta["primary_selection_metric"]
    assert primary == "Val_Macro_RMSE"
    board = pd.DataFrame(meta["validation_candidates_benchmarked"])
    best_model = board.sort_values([primary, "Val_RMSE"], ascending=True).iloc[0]["Model"]
    assert meta["champion_algorithm"] == best_model
    assert meta["feature_schema_version"] == "3.1.0-parity"

    # Independent LOCO records per-fold selection metadata
    loco = meta["leave_one_commodity_out_experiment"]
    assert set(loco.keys()) == {"Moong Dal", "Onion", "Sugar", "Tomato"}
    for comm, entry in loco.items():
        assert "selected_algorithm" in entry
        assert "remaining_val_macro_rmse" in entry

    # Verify disjoint partitions & zero target boundary crossings in dataset
    clean_df = pd.read_csv(DATA_DIR / "model_ready_unified_commodities_clean.csv")
    tr = pd.to_datetime(clean_df[clean_df["split_tag"] == "Train"]["observation_date"])
    va = pd.to_datetime(clean_df[clean_df["split_tag"] == "Validation"]["observation_date"])
    te = pd.to_datetime(clean_df[clean_df["split_tag"] == "Test"]["observation_date"])
    assert tr.max() < va.min()
    assert va.max() < te.min()


def test_specialist_validation_selection_and_train_only_clustering_outputs():
    for safe_comm in ("tomato", "onion"):
        with open(MODELS_DIR / f"model_metadata_{safe_comm}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        val_board = pd.DataFrame(meta["validation_leaderboard"])
        top_val = val_board.sort_values("Val_RMSE", ascending=True).iloc[0]["Model"]
        assert meta["model_name"] == top_val
        assert meta["feature_schema_version"] == "3.1.0-parity"

        # Test comparison file evaluates ONLY Persistence Baseline & Validation Champion (2 rows!)
        test_cmp = pd.read_csv(REPORTS_DIR / f"model_comparison_metrics_{safe_comm}.csv")
        assert len(test_cmp) == 2

        # Clustering must report Train Only fit and profiles
        cl_df = pd.read_csv(REPORTS_DIR / f"clustering_comparison_{safe_comm}.csv")
        assert (cl_df["Fitted_Split"].str.contains("Train Only")).all()
        prof_df = pd.read_csv(REPORTS_DIR / f"clustering_profiles_{safe_comm}.csv")
        assert {"Mean_Modal_Price_Rs_Qtl", "CV_Volatility_Pct", "Price_Spike_Rate_Pct"}.issubset(set(prof_df.columns))


def test_classifier_and_retail_metadata_honesty():
    with open(MODELS_DIR / "spike_classifier_metadata.json", "r", encoding="utf-8") as f:
        clf_meta = json.load(f)
    val_board = pd.DataFrame(clf_meta["validation_leaderboard"])
    top_val = val_board.sort_values(["macro_f1", "pr_auc_macro"], ascending=False).iloc[0]["Model"]
    assert clf_meta["champion_algorithm"] == top_val
    assert clf_meta["feature_schema_version"] == "3.1.0-parity"

    with open(MODELS_DIR / "retail_model_metadata.json", "r", encoding="utf-8") as f:
        ret_meta = json.load(f)
    assert ret_meta["total_same_city_same_date_pairs"] == 0
    assert ret_meta["retail_estimation_supported"] is False
