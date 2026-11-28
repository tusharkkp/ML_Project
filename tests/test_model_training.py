"""
Tests verifying leak-free model selection, chronological cutoffs, and regenerated metadata (Phase 1 & 6 & 7).
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"


def test_unified_model_selected_by_validation_and_has_disjoint_dates():
    with open(MODELS_DIR / "unified_model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    board = pd.DataFrame(meta["validation_candidates_benchmarked"])
    best_val_model = board.sort_values("Val_RMSE", ascending=True).iloc[0]["Model"]
    assert meta["champion_algorithm"] == best_val_model

    # Verify macro-average and per-commodity keys are present alongside pooled metrics
    assert "macro_average_test_metrics" in meta
    assert set(meta["per_commodity_metrics"].keys()) == {"Moong Dal", "Onion", "Sugar", "Tomato"}

    # Verify clean unified dataset has strictly disjoint chronological partitions
    clean_df = pd.read_csv(DATA_DIR / "model_ready_unified_commodities_clean.csv")
    tr = pd.to_datetime(clean_df[clean_df["split_tag"] == "Train"]["observation_date"])
    va = pd.to_datetime(clean_df[clean_df["split_tag"] == "Validation"]["observation_date"])
    te = pd.to_datetime(clean_df[clean_df["split_tag"] == "Test"]["observation_date"])
    tr_tgt = pd.to_datetime(clean_df[clean_df["split_tag"] == "Train"]["target_date_t_plus_1"])
    va_tgt = pd.to_datetime(clean_df[clean_df["split_tag"] == "Validation"]["target_date_t_plus_1"])

    assert tr.max() < va.min()
    assert va.max() < te.min()
    assert tr_tgt.max() < va.min()
    assert va_tgt.max() < te.min()


def test_specialist_models_selected_strictly_on_validation():
    for safe_comm in ("tomato", "onion"):
        with open(MODELS_DIR / f"model_metadata_{safe_comm}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        val_board = pd.DataFrame(meta["validation_leaderboard"])
        top_val = val_board.sort_values("Val_RMSE", ascending=True).iloc[0]["Model"]
        assert meta["model_name"] == top_val
        assert "target_definition" in meta
        assert meta["feature_count"] == 16


def test_classifier_and_retail_metadata_honesty():
    with open(MODELS_DIR / "spike_classifier_metadata.json", "r", encoding="utf-8") as f:
        clf_meta = json.load(f)
    val_board = pd.DataFrame(clf_meta["validation_leaderboard"])
    top_val = val_board.sort_values(["macro_f1", "pr_auc_macro"], ascending=False).iloc[0]["Model"]
    assert clf_meta["champion_algorithm"] == top_val
    assert "any_spike_pr_auc" in clf_meta["test_metrics"]

    with open(MODELS_DIR / "retail_model_metadata.json", "r", encoding="utf-8") as f:
        ret_meta = json.load(f)
    assert ret_meta["total_same_city_same_date_pairs"] == 0
    assert ret_meta["retail_estimation_supported"] is False
