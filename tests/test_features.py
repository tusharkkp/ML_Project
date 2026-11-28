"""
Phase 11A: Canonical Feature Contract, Training-Serving Parity & Purged Split Tests.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.features import (
    SPECIALIST_NUMERIC_FEATURES,
    DatePurgedWalkForwardCV,
    canonicalize_market_name,
    compute_previous_window_price_stats,
    compute_price_momentum_7_pct,
    engineer_multi_series_features,
    engineer_single_series_features,
    match_canonical_market,
    normalize_market_token,
    split_chronologically_with_purging,
)


def _make_series(commodity: str, market: str, base_price: float, n: int = 25, step: float = 10.0):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "observation_date": dates,
            "commodity_name": commodity,
            "market_name": market,
            "min_price_rs_per_qtl": [base_price + i * step - 50 for i in range(n)],
            "max_price_rs_per_qtl": [base_price + i * step + 50 for i in range(n)],
            "modal_price_rs_per_qtl": [base_price + i * step for i in range(n)],
            "arrival_quantity_tonnes": [100.0 + i * 2.0 for i in range(n)],
        }
    )


def test_training_and_inference_feature_contract_exact_parity():
    """
    Verify on a known historical sequence that `engineer_single_series_features`
    matches exact mathematical definitions and `compute_price_momentum_7_pct` (%) helper.
    """
    # Known 10-day sequence: P = [100, 110, 120, 130, 140, 150, 160, 200, 220, 240]
    prices = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 200.0, 220.0, 240.0]
    arrivals = [10.0 * (i + 1) for i in range(10)]
    dates = pd.date_range("2022-05-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "observation_date": dates,
            "min_price_rs_per_qtl": [p - 10.0 for p in prices],
            "max_price_rs_per_qtl": [p + 15.0 for p in prices],
            "modal_price_rs_per_qtl": prices,
            "arrival_quantity_tonnes": arrivals,
        }
    )

    feat = engineer_single_series_features(df, has_arrivals=True, drop_incomplete=True)
    # Index t=7 is first row with full 7 prior sessions [0..6] and target at index 8!
    row0 = feat.iloc[0]
    assert row0["modal_price_rs_per_qtl"] == 200.0  # P(t=7)
    assert row0["price_lag_1"] == 160.0             # P(t=6)
    assert row0["price_lag_3"] == 140.0             # P(t=4)
    assert row0["price_lag_7"] == 100.0             # P(t=0)

    # Prior 7 sessions strictly exclude t=7 (`[100, 110, 120, 130, 140, 150, 160]`)
    prior_7 = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
    expected_stats = compute_previous_window_price_stats(prior_7)
    assert math.isclose(row0["rolling_mean_7"], expected_stats["rolling_mean_7"], rel_tol=1e-9)
    assert math.isclose(row0["rolling_std_7"], expected_stats["rolling_std_7"], rel_tol=1e-9)
    assert math.isclose(row0["rolling_min_7"], 100.0)
    assert math.isclose(row0["rolling_max_7"], 160.0)  # Must NOT be 200.0!

    # Canonical % momentum matches helper identically: ((200 - 100) / 100) * 100 = +100.0%
    expected_mom = compute_price_momentum_7_pct(row0["modal_price_rs_per_qtl"], row0["price_lag_7"])
    assert math.isclose(row0["price_momentum_7"], expected_mom, rel_tol=1e-9)
    assert math.isclose(row0["price_momentum_7"], 100.0, rel_tol=1e-9)

    assert math.isclose(row0["daily_spread"], 25.0)
    assert math.isclose(row0["arrival_tonnes"], 80.0)
    assert math.isclose(row0["arrival_lag_1"], 70.0)
    assert math.isclose(row0["rolling_arrival_mean_7"], 40.0)  # mean([10..70])


def test_missing_arrivals_never_silently_zero_filled():
    """Missing arrival telemetry must remain NaN and get excluded cleanly when drop_incomplete=True."""
    df = _make_series("Tomato", "Pune", 1000.0, n=15)
    df.loc[9, "arrival_quantity_tonnes"] = np.nan

    raw_feat = engineer_single_series_features(df, has_arrivals=True, drop_incomplete=False)
    assert pd.isna(raw_feat.loc[9, "arrival_tonnes"])
    assert pd.isna(raw_feat.loc[10, "arrival_lag_1"])

    clean_feat = engineer_single_series_features(df, has_arrivals=True, drop_incomplete=True)
    # Days referencing index 9 as arrival_tonnes (idx 9) or arrival_lag_1 (idx 10) must be dropped cleanly
    assert len(clean_feat) < len(df) - 8
    assert clean_feat[SPECIALIST_NUMERIC_FEATURES].notna().all().all()


def test_market_normalization_and_exact_alias_guards():
    """Verify exact alias equality prevents submarkets from merging into canonical single yards."""
    assert canonicalize_market_name("  APMC   Pune ") == "Pune"
    assert canonicalize_market_name("pune apmc") == "Pune"
    assert match_canonical_market("APMC Pune ", "Pune") == "Pune"
    assert match_canonical_market("Lasalgaon", "Lasalgaon") == "Lasalgaon"

    assert match_canonical_market("Pune(Pimpri)", "Pune") is None
    assert match_canonical_market("Pune (Hadapsar)", "Pune") is None
    assert match_canonical_market("Lasalgaon(Niphad) ", "Lasalgaon") is None
    assert match_canonical_market("Mumbai-Onion & Potato Market", "Mumbai") is None
    assert normalize_market_token("Pune (Pimpri)") == "pune(pimpri)"


def test_no_cross_commodity_or_cross_market_lag_leakage():
    df_high = _make_series("Moong Dal", "Mumbai", base_price=7000.0, n=25, step=5.0)
    df_low = _make_series("Onion", "Lasalgaon", base_price=500.0, n=25, step=10.0)
    df_other_mkt = _make_series("Onion", "Pune", base_price=1500.0, n=25, step=20.0)

    combined = pd.concat([df_high, df_low, df_other_mkt], ignore_index=True).sample(frac=1.0, random_state=42)
    feat = engineer_multi_series_features(combined, drop_incomplete=True)

    onion_las = feat[(feat["commodity_name"] == "Onion") & (feat["market_name"] == "Lasalgaon")]
    onion_pune = feat[(feat["commodity_name"] == "Onion") & (feat["market_name"] == "Pune")]
    dal = feat[(feat["commodity_name"] == "Moong Dal") & (feat["market_name"] == "Mumbai")]

    assert onion_las["price_lag_1"].max() < 800.0
    assert onion_pune["price_lag_1"].min() >= 1500.0
    assert dal["rolling_mean_7"].min() >= 6900.0
    assert (onion_las["modal_price_rs_per_qtl"] != onion_las["price_lag_1"]).all()


def test_duplicate_observation_date_raises_error():
    df = _make_series("Tomato", "Pune", base_price=1000.0, n=15)
    dup_df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate observation_date"):
        engineer_single_series_features(dup_df, drop_incomplete=False)


def test_chronological_splitting_and_date_purged_walk_forward_cv():
    """Adversarial test: outer splits & internal DatePurgedWalkForwardCV keep dates atomic & purged."""
    s1 = _make_series("Tomato", "Pune", base_price=1000.0, n=60)
    s2 = _make_series("Onion", "Lasalgaon", base_price=1200.0, n=60)
    s3 = _make_series("Sugar", "Mumbai", base_price=3200.0, n=60)
    feat = engineer_multi_series_features(pd.concat([s1, s2, s3], ignore_index=True), drop_incomplete=True)

    split_df, _ = split_chronologically_with_purging(feat, train_frac=0.70, val_frac=0.15)
    tr = split_df[split_df["split_tag"] == "Train"].reset_index(drop=True)
    va = split_df[split_df["split_tag"] == "Validation"].reset_index(drop=True)
    te = split_df[split_df["split_tag"] == "Test"].reset_index(drop=True)

    assert pd.to_datetime(tr["target_date_t_plus_1"]).max() < pd.to_datetime(va["observation_date"]).min()
    assert pd.to_datetime(va["target_date_t_plus_1"]).max() < pd.to_datetime(te["observation_date"]).min()

    cv = DatePurgedWalkForwardCV(tr["observation_date"], tr["target_date_t_plus_1"], n_splits=4)
    folds = list(cv.split(tr))
    assert len(folds) >= 3
    for tr_idx, val_idx in folds:
        tr_dates = set(pd.to_datetime(tr.loc[tr_idx, "observation_date"]))
        val_dates = set(pd.to_datetime(tr.loc[val_idx, "observation_date"]))
        # Zero shared dates across train and validation fold
        assert tr_dates.isdisjoint(val_dates)
        # Future observations never appear in train
        assert max(tr_dates) < min(val_dates)
        # Zero train target reaches validation fold start date
        max_tr_tgt = pd.to_datetime(tr.loc[tr_idx, "target_date_t_plus_1"]).max()
        assert max_tr_tgt < min(val_dates)
        # Same-date rows stay atomic in one split only
        for dt in val_dates:
            all_rows_dt = set(tr.index[pd.to_datetime(tr["observation_date"]) == dt])
            assert all_rows_dt.issubset(set(val_idx))
