"""
Tests for leak-free time-series feature engineering & chronological splitting (Phase 1.2 & 1.3).
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    engineer_multi_series_features,
    engineer_single_series_features,
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
            "arrival_quantity_tonnes": [100.0 + i for i in range(n)],
        }
    )


def test_no_cross_commodity_or_cross_market_lag_leakage():
    """One commodity/market's previous price must NEVER leak into another commodity/market's lag."""
    df_high = _make_series("Moong Dal", "Mumbai", base_price=7000.0, n=25, step=5.0)
    df_low = _make_series("Onion", "Lasalgaon", base_price=500.0, n=25, step=10.0)
    df_other_mkt = _make_series("Onion", "Pune", base_price=1500.0, n=25, step=20.0)

    combined = pd.concat([df_high, df_low, df_other_mkt], ignore_index=True)
    # Shuffle rows adversarially before calling feature builder
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    feat = engineer_multi_series_features(
        combined,
        group_cols=("commodity_name", "market_name"),
        arrival_col="arrival_quantity_tonnes",
        drop_incomplete=True,
    )

    onion_las = feat[(feat["commodity_name"] == "Onion") & (feat["market_name"] == "Lasalgaon")].reset_index(drop=True)
    onion_pune = feat[(feat["commodity_name"] == "Onion") & (feat["market_name"] == "Pune")].reset_index(drop=True)
    dal = feat[(feat["commodity_name"] == "Moong Dal") & (feat["market_name"] == "Mumbai")].reset_index(drop=True)

    assert not onion_las.empty and not onion_pune.empty and not dal.empty
    assert onion_las["price_lag_1"].max() < 800.0
    assert onion_pune["price_lag_1"].min() >= 1500.0
    assert dal["rolling_mean_7"].min() >= 6900.0
    # Strict shift: P(t) != P(t-1) when prices strictly increase
    assert (onion_las["modal_price_rs_per_qtl"] != onion_las["price_lag_1"]).all()


def test_rolling_features_contain_zero_future_observations():
    """Altering future targets/prices (> idx t) must NOT alter features computed at idx t."""
    base_df = _make_series("Tomato", "Pune", base_price=1000.0, n=20, step=15.0)
    base_df = base_df.rename(columns={"arrival_quantity_tonnes": "arrival_tonnes"})

    # Compute without dropping end rows so we can inspect earlier rows cleanly
    feat_orig = engineer_single_series_features(base_df.rename(columns={"arrival_tonnes": "arrival_quantity_tonnes"}), has_arrivals=True, drop_incomplete=False)

    altered = base_df.copy()
    # Mutate days 15..19 massively in the future
    altered.loc[15:, "modal_price_rs_per_qtl"] = 99999.0
    feat_alt = engineer_single_series_features(altered.rename(columns={"arrival_tonnes": "arrival_quantity_tonnes"}), has_arrivals=True, drop_incomplete=False)

    check_cols = ["modal_price_rs_per_qtl", "price_lag_1", "price_lag_7", "rolling_mean_7", "rolling_std_7", "rolling_min_7", "rolling_max_7"]
    # At index t=12, features only depend on [5..12], completely invariant to [15..19]
    pd.testing.assert_frame_equal(
        feat_orig.loc[7:14, check_cols],
        feat_alt.loc[7:14, check_cols],
    )


def test_duplicate_observation_date_raises_error():
    """Duplicate dates in a single series must fail loudly instead of silent duplicate shifts."""
    df = _make_series("Tomato", "Pune", base_price=1000.0, n=15)
    dup_df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate observation_date"):
        engineer_single_series_features(dup_df, drop_incomplete=False)


def test_chronological_splitting_purges_cross_boundary_targets():
    """Shared global cutoffs must guarantee zero date overlap and zero target-date boundary crossings."""
    s1 = _make_series("Tomato", "Pune", base_price=1000.0, n=50)
    s2 = _make_series("Onion", "Lasalgaon", base_price=1200.0, n=50)
    feat = engineer_multi_series_features(pd.concat([s1, s2], ignore_index=True), drop_incomplete=True)

    split_df, meta = split_chronologically_with_purging(feat, train_frac=0.70, val_frac=0.15)
    assert meta["purged_boundary_rows"] >= 0

    train = split_df[split_df["split_tag"] == "Train"]
    val = split_df[split_df["split_tag"] == "Validation"]
    test = split_df[split_df["split_tag"] == "Test"]

    assert not train.empty and not val.empty and not test.empty
    assert pd.to_datetime(train["observation_date"]).max() < pd.to_datetime(val["observation_date"]).min()
    assert pd.to_datetime(val["observation_date"]).max() < pd.to_datetime(test["observation_date"]).min()
    # Critical check: target date of Train never enters Validation; target date of Val never enters Test!
    assert pd.to_datetime(train["target_date_t_plus_1"]).max() < pd.to_datetime(val["observation_date"]).min()
    assert pd.to_datetime(val["target_date_t_plus_1"]).max() < pd.to_datetime(test["observation_date"]).min()
