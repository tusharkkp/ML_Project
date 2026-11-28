"""
Tests for Exact-Market Comparison & Cross-Market Coverage Engine (Phase 3).
"""

import pandas as pd
import pytest

from src.market_comparison import (
    compare_markets,
    get_available_commodities,
    get_markets_for_commodity,
)


def test_available_commodities_and_markets():
    comms = get_available_commodities(min_markets=2, min_rows_per_market=30)
    assert "Onion" in comms
    assert "Tomato" in comms
    mkts = get_markets_for_commodity("Onion", min_rows=30)
    assert "Pune" in mkts and "Lasalgaon" in mkts


def test_compare_markets_uses_exact_markets_and_strict_overlap():
    comp = compare_markets(
        commodity="Onion",
        markets=["Pune", "Lasalgaon"],
        start_date="2016-01-01",
        end_date="2018-12-31",
    )
    summary = comp["summary_df"]
    assert list(summary["Market"]) == ["Pune", "Lasalgaon"]
    assert (summary["Trading_Sessions"] > 500).all()

    # Exact verification: no Pune submarket name leaked into "Pune" timeseries
    ts = comp["timeseries_df"]
    assert set(ts["market_name"].unique()) == {"Pune", "Lasalgaon"}

    # Overlapping comparison uses inner join strictly
    ov = comp["overlapping_comparison_df"]
    assert len(ov) == 1
    pune_dates = set(ts[ts["market_name"] == "Pune"]["observation_date"])
    las_dates = set(ts[ts["market_name"] == "Lasalgaon"]["observation_date"])
    assert int(ov.iloc[0]["Overlapping_Dates"]) == len(pune_dates & las_dates)
    assert "NOT multi-city ML price forecasting" in comp["disclaimer"]
