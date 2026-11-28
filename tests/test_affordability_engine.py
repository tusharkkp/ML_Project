"""
Tests for Household Affordability Validation, Complete-Basket Integrity, and Risk Tiers (Phase 8).
"""

import math
import pytest

from src.affordability import calculate_household_affordability, classify_affordability


def test_affordability_tiers_and_complete_basket():
    assert "Low Vulnerability" in classify_affordability(25.0)
    assert "Moderate Vulnerability" in classify_affordability(38.0)
    assert "High Vulnerability" in classify_affordability(52.0)
    assert "Critical Vulnerability" in classify_affordability(65.0)

    res = calculate_household_affordability(
        monthly_income=10000.0,
        basket_items=[
            {
                "commodity_name": "Tomato",
                "monthly_quantity_kg": 10.0,
                "current_retail_price_per_kg": 40.0,
                "predicted_retail_price_per_kg": 50.0,
            },
            {
                "commodity_name": "Onion",
                "monthly_quantity_kg": 10.0,
                "current_retail_price_per_kg": 30.0,
                "predicted_retail_price_per_kg": 35.0,
            },
        ],
        sector="Urban",
    )
    assert res["has_complete_predictions"] is True
    assert res["prediction_status"] == "complete"
    assert res["current_food_basket_cost_rs"] == 700.0
    assert res["predicted_food_basket_cost_rs"] == 850.0
    assert res["monthly_inflation_impact_rs"] == 150.0
    assert res["current_food_budget_share_pct"] == 7.0
    assert res["predicted_food_budget_share_pct"] == 8.5


def test_partial_predicted_basket_withholds_future_totals():
    res = calculate_household_affordability(
        monthly_income=12000.0,
        basket_items=[
            {
                "commodity_name": "Tomato",
                "monthly_quantity_kg": 10.0,
                "current_retail_price_per_kg": 40.0,
                "predicted_retail_price_per_kg": 55.0,
            },
            {
                "commodity_name": "Onion",
                "monthly_quantity_kg": 10.0,
                "current_retail_price_per_kg": 35.0,
                "predicted_retail_price_per_kg": None,
            },
        ],
    )
    assert res["has_complete_predictions"] is False
    assert res["prediction_status"] == "incomplete_partial_predictions"
    assert res["missing_prediction_items"] == ["Onion"]
    assert res["predicted_food_basket_cost_rs"] is None
    assert res["predicted_affordability_status"] is None


def test_invalid_inputs_raise_value_error():
    valid_item = [{"commodity_name": "Tomato", "monthly_quantity_kg": 5.0, "current_retail_price_per_kg": 40.0}]
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_household_affordability(monthly_income=0.0, basket_items=valid_item)
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_household_affordability(monthly_income=-5000.0, basket_items=valid_item)
    with pytest.raises(ValueError, match="at least one commodity"):
        calculate_household_affordability(monthly_income=15000.0, basket_items=[])
    with pytest.raises(ValueError, match="non-negative finite"):
        calculate_household_affordability(
            monthly_income=15000.0,
            basket_items=[{"commodity_name": "Tomato", "monthly_quantity_kg": float("nan"), "current_retail_price_per_kg": 40.0}],
        )
    with pytest.raises(ValueError, match="non-negative finite"):
        calculate_household_affordability(
            monthly_income=15000.0,
            basket_items=[{"commodity_name": "Tomato", "monthly_quantity_kg": 5.0, "current_retail_price_per_kg": -10.0}],
        )
