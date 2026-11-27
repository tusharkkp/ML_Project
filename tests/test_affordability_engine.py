"""
Unit Tests for Household Food Affordability & Engel's Law Engine
Course: Third-Year B.Tech AIML Practical Examination (Rubric Component H)
"""

import pytest
from src.affordability import calculate_household_affordability, classify_affordability, NSSO_BENCHMARKS


def test_classify_affordability_thresholds():
    assert "Low Vulnerability" in classify_affordability(25.0)
    assert "Moderate Vulnerability" in classify_affordability(40.0)
    assert "High Vulnerability" in classify_affordability(55.0)
    assert "Critical Food Distress" in classify_affordability(75.0)


def test_calculate_household_affordability_keys():
    basket = [
        {"commodity_name": "Rice & Cereals", "monthly_quantity_kg": 20.0, "current_retail_price_per_kg": 38.0},
        {"commodity_name": "Pulses (Moong Dal)", "monthly_quantity_kg": 4.0, "current_retail_price_per_kg": 74.0},
        {"commodity_name": "Onion", "monthly_quantity_kg": 6.0, "current_retail_price_per_kg": 25.0},
        {"commodity_name": "Tomato", "monthly_quantity_kg": 5.0, "current_retail_price_per_kg": 28.0},
        {"commodity_name": "Sugar", "monthly_quantity_kg": 3.0, "current_retail_price_per_kg": 42.0},
    ]

    res = calculate_household_affordability(monthly_income=25000.0, basket_items=basket, sector="Urban")

    # Verify both canonical and convenience alias keys exist
    assert "affordability_status" in res
    assert "affordability_classification" in res
    assert "current_food_basket_cost_rs" in res
    assert "total_food_basket_cost_current_rs" in res
    assert "discretionary_income_remaining_rs" in res
    assert "current_food_budget_share_pct" in res

    # Verify math accuracy
    # Total cost = 20*38 (760) + 4*74 (296) + 6*25 (150) + 5*28 (140) + 3*42 (126) = 1472.0
    expected_cost = 760.0 + 296.0 + 150.0 + 140.0 + 126.0
    assert res["current_food_basket_cost_rs"] == pytest.approx(expected_cost, 0.01)
    assert res["discretionary_income_remaining_rs"] == pytest.approx(25000.0 - expected_cost, 0.01)
    assert res["current_food_budget_share_pct"] == pytest.approx((expected_cost / 25000.0) * 100.0, 0.01)


def test_invalid_income_raises_value_error():
    basket = [{"commodity_name": "Tomato", "monthly_quantity_kg": 5.0, "current_retail_price_per_kg": 28.0}]
    with pytest.raises(ValueError):
        calculate_household_affordability(monthly_income=0, basket_items=basket)
