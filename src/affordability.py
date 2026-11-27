"""
Household Affordability & Food Basket Calculator
Implements deterministic econometric affordability models based on NSSO HCES 2022-23 benchmarks.

Formulas:
- Food Basket Cost = Sum of (Commodity Quantity * Commodity Retail Price)
- Food Budget Share (%) = (Food Basket Cost / Monthly Household Income) * 100
"""

from typing import List, Dict, Any, Optional

# NSSO HCES 2022-23 Factsheet Benchmark Percentages
NSSO_BENCHMARKS = {
    "Rural_Average_Food_Share_Pct": 46.38,
    "Urban_Average_Food_Share_Pct": 39.17,
    "Bottom_20_Fractile_Food_Share_Pct": 56.60,
    "Top_20_Fractile_Food_Share_Pct": 29.80,
    "Maharashtra_Rural_Average_MPCE_Rs": 4252.0,
    "Maharashtra_Urban_Average_MPCE_Rs": 6611.0,
}

def classify_affordability(food_budget_share_pct: float) -> str:
    """Classifies household vulnerability based on the proportion of income allocated to food."""
    if food_budget_share_pct < 30.0:
        return "Low Vulnerability (Food Secure — Engel Curve Top Quartile)"
    elif food_budget_share_pct <= 45.0:
        return "Moderate Vulnerability (Normal Middle-Income Range)"
    elif food_budget_share_pct <= 60.0:
        return "High Vulnerability (Food Insecure — Resembles Bottom 30% Fractile)"
    else:
        return "Critical Food Distress (Extreme Poverty / Severe Food Deprivation)"

def calculate_household_affordability(
    monthly_income: float,
    basket_items: List[Dict[str, Any]],
    sector: str = "Urban",
    household_id: str = "HH-DEFAULT"
) -> Dict[str, Any]:
    """
    Computes deterministic food basket cost and evaluates economic affordability.

    Args:
        monthly_income: Total monthly household earnings in Indian Rupees.
        basket_items: List of dicts, each with 'commodity_name', 'monthly_quantity_kg',
                      'current_retail_price_per_kg', and optional 'predicted_retail_price_per_kg'.
        sector: 'Urban' or 'Rural'
        household_id: Optional household reference identifier.
    """
    if monthly_income <= 0:
        raise ValueError("Monthly income must be strictly positive.")

    total_current_cost = 0.0
    total_predicted_cost = 0.0
    has_predictions = False

    item_breakdown = []

    for item in basket_items:
        qty = float(item["monthly_quantity_kg"])
        p_curr = float(item["current_retail_price_per_kg"])
        item_curr_cost = qty * p_curr
        total_current_cost += item_curr_cost

        item_pred_cost = None
        if "predicted_retail_price_per_kg" in item and item["predicted_retail_price_per_kg"] is not None:
            has_predictions = True
            p_pred = float(item["predicted_retail_price_per_kg"])
            item_pred_cost = qty * p_pred
            total_predicted_cost += item_pred_cost

        item_breakdown.append({
            "commodity_name": item["commodity_name"],
            "monthly_quantity_kg": qty,
            "current_retail_price_per_kg": p_curr,
            "current_cost_rs": round(item_curr_cost, 2),
            "predicted_retail_price_per_kg": item.get("predicted_retail_price_per_kg"),
            "predicted_cost_rs": round(item_pred_cost, 2) if item_pred_cost else None
        })

    # Calculate Food Budget Shares
    current_share_pct = (total_current_cost / monthly_income) * 100.0
    predicted_share_pct = ((total_predicted_cost / monthly_income) * 100.0) if has_predictions else None

    # Benchmark comparison
    benchmark_ref = (
        f"NSSO 2022-23 {sector} Baseline Average Food Share is "
        f"{NSSO_BENCHMARKS['Urban_Average_Food_Share_Pct'] if sector.lower()=='urban' else NSSO_BENCHMARKS['Rural_Average_Food_Share_Pct']}%"
    )

    # Inflation shock differential
    diff_rs = round(total_predicted_cost - total_current_cost, 2) if has_predictions else None
    if diff_rs and diff_rs > 0:
        alert = f"WARNING: Predicted commodity price increases will inflate monthly grocery expenses by Rs. {diff_rs:,.2f} (+{predicted_share_pct - current_share_pct:.2f}% of income)."
    elif diff_rs and diff_rs < 0:
        alert = f"FAVORABLE: Predicted commodity price drops will reduce monthly grocery expenses by Rs. {abs(diff_rs):,.2f}."
    else:
        alert = "STABLE: Commodity prices are forecasted to remain within normal baseline bounds."

    status = classify_affordability(current_share_pct)
    discretionary_income = round(monthly_income - total_current_cost, 2)

    return {
        "household_id": household_id,
        "monthly_income_rs": round(monthly_income, 2),
        "total_food_basket_cost_current_rs": round(total_current_cost, 2),
        "current_food_basket_cost_rs": round(total_current_cost, 2),
        "total_food_basket_cost_predicted_rs": round(total_predicted_cost, 2) if has_predictions else None,
        "predicted_food_basket_cost_rs": round(total_predicted_cost, 2) if has_predictions else None,
        "discretionary_income_remaining_rs": discretionary_income,
        "current_food_budget_share_pct": round(current_share_pct, 2),
        "predicted_food_budget_share_pct": round(predicted_share_pct, 2) if predicted_share_pct else None,
        "affordability_classification": status,
        "affordability_status": status,
        "nsso_benchmark_reference": benchmark_ref,
        "monthly_inflation_impact_rs": diff_rs,
        "net_cost_increase_rs": diff_rs,
        "vulnerability_alert": alert,
        "item_breakdown": item_breakdown
    }

if __name__ == "__main__":
    # Test case: Urban middle-income household in Pune (Income = Rs. 25,000/month)
    sample_basket = [
        {"commodity_name": "Rice", "monthly_quantity_kg": 20.0, "current_retail_price_per_kg": 45.0, "predicted_retail_price_per_kg": 46.5},
        {"commodity_name": "Wheat (Atta)", "monthly_quantity_kg": 25.0, "current_retail_price_per_kg": 35.0, "predicted_retail_price_per_kg": 36.0},
        {"commodity_name": "Tur Dal", "monthly_quantity_kg": 5.0, "current_retail_price_per_kg": 150.0, "predicted_retail_price_per_kg": 155.0},
        {"commodity_name": "Tomato", "monthly_quantity_kg": 12.0, "current_retail_price_per_kg": 30.0, "predicted_retail_price_per_kg": 45.0}, # Monsoon shock
        {"commodity_name": "Onion", "monthly_quantity_kg": 15.0, "current_retail_price_per_kg": 35.0, "predicted_retail_price_per_kg": 40.0},
        {"commodity_name": "Potato", "monthly_quantity_kg": 12.0, "current_retail_price_per_kg": 25.0, "predicted_retail_price_per_kg": 26.0},
        {"commodity_name": "Sunflower Oil", "monthly_quantity_kg": 4.0, "current_retail_price_per_kg": 140.0, "predicted_retail_price_per_kg": 142.0},
        {"commodity_name": "Milk", "monthly_quantity_kg": 30.0, "current_retail_price_per_kg": 58.0, "predicted_retail_price_per_kg": 60.0},
        {"commodity_name": "Sugar", "monthly_quantity_kg": 5.0, "current_retail_price_per_kg": 42.0, "predicted_retail_price_per_kg": 42.0}
    ]
    report = calculate_household_affordability(monthly_income=25000.0, basket_items=sample_basket, sector="Urban")
    import json
    print(json.dumps(report, indent=2))
