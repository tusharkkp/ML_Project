"""
Household Affordability & Food Basket Calculator (Phase 8).

Implements deterministic econometric affordability calculations benchmarked against
NSSO HCES 2022-23 cross-sectional consumer expenditure fractiles under Engel's Law.

Strict Correctness Guarantees:
- Rejects non-positive income, negative/NaN quantities, negative/NaN prices.
- Never calculates or displays a "complete predicted food basket" unless EVERY item
  in the basket has a valid predicted retail price; partial predictions remain flagged
  explicitly as incomplete/unavailable.
- Explicitly frames outputs as analytical estimates rather than official diagnoses.
"""

import math
from typing import Any, Dict, List, Optional


NSSO_BENCHMARKS = {
    "Rural_Average_Food_Share_Pct": 46.38,
    "Urban_Average_Food_Share_Pct": 39.17,
    "Bottom_20_Fractile_Food_Share_Pct": 56.60,
    "Top_20_Fractile_Food_Share_Pct": 29.80,
    "Maharashtra_Rural_Average_MPCE_Rs": 4252.0,
    "Maharashtra_Urban_Average_MPCE_Rs": 6611.0,
}


def classify_affordability(food_budget_share_pct: float) -> str:
    """Classify analytical vulnerability tier based on food budget share (% of income)."""
    if food_budget_share_pct < 30.0:
        return "Low Vulnerability (Analytical Estimate — Below 30% Tier)"
    elif food_budget_share_pct <= 45.0:
        return "Moderate Vulnerability (Analytical Estimate — Middle-Income Corridor)"
    elif food_budget_share_pct <= 60.0:
        return "High Vulnerability (Analytical Estimate — Near Bottom-20% HCES Tier)"
    else:
        return "Critical Vulnerability Pressure (Analytical Estimate — Above 60% Food Share)"


def _validate_non_negative_finite(val: Any, field_name: str) -> float:
    if val is None:
        raise ValueError(f"{field_name} cannot be None.")
    num = float(val)
    if not math.isfinite(num) or num < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite number (got {val}).")
    return num


def calculate_household_affordability(
    monthly_income: float,
    basket_items: List[Dict[str, Any]],
    sector: str = "Urban",
    household_id: str = "HH-DEFAULT",
    family_size: int = 4,
) -> Dict[str, Any]:
    """
    Compute deterministic current and (when complete) predicted monthly food basket cost
    and budget share under Engel's Law against NSSO HCES 2022-23 cross-sectional benchmarks.
    """
    income = float(monthly_income) if monthly_income is not None else -1.0
    if not math.isfinite(income) or income <= 0.0:
        raise ValueError("Monthly income must be strictly positive.")
    if not basket_items:
        raise ValueError("Basket must contain at least one commodity item.")
    if int(family_size) < 1:
        raise ValueError("Family size must be at least 1.")

    total_current_cost = 0.0
    total_predicted_cost = 0.0
    items_with_predictions: List[str] = []
    items_missing_predictions: List[str] = []
    item_breakdown: List[Dict[str, Any]] = []

    for item in basket_items:
        name = str(item.get("commodity_name", "Unknown")).strip()
        qty = _validate_non_negative_finite(
            item.get("monthly_quantity_kg"), f"Quantity for '{name}'"
        )
        p_curr = _validate_non_negative_finite(
            item.get("current_retail_price_per_kg"), f"Current retail price for '{name}'"
        )
        item_curr_cost = qty * p_curr
        total_current_cost += item_curr_cost

        raw_pred = item.get("predicted_retail_price_per_kg")
        item_pred_cost: Optional[float] = None
        if raw_pred is not None:
            p_pred = _validate_non_negative_finite(
                raw_pred, f"Predicted retail price for '{name}'"
            )
            item_pred_cost = qty * p_pred
            total_predicted_cost += item_pred_cost
            items_with_predictions.append(name)
        else:
            items_missing_predictions.append(name)

        item_breakdown.append(
            {
                "commodity_name": name,
                "monthly_quantity_kg": qty,
                "current_retail_price_per_kg": round(p_curr, 2),
                "current_price_source": str(item.get("current_price_source", item.get("price_source", "User Supplied"))),
                "current_cost_rs": round(item_curr_cost, 2),
                "predicted_retail_price_per_kg": round(float(raw_pred), 2) if raw_pred is not None else None,
                "predicted_cost_rs": round(item_pred_cost, 2) if item_pred_cost is not None else None,
                "prediction_source": str(
                    item.get(
                        "prediction_source",
                        "Unavailable" if raw_pred is None else "User / Scenario Provided",
                    )
                ),
            }
        )

    # Strictly require predictions for ALL basket items before reporting predicted basket totals
    has_complete_predictions = (len(items_missing_predictions) == 0) and (len(basket_items) > 0)
    has_partial_predictions = (len(items_with_predictions) > 0) and (len(items_missing_predictions) > 0)

    if has_complete_predictions:
        prediction_status = "complete"
    elif has_partial_predictions:
        prediction_status = "incomplete_partial_predictions"
    else:
        prediction_status = "unavailable"

    current_share_pct = (total_current_cost / income) * 100.0
    discretionary_income = round(income - total_current_cost, 2)
    curr_status = classify_affordability(current_share_pct)

    if has_complete_predictions:
        pred_cost_rounded = round(total_predicted_cost, 2)
        pred_share_pct = round((total_predicted_cost / income) * 100.0, 2)
        pred_status_label = classify_affordability(pred_share_pct)
        pred_discretionary = round(income - total_predicted_cost, 2)
        diff_rs = round(total_predicted_cost - total_current_cost, 2)
        share_delta = round(pred_share_pct - current_share_pct, 2)
        if diff_rs > 0:
            alert = (
                f"WARNING (Analytical Projection): Projected price increases would raise the monthly basket "
                f"by Rs. {diff_rs:,.2f} ({share_delta:+.2f} percentage points of income)."
            )
        elif diff_rs < 0:
            alert = (
                f"FAVORABLE (Analytical Projection): Projected price declines would reduce the monthly basket "
                f"by Rs. {abs(diff_rs):,.2f} ({share_delta:+.2f} percentage points of income)."
            )
        else:
            alert = "STABLE (Analytical Projection): Projected food basket cost matches current baseline."
    else:
        pred_cost_rounded = None
        pred_share_pct = None
        pred_status_label = None
        pred_discretionary = None
        diff_rs = None
        share_delta = None
        if has_partial_predictions:
            alert = (
                f"PREDICTED BASKET INCOMPLETE: Validated retail price predictions are missing for "
                f"{items_missing_predictions}. Complete predicted basket cost is withheld to avoid undercounting."
            )
        else:
            alert = "PREDICTED BASKET UNAVAILABLE: Supply current or complete predicted retail prices for all items."

    sector_norm = "Urban" if sector.strip().lower() == "urban" else "Rural"
    sector_ref_val = (
        NSSO_BENCHMARKS["Urban_Average_Food_Share_Pct"]
        if sector_norm == "Urban"
        else NSSO_BENCHMARKS["Rural_Average_Food_Share_Pct"]
    )
    benchmark_ref = (
        f"NSSO HCES 2022-23 {sector_norm} Cross-Sectional Average Food Share: {sector_ref_val}%"
    )

    return {
        "household_id": household_id,
        "sector": sector_norm,
        "family_size": int(family_size),
        "monthly_income_rs": round(income, 2),
        "mpce_rs": round(income / int(family_size), 2),
        "total_food_basket_cost_current_rs": round(total_current_cost, 2),
        "current_food_basket_cost_rs": round(total_current_cost, 2),
        "total_food_basket_cost_predicted_rs": pred_cost_rounded,
        "predicted_food_basket_cost_rs": pred_cost_rounded,
        "discretionary_income_remaining_rs": discretionary_income,
        "predicted_discretionary_income_rs": pred_discretionary,
        "current_food_budget_share_pct": round(current_share_pct, 2),
        "predicted_food_budget_share_pct": pred_share_pct,
        "budget_share_change_pct_points": share_delta,
        "affordability_classification": curr_status,
        "affordability_status": curr_status,
        "predicted_affordability_status": pred_status_label,
        "has_complete_predictions": has_complete_predictions,
        "prediction_status": prediction_status,
        "missing_prediction_items": items_missing_predictions,
        "nsso_benchmark_reference": benchmark_ref,
        "nsso_sector_baseline_pct": sector_ref_val,
        "monthly_inflation_impact_rs": diff_rs,
        "net_cost_increase_rs": diff_rs,
        "vulnerability_alert": alert,
        "methodology_disclaimer": (
            "Analytical budget estimate calibrated against cross-sectional NSSO HCES 2022-23 survey fractiles; "
            "this is NOT an official household poverty or clinical food-insecurity diagnosis."
        ),
        "item_breakdown": item_breakdown,
    }
