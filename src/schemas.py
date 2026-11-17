"""
Pydantic Schemas for AI-Based Food Inflation Prediction and Household Affordability System
Defines strong data validation contracts for REST API input and output payloads.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class CommodityPricePredictionRequest(BaseModel):
    """Input payload for single-step commodity wholesale price prediction."""
    commodity: str = Field(default="Tomato", description="Canonical commodity name")
    market: str = Field(default="Pune APMC", description="Regulated market yard name")
    price_lag_1: float = Field(..., gt=0, description="Observed modal price today at t (Rs/Quintal)")
    price_lag_3: float = Field(..., gt=0, description="Observed modal price 3 trading sessions ago (Rs/Quintal)")
    price_lag_7: float = Field(..., gt=0, description="Observed modal price 7 trading sessions ago (Rs/Quintal)")
    rolling_mean_7: float = Field(..., gt=0, description="7-session moving average price (Rs/Quintal)")
    rolling_std_7: float = Field(..., ge=0, description="7-session rolling standard deviation (Rs/Quintal)")
    rolling_min_7: float = Field(..., gt=0, description="7-session minimum modal price (Rs/Quintal)")
    rolling_max_7: float = Field(..., gt=0, description="7-session maximum modal price (Rs/Quintal)")
    price_momentum_7: float = Field(..., description="7-session price velocity (P_t - P_{t-6})")
    daily_spread: float = Field(..., ge=0, description="Intra-day auction spread (P_max - P_min)")
    arrival_tonnes: float = Field(..., ge=0, description="Physical market arrival volume today in Metric Tonnes")
    arrival_lag_1: float = Field(..., ge=0, description="Physical market arrival volume yesterday in Metric Tonnes")
    rolling_arrival_mean_7: float = Field(..., ge=0, description="7-session moving average arrival in Metric Tonnes")
    month: int = Field(..., ge=1, le=12, description="Calendar month (1 to 12)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_monsoon_season: int = Field(..., ge=0, le=1, description="Binary monsoon indicator (1 for Jun-Aug, 0 otherwise)")

class CommodityPricePredictionResponse(BaseModel):
    """Output payload from production prediction model."""
    status: str = Field(default="success")
    commodity: str
    market: str
    observed_price_rs_per_qtl: float
    predicted_next_price_rs_per_qtl: float
    observed_price_rs_per_kg: float
    predicted_next_price_rs_per_kg: float
    predicted_change_rs_per_qtl: float
    predicted_change_percentage: float
    prediction_horizon: str
    model_version: str
    timestamp: str

class FoodBasketItem(BaseModel):
    """Individual commodity in a household monthly food basket."""
    commodity_name: str = Field(..., description="Commodity name (e.g. Rice, Wheat, Tomato, Onion, Oil)")
    monthly_quantity_kg: float = Field(..., gt=0, description="Monthly household consumption in kilograms or litres")
    current_retail_price_per_kg: float = Field(..., gt=0, description="Current consumer retail price in Rs/Kg or Rs/Litre")
    predicted_retail_price_per_kg: Optional[float] = Field(None, gt=0, description="Predicted future retail price in Rs/Kg")

class HouseholdAffordabilityRequest(BaseModel):
    """Request payload for deterministic household affordability calculation."""
    household_id: Optional[str] = "HH-DEFAULT"
    household_members: int = Field(default=4, ge=1, le=20, description="Number of household members")
    monthly_household_income: float = Field(..., gt=0, description="Total monthly household income in Indian Rupees")
    sector: str = Field(default="Urban", description="Demographic sector: Urban or Rural")
    geography: str = Field(default="Maharashtra", description="State or region")
    basket_items: List[FoodBasketItem] = Field(..., description="List of food commodities consumed monthly")

class HouseholdAffordabilityResponse(BaseModel):
    """Response payload detailing food basket cost, budget share, and vulnerability index."""
    household_id: str
    monthly_income: float
    total_food_basket_cost_current: float
    total_food_basket_cost_predicted: Optional[float] = None
    current_food_budget_share_pct: float
    predicted_food_budget_share_pct: Optional[float] = None
    affordability_status: str
    nsso_benchmark_reference: str
    inflation_impact_monthly_diff_rs: Optional[float] = None
    vulnerability_alert: str
