"""
Pydantic v2 Data Contracts for AI-Based Food Inflation & Affordability Monitoring.

Schema Version: 3.1.0-parity
Synchronized verbatim with `reports/feature_contract.md`, `src/features.py`, and `src/inference.py`.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CommodityPricePredictionRequest(BaseModel):
    """Input schema for next-observed-trading-session wholesale price prediction."""

    commodity_name: str = Field(..., description="Validated commodity name (Tomato, Onion, Moong Dal, Sugar)")
    market_name: str = Field(..., description="Exact APMC mandi market (Pune, Mumbai, Lasalgaon)")
    category: Optional[str] = Field(None, description="Commodity group (Vegetables, Pulses, Sugars & Sweets)")
    modal_price_rs_per_qtl: float = Field(..., gt=0, description="Current observed session wholesale modal price (Rs/Qtl)")
    price_lag_1: float = Field(..., gt=0, description="Modal price 1 trading session prior (Rs/Qtl)")
    price_lag_3: float = Field(..., gt=0, description="Modal price 3 trading sessions prior (Rs/Qtl)")
    price_lag_7: float = Field(..., gt=0, description="Modal price 7 trading sessions prior (Rs/Qtl)")
    rolling_mean_7: float = Field(..., gt=0, description="Mean modal price across previous 7 sessions [t-7..t-1], excluding current session t (Rs/Qtl)")
    rolling_std_7: float = Field(0.0, ge=0, description="Sample std dev across previous 7 sessions [t-7..t-1], excluding t (Rs/Qtl)")
    rolling_min_7: float = Field(..., gt=0, description="Minimum modal price across previous 7 sessions [t-7..t-1], excluding t (Rs/Qtl)")
    rolling_max_7: float = Field(..., gt=0, description="Maximum modal price across previous 7 sessions [t-7..t-1], excluding t (Rs/Qtl)")
    price_momentum_7: float = Field(..., description="7-session percentage momentum ((P_t - P_{t-7}) / P_{t-7}) * 100.0 (%)")
    daily_spread: float = Field(..., ge=0, description="Current session intraday spread max - min (Rs/Qtl)")
    arrival_tonnes: float = Field(..., ge=0, description="Current session genuine market arrival volume (Tonnes)")
    arrival_lag_1: float = Field(..., ge=0, description="Previous session genuine arrival volume (Tonnes)")
    rolling_arrival_mean_7: float = Field(..., ge=0, description="Mean arrival volume across previous 7 sessions [t-7..t-1] (Tonnes)")
    month: int = Field(..., ge=1, le=12, description="Calendar month integer (1-12)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week integer (0=Mon .. 6=Sun)")
    is_monsoon_season: int = Field(0, ge=0, le=1, description="1 if June-September monsoon months, else 0")
    is_perishable: int = Field(1, ge=0, le=1, description="1 if perishable vegetable crop, else 0")


class CommodityPricePredictionResponse(BaseModel):
    """Output schema for wholesale forecast, mathematical kg equiv, and early-warning classifier."""

    commodity: str
    market: str
    feature_schema_version: str
    inference_mode: str
    is_validated_forecast: bool
    current_modal_price_rs_per_qtl: float
    predicted_next_session_rs_per_qtl: float
    wholesale_equiv_rs_per_kg: float
    predicted_change_rs_qtl: float
    predicted_change_pct: float
    prediction_horizon: str = Field(
        default="Next observed active APMC trading session (t+1; not guaranteed next calendar day)"
    )
    model_used: str
    is_validated_commodity: bool
    is_validated_pair: bool
    inflation_alert_level: str
    inflation_alert_badge: str
    retail_estimation_status: str
    experimental_disclaimer: Optional[str] = None
    early_warning_classifier: Optional[Dict[str, object]] = None


class HouseholdBasketItem(BaseModel):
    """Single household food basket entry priced per Kg (observed DCA, user-supplied, or scenario)."""

    commodity_name: str = Field(..., min_length=1)
    monthly_quantity_kg: float = Field(..., ge=0)
    current_retail_price_per_kg: float = Field(..., ge=0)
    predicted_retail_price_per_kg: Optional[float] = Field(None, ge=0)
    price_source: Optional[str] = Field(None, description="Provenance label (Actual Observed DCA, User Supplied, Hypothetical Scenario)")


class HouseholdAffordabilityRequest(BaseModel):
    """Input schema for household food affordability & budget share assessment."""

    household_id: str = "HH-DEFAULT"
    monthly_income_rs: float = Field(..., gt=0, description="Strictly positive monthly household income (Rs)")
    family_size: int = Field(4, ge=1)
    sector: str = Field("Urban", description="Urban or Rural")
    basket_items: List[HouseholdBasketItem] = Field(..., min_length=1)
