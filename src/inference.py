"""
Production Inference Engine for AI-Based Food Inflation & Affordability Monitoring.

Version: v3.0.0-leakfree
Key Guarantees:
1. Dynamic commodity & market validation against trained model metadata (raises
   UnsupportedCommodityError on unseen commodities unless explicit experimental flag set).
2. Zero lookahead leakage: operates on the exact 16-feature (Specialist) or 20-feature
   (Unified) schema including current observed session `modal_price_rs_per_qtl`.
3. Never silently fabricates 7-day rolling lags in production mode (`predict_from_features`).
4. Explicit Early-Warning 3-Class Classifier (`spike_classifier.joblib`) integration with
   honest reliability warning from untouched Test PR-AUC metadata.
5. Clear separation between Wholesale Modal Price (`Rs/Quintal`), mathematical unit
   conversion (`Wholesale Equiv Rs/Kg = price / 100`), and observed DCA Retail reference
   (`retail_estimation_status = "unavailable"` where paired longitudinal data is absent).
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features import (
    COMMODITY_CATEGORY_MAP,
    SPECIALIST_NUMERIC_FEATURES,
    UNIFIED_ALL_FEATURES,
    UNIFIED_CATEGORICAL_FEATURES,
    UNIFIED_NUMERIC_FEATURES,
    canonicalize_commodity_name,
)
from src.live_data import get_observed_retail_price

logger = logging.getLogger(__name__)

PREDICTION_HORIZON_TEXT = (
    "Next observed active APMC trading session (t+1; not guaranteed next calendar day)"
)


class UnsupportedCommodityError(ValueError):
    """Raised when inference is requested for a commodity not validated during training."""


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_alert_badge(change_pct: float) -> Tuple[str, str]:
    """Classify regression forecast delta into interpretable directional alerts."""
    if change_pct > 10.0:
        return "HIGH_SURGE_RISK", f"High Surge Alert (+{change_pct:.2f}% > +10%)"
    if change_pct > 3.0:
        return "MODERATE_UPWARD_PRESSURE", f"Moderate Upward Inflation (+{change_pct:.2f}%)"
    if change_pct < -10.0:
        return "SHARP_CRASH_RISK", f"Sharp Price Correction Risk ({change_pct:.2f}% < -10%)"
    if change_pct < -3.0:
        return "MODERATE_DECLINE", f"Moderate Price Decline ({change_pct:.2f}%)"
    return "STABLE", f"Stable Band ({change_pct:+.2f}%)"


def _validate_finite_positive(name: str, val: Any, allow_zero: bool = False) -> float:
    try:
        fval = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature '{name}' must be numeric, got {val!r}.") from exc
    if not math.isfinite(fval):
        raise ValueError(f"Feature '{name}' must be finite, got {val!r}.")
    if allow_zero:
        if fval < 0:
            raise ValueError(f"Feature '{name}' cannot be negative, got {fval}.")
    else:
        if fval <= 0:
            raise ValueError(f"Feature '{name}' must be strictly positive, got {fval}.")
    return fval


def get_retail_status_info(models_dir: Path) -> Dict[str, Any]:
    """Read audited wholesale-to-retail feasibility status from models/retail_model_metadata.json."""
    meta = _load_json(models_dir / "retail_model_metadata.json")
    if not meta:
        return {
            "retail_estimation_supported": False,
            "status": "unavailable_audit_missing",
            "explanation": (
                "Wholesale-to-retail ML estimation unavailable (audited paired longitudinal "
                "dataset overlap absent)."
            ),
        }
    return meta


class EarlyWarningClassifier:
    """Wrapper around the trained 3-class (Normal / Surge / Crash) early-warning classifier."""

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "spike_classifier.joblib"
        self.meta_path = self.models_dir / "spike_classifier_metadata.json"
        self.available = self.model_path.exists() and self.meta_path.exists()
        self.pipeline = joblib.load(self.model_path) if self.model_path.exists() else None
        self.metadata = _load_json(self.meta_path) if self.meta_path.exists() else {}

    def predict_from_features(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict next-session regime probabilities with honest reliability flagging."""
        if not self.available or self.pipeline is None:
            return {
                "available": False,
                "predicted_regime": "Unavailable",
                "class_probabilities": {},
                "any_spike_probability": None,
                "is_reliable_early_warning": False,
                "reliability_status": "Classifier artifact spike_classifier.joblib not found.",
            }

        row: Dict[str, Any] = {}
        for col in UNIFIED_ALL_FEATURES:
            if col not in features_dict:
                raise ValueError(f"Missing classifier feature '{col}'.")
            val = features_dict[col]
            if col in UNIFIED_CATEGORICAL_FEATURES:
                row[col] = str(val)
            else:
                fval = float(val)
                if not math.isfinite(fval):
                    raise ValueError(f"Classifier feature '{col}' is not finite: {val!r}")
                row[col] = fval

        X = pd.DataFrame([row])
        for col in UNIFIED_CATEGORICAL_FEATURES:
            X[col] = X[col].astype(str).astype(object)

        pred_label = str(self.pipeline.predict(X)[0])
        proba_arr = self.pipeline.predict_proba(X)[0]
        clf_classes = [str(c) for c in self.pipeline.named_steps["clf"].classes_]
        class_probs = {cls: round(float(prob), 4) for cls, prob in zip(clf_classes, proba_arr)}
        for canonical_cls in ("Normal", "Surge", "Crash"):
            class_probs.setdefault(canonical_cls, 0.0)

        any_spike_prob = round(class_probs.get("Surge", 0.0) + class_probs.get("Crash", 0.0), 4)
        test_metrics = self.metadata.get("test_metrics", {})

        return {
            "available": True,
            "predicted_regime": pred_label,
            "class_probabilities": class_probs,
            "any_spike_probability": any_spike_prob,
            "is_reliable_early_warning": bool(self.metadata.get("is_reliable_early_warning", False)),
            "reliability_status": self.metadata.get(
                "reliability_status",
                "Experimental Early-Warning Classifier (review Test PR-AUC before operational reliance).",
            ),
            "champion_algorithm": self.metadata.get("champion_algorithm", "Unknown"),
            "spike_definition": self.metadata.get(
                "spike_definition",
                "Surge > +10% | Crash < -10% | Normal within [-10%, +10%]",
            ),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_any_spike_pr_auc": test_metrics.get("any_spike_pr_auc"),
        }


class CommodityPricePredictor:
    """Dedicated single-commodity wholesale price predictor (e.g., Pune Tomato or Pune Onion)."""

    def __init__(self, models_dir: str | Path = "models", commodity: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.requested_commodity = canonicalize_commodity_name(commodity) if commodity else None

        if self.requested_commodity:
            safe_name = self.requested_commodity.lower().replace(" ", "_")
            model_path = self.models_dir / f"price_prediction_model_{safe_name}.joblib"
            meta_path = self.models_dir / f"model_metadata_{safe_name}.json"
            if not model_path.exists() or not meta_path.exists():
                default_meta = _load_json(self.models_dir / "model_metadata.json")
                default_comm = canonicalize_commodity_name(str(default_meta.get("commodity", "")))
                if default_comm == self.requested_commodity and (self.models_dir / "price_prediction_model.joblib").exists():
                    model_path = self.models_dir / "price_prediction_model.joblib"
                    meta_path = self.models_dir / "model_metadata.json"
                else:
                    raise UnsupportedCommodityError(
                        f"No dedicated specialist model exists for '{self.requested_commodity}'. "
                        f"Available specialists are Tomato and Onion."
                    )
        else:
            model_path = self.models_dir / "price_prediction_model.joblib"
            meta_path = self.models_dir / "model_metadata.json"
            if not model_path.exists():
                model_path = self.models_dir / "price_prediction_model_tomato.joblib"
                meta_path = self.models_dir / "model_metadata_tomato.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Specialist model not found at {model_path}")

        self.model = joblib.load(model_path)
        self.metadata = _load_json(meta_path)
        self.retail_metadata = get_retail_status_info(self.models_dir)
        self.commodity = str(self.metadata.get("commodity", self.requested_commodity or "Tomato"))
        self.market = str(self.metadata.get("market", "Pune"))
        self.feature_names: List[str] = self.metadata.get("feature_names", SPECIALIST_NUMERIC_FEATURES)

    def predict_from_features(
        self,
        features_dict: Dict[str, Any],
        feature_mode: str = "validated_time_series",
    ) -> Dict[str, Any]:
        """Predict next active session wholesale price from complete 16-feature dictionary."""
        row: Dict[str, float] = {}
        for col in self.feature_names:
            if col not in features_dict:
                raise ValueError(f"Missing required specialist feature '{col}'.")
            allow_zero = col in {
                "rolling_std_7",
                "price_momentum_7",
                "daily_spread",
                "arrival_tonnes",
                "arrival_lag_1",
                "rolling_arrival_mean_7",
                "day_of_week",
                "is_monsoon_season",
            }
            if col == "price_momentum_7":
                fval = float(features_dict[col])
                if not math.isfinite(fval):
                    raise ValueError("price_momentum_7 must be finite.")
                row[col] = fval
            else:
                row[col] = _validate_finite_positive(col, features_dict[col], allow_zero=allow_zero)

        current_price_qtl = row["modal_price_rs_per_qtl"]
        X_df = pd.DataFrame([row])[self.feature_names]
        raw_pred = float(self.model.predict(X_df)[0])
        pred_price_qtl = round(max(1.0, raw_pred), 2)

        change_rs_qtl = round(pred_price_qtl - current_price_qtl, 2)
        change_pct = round((change_rs_qtl / current_price_qtl) * 100.0, 2)
        wholesale_equiv_kg = round(pred_price_qtl / 100.0, 2)
        current_equiv_kg = round(current_price_qtl / 100.0, 2)
        alert_level, alert_badge = _format_alert_badge(change_pct)

        observed_retail = get_observed_retail_price(self.commodity)

        return {
            "commodity": self.commodity,
            "market": self.market,
            "current_modal_price_rs_per_qtl": round(current_price_qtl, 2),
            "current_price_rs_per_qtl": round(current_price_qtl, 2),
            "predicted_next_session_rs_per_qtl": pred_price_qtl,
            "predicted_modal_price_rs_per_qtl": pred_price_qtl,
            "predicted_price_rs_per_qtl": pred_price_qtl,
            "wholesale_equiv_rs_per_kg": wholesale_equiv_kg,
            "current_wholesale_equiv_rs_per_kg": current_equiv_kg,
            "predicted_change_rs_qtl": change_rs_qtl,
            "predicted_change_pct": change_pct,
            "price_change_pct": change_pct,
            "prediction_horizon": self.metadata.get("prediction_horizon", PREDICTION_HORIZON_TEXT),
            "target_definition": self.metadata.get("target_definition", PREDICTION_HORIZON_TEXT),
            "model_used": f"{self.metadata.get('model_name', 'Specialist Regressor')} ({self.commodity} @ {self.market})",
            "dataset_version": self.metadata.get("dataset_version", "v3.0.0-leakfree"),
            "feature_mode": feature_mode,
            "inflation_alert_level": alert_level,
            "inflation_alert_badge": alert_badge,
            "retail_estimation_status": self.retail_metadata.get("status", "unavailable"),
            "retail_estimation_supported": bool(self.retail_metadata.get("retail_estimation_supported", False)),
            "retail_estimation_note": self.retail_metadata.get(
                "explanation",
                "Retail estimation unavailable due to lack of longitudinal paired wholesale-retail records.",
            ),
            "observed_retail_reference": observed_retail,
        }

    def predict(
        self,
        modal_price: Optional[float] = None,
        price_lag_1: Optional[float] = None,
        price_lag_3: Optional[float] = None,
        price_lag_7: Optional[float] = None,
        rolling_mean_7: Optional[float] = None,
        rolling_std_7: float = 0.0,
        rolling_min_7: Optional[float] = None,
        rolling_max_7: Optional[float] = None,
        arrival_tonnes: float = 100.0,
        arrival_lag_1: Optional[float] = None,
        rolling_arrival_mean_7: Optional[float] = None,
        daily_spread: float = 0.0,
        month: int = 11,
        day_of_week: int = 2,
        allow_manual_approx: bool = False,
    ) -> Dict[str, Any]:
        """Run specialist inference from explicit parameters (manual approximation opt-in only)."""
        missing_strict = [
            name
            for name, v in [
                ("modal_price", modal_price),
                ("price_lag_1", price_lag_1),
                ("price_lag_3", price_lag_3),
                ("price_lag_7", price_lag_7),
                ("rolling_mean_7", rolling_mean_7),
                ("rolling_min_7", rolling_min_7),
                ("rolling_max_7", rolling_max_7),
                ("arrival_lag_1", arrival_lag_1),
                ("rolling_arrival_mean_7", rolling_arrival_mean_7),
            ]
            if v is None
        ]

        if missing_strict and not allow_manual_approx:
            raise ValueError(
                f"Cannot fabricate lag/rolling features in strict mode. Missing: {missing_strict}. "
                f"Supply real historical/live series features or explicitly pass allow_manual_approx=True."
            )

        if modal_price is None and price_lag_1 is None:
            raise ValueError("modal_price must be provided.")

        cur = _validate_finite_positive("modal_price", modal_price if modal_price is not None else price_lag_1)
        lag1 = float(price_lag_1 if price_lag_1 is not None else cur)
        lag3 = float(price_lag_3 if price_lag_3 is not None else lag1)
        lag7 = float(price_lag_7 if price_lag_7 is not None else lag3)
        r_mean = float(rolling_mean_7 if rolling_mean_7 is not None else np.mean([cur, lag1, lag3, lag7]))
        r_min = float(rolling_min_7 if rolling_min_7 is not None else min(cur, lag1, lag3, lag7))
        r_max = float(rolling_max_7 if rolling_max_7 is not None else max(cur, lag1, lag3, lag7))
        arr = _validate_finite_positive("arrival_tonnes", arrival_tonnes, allow_zero=True)
        arr_lag1 = float(arrival_lag_1 if arrival_lag_1 is not None else arr)
        arr_mean7 = float(rolling_arrival_mean_7 if rolling_arrival_mean_7 is not None else (arr + arr_lag1) / 2.0)

        features = {
            "modal_price_rs_per_qtl": cur,
            "price_lag_1": lag1,
            "price_lag_3": lag3,
            "price_lag_7": lag7,
            "rolling_mean_7": r_mean,
            "rolling_std_7": float(rolling_std_7),
            "rolling_min_7": r_min,
            "rolling_max_7": r_max,
            "price_momentum_7": ((cur - lag7) / lag7 * 100.0) if lag7 > 0 else 0.0,
            "daily_spread": float(daily_spread),
            "arrival_tonnes": arr,
            "arrival_lag_1": arr_lag1,
            "rolling_arrival_mean_7": arr_mean7,
            "month": int(month),
            "day_of_week": int(day_of_week),
            "is_monsoon_season": 1 if int(month) in (6, 7, 8, 9) else 0,
        }
        mode = "manual_approximated_fallback" if missing_strict else "manual_user_entered"
        return self.predict_from_features(features, feature_mode=mode)


class UnifiedCommodityPredictor:
    """
    Unified Multi-Commodity Forecaster with strict metadata-backed validation,
    unsupported-commodity rejection, early-warning classification, and honest retail framing.
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "unified_food_price_model.joblib"
        self.meta_path = self.models_dir / "unified_model_metadata.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Unified model artifact not found: {self.model_path}")

        self.pipeline = joblib.load(self.model_path)
        self.metadata = _load_json(self.meta_path)
        self.retail_metadata = get_retail_status_info(self.models_dir)
        self.classifier = EarlyWarningClassifier(models_dir=self.models_dir)

        self.supported_commodities: List[str] = self.metadata.get(
            "supported_commodities", ["Moong Dal", "Onion", "Sugar", "Tomato"]
        )
        self.supported_markets: List[str] = self.metadata.get(
            "supported_markets", ["Lasalgaon", "Mumbai", "Pune"]
        )
        self.supported_pairs: List[Tuple[str, str]] = [
            (str(p["commodity"]), str(p["market"]))
            for p in self.metadata.get("supported_commodity_market_pairs", [])
        ]

    def get_supported_commodities(self) -> List[str]:
        return list(self.supported_commodities)

    def get_supported_markets(self, commodity: Optional[str] = None) -> List[str]:
        if not commodity:
            return list(self.supported_markets)
        canon = canonicalize_commodity_name(commodity)
        paired = sorted({m for c, m in self.supported_pairs if c == canon})
        return paired if paired else list(self.supported_markets)

    def validate_commodity(
        self, commodity: str, allow_experimental: bool = False
    ) -> Tuple[str, bool, Optional[str]]:
        canon = canonicalize_commodity_name(commodity)
        is_supported = canon in self.supported_commodities
        if not is_supported and not allow_experimental:
            loco = self.metadata.get("leave_one_commodity_out_experiment", {})
            raise UnsupportedCommodityError(
                f"Commodity '{commodity}' (canonical: '{canon}') is NOT in the validated training "
                f"coverage {self.supported_commodities}. Leave-One-Commodity-Out experiments showed "
                f"unseen-crop OHE generalization does NOT reliably beat persistence baselines "
                f"({len(loco)} tested holdouts; only 1/4 beat persistence). Use one of the "
                f"validated commodities or pass allow_experimental=True for unvalidated exploration."
            )
        warning = None
        if not is_supported:
            warning = (
                f"EXPERIMENTAL UNVALIDATED COMMODITY: '{canon}' was never seen during model training. "
                f"Metrics are unvalidated; do not rely on this forecast for decisions."
            )
        return canon, is_supported, warning

    def predict_from_features(
        self,
        features_dict: Dict[str, Any],
        allow_experimental: bool = False,
        feature_mode: str = "validated_time_series",
    ) -> Dict[str, Any]:
        """Execute leak-free multi-commodity inference from complete 20-feature dictionary."""
        raw_comm = str(features_dict.get("commodity_name") or features_dict.get("commodity") or "")
        if not raw_comm:
            raise ValueError("'commodity_name' is required in features_dict.")

        canon_comm, is_validated_comm, comm_warning = self.validate_commodity(
            raw_comm, allow_experimental=allow_experimental
        )
        market = str(features_dict.get("market_name") or features_dict.get("market") or "").strip()
        if not market:
            raise ValueError("'market_name' is required in features_dict.")

        default_cat, default_perish = COMMODITY_CATEGORY_MAP.get(canon_comm, ("Vegetables", 1))
        category = str(features_dict.get("category") or default_cat)
        is_perishable = int(features_dict.get("is_perishable", default_perish))

        is_validated_pair = (canon_comm, market) in self.supported_pairs if self.supported_pairs else True
        pair_warning = None
        if is_validated_comm and not is_validated_pair:
            valid_mkts = self.get_supported_markets(canon_comm)
            pair_warning = (
                f"Market mismatch notice: '{canon_comm}' was evaluated during training in {valid_mkts}, "
                f"not '{market}'. Forecast uses OneHot unknown/cross-market encoding and is unvalidated for this pair."
            )

        numeric_values: Dict[str, float] = {}
        for col in UNIFIED_NUMERIC_FEATURES:
            if col == "is_perishable":
                numeric_values[col] = float(is_perishable)
                continue
            if col not in features_dict:
                raise ValueError(f"Missing required numeric feature '{col}'.")
            allow_zero = col in {
                "rolling_std_7",
                "price_momentum_7",
                "daily_spread",
                "arrival_tonnes",
                "arrival_lag_1",
                "rolling_arrival_mean_7",
                "day_of_week",
                "is_monsoon_season",
            }
            if col == "price_momentum_7":
                fv = float(features_dict[col])
                if not math.isfinite(fv):
                    raise ValueError("price_momentum_7 must be finite.")
                numeric_values[col] = fv
            else:
                numeric_values[col] = _validate_finite_positive(col, features_dict[col], allow_zero=allow_zero)

        full_row: Dict[str, Any] = {
            "commodity_name": canon_comm,
            "category": category,
            "market_name": market,
            **numeric_values,
        }

        X_df = pd.DataFrame([full_row])[UNIFIED_ALL_FEATURES]
        for col in UNIFIED_CATEGORICAL_FEATURES:
            X_df[col] = X_df[col].astype(str).astype(object)

        current_price_qtl = float(numeric_values["modal_price_rs_per_qtl"])
        raw_pred = float(self.pipeline.predict(X_df)[0])
        pred_price_qtl = round(max(1.0, raw_pred), 2)

        change_rs_qtl = round(pred_price_qtl - current_price_qtl, 2)
        change_pct = round((change_rs_qtl / current_price_qtl) * 100.0, 2)
        wholesale_equiv_kg = round(pred_price_qtl / 100.0, 2)
        current_equiv_kg = round(current_price_qtl / 100.0, 2)
        alert_level, alert_badge = _format_alert_badge(change_pct)

        if is_validated_comm and self.classifier.available:
            ew_res = self.classifier.predict_from_features(full_row)
        else:
            ew_res = {
                "available": False,
                "predicted_regime": "Unavailable",
                "class_probabilities": {},
                "any_spike_probability": None,
                "is_reliable_early_warning": False,
                "reliability_status": "Skipped for unvalidated commodity or missing classifier artifact.",
            }

        combined_warnings = [w for w in (comm_warning, pair_warning) if w]
        if feature_mode == "manual_approximated_fallback":
            combined_warnings.append(
                "Manual approximated lag features were supplied instead of observed historical/live daily series."
            )

        observed_retail = get_observed_retail_price(canon_comm)

        return {
            "commodity": canon_comm,
            "category": category,
            "market": market,
            "is_perishable": bool(is_perishable),
            "is_validated_commodity": bool(is_validated_comm),
            "is_validated_pair": bool(is_validated_pair),
            "validation_warning": " | ".join(combined_warnings) if combined_warnings else None,
            "current_modal_price_rs_per_qtl": round(current_price_qtl, 2),
            "current_price_rs_per_qtl": round(current_price_qtl, 2),
            "predicted_next_session_rs_per_qtl": pred_price_qtl,
            "predicted_modal_price_rs_per_qtl": pred_price_qtl,
            "predicted_price_rs_per_qtl": pred_price_qtl,
            "wholesale_equiv_rs_per_kg": wholesale_equiv_kg,
            "current_wholesale_equiv_rs_per_kg": current_equiv_kg,
            "predicted_change_rs_qtl": change_rs_qtl,
            "predicted_change_pct": change_pct,
            "price_change_pct": change_pct,
            "prediction_horizon": self.metadata.get("prediction_horizon", PREDICTION_HORIZON_TEXT),
            "model_used": f"Unified {self.metadata.get('champion_algorithm', 'Ridge Regression (L2)')}",
            "dataset_version": self.metadata.get("dataset_version", "v3.0.0-leakfree-unified"),
            "feature_mode": feature_mode,
            "inflation_alert_level": alert_level,
            "inflation_alert_badge": alert_badge,
            "early_warning_classifier": ew_res,
            "retail_estimation_status": self.retail_metadata.get("status", "unavailable"),
            "retail_estimation_supported": bool(self.retail_metadata.get("retail_estimation_supported", False)),
            "retail_estimation_note": self.retail_metadata.get(
                "explanation",
                "Retail estimation unavailable due to zero matched same-city historical wholesale-retail overlap.",
            ),
            "observed_retail_reference": observed_retail,
        }

    def predict(
        self,
        commodity: str,
        category: Optional[str] = None,
        market: str = "Pune",
        modal_price: Optional[float] = None,
        price_lag_1: Optional[float] = None,
        price_lag_3: Optional[float] = None,
        price_lag_7: Optional[float] = None,
        rolling_mean_7: Optional[float] = None,
        rolling_std_7: float = 0.0,
        rolling_min_7: Optional[float] = None,
        rolling_max_7: Optional[float] = None,
        arrival_tonnes: float = 100.0,
        arrival_lag_1: Optional[float] = None,
        rolling_arrival_mean_7: Optional[float] = None,
        daily_spread: float = 0.0,
        month: int = 11,
        day_of_week: int = 2,
        is_perishable: Optional[int] = None,
        allow_experimental: bool = False,
        allow_manual_approx: bool = False,
    ) -> Dict[str, Any]:
        """Run multi-commodity inference with strict unsupported-crop & missing-lag checks."""
        canon_comm, is_supported, _ = self.validate_commodity(commodity, allow_experimental=allow_experimental)

        missing_strict = [
            name
            for name, v in [
                ("modal_price", modal_price),
                ("price_lag_1", price_lag_1),
                ("price_lag_3", price_lag_3),
                ("price_lag_7", price_lag_7),
                ("rolling_mean_7", rolling_mean_7),
                ("rolling_min_7", rolling_min_7),
                ("rolling_max_7", rolling_max_7),
                ("arrival_lag_1", arrival_lag_1),
                ("rolling_arrival_mean_7", rolling_arrival_mean_7),
            ]
            if v is None
        ]

        if missing_strict and not allow_manual_approx:
            raise ValueError(
                f"Refusing to manufacture missing lag/rolling features in strict mode: {missing_strict}. "
                f"Use get_historical_latest_features()/fetch_latest_market_data() + predict_from_features(), "
                f"or pass allow_manual_approx=True explicitly for manual exploration."
            )

        if modal_price is None and price_lag_1 is None:
            raise ValueError("modal_price must be provided.")

        cur = _validate_finite_positive("modal_price", modal_price if modal_price is not None else price_lag_1)
        lag1 = float(price_lag_1 if price_lag_1 is not None else cur)
        lag3 = float(price_lag_3 if price_lag_3 is not None else lag1)
        lag7 = float(price_lag_7 if price_lag_7 is not None else lag3)
        r_mean = float(rolling_mean_7 if rolling_mean_7 is not None else np.mean([cur, lag1, lag3, lag7]))
        r_min = float(rolling_min_7 if rolling_min_7 is not None else min(cur, lag1, lag3, lag7))
        r_max = float(rolling_max_7 if rolling_max_7 is not None else max(cur, lag1, lag3, lag7))
        arr = _validate_finite_positive("arrival_tonnes", arrival_tonnes, allow_zero=True)
        arr_lag1 = float(arrival_lag_1 if arrival_lag_1 is not None else arr)
        arr_mean7 = float(rolling_arrival_mean_7 if rolling_arrival_mean_7 is not None else (arr + arr_lag1) / 2.0)

        default_cat, default_perish = COMMODITY_CATEGORY_MAP.get(canon_comm, ("Vegetables", 1))
        resolved_cat = category or default_cat
        resolved_perish = int(is_perishable) if is_perishable is not None else default_perish

        features = {
            "commodity_name": canon_comm,
            "category": resolved_cat,
            "market_name": market,
            "modal_price_rs_per_qtl": cur,
            "price_lag_1": lag1,
            "price_lag_3": lag3,
            "price_lag_7": lag7,
            "rolling_mean_7": r_mean,
            "rolling_std_7": float(rolling_std_7),
            "rolling_min_7": r_min,
            "rolling_max_7": r_max,
            "price_momentum_7": ((cur - lag7) / lag7 * 100.0) if lag7 > 0 else 0.0,
            "daily_spread": float(daily_spread),
            "arrival_tonnes": arr,
            "arrival_lag_1": arr_lag1,
            "rolling_arrival_mean_7": arr_mean7,
            "month": int(month),
            "day_of_week": int(day_of_week),
            "is_monsoon_season": 1 if int(month) in (6, 7, 8, 9) else 0,
            "is_perishable": resolved_perish,
        }

        if not is_supported:
            f_mode = "experimental_unseen_commodity"
        elif missing_strict:
            f_mode = "manual_approximated_fallback"
        else:
            f_mode = "manual_user_entered"

        return self.predict_from_features(
            features, allow_experimental=allow_experimental, feature_mode=f_mode
        )


if __name__ == "__main__":
    p_uni = UnifiedCommodityPredictor()
    print("Supported Commodities:", p_uni.get_supported_commodities())
    print("Supported Pairs:", p_uni.supported_pairs)
