"""
Production Inference Engine for AI-Based Food Inflation & Affordability Monitoring.

Version: v3.1.0-parity (Phases 1, 2, 5, 6)
Strict Guarantees:
1. 100% Training-Inference Feature Parity: uses exact shared helpers from `src/features.py`
   (`compute_price_momentum_7_pct` in %, strictly excluded-t rolling stats over [t-7..t-1]).
2. Two Explicit Inference Modes:
   - Mode A (`predict_validated` / `predict_from_features`): Requires real continuous APMC history,
     exact 7-session rolling features, genuine arrival measurements, validated commodity, and
     validated commodity-market pair (raises UnsupportedCommodityError / UnsupportedMarketPairError).
   - Mode B (`predict_experimental_manual`): Strictly labeled experimental exploration; excludes
     current price P(t) from previous-window summaries, discloses any subset approximations, and
     never claims Mode A reliability.
3. Schema Verification: checks matching model feature contract & version metadata cleanly.
4. Early-Warning Classifier integration (`spike_classifier.joblib`) + zero fabricated retail prices.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features import (
    COMMODITY_CATEGORY_MAP,
    FEATURE_SCHEMA_VERSION,
    SPECIALIST_NUMERIC_FEATURES,
    UNIFIED_ALL_FEATURES,
    UNIFIED_CATEGORICAL_FEATURES,
    UNIFIED_NUMERIC_FEATURES,
    canonicalize_commodity_name,
    canonicalize_market_name,
    compute_previous_window_price_stats,
    compute_price_momentum_7_pct,
)
from src.live_data import get_observed_retail_price

logger = logging.getLogger(__name__)

PREDICTION_HORIZON_TEXT = (
    "Next observed active APMC trading session (t+1; not guaranteed next calendar day)"
)

EXPERIMENTAL_MANUAL_DISCLAIMER = (
    "EXPERIMENTAL EXPLORATION ONLY (Mode B): Computed from manually entered scenario values "
    "and/or subset approximations rather than a complete continuous 7-session APMC series. "
    "Do NOT interpret as a validated forecast."
)


class UnsupportedCommodityError(ValueError):
    """Raised when inference is requested for a commodity not validated during training."""


class UnsupportedMarketPairError(ValueError):
    """Raised in Mode A when (commodity, market) was not validated in training/evaluation."""


class FeatureSchemaMismatchError(ValueError):
    """Raised when incoming features violate required contract columns or schema version."""


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_alert_badge(change_pct: float) -> Tuple[str, str]:
    """Classify regression forecast delta into interpretable directional bands."""
    if change_pct > 10.0:
        return "HIGH_SURGE_RISK", f"High Surge Alert (+{change_pct:.2f}% > +10%)"
    if change_pct > 3.0:
        return "MODERATE_UPWARD_PRESSURE", f"Moderate Upward Inflation (+{change_pct:.2f}%)"
    if change_pct < -10.0:
        return "SHARP_CRASH_RISK", f"Sharp Price Correction Risk ({change_pct:.2f}% < -10%)"
    if change_pct < -3.0:
        return "MODERATE_DECLINE", f"Moderate Price Decline ({change_pct:.2f}%)"
    return "STABLE", f"Stable Band ({change_pct:+.2f}%)"


def _validate_finite_number(name: str, val: Any, positive_only: bool = False, non_negative: bool = False) -> float:
    if val is None or isinstance(val, bool):
        raise ValueError(f"Feature '{name}' must be a valid finite number, got {val!r}.")
    try:
        fval = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature '{name}' must be numeric, got {val!r}.") from exc
    if not math.isfinite(fval):
        raise ValueError(f"Feature '{name}' must be finite (no NaN/Inf), got {val!r}.")
    if positive_only and fval <= 0.0:
        raise ValueError(f"Feature '{name}' must be strictly positive (> 0), got {fval}.")
    if non_negative and fval < 0.0:
        raise ValueError(f"Feature '{name}' cannot be negative (< 0), got {fval}.")
    return fval


def _validate_numeric_feature_dict(
    features_dict: Dict[str, Any],
    expected_features: Sequence[str],
    expected_schema_version: Optional[str] = None,
) -> Dict[str, float]:
    if not isinstance(features_dict, dict):
        raise FeatureSchemaMismatchError("features_dict must be a dictionary.")

    incoming_ver = features_dict.get("feature_schema_version")
    if (
        expected_schema_version
        and incoming_ver is not None
        and str(incoming_ver) != str(expected_schema_version)
    ):
        raise FeatureSchemaMismatchError(
            f"Feature schema version mismatch: model expects '{expected_schema_version}', "
            f"got '{incoming_ver}'."
        )

    missing = [col for col in expected_features if col not in features_dict]
    if missing:
        raise FeatureSchemaMismatchError(
            f"Missing required features matching schema {expected_schema_version or FEATURE_SCHEMA_VERSION}: {missing}"
        )

    positive_cols = {
        "modal_price_rs_per_qtl",
        "price_lag_1",
        "price_lag_3",
        "price_lag_7",
        "rolling_mean_7",
        "rolling_min_7",
        "rolling_max_7",
    }
    non_neg_cols = {
        "rolling_std_7",
        "daily_spread",
        "arrival_tonnes",
        "arrival_lag_1",
        "rolling_arrival_mean_7",
        "day_of_week",
        "is_monsoon_season",
        "is_perishable",
    }

    parsed: Dict[str, float] = {}
    for col in expected_features:
        fval = _validate_finite_number(
            col,
            features_dict[col],
            positive_only=(col in positive_cols),
            non_negative=(col in non_neg_cols),
        )
        if col == "month" and not (1 <= int(fval) <= 12):
            raise ValueError(f"month must be between 1 and 12, got {fval}.")
        if col == "day_of_week" and not (0 <= int(fval) <= 6):
            raise ValueError(f"day_of_week must be between 0 and 6, got {fval}.")
        if col in ("is_monsoon_season", "is_perishable") and fval not in (0.0, 1.0):
            raise ValueError(f"{col} must be binary 0 or 1, got {fval}.")
        parsed[col] = fval

    # Structural sanity invariants across bounds
    if parsed["rolling_min_7"] > parsed["rolling_max_7"]:
        raise ValueError(
            f"Invalid window ordering: rolling_min_7 ({parsed['rolling_min_7']}) > rolling_max_7 ({parsed['rolling_max_7']})."
        )
    return parsed


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
    """Wrapper around trained 3-class (Normal / Surge / Crash) early-warning classifier."""

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
        for col in UNIFIED_CATEGORICAL_FEATURES:
            if col not in features_dict or str(features_dict[col]).strip() == "":
                raise FeatureSchemaMismatchError(f"Missing classifier categorical feature '{col}'.")
            row[col] = str(features_dict[col]).strip()

        num_parsed = _validate_numeric_feature_dict(
            features_dict,
            UNIFIED_NUMERIC_FEATURES,
            expected_schema_version=self.metadata.get("feature_schema_version"),
        )
        row.update(num_parsed)

        X = pd.DataFrame([row])[UNIFIED_ALL_FEATURES]
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
    """Dedicated single-commodity wholesale price predictor (Tomato @ Pune / Onion @ Pune)."""

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
        self.feature_schema_version: str = str(
            self.metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        )

    def _run_validated_pipeline(
        self,
        row: Dict[str, float],
        is_validated_forecast: bool,
        inference_mode: str,
        experimental_disclaimer: Optional[str] = None,
        approximations_used: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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
            "feature_schema_version": self.feature_schema_version,
            "inference_mode": inference_mode,
            "is_validated_forecast": is_validated_forecast,
            "experimental_disclaimer": experimental_disclaimer,
            "approximations_used": approximations_used or [],
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
            "dataset_version": self.metadata.get("dataset_version", "v3.1.0-parity"),
            "feature_mode": inference_mode,
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

    def predict_validated(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Mode A: Strict validated prediction requiring canonical 16 features from genuine history."""
        row = _validate_numeric_feature_dict(
            features_dict,
            self.feature_names,
            expected_schema_version=self.feature_schema_version,
        )
        return self._run_validated_pipeline(
            row,
            is_validated_forecast=True,
            inference_mode="Mode A: Validated Forecasting",
        )

    def predict_from_features(
        self,
        features_dict: Dict[str, Any],
        feature_mode: str = "Mode A: Validated Forecasting",
    ) -> Dict[str, Any]:
        if "experimental" in str(feature_mode).lower():
            row = _validate_numeric_feature_dict(
                features_dict,
                self.feature_names,
                expected_schema_version=self.feature_schema_version,
            )
            return self._run_validated_pipeline(
                row,
                is_validated_forecast=False,
                inference_mode="Mode B: Experimental Manual Exploration",
                experimental_disclaimer=EXPERIMENTAL_MANUAL_DISCLAIMER,
            )
        return self.predict_validated(features_dict)

    def predict_experimental_manual(
        self,
        modal_price: float,
        price_lag_1: float,
        price_lag_3: float,
        price_lag_7: float,
        arrival_tonnes: float,
        arrival_lag_1: Optional[float] = None,
        rolling_arrival_mean_7: Optional[float] = None,
        prior_7_prices: Optional[Sequence[float]] = None,
        daily_spread: float = 0.0,
        month: int = 11,
        day_of_week: int = 2,
    ) -> Dict[str, Any]:
        """
        Mode B: Explicit Experimental Manual Exploration.
        Never includes current modal_price in previous-window [t-7..t-1] stats and
        clearly lists any subset approximations in `approximations_used`.
        """
        cur = _validate_finite_number("modal_price", modal_price, positive_only=True)
        l1 = _validate_finite_number("price_lag_1", price_lag_1, positive_only=True)
        l3 = _validate_finite_number("price_lag_3", price_lag_3, positive_only=True)
        l7 = _validate_finite_number("price_lag_7", price_lag_7, positive_only=True)
        arr = _validate_finite_number("arrival_tonnes", arrival_tonnes, non_negative=True)
        spread = _validate_finite_number("daily_spread", daily_spread, non_negative=True)

        approx: List[str] = []
        if prior_7_prices is not None and len(list(prior_7_prices)) == 7:
            stats = compute_previous_window_price_stats(prior_7_prices)
        else:
            # Exclude P(t) strictly: summarize only user-entered past lags [t-1, t-3, t-7]
            stats = compute_previous_window_price_stats([l1, l3, l7])
            approx.append("rolling_*_7 computed from 3 selected lags [t-1, t-3, t-7] (approximate, excludes t)")

        if arrival_lag_1 is None:
            arr_l1 = arr
            approx.append("arrival_lag_1 defaulted from user arrival input")
        else:
            arr_l1 = _validate_finite_number("arrival_lag_1", arrival_lag_1, non_negative=True)

        if rolling_arrival_mean_7 is None:
            arr_mean7 = (arr + arr_l1) / 2.0
            approx.append("rolling_arrival_mean_7 approximated from available inputs")
        else:
            arr_mean7 = _validate_finite_number("rolling_arrival_mean_7", rolling_arrival_mean_7, non_negative=True)

        features = {
            "modal_price_rs_per_qtl": cur,
            "price_lag_1": l1,
            "price_lag_3": l3,
            "price_lag_7": l7,
            "rolling_mean_7": stats["rolling_mean_7"],
            "rolling_std_7": stats["rolling_std_7"],
            "rolling_min_7": stats["rolling_min_7"],
            "rolling_max_7": stats["rolling_max_7"],
            "price_momentum_7": float(compute_price_momentum_7_pct(cur, l7)),
            "daily_spread": spread,
            "arrival_tonnes": arr,
            "arrival_lag_1": arr_l1,
            "rolling_arrival_mean_7": arr_mean7,
            "month": int(month),
            "day_of_week": int(day_of_week),
            "is_monsoon_season": 1 if int(month) in (6, 7, 8, 9) else 0,
        }
        row = _validate_numeric_feature_dict(features, self.feature_names)
        return self._run_validated_pipeline(
            row,
            is_validated_forecast=False,
            inference_mode="Mode B: Experimental Manual Exploration",
            experimental_disclaimer=EXPERIMENTAL_MANUAL_DISCLAIMER,
            approximations_used=approx or ["manual user-supplied features"],
        )


class UnifiedCommodityPredictor:
    """
    Unified Multi-Commodity Forecaster with strict Mode A (Validated) vs Mode B (Experimental Manual)
    separation, exact canonical schema & momentum parity, and early-warning classification.
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
        self.feature_schema_version: str = str(
            self.metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        )

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
        self.expected_all_features: List[str] = self.metadata.get("all_features", UNIFIED_ALL_FEATURES)
        self.expected_numeric_features: List[str] = self.metadata.get(
            "numeric_features", UNIFIED_NUMERIC_FEATURES
        )

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
                f"Commodity '{commodity}' (canonical: '{canon}') is NOT in validated training coverage "
                f"{self.supported_commodities}. Independent Leave-One-Commodity-Out tests showed "
                f"unseen-crop OHE generalization does NOT reliably beat persistence baselines "
                f"({len(loco)} holdout crops evaluated). Pass a validated commodity or opt into Mode B "
                f"Experimental exploration (allow_experimental=True)."
            )
        warning = None
        if not is_supported:
            warning = (
                f"EXPERIMENTAL UNVALIDATED COMMODITY: '{canon}' was never seen during model training. "
                f"Metrics are unvalidated; do NOT rely on this forecast."
            )
        return canon, is_supported, warning

    def validate_pair(
        self, canonical_commodity: str, market: str, allow_experimental: bool = False
    ) -> Tuple[str, bool, Optional[str]]:
        canon_mkt = canonicalize_market_name(market)
        is_pair = (canonical_commodity, canon_mkt) in self.supported_pairs if self.supported_pairs else True
        if not is_pair and not allow_experimental:
            valid_mkts = self.get_supported_markets(canonical_commodity)
            raise UnsupportedMarketPairError(
                f"Market pair ({canonical_commodity} @ {canon_mkt}) was NOT trained or validated in "
                f"Mode A. Validated markets for {canonical_commodity}: {valid_mkts}. Pass a validated "
                f"pair or explicitly enable Mode B experimental exploration."
            )
        warning = None
        if not is_pair:
            valid_mkts = self.get_supported_markets(canonical_commodity)
            warning = (
                f"Experimental Market Pair: '{canonical_commodity}' was only validated in {valid_mkts}, "
                f"not '{canon_mkt}'."
            )
        return canon_mkt, is_pair, warning

    def _run_pipeline(
        self,
        features_dict: Dict[str, Any],
        is_validated_forecast: bool,
        inference_mode: str,
        allow_experimental: bool,
        experimental_disclaimer: Optional[str] = None,
        approximations_used: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raw_comm = str(features_dict.get("commodity_name") or features_dict.get("commodity") or "").strip()
        if not raw_comm:
            raise FeatureSchemaMismatchError("'commodity_name' is required.")

        canon_comm, is_validated_comm, comm_warning = self.validate_commodity(
            raw_comm, allow_experimental=allow_experimental
        )
        raw_mkt = str(features_dict.get("market_name") or features_dict.get("market") or "").strip()
        if not raw_mkt:
            raise FeatureSchemaMismatchError("'market_name' is required.")

        canon_mkt, is_validated_pair, pair_warning = self.validate_pair(
            canon_comm, raw_mkt, allow_experimental=allow_experimental
        )

        default_cat, default_perish = COMMODITY_CATEGORY_MAP.get(canon_comm, ("Vegetables", 1))
        category = str(features_dict.get("category") or default_cat)
        is_perishable = int(features_dict.get("is_perishable", default_perish))

        merged_for_validation = dict(features_dict)
        merged_for_validation["is_perishable"] = is_perishable

        numeric_values = _validate_numeric_feature_dict(
            merged_for_validation,
            self.expected_numeric_features,
            expected_schema_version=self.feature_schema_version if is_validated_forecast else None,
        )

        full_row: Dict[str, Any] = {
            "commodity_name": canon_comm,
            "category": category,
            "market_name": canon_mkt,
            **numeric_values,
        }

        X_df = pd.DataFrame([full_row])[self.expected_all_features]
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

        if is_validated_comm and is_validated_pair and is_validated_forecast and self.classifier.available:
            ew_res = self.classifier.predict_from_features(full_row)
        else:
            ew_res = {
                "available": False,
                "predicted_regime": "Unavailable",
                "class_probabilities": {},
                "any_spike_probability": None,
                "is_reliable_early_warning": False,
                "reliability_status": (
                    "Skipped for Mode B Experimental Exploration or unvalidated commodity-market pair."
                ),
            }

        combined_warnings = [w for w in (comm_warning, pair_warning, experimental_disclaimer) if w]
        observed_retail = get_observed_retail_price(canon_comm)

        return {
            "commodity": canon_comm,
            "category": category,
            "market": canon_mkt,
            "feature_schema_version": self.feature_schema_version,
            "inference_mode": inference_mode,
            "is_validated_forecast": is_validated_forecast,
            "experimental_disclaimer": experimental_disclaimer,
            "approximations_used": approximations_used or [],
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
            "model_used": f"Unified {self.metadata.get('champion_algorithm', 'Linear/Ridge Model')}",
            "dataset_version": self.metadata.get("dataset_version", "v3.1.0-parity-unified"),
            "feature_mode": inference_mode,
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

    def predict_validated(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Mode A: Strict Validated Forecasting over complete real series & validated (commodity, market)."""
        return self._run_pipeline(
            features_dict,
            is_validated_forecast=True,
            inference_mode="Mode A: Validated Forecasting",
            allow_experimental=False,
        )

    def predict_from_features(
        self,
        features_dict: Dict[str, Any],
        allow_experimental: bool = False,
        feature_mode: str = "Mode A: Validated Forecasting",
    ) -> Dict[str, Any]:
        """Compatible entrypoint: enforces strict Mode A unless experimental exploration is requested."""
        is_mode_a = not allow_experimental and ("experimental" not in str(feature_mode).lower())
        return self._run_pipeline(
            features_dict,
            is_validated_forecast=is_mode_a,
            inference_mode="Mode A: Validated Forecasting" if is_mode_a else "Mode B: Experimental Manual Exploration",
            allow_experimental=allow_experimental,
            experimental_disclaimer=None if is_mode_a else EXPERIMENTAL_MANUAL_DISCLAIMER,
        )

    def predict_experimental_manual(
        self,
        commodity: str,
        market: str,
        modal_price: float,
        price_lag_1: float,
        price_lag_3: float,
        price_lag_7: float,
        arrival_tonnes: float,
        arrival_lag_1: Optional[float] = None,
        rolling_arrival_mean_7: Optional[float] = None,
        prior_7_prices: Optional[Sequence[float]] = None,
        daily_spread: float = 0.0,
        month: int = 11,
        day_of_week: int = 2,
        category: Optional[str] = None,
        is_perishable: Optional[int] = None,
        allow_experimental_commodity: bool = False,
    ) -> Dict[str, Any]:
        """
        Mode B: Explicit Experimental Manual Exploration.
        Raises UnsupportedCommodityError unless `allow_experimental_commodity=True`, computes
        `price_momentum_7` via shared canonical %, excludes P(t) from rolling previous window,
        and discloses every subset approximation.
        """
        canon_comm, is_sup_comm, _ = self.validate_commodity(
            commodity, allow_experimental=allow_experimental_commodity
        )
        canon_mkt, is_sup_pair, _ = self.validate_pair(
            canon_comm, market, allow_experimental=(allow_experimental_commodity or is_sup_comm)
        )

        cur = _validate_finite_number("modal_price", modal_price, positive_only=True)
        l1 = _validate_finite_number("price_lag_1", price_lag_1, positive_only=True)
        l3 = _validate_finite_number("price_lag_3", price_lag_3, positive_only=True)
        l7 = _validate_finite_number("price_lag_7", price_lag_7, positive_only=True)
        arr = _validate_finite_number("arrival_tonnes", arrival_tonnes, non_negative=True)
        spread = _validate_finite_number("daily_spread", daily_spread, non_negative=True)

        approx: List[str] = []
        if prior_7_prices is not None and len(list(prior_7_prices)) == 7:
            stats = compute_previous_window_price_stats(prior_7_prices)
        else:
            stats = compute_previous_window_price_stats([l1, l3, l7])
            approx.append("rolling_*_7 computed from 3 selected lags [t-1, t-3, t-7] (approximate, excludes t)")

        if arrival_lag_1 is None:
            arr_l1 = arr
            approx.append("arrival_lag_1 defaulted from user current arrival input")
        else:
            arr_l1 = _validate_finite_number("arrival_lag_1", arrival_lag_1, non_negative=True)

        if rolling_arrival_mean_7 is None:
            arr_mean7 = (arr + arr_l1) / 2.0
            approx.append("rolling_arrival_mean_7 approximated from available arrival inputs")
        else:
            arr_mean7 = _validate_finite_number("rolling_arrival_mean_7", rolling_arrival_mean_7, non_negative=True)

        default_cat, default_perish = COMMODITY_CATEGORY_MAP.get(canon_comm, ("Vegetables", 1))
        features = {
            "commodity_name": canon_comm,
            "category": category or default_cat,
            "market_name": canon_mkt,
            "modal_price_rs_per_qtl": cur,
            "price_lag_1": l1,
            "price_lag_3": l3,
            "price_lag_7": l7,
            "rolling_mean_7": stats["rolling_mean_7"],
            "rolling_std_7": stats["rolling_std_7"],
            "rolling_min_7": stats["rolling_min_7"],
            "rolling_max_7": stats["rolling_max_7"],
            "price_momentum_7": float(compute_price_momentum_7_pct(cur, l7)),
            "daily_spread": spread,
            "arrival_tonnes": arr,
            "arrival_lag_1": arr_l1,
            "rolling_arrival_mean_7": arr_mean7,
            "month": int(month),
            "day_of_week": int(day_of_week),
            "is_monsoon_season": 1 if int(month) in (6, 7, 8, 9) else 0,
            "is_perishable": int(is_perishable) if is_perishable is not None else default_perish,
        }

        disclaimer = EXPERIMENTAL_MANUAL_DISCLAIMER
        if not is_sup_pair:
            disclaimer = f"{disclaimer} Unvalidated pair: ({canon_comm} @ {canon_mkt})."

        return self._run_pipeline(
            features,
            is_validated_forecast=False,
            inference_mode="Mode B: Experimental Manual Exploration",
            allow_experimental=True,
            experimental_disclaimer=disclaimer,
            approximations_used=approx or ["manual user-supplied features"],
        )

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
        rolling_std_7: Optional[float] = None,
        rolling_min_7: Optional[float] = None,
        rolling_max_7: Optional[float] = None,
        arrival_tonnes: Optional[float] = None,
        arrival_lag_1: Optional[float] = None,
        rolling_arrival_mean_7: Optional[float] = None,
        daily_spread: float = 0.0,
        month: int = 11,
        day_of_week: int = 2,
        is_perishable: Optional[int] = None,
        allow_experimental: bool = False,
        allow_manual_approx: bool = False,
    ) -> Dict[str, Any]:
        """Strict wrapper: requires full validated inputs unless manual Mode B approximation is requested."""
        self.validate_commodity(commodity, allow_experimental=allow_experimental)

        missing_strict = [
            name
            for name, v in [
                ("modal_price", modal_price),
                ("price_lag_1", price_lag_1),
                ("price_lag_3", price_lag_3),
                ("price_lag_7", price_lag_7),
                ("rolling_mean_7", rolling_mean_7),
                ("rolling_std_7", rolling_std_7),
                ("rolling_min_7", rolling_min_7),
                ("rolling_max_7", rolling_max_7),
                ("arrival_tonnes", arrival_tonnes),
                ("arrival_lag_1", arrival_lag_1),
                ("rolling_arrival_mean_7", rolling_arrival_mean_7),
            ]
            if v is None
        ]

        if missing_strict and not allow_manual_approx:
            raise ValueError(
                f"Refusing to manufacture missing lag/rolling/arrival features in Mode A: {missing_strict}. "
                f"Supply real series features via `predict_validated()` or explicitly invoke `predict_experimental_manual()`."
            )

        if not missing_strict:
            canon_comm = canonicalize_commodity_name(commodity)
            default_cat, default_perish = COMMODITY_CATEGORY_MAP.get(canon_comm, ("Vegetables", 1))
            features = {
                "commodity_name": canon_comm,
                "category": category or default_cat,
                "market_name": canonicalize_market_name(market),
                "modal_price_rs_per_qtl": modal_price,
                "price_lag_1": price_lag_1,
                "price_lag_3": price_lag_3,
                "price_lag_7": price_lag_7,
                "rolling_mean_7": rolling_mean_7,
                "rolling_std_7": rolling_std_7,
                "rolling_min_7": rolling_min_7,
                "rolling_max_7": rolling_max_7,
                "price_momentum_7": float(compute_price_momentum_7_pct(modal_price, price_lag_7)),
                "daily_spread": daily_spread,
                "arrival_tonnes": arrival_tonnes,
                "arrival_lag_1": arrival_lag_1,
                "rolling_arrival_mean_7": rolling_arrival_mean_7,
                "month": month,
                "day_of_week": day_of_week,
                "is_monsoon_season": 1 if int(month) in (6, 7, 8, 9) else 0,
                "is_perishable": int(is_perishable) if is_perishable is not None else default_perish,
            }
            return self.predict_from_features(features, allow_experimental=allow_experimental)

        return self.predict_experimental_manual(
            commodity=commodity,
            market=market,
            modal_price=modal_price if modal_price is not None else (price_lag_1 or 0.0),
            price_lag_1=price_lag_1 if price_lag_1 is not None else (modal_price or 0.0),
            price_lag_3=price_lag_3 if price_lag_3 is not None else (price_lag_1 or modal_price or 0.0),
            price_lag_7=price_lag_7 if price_lag_7 is not None else (price_lag_3 or price_lag_1 or modal_price or 0.0),
            arrival_tonnes=arrival_tonnes if arrival_tonnes is not None else 0.0,
            arrival_lag_1=arrival_lag_1,
            rolling_arrival_mean_7=rolling_arrival_mean_7,
            daily_spread=daily_spread,
            month=month,
            day_of_week=day_of_week,
            category=category,
            is_perishable=is_perishable,
            allow_experimental_commodity=allow_experimental,
        )
