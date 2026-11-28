# Phase 2 Deep Engineering Audit: Training–Inference Parity & Methodology Defects

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Repository:** `https://github.com/tusharkkp/ML_Project` (`branch: main`)  
**Audited Commit:** `cf1df493911f3e09fd55d6a997a0bad123b067d5` (`v3.0.0-leakfree`, `0` ahead / `0` behind `origin/main`, clean working tree)  
**Audit Date:** 2026-11-28

---

## Executive Summary

Although `v3.0.0-leakfree` resolved initial test-set selection leakage, cross-commodity lag bleed, and retail ML infeasibility, a deep line-by-line audit of code, API records, trained artifacts, and UI workflows uncovered **10 critical/high/medium training–inference parity and methodological defects**. Left unfixed, these defects cause training–serving skew, dishonest manual/retail fallbacks, potential API submarket contamination, fold-level leakage in CV/LOCO/unsupervised analysis, and inconsistent selection metrics.

---

## Defect Inventory & Exact Locations

### Defect 1 (CRITICAL): `price_momentum_7` Definition & Unit Mismatch Between Training and Inference
- **Locations:**
  - Training (`src/features.py:147`): `out["price_momentum_7"] = out[price_col] - out[price_col].shift(7)` (**absolute price difference in `Rs/Quintal`**).
  - Inference (`src/inference.py:345, 495, 669` and `src/schemas.py:24`): `((cur - lag7) / lag7) * 100.0` (**percentage change `%`**).
  - Documentation (`src/schemas.py:20`): Inaccurately described `rolling_mean_7` as *"including current session"*, whereas `src/features.py:140-141` uses `out[price_col].shift(1).rolling(7)` (**previous 7 sessions `[t-7..t-1]`, excluding `t`**).
- **Impact:** Manual and programmatic `.predict()` calls inject `%` values (e.g., `+15.0`) into models trained on `Rs/Quintal` differences (`±2000`), silently corrupting standardized feature magnitudes.
- **Remediation (Phase 1):** Define one shared helper `compute_price_momentum_7(current_price, price_lag_7)` in `src/features.py` (using percentage change `((P_t - P_{t-7}) / P_{t-7}) * 100.0` in **`%`** for scale invariance across ₹700 Vegetables and ₹7,000 Pulses), import it directly in `src/inference.py`, update `src/schemas.py` & `reports/feature_contract.md`, regenerate all model-ready CSVs, and retrain every model.

### Defect 2 (CRITICAL): Dishonest Manual Exploratory Rolling Windows & Arrival Copying
- **Locations:** `app.py:216-225`, `gradio_app.py:72-80`, `deployment/huggingface/app.py:76-84`, `src/inference.py:324-333, 646-655`.
- **Finding:** UI manual modes compute `rolling_mean_7` from only **4 selected values including `current_price`** (`(P_t + lag1 + lag3 + lag7) / 4.0`) and silently copy `arrival_tonnes` into `arrival_lag_1` and `rolling_arrival_mean_7`, then emit forecasts labeled `manual_user_entered` without explicitly framing them as **Mode B: Experimental Manual Exploration**.
- **Impact:** Violates `[t-7..t-1]` rolling semantics (leaks `P_t` into previous-window stats), misrepresents 4-point heuristics as genuine 7-session rolling features, and masks reliability differences versus validated archive/complete time-series inference.
- **Remediation (Phase 2):** Explicitly separate **Mode A: Validated Forecasting** (genuine 7+ session series built solely via `src/features.py` + validated commodity-market pair) from **Mode B: Experimental Manual Exploration** (`is_validated_forecast=False`, excludes `P_t` from historical summary window `[lag1..lag7]`, never copies arrivals silently without explicit disclosure, renders prominent experimental banners across Streamlit/Gradio/HF).

### Defect 3 (CRITICAL): Live API Substring Market Matching Permits Unintended Submarkets
- **Location:** `src/live_data.py:252`: `if r_mkt in mkt_aliases or market.lower() in r_mkt:`
- **Verified Evidence (`data/raw/wholesale/data_gov_mandi_maharashtra_20261117_192806.json`):** Real `data.gov.in` Maharashtra market values include `'APMC Pune '`, `'Pune(Pimpri) '`, `'Pune(Hadapsar)'`, `'Pune(Khadiki) '`, `'Pune(Moshi) '`, `'APMC Gondpimpri '`, `'Lasalgaon(Niphad) '`, `'Lasalgaon(Vinchur) '`, `'APMC Mumbai '`, `'Mumbai- Fruit Market '`, `'Mumbai-Onion & Potato Market '`.
- **Impact:** The substring rule `"pune" in r_mkt` silently merges submarkets (`Pimpri`, `Khadiki`, `Moshi`) into exact `Pune`, and `"mumbai" in r_mkt` merges specialty subyards into `Mumbai`, violating single-market training semantics.
- **Remediation (Phase 3):** Implement deterministic token normalization (lowercase, strip whitespace, collapse extra spaces, normalize standard `"APMC <X>"` / `"<X> APMC"` phrasing) matched strictly by **exact normalized equality** against curated per-market alias tables, preserve raw & canonical market names, detect/reject ambiguous multi-yard payloads, log original records, and aggregate varieties only within the single exact market and observation date.

### Defect 4 (CRITICAL): Live API Silently Imputes Unavailable Arrivals as `0.0` Tonnes
- **Locations:** `src/live_data.py:130, 378`, `src/features.py:152`, `scripts/build_features.py:163`.
- **Verified Schema Evidence:** Inspected `data/raw/wholesale/data_gov_mandi_maharashtra_20261117_192806.json` (`resource 9ef84268-d588-465a-a308-a864a43d0070`). Fields are strictly `{'state', 'district', 'market', 'commodity', 'variety', 'grade', 'arrival_date', 'min_price', 'max_price', 'modal_price'}` — **`arrival_quantity_tonnes` is NOT provided by the API**.
- **Impact:** Setting `arrival_quantity_tonnes=0.0` conflates *missing telemetry* with *zero market supply* (whereas historical means range from `123.6t` Moong Dal to `1,434.0t` Lasalgaon Onion).
- **Remediation (Phase 4):** Preserve missing arrivals as `NaN`/`None`, expose explicit `arrivals_available` and `arrival_observation_date` metadata, check known units only when an explicit arrival field is supplied, and withhold automated live inference whenever continuous 7-day prices OR genuine arrival inputs are missing (`lag_features_ready=False`). Also verify in `scripts/build_features.py` & `src/features.py` that historical `NaN` arrivals remain `NaN` (not filled to `0.0`), noting empirically that all 5 2016–2018 historical series already have 100% matched arrival dates.

### Defect 5 (HIGH): Hardcoded `₹45.0/Kg` Retail Fallback Defaults
- **Locations:** `app.py:429`, `gradio_app.py:113`, `deployment/huggingface/app.py:113`.
- **Impact:** When DCA retail lookup returns `None`, all three UIs silently fall back to `45.0` Rs/Kg, risking fabricated inputs entering affordability calculations under `"DCA Obs"` labels.
- **Remediation (Phase 5):** Remove all hardcoded `45.0` defaults cleanly. Display `"Unavailable"` when `retail_price_rs_per_kg is None`; display exact centre, observation date, and provenance when present (explicitly highlighting archived historical dates); require explicit manual retail input tagged `"User Supplied"`.

### Defect 6 (HIGH): Gradio Fake Complete Future Basket (`predicted = current`) & UI Desync
- **Locations:** `gradio_app.py:118`, `deployment/huggingface/app.py:118`, `app.py:438`.
- **Finding:** `gradio_app.py` sets `"predicted_retail_price_per_kg": price` (copying current price as future predicted retail price), causing `calculate_household_affordability` to falsely report `has_complete_predictions = True` (`prediction_status = "complete"`) even when wholesale-to-retail ML estimation is `unavailable`!
- **Remediation (Phase 6):** Stop copying current prices as predictions in Gradio & HF Spaces (`predicted_retail_price_per_kg = None` unless user runs explicit hypothetical sensitivity stress scenario); distinguish `Actual Observed`, `User Supplied`, `ML Predicted (Unavailable)`, and `Hypothetical Sensitivity Scenario` uniformly across Streamlit, Gradio, and HF deployment.

### Defect 7 (HIGH): Row-Wise Internal `TimeSeriesSplit` in Multi-Series & Target-Boundary CV
- **Locations:** `scripts/train_unified_model.py:253`, `scripts/train_all_models.py:160, 259, 607`, `scripts/train_and_evaluate.py:57`.
- **Finding:** `GridSearchCV(..., cv=TimeSeriesSplit(n_splits=5))` splits `X_train` by row index rather than unique `observation_date`. In unified multi-commodity data (5 rows per date), observations on the same date can straddle CV train and validation folds, and train targets `target_date_t_plus_1` crossing into the CV validation window are not purged.
- **Remediation (Phase 7):** Build `DatePurgedWalkForwardCV` in `src/features.py` yielding `(train_idx, val_idx)` grouped strictly by unique sorted `observation_date` (`max(train_dates) < min(val_dates)`) and purging any train sample where `target_date >= min(val_dates)`. Integrate into `GridSearchCV` across unified and specialist training and `ChronologicalStackingRegressor`.

### Defect 8 (HIGH): Information Leakage in Leave-One-Commodity-Out (LOCO) Model Selection
- **Location:** `scripts/train_unified_model.py:111-155, 449`.
- **Finding:** `run_leave_one_commodity_out()` receives `fitted_val_estimators[champion_name]` (`Ridge(alpha=1.0)`), whose algorithm and hyperparameter $\alpha$ were selected using Validation data containing the held-out commodity.
- **Remediation (Phase 8):** For each held-out commodity `c`, run the full candidate search (`DatePurgedWalkForwardCV` hyperparameter tuning inside `Train_{!=c}` -> selection strictly on `Val_{!=c}` -> refit on `Train_{!=c} + Val_{!=c}` -> evaluate on `Test_{c}` vs persistence baseline) completely excluding `c` from selection.

### Defect 9 (MEDIUM): Unsupervised Clustering & PCA Uses Test Set for Scaling, `k` Selection & Fitting
- **Location:** `scripts/train_all_models.py:493-560`.
- **Finding:** `run_clustering_pca()` calls `scaler.fit(data["X_train_val"])`, combines `X_train_val` and `X_test` into `X_all`, selects `best_k` via Silhouette across `X_all`, and fits `KMeans`, `AgglomerativeClustering`, `DBSCAN`, and `PCA` on `X_all` (leaking Test into `k` choice and fit). Also reports Silhouette/DB without behavioral cluster profiles.
- **Remediation (Phase 9):** Fit scaling, select `best_k`, fit `KMeans`, `AgglomerativeClustering`, `DBSCAN`, and `PCA` strictly on **`Train` only** (`X_train`); evaluate out-of-sample cluster/PCA projection ONLY for models supporting `.predict()`/`.transform()` (`KMeans`, `PCA`) on `Validation` (keeping `Test` untouched); clearly document Hierarchical/DBSCAN out-of-sample limitation; output descriptive cluster profiles (mean price, std dev, CV%, arrivals, spike rate %) and caveat Silhouette limits.

### Defect 10 (HIGH): Pooled vs. Macro Selection Metric Discrepancy & Candidate Snooping on Test
- **Locations:** `scripts/train_unified_model.py:311` vs docs; `scripts/train_all_models.py:303-320, 449-462`.
- **Findings:**
  1. `train_unified_model.py:311` sorts `val_df_board` by pooled `Val_RMSE` instead of `Val_Macro_RMSE` even though pooled RMSE is dominated by Moong Dal (~₹7,000) & Sugar (~₹3,200) scales when claiming cross-commodity balance.
  2. `train_all_models.py` runs `final_p.predict(X_te)` across ALL 14 regression candidates and ALL 10 classifiers on `X_test`, conflicting with "Untouched Test evaluated once for the Validation Champion".
- **Remediation (Phase 10):** Explicitly standardize unified selection on `Val_Macro_RMSE` (recording pooled `Val_RMSE` as secondary), store exact `selection_metric` in metadata, evaluate **ONLY the Validation-selected Champion + Persistence/Majority Baseline** on the untouched Test partition (keeping all multi-candidate comparisons strictly in `validation_leaderboard_*.csv`), and reconcile all CSVs/JSON/docs cleanly.
