# Canonical Feature Engineering & Serving Contract (`v3.1.0-parity`)

Shared verbatim across:
- **Offline Dataset Preparation:** [src/features.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/features.py), [scripts/build_features.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/build_features.py)
- **Live & Historical Feature Construction:** [src/live_data.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/live_data.py)
- **Production Serving & Schema Validation:** [src/inference.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/inference.py), [src/schemas.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/schemas.py)

---

## 1. Numeric Input Feature Schema (`SPECIALIST_NUMERIC_FEATURES` [16] + `is_perishable` [17th])

All time-series windows are indexed over observed active APMC trading sessions ordered ascending by `observation_date` within a single exact `(commodity_name, market_name)` series.

| # | Feature Name | Mathematical Definition | Session Window | Unit | Missingness / Guard |
| :-: | :--- | :--- | :--- | :---: | :--- |
| 1 | `modal_price_rs_per_qtl` | Current session wholesale auction modal price $P_t$ (median across intraday varieties within exact market & date) | $t$ (current) | `Rs/Qtl` | Strictly finite $> 0$ |
| 2 | `price_lag_1` | Modal price 1 observed trading session prior: $P_{t-1}$ | $t - 1$ | `Rs/Qtl` | Strictly finite $> 0$ |
| 3 | `price_lag_3` | Modal price 3 observed trading sessions prior: $P_{t-3}$ | $t - 3$ | `Rs/Qtl` | Strictly finite $> 0$ |
| 4 | `price_lag_7` | Modal price 7 observed trading sessions prior: $P_{t-7}$ | $t - 7$ | `Rs/Qtl` | Strictly finite $> 0$ |
| 5 | `rolling_mean_7` | $\frac{1}{n}\sum_{k=1}^{7} P_{t-k}$ (`shift(1).rolling(7, min_periods=4).mean()`) — excludes current session $P_t$ | $[t-7 \dots t-1]$ | `Rs/Qtl` | Strictly finite $> 0$ |
| 6 | `rolling_std_7` | Sample standard deviation ($ddof=1$) of shifted window `[t-7 .. t-1]` — excludes current session $P_t$ | $[t-7 \dots t-1]$ | `Rs/Qtl` | Finite $\ge 0$ |
| 7 | `rolling_min_7` | Minimum modal price across `[t-7 .. t-1]` — excludes current session $P_t$ | $[t-7 \dots t-1]$ | `Rs/Qtl` | Strictly finite $> 0$ |
| 8 | `rolling_max_7` | Maximum modal price across `[t-7 .. t-1]` — excludes current session $P_t$ | $[t-7 \dots t-1]$ | `Rs/Qtl` | Strictly finite $> 0$ |
| 9 | `price_momentum_7` | `((P_t - P_{t-7}) / P_{t-7}) * 100.0` (canonicalized across training & inference via `compute_price_momentum_7()`) | $t$ vs $t-7$ | `%` | Finite (`P_{t-7} > 0`) |
| 10 | `daily_spread` | Current session intraday spread: $P^{\max}_t - P^{\min}_t$ (`max_price - min_price` across exact-market intraday bids) | $t$ (current) | `Rs/Qtl` | Finite $\ge 0$ |
| 11 | `arrival_tonnes` | Current session physical arrival volume summed within exact `(commodity, market, date)` | $t$ (current) | `Tonnes` | Genuine value $\ge 0$; never silently fill missing API values as `0` |
| 12 | `arrival_lag_1` | Physical arrival volume 1 trading session prior ($A_{t-1}$) | $t - 1$ | `Tonnes` | Genuine value $\ge 0$ (`NaN` drops row/withholds inference) |
| 13 | `rolling_arrival_mean_7` | Mean arrival volume across previous 7 sessions (`shift(1).rolling(7, min_periods=4).mean()`) — excludes $A_t$ | $[t-7 \dots t-1]$ | `Tonnes` | Genuine value $\ge 0$ |
| 14 | `month` | Calendar month integer of `observation_date` (`1..12`) | $t$ | `Integer (1-12)` | `[1, 12]` |
| 15 | `day_of_week` | Weekday integer of `observation_date` (`0=Monday .. 6=Sunday`) | $t$ | `Integer (0-6)` | `[0, 6]` |
| 16 | `is_monsoon_season` | Indicator: `1` if `month in {6, 7, 8, 9}`, else `0` | $t$ | `Binary {0,1}` | `{0, 1}` |
| 17 | `is_perishable` *(Unified)* | Commodity perishability indicator (`1` = Tomato, Onion; `0` = Moong Dal, Sugar) | Static | `Binary {0,1}` | `{0, 1}` |

## 2. Categorical Input Schema (`UNIFIED_CATEGORICAL_FEATURES`)

| Feature Name | Allowed Validated Values | Unit / Type | Mapping Note |
| :--- | :--- | :---: | :--- |
| `commodity_name` | `Moong Dal`, `Onion`, `Sugar`, `Tomato` | `Categorical String` | `Green Gram Dal (Moong Dal)` canonicalized to `Moong Dal` via `canonicalize_commodity_name()`. |
| `category` | `Vegetables`, `Pulses`, `Sugars & Sweets` | `Categorical String` | Derived deterministically via `COMMODITY_CATEGORY_MAP`. |
| `market_name` | `Lasalgaon`, `Mumbai`, `Pune` | `Categorical String` | Exact canonical single market via `canonicalize_market_name()`. Never aggregates submarkets. |

## 3. Target Variables & Horizon Definition (Never Features)

- **`target_price_t_plus_1`** (`Rs/Qtl`): Observed `modal_price_rs_per_qtl` on the next observed active APMC trading session (`t+1` ordered by active date, not guaranteed next calendar day).
- **`target_date_t_plus_1`**: Date of that next observed active trading session (used explicitly by `DatePurgedWalkForwardCV` and outer splitting to purge boundary crossings).
- **`spike_direction`**: 3-class shock target from `price_change_pct = ((P_{t+1} - P_t) / P_t) * 100`: `Surge` (> `+10%`), `Crash` (< `-10%`), `Normal` (`[-10%, +10%]`).
- **Wholesale-to-Kg Conversion Rule:** Dividing predicted `Rs/Qtl` by `100.0` produces `wholesale_equiv_rs_per_kg` — strictly a **unit conversion**, NOT a retail consumer price forecast (`retail_estimation_supported = False`).
