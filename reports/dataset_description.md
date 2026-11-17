# Dataset Description & Problem Formulation

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Course Component:** Practical Examination Component A (Data Collection & Dataset Description — 2 Marks)  
**Date:** 2026-11-17  
**Author:** Senior Machine Learning Engineer & Academic Project Mentor  

---

## 1. Dataset Origin, Context & Provenance

The primary modeling dataset is drawn from the **Agricultural Produce Market Committee (APMC) Mandi Trading Records** for **Pune, Maharashtra**, published through **AGMARKNET** (Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, Government of India) and the **data.gov.in** Open Government Data Platform (Resource ID: `9ef84268-d588-465a-a308-a864a43d0070`).

### Domain Context: How APMC Mandi Prices are Formed
In India, wholesale agricultural trade occurs primarily within regulated APMC market yards. 
1. **Physical Arrivals:** Farmers from surrounding agricultural talukas (e.g. Junnar, Khed, Ambegaon, Shirur for Pune) transport freshly harvested produce to the mandi early each morning.
2. **Open Auction:** Commission agents and wholesale traders bid on produce lots.
3. **Price Determination:** The auction establishes three daily price quotes:
   - **Minimum Price ($P_{min}$):** Lowest transacted price (often lower grade or end-of-day clearance).
   - **Maximum Price ($P_{max}$):** Highest transacted price (prime export/supermarket grade).
   - **Modal Price ($P_{modal}$):** The most frequently transacted price at which the largest volume of produce changed hands.
4. **Economic Significance:** The **Modal Price** is the standard benchmark used by economists, the Reserve Bank of India, and agricultural planners. It represents the actual price signal driving food inflation.

---

## 2. Selected Target Commodity: Tomato (Pune APMC)

### Why Tomato?
Tomato is a quintessential essential food commodity in Indian households and one of the "TOP" trinity of perishables (Tomato, Onion, Potato). Unlike grains, tomatoes cannot be stored for long periods without cold storage infrastructure. Consequently:
- Prices are highly sensitive to daily arrival volumes ($Q_t$).
- Sudden monsoonal rains, pest outbreaks, or transport strikes cause violent supply shocks (price surges of 300%–500% within weeks).
- Subsequent harvesting gluts cause price crashes.
This high variance and pronounced seasonality make Tomato the ultimate benchmark for evaluating whether machine learning algorithms (linear, polynomial, decision trees, random forests) can capture real non-linear market dynamics.

---

## 3. Dataset Attributes & Schema Definition

The modeling corpus pairs daily wholesale price quotes with physical mandi arrivals:

| Attribute Name | Data Type | Physical Unit | Description & Economic Meaning |
|:---|:---:|:---:|:---|
| `observation_date` | `DATE` | `YYYY-MM-DD` | Calendar date of the APMC auction session. |
| `commodity_name` | `VARCHAR(50)` | Text | Standardized canonical commodity name (`Tomato`). |
| `market_name` | `VARCHAR(100)` | Text | Regulated market yard (`Pune APMC`). |
| `district` | `VARCHAR(50)` | Text | Administrative district (`Pune`). |
| `state` | `VARCHAR(50)` | Text | State jurisdiction (`Maharashtra`). |
| `min_price_rs_per_qtl` | `FLOAT` | `Rs/Quintal` | Minimum transacted price during trading hours ($1\text{ Quintal} = 100\text{ Kg}$). |
| `max_price_rs_per_qtl` | `FLOAT` | `Rs/Quintal` | Maximum transacted price during trading hours. |
| `modal_price_rs_per_qtl` | `FLOAT` | `Rs/Quintal` | **Primary Dependent Variable ($y_t$):** Modal transacted wholesale price. |
| `arrival_quantity_tonnes`| `FLOAT` | `Tonnes` | Total physical volume unloaded at the mandi on date $t$ ($1\text{ Tonne} = 1,000\text{ Kg}$). |
| `price_unit` | `VARCHAR(20)` | Unit label | Standardized measurement unit: `Rs/Quintal`. |
| `price_type` | `VARCHAR(20)` | Constant | Categorical isolation flag: `Wholesale`. |
| `is_anomalous_flag` | `BOOLEAN` | True/False | Quality check: flags rare clerical transposition errors ($P_{min} > P_{max}$). |

---

## 4. Volume, Temporal Scope & Granularity

- **Observation Count:** **880 complete daily trading observations**.
- **Temporal Span:** **2016-01-01 to 2018-12-31** (3 full calendar years).
- **Trading Frequency:** 6 days a week (Monday through Saturday). Sundays and major religious/national holidays are official market closures.
- **Completeness:** 0 missing values in `observation_date`, `modal_price_rs_per_qtl`, and `arrival_quantity_tonnes`.
- **Integrity:** 0 duplicate dates; chronological sequence strictly preserved.

---

## 5. Machine Learning Problem Formulation

The problem is framed as a supervised multi-feature time-series regression task:

$$\hat{y}_{t+1} = f(\mathbf{x}_t)$$

Where:
- $\hat{y}_{t+1}$ is the predicted modal price of Tomato at Pune Mandi for the next active trading session ($t+1$).
- $\mathbf{x}_t$ is an information vector containing:
  - Historical price lags ($y_t, y_{t-3}, y_{t-7}$)
  - Rolling window price statistics (7-day mean, 7-day standard deviation, 7-day min/max spread)
  - Supply-side physical arrival volume ($q_t$)
  - Calendar/seasonal indicators (month of year, day of week, seasonal monsoon dummy)
- **Constraint:** All features in $\mathbf{x}_t$ are strictly observable at trading close on day $t$. No future data leakage is permitted.
