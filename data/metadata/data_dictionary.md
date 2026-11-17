# Master Data Dictionary

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Version:** 1.0.0  
**Updated:** 2026-11-17  
**Standard:** ISO 8601 Date (`YYYY-MM-DD`), Metric Units, Strict Price Type Isolation  

---

## Overview

This repository enforces strict schema consistency and physical separation between:
1. **Retail Prices** (`retail_prices.csv`): Consumer-facing end-market prices paid by households.
2. **Wholesale Mandi Prices** (`wholesale_prices.csv`): Primary and secondary APMC market transactions.
3. **Market Arrivals** (`market_arrivals.csv`): Agricultural physical supply volumes entering regulated mandis.
4. **Consumer Price Index** (`cpi_data.csv`): Official macroeconomic price indices and headline/food inflation rates.
5. **Household Expenditure** (`household_expenditure.csv`): NSSO Monthly Per Capita Consumption Expenditure (MPCE) benchmarks and food budget fractile distributions.

---

## 1. Retail Prices (`data/processed/retail_prices.csv`)

### Description
Contains consumer-level daily food prices across reporting centres in Maharashtra (Mumbai, Pune, Nashik, Nagpur, etc.) and All-India representative averages. Derived from the Department of Consumer Affairs (DCA) Price Monitoring Division.

| Column Name | Data Type | Physical Unit | Nullable | Allowed Values / Constraints | Business & Econometric Definition |
|:---|:---:|:---:|:---:|:---|:---|
| `observation_date` | `DATE` | `YYYY-MM-DD` | No | Valid historical calendar date (2014-01-01 to present) | The specific calendar day the retail price was monitored and recorded. |
| `commodity_name` | `VARCHAR(50)` | None | No | Canonical names: `Rice`, `Wheat`, `Wheat (Atta)`, `Gram Dal`, `Tur Dal`, `Urad Dal`, `Moong Dal`, `Masoor Dal`, `Groundnut Oil`, `Mustard Oil`, `Vanaspati`, `Soya Oil`, `Sunflower Oil`, `Palm Oil`, `Potato`, `Onion`, `Tomato`, `Sugar`, `Gur`, `Tea`, `Milk`, `Salt` | Standardized canonical commodity name used across the entire platform. |
| `raw_commodity_name` | `VARCHAR(100)` | None | No | Text as reported by upstream agency | Exact commodity name from the primary source, preserved for auditability. |
| `state` | `VARCHAR(50)` | None | No | e.g. `Maharashtra`, `Delhi`, `All India` | State administrative jurisdiction where the reporting centre is situated. |
| `district` | `VARCHAR(50)` | None | Yes | e.g. `Pune`, `Mumbai`, `Nashik`, `Nagpur`, `All India` | District administrative division. |
| `reporting_centre` | `VARCHAR(100)` | None | No | e.g. `Pune`, `Mumbai`, `Nashik`, `Nagpur`, `All-India Average` | Official consumer price monitoring centre or market hub where retail survey occurs. |
| `market_type` | `VARCHAR(50)` | None | No | `Reporting Centre Retail`, `Retail Average` | Clarifies whether the observation represents an urban centre retail quote or a pooled multi-centre average. |
| `retail_price` | `DECIMAL(10,2)` | Rs. per unit (`price_unit`) | No | Strictly positive float `> 0.0` | Nominal retail price per standard physical unit. |
| `price_unit` | `VARCHAR(20)` | Unit label | No | `1 Kg`, `1 Litre` | Physical measurement unit corresponding to `retail_price`. |
| `price_type` | `VARCHAR(20)` | Constant | No | `Retail` | Categorical tag ensuring retail prices are never mixed with wholesale quotes. |
| `benchmark_tag` | `VARCHAR(50)` | Category | Yes | `Historical Daily`, `Current Daily Average`, `1 Week Back`, `1 Month Back`, `6 Months Back`, `1 Year Back`, `2 Years Back` | Temporal relationship tag indicating whether the observation is daily point-in-time or an official agency benchmark. |
| `data_source` | `VARCHAR(100)` | URL/Agency | No | `DCA Retail Monitoring Historical Archive`, `DCA Price Monitoring Bulletin (fcainfoweb.nic.in)` | Provenance tag identifying upstream publishing portal. |

---

## 2. Wholesale Prices (`data/processed/wholesale_prices.csv`)

### Description
Contains transaction-level wholesale mandi market prices from Agricultural Produce Market Committees (APMCs) across Maharashtra districts (Pune, Nashik, Lasalgaon, Pimpalgaon, Nagpur, Ahmednagar, etc.). Sourced from AGMARKNET / Ministry of Agriculture & Farmers Welfare.

| Column Name | Data Type | Physical Unit | Nullable | Allowed Values / Constraints | Business & Econometric Definition |
|:---|:---:|:---:|:---:|:---|:---|
| `observation_date` | `DATE` | `YYYY-MM-DD` | No | Valid historical calendar date (2016-01-01 to present) | The specific trading day at the APMC mandi. |
| `commodity_name` | `VARCHAR(50)` | None | No | Canonical names: `Tomato`, `Onion`, `Potato`, `Tur Dal`, `Moong Dal`, `Gram Dal`, `Sugar`, etc. | Standardized canonical agricultural commodity name. |
| `raw_commodity_name` | `VARCHAR(100)` | None | No | Text as reported | Original commodity label in the APMC trading ledger. |
| `variety` | `VARCHAR(50)` | None | Yes | e.g. `Local`, `Hybrid`, `Red`, `White`, `Nashik`, `Other` | Commercial botanical variety or market trade grade. |
| `state` | `VARCHAR(50)` | None | No | `Maharashtra` (prioritized), other states | State jurisdiction where the APMC mandi operates. |
| `district` | `VARCHAR(50)` | None | No | `Pune`, `Nashik`, `Nagpur`, `Ahmednagar`, `Solapur`, etc. | District in which the regulated market yard is located. |
| `market_name` | `VARCHAR(100)` | None | No | e.g. `Lasalgaon`, `Pimpalgaon`, `Pune`, `Khadki`, `Nagpur` | Specific regulated APMC mandi yard where auction transactions occurred. |
| `min_price_rs_per_qtl` | `DECIMAL(10,2)` | Rs. per Quintal | Yes | Positive float `> 0.0` | Lowest transacted wholesale price during trading hours (1 Quintal = 100 Kg). |
| `max_price_rs_per_qtl` | `DECIMAL(10,2)` | Rs. per Quintal | Yes | Positive float `> 0.0` | Highest transacted wholesale price during trading hours. |
| `modal_price_rs_per_qtl` | `DECIMAL(10,2)` | Rs. per Quintal | No | Positive float `> 0.0` | Modal (most frequently occurring) transaction price. Primary metric for wholesale time-series modeling. |
| `price_unit` | `VARCHAR(20)` | Unit label | No | `Rs/Quintal` | Physical unit of quotation (100 Kg). |
| `price_type` | `VARCHAR(20)` | Constant | No | `Wholesale` | Categorical tag preventing conflation with consumer retail prices. |
| `is_anomalous_flag` | `BOOLEAN` | `True`/`False` | No | `True` if `min > max` or `modal < min` or `modal > max` | Data quality flag indicating upstream reporting irregularity; flagged rather than dropped for anomaly detection modeling. |
| `data_source` | `VARCHAR(100)` | Agency | No | `AGMARKNET Historical Mandi Archive`, `data.gov.in API` | Primary source origin. |

---

## 3. Market Arrivals (`data/processed/market_arrivals.csv`)

### Description
Contains agricultural physical inflow volumes arriving at regulated APMC mandis. Acts as a key leading indicator and supply-side feature for price forecasting and elasticity modeling.

| Column Name | Data Type | Physical Unit | Nullable | Allowed Values / Constraints | Business & Econometric Definition |
|:---|:---:|:---:|:---:|:---|:---|
| `observation_date` | `DATE` | `YYYY-MM-DD` | No | Valid calendar date | Date physical agricultural commodity lots entered the mandi yard. |
| `commodity_name` | `VARCHAR(50)` | None | No | Canonical names | Standardized agricultural commodity name. |
| `raw_commodity_name` | `VARCHAR(100)` | None | No | Text | Original commodity name from arrivals register. |
| `variety` | `VARCHAR(50)` | None | Yes | Text | Grade or botanical variety of arriving lots. |
| `state` | `VARCHAR(50)` | None | No | `Maharashtra` | State location of the arrival mandi. |
| `district` | `VARCHAR(50)` | None | No | District name | Administrative district. |
| `market_name` | `VARCHAR(100)` | None | No | Mandi yard name | Specific APMC yard receiving arrivals. |
| `arrival_quantity_tonnes` | `DECIMAL(10,2)` | Metric Tonnes (MT) | No | Non-negative float `>= 0.0` | Total physical volume unloaded at the mandi on the given date (1 Tonne = 1,000 Kg). |
| `unit` | `VARCHAR(20)` | Unit label | No | `Tonnes` | Unit of physical quantity. |
| `data_source` | `VARCHAR(100)` | Agency | No | `AGMARKNET Mandi Arrivals` | Upstream data provider. |

---

## 4. Consumer Price Index (`data/processed/cpi_data.csv`)

### Description
Monthly Consumer Price Index (Base 2012=100) and Year-on-Year (YoY) inflation rates published by the National Statistical Office (NSO), MoSPI. Covers Group 1 (Food and Beverages), commodity subgroups (Cereals, Pulses, Vegetables, Oils & Fats, Milk, Sugar), and item-level price series for All India and Maharashtra across Rural, Urban, and Combined sectors.

| Column Name | Data Type | Physical Unit | Nullable | Allowed Values / Constraints | Business & Econometric Definition |
|:---|:---:|:---:|:---:|:---|:---|
| `observation_date` | `DATE` | `YYYY-MM-DD` | No | Monthly sequence (`YYYY-MM-01`) | First calendar day of the index reference month (2014 to 2026). |
| `year` | `INTEGER` | Calendar Year | No | `2014` to `2026` | Calendar year of index observation. |
| `month` | `VARCHAR(20)` | Month Name | No | `January` through `December` | Calendar month of index observation. |
| `geography` | `VARCHAR(50)` | Location | No | `All India`, `Maharashtra`, etc. | Geographic domain of price monitoring. |
| `sector` | `VARCHAR(20)` | Domain | No | `Rural`, `Urban`, `Combined` | Demographic sector coverage. |
| `category_type` | `VARCHAR(30)` | Classification | No | `Group / Subgroup`, `Item` | Hierarchy level within the CPI consumption basket. |
| `category_name` | `VARCHAR(100)` | None | No | Canonical group or item name | Standardized commodity or aggregate group identifier. |
| `raw_category_name` | `VARCHAR(150)` | None | No | Raw MoSPI classification name | Original nomenclature from MoSPI index classification. |
| `subgroup` | `VARCHAR(100)` | None | Yes | e.g. `Cereals and products`, `Vegetables`, `Pulses and products` | Parent subgroup for item-level series. |
| `cpi_index` | `DECIMAL(8,2)` | Index points | Yes | Positive float `> 0.0` (Base 2012=100) | Consumer Price Index value. |
| `inflation_rate_yoy` | `DECIMAL(6,2)` | Percentage (`%`) | Yes | Float (`-100.0` to `+100.0`) | Annual percentage rate of change over the corresponding month of the previous year. |
| `base_year` | `VARCHAR(10)` | Reference base | No | `2012` | Reference benchmark year (2012 = 100). |
| `data_source` | `VARCHAR(100)` | Agency | No | `MoSPI Consumer Price Index API (api.mospi.gov.in)` | Provenance endpoint. |

---

## 5. Household Expenditure (`data/processed/household_expenditure.csv`)

### Description
Contains Household Consumption Expenditure Survey (HCES 2022-23) benchmarks published by NSSO, MoSPI. Provides Monthly Per Capita Consumption Expenditure (MPCE) across 12 fractile classes, rural/urban food budget shares, item group budget shares, and state-level averages for Maharashtra and All-India.

| Column Name | Data Type | Physical Unit | Nullable | Allowed Values / Constraints | Business & Econometric Definition |
|:---|:---:|:---:|:---:|:---|:---|
| `survey_year` | `VARCHAR(10)` | Survey period | No | `2022-23` (NSS 79th Round) | National survey reference cycle. |
| `geography` | `VARCHAR(50)` | Location | No | `All India`, `Maharashtra`, other states | Geographic jurisdiction. |
| `sector` | `VARCHAR(20)` | Demographic | No | `Rural`, `Urban` | Rural or Urban demographic sector. |
| `indicator_name` | `VARCHAR(100)` | Category | No | `Food vs Non-Food MPCE & Share`, `MPCE across Fractile Classes`, `MPCE and Share by Item Group`, `State-wise Average MPCE` | Standardized socioeconomic indicator category. |
| `item_group` | `VARCHAR(100)` | Basket item | No | `Cereals`, `Vegetables`, `Pulses and pulse products`, `Milk & milk products`, `Edible oil`, `Egg, fish & meat`, `Sugar`, `Total Food`, `Total Consumption` | Consumption item or expenditure aggregate. |
| `fractile_class` | `VARCHAR(50)` | Economic decile | No | `0-5%`, `5-10%`, `10-20%`, ..., `90-95%`, `95-100%`, `All Classes` | Household income/expenditure fractile class. Enables bottom 20% vs top 20% vulnerability indexing. |
| `mpce_value_rs` | `DECIMAL(10,2)` | Rs. per person per month | Yes | Strictly positive float `> 0.0` | Monthly Per Capita Consumption Expenditure in Indian Rupees. |
| `share_percentage` | `DECIMAL(5,2)` | Percentage (`%`) | Yes | Float (`0.0` to `100.0`) | Share of total household expenditure allocated to this item group. |
| `data_source` | `VARCHAR(150)` | Publication | No | `MoSPI NSSO HCES 2022-23 Factsheet Statement 1`, `Statement 4`, `Statement 5`, `Statement 8` | Precise official statistical statement from the HCES Factsheet. |

---

## 6. Analytical Guidelines & Best Practices

1. **Price Conversion Rules:**
   - **Never** divide wholesale prices (`Rs/Quintal`) by 100 to compare directly with retail (`Rs/Kg`) without documenting wholesale-to-retail transmission markup, APMC mandi fees, transportation costs, and retail packaging overheads.
2. **Missing Value Treatment:**
   - Market holidays and Sundays naturally produce missing mandi prices. Use forward-fill (`ffill`) bounded by 3 calendar days or Kaldate-spline interpolation only during active feature engineering, never in the raw processed layer.
3. **Outlier Filtering vs Anomaly Modeling:**
   - The `is_anomalous_flag` column in `wholesale_prices.csv` preserves genuine market distress spikes (e.g. Tomato price surge in July 2023) while flagging mechanical transmission errors (`min_price > max_price`).
4. **Inflation Calculation:**
   - Use point-in-time YoY inflation rates from `cpi_data.csv` rather than computing ad-hoc differences, ensuring strict alignment with official Reserve Bank of India (RBI) and MoSPI headline indicators.
