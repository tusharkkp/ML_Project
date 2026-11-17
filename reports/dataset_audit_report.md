# Comprehensive Dataset Audit & Evaluation Report

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Phase:** Phase 2 — Machine Learning Development (Dataset Audit)  
**Date:** 2026-11-17  
**Auditor:** Senior Machine Learning Engineer & Academic Project Mentor  
**Standard:** ISO 8601, Zero-Fabrication, Strict Retail vs. Wholesale Isolation  

---

## 1. Executive Summary of Audit

An exhaustive audit of the 5 standardized datasets in `data/processed/` and raw upstream sources was conducted. A total of **189,413 authentic records** were examined. 

The primary machine learning objective is:
> **Predicting the future price of essential food commodities using historical commodity price observations.**

### High-Level Suitability Matrix

| Dataset | Total Records | Date Span | Primary Frequency | ML Suitability Rating | Role in System Architecture |
|---|---:|:---:|:---:|:---:|---|
| **Wholesale Mandi Prices** (`wholesale_prices.csv`) | **71,307** | 2016-01-01 to 2026-11-17 | Daily | **Highest (Optimal)** | **Core ML Regression & Forecasting Target** |
| **Market Arrivals** (`market_arrivals.csv`) | **66,745** | 2016-01-01 to 2018-12-31 | Daily | **High** | **Exogenous Supply Feature for ML Models** |
| **Consumer Retail Prices** (`retail_prices.csv`) | **39,187** | 2010-01-01 to 2026-11-17 | Bi-weekly / Benchmark | **Moderate** | **Household Affordability & Markup Transmission** |
| **Consumer Price Index** (`cpi_data.csv`) | **12,022** | 2014-01-01 to 2025-12-01 | Monthly | **High (Macro)** | **Headline Food Inflation Tracking** |
| **Household Expenditure** (`household_expenditure.csv`) | **152** | 2022-23 (NSS 79th) | Survey Round | **Analytical Only** | **Deterministic Food Basket Budget Benchmarking** |

---

## 2. In-Depth Dataset Audits

---

### Dataset 1: Wholesale Mandi Prices (`data/processed/wholesale_prices.csv`)

- **Primary Source:** AGMARKNET (Directorate of Marketing & Inspection, Ministry of Agriculture) and data.gov.in API (Resource `9ef84268-d588-465a-a308-a864a43d0070`).
- **Record Count:** 71,307 rows.
- **Column Count:** 14 columns.
- **Schema & Data Types:**
  - `observation_date`: `object` (ISO 8601 string `YYYY-MM-DD`)
  - `commodity_name`: `object` (Canonical string, e.g. `Tomato`, `Onion`, `Wheat`)
  - `raw_commodity_name`: `object` (Raw reporting string)
  - `variety`: `object` (e.g. `Local`, `Hybrid`, `Red`, `Other`)
  - `state`: `object` (e.g. `Maharashtra`, `Madhya Pradesh`, `Uttar Pradesh`, `Gujarat`, `Karnataka`)
  - `district`: `object` (e.g. `Pune`, `Nashik`, `Ahmednagar`, `Nagpur`, `Jalgaon`)
  - `market_name`: `object` (e.g. `Pune`, `Sangamner`, `Lasalgaon`, `Pimpalgaon`, `Kolhapur`)
  - `min_price_rs_per_qtl`: `float64` (Minimum transacted wholesale auction price)
  - `max_price_rs_per_qtl`: `float64` (Maximum transacted wholesale auction price)
  - `modal_price_rs_per_qtl`: `float64` (Modal transacted price — most frequent quote)
  - `price_unit`: `object` (`Rs/Quintal`)
  - `price_type`: `object` (`Wholesale`)
  - `is_anomalous_flag`: `bool` (`True` if `min > max` or `modal` outside `[min, max]`)
  - `data_source`: `object` (Provenance label)
- **Temporal Coverage:** 2016-01-01 to 2026-11-17 (continuous daily observations).
- **Commodity Coverage:** 186 unique canonical commodities; heavily concentrated on essential perishables: **Onion (34,415 records)**, **Tomato (29,691 records)**, **Moong Dal (1,204 records)**, **Sugar (830 records)**, **Wheat (143 records)**.
- **Geographic Coverage:** 1,108 distinct APMC mandi market yards. 66,941 observations (93.9%) are from Maharashtra mandis (Pune, Sangamner, Pimpalgaon, Lasalgaon, Nagpur, Solapur, Kolhapur).
- **Retail vs. Wholesale Distinction:** Strictly wholesale. Reflects primary agricultural produce auctions conducted under APMC regulations.
- **Price Units:** Standardized exclusively to `Rs/Quintal` (1 Quintal = 100 kg).
- **Missing Values:** `min_price_rs_per_qtl`: 1 null (0.001%), `modal_price_rs_per_qtl`: 1 null (0.001%). All other 12 columns have 0 null values.
- **Duplicate Records:** 0 exact duplicate rows across all 71,307 observations.
- **Invalid Observations & Anomaly Flagging:** Exactly 2 records exhibit `is_anomalous_flag=True` (where `min_price > max_price` due to clerical entry transposition). Rather than dropping them silently, they are explicitly tagged.
- **Data Frequency:** Daily trading day frequency (excluding Sundays and official market closures).
- **Suitability for ML:** **Highest**. Dense historical daily observations for individual markets (e.g. Pune Tomato: 880 daily trading observations; Pune Onion: 869 daily trading observations) allow rigorous time-series train/val/test splitting, lag construction, and rolling feature engineering without cross-market pollution.

---

### Dataset 2: Market Arrivals (`data/processed/market_arrivals.csv`)

- **Primary Source:** AGMARKNET Physical Inflow Registers.
- **Record Count:** 66,745 rows.
- **Column Count:** 10 columns.
- **Schema & Data Types:**
  - `observation_date`: `object` (ISO 8601 string `YYYY-MM-DD`)
  - `commodity_name`: `object` (Canonical string)
  - `raw_commodity_name`: `object` (Raw string)
  - `variety`: `object` (Variety label)
  - `state`: `object` (`Maharashtra`)
  - `district`: `object` (District name)
  - `market_name`: `object` (APMC yard name)
  - `arrival_quantity_tonnes`: `float64` (Physical volume unloaded in metric tonnes)
  - `unit`: `object` (`Tonnes`)
  - `data_source`: `object` (Provenance label)
- **Temporal Coverage:** 2016-01-01 to 2018-12-31.
- **Commodity Coverage:** 4 core commodities: Onion (34,270 records), Tomato (30,442 records), Moong Dal (1,204 records), Sugar (829 records).
- **Geographic Coverage:** 109 APMC mandis across Maharashtra.
- **Missing Values:** `variety`: 904 nulls (1.35%). `arrival_quantity_tonnes`: 0 nulls. All other columns: 0 nulls.
- **Duplicates:** 0 duplicate rows.
- **Data Frequency:** Daily trading day frequency.
- **Suitability for ML:** **High (Exogenous Feature)**. Exactly matches the observation dates of `wholesale_prices.csv`. Provides the essential supply-side metric ($Q_t$) needed to model supply-demand elasticity and price shocks.

---

### Dataset 3: Consumer Retail Prices (`data/processed/retail_prices.csv`)

- **Primary Source:** Department of Consumer Affairs (DCA) Price Monitoring Division.
- **Record Count:** 39,187 rows.
- **Column Count:** 12 columns.
- **Schema & Data Types:**
  - `observation_date`: `object` (`YYYY-MM-DD`)
  - `commodity_name`: `object` (Canonical string)
  - `raw_commodity_name`: `object`
  - `state`: `object`
  - `district`: `object`
  - `reporting_centre`: `object` (e.g. `Mumbai`, `Pune`, `Nagpur`, `Nashik`, `All-India Average`)
  - `market_type`: `object` (`Reporting Centre Retail`, `Retail Average`)
  - `retail_price`: `float64` (Consumer price per physical unit)
  - `price_unit`: `object` (`1 Kg`, `1 Litre`)
  - `price_type`: `object` (`Retail`)
  - `benchmark_tag`: `object` (`Historical Daily`, `Current Daily Average`, etc.)
  - `data_source`: `object`
- **Temporal Coverage:** 2010-01-01 to 2026-11-17.
- **Commodity Coverage:** 41 food commodities (Rice, Wheat, Atta, Pulses, Vegetables, Edible Oils, Milk, Sugar).
- **Geographic Coverage:** 60 urban monitoring centres across India, including Mumbai, Pune, Nagpur, and Nashik.
- **Retail vs. Wholesale Distinction:** Strictly retail. Captures consumer quotes paid by urban households.
- **Price Units:** `1 Kg` (30,061 rows) and `1 Litre` (9,064 rows).
- **Missing Values:** 0 missing values across all columns.
- **Duplicates:** 0 duplicate rows.
- **Data Frequency:** Mixed historical frequency (bi-weekly reporting from 2010 to 2012; daily point-in-time and benchmark averages for 2024–2026).
- **Suitability for ML:** **Moderate (Affordability Target)**. While highly valuable for calculating the Household Food Basket Cost and studying retail markups, individual centre time series have fewer dense continuous daily records (48–50 per centre in the 2010–2012 archive) compared to the APMC wholesale series.

---

### Dataset 4: Consumer Price Index (`data/processed/cpi_data.csv`)

- **Primary Source:** Ministry of Statistics and Programme Implementation (MoSPI) API.
- **Record Count:** 12,022 rows.
- **Column Count:** 13 columns.
- **Schema & Data Types:**
  - `observation_date`: `object` (`YYYY-MM-01`)
  - `year`: `int64` (2014 to 2025)
  - `month`: `object` (e.g. `January` through `December`)
  - `geography`: `object` (`All India`, `Maharashtra`)
  - `sector`: `object` (`Rural`, `Urban`, `Combined`)
  - `category_type`: `object` (`Group / Subgroup`, `Item`)
  - `category_name`: `object` (Canonical category string)
  - `cpi_index`: `float64` (Base 2012 = 100)
  - `inflation_rate_yoy`: `float64` (Annual Year-on-Year inflation rate %)
  - `base_year`: `int64` (2012)
- **Temporal Coverage:** 2014-01-01 to 2025-12-01 (144 consecutive monthly periods).
- **Missing Values:** `subgroup`: 1,551 nulls (natural for group-level entries). `inflation_rate_yoy`: 2,456 nulls (primarily early 2014–2015 observations before base conversion).
- **Suitability for ML:** **High for Macro Forecasting**. Ideal for autoregressive monthly forecasting, but not appropriate for daily commodity price prediction.

---

### Dataset 5: Household Consumption Expenditure (`data/processed/household_expenditure.csv`)

- **Primary Source:** NSSO Household Consumption Expenditure Survey (HCES 2022-23 Factsheet).
- **Record Count:** 152 rows.
- **Column Count:** 9 columns.
- **Schema & Data Types:**
  - `survey_year`: `object` (`2022-23`)
  - `geography`: `object` (`All India`, 36 States/UTs including `Maharashtra`)
  - `sector`: `object` (`Rural`, `Urban`)
  - `indicator_name`: `object` (`Food vs Non-Food MPCE & Share`, `MPCE across Fractile Classes`, `MPCE and Share by Item Group`, `State-wise Average MPCE`)
  - `fractile_class`: `object` (12 fractile classes: `0-5%`, `5-10%`, ..., `95-100%`, `All Classes`)
  - `mpce_value_rs`: `float64` (Monthly Per Capita Consumption Expenditure)
  - `share_percentage`: `float64` (% budget share)
- **Suitability for ML:** **Analytical / Benchmark Only**. This is a cross-sectional decile survey, not a time series. It provides the ground-truth income/expenditure baselines for the **Household Affordability Analyzer** and **Household Food Basket Calculator**.

---

## 3. Selection of Primary Machine Learning Problem

### Selected Commodity and Market Series
- **Commodity:** **Tomato** (with parallel verification on **Onion**)
- **Market:** **Pune APMC Mandi, Maharashtra**
- **Justification:**
  1. **Data Density & Continuity:** 880 continuous daily observations from 2016-01-01 to 2018-12-31, representing every active auction trading day.
  2. **High Economic Volatility:** Tomato exhibits substantial price variance ($\mu = 1186.16$ Rs/Qtl, $\sigma = 1035.11$ Rs/Qtl, range: $300$ to $5700$ Rs/Qtl), reflecting genuine monsoon supply disruptions, post-harvest gluts, and price spikes. This non-linear dynamic provides a rigorous test for regression models.
  3. **Matched Exogenous Supply Inflows:** Exactly 880 corresponding daily physical arrival records ($\mu = 170.88$ Tonnes/day) are available, enabling supply-demand price interaction modeling.
  4. **Geographic Alignment:** Pune is one of the four mandated Maharashtra urban centers, directly linking wholesale market predictions to urban household grocery affordability.

### Target Variable Formulation
- Primary Target: **$t+1$ Next-Trading-Day Modal Price** (`target_price_t_plus_1`), with rolling 7-day trend analysis.
- Because agricultural mandis do not trade on Sundays and festive holidays, shifting calendar days directly introduces missing values. Formulating the target as the **next active trading session** preserves complete data integrity while enabling practical, operational trading day forecasts.

---

## 4. Methodological Compliance & Ethical Guardrails

1. **Strict Physical Separation:** Wholesale prices (`Rs/Quintal`) and Retail prices (`Rs/Kg`) are strictly maintained in separate datasets.
2. **Zero Fabrication:** Missing Sunday dates are not artificially interpolated or forward-filled in the raw storage layer.
3. **Outlier Integrity:** Extreme prices (e.g. Tomato reaching 5,700 Rs/Qtl in July 2017) are confirmed genuine supply shocks and are retained rather than discarded.
4. **Data Leakage Prevention:** Chronological train (70%), validation (15%), and test (15%) splits ensure no future information leaks into training or scaling transformations.
