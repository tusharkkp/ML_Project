# Data Quality, Coverage, and Validation Report
**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Generated:** 2026-11-17 19:29:28  
**Author:** Senior Data Engineer & Data Acquisition Specialist  

---

## 1. Executive Summary of Acquired Datasets

| Dataset | Status | Total Records | Date Coverage | Unique Commodities | Locations | Maharashtra Records | Missing Value % | Duplicates |
|---|:---:|---:|:---:|---:|---:|---:|---:|---:|
| **Retail Prices** | Verified | 39,187 | 2010-01-01 to 2026-11-17 | 41 | 60 | 1,538 | 0.0% | 0 |
| **Wholesale Mandi Prices** | Verified | 71,307 | 2016-01-01 to 2026-11-17 | 186 | 1108 | 66,941 | 0.0% | 0 |
| **Market Arrivals** | Verified | 66,745 | 2016-01-01 to 2018-12-31 | 4 | 109 | 66,745 | 0.14% | 0 |
| **Consumer Price Index (CPI)** | Verified | 12,022 | 2014-01-01 to 2025-12-01 | 24 | 2 | 5,244 | 2.56% | 0 |
| **Household Consumption Expenditure (HCES)** | Verified | 152 | 2022-23 to 2022-23 | 27 | 38 | 2 | 7.31% | 0 |

---

## 2. Dataset-by-Dataset In-Depth Analysis

### Retail Prices
- **File Path:** `data/processed/retail_prices.csv`
- **Record Count:** 39,187
- **Temporal Coverage:** 2010-01-01 to 2026-11-17
- **Geographic Granularity:** 60 reporting centres/markets across `reporting_centre`
- **Maharashtra Focus:** 1,538 records (3.9% of dataset)
- **Missing Value Ratio:** 0.0%
- **Duplicate Observations:** 0
- **Invalid Price Records:** 0
- **Price Unit Distribution:** `{'1 Kg': 30061, '1 Litre': 9064, '100 gm': 24, '250 gm': 14, '1 Dozen': 6, '1 Ltr': 6, '1 kg': 6, '50 gm': 6}`
- **Price Type Distribution:** `{'Retail': 39187}`
- **Primary Data Sources:** `{'DCA Retail Monitoring Historical Archive': 38942, 'DCA Price Monitoring Bulletin (fcainfoweb.nic.in)': 245}`
- **Top Commodities / Groups:** Sugar (2017), Rice (2015), Onion (2014), Wheat (Atta) (2012), Potato (2008)

**ML & Analytical Suitability Assessment:**
- *Time Series Forecasting:* High (Multi-year sequential observations)
- *Anomaly Detection:* High (Min/Modal/Max relationships & daily variance)
- *City Comparison:* High (Separated market & centre reporting)
- *Affordability Calculation:* High (Official MPCE and price inputs)


### Wholesale Mandi Prices
- **File Path:** `data/processed/wholesale_prices.csv`
- **Record Count:** 71,307
- **Temporal Coverage:** 2016-01-01 to 2026-11-17
- **Geographic Granularity:** 1108 reporting centres/markets across `market_name`
- **Maharashtra Focus:** 66,941 records (93.9% of dataset)
- **Missing Value Ratio:** 0.0%
- **Duplicate Observations:** 0
- **Invalid Price Records:** 2
- **Price Unit Distribution:** `{'Rs/Quintal': 71307}`
- **Price Type Distribution:** `{'Wholesale': 71307}`
- **Primary Data Sources:** `{'AGMARKNET Historical Mandi Archive': 65841, 'data.gov.in AGMARKNET API (9ef84268-d588-465a-a308-a864a43d0070)': 5466}`
- **Top Commodities / Groups:** Onion (34415), Tomato (29691), Green Gram Dal (Moong Dal) (1204), Sugar (830), Paddy (427)

**ML & Analytical Suitability Assessment:**
- *Time Series Forecasting:* High (Multi-year sequential observations)
- *Anomaly Detection:* High (Min/Modal/Max relationships & daily variance)
- *City Comparison:* High (Separated market & centre reporting)
- *Affordability Calculation:* Medium


### Market Arrivals
- **File Path:** `data/processed/market_arrivals.csv`
- **Record Count:** 66,745
- **Temporal Coverage:** 2016-01-01 to 2018-12-31
- **Geographic Granularity:** 109 reporting centres/markets across `market_name`
- **Maharashtra Focus:** 66,745 records (100.0% of dataset)
- **Missing Value Ratio:** 0.14%
- **Duplicate Observations:** 0
- **Invalid Price Records:** 0
- **Price Unit Distribution:** `{'Tonnes': 66745}`
- **Price Type Distribution:** `{}`
- **Primary Data Sources:** `{'AGMARKNET Mandi Arrivals': 66745}`
- **Top Commodities / Groups:** Onion (34270), Tomato (30442), Green Gram Dal (Moong Dal) (1204), Sugar (829)

**ML & Analytical Suitability Assessment:**
- *Time Series Forecasting:* High (Multi-year sequential observations)
- *Anomaly Detection:* Medium
- *City Comparison:* High (Separated market & centre reporting)
- *Affordability Calculation:* Medium


### Consumer Price Index (CPI)
- **File Path:** `data/processed/cpi_data.csv`
- **Record Count:** 12,022
- **Temporal Coverage:** 2014-01-01 to 2025-12-01
- **Geographic Granularity:** 2 reporting centres/markets across `geography`
- **Maharashtra Focus:** 5,244 records (43.6% of dataset)
- **Missing Value Ratio:** 2.56%
- **Duplicate Observations:** 0
- **Invalid Price Records:** 2456
- **Price Unit Distribution:** `{}`
- **Price Type Distribution:** `{}`
- **Primary Data Sources:** `{'MoSPI Consumer Price Index API (api.mospi.gov.in)': 12022}`
- **Top Commodities / Groups:** Fruits (823), Pulses And Products (820), Cereals And Products (816), Oils And Fats (816), Egg (811)

**ML & Analytical Suitability Assessment:**
- *Time Series Forecasting:* High (Multi-year sequential observations)
- *Anomaly Detection:* Medium
- *City Comparison:* Medium
- *Affordability Calculation:* Medium


### Household Consumption Expenditure (HCES)
- **File Path:** `data/processed/household_expenditure.csv`
- **Record Count:** 152
- **Temporal Coverage:** 2022-23 to 2022-23
- **Geographic Granularity:** 38 reporting centres/markets across `geography`
- **Maharashtra Focus:** 2 records (1.3% of dataset)
- **Missing Value Ratio:** 7.31%
- **Duplicate Observations:** 0
- **Invalid Price Records:** 100
- **Price Unit Distribution:** `{}`
- **Price Type Distribution:** `{}`
- **Primary Data Sources:** `{'MoSPI NSSO HCES 2022-23 Factsheet Statement 8': 74, 'MoSPI NSSO HCES 2022-23 Factsheet Statement 5': 48, 'MoSPI NSSO HCES 2022-23 Factsheet Statement 4': 26, 'MoSPI NSSO HCES 2022-23 Factsheet Statement 1': 4}`
- **Top Commodities / Groups:** Total Consumption (100), Food (2), Non-food (2), cereals & cereal substitutes (2), pulses & their products* (2)

**ML & Analytical Suitability Assessment:**
- *Time Series Forecasting:* Medium
- *Anomaly Detection:* Medium
- *City Comparison:* High (Separated market & centre reporting)
- *Affordability Calculation:* High (Official MPCE and price inputs)


---

## 3. Methodological Integrity & Validation Rules Applied

1. **Strict Retail and Wholesale Price Separation:**
   - Retail prices are stored exclusively in `retail_prices.csv` (measured in Rs/Kg or Rs/Litre).
   - Wholesale prices are stored exclusively in `wholesale_prices.csv` (measured in Rs/Quintal).
   - Neither series has been silently converted or mixed into the other.
2. **Date Standardization:**
   - All dates converted to ISO 8601 `YYYY-MM-DD`.
3. **Commodity Canonical Naming:**
   - Unified nomenclature (e.g. `Tomato`, `Onion`, `Potato`, `Tur Dal`, `Moong Dal`, `Sugar`) applied while preserving raw source names in `raw_commodity_name`.
4. **Zero Fabrication Policy:**
   - No synthetic records or undocumented interpolations have been introduced.
5. **Outlier and Anomaly Tagging:**
   - Wholesale records where `min_price > max_price` or `modal_price` falls outside `[min_price, max_price]` are flagged with `is_anomalous_flag=True` rather than silently purged, preserving data transparency for downstream Isolation Forest training.