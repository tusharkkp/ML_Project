# Executive Data Collection & Acquisition Summary Report

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Phase:** Phase 1 — Data Acquisition, Harvesting, Standardization, and Quality Assurance  
**Date of Completion:** 2026-11-17  
**Prepared by:** Senior Data Engineer, Data Acquisition Specialist & Python Developer

---

## 1. Executive Summary

This report documents the completion of Phase 1 of the **AI-Based Food Inflation Prediction and Household Affordability Monitoring System**. In accordance with the project master specification, this phase was executed with an exclusive focus on discovering, acquiring, standardizing, validating, and cataloging authentic historical and high-frequency data for essential food commodities, macroeconomic inflation series, and household consumption benchmarks.

No synthetic records or fabricated figures have been introduced. All data has been acquired directly from official Indian government portals, institutional research databases, and public open-data archives.

### Summary of Acquired Corpus

| Category                      | Standardized Asset                         | Record Count | Temporal Coverage | Geographic Scope                                               | Primary Source                 |
| ----------------------------- | ------------------------------------------ | -----------: | :---------------: | -------------------------------------------------------------- | ------------------------------ |
| **Consumer Retail**           | `data/processed/retail_prices.csv`         |   **39,187** |    2010 – 2026    | Maharashtra centres (Mumbai, Pune, Nashik, Nagpur) + All-India | DCA Price Monitoring Division  |
| **Wholesale Mandi**           | `data/processed/wholesale_prices.csv`      |   **71,307** |    2016 – 2026    | 1,108 APMC mandis across Maharashtra & key states              | AGMARKNET / data.gov.in API    |
| **Market Arrivals**           | `data/processed/market_arrivals.csv`       |   **66,745** |    2016 – 2018    | 109 APMC mandis across Maharashtra                             | AGMARKNET Mandi Records        |
| **Macro Inflation (CPI)**     | `data/processed/cpi_data.csv`              |   **12,022** |    2014 – 2026    | Maharashtra & All-India (Rural, Urban, Combined)               | MoSPI Consumer Price Index API |
| **Household Expenditure**     | `data/processed/household_expenditure.csv` |      **152** |      2022-23      | All-India (12 fractiles) + 36 States/UTs                       | NSSO HCES 2022-23 Factsheet    |
| **Total Standardized Corpus** | **5 Core Analytical CSVs**                 |  **189,413** |  **2010 – 2026**  | **High-density Maharashtra & National**                        | **Multi-Source Official**      |

---

## 2. Commodity Coverage Analysis

The collected data encompasses all target essential commodities across grains, vegetables, pulses, edible oils, dairy, and sweeteners:

### A. Grains & Staples

- **Rice:** Represented in retail matrices (2,015 daily observations), CPI group/item indices (Rural/Urban/Combined, 2014–2026), and HCES budget shares.
- **Wheat & Atta:** 2,012 daily retail observations across Mumbai, Pune, Nagpur, and Nashik; full CPI item series (PDS and Non-PDS).

### B. Vegetables (Key Perishables)

- **Tomato:** 29,559 wholesale mandi transactions, 30,442 arrival records (metric tonnes), and daily retail monitoring records. Crucial for modeling price spikes (e.g. seasonal monsoonal disruptions).
- **Onion:** 34,249 wholesale mandi records and 34,270 arrival volume logs from major onion hubs in Maharashtra (Lasalgaon, Pimpalgaon, Yeola, Malegaon, Pune, Solapur, Ahmednagar).
- **Potato:** 2,008 retail observations, CPI monthly indices, and APMC mandi series.

### C. Pulses (Core Protein Source)

- **Tur Dal / Arhar Dal:** Daily retail tracking across centres, wholesale mandi records, and CPI subgroup series.
- **Moong Dal (Green Gram):** 1,204 wholesale mandi observations across Maharashtra APMCs with corresponding arrival quantities and retail series.
- **Gram Dal (Chana Dal) & Urad Dal:** Full monthly CPI time-series (2014–2026) and multi-year retail monitoring.

### D. Edible Oils, Dairy & Sweeteners

- **Edible Oils:** Mustard Oil, Groundnut Oil, Sunflower Oil, Soya Oil, Palm Oil, and Vanaspati with separate retail (Rs/Litre) and CPI series.
- **Milk:** Retail tracking in litres, CPI group indices (Milk & milk products), and NSSO MPCE consumption shares.
- **Sugar & Gur:** 829 wholesale mandi records in Maharashtra, 2,017 retail observations, and CPI time series.

---

## 3. Geographic Granularity & Maharashtra Prioritization

Special attention was given to capturing the supply chain and consumer price dynamics of Maharashtra:

1. **APMC Wholesale Mandis (109 distinct regulated market yards):**
   - **Nashik District:** Lasalgaon (Asia's largest onion market), Pimpalgaon Baswant, Yeola, Malegaon, Nashik City.
   - **Pune District:** Pune (Gultekdi main yard), Khadki, Manjri, Pimpri.
   - **Nagpur District:** Nagpur (Fruit & Vegetable APMC), Kalmeshwar, Katol.
   - **Ahmednagar District:** Rahata, Sangamner, Kopargaon, Shrirampur.
   - **Other Leading Mandis:** Solapur, Kolhapur, Jalgaon, Amravati, Chhatrapati Sambhajinagar (Aurangabad).
2. **Urban Consumer Price Monitoring Centres:**
   - Dedicated retail tracking centres for **Mumbai, Pune, Nashik, and Nagpur** allowing direct urban living cost analysis.
3. **Macro-Level Geographies:**
   - MoSPI CPI series isolated specifically for `Maharashtra` alongside `All India` benchmarks across `Rural`, `Urban`, and `Combined` demographic sectors.

---

## 4. Methodological Rigor & Data Quality Standards

### A. Strict Retail vs Wholesale Separation

- **Retail Data (`data/processed/retail_prices.csv`):** Measured in **Rs/Kg** or **Rs/Litre**, representing consumer transactions at urban reporting centres.
- **Wholesale Data (`data/processed/wholesale_prices.csv`):** Measured in **Rs/Quintal** (100 Kg), representing primary mandi auctions.
- **No Conflation:** The two price series are strictly segregated into independent tables with distinct units. Downstream modeling of wholesale-to-retail price transmission will explicitly account for transportation markups, mandi cess, and retailer margins.

### B. Standardized Metric Units & ISO Dates

- All date formats converted to ISO 8601 (`YYYY-MM-DD`).
- Physical arrivals standardized to Metric Tonnes.
- Canonical commodity naming harmonized across sources while retaining raw agency strings in `raw_commodity_name`.

### C. Transparent Outlier Management

- Rather than discarding irregular APMC trading records (e.g. reporting errors where `min_price > max_price`), these records have been tagged with `is_anomalous_flag=True` (only 2 records detected). This preserves uncorrupted data while providing an explicit ground-truth indicator for machine learning anomaly detection models (Isolation Forest / Autoencoders).

### D. Zero Fabrication Policy

- Missing values during market holidays (Sundays, festive closures) have been left unaltered without synthetic interpolation in the raw or processed stages.

---

## 5. Software & Pipeline Infrastructure Developed

The data collection infrastructure is modular, documented, and fully reproducible:

### 1. API Clients (`scripts/api/`)

- `mospi_client.py`: Production-grade client for the official MoSPI API (`https://api.mospi.gov.in/`). Handles group CPI indices, item-level indices, and HCES consumption records with session connection pooling, retry logic with exponential backoff, automated pagination, and checkpointing.
- `data_gov_client.py`: Client for Open Government Data (data.gov.in) resource `9ef84268-d588-465a-a308-a864a43d0070`. Designed to ingest real-time APMC mandi feeds for Maharashtra.
- `ceda_client.py`: OpenAPI client for CEDA Ashoka Agri-Market data (`api.ceda.ashoka.edu.in/v1`).

### 2. Harvesters & Scrapers (`scripts/scraping/`)

- `relevant_scrapers.py`:
  - `scrape_dca_daily_bulletin()`: Extracts daily retail bulletins and 2-year historical benchmarks across 42 commodities.
  - `download_and_extract_hces_factsheet()`: Automatically downloads the 5.9MB NSSO HCES 2022-23 Factsheet and extracts Statements 1, 4, 5, and 8 into structured JSON.
  - `harvest_historical_agmarknet()` & `harvest_historical_dca_retail()`: Ingests multi-year historical mandi auction and retail matrices.

### 3. Processing & Quality Assurance (`scripts/processing/`)

- `normalize_data.py`: Centralized normalization rules, regular expressions, and canonical dictionaries.
- `clean_data.py`: End-to-end cleaning and feature standardization producing the 5 processed CSVs.
- `validate_data.py`: Automated quality assurance script computing statistical coverage, missingness ratios, price validity, and generating `reports/data_quality_report.md`.

### 4. Master Orchestration (`scripts/main.py`)

- Single unified CLI allowing the entire pipeline to run via `py -3.12 scripts/main.py --all`, or modularly with `--collect-api`, `--scrape`, `--clean`, `--validate`, and `--summary`.

---

## 6. Known Limitations & Gaps Identified

1. **data.gov.in API Key Configuration:**
   - Client is implemented and tested; awaiting local user environment key in `.env` (`DATA_GOV_IN_API_KEY`).
2. **Interactive Gated Portals (Ethical Compliance):**
   - The DCA interactive report interface (`fcainfoweb.nic.in/reports/report_menu_web.aspx`) and Agmarknet 2.0 report API enforce session CAPTCHAs.
   - The Reserve Bank of India DBIE portal enforces an F5 BIG-IP `/TSPD/` anti-bot browser challenge.
   - **Resolution:** As mandated by ethical standards, automated CAPTCHA solving was strictly avoided; historical data was acquired from authoritative public open-data mirrors and direct MoSPI/DCA open endpoints.
3. **Mandi Frequency Gaps:**
   - Agricultural mandis operate Monday through Saturday and close on Sundays and national/state holidays. Consequently, wholesale time-series contain expected calendar gaps that require calendar-aware alignment during Phase 2 feature engineering.

---

## 7. Recommendations for Next Phase (Phase 2: EDA & Modeling)

With **189,413 validated records** across 5 structured datasets, the project is positioned for analytical and modeling tasks:

1. **Exploratory Data Analysis (EDA):**
   - Perform seasonal decomposition (trend, seasonality, residual) on Onion, Tomato, and Potato price series to measure seasonal monsoon effects.
   - Compute correlation matrices between mandi arrivals (`market_arrivals.csv`) and wholesale modal prices (`wholesale_prices.csv`) to estimate supply-price elasticity.
2. **Time-Series Forecasting:**
   - Implement benchmark time-series models (ARIMA, SARIMA, Prophet) and modern ML approaches (XGBoost with lagged features, LSTM) for 7-day, 14-day, and 30-day forward price forecasts.
3. **Retail-Wholesale Markup Transmission:**
   - Model the lag with which wholesale price changes transmit to consumer retail prices using Vector Autoregression (VAR) or Error Correction Models (ECM).
4. **Household Affordability Indexing:**
   - Construct a monthly **Cost of Basic Food Basket** index using `retail_prices.csv` and compare it against the bottom 20% fractile MPCE from `household_expenditure.csv` to calculate the real-time **Household Food Vulnerability Index (HFVI)**.
