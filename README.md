# AI-Based Food Inflation Prediction and Household Affordability Monitoring System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Data Acquisition & Processing](https://img.shields.io/badge/Pipeline-Acquisition%20%26%20Cleaned-brightgreen.svg)]()

An academic data engineering and econometric pipeline designed to acquire, standardize, validate, and prepare real-world agricultural commodity prices, macroeconomic inflation indices, and household consumption expenditure data.

The project strictly focuses on high-frequency commodity monitoring in **Maharashtra (Pune, Mumbai, Nashik, Nagpur)** and All-India benchmarks for food inflation forecasting and household economic vulnerability monitoring.

---

## 1. System Architecture & Pipeline Workflow

```
       [ Public & Official Data Sources ]
  +------------------+  +------------------+  +------------------+
  |  MoSPI CPI API   |  |   data.gov.in    |  |  CEDA Agmarknet  |
  |  (Base 2012=100) |  |   Mandi API      |  |     OpenAPI      |
  +--------+---------+  +--------+---------+  +--------+---------+
           |                     |                     |
  +--------v---------+  +--------v---------+           |
  | DCA Retail Price |  | AGMARKNET Mandi  |           |
  |  Bulletin & Web  |  | Arrivals & Prices|           |
  +--------+---------+  +--------+---------+           |
           |                     |                     |
           +----------+----------+---------------------+
                      |
           [ scripts/api/ & scripts/scraping/ ]
                      |
                      v
             data/raw/ (Immutable)
    +-----------------------------------------------+
    | retail/  wholesale/  cpi/  market_arrivals/   |
    | household_expenditure/                        |
    +-----------------------+-----------------------+
                            |
           [ scripts/processing/clean_data.py ]
           [ scripts/processing/normalize_data.py ]
                            |
                            v
            data/processed/ (Standardized)
    +-----------------------------------------------+
    | 1. retail_prices.csv        (Rs/Kg, Litre)    |
    | 2. wholesale_prices.csv     (Rs/Quintal)      |
    | 3. market_arrivals.csv      (Tonnes)          |
    | 4. cpi_data.csv             (Index, YoY %)    |
    | 5. household_expenditure.csv(MPCE, Deciles)   |
    +-----------------------+-----------------------+
                            |
           [ scripts/processing/validate_data.py ]
                            |
                            v
             reports/ & data/metadata/
    +-----------------------------------------------+
    | - reports/data_quality_report.md              |
    | - reports/collection_summary.md               |
    | - reports/source_discovery_report.md          |
    | - data/metadata/data_dictionary.md            |
    +-----------------------------------------------+
```

---

## 2. Methodological Standards & Zero-Fabrication Policy

1. **Strict Retail and Wholesale Price Isolation:**
   - **Retail prices** represent consumer transactions at urban monitoring centres (Rs/Kg or Rs/Litre).
   - **Wholesale prices** represent primary APMC auction trades (Rs/Quintal).
   - **Rule:** These series are never merged or silently converted. Retail-to-wholesale markups must be modeled explicitly.
2. **Standardized Temporal & Metric Units:**
   - All dates conform to ISO 8601 (`YYYY-MM-DD`).
   - Physical arrivals are converted to Metric Tonnes.
3. **Transparent Anomaly Management:**
   - Rather than dropping rows with irregular reporting relationships (`min_price > max_price` or `modal_price` outside bounds), anomalies are tagged with `is_anomalous_flag=True` to serve as ground truth for Isolation Forest models.
4. **Ethical Harvester Boundaries:**
   - Strictly compliant with `robots.txt` and rate limits.
   - Gated endpoints requiring interactive CAPTCHAs (e.g., DCA report form) or anti-bot cookies (e.g., RBI F5 TSPD) are respected without bypass; clean academic and official open data mirrors are leveraged instead.

---

## 3. Directory Layout

```
ML_Project/
├── .env.example                       <- Template for API credentials
├── .gitignore                          <- Git exclusion rules (raw dumps, .env)
├── requirements.txt                    <- Python dependency specification
├── README.md                           <- Master documentation and user guide
│
├── data/
│   ├── raw/                            <- Immutable raw files from APIs and harvesters
│   │   ├── retail/                     <- DCA live bulletins & historical matrices
│   │   ├── wholesale/                  <- Agmarknet multi-year mandi CSVs
│   │   ├── market_arrivals/            <- Mandi arrivals
│   │   ├── cpi/                        <- MoSPI CPI JSONs, base metadata
│   │   └── household_expenditure/      <- NSSO HCES 2022-23 factsheet & statements
│   ├── processed/                      <- Five standardized analytical CSVs
│   │   ├── retail_prices.csv
│   │   ├── wholesale_prices.csv
│   │   ├── market_arrivals.csv
│   │   ├── cpi_data.csv
│   │   └── household_expenditure.csv
│   └── metadata/                       <- Catalogs, schemas, and logs
│       ├── data_dictionary.md          <- Complete data dictionary & field specifications
│       ├── data_sources.csv            <- Master catalog of all evaluated sources
│       └── collection_logs/            <- Audit logs for API, scraping, and validation
│
├── data_sources/
│   └── source_inventory.md             <- 10-source technical evaluation & feasibility matrix
│
├── notebooks/
│   └── 01_data_collection_overview.ipynb <- Interactive walkthrough of all processed corpora
│
├── reports/
│   ├── source_discovery_report.md      <- Technical discovery, endpoints, and limitations
│   ├── data_quality_report.md          <- Statistical validation & coverage audit
│   └── collection_summary.md           <- Final executive summary & phase readiness
│
└── scripts/
    ├── main.py                         <- Master pipeline CLI orchestrator
    ├── api/
    │   ├── mospi_client.py             <- MoSPI CPI & HCES API client
    │   ├── data_gov_client.py          <- data.gov.in APMC mandi client
    │   └── ceda_client.py              <- CEDA Agmarknet OpenAPI client
    ├── scraping/
    │   └── relevant_scrapers.py        <- DCA bulletin scraper & Agmarknet harvesters
    └── processing/
        ├── normalize_data.py           <- Canonical mappings (commodities, geography, dates)
        ├── clean_data.py               <- Transformation pipeline generating processed CSVs
        └── validate_data.py            <- Quality assurance and Markdown report generator
```

---

## 4. Installation & Environment Setup

### Prerequisites

- Python 3.10+ (Python 3.12 recommended)
- `pip` package manager

### 1. Clone & Install Dependencies

```bash
# Clone the repository
git clone <repository_url>
cd ML_Project

# Install required Python packages
py -3.12 -m pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env`:

```bash
copy .env.example .env
```

Open `.env` and configure your credentials:

```ini
# data.gov.in API Key (Generated from https://www.data.gov.in/)
DATA_GOV_IN_API_KEY=your_actual_api_key_here

# MoSPI API (Open public access; no token required for CPI/HCES)
MOSPI_API_BASE=https://api.mospi.gov.in/api

# CEDA API Bearer Token (Optional; register at https://api.ceda.ashoka.edu.in/)
CEDA_API_TOKEN=your_optional_ceda_token
```

---

## 5. Running the Pipeline

The master pipeline is orchestrated via `scripts/main.py`:

```bash
# Display summary of all processed datasets
py -3.12 scripts/main.py --summary

# Run the complete end-to-end pipeline (API -> Scrape -> Clean -> Validate)
py -3.12 scripts/main.py --all

# Run individual pipeline stages:
py -3.12 scripts/main.py --collect-api   # Run MoSPI and data.gov.in API clients
py -3.12 scripts/main.py --scrape        # Scrape DCA bulletins & harvest historical records
py -3.12 scripts/main.py --clean         # Standardize raw data into data/processed/
py -3.12 scripts/main.py --validate      # Validate data quality & generate reports
```

---

## 6. Acquired Datasets Summary

| Dataset                    | Scope                                                  |   Temporal Span    | Granularity              | Observations |
| -------------------------- | ------------------------------------------------------ | :----------------: | ------------------------ | -----------: |
| **Retail Prices**          | Maharashtra (Mumbai, Pune, Nashik, Nagpur) + All India |   2010 – Present   | Daily / Centre-wise      |       39,187 |
| **Wholesale Mandi Prices** | Maharashtra APMCs & Key States (Lasalgaon, Pune, etc.) |   2016 – Present   | Daily / Mandi-wise       |       71,307 |
| **Market Arrivals**        | Maharashtra APMCs                                      |    2016 – 2018     | Daily / Mandi-wise       |       66,745 |
| **MoSPI CPI Series**       | All-India & Maharashtra (Rural, Urban, Combined)       |    2014 – 2026     | Monthly / Group & Item   |       12,022 |
| **Household Expenditure**  | All-India & Maharashtra                                | NSS 79th (2022-23) | Fractile Deciles & Items |          152 |

---

## 7. Analytical Readiness for Next Phase (Phase 2: EDA & ML)

- **Time Series Forecasting:** Multi-year daily mandi and retail price series ready for SARIMA, Facebook Prophet, LSTM, and XGBoost models.
- **Supply Shock & Price Spike Detection:** Paired arrival volumes and modal prices ready for supply elasticity modeling and Isolation Forest anomaly tagging.
- **Household Affordability Indexing:** Real-time retail basket costs combined with NSSO 2022-23 fractile MPCE distributions allow monitoring food budget distress across bottom 20% vs top 20% income deciles.
