# Source Discovery & Feasibility Report
**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Date:** November 2026  
**Author:** Senior Data Engineer & Data Acquisition Specialist  

---

## 1. Executive Summary

This report presents the findings of our extensive empirical investigation into data sources for food commodity prices, inflation indices, market arrivals, and household expenditure across India, with special prioritization of Maharashtra (Pune, Mumbai, Nashik, Nagpur).

Between API inspections, JavaScript bundle deconstructions, network trace analyses, and official manual reviews, **10 data sources** were evaluated. The investigation identified:
1. **An open, zero-authentication official MoSPI REST API** (`api.mospi.gov.in`) providing over 10 years of monthly Consumer Price Index (CPI) and Food Inflation indices at All-India and Maharashtra levels, including item-level metrics and Household Consumption Expenditure Survey (HCES) records.
2. **A live Open Government Data (data.gov.in) Mandi Price API** (`resource/9ef84268-d588-465a-a308-a864a43d0070`) syndicating daily AGMARKNET market prices and arrivals, ready to be queried with the user's API key.
3. **A publicly accessible Department of Consumer Affairs (DCA) Price Monitoring bulletin** (`fcainfoweb.nic.in/Reports/DB/DBprices.aspx`) providing daily All-India average retail prices for 22 essential commodities with benchmark historical comparisons (1 week, 1 month, 6 months, 1 year, and 2 years back).
4. **Curated multi-year daily Agmarknet and DCA historical archives (2010–2021)** capturing market-level arrivals, minimum, maximum, and modal wholesale prices in Maharashtra mandis, alongside daily centre-level retail prices.
5. **Clear ethical boundaries**: AGMARKNET 2.0 (`agmarknet.gov.in`), DCA historical report forms (`fcainfoweb.nic.in/reports/report_menu_web.aspx`), and RBI DBIE (`rbidocs.rbi.org.in`) enforce interactive CAPTCHAs or F5 anti-bot challenges. In accordance with system instructions and project ethics, automated CAPTCHA bypassing will not be attempted; sanctioned open APIs and academic archives will be used instead.

---

## 2. Source-by-Source Technical Findings

### Source 1: data.gov.in — Open Government Data Platform India
* **Primary Resource Tested:** `9ef84268-d588-465a-a308-a864a43d0070` ("Current Daily Price of Various Commodities from Various Markets (Mandi)").
* **Base Endpoint:** `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070`
* **Protocol & Format:** REST API over HTTPS; responses in JSON and XML.
* **Authentication Method:** Query parameter `api-key=<KEY>`. Unauthenticated requests return HTTP 400: `{"error": "Authorization field missing"}`.
* **Query Parameters:**
  * `api-key`: User API key.
  * `format`: `json` or `xml`.
  * `offset`: Pagination offset (integer).
  * `limit`: Page size (default 10, max 1000).
  * `filters[state]`: State name (e.g., `Maharashtra`).
  * `filters[district]`: District name (e.g., `Pune`, `Mumbai`, `Nashik`, `Nagpur`).
  * `filters[commodity]`: Commodity name (e.g., `Onion`, `Tomato`, `Potato`, `Rice`, `Wheat`).
* **Returned Fields:** `state`, `district`, `market`, `commodity`, `variety`, `arrival_date`, `min_price`, `max_price`, `modal_price`.
* **Feasibility & Assessment:** Verified active and functional. Because the portal dynamically rolls recent daily observations, the API client is designed with automatic checkpointing and incremental storage so daily collections build a permanent historical database.

### Source 2: AGMARKNET (agmarknet.gov.in)
* **Website Architecture:** React Single Page Application (Agmarknet 2.0).
* **Robots.txt Analysis:** `https://agmarknet.gov.in/robots.txt` returned HTTP 200 with `User-agent: * Allow: /`.
* **Discovered Internal API:** `https://api.agmarknet.gov.in/v1/`
  * Metadata endpoints:
    * `GET /v1/commodities`: Returns 609 commodities with IDs (Open, HTTP 200).
    * `GET /v1/location/state`: Returns 37 States/UTs with IDs (Open, HTTP 200). Maharashtra ID = `20`.
    * `GET /v1/daily-price-arrival/filters`: Returns 4,176 markets, 2,154 varieties, 751 districts (Open, HTTP 200).
  * Data query endpoint:
    * `POST /v1/daily-price-arrival/report`: Tested with Maharashtra/Pune/Onion payload. Returned HTTP 400: `{"detail": "Captcha key and captcha value are required.", "code": "TOKEN_OR_CAPTCHA_REQUIRED"}`.
* **Feasibility & Assessment:** While metadata can be extracted programmatically, querying daily transactional reports directly from `api.agmarknet.gov.in` is gated behind CAPTCHA validation. Per strict project guidelines, we will not bypass CAPTCHAs. Instead, live daily mandi data is fetched via the authorized OGD `data.gov.in` API, and multi-year historical data is obtained via curated Agmarknet open data archives.

### Source 3: CEDA — Centre for Economic Data and Analysis (Ashoka University)
* **Website Architecture:** Next.js application with FastAPI/Swagger backend at `api.ceda.ashoka.edu.in`.
* **Discovered OpenAPI Specification:** Parsed from `https://api.ceda.ashoka.edu.in/documentation/swagger-ui-init.js`.
* **Base URL:** `https://api.ceda.ashoka.edu.in/v1`
* **Catalog Scope:** As stated in the OpenAPI spec: *"API documentation for the datasets provided by CEDA. Currently serving only Agmarknet data."* (The DCA Food Prices portal does not have a public API; `dca.ceda.ashoka.edu.in` domain was retired).
* **Endpoints:**
  * `GET /agmarknet/commodities`
  * `GET /agmarknet/geographies`
  * `POST /agmarknet/markets`
  * `POST /agmarknet/prices`
  * `POST /agmarknet/quantities`
* **Authentication Method:** HTTP Bearer JWT token (`Authorization: Bearer <token>`). Unauthenticated calls return HTTP 401: `{"status":"failure","message":"Unauthorised, no api key passed.","isAuthenticated":false}`.
* **Feasibility & Assessment:** The CEDA client is fully designed according to the OpenAPI specification. It is ready for optional token configuration if the user registers on the CEDA data portal.

### Source 4: Department of Consumer Affairs (DCA) Price Monitoring Division
* **Portal:** `https://fcainfoweb.nic.in/`
* **Daily Average Retail Price Bulletin:**
  * Endpoint: `https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx`
  * Status: HTTP 200, public HTML table, NO CAPTCHA.
  * Extracted Content: Table 1 contains current daily retail average prices for all 22 essential commodities, along with comparative benchmark prices:
    * 1 Week back
    * 1 Month back
    * 6 Months back
    * 1 Year back
    * 2 Years back
* **Historical Multi-Centre Web Interface:**
  * Endpoint: `https://fcainfoweb.nic.in/reports/report_menu_web.aspx`
  * Form Parameters: ASP.NET `__VIEWSTATE`, `__EVENTVALIDATION`, report type dropdown, radio buttons, and `ctl00$MainContent$Captcha`.
  * Limitation: Gated by an interactive CAPTCHA.
* **Feasibility & Assessment:**
  1. The live daily average bulletin (`DBprices.aspx`) is scraped permissibly via `requests` and `BeautifulSoup` without CAPTCHA.
  2. Multi-year daily centre-level retail prices (including Mumbai, Pune, Nagpur, Nashik) from 2010 to 2021 are acquired from open academic datasets published by IIT Delhi / DataMeet.

### Source 5: Ministry of Statistics and Programme Implementation (MoSPI)
* **Official API Portal:** `https://api.mospi.gov.in/`
* **OpenAPI Specifications Retrieved:**
  * `layout_swagger_user_cpi.yaml` (Consumer Price Index)
  * `layout_swagger_user_hces.yaml` (Household Consumption Expenditure Survey)
  * `layout_swagger_user_wpi.yaml` (Wholesale Price Index)
* **Empirical Discovery — Open Live API:**
  While the documentation mentioned Bearer tokens, empirical requests revealed that the GET endpoints are currently **fully operational without authentication**:
  * **CPI Index & Inflation Endpoint:**
    * `GET https://api.mospi.gov.in/api/cpi/getCPIIndex`
    * Parameters: `base_year=2012`, `series=Current`, `year=2014..2026`, `month_code=1..12`, `state_code=27` (Maharashtra) or `99` (All India), `group_code=1` (Food & Beverages), `sector_code=1,2,3` (Rural, Urban, Combined), `Format=JSON`.
    * Tested response: HTTP 200 with complete index and year-on-year inflation rate!
  * **Item-Level CPI Endpoint:**
    * `GET https://api.mospi.gov.in/api/cpi/getItemIndex`
    * Parameters: `base_year=2012`, `year=YYYY`, `month_code=1..12`, `item_code=...`, `Format=JSON`.
    * Tested response: HTTP 200 with item-level indices (Rice, Wheat, Potato, Onion, Tomato, Pulses).
  * **Household Expenditure Survey (HCES) Endpoint:**
    * `GET https://api.mospi.gov.in/api/hces/getHcesRecords`
    * Parameters: `indicator_code=...`, `year=2022-23`, `state_code=27`, `Format=JSON`.
    * Tested response: HTTP 200 with percentage shares of consumption expenditure.
* **MoSPI HCES Official Fact Sheet Document:**
  * Downloaded `Factsheet_HCES_2022-23.pdf` (5.9 MB).
  * Contains 20 statistical statements including Table 1 (Food vs Non-Food budget share: 46% Rural, 39% Urban), Table 4 (12 fractile classes of MPCE), Table 5/15 (cereals, pulses, vegetables, milk, oil expenditures), and Table 8/18 (State-wise MPCE for Maharashtra).
* **Feasibility & Assessment:** High reliability and historical coverage. This is the cornerstone dataset for macro inflation forecasting and household food basket affordability analysis.

### Source 6: Additional Sources & Macro Indicators
* **World Bank Open Data API:**
  * Endpoint: `https://api.worldbank.org/v2/country/IND/indicator/FP.CPI.TOTL.ZG?format=json`
  * Status: HTTP 200, JSON format, no credentials required.
  * Verified: Annual consumer price inflation for India from 1960 to 2025.
* **Reserve Bank of India (RBI DBIE):**
  * Direct downloads on `rbidocs.rbi.org.in` returned an F5 TSPD cookie anti-bot gate. Documented as restricted from automated bots; superseded by the direct MoSPI CPI API.

---

## 3. Data Pipeline Implementation Plan

| Component | Target Destination | Method | Sources |
|---|---|---|---|
| **CPI Inflation & Indices** | `data/processed/cpi_data.csv` | REST API Client (`mospi_client.py`) | MoSPI API (`api.mospi.gov.in`) |
| **Household Food Basket & MPCE** | `data/processed/household_expenditure.csv` | REST API & PDF Tabular Extraction (`clean_data.py`) | MoSPI HCES API & NSSO Factsheet |
| **Wholesale Mandi Prices & Arrivals** | `data/processed/wholesale_prices.csv` | API Client (`data_gov_client.py`) & Academic Mandi Archives | data.gov.in & Historical Maharashtra Agmarknet Archive |
| **Retail Commodity Prices** | `data/processed/retail_prices.csv` | Scraper (`dca_scraper.py`) & Historical DCA Centre Archives | DCA DBprices Bulletin & DCA Historical Archive |

All raw responses are persisted in `data/raw/` in unadulterated format (JSON, CSV, PDF, HTML) prior to cleaning and standardization.
