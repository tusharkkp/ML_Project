# Data Source Inventory & Evaluation

This document inventories, evaluates, and ranks all identified data sources for the academic research project:
**AI-Based Food Inflation Prediction and Household Affordability Monitoring System**.

---

## 1. Source Inventory Matrix

| Rank | Source | Publisher | Official URL | Acquisition Method | Data Type | Historical Coverage | Frequency | Geography | Commodities | Authentication | Permission Status | Reliability | Key Limitations |
|:---:|---|---|---|---|---|---|---|---|---|---|---|:---:|---|
| **1** | **MoSPI Consumer Price Index (CPI) API** | Ministry of Statistics and Programme Implementation (MoSPI), GoI | [api.mospi.gov.in](https://api.mospi.gov.in/) | REST API (GET) | CPI & Food Inflation Indices | 2014 – Present (Base 2012=100 & Base 2010=100) | Monthly | National (99), State-wise (Maharashtra = 27), Sector (Rural, Urban, Combined) | Food & Beverages group, Cereals, Pulses, Vegetables, Oils & Fats, Milk, Sugar, plus 299 item-level series (Rice, Wheat, Potato, Onion, Tomato, Arhar Dal, etc.) | None (Publicly open via Swagger API layer) | Verified (Official GoI Open API) | High | Monthly frequency; provides indices and inflation percentages rather than absolute mandi prices. |
| **2** | **data.gov.in — Agmarknet Mandi Prices API** | Open Government Data (OGD) Platform / Ministry of Agriculture | [data.gov.in](https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi) | REST API (JSON/XML) | Wholesale Mandi Prices & Arrivals | Current daily feed; rolling records | Daily | National, State (Maharashtra), District, Market/APMC | All essential commodities (Rice, Wheat, Onion, Potato, Tomato, Tur Dal, Chana Dal, Mustard Oil, etc.) | API Key (`DATA_GOV_IN_API_KEY`) | Verified (GODL - Government Open Data License India) | High | Dynamically updated rolling daily feed; requires pagination; historical records require periodic archival or archived OGD catalogs. |
| **3** | **MoSPI Household Consumption Expenditure Survey (HCES)** | National Sample Survey Office (NSSO) / MoSPI | [mospi.gov.in](https://www.mospi.gov.in/) & [api.mospi.gov.in](https://api.mospi.gov.in/) | REST API & Factsheet Tables (PDF/Excel extraction) | Household Expenditure & Food Baskets | HCES 2022–23 & 2023–24 | Survey Rounds (Annual / Multi-year) | All-India, Rural/Urban, State-wise (including Maharashtra), 12 Fractile Classes | Broad categories (Food vs Non-Food), Cereals, Pulses, Milk, Edible Oils, Vegetables, Fruits, Sugar | None for factsheet / Open GET API | Verified (Official GoI Report) | High | Cross-sectional survey data; updated per survey cycle rather than daily/monthly. |
| **4** | **DCA Price Monitoring System (PMS) Daily Bulletin** | Department of Consumer Affairs (DCA), Ministry of Consumer Affairs | [fcainfoweb.nic.in](https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx) | Web Scraping (HTML Table Parsing via requests/BeautifulSoup) | Retail Prices (Daily Average & Multi-Year Benchmarks) | Current day + 1-week, 1-month, 6-month, 1-year, and 2-year back benchmarks | Daily | All-India Average | 22 Essential Commodities (Rice, Wheat, Atta, Gram Dal, Tur/Arhar Dal, Urad Dal, Moong Dal, Masoor Dal, Sugar, Milk, Groundnut Oil, Mustard Oil, Vanaspati, Soya Oil, Sunflower Oil, Palm Oil, Gur, Tea, Salt, Potato, Onion, Tomato) | None | Verified (Public Government Price Bulletin) | High | DBprices page gives national averages; centre-wise historical query interface is gated by CAPTCHA. |
| **5** | **Agmarknet & DCA Multi-Year Historical Price Repositories (Academic Archive / Open Data)** | Curated Academic Mirrors (IIT Delhi / DataMeet / GODL Archives) | [github.com/ICTD-IITD/Commodity_Analysis](https://github.com/ICTD-IITD/Commodity_Analysis) | Structured CSV Retrieval & Processing | Wholesale Mandi & Retail Prices | 2010 – 2021 (Multi-Year Daily) | Daily | Market/Mandi level (Maharashtra: Ahmednagar, Pune, Nashik, Mumbai, Nagpur) & Centre-level Retail (Mumbai, Pune, etc.) | Tomato, Onion, Potato, Pulses, Mustard Oil, Sugar, Rice, Wheat | None (Public GitHub/Zenodo open repository) | Verified (Academic Research GODL mirror) | High | Spans up to 2021; requires cleaning git conflict artifacts and forward-filling market names. |
| **6** | **CEDA (Centre for Economic Data and Analysis) API** | Ashoka University | [api.ceda.ashoka.edu.in](https://api.ceda.ashoka.edu.in/documentation/) | REST API (POST/GET) | Wholesale Mandi Prices & Quantities | 2014 – Present | Daily / Monthly | National, State, District, Market | Agmarknet commodities | Account / Bearer JWT Token | Verified (Registration required on CEDA portal) | High | Only serves Agmarknet data; requires manual user registration to generate Bearer token; DCA food prices portal has no public API. |
| **7** | **AGMARKNET 2.0 Web Portal** | Directorate of Marketing & Inspection (DMI), Ministry of Agriculture | [agmarknet.gov.in](https://agmarknet.gov.in/) | REST API / Web Scraping | Wholesale Mandi Prices, Varieties, Grades & Arrivals | 2021 – Present | Daily | National, State (Maharashtra), District (Pune, Mumbai, Nashik, Nagpur), Market | 609 commodities, 2154 varieties, 18 grades | CAPTCHA / JWT required for report queries | Scraping permitted by robots.txt (`Allow: /`), but report query requires CAPTCHA | Medium | Query API returns `TOKEN_OR_CAPTCHA_REQUIRED`; ethical scraping prohibits automated CAPTCHA bypassing. |
| **8** | **DCA Historical Report Interface (fcainfoweb)** | Department of Consumer Affairs | [fcainfoweb.nic.in/reports/report_menu_web.aspx](https://fcainfoweb.nic.in/reports/report_menu_web.aspx) | Form Submission / Scraping | Daily Retail & Wholesale Prices | 2014 – Present | Daily | 500+ Reporting Centres, State-level | 22 Essential Commodities | CAPTCHA on Form | Not permitted for automated scraping due to interactive CAPTCHA | Medium | Contains interactive CAPTCHA; cannot be automated programmatically without human intervention. |
| **9** | **World Bank Global Economic & Inflation Indicators** | World Bank Open Data | [api.worldbank.org](https://api.worldbank.org/v2/country/IND/indicator/FP.CPI.TOTL.ZG) | REST API (JSON) | Headline & Food Inflation | 1960 – 2025 | Annual | National (India) | General Consumer Price Index, Macroeconomic aggregates | None | Verified (World Bank Open Data License) | High | Annual frequency; useful as macroeconomic control feature, not daily commodity series. |
| **10** | **RBI Database on Indian Economy (DBIE)** | Reserve Bank of India | [dbie.rbi.org.in](https://dbie.rbi.org.in) | Excel/CSV Downloads | CPI Combined (Food & Beverages), WPI Food Articles | 2014 – Present | Monthly | National & State-wise | Food & Beverages, Cereals, Pulses, Veg, Oils | Automated Bot Protection (F5 TSPD Challenge) | Not accessible via automated bot | High | Downloads are gated behind F5 BIG-IP anti-bot challenge cookies; requires manual browser download. |

---

## 2. Source Prioritization & Strategy

Following project priorities:
1. **CPI & Inflation Forecasting**: MoSPI CPI API (`api.mospi.gov.in`) provides 10+ years of uninterrupted monthly food inflation and index series across All-India and Maharashtra, both at group/subgroup level and item level.
2. **Wholesale Mandi Prices & Arrivals**:
   - Live/current: `data.gov.in` API (Resource `9ef84268-d588-465a-a308-a864a43d0070`).
   - Multi-year historical (2014-2021): Curated Agmarknet archives for Maharashtra mandis (Pune, Nashik, Nagpur, Mumbai).
3. **Retail Prices**:
   - Live/current: DCA DBprices bulletin (`fcainfoweb.nic.in/Reports/DB/DBprices.aspx`).
   - Multi-year historical (2010-2021): DCA RetailFCA daily centre-wise series for Maharashtra reporting centres.
4. **Household Food Basket & Affordability**: MoSPI HCES 2022-23 API and Factsheet Tables (MPCE, fractile expenditure classes, and budget shares for food subgroups).

---

## 3. Credential & Action Requirements

| Service | Credential Name | Environment Variable | Status | Action Needed |
|---|---|---|---|---|
| **data.gov.in** | Open Government Data API Key | `DATA_GOV_IN_API_KEY` | Available with User | User to set in `.env` file |
| **MoSPI CPI / HCES API** | None required | None (Public GET API) | **Active & Live** | Reusable client implemented |
| **CEDA Ashoka API** | Bearer JWT Token | `CEDA_API_TOKEN` | Optional | Register at `api.ceda.ashoka.edu.in` if additional Agmarknet querying desired |
| **AGMARKNET 2.0** | None (Free metadata) / Token for reports | `AGMARKNET_AUTH_TOKEN` | Optional | Metadata public; report querying requires CAPTCHA |
