# Project Limitations & Future Engineering Roadmap

**Project:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Course:** B.Tech AIML — Academic Project & Practical Examination  
**Date:** 2026-11-17  
**Author:** Senior Machine Learning Engineer & Academic Project Mentor  

---

## 1. Academic Transparency & Known Limitations

In scientific research and engineering practice, transparently identifying model limitations is as essential as reporting high accuracy scores. The current implementation operates within several real-world boundaries:

### 1. Single-Market Geographic Focus
- **Current Scope:** The primary predictive model was trained on the **Pune APMC Mandi** daily trading series (873 model-ready observations).
- **Limitation:** While Pune is a premier Maharashtra agricultural hub, wholesale prices in Pune are interconnected with neighboring production centers (Lasalgaon, Pimpalgaon, Sangamner, Nashik, and Solapur). A localized crop failure in Nashik directly impacts Pune arrivals 24–48 hours later.
- **Impact:** The current model treats Pune arrivals as purely exogenous without modeling spatial cross-market transmission spillover.

### 2. Multi-Step Direct vs. Recursive Forecasting Horizon
- **Current Scope:** The primary target is the **next active trading session ($t+1$)**.
- **Limitation:** For practical agricultural planning, a 14-day or 30-day forecast is highly desirable. However, in recursive autoregressive forecasting:
  $$\hat{y}_{t+2} = f(\hat{y}_{t+1}, y_t, \dots)$$
  Any estimation error in $\hat{y}_{t+1}$ is fed back into the feature vector for $t+2$, leading to **compounding error accumulation** over longer horizons.

### 3. Non-Trading Calendar Gaps (Sundays & Market Holidays)
- **Current Scope:** Time-series observations are indexed by **active trading sessions** (Monday through Saturday).
- **Limitation:** Because mandis close on Sundays, the calendar gap between Saturday and Monday is 48 hours, whereas between Tuesday and Wednesday it is 24 hours. Monday sessions frequently absorb two days of farm harvesting, leading to localized "Monday volume spikes" that simple day-of-week indicators only partially capture.

### 4. Unobserved Exogenous Shocks
- **Current Scope:** Features incorporate historical price momentum, intra-day bid spreads, and physical arrival tonnage ($Q_t$).
- **Limitation:** Commodity prices are influenced by broader macroeconomic and environmental factors not present in the mandi ledger:
  - **Diesel / Fuel Prices:** Diesel price hikes directly increase truck freight rates between farmgate and city mandis.
  - **Severe Weather Events:** Extreme unseasonal hailstorms or flash floods destroy standing crops overnight before any arrival volume changes are registered at the mandi gate.
  - **Export-Import Policies:** Sudden government export bans (e.g. minimum export price on onions) cause immediate, non-continuous step-function price drops that historical autoregression cannot anticipate.

### 5. Retail-to-Wholesale Transmission Markups
- **Current Scope:** The machine learning model forecasts the **wholesale APMC price** (`Rs/Quintal`), while household grocery expenses occur at the **consumer retail level** (`Rs/Kg`).
- **Limitation:** The transmission from wholesale auction to neighborhood vegetable vendors involves a 24–72 hour distribution lag, mandi market cess (1%–2%), handling loss (10%–15% perishability wastage for tomatoes), transport freight, and retailer markups (typically 30%–60%). A static linear conversion underestimates retail volatility in peripheral suburban markets.

---

## 2. Future Engineering & Research Roadmap

To advance the prototype into an enterprise-grade food intelligence system, the following enhancements are planned for Phase 3 and Phase 4:

```
  +-----------------------------------------------------------------------------------+
  |                             FUTURE ENGINEERING ROADMAP                            |
  |                                                                                   |
  |   1. DEEP LEARNING (Temporal Fusion Transformers, Informer, Multi-Horizon LSTM)   |
  |   2. SATELLITE REMOTE SENSING (Google Earth Engine NDVI for Yield Forecasting)    |
  |   3. SPATIAL GRAPH NEURAL NETWORKS (Inter-Mandi Supply Chain Spillover Models)   |
  |   4. MACRO-ECONOMIC FEATURE ENRICHMENT (Diesel Indices, RBI Repo, WPI Food)       |
  |   5. FULL-STACK FASTAPI & NEXT.JS HOUSEHOLD AFFORDABILITY PORTAL                  |
  +-----------------------------------------------------------------------------------+
```

### A. Deep Learning & Multi-Horizon Attention Models
- Implement **Temporal Fusion Transformers (TFT)** or **Informer** architectures. Unlike traditional RNNs or LSTMs, Transformers utilize multi-head self-attention mechanisms to learn long-range temporal dependencies and handle multi-horizon forecasting (e.g. predicting $t+1$ through $t+30$ simultaneously) with built-in prediction intervals (10th, 50th, 90th percentiles).

### B. Satellite Remote Sensing & Weather Integration
- Ingest real-time satellite imagery via **Google Earth Engine (GEE)**:
  - **Normalized Difference Vegetation Index (NDVI):** Quantifies vegetative health in major tomato-growing districts (Nashik, Ahmednagar, Pune) 30 days prior to harvest.
  - **High-Resolution Precipitation:** IMD gridded rainfall data to detect localized waterlogging shocks before produce leaves the farmgate.

### C. Spatial-Temporal Graph Neural Networks (ST-GNN)
- Construct an agricultural supply-chain graph where each node is an APMC mandi (e.g. Lasalgaon, Pune, Mumbai, Nagpur) and edges represent inter-market truck transport volume. A Spatial-Temporal GNN can forecast how a price shock in Lasalgaon propagates across Maharashtra mandis over time.

### D. Full-Stack Web Application Integration
- Package the FastAPI service into a Docker container and link it with an interactive **Next.js / Tailwind CSS dashboard**:
  - **Interactive Food Basket Builder:** Users drag-and-drop staple groceries and input their monthly household earnings.
  - **Real-Time Inflation Alerting:** Automated email/SMS notifications when predicted commodity price increases threaten to push the household's food budget share above 50%.
