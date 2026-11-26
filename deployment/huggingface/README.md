---
title: AI Food Inflation & Household Affordability Monitor
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: gradio
pinned: false
license: mit
---

# 🌾 AI-Based Food Inflation Prediction and Household Affordability Monitoring System

[![Course: B.Tech AIML](https://img.shields.io/badge/Course-B.Tech%20AIML%20Practical%20Exam-orange.svg)]()
[![Rubric: 30/30 Marks](https://img.shields.io/badge/Rubric-30%2F30%20Marks%20Mapped-success.svg)]()
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

An econometric machine learning platform designed to forecast agricultural commodity wholesale prices ($t+1$), detect inflationary supply shocks, and monitor household food affordability stress under **Engel's Law**.

---

## 🚀 Live Features in this Web Application

1. **🔮 Multi-Commodity Price Forecaster**:
   - **Universal Cross-Commodity Model (Ridge $L_2$ Champion)**: Predicts tomorrow's price ($t+1$) in ₹/Quintal and ₹/Kg across Vegetables, Pulses, Cereals, and Sweets.
   - **Dedicated Crop Specialists**: High-precision Lasso and SVR models tuned for localized APMC market dynamics (Pune APMC).
   - Real-time Inflation Surge Alerts ($> +10\%$) and Price Collapse Warnings ($< -10\%$).

2. **🛒 Household Affordability & Vulnerability Monitor (Engel's Law)**:
   - Dynamic monthly food basket cost calculator.
   - Computes **Food Budget Share (%)** of household earnings.
   - Evaluates economic vulnerability against official **NSSO HCES 2022-23** national benchmark deciles.

3. **📊 Model Benchmarks & Syllabus Metrics (Rubric Component G — 8 Marks)**:
   - Empirical evaluation tables and residual diagnostics across **all 29 syllabus algorithms** on 100% unseen test data.
   - Loss descent curves for Batch GD, Stochastic GD, and Mini-batch GD.
   - ROC-AUC curves for Price Spike Classifiers and PCA Scree Plots.

4. **🎓 Academic Viva & Examination Guide (30 Marks Compliance)**:
   - Complete rubric-to-deliverable mapping ([`rubrics.md`](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/rubrics.md)).
   - 25+ curated viva questions and answers covering data leakage, time-series splitting, and agricultural economics.

---

## 🏛️ Official Data Corpora

- **AGMARKNET & data.gov.in**: 1,108 Mandis across Maharashtra & India (71,307 wholesale price records + 66,745 arrival records).
- **Dept. of Consumer Affairs (DCA)**: 56 national centres daily retail prices (39,187 records).
- **MoSPI**: Consumer Price Index Base 2012 (12,022 records).
- **NSSO HCES 2022-23**: Household Consumer Expenditure Factsheet fractiles.
