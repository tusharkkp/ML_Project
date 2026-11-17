# Academic Evaluation & Practical Examination Rubric Mapping

**Course:** B.Tech Third-Year — Artificial Intelligence and Machine Learning Laboratory  
**Examination:** Practical Examination & Mini Project Assessment (30 Marks Rubric)  
**Project Title:** AI-Based Food Inflation Prediction and Household Affordability Monitoring System  
**Date:** 2026-11-17  
**Student Roll Number Note:** *The specific student roll-number-to-algorithm allocation table was searched across the repository and not found. The project implements the required core regression algorithms (Linear Regression, Polynomial Regression, Decision Tree Regressor, Random Forest Regressor, and Persistence Baseline) as specified by the AIML theory syllabus. The student can provide the allocation list if a specific additional algorithm is assigned.*

---

## 1. Master Evaluation Rubric Mapping

| Exam Component | Marks | Specific Syllabus & Laboratory Requirement | Implemented Deliverable & Workspace Location | Key Metrics / Techniques Evaluated |
|:---|:---:|:---|:---|:---|
| **A. Data Collection & Description** | **2** | Identification of real-world dataset, domain context, schema definition, attribute types, and volume. | - [reports/dataset_description.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/dataset_description.md)<br>- [reports/dataset_audit_report.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/dataset_audit_report.md)<br>- [data/metadata/data_dictionary.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/data/metadata/data_dictionary.md) | Official AGMARKNET Pune Mandi daily series (880 observations), 14 attributes, real economic supply-demand records. |
| **B. Statistical Summary & Operations** | **2** | Application of at least three descriptive statistical operations; interpretation of food price volatility. | - [reports/statistical_analysis.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/statistical_analysis.md)<br>- [notebooks/02_exploratory_data_analysis.ipynb](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/02_exploratory_data_analysis.ipynb) | 6 operations: Mean (1186.16), Median (800.0), Std Dev (1035.11), Min/Max (300/5700), IQR (700.0), Variance (1,071,459). |
| **C. Data Exploration & Visualization** | **2** | 3–5 meaningful, labeled visualizations with economic and statistical interpretations. | - [reports/eda_findings.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/eda_findings.md)<br>- [reports/figures/](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/figures/)<br>- [notebooks/02_exploratory_data_analysis.ipynb](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/02_exploratory_data_analysis.ipynb) | 5 visualizations: (1) Time-Series Trend, (2) Price Distribution Histogram, (3) Outlier Boxplot, (4) Seasonal Monthly Barplot, (5) Correlation Heatmap. |
| **D. Data Preprocessing & Cleaning** | **2** | Handling missing data, duplicate removal, date conversions, feature engineering, zero-leakage pipeline. | - [reports/preprocessing_explanation.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/preprocessing_explanation.md)<br>- [notebooks/03_data_preprocessing.ipynb](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/03_data_preprocessing.ipynb)<br>- [data/processed/model_ready_pune_tomato.csv](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/data/processed/model_ready_pune_tomato.csv) | Zero fabrication, lag creation ($t-1, t-3, t-7$), rolling mean, rolling std, rolling min/max, supply arrival integration. |
| **E. Data Splitting & Cross-Validation** | **2** | Chronological train/validation/test partitioning; time-series cross-validation; avoiding lookahead bias. | - [reports/preprocessing_explanation.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/preprocessing_explanation.md)<br>- [notebooks/03_data_preprocessing.ipynb](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/03_data_preprocessing.ipynb) | Chronological 70% Train (611 rows), 15% Validation (131 rows), 15% Test (131 rows); 5-fold expanding-window `TimeSeriesSplit`. |
| **F. ML Model Implementation & Tuning** | **4** | Implementation of multiple regression algorithms; systematic hyperparameter tuning. | - [reports/model_development.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/model_development.md)<br>- [notebooks/04_regression_model_development.ipynb](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/04_regression_model_development.ipynb) | 4 models: (1) Naive Persistence Baseline, (2) Multiple Linear Regression, (3) Polynomial Regression (deg 1-3), (4) Random Forest / Decision Tree. |
| **G. Model Evaluation & Comparative Analysis** | **8** | Testing on unseen data; MAE, MSE, RMSE, R²; actual vs predicted plots; residual diagnostics; error interpretation. | - [reports/model_evaluation.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/model_evaluation.md)<br>- [notebooks/05_model_evaluation.ipynb](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/notebooks/05_model_evaluation.ipynb)<br>- Comparative Performance Table | Unseen test set evaluation; actual vs predicted curve; residual histogram; residual vs predicted scatter; overfitting audit. |
| **H. Model Deployment Readiness** | **6** | Reusable inference pipeline, serialized model artifacts, metadata JSON, REST schema, independence test. | - [models/price_prediction_model.joblib](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/models/price_prediction_model.joblib)<br>- [models/model_metadata.json](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/models/model_metadata.json)<br>- [src/inference.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/inference.py)<br>- [src/schemas.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/src/schemas.py)<br>- [scripts/inference_test.py](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/scripts/inference_test.py) | Standalone loading, Pydantic request/response schemas, zero-retraining inference test passing on unseen vector. |
| **I. Presentation & Oral Exam Preparation** | **4** | Technical viva defense, theoretical depth, limitations, practical application to household affordability. | - [reports/viva_questions_and_answers.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/viva_questions_and_answers.md)<br>- [reports/limitations_and_future_work.md](file:///c:/Users/Tushar%20Kaldate/Documents/ML_Project/reports/limitations_and_future_work.md) | 25+ viva questions covering time series leakage, bias-variance tradeoff, metric selection, APMC market economics. |
| **Total** | **30** | **Comprehensive Academic & Practical AIML Rigor** | **All Artifacts Fully Executed & Verified** | **100% Verified** |

---

## 2. Verification Protocol for Practical Examiners

Examiners can reproduce and verify every step of this project by running:

```bash
# 1. Verify environment and installed dependencies
py -3.12 -c "import sklearn, matplotlib, seaborn, pandas, numpy; print('Environment Validated!')"

# 2. Re-run data cleaning and validation
py -3.12 scripts/processing/clean_data.py
py -3.12 scripts/processing/validate_data.py

# 3. Execute model training and evaluation pipeline
py -3.12 scripts/train_and_evaluate.py

# 4. Test standalone deployment inference
py -3.12 scripts/inference_test.py
```
