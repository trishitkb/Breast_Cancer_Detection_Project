# 🎗️ Breast Cancer Detection & Clinical Decision Support System

An end-to-end clinical machine learning pipeline and interactive decision-support application designed for **dual-scenario breast cancer risk assessment and diagnostic classification**.

---

## 📌 Overview

This project implements a **Dual-Scenario ML Architecture** that bridges early screening with post-test clinical diagnosis:

1. **Pre-Diagnostic Risk Screening**: Uses solely demographic and lifestyle risk factors (excluding post-facto clinical tests like mammograms or tumor measurements) to assess risk prior to clinical imaging, serving as an early warning screening tool.
2. **Diagnostic-Assessment Classification**: Integrates diagnostic findings (`Mammogram_Result`, `Lymph_Node_Involvement`, `Tumor_Size_cm`) for high-precision malignancy classification while preventing target leakage.

The system features a **FastAPI** backend that dynamically routes requests to the appropriate model engine, paired with a **Streamlit** dashboard delivering interactive risk prediction, real-time SHAP explainability, threshold tuning analytics, and downloadable clinical PDF reports.

---

## 🏗️ Architecture & Project Structure

```
Breast_Cancer_Detection_Project/
├── backend/
│   ├── main.py                     # FastAPI server with dynamic model routing & SHAP explainers
│   ├── config.py                   # Configuration & path management
│   ├── schemas.py                  # Pydantic data schemas
│   └── retrain.py                  # Retraining script
├── frontend/
│   └── app.py                      # Streamlit interactive clinical dashboard
├── notebook/
│   └── breast_cancer_pipeline_consolidated.ipynb # End-to-end reproducible research notebook
├── models/
│   ├── pre_diagnostic_pipeline.pkl          # Serialized Pre-Diagnostic model pipeline
│   ├── diagnostic_assessment_pipeline.pkl   # Serialized Diagnostic-Assessment model pipeline
│   └── model_metadata.json                  # Model thresholds, metrics, parameters & features
├── data/
│   └── raw/
│       ├── breast_cancer_prediction.csv     # Raw patient dataset (10,000 records)
│       └── data_dictionary.csv              # Attribute definitions and types
├── reports/
│   ├── figures/                             # Generated ROC, PR, Calibration, Confusion & SHAP plots
│   ├── final_scenario_summary.csv           # Test evaluation metrics across scenarios
│   ├── bootstrap_confidence_intervals.csv   # 95% Bootstrap Confidence Intervals
│   ├── calibration_summary.csv              # Brier score & calibration assessments
│   └── illustrative_cost_analysis.csv       # Clinical cost-sensitive tradeoff matrix
├── requirements.txt                         # Project dependencies
└── README.md                                # Project documentation
```

---

## 🔬 Machine Learning Pipeline & Methodology

### 1. Data Integrity & Leakage Prevention
- **Strict Partitioning**: Stratified **70% Train / 15% Validation / 15% Held-Out Test** split.
- **Categorical SMOTENC**: Applied strictly to the training fold to address class imbalance (~80:20 Benign to Malignant) without creating synthetic floating-point category artifacts.
- **Target Leakage Auditing**: Excludes direct target proxies (`Biopsy_Result`, `Cancer_Stage`) from all predictive feature spaces.

### 2. Model Space & Hyperparameter Tuning
- Evaluated models: **XGBoost (`XGBClassifier`)**, **Random Forest**, and **Logistic Regression**.
- Tuned via `RandomizedSearchCV` with 3-fold Stratified Cross-Validation on the training fold.

### 3. Validation-Based Threshold Optimization
- Operating thresholds are tuned on the **Validation Set** to optimize clinical sensitivity using the **$F_2$ metric** (prioritizing recall to minimize false negatives while maintaining reasonable precision).
- Final metrics are evaluated on the unbiased **Held-Out Test Set**.

### 4. Advanced Clinical Validation
- **Calibration Analysis**: Brier score calculation and calibration curves to ensure predicted probabilities reflect true empirical risk.
- **Bootstrap Confidence Intervals**: 1,000 iterations to compute 95% empirical confidence intervals for all core metrics.
- **Subgroup Fairness**: Evaluated across age brackets, gender, and family history.
- **Explainability**: Global and localized SHAP (SHapley Additive exPlanations) values to explain patient-specific risk drivers.

---

## 📊 Benchmark Results (Held-Out Test Set)

| Scenario | Model Architecture | Operating Threshold | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test Recall / Sensitivity (95% CI) | Test Precision (95% CI) | Test $F_1$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pre-Diagnostic Risk** | Random Forest (Tuned) | `0.230` | **0.864** (0.842 – 0.886) | **0.651** (0.593 – 0.703) | **91.72%** (88.44% – 94.80%) | 36.64% (33.28% – 39.87%) | 0.524 |
| **Diagnostic Assessment** | Random Forest (Tuned) | `0.431` | **0.983** (0.975 – 0.990) | **0.952** (0.934 – 0.966) | **94.83%** (92.15% – 97.17%) | **75.34%** (70.69% – 79.84%) | **0.840** |

*Note: In Pre-Diagnostic screening, high recall (91.7%) ensures minimal missed cases, with false positives safely triaged via subsequent diagnostic mammography.*

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Environment Setup
Clone the repository and create a Python virtual environment:

```bash
# Clone repository
git clone https://github.com/trishitkb/Breast_Cancer_Detection_Project.git
cd Breast_Cancer_Detection_Project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows (PowerShell/CMD)
# source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the ML Pipeline
To reproduce the full analysis, train models, and generate all reports/figures:
```bash
jupyter nbconvert --to notebook --execute --inplace notebook/breast_cancer_pipeline_consolidated.ipynb
```

### 3. Launch Backend API
Start the FastAPI service:
```bash
.venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Key Endpoints:
  - `GET /health` — Check server and pipeline loading status.
  - `GET /model-info` — Inspect model metadata, thresholds, and parameters.
  - `POST /predict` — Real-time single-patient prediction with dynamic routing and SHAP attribution.
  - `POST /predict_batch` — Batch CSV/list inference.

### 4. Launch Streamlit Dashboard
Start the Streamlit application:
```bash
.venv\Scripts\streamlit run frontend/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Features

- **Dynamic Form Input**: Allows entering standard demographic risk factors with optional clinical diagnostics (Mammogram result, Lymph node involvement, Tumor size).
- **Adaptive Engine Routing**: Automatically activates the Pre-Diagnostic or Diagnostic-Assessment model based on provided test results.
- **Visual Risk Gauge**: Real-time probability gauge with clinical alert thresholds.
- **Explainable AI (XAI)**: Dynamic horizontal bar chart illustrating patient-specific SHAP values (top positive and negative contributors).
- **Batch Processing**: Upload custom patient CSVs (e.g. `test.csv`) for bulk scoring and CSV export.
- **PDF Report Generation**: Instantly downloads a formatted clinical risk assessment summary.
- **Model Performance Tab**: Interactive review of held-out test confusion matrices, ROC/PR curves, calibration plots, and SHAP summary plots.

---

## ⚖️ Clinical Governance & Disclaimer

> [!WARNING]
> This software is intended strictly for research and clinical decision-support demonstration purposes. It does not replace professional medical evaluation, formal diagnostic imaging, or histological biopsy analysis. External prospective validation is required prior to any clinical deployment.
