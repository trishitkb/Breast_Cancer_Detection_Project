# Breast Cancer Prediction Project Plan
## Cleaned & Organized

---

## 📋 Project Overview

**Objective:** Build a clinical decision-support system for breast cancer prediction with focus on minimizing false negatives (missed malignant tumors).

**Key Focus Areas:**
- Exploratory Data Analysis (EDA)
- Data preprocessing & handling class imbalance
- Multiple model training & comparison
- Model explainability (SHAP)
- API deployment with dashboard

**Evaluation Emphasis:** Engineering process > 100% accuracy

---

## 🛠️ Phase 1: Project Setup

### 1.1 Environment & Version Control
- Initialize Git repository
- Create virtual environment
- Set up folder structure
- Configure `requirements.txt`

### 1.2 Required Libraries

**Core:**
- `numpy`, `pandas`
- `matplotlib`, `seaborn`

**Machine Learning:**
- `scikit-learn`
- `imbalanced-learn`
- `xgboost` (optional)
- `shap`
- `joblib`

**Deployment:**
- `FastAPI`, `uvicorn`
- `Streamlit`

**Utilities:**
- `pydantic`
- `python-dotenv`

### 1.3 Engineering Best Practices
- [x] Configuration file setup
- [x] Environment variables configured
- [x] Logging implemented
- [x] Versioning established
- [x] Exception handling
- [x] Modular architecture designed
- [x] Reproducible experiments enabled
- [x] Data validation schema defined
- [x] Feature validation implemented
- [x] Datatype validation
- [x] Range validation

---

## 📊 Phase 2: Data Processing & EDA

### 2.1 Data Understanding
- Load and inspect dataset
- Document rows, columns, feature names
- Identify target variable
- Check for missing values
- Check for duplicate values
- Verify data types

### 2.2 Exploratory Data Analysis (EDA)

**Dataset Overview:**
- Shape and size
- Statistical summaries
- Data types inspection

**Visualizations:**

| # | Figure/Chart | Description | Why It Matters |
|---|--------------|-------------|----------------|
| 1 | Pie/Donut Chart | Overall class distribution (80% Benign / 20% Malignant) | Visually proves severe imbalance problem |
| 2 | Overlapping KDE Plots | Top 4 features with separate curves for Benign (blue) and Malignant (red) | Shows if features actually separate classes |
| 3 | Correlation Heatmap | 15×15 grid showing Pearson correlation | Identifies multicollinearity (e.g., radius vs perimeter) |
| 4 | Boxplot Grid | Side-by-side boxplots faceted by Benign/Malignant | Shows which biomarkers have widest variance |

### 2.3 Data Cleaning & Feature Engineering

**Cleaning Tasks:**
- Remove duplicates
- Handle missing values
- Convert categorical values
- Fix invalid values

**Feature Engineering:**
- Correlation analysis
- Feature importance
- Recursive Feature Elimination (optional)
- Scaling and normalization
- Encoding
- Outlier analysis
- Feature selection

### 2.4 Handling Class Imbalance

**Dataset Versions for Comparison:**

| Dataset Code | Name | Description | Training Size |
|--------------|------|-------------|---------------|
| D1 | Raw Original | 80:20 ratio (Baseline) | ~8,000 |
| D2 | SMOTE Full | Oversampled to 50:50 | ~12,800 |
| D3 | SMOTE Moderate | Minority becomes 50% | ~9,600 |
| D4 | Under-sampled | Downsampled majority to 50:50 | ~3,200 |
| D5 | SMOTE + ENN | Oversampled + noisy borderline samples removed | ~11,000 |
| D6 | Weighted Data | `class_weight='balanced'` | ~8,000 |

**⚠️ Important:** Never apply SMOTE before splitting data. SMOTE only on training set.

### 2.5 Imbalance Visualizations

| # | Figure/Chart | Description | Why It Matters |
|---|--------------|-------------|----------------|
| 5 | Grouped Bar Chart | Dataset sizes with two colored bars per dataset | Visually confirms differences in dataset sizes |
| 6 | Radar/Spider Plot | Average feature values for D1 vs D2 vs D5 | Shows synthetic data looks like real data |

---

## 🤖 Phase 3: Model Training

### 3.1 Setup
- Set `random_state = 42` for reproducibility
- Use 5-Fold Cross Validation

### 3.2 Models to Train
1. **Baseline:** Logistic Regression
2. **Ensemble:** Random Forest
3. **Gradient Boosting:** XGBoost

### 3.3 Training Approach
- Implement sklearn Pipeline
- Include preprocessing inside pipeline
- Ensure inference identical to training
- Save complete pipeline

### 3.4 Hyperparameter Tuning
- GridSearchCV
- RandomSearchCV

### 3.5 Evaluation Metrics
- Accuracy
- Precision
- Recall (Sensitivity) ← **Primary focus**
- Specificity
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall Curve

---

## 📈 Phase 4: Model Evaluation & Comparison

### 4.1 Performance Comparison

**Focus:** Recall for Malignant class

**Visualizations:**

| # | Figure/Chart | Description | Why It Matters |
|---|--------------|-------------|----------------|
| 7 | Grouped Bar Chart (Recall) | X: D1-D6, Y: Recall (0-1), 3 bars/dataset (LR, RF, XGB), red line at 0.90 | The Winning Chart - shows which combo hits 90% recall |
| 8 | Heatmap Matrix (Recall) | 3-row × 6-column grid color-coded red to green | Allows spotting "sweet spot" quadrant immediately |
| 9 | Parallel Coordinates Plot | Connect metrics (Precision, Recall, F1, ROC-AUC) for Top 5 models | Shows trade-offs between models |
| 10 | ROC Curves (Overlay) | TPR vs FPR for Top 3 champion models | Shows discriminative power |
| 11 | Precision-Recall Curves | Precision vs Recall for Top 3 models | More honest for imbalanced data |

---

## ⚙️ Phase 5: Threshold Optimization

### 5.1 Goal
Minimize False Negatives (missed malignant tumors), even if False Positives increase slightly.

### 5.2 Visualizations

| # | Figure/Chart | Description | Why It Matters |
|---|--------------|-------------|----------------|
| 12 | Threshold Line Chart | X: Threshold (0.1-0.9), Y: Recall & Precision | Shows optimal threshold (e.g., 0.30) to maximize recall |
| 13 | Confusion Matrix Quadrant | 4 matrices: Worst, Best @0.5, Best @0.3, Best @0.7 | Visually proves reduced False Negatives |

---

## 🔍 Phase 6: Model Explainability (SHAP)

### 6.1 Goal
Satisfy the "Explainability" objective - build clinical trust.

### 6.2 Visualizations

| # | Figure/Chart | Description | Why It Matters |
|---|--------------|-------------|----------------|
| 14 | Global Feature Importance (Bar Chart) | Top 10 features ranked by SHAP value magnitude | Tells clinicians top 3 biomarkers to watch |
| 15 | SHAP Summary Beeswarm Plot | Features on Y-axis, SHAP values on X-axis | Shows direction of influence |
| 16 | SHAP Force Plot (Two Cases) | One Benign, one Malignant patient prediction | Proves model isn't a black box |
| 17 | Dependence Plots | Feature value vs SHAP value for top feature | Shows interaction effects |

### 6.3 Clinical Interpretation
Don't just say "Radius Mean" - say:
> "Higher Radius Mean generally increases predicted malignancy probability."

### 6.4 SHAP Components
- [x] Global SHAP (overall feature importance)
- [x] Local SHAP (explain individual predictions)
- [x] Compare SHAP with Feature Importance
- [x] Explain why they differ

---

## 💼 Phase 7: Business Impact Analysis

### 7.1 Objective
Translate metrics into tangible healthcare stories.

### 7.2 Visualizations

| # | Figure/Chart | Description | Why It Matters |
|---|--------------|-------------|----------------|
| 18 | "Catch Rate" Waterfall Chart | 100 actual Malignant patients: Raw Model (75) vs Champion Model (96) | Translates 0.96 Recall into real impact |
| 19 | Gain/Lift Curve | Cumulative gains vs random baseline | Proves model is 2x-3x better than random |

### 7.3 Interpretation
- Explain why prediction occurred
- Which features mattered
- Confidence level

---

## 🚀 Phase 8: Model Saving

### 8.1 Save Objects
- [x] Full pipeline
- [x] Scaler
- [x] Encoder
- [x] Classifier (best model)
- [x] Training metadata

### 8.2 Save Metadata
- [x] Model version
- [x] Training date
- [x] Parameters used
- [x] Dataset version
- [x] Threshold selected

### 8.3 Experiment Tracking
Record for each experiment:
- Model used
- Hyperparameters
- Metrics achieved
- Timestamp
- Dataset version
- Threshold value
- Preprocessing steps

**Options:**
- MLflow (recommended)
- Weights & Biases
- Simple CSV/JSON log

---

## 🔧 Phase 9: Backend (FastAPI)

### 9.1 Endpoints
- `GET /` - Health check
- `GET /health` - Service status
- `GET /model-info` - Model information
- `POST /predict` - Make prediction

### 9.2 Prediction Input
```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "area_mean": 1001.0
  // ... all required features
}
```

### 9.3 Prediction Output
```json
{
  "prediction": "Malignant",
  "probability": 0.92,
  "confidence": 92.0,
  "risk_level": "High",
  "threshold": 0.30,
  "top_features": [
    {"feature": "concave_points_mean", "importance": 0.35},
    {"feature": "area_mean", "importance": 0.28}
  ],
  "prediction_time": "2026-08-22T10:30:00Z"
}
```

### 9.4 Additional Features
- [x] Pydantic input validation
- [x] Error handling
- [x] Logging and monitoring
- [x] Store prediction history
- [x] Configurable prediction threshold via .env
- [x] Batch CSV prediction upload

---

## 📱 Phase 10: Frontend (Streamlit Dashboard)

### 10.1 Pages
1. **Home** - Project overview
2. **Dataset Overview** - Data summary
3. **EDA** - Visualizations
4. **Model Comparison** - Performance metrics
5. **Prediction** - Interactive prediction
6. **Explainability** - SHAP explanations
7. **About** - Project information

### 10.2 Prediction Page Features
- [x] Prediction result
- [x] Confidence score
- [x] Probability gauge
- [x] Risk level indicator
- [x] Top contributing features

### 10.3 Additional Dashboard Features
- [x] Probability gauge (e.g., 92% Malignant)
- [x] Risk meter visualization
- [x] Downloadable prediction report (PDF)
- [x] "Why this prediction?" section using SHAP
- [x] Input validation for impossible values
- [x] Model version information
- [x] Basic logging for API requests
- [x] Batch CSV upload capability

---

## 📄 Phase 11: Report Generation

### 11.1 PDF Report Includes
- Prediction result
- Probability/confidence score
- Top contributing features
- SHAP explanation
- Timestamp
- Model version

---

## 🧪 Phase 12: Testing

### 12.1 Test Categories
- [x] Data loading tests
- [x] Preprocessing tests
- [x] Prediction pipeline tests
- [x] API endpoint tests
- [x] Dashboard component tests

---

## 🌐 Phase 13: Deployment

### 13.1 Deployment Targets
- **Backend:** FastAPI on Render, Railway, or Azure
- **Dashboard:** Streamlit Community Cloud

### 13.2 Repository Setup
- [x] GitHub repository
- [x] Releases configured
- [x] Documentation hosted

---

## 📚 Phase 14: Documentation

### 14.1 Required Documentation
- [x] README.md
- [x] Architecture diagram
- [x] ML pipeline diagram
- [x] API documentation
- [x] Installation guide
- [x] Usage guide
- [x] Results section
- [x] Limitations
- [x] Future work

---

## 📋 Project Timeline Summary

| Phase | Title | Key Activities |
|-------|-------|----------------|
| 1 | Project Setup | Environment, structure, dependencies |
| 2 | Data Processing & EDA | Cleaning, exploration, SMOTE |
| 3 | Model Training | Train, evaluate, tune, save |
| 4 | Model Comparison | Performance visualization |
| 5 | Threshold Optimization | Clinical threshold tuning |
| 6 | Explainability | SHAP implementation |
| 7 | Business Impact | Real-world translation |
| 8 | Model Saving | Pipeline, metadata, tracking |
| 9 | Backend | FastAPI implementation |
| 10 | Frontend | Streamlit dashboard |
| 11 | Report Generation | PDF reports |
| 12 | Testing | Validation suite |
| 13 | Deployment | Production hosting |
| 14 | Documentation | Complete project docs |

---

## 🎯 Key Success Criteria

1. ✅ EDA performed comprehensively
2. ✅ Multiple imbalance handling techniques compared
3. ✅ 18+ model-dataset combinations evaluated
4. ✅ Recall prioritized over accuracy
5. ✅ SHAP explainability implemented
6. ✅ FastAPI backend deployed
7. ✅ Streamlit dashboard functional
8. ✅ Clinical interpretation provided
9. ✅ Business impact demonstrated
10. ✅ Complete documentation available

---

## 📌 Notes for Presentation

- **Focus on Recall** - The clinical priority is catching malignant cases
- **Show the Journey** - Demonstrate how you improved from baseline
- **Explain the "Why"** - Clinical interpretation matters
- **Visual Storytelling** - Use all 19 visualizations effectively
- **Real-World Impact** - Translate metrics to lives saved

---

*Document Version: 1.0*  
*Prepared for: Breast Cancer Prediction Project*  
*Last Updated: August 22, 2026*