# Architecture and ML Pipeline

This document details the architectural decisions and the underlying Machine Learning pipeline that powers the Breast Cancer Hybrid Prediction System (BC-HPS).

## 1. System Architecture

The BC-HPS application follows a decoupled client-server architecture. This separation of concerns ensures that the computationally heavy machine learning inference is completely isolated from the user interface rendering.

- **Frontend (Streamlit)**: Serves as the interactive dashboard. It handles user inputs (forms, CSV uploads), form validation, and the rendering of interactive Plotly charts and PDF generation logic.
- **Backend (FastAPI)**: Serves as the high-performance inference engine. It loads the `joblib` model pipelines into memory upon startup, handles data validation via Pydantic schemas, runs predictions, and calculates local SHAP explanations using `shap.KernelExplainer`.

Communication between the two layers happens entirely via REST API endpoints (`/health`, `/model-info`, `/predict`, `/predict_batch`).

## 2. Machine Learning Pipeline

The system employs a unique **Dual-Model** approach (also referred to as Hybrid Prediction) to handle patients at different stages of their clinical journey.

### 2.1 Pre-Diagnostic Model
Designed for patients who have only undergone preliminary screening and clinical questionnaires.
- **Features Used**: Age, Gender, BMI, Family History, Smoking, Alcohol Consumption, Physical Activity, Hormone Therapy, Menopause Status, Genetic Mutation, Blood Pressure, Cholesterol, Diabetes, Exercise Days Per Week, Breastfeeding History, Annual Income.
- **Model Purpose**: Initial risk stratification to determine if the patient requires immediate diagnostic testing (like a mammogram) or standard routine screening.

### 2.2 Diagnostic Assessment Model
Designed for patients who have undergone medical testing.
- **Features Used**: All Pre-Diagnostic features + **Mammogram Result**, **Tumor Size (cm)**, and **Lymph Node Involvement**.
- **Model Purpose**: High-confidence classification of malignancy based on definitive clinical indicators.

### 2.3 Data Preprocessing Strategy
As explored in the original Jupyter Notebook (`notebook/breast_cancer_pipeline_consolidated.ipynb`):
1. **Handling Missing Values**: Categorical data is imputed with the most frequent values, and numerical data is imputed using median strategies.
2. **Encoding**: One-Hot Encoding is used for non-ordinal categorical variables, while Label Encoding is applied to strictly ordinal variables (e.g., Cancer Stage).
3. **Scaling**: `StandardScaler` is applied to numerical features (like BMI, Age, and Blood Pressure) to ensure the Random Forest algorithm is not biased by varying scales.
4. **Class Imbalance**: Synthetic Minority Over-sampling Technique (SMOTE) was heavily utilized during training. Since breast cancer datasets are highly imbalanced (far more benign cases than malignant), SMOTE was used to synthesize malignant cases, resulting in a robust, recall-optimized model.

### 2.4 Explanations via SHAP
Instead of operating as a "black box," the pipeline incorporates SHAP (SHapley Additive exPlanations). 
- A background dataset is sampled during backend startup.
- During inference, `KernelExplainer` simulates the impact of each provided feature against the background data.
- The results are visualized as a horizontal bar chart on the frontend, empowering clinicians to understand *why* a particular probability was assigned.
