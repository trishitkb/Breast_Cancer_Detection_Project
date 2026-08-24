# Breast Cancer Prediction - Presentation Outline & Speaker Notes

This document provides a structured outline for presenting the BC-HPS project, including specific speaker notes and strategies for handling audience Q&A.

## Slide 1: Problem Statement
- **Visuals**: A clean title slide with the project name.
- **Talking Points**: 
  - Breast cancer diagnosis can be highly subjective and slow when reliant purely on human analysis of sprawling tabular data.
  - Inconsistent thresholds often lead to varying diagnoses between clinics.
  - **Goal**: We built an automated, data-driven, and explainable decision-support tool to minimize missed detections (focusing on Recall).

## Slide 2: Dataset Overview & Challenges
- **Visuals**: A pie chart showing class imbalance and a list of key feature categories.
- **Talking Points**:
  - We utilized a 10,000-patient dataset covering demographics, lifestyle, and clinical markers.
  - **Challenge**: The data was highly imbalanced. Far fewer malignant cases exist compared to benign. We couldn't train a model on raw data without it becoming biased toward predicting "Benign".

## Slide 3: Exploratory Data Analysis & Preprocessing
- **Visuals**: A correlation heatmap (from the `reports/figures/` directory).
- **Talking Points**:
  - We applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the malignant and benign classes for robust training.
  - Numerical features were standardized so that factors like Income or Blood Pressure wouldn't overpower smaller measurements like Tumor Size.

## Slide 4: The Dual-Model Architecture (Core Innovation)
- **Visuals**: A flow chart showing a patient splitting into "Pre-Diagnostic" and "Diagnostic Assessment" paths.
- **Talking Points**:
  - **Key Strategy**: Not all patients have imaging data when they walk into a clinic. 
  - We built *two* pipelines. A Pre-Diagnostic model assesses baseline risk using demographics and lifestyle. If a patient gets a mammogram or biopsy, the system dynamically shifts to the Diagnostic Assessment model, utilizing those new clinical markers for extreme accuracy.

## Slide 5: Model Selection & Tech Stack
- **Visuals**: Logos for FastAPI, Streamlit, and Scikit-Learn.
- **Talking Points**:
  - Selected **Random Forest** over Deep Learning because tabular medical data requires robust handling of non-linear features without the "black box" nature of neural networks.
  - We decoupled the architecture: A FastAPI backend acts as a high-performance inference engine, while Streamlit provides an interactive, data-centric dashboard.

## Slide 6: Explainable AI (SHAP)
- **Visuals**: A screenshot of the SHAP horizontal bar chart from the Hybrid Prediction page.
- **Talking Points**:
  - Clinical trust is paramount. A doctor won't trust a binary "High Risk" alert without reasoning.
  - We integrated SHAP (SHapley Additive exPlanations) to provide local interpretability. The system explicitly details how much a patient's specific BMI or Tumor Size influenced the final probability.

## Slide 7: Dashboard & Batch Processing Demo
- **Visuals**: Screenshots of the UI, specifically the Batch Processing ZIP feature.
- **Talking Points**:
  - Show the Hybrid Prediction tool with real-time risk assessment.
  - **Highlight**: The Batch Processing feature. Clinics can upload a CSV of thousands of patients, and the system will automatically parse them, drop invalid rows, and generate a ZIP archive of individual PDF reports sorted into Safe and Critical folders.

---

## 🎤 Q&A Preparation Reminders

**Q: Why optimize for Recall instead of Accuracy?**
**A**: In medical diagnostics, a False Negative (telling a patient they are fine when they have cancer) is far more dangerous than a False Positive (ordering an extra screening for a healthy patient). We tuned the threshold to aggressively catch malignant cases.

**Q: Why not use a Neural Network?**
**A**: Neural networks require massive amounts of data to avoid overfitting and offer very poor interpretability natively. For 10,000 rows of tabular data, ensemble tree methods like Random Forest are computationally cheaper, more robust to outliers, and naturally suited for feature importance extraction (SHAP).

**Q: How do you handle missing data in the Batch Processor?**
**A**: The FastAPI backend is strictly typed using Pydantic. If a required field is missing, it will reject the payload. To prevent the entire batch from failing, the Streamlit frontend intercepts missing data, extracts those invalid rows into a downloadable CSV for the user to correct, and processes the remaining valid rows seamlessly.
