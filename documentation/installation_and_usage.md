# Installation and Usage Guide

Welcome to the Breast Cancer Hybrid Prediction System (BC-HPS) documentation. This guide covers how to set up the environment, run the servers, and utilize the user interface effectively.

## 1. Prerequisites

- **Python**: Version 3.9 to 3.11 is recommended.
- **Git**: To clone the repository (if applicable).
- **uv / pip**: Package managers to install dependencies.

## 2. Environment Setup

It is highly recommended to run this project inside a virtual environment to isolate dependencies.

### Windows (PowerShell)
```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### macOS / Linux (Bash/Zsh)
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

## 3. Installing Dependencies

Once your virtual environment is active, install the required packages:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> The `requirements.txt` includes essential packages like `fastapi`, `uvicorn`, `streamlit`, `pandas`, `scikit-learn`, `shap`, and `fpdf`. Ensure installation finishes without errors.

## 4. Running the Application

This system utilizes a decoupled architecture. You must run the **FastAPI Backend** and the **Streamlit Frontend** simultaneously in two separate terminal windows.

### Terminal 1: Start the FastAPI Backend
The backend serves the ML models and calculates SHAP values.
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- You can access the API documentation (Swagger UI) at: `http://127.0.0.1:8000/docs`
- To verify health, visit: `http://127.0.0.1:8000/health`

### Terminal 2: Start the Streamlit Frontend
The frontend provides the interactive user interface. Ensure your virtual environment is activated in this terminal as well.
```bash
streamlit run frontend/app.py --server.port 8501
```
- The application will automatically open in your default web browser at `http://localhost:8501`.

## 5. Usage Guide

Once both servers are running, you can navigate the system using the left sidebar in the Streamlit UI.

### Home / Overview
Provides a high-level summary of the BC-HPS dual-phase diagnostic approach. 

### Hybrid Prediction
This is the core diagnostic tool. 
- **Pre-Diagnostic Phase**: Enter demographic, lifestyle, and basic clinical data (e.g., Age, BMI, Blood Pressure). The system evaluates these factors to generate a base risk assessment.
- **Diagnostic Assessment Phase**: If available, enter diagnostic imaging or pathology results (e.g., Mammogram Result, Tumor Size). The system will automatically upgrade the prediction model to provide a highly accurate post-diagnostic classification.
- **SHAP Explanations**: Below the prediction gauge, the system highlights which specific factors contributed most to the prediction.

### Batch Processing
Designed for clinical analysts managing bulk records.
1. Download or prepare a CSV file with patient records (e.g., `test.csv`). The CSV must contain the required clinical features.
2. Upload the CSV. (If your CSV lacks headers, the system will attempt to auto-assign them).
3. Click **Run Batch Prediction**.
4. Review the generated statistics summary.
5. Click **Download All Reports (ZIP)** to download a generated ZIP archive containing individual PDF reports categorized into `safe/` and `critical/` folders.

> [!TIP]
> If any rows in your uploaded CSV contain missing or null values for required fields, they will be automatically separated. You can download these excluded rows via a dedicated button before running the batch prediction.

### Model Diagnostics
Intended for data scientists and technical stakeholders. View the ROC-AUC curves and global SHAP summary plots generated during the model's training phase to understand overall model behavior and feature importance across the entire dataset.
