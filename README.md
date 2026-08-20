# Breast Cancer Prediction Project

This project provides an end-to-end Machine Learning pipeline to predict whether a breast tumor is malignant or benign.

## Project Structure
- `data/`: Raw and processed dataset
- `notebook/`: Scripts for EDA, preprocessing, training, and explainability
- `models/`: Saved models and scalers
- `reports/figures/`: EDA and SHAP visualizations
- `backend/`: FastAPI application
- `frontend/`: Streamlit dashboard
- `documentation/`: Additional docs and presentation material

## Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate  # Windows
   # or source .venv/bin/activate # Linux/Mac
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Pipeline & Training
Run the pipeline scripts in order:
```bash
jupyter nbconvert --to notebook --execute notebook/breast_cancer_pipeline.ipynb
```

### 2. Run the Backend API
Start the FastAPI server:
```bash
.venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
API docs available at `http://localhost:8000/docs`.

### 3. Run the Dashboard
Start the Streamlit app:
```bash
.venv\Scripts\streamlit run frontend\app.py --server.port 8501
```
Open your browser to `http://localhost:8501`.
