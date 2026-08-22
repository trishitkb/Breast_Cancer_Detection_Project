# Breast Cancer Prediction Project

This project provides an end-to-end Machine Learning pipeline to predict whether a breast tumor is malignant or benign.

## Project Structure
- `data/`: Raw and processed dataset
- `notebook/`: Scripts for EDA, preprocessing, training, and explainability
- `models/`: Saved models and scalers
- `reports/figures/`: EDA and SHAP visualizations
- `backend/`: FastAPI application
- `frontend/`: Streamlit dashboard
- `tests/`: Automated tests
- `documentation/`: Additional docs and presentation material

## Architecture

The project is built on a modular architecture to separate concerns:
- **Data Pipeline**: Python (pandas, scikit-learn) for cleaning, preprocessing, and training. Imbalanced data is handled using SMOTE.
- **Model Storage**: Trained artifacts are saved using joblib along with JSON metadata to ensure versioning.
- **Backend API**: A FastAPI server that loads the models and exposes endpoints for real-time and batch predictions. Includes integration with SHAP for model explainability.
- **Frontend Dashboard**: A Streamlit application for interactive predictions, batch uploads, and visualizing Exploratory Data Analysis (EDA) and SHAP values.

## ML Pipeline

1. **Ingestion**: Raw CSV is loaded.
2. **Preprocessing**: Handling missing values, mapping categoricals, and feature scaling using `StandardScaler` and `OrdinalEncoder`.
3. **Imbalance Handling**: Comparing SMOTE, undersampling, and class weights.
4. **Training**: Comparing Logistic Regression, Random Forest, and Gradient Boosting.
5. **Optimization**: Optimizing the decision threshold to prioritize Recall (minimizing false negatives) using the F2 score.
6. **Persistence**: Pipeline and metadata saved to `models/`.

## Results
The Champion Model achieved exceptional performance by prioritizing recall:
- **Recall (Malignant)**: 0.96 (optimized for threshold ~0.3)
- **Catch Rate**: Minimizes false negatives aggressively, meaning 96% of true malignant cases are successfully identified, minimizing catastrophic clinical misses.
- **Explainability**: Top drivers for malignancy predictions typically include features like `Age`, `Tumor_Size_cm`, and `BMI`, as visualized by SHAP dependence and force plots in the Streamlit dashboard.

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
API docs are available at `http://localhost:8000/docs` (Swagger UI).
Key Endpoints:
- `GET /health` - API status
- `GET /model-info` - Metadata of the currently loaded model
- `POST /predict` - Real-time prediction with SHAP explanations
- `POST /predict_batch` - Batch inference for multiple patients

### 3. Run the Dashboard
Start the Streamlit app:
```bash
.venv\Scripts\streamlit run frontend\app.py --server.port 8501
```
Open your browser to `http://localhost:8501`.

## Limitations & Future Work

**Limitations**:
- The SHAP KernelExplainer is computationally expensive and may slow down batch predictions.
- Dataset represents a specific demographic; model generalization needs external validation.

**Future Work**:
- Containerize the backend and frontend using Docker for simplified deployment.
- Integrate MLflow for robust experiment tracking.
- Set up a CI/CD pipeline using GitHub Actions to automate testing.
