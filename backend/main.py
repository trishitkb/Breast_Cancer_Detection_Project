from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import json
import os
import datetime
import logging
import shap
import warnings
from backend.config import config

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predicting breast cancer risk using a trained ML Pipeline with optimized threshold. Supports Hybrid Pre-Diagnostic and Post-Diagnostic models.",
    version="2.0.0"
)

# Global variables for models and metadata
pre_pipeline = None
post_pipeline = None
metadata = None
pre_explainer = None
post_explainer = None
pre_features = []
post_features = []
HISTORY_FILE = config.HISTORY_FILE

@app.on_event("startup")
def load_assets():
    global pre_pipeline, post_pipeline, metadata, pre_explainer, post_explainer, pre_features, post_features
    import sklearn
    logger.info(f"Loaded scikit-learn version: {sklearn.__version__}")
    logger.info("Loading model pipelines and metadata...")
    
    try:
        # Load metadata
        with open('models/model_metadata.json', "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        pre_features = metadata['pre_diagnostic']['features']
        post_features = metadata['diagnostic_assessment']['features']

        # Load Pre-Diagnostic Model
        pre_pipeline = joblib.load('models/pre_diagnostic_pipeline.pkl')
        
        # Load Post-Diagnostic Model
        post_pipeline = joblib.load('models/diagnostic_assessment_pipeline.pkl')

        # Dynamically generate SHAP background data strictly aligned with training features
        raw_path = 'data/raw/breast_cancer_prediction.csv'
        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path)
            
            bg_pre = df_raw[pre_features].sample(n=min(50, len(df_raw)), random_state=42)
            bg_post = df_raw[post_features].sample(n=min(50, len(df_raw)), random_state=42)
            
            def predict_fn_pre(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=pre_features)
                return pre_pipeline.predict_proba(X)[:, 1]
            pre_explainer = shap.KernelExplainer(predict_fn_pre, bg_pre)
            
            def predict_fn_post(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=post_features)
                return post_pipeline.predict_proba(X)[:, 1]
            post_explainer = shap.KernelExplainer(predict_fn_post, bg_post)
            logger.info("Models and SHAP explainers initialized successfully with aligned features.")
        else:
            logger.warning("Raw dataset not found; SHAP explainers skipped.")
            
    except Exception as e:
        logger.error(f"Failed to load required model assets: {e}")

class PatientData(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Gender: str
    BMI: float
    Family_History: str
    Smoking: str
    Alcohol_Consumption: str
    Physical_Activity: str
    Hormone_Therapy: str
    Menopause_Status: str
    Genetic_Mutation: str
    Blood_Pressure: int
    Cholesterol: int
    Diabetes: str
    Exercise_Days_Per_Week: int
    Breastfeeding_History: str
    Annual_Income_USD: int
    
    # Optional diagnostic features
    Mammogram_Result: Optional[str] = None
    Lymph_Node_Involvement: Optional[str] = None
    Tumor_Size_cm: Optional[float] = None

class BatchPatientData(BaseModel):
    patients: list[PatientData]

def save_prediction_history(df, probabilities, predictions, model_types):
    """Append predictions to a CSV file for tracking and monitoring."""
    try:
        history_df = df.copy()
        history_df['timestamp'] = datetime.datetime.now().isoformat()
        history_df['probability'] = probabilities
        history_df['prediction'] = predictions
        history_df['model_type'] = model_types
        
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
            history_df.to_csv(HISTORY_FILE, index=False)
        else:
            history_df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    except Exception as e:
        logger.error(f"Failed to save prediction history: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "pre_model_loaded": pre_pipeline is not None, "post_model_loaded": post_pipeline is not None}

@app.get("/model-info")
def model_info():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return metadata

@app.post("/predict")
def predict(patient: PatientData):
    if pre_pipeline is None or post_pipeline is None:
        raise HTTPException(status_code=503, detail="Models are currently unavailable.")
        
    try:
        # Determine if we have valid diagnostic data
        has_diagnostic = False
        if patient.Mammogram_Result and patient.Mammogram_Result != "Not Tested" and \
           patient.Lymph_Node_Involvement and patient.Lymph_Node_Involvement != "Not Tested" and \
           patient.Tumor_Size_cm is not None and patient.Tumor_Size_cm > 0.0:
            has_diagnostic = True

        raw_dict = patient.dict()
        
        if has_diagnostic:
            pipeline = post_pipeline
            explainer = post_explainer
            scenario = "diagnostic_assessment"
            model_type = "Post-Test Diagnostic Classification"
            features = post_features
        else:
            pipeline = pre_pipeline
            explainer = pre_explainer
            scenario = "pre_diagnostic"
            model_type = "Pre-Test Risk Assessment"
            features = pre_features
        
        # Build DataFrame with strictly ordered feature columns
        df = pd.DataFrame([{col: raw_dict.get(col) for col in features}])
        
        # Predict probability
        prob = pipeline.predict_proba(df)[0, 1]
        
        # Apply optimized threshold
        threshold = metadata.get(scenario, {}).get("threshold", 0.5)
        pred = 1 if prob >= threshold else 0
        
        # Calculate Local SHAP values
        shap_values_dict = {}
        if explainer is not None:
            shap_vals = explainer.shap_values(df, nsamples=50)
            vals = np.array(shap_vals)
            if len(vals.shape) == 2:
                vals = vals[0]
            
            for feature, val in zip(df.columns, vals):
                shap_values_dict[feature] = float(val)
                
            shap_values_dict = dict(sorted(shap_values_dict.items(), key=lambda item: abs(item[1]), reverse=True))

        # Save to history
        save_prediction_history(df, [float(prob)], [int(pred)], [model_type])
        
        logger.info(f"Prediction made ({model_type}): Risk={pred}, Prob={prob:.4f}")
        
        return {
            "prediction": int(pred),
            "cancer_risk": "High Risk (Malignant)" if pred == 1 else "Low Risk (Benign)",
            "probability": float(prob),
            "threshold_used": float(threshold),
            "model_type": model_type,
            "shap_explanations": shap_values_dict
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(batch: BatchPatientData):
    if pre_pipeline is None or post_pipeline is None:
        raise HTTPException(status_code=503, detail="Models are currently unavailable.")
        
    try:
        results = []
        probabilities_list = []
        predictions_list = []
        model_types_list = []
        
        for p in batch.patients:
            raw_dict = p.dict()
            has_diagnostic = (raw_dict.get('Mammogram_Result') and raw_dict.get('Mammogram_Result') != "Not Tested" and 
                              raw_dict.get('Lymph_Node_Involvement') and raw_dict.get('Lymph_Node_Involvement') != "Not Tested" and 
                              raw_dict.get('Tumor_Size_cm') is not None and float(raw_dict.get('Tumor_Size_cm') or 0.0) > 0.0)
            
            if has_diagnostic:
                pipeline = post_pipeline
                scenario = "diagnostic_assessment"
                model_type = "Post-Test Diagnostic Classification"
                features = post_features
            else:
                pipeline = pre_pipeline
                scenario = "pre_diagnostic"
                model_type = "Pre-Test Risk Assessment"
                features = pre_features
                
            row_df = pd.DataFrame([{col: raw_dict.get(col) for col in features}])
            prob = pipeline.predict_proba(row_df)[0, 1]
            threshold = metadata.get(scenario, {}).get("threshold", 0.5)
            pred = 1 if prob >= threshold else 0
            
            results.append({
                "prediction": int(pred),
                "cancer_risk": "High Risk (Malignant)" if pred == 1 else "Low Risk (Benign)",
                "probability": float(prob),
                "threshold_used": float(threshold),
                "model_type": model_type
            })
            probabilities_list.append(prob)
            predictions_list.append(pred)
            model_types_list.append(model_type)
            
        full_df = pd.DataFrame([p.dict() for p in batch.patients])
        save_prediction_history(full_df, probabilities_list, predictions_list, model_types_list)
        logger.info(f"Batch prediction completed for {len(batch.patients)} records.")
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
