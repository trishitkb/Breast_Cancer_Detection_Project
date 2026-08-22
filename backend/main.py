from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
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
    description="API for predicting breast cancer risk using a trained ML Pipeline with optimized threshold.",
    version="2.0.0"
)

# Global variables for models and metadata
pipeline = None
metadata = None
explainer = None
HISTORY_FILE = config.HISTORY_FILE

@app.on_event("startup")
def load_assets():
    global pipeline, metadata, explainer
    logger.info("Loading model pipeline and metadata...")
    
    try:
        pipeline = joblib.load(config.MODEL_PATH)
        
        with open(config.METADATA_PATH, "r") as f:
            metadata = json.load(f)
            
        logger.info(f"Loaded {metadata.get('model_name')} optimized for {metadata.get('dataset_balancing')}.")
        logger.info(f"Optimal Threshold: {metadata.get('optimal_threshold')}")
        
        # Load SHAP explainer
        # For CalibratedClassifierCV or pipelines, KernelExplainer with a background dataset is safest
        try:
            background_data = pd.read_csv(config.BACKGROUND_DATA_PATH)
            # Create a prediction function that outputs the probability of class 1
            def predict_fn(X):
                # Ensure X is a DataFrame with correct columns if background_data has columns
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=background_data.columns)
                return pipeline.predict_proba(X)[:, 1]
            
            # Use kmeans to summarize background data if it's too large, but 100 samples is small enough
            import shap
            import numpy as np
            explainer = shap.KernelExplainer(predict_fn, background_data)
            logger.info("SHAP KernelExplainer initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            
    except Exception as e:
        logger.error(f"Failed to load required model assets: {e}")
        # In a real production environment, you might want to raise here. 
        # But for development, we just log it.

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

class BatchPatientData(BaseModel):
    patients: list[PatientData]

def save_prediction_history(df, probabilities, predictions):
    """Append predictions to a CSV file for tracking and monitoring."""
    try:
        history_df = df.copy()
        history_df['timestamp'] = datetime.datetime.now().isoformat()
        history_df['probability'] = probabilities
        history_df['prediction'] = predictions
        
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
            history_df.to_csv(HISTORY_FILE, index=False)
        else:
            history_df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    except Exception as e:
        logger.error(f"Failed to save prediction history: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.get("/model-info")
def model_info():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return metadata

@app.post("/predict")
def predict(patient: PatientData):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # Convert to DataFrame
        df = pd.DataFrame([patient.dict()])
        
        # Predict probability
        prob = pipeline.predict_proba(df)[0, 1]
        
        # Apply optimized threshold or fallback to env threshold
        threshold = metadata.get("optimal_threshold", config.PREDICTION_THRESHOLD)
        pred = 1 if prob >= threshold else 0
        
        # Calculate Local SHAP values
        shap_values_dict = {}
        if explainer is not None:
            # Calculate SHAP value for this single instance
            shap_vals = explainer.shap_values(df)
            # shap_vals is an array. For KernelExplainer and single output, it's 1D or 2D depending on shap version.
            # Usually shap_vals is a list if multi-class, but predict_fn returns 1D array (probabilities),
            # so shap_vals should be shape (1, num_features)
            import numpy as np
            vals = np.array(shap_vals)
            if len(vals.shape) == 2:
                vals = vals[0]
            
            # Map features to their SHAP values
            for feature, val in zip(df.columns, vals):
                shap_values_dict[feature] = float(val)
                
            # Sort by absolute impact
            shap_values_dict = dict(sorted(shap_values_dict.items(), key=lambda item: abs(item[1]), reverse=True))

        # Save to history
        save_prediction_history(df, [float(prob)], [int(pred)])
        
        logger.info(f"Prediction made: Risk={pred}, Prob={prob:.4f}")
        
        return {
            "prediction": int(pred),
            "cancer_risk": "High Risk (Malignant)" if pred == 1 else "Low Risk (Benign)",
            "probability": float(prob),
            "threshold_used": float(threshold),
            "shap_explanations": shap_values_dict
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(batch: BatchPatientData):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        df = pd.DataFrame([p.dict() for p in batch.patients])
        
        probabilities = pipeline.predict_proba(df)[:, 1]
        threshold = metadata.get("optimal_threshold", config.PREDICTION_THRESHOLD)
        predictions = (probabilities >= threshold).astype(int)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "cancer_risk": "High Risk" if pred == 1 else "Low Risk",
                "probability": float(prob)
            })
            
        save_prediction_history(df, probabilities.tolist(), predictions.tolist())
        logger.info(f"Batch prediction completed for {len(df)} records.")
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
