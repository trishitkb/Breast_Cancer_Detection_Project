from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from .schemas import PatientData

app = FastAPI(title="Breast Cancer Prediction API", version="1.0")

# Globals for models
model = None
scaler = None
label_encoders = None

@app.on_event("startup")
def load_models():
    global model, scaler, label_encoders
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
        encoders_path = os.path.join(os.path.dirname(__file__), '../models/label_encoders.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(patient: PatientData):
    if model is None or scaler is None or label_encoders is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")
        
    try:
        # Convert to dict and then DataFrame
        data = patient.model_dump()
        df = pd.DataFrame([data])
        
        # Apply Label Encoding
        for col, le in label_encoders.items():
            if col in df.columns:
                # Handle unseen labels by setting to a default or handling exception
                # For simplicity, we just transform assuming the input matches the training domain
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # In a robust system, we handle unseen categorical values. 
                    # Here we might assign it to a default category class 0
                    df[col] = 0
                    
        # Ensure the column order matches the model training exactly
        # The schema order should match, but let's ensure it's scaled correctly
        X_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else None
        
        return {
            "prediction": int(prediction),
            "cancer_risk": "High" if prediction == 1 else "Low",
            "probability": float(probability) if probability is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
