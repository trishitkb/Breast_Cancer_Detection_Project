import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()

def test_model_info():
    response = client.get("/model-info")
    # If the model metadata is loaded, it should return 200
    if response.status_code == 200:
        data = response.json()
        assert "optimal_threshold" in data
        assert "model_name" in data
    else:
        # 503 means model not loaded (perhaps tests running before startup is complete)
        assert response.status_code == 503

def test_predict_endpoint_validation():
    # Missing fields should trigger a 422 Unprocessable Entity
    response = client.post("/predict", json={"Age": 45})
    assert response.status_code == 422

def test_predict_endpoint_success():
    # Valid payload
    payload = {
        "Age": 55,
        "Gender": "Female",
        "BMI": 25.0,
        "Family_History": "Yes",
        "Smoking": "No",
        "Alcohol_Consumption": "No",
        "Physical_Activity": "Moderate",
        "Hormone_Therapy": "No",
        "Menopause_Status": "Post",
        "Genetic_Mutation": "Positive",
        "Tumor_Size_cm": 3.0,
        "Lymph_Node_Involvement": "Yes",
        "Mammogram_Result": "Suspicious",
        "Blood_Pressure": 130,
        "Cholesterol": 200,
        "Diabetes": "No",
        "Exercise_Days_Per_Week": 3,
        "Breastfeeding_History": "Yes",
        "Annual_Income_USD": 60000
    }
    
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "threshold_used" in data
        # SHAP explanations might be empty if explainer failed to load, but the key should exist
        assert "shap_explanations" in data
    else:
        assert response.status_code == 503 # Model unavailable
