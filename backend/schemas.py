from pydantic import BaseModel
from typing import Optional, List

class PatientData(BaseModel):
    Age: int
    Gender: str
    BMI: float
    Family_History: str
    Smoking: str
    Alcohol_Consumption: str
    Physical_Activity: str
    Hormone_Therapy: str
    Menopause_Status: str
    Genetic_Mutation: str
    Tumor_Size_cm: float
    Lymph_Node_Involvement: str
    Mammogram_Result: str
    Blood_Pressure: int
    Cholesterol: int
    Diabetes: str
    Exercise_Days_Per_Week: int
    Breastfeeding_History: str
    Annual_Income_USD: int

class BatchPatientData(BaseModel):
    patients: List[PatientData]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
