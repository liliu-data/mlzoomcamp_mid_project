from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import xgboost as xgb
import numpy as np

app = FastAPI()

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dv.pkl', 'rb') as f:
    dv = pickle.load(f)

class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/")
def home():
    return {"message": "Stroke Prediction API - Send POST request to /predict"}

@app.post("/predict")
def predict(patient: PatientData):
    # Convert input to dict
    patient_dict = patient.dict()
    
    # Transform features
    X = dv.transform([patient_dict])
    
    # Make prediction
    dmatrix = xgb.DMatrix(X)
    probability = float(model.predict(dmatrix)[0])
    
    # Return result
    return {
        "stroke_probability": probability,
        "risk": "high" if probability > 0.5 else "low"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)