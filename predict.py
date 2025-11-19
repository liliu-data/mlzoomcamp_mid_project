import pickle
from typing import Literal
from pydantic import BaseModel, Field
import xgboost as xgb

from fastapi import FastAPI
import uvicorn

class PatientData(BaseModel):
    gender: Literal['male', 'female', 'other']
    age: float
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: Literal['yes', 'no']
    work_type: Literal['private', 'self-employed', 'govt_job', 'children', 'never_worked']
    residence_type: Literal['urban', 'rural']
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)
    smoking_status: Literal['never_smoked', 'formerly_smoked', 'smokes', 'unknown']

class PredictResponse(BaseModel):
    stroke_probability: float
    stroke: bool

app = FastAPI(title='Stroke Prediction API')

with open('model.bin', 'rb') as f_in:
    model, dv = pickle.load(f_in)

def predict_single(patient: PatientData):
    patient_dict = patient.dict()
    X = dv.transform([patient_dict])
    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    y_pred = model.predict(dmatrix)[0]
    return float(y_pred)

@app.post('/predict')
def predict_endpoint(patient: PatientData) -> PredictResponse:
    prob = predict_single(patient)
    
    return PredictResponse(
        stroke_probability=prob,
        stroke=bool(prob >= 0.5)
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

