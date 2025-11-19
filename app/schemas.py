from pydantic import BaseModel, Field
from typing import Literal

class StrokePredictionInput(BaseModel):
    gender: Literal['male', 'female', 'other'] = Field(..., description="Gender of the patient")
    age: float = Field(..., ge=0, le=120, description="Age of the patient")
    hypertension: Literal[0, 1] = Field(..., description="Hypertension status (0: No, 1: Yes)")
    heart_disease: Literal[0, 1] = Field(..., description="Heart disease status (0: No, 1: Yes)")
    ever_married: Literal['yes', 'no'] = Field(..., description="Marital status")
    work_type: Literal['private', 'self-employed', 'govt_job', 'children', 'never_worked'] = Field(..., description="Type of work")
    residence_type: Literal['urban', 'rural'] = Field(..., description="Type of residence")
    avg_glucose_level: float = Field(..., ge=50, le=300, description="Average glucose level")
    bmi: float = Field(..., ge=10, le=50, description="Body Mass Index")
    smoking_status: Literal['formerly_smoked', 'never_smoked', 'smokes', 'unknown'] = Field(..., description="Smoking status")

class StrokePredictionOutput(BaseModel):
    stroke_probability: float = Field(..., ge=0, le=1, description="Probability of stroke")
    prediction: Literal['low_risk', 'high_risk'] = Field(..., description="Risk classification")
    risk_level: Literal['low', 'high'] = Field(..., description="Risk level")