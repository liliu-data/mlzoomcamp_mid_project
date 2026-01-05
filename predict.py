import pickle
from typing import Literal, Optional
from pydantic import BaseModel, Field
import xgboost as xgb
import lightgbm as lgb
import numpy as np

from fastapi import FastAPI, HTTPException
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
    model_used: str

class PredictWithModelResponse(BaseModel):
    stroke_probability: float
    stroke: bool
    model_used: str
    all_models: dict

app = FastAPI(
    title='Stroke Prediction API',
    description='API for stroke prediction using multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)',
    version='2.0.0'
)

# Load model
try:
    with open('model.bin', 'rb') as f_in:
        model_data = pickle.load(f_in)
        if len(model_data) == 3:
            model, dv, model_name = model_data
        else:
            # Backward compatibility
            model, dv = model_data
            model_name = "XGBoost"
    print(f"Loaded model: {model_name}")
except FileNotFoundError:
    import sys
    print("ERROR: Model file 'model.bin' not found. Please run train.py first.", file=sys.stderr)
    raise FileNotFoundError("Model file 'model.bin' not found. Please run train.py first.")
except Exception as e:
    import sys
    print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
    raise

def predict_single(patient: PatientData, model_obj, vectorizer, model_type: str):
    """Make prediction using a specific model."""
    patient_dict = patient.dict()
    X = vectorizer.transform([patient_dict])
    
    if model_type == 'XGBoost':
        dmatrix = xgb.DMatrix(X, feature_names=list(vectorizer.get_feature_names_out()))
        y_pred = model_obj.predict(dmatrix)[0]
    elif model_type == 'LightGBM':
        y_pred = model_obj.predict(X)[0]
    elif model_type in ['Logistic Regression', 'Random Forest']:
        y_pred = model_obj.predict_proba(X)[0, 1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return float(y_pred)

@app.get('/')
def root():
    return {
        "message": "Stroke Prediction API",
        "version": "2.0.0",
        "model": model_name,
        "endpoints": {
            "/predict": "Predict using the best model",
            "/predict/all": "Get predictions from all available models",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }

@app.get('/health')
def health():
    return {"status": "healthy", "model": model_name}

@app.post('/predict')
def predict_endpoint(patient: PatientData) -> PredictResponse:
    """Predict stroke probability using the best trained model."""
    try:
        prob = predict_single(patient, model, dv, model_name)
        
        return PredictResponse(
            stroke_probability=prob,
            stroke=bool(prob >= 0.5),
            model_used=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post('/predict/all')
def predict_all_models(patient: PatientData) -> PredictWithModelResponse:
    """Get predictions from all available models (if all models file exists)."""
    try:
        # Try to load all models
        try:
            with open('models_all_latest.bin', 'rb') as f_in:
                all_models_data = pickle.load(f_in)
                if len(all_models_data) == 3:
                    all_models_dict, vectorizer, _ = all_models_data
                else:
                    raise HTTPException(
                        status_code=404, 
                        detail="All models file not found. Run train.py to generate it."
                    )
        except FileNotFoundError:
            # Fallback to just the best model
            all_models_dict = {model_name: model}
            vectorizer = dv
        
        results = {}
        for model_name_key, model_obj in all_models_dict.items():
            try:
                prob = predict_single(patient, model_obj, vectorizer, model_name_key)
                results[model_name_key] = {
                    "stroke_probability": prob,
                    "stroke": bool(prob >= 0.5)
                }
            except Exception as e:
                results[model_name_key] = {"error": str(e)}
        
        # Use the best model's prediction as primary
        best_prob = predict_single(patient, model, dv, model_name)
        
        return PredictWithModelResponse(
            stroke_probability=best_prob,
            stroke=bool(best_prob >= 0.5),
            model_used=model_name,
            all_models=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
