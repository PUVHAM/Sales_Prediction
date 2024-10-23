import os
import joblib
import numpy as np
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.config import ModelConfig
from backend.train import train_models_parallel, SalesModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    model_type: str
    input_data: Dict[str, Any]

@app.post("/train_model/")
def train_model():
    model_types = list(ModelConfig.MODEL_TYPES.values())
    
    _, results = train_models_parallel(model_types)
    
    os.makedirs("backend/models", exist_ok=True)
    
    return {
        "status": "success",
        "result": results
    }
    
@app.post("/predict")
def predict(request: PredictRequest):
    input_data = np.array(list(request.input_data.values())).reshape(1, -1)

    model_path = f"backend/models/{request.model_type}_model.pkl"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")

    try:
        # Load model using the class method
        model = SalesModel.load(model_path)
        
        scaler = joblib.load("backend/data/scaler.pkl")
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))