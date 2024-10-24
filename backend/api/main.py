import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.config import ModelConfig, DatasetConfig
from src.train import train_models_parallel, SalesModel
from src.schemas.api_schema import PredictRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train_model/")
def train_model():
    model_types = list(ModelConfig.MODEL_TYPES.values())
    
    os.makedirs(ModelConfig.MODEL_DIR, exist_ok=True)
    
    _, results = train_models_parallel(model_types)
    
    return {
        "status": "success",
        "result": results
    }
    
@app.post("/predict")
def predict(request: PredictRequest):
    input_data = np.array(list(request.input_data.values())).reshape(1, -1)

    model_path = ModelConfig.get_model_path(request.model_type)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")

    try:
        # Load model using the class method
        model = SalesModel.load(model_path)
        
        scaler = joblib.load(DatasetConfig.SCALER_PATH)
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/health")
def health_check():
    return {"status": "healthy"}