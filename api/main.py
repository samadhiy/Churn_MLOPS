from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

app = FastAPI()

# 1. Define a Request Body Model
class PredictionRequest(BaseModel):
    data: List[float]

# Locate the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "RandomForest.pkl")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.post("/predict")
def predict(request: PredictionRequest): # Use the Pydantic model here
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found or corrupted.")
    
    try:
        # Extract the list from the request object
        data_array = np.array(request.data).reshape(1, -1)
        
        # Get probability
        prob = model.predict_proba(data_array)[0][1]
        prediction = "Yes" if prob > 0.5 else "No"

        return {
            "churn_probability": float(prob),
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")