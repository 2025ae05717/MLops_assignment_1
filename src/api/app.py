"""
FastAPI application for Heart Disease Prediction API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions', ['model', 'result'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

# Load model and preprocessing components
MODEL_PATH = Path("models/best_model.joblib")
SCALER_PATH = Path("data/scaler.joblib")
FEATURE_COLUMNS_PATH = Path("data/feature_columns.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    logger.info("Model and preprocessing components loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or preprocessing components: {e}")
    model = None
    scaler = None
    feature_columns = None


# Define input schema
class HeartDiseaseInput(BaseModel):
    """Input schema for heart disease prediction."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0 = female, 1 = male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0 = no, 1 = yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 2,
                "thal": 3
            }
        }
    }


# Define output schema
class PredictionResponse(BaseModel):
    """Response schema for heart disease prediction."""

    prediction: int = Field(..., description="Predicted class (0 = no disease, 1 = disease)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "healthy" if model is not None else "unhealthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None or scaler is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model or preprocessing components not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: HeartDiseaseInput):
    """
    Predict heart disease risk.

    Args:
        input_data: Heart disease features

    Returns:
        Prediction with confidence score and risk level
    """
    if model is None or scaler is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model or preprocessing components not loaded")

    started_at = time.perf_counter()
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Ensure correct column order
        input_df = input_df[feature_columns]

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # Get confidence (probability of predicted class)
        confidence = probability[prediction]

        # Risk is based on estimated disease probability, not confidence in either class.
        disease_probability = float(probability[1])
        if disease_probability < 0.5:
            risk_level = "Low"
        elif disease_probability < 0.75:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Log prediction
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        prediction_counter.labels(
            model=model.__class__.__name__,
            result="disease" if prediction == 1 else "no_disease"
        ).inc()

        return PredictionResponse(
            prediction=int(prediction),
            confidence=float(confidence),
            risk_level=risk_level
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        prediction_latency.observe(time.perf_counter() - started_at)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
