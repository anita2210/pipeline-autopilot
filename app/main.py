from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.alert_system import send_alert

logger = logging.getLogger(__name__)
app = FastAPI(title="PipelineGuard API", version="1.0.0")

class PredictRequest(BaseModel):
    features: dict
    pipeline_name: Optional[str] = "unknown_pipeline"

class PredictResponse(BaseModel):
    probability: float
    prediction: int
    risk_level: str
    top_shap_features: List[str]
    alert_sent: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Member 2 will replace stub with real model inference
    probability = 0.5
    prediction = int(probability > 0.5)
    top_shap_features = ["retry_count", "duration_deviation", "failures_last_7_runs"]

    risk_level = _risk_level(probability)

    alert_sent = send_alert(
        pipeline_name=request.pipeline_name,
        risk_score=probability,
        top_shap_features=top_shap_features,
    )

    return PredictResponse(
        probability=probability,
        prediction=prediction,
        risk_level=risk_level,
        top_shap_features=top_shap_features,
        alert_sent=alert_sent,
    )

def _risk_level(score: float) -> str:
    if score >= 0.90: return "CRITICAL"
    if score >= 0.75: return "HIGH"
    if score >= 0.50: return "MEDIUM"
    return "LOW"
