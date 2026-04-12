"""
main.py
-------
FastAPI backend for PipelineGuard. Exposes:
  - /predict    : Single pipeline run failure prediction
  - /stream     : Live SSE stream of predictions from GitHub Actions API
  - /explain    : RAG diagnosis (WHY + FIX + CODE) via rag_chatbot.py
  - /health     : Health check
  - /metrics    : Last 100 predictions summary

Author  : Member 2 (Backend + ML Engineer)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from app.alert_system import send_alert
from app.live_poller import live_prediction_generator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PipelineGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/trained/best_model.joblib")
model = None

def get_model():
    global model
    if model is None:
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                logger.info("Model loaded from: %s", MODEL_PATH)
            except Exception as exc:
                logger.warning("Model load failed: %s — using stub predictions", exc)
                model = None
        else:
            logger.warning("Model not found at %s — using stub predictions", MODEL_PATH)
    return model

# ---------------------------------------------------------------------------
# Prediction history (in-memory, last 100)
# ---------------------------------------------------------------------------
prediction_history: list = []


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: dict
    pipeline_name: Optional[str] = "unknown_pipeline"


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    risk_level: str
    top_shap_features: List[str]
    alert_sent: bool


class ExplainRequest(BaseModel):
    features: dict
    failure_prob: float
    pipeline_name: Optional[str] = "unknown_pipeline"


# ---------------------------------------------------------------------------
# Core predict function (used by /predict and /stream)
# ---------------------------------------------------------------------------

def run_prediction(features: dict, pipeline_name: str = "unknown") -> dict:
    """
    Run model inference on a feature dict.

    Parameters
    ----------
    features      : Feature dict with pipeline run metadata.
    pipeline_name : Name of the pipeline being predicted.

    Returns
    -------
    dict with probability, prediction, risk_level, top_shap_features, alert_sent.
    """
    loaded_model = get_model()

    if loaded_model is not None:
        try:
            # Build feature array in correct order
            drop_cols = {"run_id", "pipeline_name", "trigger_time", "failed",
                        "failure_type", "error_message", "repo"}
            feature_values = {k: v for k, v in features.items() if k not in drop_cols}
            X = np.array(list(feature_values.values())).reshape(1, -1)
            probability = float(loaded_model.predict_proba(X)[0][1])
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            probability = 0.5
    else:
        probability = 0.5

    prediction = int(probability >= 0.5)
    risk_level = _risk_level(probability)
    top_shap_features = ["retry_count", "duration_deviation", "failures_last_7_runs"]

    alert_sent = False
    if probability >= 0.75:
        alert_sent = send_alert(
            pipeline_name=pipeline_name,
            risk_score=probability,
            top_shap_features=top_shap_features,
        )

    result = {
        "probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "top_shap_features": top_shap_features,
        "alert_sent": alert_sent,
    }

    # Store in history
    prediction_history.append({**result, "pipeline_name": pipeline_name})
    if len(prediction_history) > 100:
        prediction_history.pop(0)

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": False,
        "version": "2.0.0",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Single pipeline run failure prediction.
    Accepts 21 pipeline features, returns probability + risk level + SHAP features.
    """
    result = run_prediction(request.features, request.pipeline_name)
    return PredictResponse(**result)


@app.get("/stream")
async def stream():
    """
    Live SSE stream of pipeline failure predictions.
    Polls GitHub Actions API every 30 seconds across monitored repos.
    Connect from Streamlit using SSE client.
    """
    return StreamingResponse(
        live_prediction_generator(predict_fn=run_prediction),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/explain")
def explain(request: ExplainRequest):
    """
    RAG diagnosis endpoint. Returns WHY + FIX + CODE for a pipeline failure.
    Calls rag_chatbot.get_diagnosis() with pipeline features and failure probability.
    """
    try:
        from app.rag_chatbot import get_diagnosis
        diagnosis = get_diagnosis(request.features, request.failure_prob)
        return {"status": "ok", "diagnosis": diagnosis}
    except Exception as exc:
        logger.error("RAG diagnosis failed: %s", exc)
        return {
            "status": "error",
            "diagnosis": {
                "why": "Unable to generate diagnosis at this time.",
                "fix": "Please check pipeline logs manually.",
                "code": "",
            }
        }


@app.get("/metrics")
def metrics():
    """
    Summary of the last 100 predictions.
    Returns counts by risk level and average probability.
    """
    if not prediction_history:
        return {"message": "No predictions yet", "total": 0}

    total = len(prediction_history)
    avg_prob = round(sum(p["probability"] for p in prediction_history) / total, 4)
    risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for p in prediction_history:
        risk_counts[p.get("risk_level", "LOW")] += 1

    return {
        "total_predictions": total,
        "average_probability": avg_prob,
        "risk_distribution": risk_counts,
        "alert_rate": round(
            sum(1 for p in prediction_history if p.get("alert_sent")) / total, 4
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score >= 0.90: return "CRITICAL"
    if score >= 0.75: return "HIGH"
    if score >= 0.50: return "MEDIUM"
    return "LOW"