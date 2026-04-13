"""
app/main.py
-----------
FastAPI backend for Pipeline Autopilot.

Endpoints:
  POST /predict  — accepts 21 features, returns probability + SHAP
  GET  /health   — model version + status
  GET  /metrics  — last 100 predictions summary

Author  : Member 2 (Anita)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models" / "trained"
MODEL_PATH  = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
META_PATH   = MODELS_DIR / "model_metadata.json"
FEAT_PATH   = MODELS_DIR / "feature_names.json"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PipelineGuard API",
    description="CI/CD Pipeline Failure Prediction System",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
model        = None
scaler       = None
explainer    = None
feature_names = None
metadata     = {}
pred_history  = deque(maxlen=100)   # rolling log of last 100 predictions

@app.on_event("startup")
def load_artifacts():
    global model, scaler, explainer, feature_names, metadata
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Model loaded: {MODEL_PATH.name}")

    # Feature names
    if FEAT_PATH.exists():
        with open(FEAT_PATH) as f:
            feature_names = json.load(f)
    
    # Metadata
    if META_PATH.exists():
        with open(META_PATH) as f:
            metadata = json.load(f)

    # SHAP explainer — TreeExplainer for XGBoost/RF, KernelExplainer fallback
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer loaded.")
    except Exception:
        logger.warning("TreeExplainer failed, using LinearExplainer fallback.")
        try:
            explainer = shap.LinearExplainer(model, shap.maskers.Independent(np.zeros((1, len(feature_names or [])))))
        except Exception as e:
            logger.warning(f"SHAP explainer not loaded: {e}")
            explainer = None

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PipelineFeatures(BaseModel):
    # Numerical features (21 total — matches processed dataset)
    duration_seconds        : float = Field(default=300.0)
    retry_count             : int   = Field(default=0)
    total_jobs              : int   = Field(default=10)
    failed_jobs             : int   = Field(default=0)
    queue_time_seconds      : float = Field(default=30.0)
    avg_duration_7_runs     : float = Field(default=300.0)
    duration_deviation      : float = Field(default=0.0)
    failures_last_7_runs    : int   = Field(default=0)
    hour                    : int   = Field(default=12)
    day_of_week             : int   = Field(default=1)
    is_weekend              : int   = Field(default=0)
    is_bot_triggered        : int   = Field(default=0)
    is_main_branch          : int   = Field(default=1)
    pipeline_name_encoded   : float = Field(default=0.0)
    repo_encoded            : float = Field(default=0.0)
    trigger_type_encoded    : float = Field(default=0.0)
    head_branch_encoded     : float = Field(default=0.0)
    failure_rate_7_runs     : float = Field(default=0.0)
    success_streak          : int   = Field(default=5)
    avg_queue_time_7_runs   : float = Field(default=30.0)
    concurrency_level       : int   = Field(default=1)

class PredictionResponse(BaseModel):
    probability     : float
    prediction      : int
    risk_level      : str
    top_shap_features: list
    timestamp       : str
    model_name      : str

# ---------------------------------------------------------------------------
# Helper: build SHAP top features
# ---------------------------------------------------------------------------
def get_shap_features(input_array: np.ndarray) -> list:
    if explainer is None or feature_names is None:
        return []
    try:
        shap_vals = explainer.shap_values(input_array)
        # For binary classifiers shap_values may return list[2]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        vals   = shap_vals[0]
        top_idx = np.argsort(np.abs(vals))[::-1][:5]
        return [
            {
                "feature"   : feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "shap_value": round(float(vals[i]), 4),
                "direction" : "increases_risk" if vals[i] > 0 else "reduces_risk",
            }
            for i in top_idx
        ]
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return []

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status"      : "healthy" if model is not None else "model_not_loaded",
        "model_name"  : metadata.get("model_name", "unknown"),
        "trained_at"  : metadata.get("trained_at", "unknown"),
        "auc_roc"     : metadata.get("test_metrics", {}).get("auc_roc", "unknown"),
        "version"     : metadata.get("test_metrics", {}).get("model", "v1"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PipelineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build input DataFrame in correct column order
    input_dict = features.dict()
    if feature_names:
        # Align to trained feature order, fill missing with 0
        row = {f: input_dict.get(f, 0.0) for f in feature_names}
        input_df = pd.DataFrame([row])[feature_names]
    else:
        input_df = pd.DataFrame([input_dict])

    input_arr  = scaler.transform(input_df)
    probability = float(model.predict_proba(input_arr)[0][1])

    # Threshold = 0.75 (optimized for precision/recall balance)
    # At t=0.75: Precision=0.9036, Recall=0.9265, F1=0.9149
    DECISION_THRESHOLD = 0.75
    prediction = int(probability >= DECISION_THRESHOLD)

    if probability >= 0.75:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    top_shap = get_shap_features(input_arr)

    result = {
        "probability"      : round(probability, 4),
        "prediction"       : prediction,
        "risk_level"       : risk_level,
        "top_shap_features": top_shap,
        "timestamp"        : datetime.utcnow().isoformat(),
        "model_name"       : metadata.get("model_name", "unknown"),
    }

    # Store in rolling history
    pred_history.append({
        "probability": probability,
        "prediction" : prediction,
        "risk_level" : risk_level,
        "timestamp"  : result["timestamp"],
    })

    logger.info(f"Prediction: prob={probability:.4f} risk={risk_level}")
    return result


class ExplainRequest(BaseModel):
    pipeline_features : dict  = Field(default={})
    failure_prob      : float = Field(default=0.5)
    user_message      : str   = Field(default="Give me a full diagnosis of this pipeline run.")
    chat_history      : list  = Field(default=[])

@app.post("/explain")
def explain(request: ExplainRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        from app.rag_chatbot import get_diagnosis
        response = get_diagnosis(
            pipeline_features=request.pipeline_features,
            failure_prob=request.failure_prob,
            user_message=request.user_message,
            chat_history=request.chat_history,
        )
        return {"diagnosis": response, "status": "success"}
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return {"diagnosis": f"RAG error: {str(e)}", "status": "error"}


@app.get("/metrics")
def metrics():
    if not pred_history:
        return {"message": "No predictions yet", "count": 0}
    probs  = [p["probability"] for p in pred_history]
    preds  = [p["prediction"]  for p in pred_history]
    risks  = [p["risk_level"]  for p in pred_history]
    return {
        "total_predictions"  : len(pred_history),
        "avg_probability"    : round(float(np.mean(probs)), 4),
        "high_risk_count"    : risks.count("HIGH"),
        "medium_risk_count"  : risks.count("MEDIUM"),
        "low_risk_count"     : risks.count("LOW"),
        "predicted_failures" : sum(preds),
        "model_auc_roc"      : metadata.get("test_metrics", {}).get("auc_roc", "unknown"),
        "window"             : "last 100 predictions",
    }