import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MLRUNS_DIR = BASE_DIR / "mlruns"
MODELS_DIR = BASE_DIR / "models"

EXPERIMENT_NAME = "pipelineguard-model-dev"
TRACKING_URI    = "file:///opt/airflow/mlruns"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow() -> str:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        logger.info(f"Created MLflow experiment '{EXPERIMENT_NAME}'")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{EXPERIMENT_NAME}'")
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def log_experiment(run_name, model, params, metrics, y_true, y_pred, y_prob, model_type="sklearn", extra_artifacts=None):
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
        cm_path = _save_confusion_matrix(y_true, y_pred, run_name)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        metrics_path = MODELS_DIR / "registry" / f"{run_name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"run_id": run_id, "run_name": run_name, **metrics, **params}, f, indent=2)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        logger.info(f"[MLflow] Run '{run_name}' logged — run_id={run_id}, AUC={metrics.get('auc_roc', 0):.4f}")
        return run_id


def _save_confusion_matrix(y_true, y_pred, label):
    out_dir = MODELS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = str(out_dir / f"cm_{label}.png")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["pass", "fail"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_auc_comparison_chart(results):
    out_dir = MODELS_DIR / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = str(out_dir / "auc_comparison.png")
    names  = list(results.keys())
    scores = [results[n] for n in names]
    colors = ["#4CAF50" if s == max(scores) else "#90CAF9" for s in scores]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, scores, color=colors, edgecolor="white")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Model AUC-ROC Comparison")
    ax.axhline(0.85, color="red", linestyle="--", linewidth=1, label="Min threshold 0.85")
    ax.legend()
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005, f"{score:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"AUC comparison chart saved -> {path}")
    return path


def register_best_model(run_id, model_name="pipelineguard-xgboost"):
    setup_mlflow()
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    result  = mlflow.register_model(model_uri, model_name)
    version = result.version
    logger.info(f"[MLflow Registry] '{model_name}' v{version} registered successfully")
    return version


if __name__ == "__main__":
    eid = setup_mlflow()
    logger.info(f"MLflow ready. Experiment ID: {eid}")
