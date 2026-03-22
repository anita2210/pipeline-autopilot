"""
experiment_tracking.py
======================
MLflow experiment tracking, logging, model versioning, and results visualization.

Author: Member 2 (MLOps Engineer)
Created: March 2026
Project: Pipeline Autopilot - CI/CD Failure Prediction System

Responsibilities:
    1. Configure MLflow tracking server (local file-based)
    2. Log experiments: params, metrics, artifacts (confusion matrix, model)
    3. Register best model in MLflow Model Registry (Staging -> Production)
    4. Generate comparison visualizations across models
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import joblib

# Add parent directory to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.config import (
    BASE_DIR, PROCESSED_DATA_FILE, REPORTS_DIR,
    TARGET_COLUMN, LOGGING_CONFIG
)

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

# MLflow settings
MLFLOW_TRACKING_URI = f"file:///{str(BASE_DIR / 'mlruns').replace(os.sep, '/')}"
EXPERIMENT_NAME = "pipelineguard-model-dev"

# Model directories
MODELS_DIR = BASE_DIR / "models"
TRAINED_DIR = MODELS_DIR / "trained"
REGISTRY_DIR = MODELS_DIR / "registry"
SENSITIVITY_DIR = MODELS_DIR / "sensitivity"

# Validation gate thresholds
MIN_AUC_THRESHOLD = 0.85
MIN_F1_THRESHOLD = 0.50

# Logger setup
logger = logging.getLogger("experiment_tracking")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        LOGGING_CONFIG.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ============================================================================
# 1. MLFLOW SETUP
# ============================================================================

def setup_mlflow():
    """
    Configure MLflow tracking URI and create/set experiment.

    Returns:
        str: The experiment ID for 'pipelineguard-model-dev'.
    """
    logger.info("Setting up MLflow tracking...")

    # Ensure mlruns directory exists
    mlruns_dir = BASE_DIR / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Set tracking URI (local file-based)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        # Use file:/// URI for artifact location on Windows
        artifact_loc = mlruns_dir / EXPERIMENT_NAME
        artifact_uri = artifact_loc.as_uri() if hasattr(artifact_loc, 'as_uri') else \
            "file:///" + str(artifact_loc).replace("\\", "/")
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_uri
        )
        logger.info(f"Created new experiment '{EXPERIMENT_NAME}' (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment '{EXPERIMENT_NAME}' (ID: {experiment_id})")

    mlflow.set_experiment(EXPERIMENT_NAME)

    return experiment_id


# ============================================================================
# 2. EXPERIMENT LOGGING
# ============================================================================

def log_experiment(model_name, model, X_train, y_train, X_test, y_test,
                   params, metrics=None, extra_artifacts=None):
    """
    Log a complete ML experiment to MLflow.

    Args:
        model_name (str): Name of the model (e.g., 'LogisticRegression', 'XGBoost').
        model (object): Trained sklearn/xgboost model object.
        X_train (pd.DataFrame): Training features (for signature inference).
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features (for evaluation).
        y_test (pd.Series): Test labels.
        params (dict): Hyperparameters to log.
        metrics (dict, optional): Pre-computed metrics. If None, computed automatically.
        extra_artifacts (dict, optional): Additional artifacts {name: filepath} to log.

    Returns:
        dict: Run info with 'run_id', 'metrics', and 'model_name'.
    """
    logger.info(f"Logging experiment for model: {model_name}")

    # Compute predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    # Compute metrics if not provided
    if metrics is None:
        metrics = compute_metrics(y_test, y_pred, y_prob)

    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # --- Log parameters ---
        mlflow.log_params(params)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset_size", len(X_train) + len(X_test))
        mlflow.set_tag("n_features", X_train.shape[1])
        mlflow.set_tag("target", TARGET_COLUMN)

        # --- Log metrics ---
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, round(metric_value, 4))

        # --- Log confusion matrix as artifact ---
        cm_path = _save_confusion_matrix(y_test, y_pred, model_name, run_id)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # --- Log ROC curve if probabilities available ---
        if y_prob is not None:
            roc_path = _save_roc_curve(y_test, y_prob, model_name, run_id)
            mlflow.log_artifact(roc_path, artifact_path="plots")

        # --- Log classification report as JSON ---
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = REGISTRY_DIR / f"classification_report_{model_name}_{run_id[:8]}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # --- Log the model ---
        try:
            signature = infer_signature(X_train, model.predict(X_train))
        except Exception:
            signature = None

        if "xgb" in model_name.lower() or "xgboost" in model_name.lower():
            mlflow.xgboost.log_model(model, artifact_path="model", signature=signature)
        else:
            mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

        # --- Save model locally as .joblib ---
        joblib_path = TRAINED_DIR / f"{model_name}_best.joblib"
        joblib_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, joblib_path)
        mlflow.log_artifact(str(joblib_path), artifact_path="model_artifacts")
        logger.info(f"Model saved to {joblib_path}")

        # --- Log any extra artifacts ---
        if extra_artifacts:
            for name, filepath in extra_artifacts.items():
                if Path(filepath).exists():
                    mlflow.log_artifact(filepath, artifact_path=name)

        logger.info(f"Experiment logged successfully | AUC: {metrics.get('auc', 'N/A')} | "
                     f"F1: {metrics.get('f1', 'N/A')}")

    return {
        "run_id": run_id,
        "model_name": model_name,
        "metrics": metrics,
    }


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional).

    Returns:
        dict: Dictionary of metric_name -> value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auc"] = None
        metrics["avg_precision"] = None

    return metrics


# ============================================================================
# 3. MODEL VERSIONING & REGISTRY
# ============================================================================

def register_best_model(experiment_id, metric="auc"):
    """
    Find the best model across all runs and register it in MLflow Model Registry.
    Transitions: None -> Staging -> Production.

    Args:
        experiment_id (str): MLflow experiment ID.
        metric (str): Metric to compare (default: 'auc').

    Returns:
        dict: Best model info with 'run_id', 'model_name', 'metric_value', 'version'.
    """
    logger.info(f"Finding best model by '{metric}' in experiment {experiment_id}...")

    client = MlflowClient()

    # Query all completed runs sorted by metric (descending)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=50,
        filter_string="status = 'FINISHED'"
    )

    if not runs:
        logger.warning("No completed runs found in experiment.")
        return None

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_metric = best_run.data.metrics.get(metric, 0)
    best_model_name = best_run.data.tags.get("model_name", "unknown")

    logger.info(f"Best model: {best_model_name} | Run: {best_run_id} | "
                f"{metric}: {best_metric:.4f}")

    # Validation gate check
    if metric == "auc" and best_metric < MIN_AUC_THRESHOLD:
        logger.warning(f"Best AUC ({best_metric:.4f}) is below threshold "
                       f"({MIN_AUC_THRESHOLD}). Skipping registration.")
        return {
            "run_id": best_run_id,
            "model_name": best_model_name,
            "metric_value": best_metric,
            "version": None,
            "status": "FAILED_VALIDATION",
        }

    # Register the model
    registry_name = "PipelineGuard-BestModel"
    model_uri = f"runs:/{best_run_id}/model"

    try:
        # Register model (creates a new version automatically)
        result = mlflow.register_model(model_uri, registry_name)
        version = result.version
        logger.info(f"Registered model '{registry_name}' version {version}")

        # Transition to Staging
        client.transition_model_version_stage(
            name=registry_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )
        logger.info(f"Model version {version} transitioned to Staging")

        # Transition to Production (archive previous production models)
        client.transition_model_version_stage(
            name=registry_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Model version {version} transitioned to Production")

        # Save registry metadata locally
        metadata = {
            "registry_name": registry_name,
            "version": version,
            "run_id": best_run_id,
            "model_name": best_model_name,
            "metric": metric,
            "metric_value": round(best_metric, 4),
            "stage": "Production",
            "registered_at": datetime.now().isoformat(),
        }
        metadata_path = REGISTRY_DIR / "best_model_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Registry metadata saved to {metadata_path}")

        return {
            "run_id": best_run_id,
            "model_name": best_model_name,
            "metric_value": best_metric,
            "version": version,
            "status": "PRODUCTION",
        }

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return {
            "run_id": best_run_id,
            "model_name": best_model_name,
            "metric_value": best_metric,
            "version": None,
            "status": f"ERROR: {str(e)}",
        }


def compare_with_previous_best(experiment_id, new_run_id, metric="auc"):
    """
    Compare a new model run against the current production model.
    Implements rollback: if new model is worse, keep the old one.

    Args:
        experiment_id (str): MLflow experiment ID.
        new_run_id (str): Run ID of the newly trained model.
        metric (str): Metric to compare.

    Returns:
        dict: Comparison result with 'action' ('DEPLOY' or 'ROLLBACK').
    """
    client = MlflowClient()

    # Get new model metrics
    new_run = client.get_run(new_run_id)
    new_metric = new_run.data.metrics.get(metric, 0)

    # Get current production model
    registry_name = "PipelineGuard-BestModel"
    try:
        versions = client.get_latest_versions(registry_name, stages=["Production"])
        if not versions:
            logger.info("No production model found. New model will be deployed.")
            return {"action": "DEPLOY", "reason": "No existing production model"}

        prod_version = versions[0]
        prod_run = client.get_run(prod_version.run_id)
        prod_metric = prod_run.data.metrics.get(metric, 0)

        logger.info(f"Current production {metric}: {prod_metric:.4f} | "
                     f"New model {metric}: {new_metric:.4f}")

        if new_metric > prod_metric:
            logger.info("New model is better. Proceeding with deployment.")
            return {
                "action": "DEPLOY",
                "reason": f"New {metric} ({new_metric:.4f}) > Production ({prod_metric:.4f})",
            }
        else:
            logger.info("New model is NOT better. Rolling back to existing production model.")
            return {
                "action": "ROLLBACK",
                "reason": f"New {metric} ({new_metric:.4f}) <= Production ({prod_metric:.4f})",
            }

    except Exception as e:
        logger.warning(f"Could not compare with production model: {e}")
        return {"action": "DEPLOY", "reason": f"Comparison failed: {str(e)}"}


# ============================================================================
# 4. VISUALIZATION & COMPARISON PLOTS
# ============================================================================

def generate_comparison_plots(experiment_id):
    """
    Generate bar plots comparing metrics across all logged models.
    Saves plots to models/registry/.

    Args:
        experiment_id (str): MLflow experiment ID.

    Returns:
        list: Paths to saved plot files.
    """
    logger.info("Generating model comparison plots...")

    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.auc DESC"],
    )

    if not runs:
        logger.warning("No completed runs found. Cannot generate plots.")
        return []

    # Collect metrics from all runs
    records = []
    for run in runs:
        model_name = run.data.tags.get("model_name", "Unknown")
        record = {
            "model": model_name,
            "run_id": run.info.run_id[:8],
            "auc": run.data.metrics.get("auc", 0),
            "f1": run.data.metrics.get("f1", 0),
            "precision": run.data.metrics.get("precision", 0),
            "recall": run.data.metrics.get("recall", 0),
            "accuracy": run.data.metrics.get("accuracy", 0),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # If multiple runs per model, keep the best AUC per model
    df_best = df.sort_values("auc", ascending=False).drop_duplicates(subset=["model"], keep="first")

    saved_plots = []
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: AUC Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df_best))
    bars = ax.barh(df_best["model"], df_best["auc"], color=colors)
    ax.axvline(x=MIN_AUC_THRESHOLD, color="red", linestyle="--", label=f"Threshold ({MIN_AUC_THRESHOLD})")
    ax.set_xlabel("AUC-ROC Score", fontsize=12)
    ax.set_title("Model Comparison: AUC-ROC", fontsize=14, fontweight="bold")
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, df_best["auc"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    auc_plot_path = REGISTRY_DIR / "model_comparison_auc.png"
    fig.savefig(auc_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(str(auc_plot_path))
    logger.info(f"AUC comparison plot saved: {auc_plot_path}")

    # --- Plot 2: Multi-Metric Grouped Bar Chart ---
    metric_cols = ["auc", "f1", "precision", "recall", "accuracy"]
    df_melted = df_best.melt(id_vars=["model"], value_vars=metric_cols,
                              var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="model", ax=ax, palette="Set2")
    ax.set_title("Model Comparison: All Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    multi_plot_path = REGISTRY_DIR / "model_comparison_all_metrics.png"
    fig.savefig(multi_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(str(multi_plot_path))
    logger.info(f"Multi-metric comparison plot saved: {multi_plot_path}")

    # --- Plot 3: Metrics Summary Table as Image ---
    fig, ax = plt.subplots(figsize=(10, max(2, len(df_best) + 1)))
    ax.axis("off")
    table_data = df_best[["model"] + metric_cols].round(4)
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Model Metrics Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    table_path = REGISTRY_DIR / "model_metrics_summary_table.png"
    fig.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(str(table_path))
    logger.info(f"Metrics summary table saved: {table_path}")

    logger.info(f"Generated {len(saved_plots)} comparison plots.")
    return saved_plots


# ============================================================================
# HELPER FUNCTIONS (PRIVATE)
# ============================================================================

def _save_confusion_matrix(y_true, y_pred, model_name, run_id):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Failed (0)", "Failed (1)"],
                yticklabels=["Not Failed (0)", "Failed (1)"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = REGISTRY_DIR / f"confusion_matrix_{model_name}_{run_id[:8]}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _save_roc_curve(y_true, y_prob, model_name, run_id):
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    save_path = REGISTRY_DIR / f"roc_curve_{model_name}_{run_id[:8]}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


# ============================================================================
# MASTER FUNCTION
# ============================================================================

def run_experiment_tracking(models_dict, X_train, y_train, X_test, y_test):
    """
    Master function to run full experiment tracking pipeline.

    Args:
        models_dict (dict): {model_name: {'model': trained_model, 'params': params_dict}}
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        dict: Summary with experiment_id, all run results, and best model info.
    """
    logger.info("=" * 60)
    logger.info("STARTING EXPERIMENT TRACKING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Setup MLflow
    experiment_id = setup_mlflow()

    # Step 2: Log each model
    all_results = []
    for model_name, model_info in models_dict.items():
        result = log_experiment(
            model_name=model_name,
            model=model_info["model"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=model_info["params"],
        )
        all_results.append(result)

    # Step 3: Register best model
    best_model_info = register_best_model(experiment_id, metric="auc")

    # Step 4: Generate comparison plots
    plot_paths = generate_comparison_plots(experiment_id)

    # Summary
    summary = {
        "experiment_id": experiment_id,
        "total_models_logged": len(all_results),
        "all_results": all_results,
        "best_model": best_model_info,
        "comparison_plots": plot_paths,
    }

    logger.info("=" * 60)
    logger.info("EXPERIMENT TRACKING COMPLETE")
    if best_model_info:
        logger.info(f"Best Model: {best_model_info.get('model_name')} | "
                     f"AUC: {best_model_info.get('metric_value', 'N/A')}")
    logger.info("=" * 60)

    return summary


# ============================================================================
# MAIN (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    """Quick test: creates a dummy model and logs it to MLflow."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("EXPERIMENT TRACKING - STANDALONE TEST")
    print("=" * 60)

    # Ensure directories exist
    for d in [TRAINED_DIR, REGISTRY_DIR, SENSITIVITY_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy dataset
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_classes=2, weights=[0.88, 0.12],
                               random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y, name="failed")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Train dummy model
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)

    # Run full pipeline
    models_dict = {
        "LogisticRegression": {
            "model": lr,
            "params": {"max_iter": 500, "solver": "lbfgs", "C": 1.0},
        }
    }

    summary = run_experiment_tracking(models_dict, X_train, y_train, X_test, y_test)

    print("\n--- Summary ---")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "all_results"}, indent=2, default=str))
    print("\nRun 'mlflow ui --port 5001' and open http://localhost:5001 to view experiments.")