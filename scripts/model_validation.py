"""
model_validation.py
-------------------
Validates the XGBoost pipeline failure prediction model.

Tasks:
1. Hold-out evaluation  — accuracy, precision, recall, F1, AUC-ROC, AUC-PR
2. Confusion matrix      — heatmap PNG + classification report JSON
3. Threshold analysis    — vary 0.1-0.9, find optimal F1, generate plot
4. Validation gate       — pass/fail logic (AUC-ROC must exceed 0.85)
5. Rollback mechanism    — compare new vs previous best model, reject if worse

Author  : Member 3 (Data Scientist)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : February 2026
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.config import (
    LOGGING_CONFIG,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    ensure_directories_exist,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["log_level"], logging.INFO),
    format=LOGGING_CONFIG["log_format"],
    datefmt=LOGGING_CONFIG["date_format"],
)
logger = logging.getLogger("model_validation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DATASET  = PROCESSED_DATA_DIR / "processed_dataset.csv"
MODEL_DIR          = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH         = MODEL_DIR / "best_model.joblib"
PREV_MODEL_PATH    = MODEL_DIR / "previous_model.joblib"
PREV_METRICS_PATH  = MODEL_DIR / "previous_metrics.json"

VALIDATION_REPORT  = REPORTS_DIR / "validation_report.json"
CONFUSION_MATRIX   = REPORTS_DIR / "confusion_matrix.png"
THRESHOLD_PLOT     = REPORTS_DIR / "threshold_analysis.png"

MIN_AUC_THRESHOLD  = 0.85   # Validation gate — model must exceed this
TARGET_COLUMN      = "failed"

# Columns to drop before training (non-feature columns)
DROP_COLUMNS = [
    "run_id", "trigger_time",
    "failure_type", "error_message",
    "pipeline_name", "repo", "head_branch",
    "trigger_type",
]


# ---------------------------------------------------------------------------
# 1. Data Loading & Model Training
# ---------------------------------------------------------------------------

def load_and_split_data():
    """
    Load processed dataset and split into train/test (85/15).

    Returns
    -------
    X_train, X_test, y_train, y_test : split DataFrames/Series
    """
    logger.info("Loading processed dataset from: %s", PROCESSED_DATASET)

    if not PROCESSED_DATASET.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_DATASET}")

    df = pd.read_csv(PROCESSED_DATASET)
    logger.info("Dataset loaded: %d rows x %d columns", len(df), df.shape[1])

    # Drop non-feature columns that exist in this dataset
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop + [TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    logger.info("Features used for training: %d", X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    logger.info(
        "Split — Train: %d | Test: %d | Failure rate in test: %.2f%%",
        len(X_train), len(X_test), y_test.mean() * 100
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> XGBClassifier:
    """
    Train XGBoost classifier on training data.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    XGBClassifier — trained model
    """
    logger.info("Training XGBoost model...")

    # Handle class imbalance with scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    logger.info("Class imbalance ratio: %.2f (scale_pos_weight=%.2f)", scale, scale)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost model training complete.")
    return model


def save_model(model: XGBClassifier) -> None:
    """Save trained model to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # If a best model already exists, move it to previous
    if MODEL_PATH.exists():
        logger.info("Moving existing model to previous_model.joblib")
        prev_model = joblib.load(MODEL_PATH)
        joblib.dump(prev_model, PREV_MODEL_PATH)

    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved to: %s", MODEL_PATH)


def load_model() -> XGBClassifier:
    """Load trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# 2. Hold-out Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model on hold-out test set (15%).

    Computes: accuracy, precision, recall, F1, AUC-ROC, AUC-PR

    Parameters
    ----------
    model   : trained XGBClassifier
    X_test  : pd.DataFrame
    y_test  : pd.Series

    Returns
    -------
    dict — evaluation metrics
    """
    logger.info("--- Hold-out Evaluation (15%% test set) ---")

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Core metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc_roc   = roc_auc_score(y_test, y_pred_proba)

    # AUC-PR
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(r_curve, p_curve)

    metrics = {
        "accuracy"  : round(float(accuracy),  4),
        "precision" : round(float(precision), 4),
        "recall"    : round(float(recall),    4),
        "f1_score"  : round(float(f1),        4),
        "auc_roc"   : round(float(auc_roc),   4),
        "auc_pr"    : round(float(auc_pr),    4),
        "test_size" : len(y_test),
        "n_failures": int(y_test.sum()),
    }

    logger.info("Accuracy  : %.4f", accuracy)
    logger.info("Precision : %.4f", precision)
    logger.info("Recall    : %.4f", recall)
    logger.info("F1 Score  : %.4f", f1)
    logger.info("AUC-ROC   : %.4f", auc_roc)
    logger.info("AUC-PR    : %.4f", auc_pr)

    return metrics, y_pred, y_pred_proba


# ---------------------------------------------------------------------------
# 3. Confusion Matrix & Classification Report
# ---------------------------------------------------------------------------

def generate_confusion_matrix(y_test, y_pred) -> None:
    """
    Generate and save confusion matrix heatmap as PNG.

    Parameters
    ----------
    y_test : pd.Series — true labels
    y_pred : np.ndarray — predicted labels
    """
    logger.info("--- Generating Confusion Matrix ---")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure (0)", "Failure (1)"],
        yticklabels=["No Failure (0)", "Failure (1)"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Pipeline Failure Prediction", fontsize=14, pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Add metric annotations
    tn, fp, fn, tp = cm.ravel()
    ax.text(
        0.5, -0.12,
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        transform=ax.transAxes,
        ha="center", fontsize=10, color="gray"
    )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to: %s", CONFUSION_MATRIX)


def generate_classification_report(y_test, y_pred) -> dict:
    """
    Generate classification report as dictionary.

    Parameters
    ----------
    y_test : pd.Series
    y_pred : np.ndarray

    Returns
    -------
    dict — classification report
    """
    report = classification_report(
        y_test, y_pred,
        target_names=["No Failure", "Failure"],
        output_dict=True,
        zero_division=0,
    )
    logger.info("Classification report generated.")
    return report


# ---------------------------------------------------------------------------
# 4. Threshold Analysis
# ---------------------------------------------------------------------------

def threshold_analysis(y_test, y_pred_proba) -> dict:
    """
    Vary decision threshold from 0.1 to 0.9 and compute metrics at each threshold.
    Finds the optimal threshold for F1 score.
    Generates and saves threshold-vs-metric plot.

    Parameters
    ----------
    y_test       : pd.Series — true labels
    y_pred_proba : np.ndarray — predicted probabilities

    Returns
    -------
    dict — threshold analysis results including optimal threshold
    """
    logger.info("--- Threshold Analysis (0.1 to 0.9) ---")

    thresholds = np.arange(0.1, 1.0, 0.1).round(1)
    results = []

    for thresh in thresholds:
        y_pred_t   = (y_pred_proba >= thresh).astype(int)
        precision  = precision_score(y_test, y_pred_t, zero_division=0)
        recall     = recall_score(y_test, y_pred_t, zero_division=0)
        f1         = f1_score(y_test, y_pred_t, zero_division=0)
        accuracy   = accuracy_score(y_test, y_pred_t)

        results.append({
            "threshold" : round(float(thresh), 1),
            "precision" : round(float(precision), 4),
            "recall"    : round(float(recall),    4),
            "f1_score"  : round(float(f1),        4),
            "accuracy"  : round(float(accuracy),  4),
        })
        logger.info(
            "Threshold %.1f — Precision: %.4f | Recall: %.4f | F1: %.4f",
            thresh, precision, recall, f1
        )

    # Find optimal threshold (max F1)
    best = max(results, key=lambda x: x["f1_score"])
    logger.info(
        "Optimal threshold: %.1f (F1=%.4f)", best["threshold"], best["f1_score"]
    )

    # ── Plot ────────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: metrics vs threshold
    ax1.plot(df_results["threshold"], df_results["f1_score"],
             "b-o", label="F1 Score", linewidth=2)
    ax1.plot(df_results["threshold"], df_results["precision"],
             "g-s", label="Precision", linewidth=2)
    ax1.plot(df_results["threshold"], df_results["recall"],
             "r-^", label="Recall", linewidth=2)
    ax1.plot(df_results["threshold"], df_results["accuracy"],
             "m-d", label="Accuracy", linewidth=2)
    ax1.axvline(x=best["threshold"], color="orange", linestyle="--",
                linewidth=2, label=f"Optimal={best['threshold']}")
    ax1.set_xlabel("Decision Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Metrics vs Decision Threshold", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(thresholds)

    # Right: F1 bar chart per threshold
    colors = ["orange" if t == best["threshold"] else "steelblue"
              for t in df_results["threshold"]]
    ax2.bar(df_results["threshold"].astype(str), df_results["f1_score"],
            color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Decision Threshold", fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("F1 Score per Threshold (orange = optimal)", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Threshold Analysis — Pipeline Failure Prediction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(THRESHOLD_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Threshold analysis plot saved to: %s", THRESHOLD_PLOT)

    return {
        "thresholds"         : results,
        "optimal_threshold"  : best["threshold"],
        "optimal_f1"         : best["f1_score"],
        "optimal_precision"  : best["precision"],
        "optimal_recall"     : best["recall"],
    }


# ---------------------------------------------------------------------------
# 5. Validation Gate
# ---------------------------------------------------------------------------

def validation_gate(auc_roc: float) -> dict:
    """
    Pass/fail logic: model must exceed MIN_AUC_THRESHOLD (0.85) to proceed.
    Blocks deployment if AUC-ROC is below threshold.

    Parameters
    ----------
    auc_roc : float — model's AUC-ROC score

    Returns
    -------
    dict — gate result with status and message
    """
    logger.info("--- Validation Gate ---")
    logger.info("AUC-ROC: %.4f | Minimum required: %.2f", auc_roc, MIN_AUC_THRESHOLD)

    passed = auc_roc >= MIN_AUC_THRESHOLD

    if passed:
        status  = "PASSED"
        message = (
            f"Model PASSED validation gate. "
            f"AUC-ROC={auc_roc:.4f} >= threshold={MIN_AUC_THRESHOLD}. "
            f"Deployment approved."
        )
        logger.info("VALIDATION GATE: PASSED ✓")
    else:
        status  = "FAILED"
        message = (
            f"Model FAILED validation gate. "
            f"AUC-ROC={auc_roc:.4f} < threshold={MIN_AUC_THRESHOLD}. "
            f"Deployment BLOCKED."
        )
        logger.warning("VALIDATION GATE: FAILED ✗ — Deployment blocked!")

    return {
        "status"        : status,
        "auc_roc"       : round(auc_roc, 4),
        "min_threshold" : MIN_AUC_THRESHOLD,
        "passed"        : passed,
        "message"       : message,
    }


# ---------------------------------------------------------------------------
# 6. Rollback Mechanism
# ---------------------------------------------------------------------------

def rollback_check(current_auc: float) -> dict:
    """
    Compare new model AUC vs previous best model.
    If new model is worse, reject it and keep previous model.

    Parameters
    ----------
    current_auc : float — new model's AUC-ROC

    Returns
    -------
    dict — rollback decision
    """
    logger.info("--- Rollback Mechanism ---")

    # No previous model exists — first run
    if not PREV_METRICS_PATH.exists():
        logger.info("No previous model found. Current model accepted as baseline.")
        return {
            "decision"       : "ACCEPTED",
            "reason"         : "No previous model exists. Current model set as baseline.",
            "current_auc"    : round(current_auc, 4),
            "previous_auc"   : None,
            "rollback_needed": False,
        }

    # Load previous metrics
    with open(PREV_METRICS_PATH) as f:
        prev_metrics = json.load(f)

    prev_auc = prev_metrics.get("auc_roc", 0.0)
    logger.info("Current AUC:  %.4f", current_auc)
    logger.info("Previous AUC: %.4f", prev_auc)

    if current_auc >= prev_auc:
        decision = "ACCEPTED"
        reason   = (
            f"New model (AUC={current_auc:.4f}) is better than or equal to "
            f"previous (AUC={prev_auc:.4f}). Keeping new model."
        )
        logger.info("ROLLBACK: Not needed. New model accepted.")
    else:
        decision = "ROLLED BACK"
        reason   = (
            f"New model (AUC={current_auc:.4f}) is WORSE than "
            f"previous (AUC={prev_auc:.4f}). Rolling back to previous model."
        )
        logger.warning("ROLLBACK: New model rejected! Restoring previous model.")

        # Restore previous model
        if PREV_MODEL_PATH.exists():
            prev_model = joblib.load(PREV_MODEL_PATH)
            joblib.dump(prev_model, MODEL_PATH)
            logger.info("Previous model restored successfully.")

    return {
        "decision"       : decision,
        "reason"         : reason,
        "current_auc"    : round(current_auc, 4),
        "previous_auc"   : round(prev_auc, 4),
        "rollback_needed": decision == "ROLLED BACK",
    }


# ---------------------------------------------------------------------------
# Save Validation Report
# ---------------------------------------------------------------------------

def save_validation_report(
    metrics: dict,
    classification_rep: dict,
    threshold_results: dict,
    gate_result: dict,
    rollback_result: dict,
) -> Path:
    """
    Save full validation report as JSON.

    Returns
    -------
    Path — path to saved report
    """
    report = {
        "model"               : "XGBoost",
        "dataset"             : str(PROCESSED_DATASET),
        "test_size_percent"   : 15,
        "hold_out_metrics"    : metrics,
        "classification_report": classification_rep,
        "threshold_analysis"  : threshold_results,
        "validation_gate"     : gate_result,
        "rollback"            : rollback_result,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report saved to: %s", VALIDATION_REPORT)
    return VALIDATION_REPORT


# ---------------------------------------------------------------------------
# Save metrics for future rollback comparison
# ---------------------------------------------------------------------------

def save_current_metrics(metrics: dict) -> None:
    """Save current model metrics for future rollback comparison."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREV_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Current metrics saved for future rollback comparison.")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_model_validation() -> dict:
    """
    Full model validation pipeline:
    1. Load data and train model
    2. Hold-out evaluation
    3. Confusion matrix + classification report
    4. Threshold analysis
    5. Validation gate
    6. Rollback check
    7. Save all outputs
    """
    logger.info("=" * 60)
    logger.info("MODEL VALIDATION START")
    logger.info("=" * 60)

    ensure_directories_exist()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and train
    X_train, X_test, y_train, y_test = load_and_split_data()
    model = train_model(X_train, y_train)
    save_model(model)

    # 2. Hold-out evaluation
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

    # 3. Confusion matrix + classification report
    generate_confusion_matrix(y_test, y_pred)
    class_report = generate_classification_report(y_test, y_pred)

    # 4. Threshold analysis
    threshold_results = threshold_analysis(y_test, y_pred_proba)

    # 5. Validation gate
    gate_result = validation_gate(metrics["auc_roc"])

    # 6. Rollback check
    rollback_result = rollback_check(metrics["auc_roc"])

    # 7. Save report + metrics
    save_validation_report(
        metrics, class_report, threshold_results, gate_result, rollback_result
    )
    save_current_metrics(metrics)

    # Final summary
    logger.info("=" * 60)
    logger.info("MODEL VALIDATION COMPLETE")
    logger.info("AUC-ROC         : %.4f", metrics["auc_roc"])
    logger.info("F1 Score        : %.4f", metrics["f1_score"])
    logger.info("Optimal Threshold: %.1f", threshold_results["optimal_threshold"])
    logger.info("Validation Gate : %s", gate_result["status"])
    logger.info("Rollback        : %s", rollback_result["decision"])
    logger.info("=" * 60)

    return {
        "metrics"          : metrics,
        "gate_result"      : gate_result,
        "rollback_result"  : rollback_result,
        "threshold_results": threshold_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_model_validation()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy         : {results['metrics']['accuracy']:.4f}")
    print(f"Precision        : {results['metrics']['precision']:.4f}")
    print(f"Recall           : {results['metrics']['recall']:.4f}")
    print(f"F1 Score         : {results['metrics']['f1_score']:.4f}")
    print(f"AUC-ROC          : {results['metrics']['auc_roc']:.4f}")
    print(f"AUC-PR           : {results['metrics']['auc_pr']:.4f}")
    print(f"Optimal Threshold: {results['threshold_results']['optimal_threshold']}")
    print(f"Validation Gate  : {results['gate_result']['status']}")
    print(f"Rollback         : {results['rollback_result']['decision']}")
    print("=" * 60)
