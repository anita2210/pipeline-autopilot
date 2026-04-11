"""
performance_monitor.py
----------------------
Logs every prediction made by the model to a rolling CSV log. Computes
rolling AUC and F1 every 100 predictions. Flags if AUC drops below the
0.85 threshold and triggers an alert. Outputs performance status to
Airflow XCom when called from the DAG.

Author  : Member 4 (MLOps Monitor)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import LOGGING_CONFIG

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get("log_level", "INFO"), logging.INFO),
    format=LOGGING_CONFIG.get("log_format", "%(asctime)s %(levelname)s %(message)s"),
    datefmt=LOGGING_CONFIG.get("date_format", "%Y-%m-%d %H:%M:%S"),
)
logger = logging.getLogger("performance_monitor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PREDICTIONS_LOG_PATH = Path("data/monitoring/predictions_log.csv")
PERFORMANCE_REPORT_PATH = Path("data/reports/performance/performance_report.json")
PROCESSED_DATA_PATH = Path("data/processed/processed_dataset.csv")

AUC_THRESHOLD = 0.85
ROLLING_WINDOW = 100   # compute metrics every N predictions
LOG_COLUMNS = [
    "timestamp",
    "run_id",
    "probability",
    "prediction",
    "actual",
    "risk_level",
    "retry_count",
    "duration_deviation",
    "failures_last_7_runs",
    "workflow_failure_rate",
    "concurrent_runs",
]


# ---------------------------------------------------------------------------
# Log predictions
# ---------------------------------------------------------------------------

def log_prediction(
    run_id: str,
    probability: float,
    prediction: int,
    actual: Optional[int],
    features: dict,
    log_path: Path = PREDICTIONS_LOG_PATH,
) -> None:
    """
    Append a single prediction to the predictions log CSV.

    Parameters
    ----------
    run_id      : Unique identifier for the pipeline run.
    probability : Model's predicted failure probability (0-1).
    prediction  : Binary prediction (0 = pass, 1 = fail).
    actual      : True label if available, else None.
    features    : Dict of feature values for the monitored features.
    log_path    : Path to the predictions log CSV.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    risk_level = "HIGH" if probability >= 0.75 else "MEDIUM" if probability >= 0.5 else "LOW"

    row = {
        "timestamp": datetime.now().isoformat(),
        "run_id": str(run_id),
        "probability": round(float(probability), 4),
        "prediction": int(prediction),
        "actual": int(actual) if actual is not None else None,
        "risk_level": risk_level,
        "retry_count": features.get("retry_count", None),
        "duration_deviation": features.get("duration_deviation", None),
        "failures_last_7_runs": features.get("failures_last_7_runs", None),
        "workflow_failure_rate": features.get("workflow_failure_rate", None),
        "concurrent_runs": features.get("concurrent_runs", None),
    }

    df_row = pd.DataFrame([row])

    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)

    logger.debug("Logged prediction for run_id=%s | prob=%.4f | risk=%s", run_id, probability, risk_level)


# ---------------------------------------------------------------------------
# Compute rolling metrics
# ---------------------------------------------------------------------------

def compute_rolling_metrics(
    log_path: Path = PREDICTIONS_LOG_PATH,
    window: int = ROLLING_WINDOW,
) -> dict:
    """
    Compute rolling AUC and F1 over the last N predictions with known actuals.

    Parameters
    ----------
    log_path : Path to the predictions log CSV.
    window   : Number of recent predictions to use for metric computation.

    Returns
    -------
    dict with keys:
        - auc           : float — Rolling AUC score
        - f1            : float — Rolling F1 score
        - n_samples     : int   — Number of labeled samples used
        - auc_flagged   : bool  — True if AUC < threshold
        - threshold     : float — AUC threshold
        - timestamp     : str   — When metrics were computed

    Raises
    ------
    FileNotFoundError : If predictions log does not exist.
    ValueError        : If not enough labeled predictions to compute metrics.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Predictions log not found at: {log_path}")

    df = pd.read_csv(log_path)
    labeled = df.dropna(subset=["actual"]).tail(window)

    if len(labeled) < 10:
        raise ValueError(
            f"Not enough labeled predictions to compute metrics. "
            f"Need at least 10, got {len(labeled)}."
        )

    y_true = labeled["actual"].astype(int).values
    y_prob = labeled["probability"].values
    y_pred = labeled["prediction"].astype(int).values

    # Handle case where only one class present
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in labels — AUC cannot be computed. Using 0.0.")
        auc = 0.0
    else:
        auc = round(float(roc_auc_score(y_true, y_prob)), 4)

    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    auc_flagged = auc < AUC_THRESHOLD

    logger.info(
        "Rolling metrics | AUC: %.4f | F1: %.4f | Samples: %d | Flagged: %s",
        auc, f1, len(labeled), auc_flagged
    )

    if auc_flagged:
        logger.warning(
            "AUC %.4f is below threshold %.2f — retraining may be needed.",
            auc, AUC_THRESHOLD
        )

    return {
        "auc": auc,
        "f1": f1,
        "n_samples": len(labeled),
        "auc_flagged": auc_flagged,
        "threshold": AUC_THRESHOLD,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Save performance report
# ---------------------------------------------------------------------------

def save_performance_report(
    metrics: dict,
    report_path: Path = PERFORMANCE_REPORT_PATH,
) -> dict:
    """
    Save performance metrics to a JSON report file.

    Parameters
    ----------
    metrics     : Output of compute_rolling_metrics().
    report_path : Path to save the JSON report.

    Returns
    -------
    dict — The full report saved to disk.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": metrics["timestamp"],
        "rolling_window": ROLLING_WINDOW,
        "auc_threshold": AUC_THRESHOLD,
        "auc": metrics["auc"],
        "f1": metrics["f1"],
        "n_samples": metrics["n_samples"],
        "auc_flagged": metrics["auc_flagged"],
        "action": "RETRAIN" if metrics["auc_flagged"] else "NO_ACTION",
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Performance report saved to: %s", report_path)
    logger.info("Action: %s", report["action"])
    return report


# ---------------------------------------------------------------------------
# Simulate predictions from processed data (for testing)
# ---------------------------------------------------------------------------

def simulate_predictions_from_data(
    data_path: Path = PROCESSED_DATA_PATH,
    n_rows: int = 200,
    auc_drop: bool = False,
    log_path: Path = PREDICTIONS_LOG_PATH,
) -> None:
    """
    Simulate predictions by loading real processed data and generating
    mock probabilities. Used for testing when no live model is available.

    Parameters
    ----------
    data_path : Path to processed dataset.
    n_rows    : Number of rows to simulate (default 200).
    auc_drop  : If True, inject bad predictions to simulate AUC drop.
    log_path  : Path to write predictions log.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at: {data_path}")

    logger.info("Simulating %d predictions from: %s", n_rows, data_path)
    df = pd.read_csv(data_path, low_memory=False)
    df["run_id"] = df["run_id"].astype(str) if "run_id" in df.columns else [str(i) for i in range(len(df))]

    sample = df.sample(n=min(n_rows, len(df)), random_state=42).reset_index(drop=True)

    rng = np.random.default_rng(42)

    for _, row in sample.iterrows():
        actual = int(row["failed"]) if "failed" in row else None

        if auc_drop:
            # Simulate bad model — random predictions
            probability = float(rng.uniform(0, 1))
        else:
            # Simulate good model — predictions correlated with actual
            if actual == 1:
                probability = float(rng.beta(5, 2))   # skewed high
            else:
                probability = float(rng.beta(2, 5))   # skewed low

        prediction = 1 if probability >= 0.5 else 0

        features = {
            "retry_count": row.get("retry_count", 0),
            "duration_deviation": row.get("duration_deviation", 0.0),
            "failures_last_7_runs": row.get("failures_last_7_runs", 0),
            "workflow_failure_rate": row.get("workflow_failure_rate", 0.0),
            "concurrent_runs": row.get("concurrent_runs", 0),
        }

        log_prediction(
            run_id=row["run_id"],
            probability=probability,
            prediction=prediction,
            actual=actual,
            features=features,
            log_path=log_path,
        )

    logger.info("Simulation complete. %d predictions logged to: %s", len(sample), log_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_performance_monitor(
    simulate: bool = False,
    auc_drop: bool = False,
    data_path: Path = PROCESSED_DATA_PATH,
    log_path: Path = PREDICTIONS_LOG_PATH,
    report_path: Path = PERFORMANCE_REPORT_PATH,
) -> dict:
    """
    End-to-end performance monitoring pipeline. Called by Airflow DAG.

    Parameters
    ----------
    simulate    : Simulate predictions from processed data (for testing).
    auc_drop    : If simulate=True, inject AUC drop scenario.
    data_path   : Path to processed dataset.
    log_path    : Path to predictions log CSV.
    report_path : Path to save performance report JSON.

    Returns
    -------
    dict — Performance report with AUC, F1, and action.
    """
    logger.info("=" * 60)
    logger.info("PERFORMANCE MONITOR START")
    logger.info("=" * 60)

    if simulate:
        logger.info("Simulation mode — generating synthetic predictions...")
        simulate_predictions_from_data(
            data_path=data_path,
            n_rows=200,
            auc_drop=auc_drop,
            log_path=log_path,
        )

    metrics = compute_rolling_metrics(log_path=log_path)
    report = save_performance_report(metrics, report_path=report_path)

    logger.info("PERFORMANCE MONITOR COMPLETE")
    logger.info("=" * 60)
    return report


# ---------------------------------------------------------------------------
# Airflow XCom wrapper
# ---------------------------------------------------------------------------

def run_performance_monitor_airflow(**context) -> float:
    """
    Airflow-compatible wrapper. Pushes AUC score to XCom.

    Returns
    -------
    float — Rolling AUC score pushed to XCom as 'rolling_auc'.
    """
    report = run_performance_monitor()
    auc = report["auc"]
    context["ti"].xcom_push(key="rolling_auc", value=auc)
    logger.info("Rolling AUC pushed to XCom: %.4f", auc)
    return auc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Autopilot - Performance Monitor"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Simulate predictions from processed data"
    )
    parser.add_argument(
        "--auc-drop", action="store_true",
        help="Inject AUC drop scenario (only with --simulate)"
    )
    parser.add_argument(
        "--data", type=str, default=str(PROCESSED_DATA_PATH),
        help="Path to processed dataset CSV"
    )
    parser.add_argument(
        "--log", type=str, default=str(PREDICTIONS_LOG_PATH),
        help="Path to predictions log CSV"
    )
    args = parser.parse_args()

    report = run_performance_monitor(
        simulate=args.simulate,
        auc_drop=args.auc_drop,
        data_path=Path(args.data),
        log_path=Path(args.log),
    )

    print(f"\nRolling AUC : {report['auc']}")
    print(f"Rolling F1  : {report['f1']}")
    print(f"Samples     : {report['n_samples']}")
    print(f"AUC flagged : {report['auc_flagged']}")
    print(f"Action      : {report['action']}")