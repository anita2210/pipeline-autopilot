"""
drift_detection.py
------------------
Compares incoming prediction data against the training distribution for 5 key
features using Evidently AI. Generates an HTML drift report and flags if the
drift score exceeds the threshold. Outputs drift score to Airflow XCom when
called from the DAG.

Author  : Member 4 (MLOps Monitor)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

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
logger = logging.getLogger("drift_detection")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DATA_PATH = Path("data/processed/processed_dataset.csv")
PREDICTIONS_LOG_PATH = Path("data/monitoring/predictions_log.csv")
DRIFT_REPORT_DIR = Path("data/reports/drift")
DRIFT_SUMMARY_PATH = Path("data/reports/drift/drift_summary.json")

DRIFT_THRESHOLD = 0.3

MONITORED_FEATURES = [
    "retry_count",
    "duration_deviation",
    "failures_last_7_runs",
    "workflow_failure_rate",
    "concurrent_runs",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reference_data(
    data_path: Path = PROCESSED_DATA_PATH,
    sample_size: int = 5000,
) -> pd.DataFrame:
    """
    Load a sample of the training data as the reference distribution.

    Parameters
    ----------
    data_path   : Path to the processed training dataset.
    sample_size : Number of rows to sample as reference (default 5000).

    Returns
    -------
    pd.DataFrame — Reference dataset with monitored features only.

    Raises
    ------
    FileNotFoundError : If processed data file does not exist.
    ValueError        : If monitored features are missing from the dataset.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at: {data_path}")

    logger.info("Loading reference data from: %s", data_path)
    df = pd.read_csv(data_path, low_memory=False)

    missing = set(MONITORED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing monitored features in reference data: {missing}")

    reference = df[MONITORED_FEATURES].dropna().sample(
        n=min(sample_size, len(df)), random_state=42
    )
    logger.info("Reference data loaded: %d rows x %d features", len(reference), len(MONITORED_FEATURES))
    return reference


def load_current_data(
    predictions_log_path: Path = PREDICTIONS_LOG_PATH,
    window: int = 500,
) -> pd.DataFrame:
    """
    Load the most recent predictions as the current distribution to compare.

    Parameters
    ----------
    predictions_log_path : Path to predictions_log.csv written by performance_monitor.py.
    window               : Number of most recent predictions to use (default 500).

    Returns
    -------
    pd.DataFrame — Current dataset with monitored features only.

    Raises
    ------
    FileNotFoundError : If predictions log does not exist.
    ValueError        : If monitored features are missing from the log.
    """
    if not predictions_log_path.exists():
        raise FileNotFoundError(
            f"Predictions log not found at: {predictions_log_path}. "
            "Run performance_monitor.py first to generate it."
        )

    logger.info("Loading current data from: %s", predictions_log_path)
    df = pd.read_csv(predictions_log_path, low_memory=False)

    missing = set(MONITORED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing monitored features in predictions log: {missing}")

    current = df[MONITORED_FEATURES].tail(window).dropna()
    logger.info("Current data loaded: %d rows x %d features", len(current), len(MONITORED_FEATURES))
    return current


def generate_synthetic_current(
    reference: pd.DataFrame,
    drift: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic current data for testing when no predictions log exists.

    Parameters
    ----------
    reference : Reference DataFrame to base distributions on.
    drift     : If True, inject drift by shifting distributions. Default False.
    seed      : Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame — Synthetic current data.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    synthetic = reference.copy().reset_index(drop=True)
    if drift:
        logger.info("Injecting synthetic drift for testing...")
        for col in MONITORED_FEATURES:
            synthetic[col] = synthetic[col] * rng.uniform(1.5, 2.5)
    else:
        noise = rng.normal(0, 0.05, size=synthetic.shape)
        std_vals = synthetic.std().values
        synthetic = synthetic + noise * std_vals

    return synthetic


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_dir: Path = DRIFT_REPORT_DIR,
) -> Report:
    """
    Run Evidently DataDriftPreset report comparing reference vs current data.

    Parameters
    ----------
    reference  : Reference (training) distribution DataFrame.
    current    : Current (incoming) distribution DataFrame.
    report_dir : Directory to save the HTML report.

    Returns
    -------
    evidently.report.Report — The completed Evidently report object.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = MONITORED_FEATURES

    report = Report(metrics=[DataDriftPreset()])
    logger.info("Running Evidently DataDrift report...")
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = report_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))
    logger.info("Drift report saved to: %s", html_path)

    return report


def extract_drift_score(report: Report) -> dict:
    """
    Extract per-feature drift scores and overall drift flag from Evidently report.

    Parameters
    ----------
    report : Completed Evidently Report object.

    Returns
    -------
    dict with keys:
        - overall_drift_score   : float
        - per_feature           : dict
        - drifted_features      : list
        - drift_detected        : bool
        - threshold             : float
    """
    result = report.as_dict()
    metrics = result.get("metrics", [])

    per_feature = {}
    drifted_features = []

    for metric in metrics:
        result_data = metric.get("result", {})
        drift_by_columns = result_data.get("drift_by_columns", {})
        for col, col_data in drift_by_columns.items():
            if col in MONITORED_FEATURES:
                score = col_data.get("drift_score", 0.0)
                drifted = col_data.get("drift_detected", False)
                per_feature[col] = round(float(score), 4)
                if drifted:
                    drifted_features.append(col)

    for col in MONITORED_FEATURES:
        if col not in per_feature:
            per_feature[col] = 0.0

    overall_score = round(sum(per_feature.values()) / len(per_feature), 4)
    drift_detected = overall_score > DRIFT_THRESHOLD

    logger.info(
        "Drift score: %.4f | Threshold: %.2f | Detected: %s",
        overall_score, DRIFT_THRESHOLD, drift_detected
    )
    if drifted_features:
        logger.warning("Drifted features: %s", drifted_features)

    return {
        "overall_drift_score": overall_score,
        "per_feature": per_feature,
        "drifted_features": drifted_features,
        "drift_detected": drift_detected,
        "threshold": DRIFT_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------

def save_drift_summary(
    drift_result: dict,
    summary_path: Path = DRIFT_SUMMARY_PATH,
) -> dict:
    """
    Save drift detection results to a JSON summary file.

    Parameters
    ----------
    drift_result : Output of extract_drift_score().
    summary_path : Path to save the JSON summary.

    Returns
    -------
    dict — The full summary dict saved to disk.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "monitored_features": MONITORED_FEATURES,
        "drift_threshold": DRIFT_THRESHOLD,
        "overall_drift_score": drift_result["overall_drift_score"],
        "drift_detected": drift_result["drift_detected"],
        "drifted_features": drift_result["drifted_features"],
        "per_feature_scores": drift_result["per_feature"],
        "action": "RETRAIN" if drift_result["drift_detected"] else "NO_ACTION",
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Drift summary saved to: %s", summary_path)
    logger.info("Action: %s", summary["action"])
    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_drift_detection(
    use_synthetic: bool = False,
    inject_drift: bool = False,
    reference_path: Path = PROCESSED_DATA_PATH,
    predictions_log_path: Path = PREDICTIONS_LOG_PATH,
    report_dir: Path = DRIFT_REPORT_DIR,
    summary_path: Path = DRIFT_SUMMARY_PATH,
) -> dict:
    """
    End-to-end drift detection pipeline. Called by Airflow DAG daily.

    Parameters
    ----------
    use_synthetic        : Use synthetic current data (for testing). Default False.
    inject_drift         : If use_synthetic=True, inject drift. Default False.
    reference_path       : Path to processed training data.
    predictions_log_path : Path to predictions log CSV.
    report_dir           : Output directory for HTML reports.
    summary_path         : Output path for drift summary JSON.

    Returns
    -------
    dict — Drift summary including score, flagged features, and action.
    """
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION START")
    logger.info("=" * 60)

    reference = load_reference_data(reference_path)

    if use_synthetic:
        logger.info("Using synthetic current data (testing mode)...")
        current = generate_synthetic_current(reference, drift=inject_drift)
    else:
        try:
            current = load_current_data(predictions_log_path)
        except FileNotFoundError:
            logger.warning(
                "Predictions log not found — falling back to synthetic data for demo."
            )
            current = generate_synthetic_current(reference, drift=False)

    report = run_evidently_report(reference, current, report_dir)
    drift_result = extract_drift_score(report)
    summary = save_drift_summary(drift_result, summary_path)

    logger.info("DRIFT DETECTION COMPLETE")
    logger.info("=" * 60)
    return summary


# ---------------------------------------------------------------------------
# Airflow XCom wrapper
# ---------------------------------------------------------------------------

def run_drift_detection_airflow(**context) -> float:
    """
    Airflow-compatible wrapper. Pushes drift score to XCom.

    Returns
    -------
    float — Overall drift score pushed to XCom as 'drift_score'.
    """
    summary = run_drift_detection()
    drift_score = summary["overall_drift_score"]
    context["ti"].xcom_push(key="drift_score", value=drift_score)
    logger.info("Drift score pushed to XCom: %.4f", drift_score)
    return drift_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Autopilot - Drift Detection"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic current data instead of predictions log"
    )
    parser.add_argument(
        "--inject-drift", action="store_true",
        help="Inject synthetic drift (only with --synthetic)"
    )
    parser.add_argument(
        "--reference", type=str, default=str(PROCESSED_DATA_PATH),
        help="Path to reference (training) data CSV"
    )
    parser.add_argument(
        "--predictions-log", type=str, default=str(PREDICTIONS_LOG_PATH),
        help="Path to predictions log CSV"
    )
    args = parser.parse_args()

    summary = run_drift_detection(
        use_synthetic=args.synthetic,
        inject_drift=args.inject_drift,
        reference_path=Path(args.reference),
        predictions_log_path=Path(args.predictions_log),
    )

    print(f"\nOverall drift score : {summary['overall_drift_score']}")
    print(f"Drift detected      : {summary['drift_detected']}")
    print(f"Drifted features    : {summary['drifted_features']}")
    print(f"Action              : {summary['action']}")