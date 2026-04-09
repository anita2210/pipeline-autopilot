"""
retrain_trigger.py
------------------
Reads drift score (from drift_detection.py) and rolling AUC (from
performance_monitor.py) and triggers Airflow DAG retraining via the
Airflow REST API if either threshold is exceeded. Sends a notification
email when retraining is triggered.

Author  : Member 4 (MLOps Monitor)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import json
import logging
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests

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
logger = logging.getLogger("retrain_trigger")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DRIFT_SUMMARY_PATH = Path("data/reports/drift/drift_summary.json")
PERFORMANCE_REPORT_PATH = Path("data/reports/performance/performance_report.json")
RETRAIN_LOG_PATH = Path("data/monitoring/retrain_log.json")

DRIFT_THRESHOLD = 0.3
AUC_THRESHOLD = 0.85

# Airflow REST API config
AIRFLOW_BASE_URL = "http://localhost:8080"
AIRFLOW_DAG_ID = "pipeline_autopilot_data_pipeline"
AIRFLOW_USERNAME = "airflow"
AIRFLOW_PASSWORD = "airflow"

# Email config (set via environment or update directly)
ALERT_EMAIL_TO = "team@pipelineautopilot.com"
SMTP_HOST = "localhost"
SMTP_PORT = 25


# ---------------------------------------------------------------------------
# Read monitoring outputs
# ---------------------------------------------------------------------------

def read_drift_summary(
    summary_path: Path = DRIFT_SUMMARY_PATH,
) -> dict:
    """
    Read the latest drift summary from drift_detection.py output.

    Parameters
    ----------
    summary_path : Path to drift_summary.json.

    Returns
    -------
    dict — Drift summary with score, detected flag, and drifted features.

    Raises
    ------
    FileNotFoundError : If drift summary does not exist.
    """
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Drift summary not found at: {summary_path}. "
            "Run drift_detection.py first."
        )

    with open(summary_path, "r") as f:
        summary = json.load(f)

    logger.info(
        "Drift summary loaded | Score: %.4f | Detected: %s",
        summary.get("overall_drift_score", 0.0),
        summary.get("drift_detected", False),
    )
    return summary


def read_performance_report(
    report_path: Path = PERFORMANCE_REPORT_PATH,
) -> dict:
    """
    Read the latest performance report from performance_monitor.py output.

    Parameters
    ----------
    report_path : Path to performance_report.json.

    Returns
    -------
    dict — Performance report with AUC, F1, and flagged status.

    Raises
    ------
    FileNotFoundError : If performance report does not exist.
    """
    if not report_path.exists():
        raise FileNotFoundError(
            f"Performance report not found at: {report_path}. "
            "Run performance_monitor.py first."
        )

    with open(report_path, "r") as f:
        report = json.load(f)

    logger.info(
        "Performance report loaded | AUC: %.4f | Flagged: %s",
        report.get("auc", 0.0),
        report.get("auc_flagged", False),
    )
    return report


# ---------------------------------------------------------------------------
# Trigger decision
# ---------------------------------------------------------------------------

def should_retrain(
    drift_summary: dict,
    performance_report: dict,
) -> tuple:
    """
    Decide whether retraining should be triggered based on drift and AUC.

    Parameters
    ----------
    drift_summary      : Output of read_drift_summary().
    performance_report : Output of read_performance_report().

    Returns
    -------
    tuple (bool, str) — (should_retrain, reason)
    """
    drift_score = drift_summary.get("overall_drift_score", 0.0)
    drift_detected = drift_summary.get("drift_detected", False)
    auc = performance_report.get("auc", 1.0)
    auc_flagged = performance_report.get("auc_flagged", False)

    reasons = []

    if drift_detected:
        reasons.append(
            f"Data drift detected (score={drift_score:.4f} > threshold={DRIFT_THRESHOLD})"
        )
    if auc_flagged:
        reasons.append(
            f"AUC degradation detected (AUC={auc:.4f} < threshold={AUC_THRESHOLD})"
        )

    trigger = len(reasons) > 0
    reason = " | ".join(reasons) if reasons else "No retraining needed"

    if trigger:
        logger.warning("RETRAIN TRIGGERED: %s", reason)
    else:
        logger.info("No retraining needed | Drift: %.4f | AUC: %.4f", drift_score, auc)

    return trigger, reason


# ---------------------------------------------------------------------------
# Trigger Airflow DAG
# ---------------------------------------------------------------------------

def trigger_airflow_dag(
    dag_id: str = AIRFLOW_DAG_ID,
    base_url: str = AIRFLOW_BASE_URL,
    username: str = AIRFLOW_USERNAME,
    password: str = AIRFLOW_PASSWORD,
    reason: str = "",
) -> bool:
    """
    Trigger an Airflow DAG run via the Airflow REST API.

    Parameters
    ----------
    dag_id   : ID of the DAG to trigger.
    base_url : Airflow webserver base URL.
    username : Airflow username.
    password : Airflow password.
    reason   : Reason string to include in DAG run conf.

    Returns
    -------
    bool — True if trigger succeeded, False otherwise.
    """
    url = f"{base_url}/api/v1/dags/{dag_id}/dagRuns"
    payload = {
        "dag_run_id": f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "conf": {
            "trigger_reason": reason,
            "triggered_by": "retrain_trigger.py",
            "timestamp": datetime.now().isoformat(),
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            auth=(username, password),
            timeout=30,
        )
        if response.status_code in (200, 201):
            logger.info(
                "Airflow DAG '%s' triggered successfully. Run ID: %s",
                dag_id,
                payload["dag_run_id"],
            )
            return True
        else:
            logger.error(
                "Failed to trigger DAG '%s'. Status: %d | Response: %s",
                dag_id,
                response.status_code,
                response.text[:200],
            )
            return False
    except requests.exceptions.ConnectionError:
        logger.warning(
            "Could not connect to Airflow at %s — DAG trigger skipped (non-fatal in local mode).",
            base_url,
        )
        return False
    except Exception as exc:
        logger.error("Unexpected error triggering DAG: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Send notification email
# ---------------------------------------------------------------------------

def send_notification_email(
    reason: str,
    drift_summary: dict,
    performance_report: dict,
    to_email: str = ALERT_EMAIL_TO,
) -> bool:
    """
    Send a notification email when retraining is triggered.

    Parameters
    ----------
    reason             : Reason for retraining trigger.
    drift_summary      : Drift detection results dict.
    performance_report : Performance monitoring results dict.
    to_email           : Recipient email address.

    Returns
    -------
    bool — True if email sent, False otherwise.
    """
    subject = "[PipelineGuard] Retraining Triggered — Action Required"

    drift_score = drift_summary.get("overall_drift_score", "N/A")
    drifted_features = drift_summary.get("drifted_features", [])
    auc = performance_report.get("auc", "N/A")
    f1 = performance_report.get("f1", "N/A")

    body = f"""
PipelineGuard Auto-Retrain Notification
========================================

Timestamp : {datetime.now().isoformat()}
Reason    : {reason}

Drift Detection
---------------
Overall Drift Score : {drift_score}
Drifted Features    : {', '.join(drifted_features) if drifted_features else 'None'}
Drift Threshold     : {DRIFT_THRESHOLD}

Model Performance
-----------------
Rolling AUC : {auc}
Rolling F1  : {f1}
AUC Threshold : {AUC_THRESHOLD}

Action
------
Airflow DAG '{AIRFLOW_DAG_ID}' has been triggered for retraining.
Please monitor the DAG run at: {AIRFLOW_BASE_URL}

— PipelineGuard Monitoring System
"""

    msg = MIMEMultipart()
    msg["From"] = "pipelineguard@monitor.com"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.sendmail(msg["From"], to_email, msg.as_string())
        logger.info("Notification email sent to: %s", to_email)
        return True
    except Exception as exc:
        logger.warning(
            "Email notification skipped (non-fatal): %s", exc
        )
        return False


# ---------------------------------------------------------------------------
# Save retrain log
# ---------------------------------------------------------------------------

def save_retrain_log(
    triggered: bool,
    reason: str,
    dag_triggered: bool,
    drift_summary: dict,
    performance_report: dict,
    log_path: Path = RETRAIN_LOG_PATH,
) -> dict:
    """
    Append a retrain trigger event to the retrain log JSON.

    Parameters
    ----------
    triggered          : Whether retraining was triggered.
    reason             : Reason for trigger decision.
    dag_triggered      : Whether Airflow DAG was successfully triggered.
    drift_summary      : Drift detection results.
    performance_report : Performance monitoring results.
    log_path           : Path to retrain log JSON.

    Returns
    -------
    dict — The log entry saved.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "retrain_triggered": triggered,
        "reason": reason,
        "dag_triggered": dag_triggered,
        "drift_score": drift_summary.get("overall_drift_score", 0.0),
        "drift_detected": drift_summary.get("drift_detected", False),
        "drifted_features": drift_summary.get("drifted_features", []),
        "auc": performance_report.get("auc", 0.0),
        "auc_flagged": performance_report.get("auc_flagged", False),
    }

    # Load existing log or start fresh
    if log_path.exists():
        with open(log_path, "r") as f:
            try:
                log = json.load(f)
                if not isinstance(log, list):
                    log = [log]
            except json.JSONDecodeError:
                log = []
    else:
        log = []

    log.append(entry)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    logger.info("Retrain log updated at: %s", log_path)
    return entry


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_retrain_trigger(
    drift_summary_path: Path = DRIFT_SUMMARY_PATH,
    performance_report_path: Path = PERFORMANCE_REPORT_PATH,
    dry_run: bool = False,
) -> dict:
    """
    End-to-end retrain trigger pipeline. Called by Airflow DAG or CLI.

    Parameters
    ----------
    drift_summary_path      : Path to drift_summary.json.
    performance_report_path : Path to performance_report.json.
    dry_run                 : If True, check thresholds but do not trigger DAG or send email.

    Returns
    -------
    dict — Retrain log entry with trigger decision and details.
    """
    logger.info("=" * 60)
    logger.info("RETRAIN TRIGGER START")
    logger.info("=" * 60)

    # 1. Read monitoring outputs
    drift_summary = read_drift_summary(drift_summary_path)
    performance_report = read_performance_report(performance_report_path)

    # 2. Decide whether to retrain
    triggered, reason = should_retrain(drift_summary, performance_report)

    dag_triggered = False

    if triggered and not dry_run:
        # 3. Trigger Airflow DAG
        dag_triggered = trigger_airflow_dag(reason=reason)

        # 4. Send notification email
        send_notification_email(reason, drift_summary, performance_report)

    elif triggered and dry_run:
        logger.info("DRY RUN — retraining would be triggered but skipped.")

    # 5. Save retrain log
    entry = save_retrain_log(
        triggered=triggered,
        reason=reason,
        dag_triggered=dag_triggered,
        drift_summary=drift_summary,
        performance_report=performance_report,
    )

    logger.info("RETRAIN TRIGGER COMPLETE")
    logger.info("=" * 60)
    return entry


# ---------------------------------------------------------------------------
# Airflow XCom wrapper
# ---------------------------------------------------------------------------

def run_retrain_trigger_airflow(**context) -> bool:
    """
    Airflow-compatible wrapper. Reads drift_score and rolling_auc from XCom.

    Returns
    -------
    bool — Whether retraining was triggered.
    """
    entry = run_retrain_trigger()
    context["ti"].xcom_push(key="retrain_triggered", value=entry["retrain_triggered"])
    return entry["retrain_triggered"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Autopilot - Retrain Trigger"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check thresholds but do not trigger DAG or send email"
    )
    parser.add_argument(
        "--drift-summary", type=str, default=str(DRIFT_SUMMARY_PATH),
        help="Path to drift_summary.json"
    )
    parser.add_argument(
        "--performance-report", type=str, default=str(PERFORMANCE_REPORT_PATH),
        help="Path to performance_report.json"
    )
    args = parser.parse_args()

    entry = run_retrain_trigger(
        drift_summary_path=Path(args.drift_summary),
        performance_report_path=Path(args.performance_report),
        dry_run=args.dry_run,
    )

    print(f"\nRetrain triggered : {entry['retrain_triggered']}")
    print(f"Reason            : {entry['reason']}")
    print(f"DAG triggered     : {entry['dag_triggered']}")
    print(f"Drift score       : {entry['drift_score']}")
    print(f"AUC               : {entry['auc']}")