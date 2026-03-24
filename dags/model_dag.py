"""
model_dag.py

Airflow DAG for PipelineGuard Model Development Pipeline (Assignment 2).

Execution Order:
    load_processed_data
            ↓
    train_models  (Member 1 + Member 2)
            ↓
    validate_model  (Member 3)
            ↓
    model_bias_detection ──── sensitivity_analysis   ← PARALLEL (Member 4 + Member 5)
            └──────────┬───────────────┘
                       ↓
               validation_gate
                       ↓
               push_to_registry  (Member 5)
                       ↓
           model_pipeline_complete
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# Add scripts to path
sys.path.insert(0, "/opt/airflow/scripts")

# ── Import Member scripts ─────────────────────────────────────────────────────
from experiment_tracking import setup_mlflow          # Member 2
from model_training      import main as train_models  # Member 1
from model_validation    import run_model_validation  # Member 3
from model_bias_detection import run_bias_detection   # Member 4
from model_sensitivity   import run_sensitivity_analysis  # Member 5
from model_registry      import run_model_registry    # Member 5

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# =============================================================================
# DAG DEFAULT ARGS
# =============================================================================

default_args = {
    "owner":             "pipeline-autopilot",
    "email":             ["team@pipelineautopilot.com"],
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id="pipelineguard_model_pipeline",
    description="PipelineGuard ML model training, validation, bias detection, and registry push",
    default_args=default_args,
    schedule_interval=None,   # Triggered manually or via CI/CD (GitHub Actions)
    start_date=datetime(2026, 3, 1),
    catchup=False,
    max_active_runs=1,
    tags=["pipeline-autopilot", "model-pipeline", "assignment-2"],
) as dag:

    # ── Task 0: MLflow setup (Member 2) ──────────────────────────────────────
    mlflow_setup = PythonOperator(
        task_id="mlflow_setup",
        python_callable=setup_mlflow,
    )

    # ── Task 1: Model training + hyperparameter tuning (Member 1) ────────────
    train_model = PythonOperator(
        task_id="train_models",
        python_callable=train_models,
    )

    # ── Task 2: Model validation — hold-out eval, threshold, rollback (M3) ───
    validate_model = PythonOperator(
        task_id="validate_model",
        python_callable=run_model_validation,
    )

    # ── Task 3a: Model bias detection with Fairlearn (Member 4) ─────────────
    model_bias_detection = PythonOperator(
        task_id="model_bias_detection",
        python_callable=run_bias_detection,
    )

    # ── Task 3b: Sensitivity analysis — SHAP + hyperparams (Member 5) ───────
    sensitivity_analysis = PythonOperator(
        task_id="sensitivity_analysis",
        python_callable=run_sensitivity_analysis,
    )

    # ── Task 4: Validation gate — AUC > 0.85 + no critical bias flags ────────
    def validation_gate(**context):
        """
        Pull results from upstream tasks and decide whether to proceed.
        Blocks push_to_registry if AUC < 0.85 or critical bias detected.
        """
        import json
        from pathlib import Path

        BASE_DIR = Path("/opt/airflow")

        # Check validation report
        val_rpt = BASE_DIR / "data" / "reports" / "validation_report.json"
        if val_rpt.exists():
            with open(val_rpt) as f:
                rpt = json.load(f)
            auc = rpt.get("auc_roc") or rpt.get("metrics", {}).get("auc_roc", 0)
            gate = rpt.get("gate_result", {}).get("status", "unknown")
            logger.info("Validation report — AUC: %.4f | Gate: %s", auc, gate)
            if float(auc) < 0.85:
                raise ValueError(f"Validation gate FAILED: AUC {auc} < 0.85")
        else:
            logger.warning("validation_report.json not found — skipping AUC gate check.")

        # Check bias report
        bias_rpt = BASE_DIR / "models" / "model_bias_report.json"
        if bias_rpt.exists():
            with open(bias_rpt) as f:
                brpt = json.load(f)
            flagged = brpt.get("flagged_biases", [])
            if flagged:
                logger.warning("Bias flags detected: %s", flagged)
            # Hard block only on critical disparity (>3x)
            disparities = [
                v.get("disparity_ratio", 1.0)
                for v in brpt.get("slices", {}).values()
                if isinstance(v, dict)
            ]
            if disparities and max(disparities) > 3.0:
                raise ValueError(
                    f"Validation gate FAILED: Max bias disparity "
                    f"{max(disparities):.2f} exceeds hard limit of 3.0"
                )

        logger.info("Validation gate PASSED — proceeding to registry push.")

    validation_gate_task = PythonOperator(
        task_id="validation_gate",
        python_callable=validation_gate,
        provide_context=True,
    )

    # ── Task 5: Push to GCP Artifact Registry (Member 5) ────────────────────
    push_to_registry = PythonOperator(
        task_id="push_to_registry",
        python_callable=lambda: run_model_registry(dry_run=False),
    )

    # ── Task 6: Pipeline complete ─────────────────────────────────────────────
    model_pipeline_complete = EmptyOperator(
        task_id="model_pipeline_complete",
    )

    # =========================================================================
    # DEPENDENCIES
    # =========================================================================
    #
    #   mlflow_setup
    #        ↓
    #   train_models
    #        ↓
    #   validate_model
    #        ↓              ↓
    #   model_bias_detection   sensitivity_analysis   ← PARALLEL
    #        └──────────┬────────────────┘
    #                   ↓
    #           validation_gate
    #                   ↓
    #           push_to_registry
    #                   ↓
    #       model_pipeline_complete
    #
    mlflow_setup >> train_model >> validate_model
    validate_model >> [model_bias_detection, sensitivity_analysis]
    [model_bias_detection, sensitivity_analysis] >> validation_gate_task
    validation_gate_task >> push_to_registry
    push_to_registry >> model_pipeline_complete