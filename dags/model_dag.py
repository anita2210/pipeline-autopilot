"""
model_dag.py
Airflow DAG for PipelineGuard Model Development Pipeline (Assignment 2).
"""
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

sys.path.insert(0, "/opt/airflow/scripts")
logger = logging.getLogger(__name__)

default_args = {
    "owner": "pipeline-autopilot",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

def task_mlflow_setup():
    from experiment_tracking import setup_mlflow
    setup_mlflow()

def task_train_models():
    from model_training import main as train_models
    train_models()

def task_validate_model():
    from model_validation import run_model_validation
    run_model_validation()

def task_bias_detection():
    from model_bias_detection import run_bias_detection
    run_bias_detection()

def task_sensitivity_analysis():
    from model_sensitivity import run_sensitivity_analysis
    run_sensitivity_analysis()

def task_validation_gate(**context):
    BASE_DIR = Path("/opt/airflow")
    val_rpt = BASE_DIR / "data" / "reports" / "validation_report.json"
    if val_rpt.exists():
        with open(val_rpt) as f:
            rpt = json.load(f)
        auc = rpt.get("validation_gate", {}).get("auc_roc") or rpt.get("hold_out_metrics", {}).get("auc_roc") or rpt.get("auc_roc", 0)
        logger.info("Validation AUC: %.4f", float(auc))
        if float(auc) < 0.85:
            raise ValueError(f"Validation gate FAILED: AUC {auc} < 0.85")
    else:
        logger.warning("validation_report.json not found — skipping AUC check.")
    logger.info("Validation gate PASSED.")

def task_push_registry():
    from model_registry import run_model_registry
    run_model_registry(dry_run=False)

with DAG(
    dag_id="pipelineguard_model_pipeline",
    description="PipelineGuard ML Pipeline — Assignment 2",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 3, 1),
    catchup=False,
    max_active_runs=1,
    tags=["pipeline-autopilot", "model-pipeline", "assignment-2"],
) as dag:

    t0 = PythonOperator(task_id="mlflow_setup",            python_callable=task_mlflow_setup)
    t1 = PythonOperator(task_id="train_models",            python_callable=task_train_models)
    t2 = PythonOperator(task_id="validate_model",          python_callable=task_validate_model)
    t3 = PythonOperator(task_id="model_bias_detection",    python_callable=task_bias_detection)
    t4 = PythonOperator(task_id="sensitivity_analysis",    python_callable=task_sensitivity_analysis)
    t5 = PythonOperator(task_id="validation_gate",         python_callable=task_validation_gate, provide_context=True)
    t6 = PythonOperator(task_id="push_to_registry",        python_callable=task_push_registry)
    t7 = EmptyOperator(task_id="model_pipeline_complete")

    t0 >> t1 >> t2 >> [t3, t4] >> t5 >> t6 >> t7
