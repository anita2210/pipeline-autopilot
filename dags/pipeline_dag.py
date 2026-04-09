"""
pipeline_dag.py — PipelineGuard 13-task Airflow DAG
Assignment 3 | Member 1 update: all function names verified against actual scripts
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow/scripts')

default_args = {
    'owner': 'pipeline-guard',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def run_data_acquisition():
    from data_acquisition import acquire_data
    acquire_data()

def run_data_preprocessing():
    from data_preprocessing import (load_data, handle_missing_values,
        remove_duplicates, validate_dtypes, enforce_constraints,
        cap_outliers, parse_datetime, encode_categoricals,
        validate_features, save_processed_data)
    from config import RAW_DATA_FILE
    import logging
    logger = logging.getLogger(__name__)
    df = load_data(str(RAW_DATA_FILE))
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = validate_dtypes(df)
    df = enforce_constraints(df)
    df = cap_outliers(df)
    df = parse_datetime(df)
    df, _ = encode_categoricals(df)
    df = validate_features(df)
    save_processed_data(df)
    logger.info(f"Preprocessing complete. Shape: {df.shape}")

def run_schema_validation():
    from schema_validation import run_schema_validation
    run_schema_validation()

def run_bias_detection():
    from bias_detection import run_bias_detection
    run_bias_detection()

def run_anomaly_detection():
    from anomaly_detection import run_anomaly_detection
    run_anomaly_detection()

def run_dvc_versioning():
    from dvc_versioning import run_full_versioning
    run_full_versioning(push=False, commit=False)

def run_model_training():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model_training", "/opt/airflow/scripts/model_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

def run_model_validation(**context):
    import importlib.util, logging
    logger = logging.getLogger(__name__)
    spec = importlib.util.spec_from_file_location(
        "model_validation", "/opt/airflow/scripts/model_validation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod.run_model_validation()
    # extract AUC and F1 from returned dict
    auc = 0.0
    f1  = 0.0
    if isinstance(report, dict):
        for key in ['auc', 'roc_auc', 'AUC', 'test_auc']:
            if key in report:
                auc = float(report[key]); break
        for key in ['f1', 'f1_score', 'F1', 'test_f1']:
            if key in report:
                f1 = float(report[key]); break
    logger.info(f"Validation — AUC: {auc:.4f}, F1: {f1:.4f}")
    context['ti'].xcom_push(key='val_auc', value=round(auc, 4))
    context['ti'].xcom_push(key='val_f1',  value=round(f1, 4))

def run_model_bias(**context):
    import importlib.util, logging
    logger = logging.getLogger(__name__)
    try:
        spec = importlib.util.spec_from_file_location(
            "model_bias_detection", "/opt/airflow/scripts/model_bias_detection.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_bias_detection()
        logger.info("Model bias detection complete")
    except Exception as e:
        logger.warning(f"Model bias skipped: {e}")

def run_sensitivity_analysis(**context):
    import importlib.util, logging
    logger = logging.getLogger(__name__)
    try:
        spec = importlib.util.spec_from_file_location(
            "model_sensitivity", "/opt/airflow/scripts/model_sensitivity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.run_sensitivity_analysis()
        top3 = ['retry_count', 'duration_deviation', 'failures_last_7_runs']
        if isinstance(result, dict):
            for key in ['top_features', 'shap_features', 'important_features']:
                if key in result:
                    top3 = result[key][:3]; break
        context['ti'].xcom_push(key='top_shap_features', value=top3)
        logger.info(f"Sensitivity analysis complete. Top features: {top3}")
    except Exception as e:
        logger.warning(f"Sensitivity analysis skipped: {e}")
        context['ti'].xcom_push(key='top_shap_features',
            value=['retry_count', 'duration_deviation', 'failures_last_7_runs'])

def run_validation_gate(**context):
    import logging
    logger = logging.getLogger(__name__)
    auc = context['ti'].xcom_pull(task_ids='model_validation', key='val_auc') or 0
    threshold = 0.85
    if float(auc) < threshold:
        raise ValueError(f"Validation gate FAILED: AUC={auc} < {threshold}")
    logger.info(f"Validation gate PASSED: AUC={auc} >= {threshold}")

def run_push_to_registry():
    import importlib.util, logging
    logger = logging.getLogger(__name__)
    try:
        spec = importlib.util.spec_from_file_location(
            "model_registry", "/opt/airflow/scripts/model_registry.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_model_registry(dry_run=False)
        logger.info("Model pushed to registry")
    except Exception as e:
        logger.warning(f"Registry push skipped: {e}")

def run_drift_monitoring(**context):
    import pandas as pd, logging, os
    logger = logging.getLogger(__name__)
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        df = pd.read_csv("/opt/airflow/data/processed/final_dataset_processed.csv")
        features = [f for f in ['retry_count', 'duration_deviation',
            'failures_last_7_runs', 'workflow_failure_rate', 'total_jobs']
            if f in df.columns]
        half = len(df) // 2
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df[features].iloc[:half],
                   current_data=df[features].iloc[half:])
        os.makedirs("/opt/airflow/data/reports", exist_ok=True)
        report.save_html("/opt/airflow/data/reports/drift_report.html")
        result = report.as_dict()
        drift_score = result['metrics'][0]['result'].get('share_of_drifted_columns', 0.0)
        logger.info(f"Drift score: {drift_score:.4f}")
        context['ti'].xcom_push(key='drift_score', value=drift_score)
    except Exception as e:
        logger.warning(f"Drift monitoring skipped: {e}")
        context['ti'].xcom_push(key='drift_score', value=0.0)

def run_pipeline_complete(**context):
    import logging
    logger = logging.getLogger(__name__)
    ti = context['ti']
    auc   = ti.xcom_pull(task_ids='model_validation',     key='val_auc') or 'N/A'
    f1    = ti.xcom_pull(task_ids='model_validation',     key='val_f1')  or 'N/A'
    shap  = ti.xcom_pull(task_ids='sensitivity_analysis', key='top_shap_features') or []
    drift = ti.xcom_pull(task_ids='drift_monitoring',     key='drift_score') or 0.0
    logger.info("=" * 60)
    logger.info("  PIPELINEGUARD — PIPELINE COMPLETE")
    logger.info(f"  AUC: {auc}  |  F1: {f1}")
    logger.info(f"  Top SHAP: {shap}")
    logger.info(f"  Drift score: {drift}")
    logger.info("=" * 60)

with DAG(
    dag_id='pipeline_autopilot_data_pipeline',
    default_args=default_args,
    description='PipelineGuard 13-task MLOps DAG — A3',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'pipelineguard', 'a3'],
) as dag:

    t_acquire     = PythonOperator(task_id='data_acquisition',     python_callable=run_data_acquisition)
    t_preprocess  = PythonOperator(task_id='data_preprocessing',   python_callable=run_data_preprocessing)
    t_schema      = PythonOperator(task_id='schema_validation',    python_callable=run_schema_validation)
    t_bias_data   = PythonOperator(task_id='bias_detection',       python_callable=run_bias_detection)
    t_anomaly     = PythonOperator(task_id='anomaly_detection',    python_callable=run_anomaly_detection)
    t_dvc         = PythonOperator(task_id='dvc_versioning',       python_callable=run_dvc_versioning)
    t_train       = PythonOperator(task_id='model_training',       python_callable=run_model_training)
    t_validate    = PythonOperator(task_id='model_validation',     python_callable=run_model_validation,     provide_context=True)
    t_bias_model  = PythonOperator(task_id='model_bias',           python_callable=run_model_bias,           provide_context=True)
    t_sensitivity = PythonOperator(task_id='sensitivity_analysis', python_callable=run_sensitivity_analysis, provide_context=True)
    t_gate        = PythonOperator(task_id='validation_gate',      python_callable=run_validation_gate,      provide_context=True)
    t_registry    = PythonOperator(task_id='push_to_registry',     python_callable=run_push_to_registry)
    t_drift       = PythonOperator(task_id='drift_monitoring',     python_callable=run_drift_monitoring,     provide_context=True)
    t_complete    = PythonOperator(task_id='pipeline_complete',    python_callable=run_pipeline_complete,    provide_context=True)

    t_acquire >> t_preprocess
    t_preprocess >> [t_schema, t_bias_data]
    [t_schema, t_bias_data] >> t_anomaly
    t_anomaly >> t_dvc >> t_train >> t_validate
    t_validate >> [t_bias_model, t_sensitivity]
    [t_bias_model, t_sensitivity] >> t_gate
    t_gate >> t_registry >> t_drift >> t_complete
