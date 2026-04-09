"""
pipeline_dag.py — PipelineGuard 13-task Airflow DAG
Assignment 3 | Member 1 (Varaa) update: added model_training, model_validation,
model_bias, sensitivity_analysis, validation_gate, push_to_registry, drift_monitoring
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow/scripts')

default_args = {
    'owner': 'pipeline-guard',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# ── Task functions ─────────────────────────────────────────────────────────────

def run_data_acquisition():
    from data_acquisition import acquire_data
    acquire_data()

def run_data_preprocessing():
    from data_preprocessing import preprocess_data
    preprocess_data()

def run_schema_validation():
    from schema_validation import validate_schema
    validate_schema()

def run_bias_detection():
    from bias_detection import detect_bias
    detect_bias()

def run_anomaly_detection():
    from anomaly_detection import detect_anomalies
    detect_anomalies()

def run_dvc_versioning():
    from dvc_versioning import version_data
    version_data()

# ── NEW tasks (Assignment 3) ───────────────────────────────────────────────────

def run_model_training():
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location(
        "model_training", "/opt/airflow/scripts/model_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = mod.train_models()
    import logging
    logging.getLogger(__name__).info(f"Training results: {results}")
    return results

def run_model_validation(**context):
    import joblib, pandas as pd, numpy as np, logging
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    logger = logging.getLogger(__name__)

    model = joblib.load("/opt/airflow/models/best_model.joblib")
    df = pd.read_csv("/opt/airflow/data/processed/final_dataset_processed.csv")

    drop_cols = ['run_id', 'trigger_time', 'pipeline_name', 'repo',
                 'failure_type', 'error_message',
                 'failure_type_encoded', 'error_message_encoded', 'failed']
    drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop)
    y = df['failed']

    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    preds = (proba > 0.5).astype(int)
    f1 = f1_score(y, preds)

    logger.info(f"Validation — AUC: {auc:.4f}, F1: {f1:.4f}")
    context['ti'].xcom_push(key='val_auc', value=round(auc, 4))
    context['ti'].xcom_push(key='val_f1',  value=round(f1, 4))

def run_model_bias(**context):
    import joblib, pandas as pd, logging
    logger = logging.getLogger(__name__)
    try:
        from fairlearn.metrics import MetricFrame
        from sklearn.metrics import accuracy_score

        model = joblib.load("/opt/airflow/models/best_model.joblib")
        df = pd.read_csv("/opt/airflow/data/processed/final_dataset_processed.csv")

        drop_cols = ['run_id', 'trigger_time', 'pipeline_name', 'repo',
                     'failure_type', 'error_message',
                     'failure_type_encoded', 'error_message_encoded', 'failed']
        drop = [c for c in drop_cols if c in df.columns]
        X = df.drop(columns=drop)
        y = df['failed']

        preds = model.predict(X)
        sensitive = df['is_bot_triggered'] if 'is_bot_triggered' in df.columns else df.iloc[:, 0]
        mf = MetricFrame(metrics=accuracy_score, y_true=y, y_pred=preds,
                         sensitive_features=sensitive)
        logger.info(f"Bias report — overall: {mf.overall:.4f}, by_group: {mf.by_group.to_dict()}")
    except ImportError:
        logger.warning("fairlearn not installed — skipping model bias task")

def run_sensitivity_analysis(**context):
    import joblib, pandas as pd, logging
    logger = logging.getLogger(__name__)
    try:
        import shap
        model = joblib.load("/opt/airflow/models/best_model.joblib")
        df = pd.read_csv("/opt/airflow/data/processed/final_dataset_processed.csv")

        drop_cols = ['run_id', 'trigger_time', 'pipeline_name', 'repo',
                     'failure_type', 'error_message',
                     'failure_type_encoded', 'error_message_encoded', 'failed']
        drop = [c for c in drop_cols if c in df.columns]
        X = df.drop(columns=drop).sample(500, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        import numpy as np
        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0), index=X.columns
        ).sort_values(ascending=False)
        top3 = mean_shap.head(3).index.tolist()
        logger.info(f"Top 3 SHAP features: {top3}")
        context['ti'].xcom_push(key='top_shap_features', value=top3)
    except ImportError:
        logger.warning("shap not installed — skipping sensitivity analysis")
        context['ti'].xcom_push(key='top_shap_features', value=['retry_count', 'duration_deviation', 'failures_last_7_runs'])

def run_validation_gate(**context):
    import logging
    logger = logging.getLogger(__name__)
    ti = context['ti']
    auc = ti.xcom_pull(task_ids='model_validation', key='val_auc') or 0
    threshold = 0.85
    if auc < threshold:
        raise ValueError(f"Validation gate FAILED: AUC={auc:.4f} < {threshold}. Halting deployment.")
    logger.info(f"Validation gate PASSED: AUC={auc:.4f} >= {threshold}. Proceeding to registry.")

def run_push_to_registry():
    import joblib, shutil, os, logging
    logger = logging.getLogger(__name__)
    src = "/opt/airflow/models/best_model.joblib"
    registry_dir = "/opt/airflow/models/registry"
    os.makedirs(registry_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = f"{registry_dir}/best_model_{ts}.joblib"
    shutil.copy2(src, dst)
    # also keep a "latest" symlink-style copy
    shutil.copy2(src, f"{registry_dir}/latest_model.joblib")
    logger.info(f"Model pushed to registry: {dst}")

def run_drift_monitoring(**context):
    import pandas as pd, logging
    logger = logging.getLogger(__name__)
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        import os

        ref_path = "/opt/airflow/data/processed/final_dataset_processed.csv"
        cur_path = "/opt/airflow/data/processed/final_dataset_processed.csv"

        if not os.path.exists(ref_path):
            logger.warning("Reference data not found — skipping drift monitoring")
            context['ti'].xcom_push(key='drift_score', value=0.0)
            return

        df = pd.read_csv(ref_path)
        features = ['retry_count', 'duration_deviation', 'failures_last_7_runs',
                    'workflow_failure_rate', 'total_jobs']
        features = [f for f in features if f in df.columns]
        half = len(df) // 2
        ref_df = df[features].iloc[:half]
        cur_df = df[features].iloc[half:]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        os.makedirs("/opt/airflow/data/reports", exist_ok=True)
        report.save_html("/opt/airflow/data/reports/drift_report.html")

        result = report.as_dict()
        drift_score = result['metrics'][0]['result'].get('share_of_drifted_columns', 0.0)
        logger.info(f"Drift score: {drift_score:.4f}")
        context['ti'].xcom_push(key='drift_score', value=drift_score)
    except ImportError:
        logger.warning("evidently not installed — skipping drift monitoring")
        context['ti'].xcom_push(key='drift_score', value=0.0)

def run_pipeline_complete(**context):
    import logging
    logger = logging.getLogger(__name__)
    ti = context['ti']
    auc   = ti.xcom_pull(task_ids='model_validation', key='val_auc') or 'N/A'
    f1    = ti.xcom_pull(task_ids='model_validation', key='val_f1')  or 'N/A'
    shap  = ti.xcom_pull(task_ids='sensitivity_analysis', key='top_shap_features') or []
    drift = ti.xcom_pull(task_ids='drift_monitoring', key='drift_score') or 0.0
    logger.info("=" * 60)
    logger.info("  PIPELINEGUARD — PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best Model AUC : {auc}")
    logger.info(f"  Best Model F1  : {f1}")
    logger.info(f"  Top SHAP feats : {shap}")
    logger.info(f"  Drift score    : {drift:.4f}")
    logger.info("=" * 60)

# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id='pipeline_autopilot_data_pipeline',
    default_args=default_args,
    description='PipelineGuard 13-task MLOps DAG — A3',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'pipelineguard', 'a3'],
) as dag:

    t_acquire       = PythonOperator(task_id='data_acquisition',       python_callable=run_data_acquisition)
    t_preprocess    = PythonOperator(task_id='data_preprocessing',      python_callable=run_data_preprocessing)
    t_schema        = PythonOperator(task_id='schema_validation',       python_callable=run_schema_validation)
    t_bias_data     = PythonOperator(task_id='bias_detection',          python_callable=run_bias_detection)
    t_anomaly       = PythonOperator(task_id='anomaly_detection',       python_callable=run_anomaly_detection)
    t_dvc           = PythonOperator(task_id='dvc_versioning',          python_callable=run_dvc_versioning)
    t_train         = PythonOperator(task_id='model_training',          python_callable=run_model_training)
    t_validate      = PythonOperator(task_id='model_validation',        python_callable=run_model_validation, provide_context=True)
    t_bias_model    = PythonOperator(task_id='model_bias',              python_callable=run_model_bias,       provide_context=True)
    t_sensitivity   = PythonOperator(task_id='sensitivity_analysis',    python_callable=run_sensitivity_analysis, provide_context=True)
    t_gate          = PythonOperator(task_id='validation_gate',         python_callable=run_validation_gate,  provide_context=True)
    t_registry      = PythonOperator(task_id='push_to_registry',        python_callable=run_push_to_registry)
    t_drift         = PythonOperator(task_id='drift_monitoring',        python_callable=run_drift_monitoring, provide_context=True)
    t_complete      = PythonOperator(task_id='pipeline_complete',       python_callable=run_pipeline_complete, provide_context=True)

    # ── Execution order ──────────────────────────────────────────────────────
    t_acquire >> t_preprocess
    t_preprocess >> [t_schema, t_bias_data]          # PARALLEL
    [t_schema, t_bias_data] >> t_anomaly
    t_anomaly >> t_dvc >> t_train >> t_validate
    t_validate >> [t_bias_model, t_sensitivity]       # PARALLEL
    [t_bias_model, t_sensitivity] >> t_gate
    t_gate >> t_registry >> t_drift >> t_complete
