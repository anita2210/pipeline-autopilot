import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import MetricFrame
import mlflow
import xgboost

# --- FIX: Set MLflow to use a local relative path to avoid "Aishwarya%20V" error ---
os.makedirs('/opt/airflow/mlruns', exist_ok=True)
mlflow.set_tracking_uri("file:///" + "/opt/airflow/mlruns".replace("\\", "/"))

# --- STEP 2: Load Assets & Reproduce Split ---
def load_assets():
    model = joblib.load('models/best_model.joblib')
    # Using low_memory=False to stop the DtypeWarning
    df = pd.read_csv('scripts/final_dataset_processed.csv', low_memory=False)

    X = df.drop(columns=['failed', 'run_id', 'trigger_time'])
    y = df['failed']

    # Reproduce the exact 70/15/15 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # XGBOOST FIX: Select only the 17 numeric features the model was trained on
    expected_features = [
        'day_of_week', 'hour', 'is_weekend', 'duration_seconds', 
        'avg_duration_7_runs', 'duration_deviation', 'prev_run_status', 
        'failures_last_7_runs', 'workflow_failure_rate', 'hours_since_last_run', 
        'total_jobs', 'failed_jobs', 'retry_count', 'concurrent_runs', 
        'is_main_branch', 'is_first_run', 'is_bot_triggered'
    ]
    
    y_pred = model.predict(X_test[expected_features])
    
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df['prediction'] = y_pred
    
    return test_df

# --- STEP 4: Define Metrics for MetricFrame ---
def false_positive_rate(y_true, y_pred):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def false_negative_rate(y_true, y_pred):
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    return fn / (fn + tp) if (fn + tp) > 0 else 0

# --- MAIN EXECUTION ---
test_df = load_assets()

# STEP 3: Define Sensitive Features
sensitive_features = ['trigger_type', 'is_bot_triggered', 'is_main_branch', 'is_weekend', 'repo']
top_10_repos = test_df['repo'].value_counts().head(10).index.tolist()

metrics = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'fpr': false_positive_rate,
    'fnr': false_negative_rate
}

disparity_results = {}
all_metrics_frames = {}

# STEP 5: Compute Metrics and Disparity
for feature in sensitive_features:
    # Filter for repo to focus on top 10
    current_test = test_df[test_df['repo'].isin(top_10_repos)] if feature == 'repo' else test_df
    
    mf = MetricFrame(
        metrics=metrics, 
        y_true=current_test['target'], 
        y_pred=current_test['prediction'], 
        sensitive_features=current_test[feature]
    )
    all_metrics_frames[feature] = mf

    # Flag bias if ratio < 0.67 (the 1.5x disparity threshold)
    ratio = mf.ratio()
    flagged_metrics = {m: val for m, val in ratio.items() if val < 0.67}
    
    disparity_results[feature] = {
        'ratio': ratio.to_dict(),
        'flagged': len(flagged_metrics) > 0,
        'flagged_metrics': list(flagged_metrics.keys())
    }

# STEP 7: Save JSON Report
report = {
    'timestamp': datetime.now().isoformat(),
    'overall_pass': not any(res['flagged'] for res in disparity_results.values()),
    'results': disparity_results
}

os.makedirs('/opt/airflow/models/registry', exist_ok=True)
with open('models/registry/model_bias_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# STEP 8: Generate Visualizations
os.makedirs('/opt/airflow/data/reports', exist_ok=True)
for feature, mf in all_metrics_frames.items():
    mf.by_group.plot(kind='bar', figsize=(10, 5), title=f'Fairness: {feature}')
    plt.tight_layout()
    plt.savefig(f'data/reports/model_bias_{feature}.png')
    plt.close()

# STEP 9: Log to MLflow
mlflow.set_experiment("pipelineguard-model-dev")
with mlflow.start_run(run_name="model_bias_detection"):
    mlflow.log_dict(report, "model_bias_report.json")
    mlflow.log_metric("overall_bias_pass", int(report['overall_pass']))
    for feature in sensitive_features:
        mlflow.log_artifact(f'data/reports/model_bias_{feature}.png')

print("Success! Check 'models/registry/' for the report and 'data/reports/' for the 5 PNGs.")