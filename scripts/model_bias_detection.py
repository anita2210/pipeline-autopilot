import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import MetricFrame
import mlflow

BASE_DIR = "/opt/airflow" if os.environ.get("AIRFLOW_HOME") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def false_positive_rate(y_true, y_pred):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def false_negative_rate(y_true, y_pred):
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0

def load_assets():
    model_path = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
    data_path  = os.path.join(BASE_DIR, 'data', 'processed', 'final_dataset_processed.csv')
    model = joblib.load(model_path)
    df    = pd.read_csv(data_path)
    return model, df

def run_bias_detection():
    os.makedirs(os.path.join(BASE_DIR, 'mlruns'), exist_ok=True)
    mlflow.set_tracking_uri("file:///" + os.path.join(BASE_DIR, 'mlruns'))

    model, df = load_assets()

    drop_cols = ["run_id", "trigger_time", "failed", "failure_type",
                 "error_message", "pipeline_name", "repo", "head_branch"]
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
    X = df[feature_cols]
    y = df["failed"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    df_test = df.loc[X_test.index].copy()

    y_pred = model.predict(X_test)

    sensitive_features = ["trigger_type", "is_bot_triggered", "is_main_branch", "is_weekend"]
    available = [f for f in sensitive_features if f in df_test.columns]

    metrics = {
        "accuracy":          accuracy_score,
        "f1":                f1_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

    all_metrics_frames = {}
    disparity_results  = {}

    for feature in available:
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=pd.Series(y_pred, index=y_test.index),
            sensitive_features=df_test[feature],
        )
        all_metrics_frames[feature] = mf
        overall_f1 = float(mf.overall["f1"])
        min_f1     = float(mf.by_group["f1"].min())
        disparity  = overall_f1 / min_f1 if min_f1 > 0 else 0
        disparity_results[feature] = {
            "overall_f1":  round(overall_f1, 4),
            "min_group_f1": round(min_f1, 4),
            "disparity_ratio": round(disparity, 4),
            "flagged": disparity > 1.5,
        }

    report = {
        "timestamp":    datetime.now().isoformat(),
        "overall_pass": not any(r["flagged"] for r in disparity_results.values()),
        "results":      disparity_results,
    }

    os.makedirs(os.path.join(BASE_DIR, 'models', 'registry'), exist_ok=True)
    report_path = os.path.join(BASE_DIR, 'models', 'registry', 'model_bias_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    os.makedirs(os.path.join(BASE_DIR, 'data', 'reports'), exist_ok=True)
    for feature, mf in all_metrics_frames.items():
        mf.by_group.plot(kind='bar', figsize=(10, 5), title=f'Fairness: {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'data', 'reports', f'model_bias_{feature}.png'))
        plt.close()

    print(f"Bias detection complete. overall_pass={report['overall_pass']}")
    return report

if __name__ == "__main__":
    run_bias_detection()
