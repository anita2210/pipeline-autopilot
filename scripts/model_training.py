"""
model_training.py
-----------------
Trains 5 models for pipeline failure prediction:
  1. Logistic Regression (baseline)
  2. Random Forest (baseline)
  3. XGBoost Default
  4. XGBoost Tuned
  5. MLP Neural Network

Selects best model by AUC-ROC, saves as best_model.joblib.

Author  : Member 2 (Anita)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, average_precision_score,
    classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  — FIXED: was "scripts/final_dataset_processed.csv"
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "processed" / "processed_dataset.csv"
MODELS_DIR  = BASE_DIR / "models" / "trained"
MODEL_PATH  = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

for d in [MODELS_DIR,
          BASE_DIR / "models" / "registry",
          BASE_DIR / "models" / "sensitivity"]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL  = "failed"
RANDOM_SEED = 42

DROP_COLS = [
    "run_id",
    "trigger_time",
    "failure_type",           # post-run metadata
    "error_message",          # post-run metadata
    "failure_type_encoded",   # post-run metadata encoded
    "error_message_encoded",  # post-run metadata encoded
    "pipeline_name",          # high cardinality ID
    "repo",                   # high cardinality ID
    "failed_jobs",            # LEAKAGE: only known after pipeline completes
                              # 96.6% of failures have failed_jobs>0, making it
                              # a near-perfect proxy for the target
]

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    logger.info(f"Loading data from: {path}")
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    path = Path(path)
    if path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    logger.info(f"Features: {X.shape[1]}  Samples: {len(y)}  Failure rate: {y.mean():.4f}")
    return X, y

# ---------------------------------------------------------------------------
# 2. Split Data
# ---------------------------------------------------------------------------
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )
    logger.info(f"Split → Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------------------------------------------------------------------------
# 3. Scale Features
# ---------------------------------------------------------------------------
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to: {SCALER_PATH}")
    return X_train_s, X_val_s, X_test_s, scaler

# ---------------------------------------------------------------------------
# 4. Evaluate helper
# ---------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model"    : name,
        "accuracy" : round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall"   : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1"       : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc"  : round(float(roc_auc_score(y_test, y_prob)), 4),
        "auc_pr"   : round(float(average_precision_score(y_test, y_prob)), 4),
    }
    logger.info(f"\n── {name} ──")
    for k, v in metrics.items():
        if k != "model":
            logger.info(f"  {k}: {v}")
    return metrics

# ---------------------------------------------------------------------------
# 5. Train all 5 models
# ---------------------------------------------------------------------------
def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test):
    # Class imbalance weight for XGBoost (~11% failure rate → scale ~8x)
    neg, pos = np.bincount(y_train)
    scale_pos = round(neg / pos, 2)
    logger.info(f"Class imbalance scale_pos_weight: {scale_pos}")

    results = {}

    # ── 1. Logistic Regression ──────────────────────────────────────────────
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1
    )
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = {"model": lr, "metrics": evaluate("Logistic Regression", lr, X_test, y_test)}

    # ── 2. Random Forest ────────────────────────────────────────────────────
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["Random Forest"] = {"model": rf, "metrics": evaluate("Random Forest", rf, X_test, y_test)}

    # ── 3. XGBoost Default ──────────────────────────────────────────────────
    logger.info("Training XGBoost Default...")
    xgb_default = XGBClassifier(
        n_estimators=200, scale_pos_weight=scale_pos,
        random_state=RANDOM_SEED, eval_metric="auc",
        use_label_encoder=False, verbosity=0
    )
    xgb_default.fit(X_train, y_train)
    results["XGBoost Default"] = {"model": xgb_default, "metrics": evaluate("XGBoost Default", xgb_default, X_test, y_test)}

    # ── 4. XGBoost Tuned (RandomizedSearchCV) ───────────────────────────────
    logger.info("Training XGBoost Tuned (RandomizedSearchCV, 3-fold)...")
    param_dist = {
        "n_estimators"    : [100, 200, 300],
        "max_depth"       : [3, 5, 7, 9],
        "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
        "subsample"       : [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos, random_state=RANDOM_SEED,
        eval_metric="auc", use_label_encoder=False, verbosity=0
    )
    xgb_tuned = RandomizedSearchCV(
        xgb_base, param_dist, n_iter=20, cv=3,
        scoring="roc_auc", random_state=RANDOM_SEED,
        n_jobs=-1, verbose=0
    )
    xgb_tuned.fit(X_train, y_train)
    best_xgb = xgb_tuned.best_estimator_
    logger.info(f"Best XGBoost params: {xgb_tuned.best_params_}")
    results["XGBoost Tuned"] = {"model": best_xgb, "metrics": evaluate("XGBoost Tuned", best_xgb, X_test, y_test)}

    # ── 5. MLP Neural Network ───────────────────────────────────────────────
    logger.info("Training MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        solver="adam", alpha=0.001, batch_size=256,
        learning_rate="adaptive", learning_rate_init=0.001,
        max_iter=100, random_state=RANDOM_SEED,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, verbose=False,
    )
    mlp.fit(X_train, y_train)
    results["MLP Neural Network"] = {"model": mlp, "metrics": evaluate("MLP Neural Network", mlp, X_test, y_test)}

    return results

# ---------------------------------------------------------------------------
# 6. Select best model by AUC-ROC
# ---------------------------------------------------------------------------
def select_best(results):
    # Break AUC ties using F1 score
    best_name = max(results, key=lambda k: (results[k]["metrics"]["auc_roc"], results[k]["metrics"]["f1"]))
    best      = results[best_name]
    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_name}  AUC={best['metrics']['auc_roc']}")
    logger.info(f"{'='*50}")
    return best_name, best["model"], best["metrics"]

# ---------------------------------------------------------------------------
# 7. Save model + metadata + comparison chart
# ---------------------------------------------------------------------------
def save_all(best_name, best_model, best_metrics, all_results, feature_names):
    # Backup old model (skip if old model used dill or incompatible pickle)
    if MODEL_PATH.exists():
        try:
            joblib.dump(joblib.load(MODEL_PATH), MODELS_DIR / "previous_model.joblib")
            logger.info("Previous model backed up.")
        except Exception as e:
            logger.warning(f"Could not backup previous model (skipping): {e}")

    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"Best model saved → {MODEL_PATH}")

    # Save feature names for FastAPI
    feat_path = MODELS_DIR / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(list(feature_names), f, indent=2)
    logger.info(f"Feature names saved → {feat_path}")

    # Metadata
    metadata = {
        "model_name" : best_name,
        "trained_at" : datetime.now().isoformat(),
        "test_metrics": best_metrics,
        "model_path" : str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "all_model_metrics": {k: v["metrics"] for k, v in all_results.items()},
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved → {meta_path}")

    # Also save to models/ root for model_registry.py compatibility
    root_meta = BASE_DIR / "models" / "previous_metrics.json"
    with open(root_meta, "w") as f:
        json.dump(best_metrics, f, indent=2)

    # Comparison bar chart
    names  = list(all_results.keys())
    aucs   = [all_results[n]["metrics"]["auc_roc"] for n in names]
    f1s    = [all_results[n]["metrics"]["f1"]       for n in names]
    x      = np.arange(len(names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, aucs, width, label="AUC-ROC", color="#4C72B0")
    ax.bar(x + width/2, f1s,  width, label="F1 Score", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — AUC-ROC vs F1")
    ax.legend()
    ax.axhline(0.85, color="red", linestyle="--", linewidth=1, label="Min AUC threshold")
    plt.tight_layout()
    chart_path = BASE_DIR / "models" / "sensitivity" / "model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"Comparison chart saved → {chart_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Pipeline Autopilot — 5-Model Training")
    logger.info("=" * 60)

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)

    all_results = train_all_models(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)
    best_name, best_model, best_metrics = select_best(all_results)
    save_all(best_name, best_model, best_metrics, all_results, X.columns)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"  Best Model : {best_name}")
    logger.info(f"  AUC-ROC    : {best_metrics['auc_roc']}")
    logger.info(f"  F1         : {best_metrics['f1']}")
    logger.info(f"  Precision  : {best_metrics['precision']}")
    logger.info(f"  Recall     : {best_metrics['recall']}")
    logger.info(f"  Saved to   : {MODEL_PATH}")
    logger.info("=" * 60)

    # Print summary table for group chat
    print("\n📊 MODEL COMPARISON SUMMARY")
    print(f"{'Model':<22} {'AUC-ROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)
    for name, res in all_results.items():
        m = res["metrics"]
        marker = " ✅" if name == best_name else ""
        print(f"{name:<22} {m['auc_roc']:>8.4f} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}{marker}")


if __name__ == "__main__":
    main()