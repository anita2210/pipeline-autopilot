import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics         import (roc_auc_score, f1_score, precision_score,
                                     recall_score, accuracy_score, average_precision_score,
                                     classification_report)
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
import xgboost as xgb

from experiment_tracking import (setup_mlflow, log_experiment,
                                  save_auc_comparison_chart, register_best_model)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "processed" / "final_dataset_processed.csv"
MODELS_DIR = BASE_DIR / "models" / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models" / "registry").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models" / "sensitivity").mkdir(parents=True, exist_ok=True)

TARGET_COL  = "failed"
RANDOM_SEED = 42
DROP_COLS   = ["run_id","trigger_time","failure_type","error_message","pipeline_name","repo"]


def load_data(path=DATA_PATH):
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
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


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED)
    logger.info(f"Split -> Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)), 6),
        "auc_pr":    round(float(average_precision_score(y_true, y_prob)), 6),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
    }


def compute_scale_pos_weight(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / n_pos
    logger.info(f"scale_pos_weight = {spw:.2f}  (neg={n_neg}, pos={n_pos})")
    return spw


def train_logistic_regression(X_train, X_val, y_train, y_val):
    logger.info("Training Logistic Regression baseline...")
    params = {"model":"logistic_regression","C":0.1,"max_iter":1000,"class_weight":"balanced","solver":"lbfgs"}
    pipe   = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced",
                                                  solver="lbfgs", random_state=RANDOM_SEED))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    y_prob = pipe.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_prob)
    logger.info(f"  LR  AUC-ROC={metrics['auc_roc']:.4f}  F1={metrics['f1']:.4f}")
    return pipe, metrics, params


def train_random_forest(X_train, X_val, y_train, y_val):
    logger.info("Training Random Forest baseline...")
    params = {"model":"random_forest","n_estimators":200,"max_depth":15,"min_samples_leaf":5,"class_weight":"balanced"}
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5,
                                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    y_prob = rf.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_prob)
    logger.info(f"  RF  AUC-ROC={metrics['auc_roc']:.4f}  F1={metrics['f1']:.4f}")
    return rf, metrics, params


def train_xgboost_default(X_train, X_val, y_train, y_val):
    logger.info("Training XGBoost (default + scale_pos_weight)...")
    spw    = compute_scale_pos_weight(y_train)
    params = {"model":"xgboost_default","n_estimators":300,"max_depth":6,"learning_rate":0.1,
              "subsample":0.8,"colsample_bytree":0.8,"scale_pos_weight":round(spw,3)}
    model  = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                eval_metric="auc", random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_prob)
    logger.info(f"  XGB(default) AUC-ROC={metrics['auc_roc']:.4f}  F1={metrics['f1']:.4f}")
    return model, metrics, params


def tune_xgboost(X_train, X_val, y_train, y_val):
    logger.info("Starting XGBoost hyperparameter tuning (RandomizedSearchCV)...")
    spw = compute_scale_pos_weight(y_train)
    param_dist = {
        "n_estimators":     [200, 300, 400, 500],
        "max_depth":        [4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.1, 0.2, 0.3],
        "reg_alpha":        [0, 0.01, 0.05, 0.1],
        "reg_lambda":       [1, 1.5, 2.0, 2.5],
    }
    base_xgb = xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="auc",
                                   random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
    cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(estimator=base_xgb, param_distributions=param_dist,
                                n_iter=30, scoring="roc_auc", cv=cv,
                                random_state=RANDOM_SEED, n_jobs=-1, verbose=0, refit=True)
    search.fit(X_train, y_train)
    best_model  = search.best_estimator_
    best_cv_auc = search.best_score_
    y_pred  = best_model.predict(X_val)
    y_prob  = best_model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred, y_prob)
    metrics["cv_auc_mean"] = round(float(best_cv_auc), 6)
    best_params = {"model":"xgboost_tuned","scale_pos_weight":round(spw,3), **search.best_params_}
    logger.info(f"  XGB(tuned) CV-AUC={best_cv_auc:.4f}  Val AUC={metrics['auc_roc']:.4f}  F1={metrics['f1']:.4f}")
    return best_model, metrics, best_params


def select_and_save_best(models_dict, X_test, y_test):
    logger.info("\n── Model Comparison (Validation AUC-ROC) ──")
    best_name = max(models_dict, key=lambda k: models_dict[k][1]["auc_roc"])
    auc_table = {k: v[1]["auc_roc"] for k, v in models_dict.items()}
    for name, auc in sorted(auc_table.items(), key=lambda x: -x[1]):
        flag = " <- BEST" if name == best_name else ""
        logger.info(f"  {name:<30s}  AUC-ROC = {auc:.4f}{flag}")
    best_model, val_metrics, best_params = models_dict[best_name]
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)
    logger.info(f"\nTest set results for '{best_name}':")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred_test, target_names=["pass","fail"]))
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved -> {model_path}")
    metadata = {
        "best_model_name": best_name,
        "model_path": str(model_path),
        "trained_at": datetime.now().isoformat(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_params": best_params,
        "validation_auc_all": auc_table,
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved -> {meta_path}")
    return best_name, best_model, test_metrics


def main():
    logger.info("="*60)
    logger.info("PipelineGuard — Model Training & Selection")
    logger.info("="*60)
    setup_mlflow()
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    models_dict = {}

    lr_model, lr_metrics, lr_params = train_logistic_regression(X_train, X_val, y_train, y_val)
    models_dict["logistic_regression"] = (lr_model, lr_metrics, lr_params)
    log_experiment("logistic_regression", lr_model, lr_params, lr_metrics,
                   y_val, lr_model.predict(X_val), lr_model.predict_proba(X_val)[:,1], "sklearn")

    rf_model, rf_metrics, rf_params = train_random_forest(X_train, X_val, y_train, y_val)
    models_dict["random_forest"] = (rf_model, rf_metrics, rf_params)
    log_experiment("random_forest", rf_model, rf_params, rf_metrics,
                   y_val, rf_model.predict(X_val), rf_model.predict_proba(X_val)[:,1], "sklearn")

    xgb_model, xgb_metrics, xgb_params = train_xgboost_default(X_train, X_val, y_train, y_val)
    models_dict["xgboost_default"] = (xgb_model, xgb_metrics, xgb_params)
    log_experiment("xgboost_default", xgb_model, xgb_params, xgb_metrics,
                   y_val, xgb_model.predict(X_val), xgb_model.predict_proba(X_val)[:,1], "xgboost")

    xgb_tuned, xgb_tuned_metrics, xgb_tuned_params = tune_xgboost(X_train, X_val, y_train, y_val)
    models_dict["xgboost_tuned"] = (xgb_tuned, xgb_tuned_metrics, xgb_tuned_params)
    run_id = log_experiment("xgboost_tuned", xgb_tuned, xgb_tuned_params, xgb_tuned_metrics,
                            y_val, xgb_tuned.predict(X_val), xgb_tuned.predict_proba(X_val)[:,1], "xgboost")

    auc_scores = {k: v[1]["auc_roc"] for k, v in models_dict.items()}
    save_auc_comparison_chart(auc_scores)

    best_name, best_model, test_metrics = select_and_save_best(models_dict, X_test, y_test)

    if "xgboost" in best_name:
        register_best_model(run_id, model_name="pipelineguard-xgboost")

    logger.info("\n" + "="*60)
    logger.info(f"Training complete. Best model: {best_name}")
    logger.info(f"  Test AUC-ROC : {test_metrics['auc_roc']:.4f}")
    logger.info(f"  Test F1      : {test_metrics['f1']:.4f}")
    logger.info(f"  Saved to     : {MODELS_DIR / 'best_model.joblib'}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
