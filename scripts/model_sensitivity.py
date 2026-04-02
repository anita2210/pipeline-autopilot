"""
model_sensitivity.py

Sensitivity Analysis (SHAP) & Feature Importance for Pipeline Autopilot.

Member 5 — ML Engineer
Deliverables:
  - models/sensitivity/shap_summary.png
  - models/sensitivity/shap_bar.png
  - models/sensitivity/feature_importance_comparison.png
  - models/sensitivity/hyperparameter_sensitivity/sensitivity_<param>.png
  - models/sensitivity/hyperparameter_sensitivity_combined.png
  - models/sensitivity/sensitivity_analysis_summary.json
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_DIR,
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
)

# =============================================================================
# PATHS
# =============================================================================

MODELS_DIR = BASE_DIR / "models"
SENSITIVITY_DIR = MODELS_DIR / "sensitivity"
TRAINED_DIR = MODELS_DIR / "trained"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = TRAINED_DIR / "model_metadata.json"

SHAP_SUMMARY_PATH = SENSITIVITY_DIR / "shap_summary.png"
SHAP_BAR_PATH = SENSITIVITY_DIR / "shap_bar.png"
FEATURE_IMPORTANCE_COMPARISON_PATH = SENSITIVITY_DIR / "feature_importance_comparison.png"
HYPERPARAM_SENSITIVITY_DIR = SENSITIVITY_DIR / "hyperparameter_sensitivity"

# =============================================================================
# SETTINGS
# =============================================================================

SHAP_SAMPLE_SIZE = 1000
SHAP_MAX_DISPLAY = 20

HYPERPARAM_RANGES = {
    "max_depth":     [3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "n_estimators":  [50, 100, 150, 200, 250, 300, 400, 500],
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SETUP
# =============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist."""
    SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
    HYPERPARAM_SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories verified: {SENSITIVITY_DIR}")


def load_model() -> Any:
    """Load the trained XGBoost model."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")
    model = joblib.load(BEST_MODEL_PATH)
    logger.info(f"Model loaded from {BEST_MODEL_PATH}")
    return model


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed data and split into features and target."""
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(f"Data not found: {PROCESSED_DATA_FILE}")

    df = pd.read_csv(PROCESSED_DATA_FILE)
    logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

    drop_cols = [TARGET_COLUMN, 'run_id', 'trigger_time', 'failure_type', 'error_message']
    available_features = [col for col in df.columns if col not in drop_cols]

    X = df[available_features].select_dtypes(include=[np.number])
    y = df[TARGET_COLUMN]

    logger.info(f"Features: {X.shape[1]} | Target: {TARGET_COLUMN} | Positive rate: {y.mean():.2%}")
    return X, y


# =============================================================================
# TASK 1: SHAP Analysis
# =============================================================================

def compute_shap_values(
    model: Any, X: pd.DataFrame, sample_size: int = SHAP_SAMPLE_SIZE
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for the model on a sample of X."""
    logger.info(f"Computing SHAP values (sample size: {sample_size})...")

    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    logger.info(f"SHAP values computed for {len(X_sample)} samples")
    return shap_values, X_sample


def generate_shap_summary_plot(shap_values: np.ndarray, X_sample: pd.DataFrame) -> str:
    """Generate SHAP beeswarm summary plot."""
    logger.info("Generating SHAP summary plot (beeswarm)...")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=SHAP_MAX_DISPLAY,
        show=False,
        plot_size=(12, 10)
    )
    plt.title("SHAP Summary Plot — Feature Impact on Pipeline Failure Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP summary plot saved: {SHAP_SUMMARY_PATH}")
    return str(SHAP_SUMMARY_PATH)


def generate_shap_bar_plot(shap_values: np.ndarray, X_sample: pd.DataFrame) -> str:
    """Generate SHAP bar plot (mean absolute SHAP values)."""
    logger.info("Generating SHAP bar plot...")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar",
        max_display=SHAP_MAX_DISPLAY,
        show=False
    )
    plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=14)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP bar plot saved: {SHAP_BAR_PATH}")
    return str(SHAP_BAR_PATH)


# =============================================================================
# TASK 2: Feature Importance Comparison
# =============================================================================

def get_xgboost_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Get XGBoost built-in feature importance (gain)."""
    return pd.DataFrame({
        'feature': feature_names,
        'xgboost_importance': model.feature_importances_
    }).sort_values('xgboost_importance', ascending=False)


def get_shap_feature_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Get SHAP-based feature importance (mean |SHAP|)."""
    return pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)


def generate_feature_importance_comparison(
    model: Any,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 15
) -> str:
    """Generate side-by-side comparison of SHAP vs XGBoost feature importance."""
    logger.info("Generating feature importance comparison plot...")

    feature_names = X_sample.columns.tolist()

    xgb_imp  = get_xgboost_feature_importance(model, feature_names)
    shap_imp = get_shap_feature_importance(shap_values, feature_names)

    df = xgb_imp.merge(shap_imp, on='feature')
    df['xgboost_normalized'] = df['xgboost_importance'] / df['xgboost_importance'].max()
    df['shap_normalized']    = df['shap_importance']    / df['shap_importance'].max()
    df['avg_importance']     = (df['xgboost_normalized'] + df['shap_normalized']) / 2
    df = df.sort_values('avg_importance', ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    x = np.arange(len(df))
    w = 0.35
    axes[0].barh(x - w/2, df['xgboost_normalized'], w, label='XGBoost (Gain)',      color='#3498db', alpha=0.8)
    axes[0].barh(x + w/2, df['shap_normalized'],    w, label='SHAP (Mean |Value|)', color='#e74c3c', alpha=0.8)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(df['feature'])
    axes[0].set_xlabel('Normalized Importance')
    axes[0].set_title('Feature Importance: XGBoost vs SHAP', fontsize=12)
    axes[0].legend()
    axes[0].invert_yaxis()

    axes[1].scatter(df['xgboost_normalized'], df['shap_normalized'],
                    s=100, alpha=0.7, c='#2ecc71', edgecolors='black')
    for _, row in df.iterrows():
        axes[1].annotate(row['feature'][:15],
                         (row['xgboost_normalized'], row['shap_normalized']),
                         fontsize=8, alpha=0.7)
    max_val = max(df['xgboost_normalized'].max(), df['shap_normalized'].max())
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
    axes[1].set_xlabel('XGBoost Importance (Normalized)')
    axes[1].set_ylabel('SHAP Importance (Normalized)')
    axes[1].set_title('Correlation: XGBoost vs SHAP', fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_COMPARISON_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    # Also save as JSON
    df.to_json(SENSITIVITY_DIR / "feature_importance_comparison.json",
               orient='records', indent=2)

    logger.info(f"Feature importance comparison saved: {FEATURE_IMPORTANCE_COMPARISON_PATH}")
    return str(FEATURE_IMPORTANCE_COMPARISON_PATH)


# =============================================================================
# TASK 3: Hyperparameter Sensitivity
# =============================================================================

def run_hyperparameter_sensitivity(
    X: pd.DataFrame,
    y: pd.Series,
    base_params: Optional[Dict] = None,
    cv_folds: int = 3
) -> Dict[str, pd.DataFrame]:
    """Vary top 3 hyperparameters independently, record CV AUC per value."""
    logger.info("Running hyperparameter sensitivity analysis...")

    if base_params is None:
        base_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
        }

    # Subsample for speed
    sample_size = min(10000, len(X))
    X_s, _, y_s, _ = train_test_split(X, y, train_size=sample_size,
                                       stratify=y, random_state=42)
    logger.info(f"Using {len(X_s)} samples for sensitivity analysis")

    results = {}
    for param_name, param_values in HYPERPARAM_RANGES.items():
        logger.info(f"  Sweeping {param_name}: {param_values}")
        rows = []
        for value in param_values:
            params = {**base_params, param_name: value}
            try:
                scores = cross_val_score(
                    xgb.XGBClassifier(**params), X_s, y_s,
                    cv=cv_folds, scoring='roc_auc', n_jobs=-1
                )
                rows.append({'param_value': value,
                             'mean_auc': scores.mean(),
                             'std_auc':  scores.std()})
                logger.info(f"    {param_name}={value} → AUC={scores.mean():.4f}")
            except Exception as e:
                logger.warning(f"    Skipped {param_name}={value}: {e}")
        results[param_name] = pd.DataFrame(rows)

    return results


def generate_hyperparameter_sensitivity_plots(
    results: Dict[str, pd.DataFrame]
) -> List[str]:
    """Generate individual + combined sensitivity plots."""
    logger.info("Generating hyperparameter sensitivity plots...")
    saved = []

    for param_name, df in results.items():
        if df.empty:
            continue
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['param_value'], df['mean_auc'], yerr=df['std_auc'],
                     marker='o', markersize=8, capsize=5, linewidth=2, color='#3498db')
        best = df.loc[df['mean_auc'].idxmax()]
        plt.scatter([best['param_value']], [best['mean_auc']],
                    color='#e74c3c', s=200, zorder=5,
                    label=f"Best: {best['param_value']} (AUC={best['mean_auc']:.4f})")
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('Mean AUC-ROC', fontsize=12)
        plt.title(f'Hyperparameter Sensitivity: {param_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = HYPERPARAM_SENSITIVITY_DIR / f"sensitivity_{param_name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved.append(str(path))
        logger.info(f"Saved: {path}")

    # Combined 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (param_name, df) in enumerate(results.items()):
        if df.empty or idx >= 3:
            continue
        ax = axes[idx]
        ax.errorbar(df['param_value'], df['mean_auc'], yerr=df['std_auc'],
                    marker='o', capsize=5, linewidth=2)
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('Mean AUC-ROC', fontsize=11)
        ax.set_title(f'{param_name} Sensitivity', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Hyperparameter Sensitivity Analysis — XGBoost', fontsize=14, y=1.02)
    plt.tight_layout()
    combined = SENSITIVITY_DIR / "hyperparameter_sensitivity_combined.png"
    plt.savefig(combined, dpi=150, bbox_inches='tight')
    plt.close()
    saved.append(str(combined))
    logger.info(f"Combined sensitivity plot saved: {combined}")

    return saved


# =============================================================================
# MAIN
# =============================================================================

def run_sensitivity_analysis() -> Dict[str, Any]:
    """Run the complete sensitivity analysis pipeline."""
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS — Starting")
    logger.info("=" * 60)

    results = {
        "status": "success",
        "shap_plots": [],
        "feature_importance": {},
        "hyperparameter_sensitivity": {},
        "output_paths": []
    }

    try:
        ensure_directories()
        model = load_model()
        X, y  = load_data()

        # Task 1: SHAP
        logger.info("Task 1: SHAP Analysis")
        shap_values, X_sample = compute_shap_values(model, X)

        path = generate_shap_summary_plot(shap_values, X_sample)
        results["shap_plots"].append(path)
        results["output_paths"].append(path)

        path = generate_shap_bar_plot(shap_values, X_sample)
        results["shap_plots"].append(path)
        results["output_paths"].append(path)

        # Task 2: Feature importance comparison
        logger.info("Task 2: Feature Importance Comparison")
        path = generate_feature_importance_comparison(model, shap_values, X_sample)
        results["feature_importance"]["comparison_plot"] = path
        results["output_paths"].append(path)

        # Task 3: Hyperparameter sensitivity
        logger.info("Task 3: Hyperparameter Sensitivity Analysis")
        sensitivity_results = run_hyperparameter_sensitivity(X, y)
        sensitivity_plots   = generate_hyperparameter_sensitivity_plots(sensitivity_results)

        results["hyperparameter_sensitivity"] = {
            param: df.to_dict('records')
            for param, df in sensitivity_results.items()
        }
        results["output_paths"].extend(sensitivity_plots)

        # Save summary JSON
        summary_path = SENSITIVITY_DIR / "sensitivity_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("SENSITIVITY ANALYSIS COMPLETE")
        logger.info(f"Output directory : {SENSITIVITY_DIR}")
        logger.info(f"Plots generated  : {len(results['output_paths'])}")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}", exc_info=True)
        results["status"] = "failed"
        results["error"]  = str(e)
        return results


if __name__ == "__main__":
    results = run_sensitivity_analysis()
    if results["status"] == "success":
        print("\nSensitivity analysis completed successfully!")
        print(f"Check outputs in: {SENSITIVITY_DIR}")
    else:
        print(f"\nAnalysis failed: {results.get('error', 'Unknown error')}")