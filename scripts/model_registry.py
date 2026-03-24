"""
model_registry.py

Push Best Model to GCP Artifact Registry (or GCS bucket fallback).

Member 5 — ML Engineer
Deliverables:
  - models/registry/registry_manifest.json
  - Model pushed to GCP Artifact Registry OR GCS bucket

Environment variables (set in .env):
  GCP_PROJECT_ID                 — your GCP project id
  GCP_REGION                     — e.g. us-central1
  GCP_REPOSITORY                 — Artifact Registry repo name
  GCS_BUCKET                     — fallback GCS bucket name
  GOOGLE_APPLICATION_CREDENTIALS — path to service account JSON key

Usage:
  python scripts/model_registry.py            # real push
  python scripts/model_registry.py --dry-run  # local test, no GCP
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_DIR,
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
)

# =============================================================================
# PATHS
# =============================================================================

MODELS_DIR      = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH   = MODELS_DIR / "trained" / "model_metadata.json"
REGISTRY_DIR    = MODELS_DIR / "registry"
VALIDATION_RPT  = BASE_DIR / "data" / "reports" / "validation_report.json"

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GCP CONFIG (from environment)
# =============================================================================

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID",  "your-gcp-project-id")
GCP_REGION     = os.getenv("GCP_REGION",      "us-central1")
GCP_REPOSITORY = os.getenv("GCP_REPOSITORY",  "pipelineguard-models")
GCS_BUCKET     = os.getenv("GCS_BUCKET",      "pipelineguard-model-bucket")
MODEL_NAME     = "pipelineguard-failure-predictor"

MIN_AUC = 0.85

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def compute_model_hash(path: Path) -> str:
    """SHA-256 hash (first 16 chars) of model file for integrity/versioning."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_version_tag() -> str:
    today = datetime.date.today().strftime("%Y%m%d")
    h     = compute_model_hash(BEST_MODEL_PATH)
    return f"v{today}-{h}"


def load_metrics() -> Dict:
    """Load AUC from validation report → model metadata → recompute (priority order)."""
    # 1. validation_report.json
    if VALIDATION_RPT.exists():
        with open(VALIDATION_RPT) as f:
            rpt = json.load(f)
        auc = rpt.get("auc_roc") or rpt.get("metrics", {}).get("auc_roc")
        if auc:
            return {"auc_roc": float(auc), "source": "validation_report.json"}

    # 2. model_metadata.json
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        auc = (meta.get("metrics", {}).get("auc_roc")
               or meta.get("test_auc") or meta.get("auc"))
        if auc:
            return {"auc_roc": float(auc), "source": "model_metadata.json"}

    # 3. Recompute on test set
    logger.warning("No cached metrics found — recomputing AUC on test set.")
    return _recompute_auc()


def _recompute_auc() -> Dict:
    df = pd.read_csv(PROCESSED_DATA_FILE)
    drop = [TARGET_COLUMN, 'run_id', 'trigger_time', 'failure_type', 'error_message']
    drop = [c for c in drop if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    model  = joblib.load(BEST_MODEL_PATH)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = float(roc_auc_score(y_test, y_prob))
    return {"auc_roc": round(auc, 6), "source": "recomputed"}


# =============================================================================
# VALIDATION GATE
# =============================================================================

def run_validation_gate(metrics: Dict) -> bool:
    """Block push if AUC is below minimum threshold."""
    auc = metrics.get("auc_roc", 0.0)
    if auc < MIN_AUC:
        logger.error("VALIDATION GATE FAILED: AUC %.4f < threshold %.2f", auc, MIN_AUC)
        return False
    logger.info("Validation gate PASSED: AUC %.4f >= %.2f", auc, MIN_AUC)
    return True


# =============================================================================
# PUSH STRATEGIES
# =============================================================================

def push_to_gcs(version_tag: str, metrics: Dict) -> Dict:
    """Upload model + metadata to GCS bucket."""
    try:
        from google.cloud import storage as gcs
    except ImportError:
        raise ImportError("Install google-cloud-storage: pip install google-cloud-storage")

    logger.info("Pushing to GCS: gs://%s/%s/%s", GCS_BUCKET, MODEL_NAME, version_tag)
    client = gcs.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    prefix = f"{MODEL_NAME}/{version_tag}"

    model_blob = bucket.blob(f"{prefix}/best_model.joblib")
    model_blob.upload_from_filename(str(BEST_MODEL_PATH))
    model_uri = f"gs://{GCS_BUCKET}/{prefix}/best_model.joblib"
    logger.info("Uploaded model → %s", model_uri)

    if METADATA_PATH.exists():
        bucket.blob(f"{prefix}/model_metadata.json").upload_from_filename(str(METADATA_PATH))

    return {"registry_type": "gcs", "bucket": GCS_BUCKET, "model_uri": model_uri}


def push_to_artifact_registry(version_tag: str, metrics: Dict) -> Dict:
    """Push to GCP Artifact Registry via gcloud CLI; falls back to GCS on failure."""
    try:
        import subprocess, tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shutil.copy(BEST_MODEL_PATH, tmp / "best_model.joblib")
            if METADATA_PATH.exists():
                shutil.copy(METADATA_PATH, tmp / "model_metadata.json")

            manifest = {
                "model_name": MODEL_NAME,
                "version":    version_tag,
                "metrics":    metrics,
                "model_hash": compute_model_hash(BEST_MODEL_PATH),
                "pushed_at":  datetime.datetime.utcnow().isoformat() + "Z",
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))

            cmd = [
                "gcloud", "artifacts", "generic", "upload",
                f"--project={GCP_PROJECT_ID}",
                f"--location={GCP_REGION}",
                f"--repository={GCP_REPOSITORY}",
                f"--package={MODEL_NAME}",
                f"--version={version_tag}",
                f"--source={tmpdir}",
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("gcloud output: %s", result.stdout.strip())

        registry_uri = (
            f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/"
            f"{GCP_REPOSITORY}/{MODEL_NAME}:{version_tag}"
        )
        return {
            "registry_type": "artifact_registry",
            "project":       GCP_PROJECT_ID,
            "region":        GCP_REGION,
            "repository":    GCP_REPOSITORY,
            "model_name":    MODEL_NAME,
            "version_tag":   version_tag,
            "registry_uri":  registry_uri,
        }

    except Exception as e:
        logger.warning("Artifact Registry push failed (%s) — falling back to GCS.", e)
        return push_to_gcs(version_tag, metrics)


# =============================================================================
# LOCAL REGISTRY COPY
# =============================================================================

def save_local_registry_copy(version_tag: str, metrics: Dict,
                              push_info: Dict) -> Path:
    """Always save a versioned local copy regardless of GCP outcome."""
    version_dir = REGISTRY_DIR / version_tag
    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(BEST_MODEL_PATH, version_dir / "best_model.joblib")

    manifest = {
        "model_name":       MODEL_NAME,
        "version":          version_tag,
        "pushed_at":        datetime.datetime.utcnow().isoformat() + "Z",
        "metrics":          metrics,
        "model_hash":       compute_model_hash(BEST_MODEL_PATH),
        "min_auc_gate":     MIN_AUC,
        "push_info":        push_info,
        "model_path_local": str(BEST_MODEL_PATH),
    }
    manifest_path = REGISTRY_DIR / "registry_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Local registry copy saved: %s", version_dir)
    logger.info("Registry manifest: %s", manifest_path)
    return manifest_path


# =============================================================================
# MAIN
# =============================================================================

def run_model_registry(dry_run: bool = False) -> Dict[str, Any]:
    """Full registry push flow."""
    logger.info("=" * 60)
    logger.info("MODEL REGISTRY PUSH — Starting")
    logger.info("=" * 60)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"best_model.joblib not found at {BEST_MODEL_PATH}. "
            "Run model_training.py (Member 1) first."
        )

    metrics     = load_metrics()
    version_tag = build_version_tag()

    logger.info("Version tag : %s", version_tag)
    logger.info("AUC         : %.4f (source: %s)",
                metrics["auc_roc"], metrics.get("source", "?"))

    # Validation gate
    if not run_validation_gate(metrics):
        return {"status": "blocked", "reason": "validation_gate_failed",
                "metrics": metrics}

    # Push to GCP (or dry run)
    if dry_run or GCP_PROJECT_ID == "your-gcp-project-id":
        logger.warning("DRY RUN — skipping actual GCP upload.")
        push_info = {"registry_type": "dry_run",
                     "model_uri": f"[dry-run] {MODEL_NAME}:{version_tag}"}
    else:
        push_info = push_to_artifact_registry(version_tag, metrics)

    # Always save local copy + manifest
    manifest_path = save_local_registry_copy(version_tag, metrics, push_info)

    result = {
        "status":        "success",
        "version_tag":   version_tag,
        "metrics":       metrics,
        "push_info":     push_info,
        "manifest_path": str(manifest_path),
    }

    logger.info("=" * 60)
    logger.info("REGISTRY PUSH COMPLETE")
    logger.info("  Version    : %s", version_tag)
    logger.info("  AUC        : %.4f", metrics["auc_roc"])
    logger.info("  Destination: %s", push_info.get("model_uri", push_info))
    logger.info("  Manifest   : %s", manifest_path)
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push best model to GCP registry.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip actual GCP upload (test locally).")
    args = parser.parse_args()
    run_model_registry(dry_run=args.dry_run)