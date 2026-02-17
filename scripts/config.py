"""
config.py
=========
Central configuration file for Pipeline Autopilot.
Contains all paths, column definitions, validation rules, and settings.

Author: Member 1 (Pipeline Architect)
Updated: February 2026 (Added Member 3's preprocessing requirements)
Project: Pipeline Autopilot - CI/CD Failure Prediction System
"""

import os
from pathlib import Path
from datetime import timedelta

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base directory (works in both local and Docker environments)
if os.environ.get("AIRFLOW_HOME"):
    # Running inside Airflow Docker container
    BASE_DIR = Path("/opt/airflow")
else:
    # Running locally
    BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCHEMA_DIR = DATA_DIR / "schema"
REPORTS_DIR = DATA_DIR / "reports"

# Other directories
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"
TESTS_DIR = BASE_DIR / "tests"
DAGS_DIR = BASE_DIR / "dags"

# =============================================================================
# FILE PATHS
# =============================================================================

# Input files
RAW_DATA_FILE = RAW_DATA_DIR / "final_dataset.csv"
RAW_DATASET_PATH = RAW_DATA_FILE  # Alias for backward compatibility

# Output files (Updated for Member 3's preprocessing)
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "final_dataset_processed.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_FILE  # Alias for backward compatibility
TRAIN_DATASET_PATH = PROCESSED_DATA_DIR / "train_dataset.csv"
TEST_DATASET_PATH = PROCESSED_DATA_DIR / "test_dataset.csv"

# Schema files
SCHEMA_FILE_PATH = SCHEMA_DIR / "data_schema.json"
VALIDATION_REPORT_PATH = SCHEMA_DIR / "validation_report.json"

# Report files
ANOMALY_REPORT_PATH = REPORTS_DIR / "anomaly_report.json"
BIAS_REPORT_PATH = REPORTS_DIR / "bias_report.json"
PIPELINE_REPORT_PATH = REPORTS_DIR / "pipeline_report.json"
PREPROCESSING_REPORT_PATH = REPORTS_DIR / "preprocessing_report.json"

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

# Primary identifier
ID_COLUMN = "run_id"

# Target variable
TARGET_COLUMN = "failed"

# Datetime column
DATETIME_COLUMN = "trigger_time"
DATETIME_COLUMNS = ["trigger_time"]

# Categorical columns (need encoding for ML)
CATEGORICAL_COLUMNS = [
    "pipeline_name",
    "repo",
    "head_branch",
    "trigger_type",
    "failure_type",
    "error_message",
]

# NEW: High cardinality categorical columns (many unique values - use frequency encoding)
HIGH_CARDINALITY_COLUMNS = [
    "pipeline_name",
    "repo",
    "head_branch",
    "error_message",
]

# NEW: Low cardinality categorical columns (few unique values - use label encoding)
LOW_CARDINALITY_COLUMNS = [
    "trigger_type",
    "failure_type",
]

# Numerical columns (continuous + discrete)
NUMERICAL_COLUMNS = [
    "day_of_week",
    "hour",
    "is_weekend",
    "duration_seconds",
    "avg_duration_7_runs",
    "duration_deviation",
    "prev_run_status",
    "failures_last_7_runs",
    "workflow_failure_rate",
    "hours_since_last_run",
    "total_jobs",
    "failed_jobs",
    "retry_count",
    "concurrent_runs",
    "is_main_branch",
    "is_first_run",
    "is_bot_triggered",
]

# Binary columns (subset of numerical, 0/1 values)
BINARY_COLUMNS = [
    "is_weekend",
    "is_main_branch",
    "is_first_run",
    "is_bot_triggered",
    "prev_run_status",
]

# NEW: Columns to apply outlier capping (IQR method)
OUTLIER_CAP_COLUMNS = [
    "duration_seconds",
    "avg_duration_7_runs",
    "duration_deviation",
    "hours_since_last_run",
    "total_jobs",
    "failed_jobs",
    "retry_count",
    "concurrent_runs",
]

# NEW: IQR multiplier for outlier detection
OUTLIER_IQR_MULTIPLIER = 1.5

# All feature columns (excluding ID, target, datetime)
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

# All columns in expected order
ALL_COLUMNS = [ID_COLUMN, DATETIME_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# =============================================================================
# DATA VALIDATION RULES (Updated for Member 3's preprocessing)
# =============================================================================

VALIDATION_RULES = {
    # Row count expectations
    "min_row_count": 100000,
    "max_row_count": 200000,
    
    # Missing value thresholds (percentage)
    "max_missing_percent": 5.0,
    
    # Column-specific rules (UPDATED)
    "column_rules": {
        "run_id": {
            "dtype": "int64",
            "unique": True,
            "nullable": False,
        },
        "failed": {
            "dtype": "int64",
            "allowed_values": [0, 1],
            "nullable": False,
        },
        "day_of_week": {
            "dtype": "int64",
            "min_value": 0,
            "max_value": 6,
            "nullable": False,
        },
        "hour": {
            "dtype": "int64",
            "min_value": 0,
            "max_value": 23,
            "nullable": False,
        },
        "is_weekend": {
            "dtype": "int64",
            "allowed_values": [0, 1],
            "nullable": False,
        },
        "is_main_branch": {
            "dtype": "int64",
            "allowed_values": [0, 1],
            "nullable": False,
        },
        "is_first_run": {
            "dtype": "int64",
            "allowed_values": [0, 1],
            "nullable": False,
        },
        "is_bot_triggered": {
            "dtype": "int64",
            "allowed_values": [0, 1],
            "nullable": False,
        },
        # UPDATED: prev_run_status uses allowed_values instead of min/max
        "prev_run_status": {
            "dtype": "int64",
            "allowed_values": [0, 1],
            "nullable": False,
        },
        # UPDATED: duration_seconds has no upper cap
        "duration_seconds": {
            "dtype": "float64",
            "min_value": 0,
            # No max_value - will be handled by IQR outlier capping
            "nullable": False,
        },
        "avg_duration_7_runs": {
            "dtype": "float64",
            "min_value": 0,
            "nullable": False,
        },
        "duration_deviation": {
            "dtype": "float64",
            # Can be negative (faster than average)
            "nullable": False,
        },
        "workflow_failure_rate": {
            "dtype": "float64",
            "min_value": 0.0,
            "max_value": 1.0,
            "nullable": False,
        },
        "failures_last_7_runs": {
            "dtype": "float64",
            "min_value": 0,
            "max_value": 7,
            "nullable": False,
        },
        # UPDATED: hours_since_last_run clips at 0 (no negative values)
        "hours_since_last_run": {
            "dtype": "float64",
            "min_value": 0,
            # No max_value - will be handled by IQR outlier capping
            "nullable": False,
        },
        "total_jobs": {
            "dtype": "int64",
            "min_value": 1,
            "nullable": False,
        },
        "failed_jobs": {
            "dtype": "int64",
            "min_value": 0,
            "nullable": False,
        },
        "retry_count": {
            "dtype": "int64",
            "min_value": 0,
            "nullable": False,
        },
        # UPDATED: concurrent_runs has no upper cap
        "concurrent_runs": {
            "dtype": "int64",
            "min_value": 0,
            # No max_value - will be handled by IQR outlier capping
            "nullable": False,
        },
    },
    
    # Cross-column validation rules
    "cross_column_rules": {
        "failed_jobs_lte_total_jobs": {
            "description": "failed_jobs must be <= total_jobs",
            "check": "failed_jobs <= total_jobs",
        },
    },
}

# =============================================================================
# ANOMALY DETECTION SETTINGS
# =============================================================================

ANOMALY_SETTINGS = {
    # Z-score threshold for numerical columns
    "zscore_threshold": 3.0,
    
    # IQR multiplier for outlier detection
    "iqr_multiplier": OUTLIER_IQR_MULTIPLIER,
    
    # Minimum percentage of anomalies to flag dataset
    "anomaly_flag_threshold": 5.0,
    
    # Columns to check for anomalies
    "columns_to_check": OUTLIER_CAP_COLUMNS,
}

# =============================================================================
# BIAS DETECTION SETTINGS
# =============================================================================

BIAS_SETTINGS = {
    # Columns to check for bias
    "sensitive_columns": [
        "repo",
        "pipeline_name",
        "trigger_type",
        "is_weekend",
        "is_bot_triggered",
    ],
    
    # Minimum representation threshold (percentage)
    "min_representation": 1.0,
    
    # Maximum class imbalance ratio
    "max_imbalance_ratio": 10.0,
    
    # Target column for bias analysis
    "target_column": TARGET_COLUMN,
}

# =============================================================================
# PREPROCESSING SETTINGS (NEW - for Member 3)
# =============================================================================

PREPROCESSING_SETTINGS = {
    # Missing value strategies
    "missing_value_strategy": {
        "numerical": "median",
        "categorical": "mode",
        "binary": "mode",
    },
    
    # Encoding strategies
    "encoding_strategy": {
        "high_cardinality": "frequency",  # Frequency encoding
        "low_cardinality": "label",       # Label encoding
    },
    
    # Datetime parsing
    "datetime_format": "%Y-%m-%d %H:%M:%S",
    "normalize_datetime_separator": True,  # Replace "T" with " " before parsing
    
    # Outlier handling
    "outlier_method": "iqr",
    "outlier_iqr_multiplier": OUTLIER_IQR_MULTIPLIER,
    "outlier_cap_columns": OUTLIER_CAP_COLUMNS,
}

# =============================================================================
# AIRFLOW DAG SETTINGS
# =============================================================================

DAG_CONFIG = {
    "dag_id": "pipeline_autopilot_data_pipeline",
    "description": "Data pipeline for CI/CD failure prediction system",
    "schedule_interval": "@daily",
    "start_date": "2026-02-01",
    "catchup": False,
    "max_active_runs": 1,
    "tags": ["pipeline-autopilot", "data-pipeline", "mlops"],
    
    # Default task arguments
    "default_args": {
        "owner": "pipeline-architect",
        "email": ["team@pipelineautopilot.com"],
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=1),
    },
}

# =============================================================================
# DVC SETTINGS
# =============================================================================

DVC_CONFIG = {
    "remote_name": "gcs_remote",
    "remote_url": "gs://pipeline-autopilot-data",  # Update with your GCS bucket
    "tracked_files": [
        str(RAW_DATA_FILE),
        str(PROCESSED_DATA_FILE),
        str(TRAIN_DATASET_PATH),
        str(TEST_DATASET_PATH),
    ],
}

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": LOGS_DIR / "pipeline.log",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories_exist():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SCHEMA_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_path_str(path: Path) -> str:
    """Convert Path object to string for compatibility."""
    return str(path)


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("PIPELINE AUTOPILOT CONFIGURATION")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Raw Data Path: {RAW_DATA_FILE}")
    print(f"Processed Data Path: {PROCESSED_DATA_FILE}")
    print(f"Number of Features: {len(FEATURE_COLUMNS)}")
    print(f"Categorical Columns: {len(CATEGORICAL_COLUMNS)}")
    print(f"  - High Cardinality: {len(HIGH_CARDINALITY_COLUMNS)}")
    print(f"  - Low Cardinality: {len(LOW_CARDINALITY_COLUMNS)}")
    print(f"Numerical Columns: {len(NUMERICAL_COLUMNS)}")
    print(f"Binary Columns: {len(BINARY_COLUMNS)}")
    print(f"Outlier Cap Columns: {len(OUTLIER_CAP_COLUMNS)}")
    print(f"Target Column: {TARGET_COLUMN}")
    print("=" * 60)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test configuration
    print_config()
    
    # Ensure directories exist
    ensure_directories_exist()
    print("\n✅ All directories verified/created!")
    
    # Check if raw data exists
    if RAW_DATA_FILE.exists():
        print(f"✅ Raw dataset found: {RAW_DATA_FILE}")
    else:
        print(f"❌ Raw dataset NOT found: {RAW_DATA_FILE}")
    
    # Check if processed data exists
    if PROCESSED_DATA_FILE.exists():
        print(f"✅ Processed dataset found: {PROCESSED_DATA_FILE}")
    else:
        print(f"⏳ Processed dataset not yet created: {PROCESSED_DATA_FILE}")