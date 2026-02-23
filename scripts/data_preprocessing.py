"""
Data Preprocessing Pipeline for Pipeline Autopilot
Handles missing values, duplicates, constraints, outliers, datetime parsing, and encoding.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("preprocessing")

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Paths ---
RAW_DATA_FILE = "final_dataset.csv"
PROCESSED_DATA_FILE = "final_dataset_processed.csv"
SUMMARY_FILE = "preprocessing_summary.json"

# --- Column Definitions ---
TARGET_COLUMN = "failed"

CATEGORICAL_COLUMNS = [
    "pipeline_name", "repo", "head_branch", "trigger_type",
    "failure_type", "error_message",
]

HIGH_CARDINALITY_COLUMNS = [
    "pipeline_name", "head_branch", "error_message",
]

LOW_CARDINALITY_COLUMNS = [
    "repo", "trigger_type", "failure_type",
]

NUMERICAL_COLUMNS = [
    "day_of_week", "hour", "is_weekend",
    "duration_seconds", "avg_duration_7_runs", "duration_deviation",
    "prev_run_status", "failures_last_7_runs", "workflow_failure_rate",
    "hours_since_last_run", "total_jobs", "failed_jobs",
    "retry_count", "concurrent_runs",
    "is_main_branch", "is_first_run", "is_bot_triggered",
]

BINARY_COLUMNS = ["is_weekend", "is_main_branch", "is_first_run", "is_bot_triggered"]
ID_COLUMNS = ["run_id"]
DATETIME_COLUMNS = ["trigger_time"]

# --- Validation Rules ---
VALIDATION_RULES = {
    "duration_seconds": {"min": 0, "max": 86400},
    "workflow_failure_rate": {"min": 0.0, "max": 1.0},
    "day_of_week": {"min": 0, "max": 6},
    "hour": {"min": 0, "max": 23},
    "total_jobs": {"min": 1, "max": 30},
    "failed_jobs": {"min": 0, "max": 30},
    "retry_count": {"min": 0, "max": 24},
    "concurrent_runs": {"min": 0},
    "hours_since_last_run": {"min": 0},
    "failures_last_7_runs": {"min": 0, "max": 7},
    "avg_duration_7_runs": {"min": 0},
    "prev_run_status": {"allowed_values": [0.0, 1.0]},
    "failed_jobs_leq_total": True,
}

# --- Outlier Settings ---
OUTLIER_CAP_COLUMNS = [
    "duration_seconds", "avg_duration_7_runs",
    "hours_since_last_run", "concurrent_runs",
]
OUTLIER_IQR_MULTIPLIER = 3.0


# ============================================================================
# FUNCTIONS (Can be imported by tests)
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data with validation."""
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, dtype={"run_id": str}, low_memory=False)

    if df.empty:
        raise ValueError(f"Loaded file is empty: {filepath}")

    expected = set(
        ID_COLUMNS + CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
        + DATETIME_COLUMNS + [TARGET_COLUMN]
    )
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    logger.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} cols from {filepath.name}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: median for numerical, mode for categorical."""
    initial_nulls = df.isnull().sum().sum()

    if initial_nulls == 0:
        logger.info("No missing values found.")
        return df

    logger.warning(f"Found {initial_nulls} total missing values.")

    # Drop rows with missing target
    if df[TARGET_COLUMN].isnull().any():
        before = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        logger.warning(f"Dropped {before - len(df)} rows with missing target")

    # Fill numerical with median
    for col in NUMERICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {count} nulls in '{col}' with median={median_val:.4f}")

    # Fill categorical with mode
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled {count} nulls in '{col}' with mode='{mode_val}'")

    remaining = df.isnull().sum().sum()
    logger.info(f"Missing values: {initial_nulls} → {remaining}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on run_id."""
    before = len(df)
    df = df.drop_duplicates(subset=ID_COLUMNS, keep="first")
    removed = before - len(df)

    if removed > 0:
        logger.warning(f"Removed {removed} duplicate rows based on {ID_COLUMNS}")
    else:
        logger.info("No duplicate run_id values found.")

    return df


def validate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and cast data types."""
    dtype_fixes = 0

    if "run_id" in df.columns and not pd.api.types.is_string_dtype(df["run_id"]):
        df["run_id"] = df["run_id"].astype(str)
        dtype_fixes += 1

    for col in NUMERICAL_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            dtype_fixes += 1

    for col in BINARY_COLUMNS:
        if col in df.columns and df[col].dtype != np.int64:
            df[col] = df[col].astype(np.int64)
            dtype_fixes += 1

    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype != np.int64:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.int64)
        dtype_fixes += 1

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
            dtype_fixes += 1

    logger.info(f"Dtype validation complete. Fixed {dtype_fixes} column(s).")
    return df


def enforce_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply validation rules and fix constraint violations."""
    total_violations = 0

    # Range constraints
    for col, rules in VALIDATION_RULES.items():
        if not isinstance(rules, dict) or "allowed_values" in rules:
            continue
        if col not in df.columns:
            continue

        violations = 0
        if "min" in rules:
            below = (df[col] < rules["min"]).sum()
            if below > 0:
                df[col] = df[col].clip(lower=rules["min"])
                violations += below

        if "max" in rules:
            above = (df[col] > rules["max"]).sum()
            if above > 0:
                df[col] = df[col].clip(upper=rules["max"])
                violations += above

        if violations > 0:
            total_violations += violations
            logger.warning(f"Clipped {violations} values in '{col}'")

    # Allowed values check
    for col, rules in VALIDATION_RULES.items():
        if not isinstance(rules, dict) or "allowed_values" not in rules:
            continue
        if col not in df.columns:
            continue

        allowed = rules["allowed_values"]
        mask = ~df[col].isin(allowed)
        count = mask.sum()
        if count > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else allowed[0]
            df.loc[mask, col] = mode_val
            total_violations += count
            logger.warning(f"Fixed {count} values in '{col}' not in {allowed}")

    # Relational check: failed_jobs <= total_jobs
    if VALIDATION_RULES.get("failed_jobs_leq_total"):
        if "failed_jobs" in df.columns and "total_jobs" in df.columns:
            mask = df["failed_jobs"] > df["total_jobs"]
            count = mask.sum()
            if count > 0:
                df.loc[mask, "failed_jobs"] = df.loc[mask, "total_jobs"]
                total_violations += count

    if total_violations == 0:
        logger.info("All constraints passed.")
    else:
        logger.info(f"Constraints enforced. Total fixes: {total_violations}")

    return df


def cap_outliers(
    df: pd.DataFrame,
    columns: list = None,
    multiplier: float = None
) -> pd.DataFrame:
    """Cap extreme outliers using IQR method."""
    columns = columns or OUTLIER_CAP_COLUMNS
    multiplier = multiplier or OUTLIER_IQR_MULTIPLIER
    total_capped = 0

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        col_rules = VALIDATION_RULES.get(col, {})
        if isinstance(col_rules, dict) and "min" in col_rules:
            lower = max(lower, col_rules["min"])

        before_below = (df[col] < lower).sum()
        before_above = (df[col] > upper).sum()
        capped = before_below + before_above

        if capped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_capped += capped
            logger.info(f"Capped '{col}': {capped} outliers → [{lower:.2f}, {upper:.2f}]")

    if total_capped == 0:
        logger.info("No outliers capped.")
    else:
        logger.info(f"Total capped: {total_capped}")

    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse trigger_time handling mixed formats."""
    if "trigger_time" not in df.columns:
        logger.warning("'trigger_time' not found. Skipping.")
        return df

    # Normalize format
    df["trigger_time"] = df["trigger_time"].astype(str).str.replace("T", " ", regex=False)
    df["trigger_time"] = pd.to_datetime(df["trigger_time"], errors="coerce", utc=True)

    null_dates = df["trigger_time"].isnull().sum()
    if null_dates > 0:
        logger.warning(f"{null_dates} rows have unparseable trigger_time")
    else:
        logger.info(f"All trigger_time values parsed successfully")

    valid = df["trigger_time"].notna()

    # Validate time features
    if "day_of_week" in df.columns:
        expected = df.loc[valid, "trigger_time"].dt.dayofweek
        mismatch = (df.loc[valid, "day_of_week"] != expected).sum()
        if mismatch > 0:
            df.loc[valid, "day_of_week"] = expected

    if "hour" in df.columns:
        expected = df.loc[valid, "trigger_time"].dt.hour
        mismatch = (df.loc[valid, "hour"] != expected).sum()
        if mismatch > 0:
            df.loc[valid, "hour"] = expected

    if "is_weekend" in df.columns:
        expected = (df.loc[valid, "trigger_time"].dt.dayofweek >= 5).astype(int)
        mismatch = (df.loc[valid, "is_weekend"] != expected).sum()
        if mismatch > 0:
            df.loc[valid, "is_weekend"] = expected

    return df


def encode_categoricals(
    df: pd.DataFrame,
    method: str = "frequency"
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Encode categorical variables."""
    encoding_maps = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()
        use_method = "frequency" if col in HIGH_CARDINALITY_COLUMNS else method

        if use_method == "frequency":
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f"{col}_encoded"] = df[col].map(freq_map).astype(np.float64)
            encoding_maps[col] = {
                "method": "frequency",
                "n_unique": n_unique,
                "top_5": dict(list(freq_map.items())[:5]),
            }
            logger.info(f"Frequency encoded '{col}' ({n_unique} unique)")

        elif use_method == "label":
            unique_vals = sorted(df[col].dropna().unique())
            label_map = {val: idx for idx, val in enumerate(unique_vals)}
            df[f"{col}_encoded"] = df[col].map(label_map).astype(np.int64)
            encoding_maps[col] = {
                "method": "label",
                "n_unique": n_unique,
                "top_5": dict(list(label_map.items())[:5]),
            }
            logger.info(f"Label encoded '{col}' ({n_unique} unique)")

    return df, encoding_maps


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate pre-computed features."""
    if "avg_duration_7_runs" in df.columns:
        mask = df["avg_duration_7_runs"] < 0
        if mask.sum() > 0:
            df.loc[mask, "avg_duration_7_runs"] = 0.0

    if "duration_deviation" in df.columns:
        mask = ~np.isfinite(df["duration_deviation"])
        if mask.sum() > 0:
            median_val = df.loc[np.isfinite(df["duration_deviation"]), "duration_deviation"].median()
            df.loc[mask, "duration_deviation"] = median_val

    if "failures_last_7_runs" in df.columns:
        df["failures_last_7_runs"] = df["failures_last_7_runs"].clip(0, 7)

    logger.info("Feature validation complete")
    return df


def generate_summary(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    encoding_maps: Dict[str, dict]
) -> dict:
    """Generate preprocessing summary report."""
    return {
        "raw_shape": {"rows": df_raw.shape[0], "cols": df_raw.shape[1]},
        "processed_shape": {"rows": df_processed.shape[0], "cols": df_processed.shape[1]},
        "rows_removed": df_raw.shape[0] - df_processed.shape[0],
        "columns_added": df_processed.shape[1] - df_raw.shape[1],
        "new_columns": sorted([c for c in df_processed.columns if c not in df_raw.columns]),
        "null_counts_after": int(df_processed.isnull().sum().sum()),
        "target_distribution": df_processed[TARGET_COLUMN].value_counts().to_dict(),
        "encoding_info": encoding_maps,
    }


def save_processed_data(
    df: pd.DataFrame,
    filepath: str,
    summary: Optional[dict] = None
) -> None:
    """Save processed data and summary."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
    logger.info(f"Saved: {df.shape[0]:,} rows × {df.shape[1]} cols → {filepath.name}")

    if summary:
        summary_path = filepath.parent / "preprocessing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved → {summary_path.name}")


# ============================================================================
# MAIN EXECUTION (Only runs when script is executed directly)
# ============================================================================

if __name__ == "__main__":
    # Load data
    df = load_data(RAW_DATA_FILE)
    df_raw = df.copy()

    # Apply all preprocessing steps
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = validate_dtypes(df)
    df = enforce_constraints(df)
    df = cap_outliers(df)
    df = parse_datetime(df)
    df, encoding_maps = encode_categoricals(df)
    df = validate_features(df)

    # Generate summary and save
    summary = generate_summary(df_raw, df, encoding_maps)
    save_processed_data(df, PROCESSED_DATA_FILE, summary)

    print("\n📊 PREPROCESSING COMPLETE")
    print(f"  Raw: {summary['raw_shape']}")
    print(f"  Processed: {summary['processed_shape']}")
    print(f"  Rows removed: {summary['rows_removed']}")
    print(f"  Columns added: {summary['columns_added']}")