import pandas as pd
df = pd.read_csv("final_dataset.csv")

# Quick look
print(df.head())
print(df.info())

df.info()

import pandas as pd

df = pd.read_csv("final_dataset.csv")
print(df.shape)
print(df.dtypes)
print("---")
print(df.head(3).to_string())
print("---")
print(df.describe())
print("---")
# Check unique values for categoricals
for col in ['pipeline_name', 'repo', 'trigger_type', 'failure_type', 'head_branch']:
    print(f"{col}: {df[col].nunique()} unique → {df[col].unique()[:5]}")
print("---")
print(f"Target distribution:\n{df['failed'].value_counts()}")
print(f"error_message unique: {df['error_message'].nunique()}")

# Imports & Setup
# WHY: Load all the Python libraries we need for data processing

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

# Logger setup — so that we can track what happens at each step
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("preprocessing")

# CELL 2: Configuration
# WHY: Define all column names, file paths, and rules in ONE place so nothing is hardcoded inside functions

# --- Paths ---
RAW_DATA_FILE = "final_dataset.csv"                     
PROCESSED_DATA_FILE = "final_dataset_processed.csv"
SUMMARY_FILE = "preprocessing_summary.json"

# --- Column Definitions ---
TARGET_COLUMN = "failed"

CATEGORICAL_COLUMNS = [
    "pipeline_name",       
    "repo",               
    "head_branch",         
    "trigger_type",       
    "failure_type",       
    "error_message",       
]

HIGH_CARDINALITY_COLUMNS = [
    "pipeline_name",       
    "head_branch",         
    "error_message",       
]

LOW_CARDINALITY_COLUMNS = [
    "repo",                
    "trigger_type",        
    "failure_type",       
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
    "duration_seconds": {"min": 0},                          # Can't be negative
    "workflow_failure_rate": {"min": 0.0, "max": 1.0},       # It's a rate, must be 0-1
    "day_of_week": {"min": 0, "max": 6},                     # Monday=0 to Sunday=6
    "hour": {"min": 0, "max": 23},                           # 24-hour clock
    "total_jobs": {"min": 1, "max": 30},                     # At least 1 job per run
    "failed_jobs": {"min": 0, "max": 30},                    # Can't fail more than total
    "retry_count": {"min": 0, "max": 24},                    # Max seen in data is 24
    "concurrent_runs": {"min": 0},                           # No upper cap — max=526 is real
    "hours_since_last_run": {"min": 0},                      # Has negatives! min=-64.97
    "failures_last_7_runs": {"min": 0, "max": 7},            # Max 7 failures in 7 runs
    "avg_duration_7_runs": {"min": 0},                       # Average can't be negative
    "prev_run_status": {"allowed_values": [0.0, 1.0]},       # Binary: passed or failed
    "failed_jobs_leq_total": True,                           # failed_jobs must be <= total_jobs
}

# --- Outlier Settings ---
OUTLIER_CAP_COLUMNS = [
    "duration_seconds",        # max=34.6M but 75th percentile=467 — extreme skew
    "avg_duration_7_runs",     # max=34.6M but 75th percentile=719
    "hours_since_last_run",    # max=9487 but 75th percentile=8.4
    "concurrent_runs",         # max=526 but 75th percentile=21
]
OUTLIER_IQR_MULTIPLIER = 3.0  # 3x IQR — conservative to keep real variation


# CELL 3: Load Data
# WHY: Read the CSV into a DataFrame and make sure it has all the columns we expect before doing anything else

def load_data(filepath: str) -> pd.DataFrame:
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    # dtype={"run_id": str} fixes the mixed-type warning we saw
    df = pd.read_csv(filepath, dtype={"run_id": str}, low_memory=False)

    if df.empty:
        raise ValueError(f"Loaded file is empty: {filepath}")

    # Check all expected columns are present
    expected = set(
        ID_COLUMNS + CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
        + DATETIME_COLUMNS + [TARGET_COLUMN]
    )
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    logger.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} cols from {filepath.name}")
    logger.info(f"Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    logger.info(f"Null count: {df.isnull().sum().sum()}")

    return df

df = load_data(RAW_DATA_FILE)
df_raw = df.copy()  # Keep original copy for comparison later
print(f"\nShape: {df.shape}")
print(f"Target: {df['failed'].value_counts().to_dict()}")
df.head(5)


# CELL 4: Handle Missing Values
# WHY: Our data has 0 nulls right now, but if new data comes in with missing values, this function catches and fills them
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    initial_nulls = df.isnull().sum().sum()

    if initial_nulls == 0:
        logger.info("No missing values found (0 nulls across all columns).")
        return df

    logger.warning(f"Found {initial_nulls} total missing values — applying fixes.")

    # Can't train without labels — drop rows with missing target
    if df[TARGET_COLUMN].isnull().any():
        before = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        logger.warning(f"Dropped {before - len(df)} rows with missing target")

    # Numbers → fill with middle value (median)
    for col in NUMERICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {count} nulls in '{col}' with median={median_val:.4f}")

    # Text → fill with most common value (mode)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled {count} nulls in '{col}' with mode='{mode_val}'")

    remaining = df.isnull().sum().sum()
    logger.info(f"Missing values: {initial_nulls} → {remaining}")
    return df

df = handle_missing_values(df)
print(f"\nShape after: {df.shape}")
print(f"Remaining nulls: {df.isnull().sum().sum()}")

# CELL 5: Remove Duplicates
# WHY: Each pipeline run should have a unique run_id — duplicate rows would bias our model training

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

df = remove_duplicates(df)
print(f"\nShape after: {df.shape}")
print(f"Unique run_ids: {df['run_id'].nunique():,}")


# CELL 6: Validate & Cast Data Types
# WHY: Make sure every column is the correct type (numbers are numbers,text is text) — wrong types cause errors in ML models

def validate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    dtype_fixes = 0
    # run_id should be string (it had mixed types warning)
    if "run_id" in df.columns and not pd.api.types.is_string_dtype(df["run_id"]):
        df["run_id"] = df["run_id"].astype(str)
        dtype_fixes += 1
    # Numerical columns must be numeric
    for col in NUMERICAL_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            dtype_fixes += 1
            logger.info(f"Converted '{col}' to numeric")
    # Binary columns must be integers (0 or 1)
    for col in BINARY_COLUMNS:
        if col in df.columns and df[col].dtype != np.int64:
            df[col] = df[col].astype(np.int64)
            dtype_fixes += 1
    # Target must be integer
    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype != np.int64:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.int64)
        dtype_fixes += 1
    # Categorical columns must be strings
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
            dtype_fixes += 1
    logger.info(f"Dtype validation complete. Fixed {dtype_fixes} column(s).")
    return df

print("Before:")
print(df.dtypes.to_string())
df = validate_dtypes(df)
print(f"\nAfter: {df.dtypes.value_counts().to_dict()}")

# CELL 7: Enforce Constraints
# WHY: Fix impossible values — like negative time durations,failure rates above 100%, or more failed jobs than total jobs

def enforce_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply validation rules — clip out-of-range values, fix relational constraints.

    Key fixes for this dataset:
        - hours_since_last_run has negatives (min=-64.97) → clip to 0
        - workflow_failure_rate must be [0, 1]
        - failed_jobs must be <= total_jobs
        - prev_run_status must be 0 or 1
    """
    total_violations = 0
    violation_details = {}

    # Range constraints (min/max checks)
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
            violation_details[col] = int(violations)
            logger.warning(f"Clipped {violations} values in '{col}'")

    # Allowed values check (prev_run_status must be exactly 0 or 1)
    for col, rules in VALIDATION_RULES.items():
        if not isinstance(rules, dict) or "allowed_values" not in rules:
            continue
        if col not in df.columns:
            continue

        allowed = rules["allowed_values"]
        mask = ~df[col].isin(allowed)
        count = mask.sum()
        if count > 0:
            df.loc[mask, col] = df[col].mode()[0]
            total_violations += count
            violation_details[f"{col} not in {allowed}"] = int(count)
            logger.warning(f"Fixed {count} values in '{col}' not in {allowed}")

    # Relational check: you can't fail more jobs than you ran
    if VALIDATION_RULES.get("failed_jobs_leq_total"):
        if "failed_jobs" in df.columns and "total_jobs" in df.columns:
            mask = df["failed_jobs"] > df["total_jobs"]
            count = mask.sum()
            if count > 0:
                df.loc[mask, "failed_jobs"] = df.loc[mask, "total_jobs"]
                total_violations += count
                violation_details["failed_jobs > total_jobs"] = int(count)

    if total_violations == 0:
        logger.info("All constraints passed. No violations.")
    else:
        logger.info(f"Constraints enforced. Total fixes: {total_violations}")
        logger.info(f"Breakdown: {violation_details}")

    return df

print("Before constraints:")
print(f"  hours_since_last_run min: {df['hours_since_last_run'].min():.2f}")
print(f"  failed_jobs > total_jobs: {(df['failed_jobs'] > df['total_jobs']).sum()}")

df = enforce_constraints(df)

print(f"\nAfter constraints:")
print(f"  hours_since_last_run min: {df['hours_since_last_run'].min():.2f}")
print(f"  failed_jobs > total_jobs: {(df['failed_jobs'] > df['total_jobs']).sum()}")


# CELL 8: Cap Outliers (IQR-based)
# WHY: Some columns have extreme values (duration_seconds max=34.6 MILLION) that would dominate ML models — we cap them using a statistical method(3x IQR) that keeps real variation but removes extreme tails

def cap_outliers(
    df: pd.DataFrame,
    columns: list = None,
    multiplier: float = None
) -> pd.DataFrame:
    """Cap extreme outliers using IQR method.

    IQR = Q3 - Q1 (the middle 50% spread)
    Upper bound = Q3 + 3 * IQR
    Lower bound = Q1 - 3 * IQR
    Anything beyond these bounds gets clipped.
    """
    columns = columns or OUTLIER_CAP_COLUMNS
    multiplier = multiplier or OUTLIER_IQR_MULTIPLIER
    total_capped = 0
    cap_details = {}

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        # Don't go below the minimum constraint (e.g., duration can't be < 0)
        col_rules = VALIDATION_RULES.get(col, {})
        if isinstance(col_rules, dict) and "min" in col_rules:
            lower = max(lower, col_rules["min"])

        before_below = (df[col] < lower).sum()
        before_above = (df[col] > upper).sum()
        capped = before_below + before_above

        if capped > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_capped += capped
            cap_details[col] = {
                "capped": int(capped),
                "lower": round(lower, 2),
                "upper": round(upper, 2),
                "below": int(before_below),
                "above": int(before_above),
            }
            logger.info(f"Capped '{col}': {capped} outliers → [{lower:.2f}, {upper:.2f}]")

    if total_capped == 0:
        logger.info("No outliers capped.")
    else:
        logger.info(f"Total capped: {total_capped}")

    return df

print("Before capping:")
for col in OUTLIER_CAP_COLUMNS:
    print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

df = cap_outliers(df)

print(f"\nAfter capping:")
for col in OUTLIER_CAP_COLUMNS:
    print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

# CELL 9: Parse & Validate Datetime
# WHY: trigger_time is stored as text in TWO formats —
#       "2026-02-09 10:12:46+00:00" (space separator, 100K real rows)
#       "2025-03-23T01:39:00+00:00" (T separator, 50K simulated rows)
#       We normalize both to a single datetime format, then verify
#       that day_of_week, hour, is_weekend actually match

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse trigger_time handling mixed formats and validate time features.

    Two formats in data:
        Real data (100K):      '2026-02-09 10:12:46+00:00' (space separator)
        Simulated data (50K):  '2025-03-23T01:39:00+00:00' (T separator)

    Solution: Normalize the T-separator format to space-separator first,
    then parse everything at once.
    """
    if "trigger_time" not in df.columns:
        logger.warning("'trigger_time' not found. Skipping.")
        return df

    # Step 1: Normalize — replace 'T' separator with space for consistency
    df["trigger_time"] = df["trigger_time"].astype(str).str.replace("T", " ", regex=False)
    logger.info("Normalized trigger_time format (replaced T separator with space)")

    # Step 2: Parse all at once — now both formats are the same
    df["trigger_time"] = pd.to_datetime(df["trigger_time"], errors="coerce", utc=True)
    null_dates = df["trigger_time"].isnull().sum()

    if null_dates > 0:
        logger.warning(f"{null_dates} rows still have unparseable trigger_time")
    else:
        logger.info(f"All {len(df):,} trigger_time values parsed successfully (0 nulls)")

    valid = df["trigger_time"].notna()
    mismatch_summary = {}

    # Check: does day_of_week match the actual day from trigger_time?
    if "day_of_week" in df.columns:
        expected = df.loc[valid, "trigger_time"].dt.dayofweek
        mismatch = (df.loc[valid, "day_of_week"] != expected).sum()
        if mismatch > 0:
            df.loc[valid, "day_of_week"] = expected
            mismatch_summary["day_of_week"] = int(mismatch)

    # Check: does hour match the actual hour from trigger_time?
    if "hour" in df.columns:
        expected = df.loc[valid, "trigger_time"].dt.hour
        mismatch = (df.loc[valid, "hour"] != expected).sum()
        if mismatch > 0:
            df.loc[valid, "hour"] = expected
            mismatch_summary["hour"] = int(mismatch)

    # Check: does is_weekend match (Saturday=5, Sunday=6)?
    if "is_weekend" in df.columns:
        expected = (df.loc[valid, "trigger_time"].dt.dayofweek >= 5).astype(int)
        mismatch = (df.loc[valid, "is_weekend"] != expected).sum()
        if mismatch > 0:
            df.loc[valid, "is_weekend"] = expected
            mismatch_summary["is_weekend"] = int(mismatch)

    if not mismatch_summary:
        logger.info("Datetime validation passed. All time features consistent.")
    else:
        logger.warning(f"Mismatches corrected: {mismatch_summary}")

    return df

# --- Run ---
print(f"Before: trigger_time dtype = {df['trigger_time'].dtype}")
print(f"Sample: {df['trigger_time'].iloc[0]}")

df = parse_datetime(df)

print(f"\nAfter: trigger_time dtype = {df['trigger_time'].dtype}")
print(f"Sample: {df['trigger_time'].iloc[0]}")
print(f"day_of_week range: [{df['day_of_week'].min()}, {df['day_of_week'].max()}]")
print(f"hour range: [{df['hour'].min()}, {df['hour'].max()}]")


# CELL 10: Encode Categorical Variables
# WHY: ML models can't read text — we convert category names into numbers.Frequency encoding = replace each value with how often it appears(e.g., "push" appears 40% of the time → 0.40) We keep the original text columns for bias analysis later.

def encode_categoricals(
    df: pd.DataFrame,
    method: str = "frequency"
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Encode categoricals, keeping originals for bias slicing.

    High cardinality (pipeline_name, head_branch, error_message):
        → Always frequency encoding (too many values for label encoding)
    Low cardinality (repo, trigger_type, failure_type):
        → Uses the method parameter (default: frequency)
    """
    encoding_maps = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()

        # Force frequency encoding for high cardinality columns
        use_method = "frequency" if col in HIGH_CARDINALITY_COLUMNS else method

        if use_method == "frequency":
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f"{col}_encoded"] = df[col].map(freq_map).astype(np.float64)
            encoding_maps[col] = {
                "method": "frequency",
                "n_unique": n_unique,
                "top_5": dict(list(freq_map.items())[:5]),
            }
            logger.info(f"Frequency encoded '{col}' → '{col}_encoded' ({n_unique} unique)")

        elif use_method == "label":
            unique_vals = sorted(df[col].dropna().unique())
            label_map = {val: idx for idx, val in enumerate(unique_vals)}
            df[f"{col}_encoded"] = df[col].map(label_map).astype(np.int64)
            encoding_maps[col] = {
                "method": "label",
                "n_unique": n_unique,
                "top_5": dict(list(label_map.items())[:5]),
            }
            logger.info(f"Label encoded '{col}' → '{col}_encoded' ({n_unique} unique)")

        else:
            raise ValueError(f"Unsupported method: '{use_method}'")

    logger.info(f"Encoding complete. {len(encoding_maps)} columns encoded.")
    return df, encoding_maps

# --- Run ---
print(f"Shape before encoding: {df.shape}")
df, encoding_maps = encode_categoricals(df, method="frequency")
print(f"Shape after encoding: {df.shape}")

print("\nEncoding Summary:")
for col, info in encoding_maps.items():
    print(f"  {col}: {info['method']}, {info['n_unique']} unique")
    print(f"    Top 5: {info['top_5']}")


# CELL 11: Validate Pre-computed Features
# WHY: Some columns (like avg_duration_7_runs, failures_last_7_runs) were calculated before we got the data — we double-check they make sense (no negatives, no infinities, within range)
def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate pre-computed rolling/aggregate features.

    Checks:
        - avg_duration_7_runs >= 0 (averages can't be negative)
        - duration_deviation is finite (no inf or NaN)
        - failures_last_7_runs in [0, 7] (max 7 failures in 7 runs)
    """
    fixes = 0
    fix_details = {}

    # Average duration can't be negative
    if "avg_duration_7_runs" in df.columns:
        mask = df["avg_duration_7_runs"] < 0
        count = mask.sum()
        if count > 0:
            df.loc[mask, "avg_duration_7_runs"] = 0.0
            fixes += count
            fix_details["avg_duration_7_runs < 0"] = int(count)

    # Deviation must be a real number (not infinity or NaN)
    if "duration_deviation" in df.columns:
        mask = ~np.isfinite(df["duration_deviation"])
        count = mask.sum()
        if count > 0:
            median_val = df.loc[
                np.isfinite(df["duration_deviation"]), "duration_deviation"
            ].median()
            df.loc[mask, "duration_deviation"] = median_val
            fixes += count
            fix_details["duration_deviation non-finite"] = int(count)

    # Can't have more than 7 failures in 7 runs
    if "failures_last_7_runs" in df.columns:
        mask = (df["failures_last_7_runs"] < 0) | (df["failures_last_7_runs"] > 7)
        count = mask.sum()
        if count > 0:
            df["failures_last_7_runs"] = df["failures_last_7_runs"].clip(0, 7)
            fixes += count
            fix_details["failures_last_7_runs out of [0,7]"] = int(count)

    if fixes == 0:
        logger.info("All pre-computed features valid. No fixes needed.")
    else:
        logger.warning(f"Feature validation: {fixes} fixes → {fix_details}")

    return df

df = validate_features(df)
print("Feature validation complete")


# CELL 12: Generate Summary & Save
# WHY: Save the cleaned data to a new CSV and create a JSON report showing what changed (rows removed, columns added, etc.)so the team can review what preprocessing did

def generate_summary(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    encoding_maps: Dict[str, dict]
) -> dict:
    """Generate preprocessing summary report."""
    summary = {
        "raw_shape": {"rows": df_raw.shape[0], "cols": df_raw.shape[1]},
        "processed_shape": {"rows": df_processed.shape[0], "cols": df_processed.shape[1]},
        "rows_removed": df_raw.shape[0] - df_processed.shape[0],
        "columns_added": df_processed.shape[1] - df_raw.shape[1],
        "new_columns": sorted([
            c for c in df_processed.columns if c not in df_raw.columns
        ]),
        "null_counts_after": int(df_processed.isnull().sum().sum()),
        "target_distribution": df_processed[TARGET_COLUMN].value_counts().to_dict(),
        "target_imbalance_ratio": round(
            df_processed[TARGET_COLUMN].value_counts().min()
            / df_processed[TARGET_COLUMN].value_counts().max(), 4
        ),
        "encoding_info": encoding_maps,
    }
    return summary

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


summary = generate_summary(df_raw, df, encoding_maps)
save_processed_data(df, PROCESSED_DATA_FILE, summary)

print("\n📊 PREPROCESSING SUMMARY")
print(f"  Raw:       {summary['raw_shape']}")
print(f"  Processed: {summary['processed_shape']}")
print(f"  Rows removed: {summary['rows_removed']}")
print(f"  Columns added: {summary['columns_added']}")
print(f"  New columns: {summary['new_columns']}")
print(f"  Target distribution: {summary['target_distribution']}")
print(f"  Imbalance ratio: {summary['target_imbalance_ratio']}")
print(f"\nFiles saved: {PROCESSED_DATA_FILE}, preprocessing_summary.json")


# CELL 13: Final Verification
# WHY: Reload the saved file and run ALL checks to confirm everything is correct — this is our quality gate before passing data to the next pipeline step


# Reload saved file to verify it was written correctly
df_check = pd.read_csv(PROCESSED_DATA_FILE, dtype={"run_id": str}, low_memory=False)
print(f"\nReloaded shape: {df_check.shape}")
print(f"Nulls: {df_check.isnull().sum().sum()}")
print(f"Target: {df_check['failed'].value_counts().to_dict()}")

# Run all constraint checks — everything should be True
print(f"\n Constraint Checks:")
print(f"  hours_since_last_run >= 0: {(df_check['hours_since_last_run'] >= 0).all()}")
print(f"  workflow_failure_rate in [0,1]: {df_check['workflow_failure_rate'].between(0, 1).all()}")
print(f"  failed_jobs <= total_jobs: {(df_check['failed_jobs'] <= df_check['total_jobs']).all()}")
print(f"  day_of_week in [0,6]: {df_check['day_of_week'].between(0, 6).all()}")
print(f"  hour in [0,23]: {df_check['hour'].between(0, 23).all()}")

# Verify encoded columns were created
encoded_cols = [c for c in df_check.columns if c.endswith("_encoded")]
print(f"\n Encoded columns ({len(encoded_cols)}): {encoded_cols}")

# Make sure encoded columns have no nulls
encoded_nulls = df_check[encoded_cols].isnull().sum().sum()
print(f"  Encoded columns nulls: {encoded_nulls}")

# Show dtype distribution
print(f"\nDtype distribution:")
print(df_check.dtypes.value_counts().to_string())


print(df_check.isnull().sum()[df_check.isnull().sum() > 0])
print("---")
print(df_check['trigger_time'].head())
print(df_check['trigger_time'].isnull().sum())

# Find rows where trigger_time is null BEFORE saving (in current df)
null_mask = pd.to_datetime(df_raw['trigger_time'], errors='coerce', utc=True).isnull()
print(f"Unparseable trigger_time rows: {null_mask.sum()}")
print("\nSample unparseable values:")
print(df_raw.loc[null_mask, 'trigger_time'].head(10))
print("\nSample parseable values:")
print(df_raw.loc[~null_mask, 'trigger_time'].head(5))





