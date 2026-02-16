"""
schema_validation.py
====================
Auto-generate schema & statistics, validate data against baseline.

Author: Member 4 (Data Quality Engineer) - Aishwarya
Date: February 2026
Project: Pipeline Autopilot - CI/CD Failure Prediction System
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Import config
from config import (
    PROCESSED_DATASET_PATH,
    SCHEMA_DIR,
    SCHEMA_FILE_PATH,
    VALIDATION_REPORT_PATH,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    BINARY_COLUMNS,
    ALL_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    DATETIME_COLUMN,
    VALIDATION_RULES,
    LOGGING_CONFIG,
    ensure_directories_exist,
)

# Set up logging
logging.basicConfig(
    level=LOGGING_CONFIG["log_level"],
    format=LOGGING_CONFIG["log_format"],
    datefmt=LOGGING_CONFIG["date_format"],
)
logger = logging.getLogger(__name__)


# =============================================================================
# STATISTICS GENERATION
# =============================================================================

def generate_statistics(df: pd.DataFrame) -> dict:
    """
    Generate statistics for all columns in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Statistics for each column
    """
    logger.info("Generating dataset statistics...")
    
    stats = {
        "generated_at": datetime.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": {}
    }
    
    # Numerical column stats
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            stats["columns"][col] = {
                "type": "numerical",
                "dtype": str(df[col].dtype),
                "count": int(df[col].count()),
                "missing": int(df[col].isna().sum()),
                "missing_percent": round(df[col].isna().sum() / len(df) * 100, 2),
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "median": round(float(df[col].median()), 4),
                "q1": round(float(df[col].quantile(0.25)), 4),
                "q3": round(float(df[col].quantile(0.75)), 4),
            }
    
    # Categorical column stats
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            value_counts = df[col].value_counts()
            stats["columns"][col] = {
                "type": "categorical",
                "dtype": str(df[col].dtype),
                "count": int(df[col].count()),
                "missing": int(df[col].isna().sum()),
                "missing_percent": round(df[col].isna().sum() / len(df) * 100, 2),
                "unique_count": int(df[col].nunique()),
                #"unique_values": df[col].dropna().unique().tolist()[:50],  # Top 50 unique values
                "unique_values": df[col].dropna().unique().tolist(),  # All unique values
                "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            }
    
    # ID column stats
    if ID_COLUMN in df.columns:
        stats["columns"][ID_COLUMN] = {
            "type": "id",
            "dtype": str(df[ID_COLUMN].dtype),
            "count": int(df[ID_COLUMN].count()),
            "unique_count": int(df[ID_COLUMN].nunique()),
            "is_unique": df[ID_COLUMN].nunique() == len(df),
        }
    
    # Target column stats
    if TARGET_COLUMN in df.columns:
        value_counts = df[TARGET_COLUMN].value_counts()
        stats["columns"][TARGET_COLUMN] = {
            "type": "target",
            "dtype": str(df[TARGET_COLUMN].dtype),
            "count": int(df[TARGET_COLUMN].count()),
            "unique_values": df[TARGET_COLUMN].unique().tolist(),
            "distribution": {str(k): int(v) for k, v in value_counts.items()},
            "positive_rate": round(float(df[TARGET_COLUMN].mean()), 4),
        }
    
    # Datetime column stats
    if DATETIME_COLUMN in df.columns:
        dt_col = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce')
        stats["columns"][DATETIME_COLUMN] = {
            "type": "datetime",
            "dtype": str(df[DATETIME_COLUMN].dtype),
            "count": int(df[DATETIME_COLUMN].count()),
            "missing": int(df[DATETIME_COLUMN].isna().sum()),
            "min_date": str(dt_col.min()),
            "max_date": str(dt_col.max()),
        }
    
    logger.info(f"Statistics generated for {len(stats['columns'])} columns")
    return stats


def save_statistics(stats: dict, filepath: Path = None) -> None:
    """
    Save statistics to a JSON file.
    
    Args:
        stats: Statistics dictionary
        filepath: Path to save (default: SCHEMA_DIR/stats.json)
    """
    if filepath is None:
        filepath = SCHEMA_DIR / "stats.json"
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"Statistics saved to {filepath}")


# =============================================================================
# SCHEMA GENERATION
# =============================================================================

def generate_schema(df: pd.DataFrame, stats: dict = None) -> dict:
    """
    Generate schema from DataFrame and statistics.
    
    Args:
        df: Input DataFrame
        stats: Pre-computed statistics (optional)
        
    Returns:
        dict: Schema definition
    """
    logger.info("Generating data schema...")
    
    if stats is None:
        stats = generate_statistics(df)
    
    schema = {
        "generated_at": datetime.now().isoformat(),
        "version": "1.0",
        "expected_columns": list(df.columns),
        "column_count": len(df.columns),
        "columns": {}
    }
    
    # Generate schema for each column
    for col in df.columns:
        col_schema = {
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isna().any()),
        }
        
        # Add type-specific schema
        if col in NUMERICAL_COLUMNS:
            col_stats = stats["columns"].get(col, {})
            col_schema.update({
                "type": "numerical",
                "min": col_stats.get("min"),
                "max": col_stats.get("max"),
                "mean": col_stats.get("mean"),
                "std": col_stats.get("std"),
            })
            
            # Check if binary
            if col in BINARY_COLUMNS:
                col_schema["type"] = "binary"
                col_schema["allowed_values"] = [0, 1]
        
        elif col in CATEGORICAL_COLUMNS:
            col_stats = stats["columns"].get(col, {})
            col_schema.update({
                "type": "categorical",
                "unique_count": col_stats.get("unique_count"),
                "allowed_values": col_stats.get("unique_values", []),
            })
        
        elif col == ID_COLUMN:
            col_schema["type"] = "id"
            col_schema["unique"] = True
        
        elif col == TARGET_COLUMN:
            col_schema["type"] = "target"
            col_schema["allowed_values"] = [0, 1]
        
        elif col == DATETIME_COLUMN:
            col_schema["type"] = "datetime"
        
        # Add validation rules from config
        if col in VALIDATION_RULES.get("column_rules", {}):
            col_schema["validation_rules"] = VALIDATION_RULES["column_rules"][col]
        
        schema["columns"][col] = col_schema
    
    logger.info(f"Schema generated with {len(schema['columns'])} columns")
    return schema


def save_schema(schema: dict, filepath: Path = None) -> None:
    """
    Save schema to a JSON file.
    
    Args:
        schema: Schema dictionary
        filepath: Path to save (default: SCHEMA_FILE_PATH)
    """
    if filepath is None:
        filepath = SCHEMA_FILE_PATH
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(schema, f, indent=2, default=str)
    
    logger.info(f"Schema saved to {filepath}")


def load_schema(filepath: Path = None) -> Optional[dict]:
    """
    Load schema from a JSON file.
    
    Args:
        filepath: Path to schema file (default: SCHEMA_FILE_PATH)
        
    Returns:
        dict: Schema dictionary or None if file not found
    """
    if filepath is None:
        filepath = SCHEMA_FILE_PATH
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Schema file not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        schema = json.load(f)
    
    logger.info(f"Schema loaded from {filepath}")
    return schema


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def validate_against_schema(df: pd.DataFrame, schema: dict) -> dict:
    """
    Validate DataFrame against a baseline schema.
    
    Args:
        df: Input DataFrame to validate
        schema: Baseline schema to validate against
        
    Returns:
        dict: Validation report with anomalies
    """
    logger.info("Validating data against schema...")
    
    report = {
        "validated_at": datetime.now().isoformat(),
        "is_valid": True,
        "total_errors": 0,
        "total_warnings": 0,
        "errors": [],
        "warnings": [],
        "column_reports": {}
    }
    
    # Check for missing columns
    expected_cols = set(schema.get("expected_columns", []))
    actual_cols = set(df.columns)
    
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols
    
    if missing_cols:
        report["errors"].append({
            "type": "missing_columns",
            "message": f"Missing columns: {list(missing_cols)}",
            "columns": list(missing_cols)
        })
        report["is_valid"] = False
        report["total_errors"] += 1
    
    if extra_cols:
        report["warnings"].append({
            "type": "extra_columns",
            "message": f"Unexpected columns: {list(extra_cols)}",
            "columns": list(extra_cols)
        })
        report["total_warnings"] += 1
    
    # Validate each column
    for col_name, col_schema in schema.get("columns", {}).items():
        if col_name not in df.columns:
            continue
        
        col_report = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        col_data = df[col_name]
        
        # Check dtype
        expected_dtype = col_schema.get("dtype")
        if expected_dtype and str(col_data.dtype) != expected_dtype:
            col_report["warnings"].append({
                "type": "dtype_mismatch",
                "message": f"Expected {expected_dtype}, got {col_data.dtype}",
                "expected": expected_dtype,
                "actual": str(col_data.dtype)
            })
            col_report["is_valid"] = False
            report["total_warnings"] += 1
        
        # Check nullable
        if not col_schema.get("nullable", True) and col_data.isna().any():
            null_count = int(col_data.isna().sum())
            col_report["errors"].append({
                "type": "unexpected_nulls",
                "message": f"Found {null_count} null values in non-nullable column",
                "null_count": null_count
            })
            col_report["is_valid"] = False
            report["is_valid"] = False
            report["total_errors"] += 1
        
        # Check allowed values (for categorical/binary)
        if "allowed_values" in col_schema:
            allowed = set(col_schema["allowed_values"])
            actual_values = set(col_data.dropna().unique())
            new_values = actual_values - allowed
            
            if new_values:
                # For categorical, new values might be okay (warning)
                if col_schema.get("type") == "categorical":
                    col_report["warnings"].append({
                        "type": "new_categorical_values",
                        "message": f"Found {len(new_values)} new categorical values",
                        "new_values": list(new_values)[:20]  # Limit to 20
                    })
                    report["total_warnings"] += 1
                else:
                    # For binary/target, new values are errors
                    col_report["errors"].append({
                        "type": "invalid_values",
                        "message": f"Found invalid values: {list(new_values)}",
                        "invalid_values": list(new_values)
                    })
                    col_report["is_valid"] = False
                    report["is_valid"] = False
                    report["total_errors"] += 1
        
        # Check min/max for numerical
        if col_schema.get("type") == "numerical":
            if "min" in col_schema and col_schema["min"] is not None:
                schema_min = col_schema["min"]
                actual_min = col_data.min()
                # Allow 10% deviation
                if actual_min < schema_min * 0.9 - 1:  # -1 for columns near 0
                    col_report["warnings"].append({
                        "type": "min_drift",
                        "message": f"Min value drifted: schema={schema_min}, actual={actual_min}",
                        "schema_min": schema_min,
                        "actual_min": float(actual_min)
                    })
                    report["total_warnings"] += 1
            
            if "max" in col_schema and col_schema["max"] is not None:
                schema_max = col_schema["max"]
                actual_max = col_data.max()
                # Allow 10% deviation
                if actual_max > schema_max * 1.1 + 1:  # +1 for columns near 0
                    col_report["warnings"].append({
                        "type": "max_drift",
                        "message": f"Max value drifted: schema={schema_max}, actual={actual_max}",
                        "schema_max": schema_max,
                        "actual_max": float(actual_max)
                    })
                    report["total_warnings"] += 1
        
        # Check validation rules from config
        validation_rules = col_schema.get("validation_rules", {})
        
        if "min_value" in validation_rules:
            min_val = validation_rules["min_value"]
            violations = (col_data < min_val).sum()
            if violations > 0:
                col_report["errors"].append({
                    "type": "min_value_violation",
                    "message": f"{violations} values below minimum {min_val}",
                    "violation_count": int(violations),
                    "min_allowed": min_val
                })
                col_report["is_valid"] = False
                report["is_valid"] = False
                report["total_errors"] += 1
        
        if "max_value" in validation_rules:
            max_val = validation_rules["max_value"]
            violations = (col_data > max_val).sum()
            if violations > 0:
                col_report["errors"].append({
                    "type": "max_value_violation",
                    "message": f"{violations} values above maximum {max_val}",
                    "violation_count": int(violations),
                    "max_allowed": max_val
                })
                col_report["is_valid"] = False
                report["is_valid"] = False
                report["total_errors"] += 1
        
        report["column_reports"][col_name] = col_report
    
    logger.info(f"Validation complete: {report['total_errors']} errors, {report['total_warnings']} warnings")
    return report


def save_validation_report(report: dict, filepath: Path = None) -> None:
    """
    Save validation report to a JSON file.
    
    Args:
        report: Validation report dictionary
        filepath: Path to save (default: VALIDATION_REPORT_PATH)
    """
    if filepath is None:
        filepath = VALIDATION_REPORT_PATH
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Validation report saved to {filepath}")


# =============================================================================
# MASTER FUNCTION
# =============================================================================

def run_schema_validation(data_path: Path = None, baseline_schema_path: Path = None) -> Tuple[dict, dict, dict]:
    """
    Run complete schema validation pipeline.
    
    Args:
        data_path: Path to data file (default: PROCESSED_DATASET_PATH)
        baseline_schema_path: Path to baseline schema (default: SCHEMA_FILE_PATH)
        
    Returns:
        Tuple of (statistics, schema, validation_report)
    """
    logger.info("=" * 60)
    logger.info("STARTING SCHEMA VALIDATION PIPELINE")
    logger.info("=" * 60)
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Set defaults
    if data_path is None:
        data_path = PROCESSED_DATASET_PATH
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Generate statistics
    stats = generate_statistics(df)
    save_statistics(stats)
    
    # Check if baseline schema exists
    baseline_schema = load_schema(baseline_schema_path)
    
    if baseline_schema is None:
        # First run - generate and save schema
        logger.info("No baseline schema found. Generating new schema...")
        schema = generate_schema(df, stats)
        save_schema(schema)
        
        # Create a simple validation report (no comparison)
        validation_report = {
            "validated_at": datetime.now().isoformat(),
            "is_valid": True,
            "message": "Baseline schema created. No comparison performed.",
            "total_errors": 0,
            "total_warnings": 0,
            "errors": [],
            "warnings": [],
        }
    else:
        # Compare against baseline
        logger.info("Validating against baseline schema...")
        schema = baseline_schema  # Use existing schema
        validation_report = validate_against_schema(df, baseline_schema)
    
    # Save validation report
    save_validation_report(validation_report)
    
    # Summary
    logger.info("=" * 60)
    logger.info("SCHEMA VALIDATION COMPLETE")
    logger.info(f"  Valid: {validation_report['is_valid']}")
    logger.info(f"  Errors: {validation_report['total_errors']}")
    logger.info(f"  Warnings: {validation_report['total_warnings']}")
    logger.info("=" * 60)
    
    return stats, schema, validation_report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run the full pipeline
    stats, schema, report = run_schema_validation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCHEMA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Rows: {stats['row_count']}")
    print(f"Columns: {stats['column_count']}")
    print(f"Valid: {report['is_valid']}")
    print(f"Errors: {report['total_errors']}")
    print(f"Warnings: {report['total_warnings']}")
    
    if report['errors']:
        print("\nErrors:")
        for err in report['errors']:
            print(f"  - {err['message']}")
    
    if report['warnings']:
        print("\nWarnings:")
        for warn in report['warnings']:
            print(f"  - {warn['message']}")
    
    print("=" * 60)