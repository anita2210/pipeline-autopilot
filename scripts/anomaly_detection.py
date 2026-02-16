"""
anomaly_detection.py
====================
Detect data anomalies and trigger alerts.

Author: Member 4 (Data Quality Engineer) - Aishwarya
Date: February 2026
Project: Pipeline Autopilot - CI/CD Failure Prediction System
"""

import pandas as pd
import numpy as np
import json
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Import config
from config import (
    PROCESSED_DATASET_PATH,
    ANOMALY_REPORT_PATH,
    REPORTS_DIR,
    SCHEMA_FILE_PATH,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    BINARY_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    VALIDATION_RULES,
    ANOMALY_SETTINGS,
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

# Alert settings (configure these for your team)
SLACK_WEBHOOK_URL = None  # Set your Slack webhook URL here
ALERT_EMAIL = None  # Set alert email here


# =============================================================================
# MISSING VALUE CHECKS
# =============================================================================

def check_missing_values(df: pd.DataFrame, threshold: float = 5.0) -> dict:
    """
    Check for columns with missing values above threshold.
    
    Args:
        df: Input DataFrame
        threshold: Maximum allowed missing percentage
        
    Returns:
        dict: Missing value anomalies
    """
    logger.info("Checking for missing values...")
    
    result = {
        "check_name": "missing_values",
        "threshold": threshold,
        "passed": True,
        "anomalies": [],
        "summary": {}
    }
    
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round(missing_count / len(df) * 100, 2)
        
        result["summary"][col] = {
            "missing_count": missing_count,
            "missing_percent": missing_pct
        }
        
        if missing_pct > threshold:
            result["passed"] = False
            result["anomalies"].append({
                "column": col,
                "missing_count": missing_count,
                "missing_percent": missing_pct,
                "message": f"Column '{col}' has {missing_pct}% missing values (threshold: {threshold}%)"
            })
    
    logger.info(f"Missing values check: {'PASSED' if result['passed'] else 'FAILED'} - {len(result['anomalies'])} anomalies")
    return result


# =============================================================================
# RANGE VIOLATION CHECKS
# =============================================================================

def check_range_violations(df: pd.DataFrame) -> dict:
    """
    Check for values outside expected ranges.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Range violation anomalies
    """
    logger.info("Checking for range violations...")
    
    result = {
        "check_name": "range_violations",
        "passed": True,
        "anomalies": [],
        "summary": {}
    }
    
    # Get column rules from config
    column_rules = VALIDATION_RULES.get("column_rules", {})
    
    for col, rules in column_rules.items():
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        violations = []
        
        # Check min_value
        if "min_value" in rules:
            min_val = rules["min_value"]
            below_min = (col_data < min_val).sum()
            if below_min > 0:
                violations.append({
                    "type": "below_minimum",
                    "min_allowed": min_val,
                    "violation_count": int(below_min),
                    "min_found": float(col_data.min())
                })
        
        # Check max_value
        if "max_value" in rules:
            max_val = rules["max_value"]
            above_max = (col_data > max_val).sum()
            if above_max > 0:
                violations.append({
                    "type": "above_maximum",
                    "max_allowed": max_val,
                    "violation_count": int(above_max),
                    "max_found": float(col_data.max())
                })
        
        # Check allowed_values
        if "allowed_values" in rules:
            allowed = set(rules["allowed_values"])
            invalid_mask = ~col_data.isin(allowed)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                invalid_values = col_data[invalid_mask].unique().tolist()[:10]
                violations.append({
                    "type": "invalid_values",
                    "allowed_values": list(allowed),
                    "violation_count": int(invalid_count),
                    "invalid_values_sample": invalid_values
                })
        
        result["summary"][col] = {
            "violations_found": len(violations) > 0,
            "violation_count": sum(v.get("violation_count", 0) for v in violations)
        }
        
        if violations:
            result["passed"] = False
            result["anomalies"].append({
                "column": col,
                "violations": violations,
                "message": f"Column '{col}' has {len(violations)} type(s) of range violations"
            })
    
    logger.info(f"Range violations check: {'PASSED' if result['passed'] else 'FAILED'} - {len(result['anomalies'])} anomalies")
    return result


# =============================================================================
# CONSTRAINT VIOLATION CHECKS
# =============================================================================

def check_constraint_violations(df: pd.DataFrame) -> dict:
    """
    Check for logical constraint violations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Constraint violation anomalies
    """
    logger.info("Checking for constraint violations...")
    
    result = {
        "check_name": "constraint_violations",
        "passed": True,
        "anomalies": [],
        "constraints_checked": []
    }
    
    # Constraint 1: failed_jobs <= total_jobs
    if "failed_jobs" in df.columns and "total_jobs" in df.columns:
        violation_mask = df["failed_jobs"] > df["total_jobs"]
        violation_count = violation_mask.sum()
        
        result["constraints_checked"].append("failed_jobs <= total_jobs")
        
        if violation_count > 0:
            result["passed"] = False
            result["anomalies"].append({
                "constraint": "failed_jobs <= total_jobs",
                "violation_count": int(violation_count),
                "violation_percent": round(violation_count / len(df) * 100, 2),
                "message": f"{violation_count} rows have failed_jobs > total_jobs"
            })
    
    # Constraint 2: is_weekend matches day_of_week
    if "is_weekend" in df.columns and "day_of_week" in df.columns:
        # day_of_week: 0=Monday, 6=Sunday; Weekend = 5 (Sat) or 6 (Sun)
        expected_weekend = df["day_of_week"].isin([5, 6]).astype(int)
        mismatch_mask = df["is_weekend"] != expected_weekend
        mismatch_count = mismatch_mask.sum()
        
        result["constraints_checked"].append("is_weekend matches day_of_week (5,6)")
        
        if mismatch_count > 0:
            result["passed"] = False
            result["anomalies"].append({
                "constraint": "is_weekend matches day_of_week",
                "violation_count": int(mismatch_count),
                "violation_percent": round(mismatch_count / len(df) * 100, 2),
                "message": f"{mismatch_count} rows have is_weekend not matching day_of_week"
            })
    
    # Constraint 3: workflow_failure_rate in [0, 1]
    if "workflow_failure_rate" in df.columns:
        invalid_rate = ((df["workflow_failure_rate"] < 0) | (df["workflow_failure_rate"] > 1)).sum()
        
        result["constraints_checked"].append("workflow_failure_rate in [0, 1]")
        
        if invalid_rate > 0:
            result["passed"] = False
            result["anomalies"].append({
                "constraint": "workflow_failure_rate in [0, 1]",
                "violation_count": int(invalid_rate),
                "violation_percent": round(invalid_rate / len(df) * 100, 2),
                "message": f"{invalid_rate} rows have workflow_failure_rate outside [0, 1]"
            })
    
    # Constraint 4: duration_seconds >= 0
    if "duration_seconds" in df.columns:
        negative_duration = (df["duration_seconds"] < 0).sum()
        
        result["constraints_checked"].append("duration_seconds >= 0")
        
        if negative_duration > 0:
            result["passed"] = False
            result["anomalies"].append({
                "constraint": "duration_seconds >= 0",
                "violation_count": int(negative_duration),
                "violation_percent": round(negative_duration / len(df) * 100, 2),
                "message": f"{negative_duration} rows have negative duration_seconds"
            })
    
    # Constraint 5: failures_last_7_runs <= 7
    if "failures_last_7_runs" in df.columns:
        invalid_failures = (df["failures_last_7_runs"] > 7).sum()
        
        result["constraints_checked"].append("failures_last_7_runs <= 7")
        
        if invalid_failures > 0:
            result["passed"] = False
            result["anomalies"].append({
                "constraint": "failures_last_7_runs <= 7",
                "violation_count": int(invalid_failures),
                "violation_percent": round(invalid_failures / len(df) * 100, 2),
                "message": f"{invalid_failures} rows have failures_last_7_runs > 7"
            })
    
    logger.info(f"Constraint violations check: {'PASSED' if result['passed'] else 'FAILED'} - {len(result['anomalies'])} anomalies")
    return result


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

def check_outliers(df: pd.DataFrame, method: str = "iqr") -> dict:
    """
    Detect outliers in numerical columns using IQR or Z-score method.
    
    Args:
        df: Input DataFrame
        method: 'iqr' or 'zscore'
        
    Returns:
        dict: Outlier anomalies
    """
    logger.info(f"Checking for outliers using {method.upper()} method...")
    
    result = {
        "check_name": "outliers",
        "method": method,
        "passed": True,
        "anomalies": [],
        "summary": {}
    }
    
    # Get columns to check from config
    columns_to_check = ANOMALY_SETTINGS.get("columns_to_check", NUMERICAL_COLUMNS)
    iqr_multiplier = ANOMALY_SETTINGS.get("iqr_multiplier", 1.5)
    zscore_threshold = ANOMALY_SETTINGS.get("zscore_threshold", 3.0)
    
    for col in columns_to_check:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        if method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count = outlier_mask.sum()
            
            result["summary"][col] = {
                "method": "iqr",
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
                "outlier_count": int(outlier_count),
                "outlier_percent": round(outlier_count / len(col_data) * 100, 2)
            }
        
        elif method == "zscore":
            mean = col_data.mean()
            std = col_data.std()
            
            if std == 0:
                outlier_count = 0
            else:
                z_scores = np.abs((col_data - mean) / std)
                outlier_mask = z_scores > zscore_threshold
                outlier_count = outlier_mask.sum()
            
            result["summary"][col] = {
                "method": "zscore",
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
                "threshold": zscore_threshold,
                "outlier_count": int(outlier_count),
                "outlier_percent": round(outlier_count / len(col_data) * 100, 2)
            }
        
        # Flag if outliers exceed threshold
        anomaly_threshold = ANOMALY_SETTINGS.get("anomaly_flag_threshold", 5.0)
        outlier_pct = result["summary"][col]["outlier_percent"]
        
        if outlier_pct > anomaly_threshold:
            result["passed"] = False
            result["anomalies"].append({
                "column": col,
                "outlier_count": result["summary"][col]["outlier_count"],
                "outlier_percent": outlier_pct,
                "message": f"Column '{col}' has {outlier_pct}% outliers (threshold: {anomaly_threshold}%)"
            })
    
    logger.info(f"Outliers check: {'PASSED' if result['passed'] else 'FAILED'} - {len(result['anomalies'])} anomalies")
    return result


# =============================================================================
# SCHEMA VIOLATION CHECKS
# =============================================================================

def check_schema_violations(df: pd.DataFrame, schema: dict = None) -> dict:
    """
    Check for schema violations (unexpected dtypes, new categorical values).
    
    Args:
        df: Input DataFrame
        schema: Schema to validate against (loads from file if None)
        
    Returns:
        dict: Schema violation anomalies
    """
    logger.info("Checking for schema violations...")
    
    result = {
        "check_name": "schema_violations",
        "passed": True,
        "anomalies": [],
        "summary": {}
    }
    
    # Load schema if not provided
    if schema is None:
        if SCHEMA_FILE_PATH.exists():
            with open(SCHEMA_FILE_PATH, 'r') as f:
                schema = json.load(f)
        else:
            logger.warning("No schema file found. Skipping schema violation check.")
            result["message"] = "No schema file found"
            return result
    
    schema_columns = schema.get("columns", {})
    
    for col_name, col_schema in schema_columns.items():
        if col_name not in df.columns:
            result["anomalies"].append({
                "column": col_name,
                "type": "missing_column",
                "message": f"Expected column '{col_name}' not found in data"
            })
            result["passed"] = False
            continue
        
        col_violations = []
        
        # Check dtype
        expected_dtype = col_schema.get("dtype")
        actual_dtype = str(df[col_name].dtype)
        
        if expected_dtype and actual_dtype != expected_dtype:
            col_violations.append({
                "type": "dtype_mismatch",
                "expected": expected_dtype,
                "actual": actual_dtype
            })
        
        # Check for new categorical values
        if col_schema.get("type") == "categorical" and "allowed_values" in col_schema:
            allowed_values = set(col_schema["allowed_values"])
            actual_values = set(df[col_name].dropna().unique())
            new_values = actual_values - allowed_values
            
            if new_values:
                col_violations.append({
                    "type": "new_categorical_values",
                    "new_values_count": len(new_values),
                    "new_values_sample": list(new_values)[:10]
                })
        
        result["summary"][col_name] = {
            "violations_found": len(col_violations) > 0
        }
        
        if col_violations:
            result["passed"] = False
            result["anomalies"].append({
                "column": col_name,
                "violations": col_violations,
                "message": f"Column '{col_name}' has {len(col_violations)} schema violation(s)"
            })
    
    logger.info(f"Schema violations check: {'PASSED' if result['passed'] else 'FAILED'} - {len(result['anomalies'])} anomalies")
    return result


# =============================================================================
# ANOMALY REPORT GENERATION
# =============================================================================

def generate_anomaly_report(check_results: List[dict]) -> dict:
    """
    Generate comprehensive anomaly report from all checks.
    
    Args:
        check_results: List of check result dictionaries
        
    Returns:
        dict: Complete anomaly report
    """
    logger.info("Generating anomaly report...")
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "overall_status": "PASSED",
        "total_checks": len(check_results),
        "checks_passed": 0,
        "checks_failed": 0,
        "total_anomalies": 0,
        "checks": {},
        "all_anomalies": []
    }
    
    for check in check_results:
        check_name = check.get("check_name", "unknown")
        passed = check.get("passed", True)
        anomalies = check.get("anomalies", [])
        
        report["checks"][check_name] = {
            "passed": passed,
            "anomaly_count": len(anomalies)
        }
        
        if passed:
            report["checks_passed"] += 1
        else:
            report["checks_failed"] += 1
            report["overall_status"] = "FAILED"
        
        report["total_anomalies"] += len(anomalies)
        
        for anomaly in anomalies:
            anomaly["check_name"] = check_name
            report["all_anomalies"].append(anomaly)
    
    logger.info(f"Anomaly report generated: {report['overall_status']}")
    return report


def save_anomaly_report(report: dict, filepath: Path = None) -> None:
    """
    Save anomaly report to JSON file.
    
    Args:
        report: Anomaly report dictionary
        filepath: Path to save (default: ANOMALY_REPORT_PATH)
    """
    if filepath is None:
        filepath = ANOMALY_REPORT_PATH
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Anomaly report saved to {filepath}")


# =============================================================================
# ALERT SYSTEM
# =============================================================================

def send_slack_alert(message: str, webhook_url: str = None) -> bool:
    """
    Send alert to Slack channel.
    
    Args:
        message: Alert message
        webhook_url: Slack webhook URL
        
    Returns:
        bool: True if sent successfully
    """
    if webhook_url is None:
        webhook_url = SLACK_WEBHOOK_URL
    
    if webhook_url is None:
        logger.warning("Slack webhook URL not configured. Skipping Slack alert.")
        return False
    
    try:
        payload = {
            "text": f"🚨 *Pipeline Autopilot Alert*\n{message}",
            "username": "Pipeline Autopilot",
            "icon_emoji": ":warning:"
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("Slack alert sent successfully")
            return True
        else:
            logger.error(f"Failed to send Slack alert: {response.status_code}")
            return False
    
    except Exception as e:
        logger.error(f"Error sending Slack alert: {e}")
        return False


def send_email_alert(message: str, email: str = None) -> bool:
    """
    Send alert via email (placeholder - implement with your email service).
    
    Args:
        message: Alert message
        email: Recipient email
        
    Returns:
        bool: True if sent successfully
    """
    if email is None:
        email = ALERT_EMAIL
    
    if email is None:
        logger.warning("Alert email not configured. Skipping email alert.")
        return False
    
    # Placeholder for email implementation
    # You can implement this with smtplib, SendGrid, AWS SES, etc.
    logger.info(f"Email alert would be sent to {email}: {message}")
    return True


def send_alert(report: dict, channel: str = "slack") -> None:
    """
    Send alert based on anomaly report.
    
    Args:
        report: Anomaly report
        channel: Alert channel ('slack', 'email', 'both')
    """
    if report["overall_status"] == "PASSED":
        logger.info("No anomalies detected. Skipping alert.")
        return
    
    # Build alert message
    message = f"""
*Anomaly Detection Report*
Status: {report['overall_status']}
Total Checks: {report['total_checks']}
Checks Failed: {report['checks_failed']}
Total Anomalies: {report['total_anomalies']}

*Failed Checks:*
"""
    
    for check_name, check_info in report["checks"].items():
        if not check_info["passed"]:
            message += f"• {check_name}: {check_info['anomaly_count']} anomalies\n"
    
    message += f"\nTimestamp: {report['generated_at']}"
    
    # Send alerts
    if channel in ["slack", "both"]:
        send_slack_alert(message)
    
    if channel in ["email", "both"]:
        send_email_alert(message)


# =============================================================================
# MASTER FUNCTION
# =============================================================================

def run_anomaly_detection(data_path: Path = None, alert_channel: str = "slack") -> dict:
    """
    Run complete anomaly detection pipeline.
    
    Args:
        data_path: Path to data file (default: PROCESSED_DATASET_PATH)
        alert_channel: Alert channel ('slack', 'email', 'both', None)
        
    Returns:
        dict: Complete anomaly report
    """
    logger.info("=" * 60)
    logger.info("STARTING ANOMALY DETECTION PIPELINE")
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
    
    # Run all checks
    check_results = []
    
    # 1. Missing values check
    missing_result = check_missing_values(df, threshold=VALIDATION_RULES.get("max_missing_percent", 5.0))
    check_results.append(missing_result)
    
    # 2. Range violations check
    range_result = check_range_violations(df)
    check_results.append(range_result)
    
    # 3. Constraint violations check
    constraint_result = check_constraint_violations(df)
    check_results.append(constraint_result)
    
    # 4. Outlier detection (IQR)
    outlier_result = check_outliers(df, method="iqr")
    check_results.append(outlier_result)
    
    # 5. Schema violations check
    schema_result = check_schema_violations(df)
    check_results.append(schema_result)
    
    # Generate report
    report = generate_anomaly_report(check_results)
    
    # Save report
    save_anomaly_report(report)
    
    # Send alerts if needed
    if alert_channel:
        send_alert(report, channel=alert_channel)
    
    # Summary
    logger.info("=" * 60)
    logger.info("ANOMALY DETECTION COMPLETE")
    logger.info(f"  Status: {report['overall_status']}")
    logger.info(f"  Checks Passed: {report['checks_passed']}/{report['total_checks']}")
    logger.info(f"  Total Anomalies: {report['total_anomalies']}")
    logger.info("=" * 60)
    
    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run the full pipeline (without alerts for testing)
    report = run_anomaly_detection(alert_channel=None)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    print(f"Status: {report['overall_status']}")
    print(f"Checks Passed: {report['checks_passed']}/{report['total_checks']}")
    print(f"Total Anomalies: {report['total_anomalies']}")
    
    print("\nCheck Results:")
    for check_name, check_info in report["checks"].items():
        status = "✅ PASSED" if check_info["passed"] else "❌ FAILED"
        print(f"  {check_name}: {status} ({check_info['anomaly_count']} anomalies)")
    
    if report["all_anomalies"]:
        print("\nAnomalies Found:")
        for anomaly in report["all_anomalies"][:10]:  # Show first 10
            print(f"  - [{anomaly['check_name']}] {anomaly.get('message', 'No message')}")
        
        if len(report["all_anomalies"]) > 10:
            print(f"  ... and {len(report['all_anomalies']) - 10} more")
    
    print("=" * 60)