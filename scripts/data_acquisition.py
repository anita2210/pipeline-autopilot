"""
data_acquisition.py
-------------------
Fetches and validates raw CI/CD pipeline log data from a local CSV or remote URL.
Saves validated data to RAW_DATASET_PATH for downstream DVC-tracked pipeline stages.

Author  : Member 2 (Data Engineer)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : February 2026
"""

import logging
import sys
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path setup — works whether run directly or imported
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    ALL_COLUMNS,
    LOGGING_CONFIG,
    RAW_DATA_DIR,
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    ensure_directories_exist,
)

# ---------------------------------------------------------------------------
# Logging — uses same format as the rest of the project
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["log_level"], logging.INFO),
    format=LOGGING_CONFIG["log_format"],
    datefmt=LOGGING_CONFIG["date_format"],
)
logger = logging.getLogger("data_acquisition")


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_from_csv(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw pipeline data from a local CSV file.

    Parameters
    ----------
    file_path : Path, optional
        Path to CSV. Defaults to RAW_DATASET_PATH from config.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError : If the file does not exist.
    ValueError        : If the file is empty or unparseable.
    """
    path = Path(file_path) if file_path else RAW_DATASET_PATH
    logger.info("Loading data from local CSV: %s", path)

    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"CSV not found at: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        logger.error("CSV file is empty: %s", path)
        raise ValueError(f"CSV file is empty: {path}") from exc
    except Exception as exc:
        logger.error("Failed to parse CSV: %s", exc)
        raise

    logger.info("Loaded %d rows x %d columns from CSV.", len(df), df.shape[1])
    return df


def load_from_url(url: str, timeout: int = 60) -> pd.DataFrame:
    """
    Download raw pipeline data from a remote URL (CSV format).

    Parameters
    ----------
    url     : str  — Public URL pointing to a CSV file.
    timeout : int  — Request timeout in seconds (default 60).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    requests.exceptions.RequestException : On any network error.
    ValueError                           : If response is not valid CSV.
    """
    logger.info("Downloading data from URL: %s", url)

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Request timed out after %ds: %s", timeout, url)
        raise
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while reaching: %s", url)
        raise
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP %s error for URL: %s", exc.response.status_code, url)
        raise

    try:
        df = pd.read_csv(StringIO(response.text))
    except Exception as exc:
        logger.error("Failed to parse response as CSV: %s", exc)
        raise ValueError("Response from URL is not valid CSV.") from exc

    logger.info("Downloaded %d rows x %d columns from URL.", len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded DataFrame against the project schema.

    Checks
    ------
    1. Non-empty (row count > 0)
    2. All expected columns from ALL_COLUMNS are present
    3. Target column contains only binary values (0/1)
    4. Row count within VALIDATION_RULES bounds (100K-200K)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bool — True if all checks pass.

    Raises
    ------
    ValueError : On any failed check.
    """
    logger.info("Running data validation...")

    # 1. Non-empty
    if len(df) == 0:
        raise ValueError("Validation failed: DataFrame has 0 rows.")

    # 2. Expected columns
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Validation failed: Missing columns: {sorted(missing)}"
        )

    # 3. Binary target
    bad_targets = df[TARGET_COLUMN].dropna()
    bad_targets = bad_targets[~bad_targets.isin([0, 1])]
    if not bad_targets.empty:
        raise ValueError(
            f"Validation failed: '{TARGET_COLUMN}' has non-binary values: "
            f"{bad_targets.unique()}"
        )

    # 4. Row count sanity (warn, don't fail)
    if len(df) < 100_000:
        logger.warning(
            "Row count %d is below expected minimum of 100,000.", len(df)
        )

    failure_rate = df[TARGET_COLUMN].mean() * 100
    logger.info(
        "Validation passed | Rows: %d | Columns: %d | Failure rate: %.2f%%",
        len(df), df.shape[1], failure_rate,
    )
    return True


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_raw_data(df: pd.DataFrame, file_path: Optional[Path] = None) -> Path:
    """
    Save raw DataFrame to disk as CSV. Creates parent directories if needed.

    Parameters
    ----------
    df        : pd.DataFrame — Data to save.
    file_path : Path, optional — Defaults to RAW_DATASET_PATH from config.

    Returns
    -------
    Path — Where the file was saved.
    """
    path = Path(file_path) if file_path else RAW_DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Raw data saved to %s (%d rows)", path, len(df))
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def acquire_data(
    source: str = "csv",
    url: Optional[str] = None,
    file_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main entry point for data acquisition. Called by Airflow DAG and dvc.yaml.

    Parameters
    ----------
    source    : 'csv' (default) or 'url'
    url       : Required when source='url'
    file_path : Override for the local CSV path

    Returns
    -------
    pd.DataFrame — Validated raw dataset.
    """
    logger.info("=" * 60)
    logger.info("DATA ACQUISITION START  (source=%s)", source)
    logger.info("=" * 60)

    ensure_directories_exist()

    if source == "csv":
        df = load_from_csv(file_path)
    elif source == "url":
        if not url:
            raise ValueError("'url' argument required when source='url'.")
        df = load_from_url(url)
        save_raw_data(df, file_path)
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'csv' or 'url'.")

    validate_data(df)

    logger.info("DATA ACQUISITION COMPLETE")
    logger.info("=" * 60)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Autopilot - Data Acquisition"
    )
    parser.add_argument(
        "--source", choices=["csv", "url"], default="csv",
        help="Data source type (default: csv)"
    )
    parser.add_argument("--url",  type=str, default=None, help="Remote CSV URL")
    parser.add_argument("--file", type=str, default=None, help="Override local CSV path")
    args = parser.parse_args()

    df = acquire_data(
        source=args.source,
        url=args.url,
        file_path=Path(args.file) if args.file else None,
    )
    print(f"\nShape: {df.shape}")
    print(df.head(3))