"""
Shared Logging Configuration for Pipeline Autopilot
Provides consistent logging across all pipeline scripts.

Usage:
    from logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Your message here")
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotating log settings
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Keep 5 backup files


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_logging(
    log_file: Path = LOG_FILE,
    log_level: int = LOG_LEVEL,
    log_format: str = LOG_FORMAT,
    date_format: str = DATE_FORMAT,
    max_bytes: int = MAX_LOG_SIZE,
    backup_count: int = BACKUP_COUNT
) -> None:
    """
    Set up logging configuration for the entire application.
    
    Creates:
        - Console handler (stdout) for INFO and above
        - Rotating file handler for all logs
        
    Args:
        log_file: Path to log file
        log_level: Logging level (default: INFO)
        log_format: Log message format
        date_format: Date/time format
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Ensure log directory exists
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log the setup
    root_logger.info("=" * 80)
    root_logger.info("Logging initialized")
    root_logger.info(f"Log file: {log_file.absolute()}")
    root_logger.info(f"Log level: {logging.getLevelName(log_level)}")
    root_logger.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    
    Args:
        name: Logger name (usually __name__ of the module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str, width: int = 80) -> None:
    """
    Log a section header for better readability.
    
    Args:
        logger: Logger instance
        title: Section title
        width: Width of the header line
        
    Example:
        log_section(logger, "DATA LOADING")
        # Output:
        # ================================================================================
        # DATA LOADING
        # ================================================================================
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame") -> None:
    """
    Log information about a pandas DataFrame.
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name/description of the DataFrame
    """
    logger.info(f"{name} info:")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.info(f"  Missing values: {nulls:,}")
    else:
        logger.info("  Missing values: 0")


def log_step(logger: logging.Logger, step_name: str, status: str = "START") -> None:
    """
    Log a pipeline step with timestamp.
    
    Args:
        logger: Logger instance
        step_name: Name of the step
        status: Status (START, COMPLETE, FAILED, etc.)
        
    Example:
        log_step(logger, "Data Preprocessing", "START")
        # ... do work ...
        log_step(logger, "Data Preprocessing", "COMPLETE")
    """
    logger.info(f"[{status}] {step_name}")


def log_metrics(logger: logging.Logger, metrics: dict, title: str = "Metrics") -> None:
    """
    Log a dictionary of metrics in a readable format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric_name: value
        title: Title for the metrics section
        
    Example:
        log_metrics(logger, {"Accuracy": 0.95, "F1": 0.92}, "Model Performance")
    """
    logger.info(f"{title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


# =============================================================================
# AUTO-SETUP (runs when module is imported)
# =============================================================================

# Automatically set up logging when this module is imported
setup_logging()