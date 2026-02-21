"""
Unit tests for logging_config.py
Tests logging setup and utility functions.
"""

import pytest
import logging
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from logging_config import (
    get_logger,
    log_section,
    log_step,
    log_metrics,
    setup_logging
)


class TestLoggingSetup:
    """Test logging configuration setup."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_logger_has_handlers(self):
        """Test that loggers have handlers configured."""
        logger = get_logger("test_handlers")
        root_logger = logging.getLogger()
        
        # Root logger should have handlers (console + file)
        assert len(root_logger.handlers) >= 1


class TestLoggingUtilities:
    """Test logging utility functions."""
    
    def test_log_section_runs(self):
        """Test that log_section executes without error."""
        logger = get_logger("test_section")
        
        # Should not raise any exceptions
        log_section(logger, "TEST SECTION")
    
    def test_log_step_runs(self):
        """Test that log_step executes without error."""
        logger = get_logger("test_step")
        
        log_step(logger, "Test Step", "START")
        log_step(logger, "Test Step", "COMPLETE")
    
    def test_log_metrics_with_dict(self):
        """Test that log_metrics handles dictionary input."""
        logger = get_logger("test_metrics")
        
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "count": 100
        }
        
        # Should not raise any exceptions
        log_metrics(logger, metrics, "Test Metrics")
    
    def test_log_metrics_with_empty_dict(self):
        """Test that log_metrics handles empty dictionary."""
        logger = get_logger("test_empty")
        
        log_metrics(logger, {}, "Empty Metrics")


class TestLoggerFunctionality:
    """Test actual logging functionality."""
    
    def test_logger_info_level(self):
        """Test that logger logs INFO level messages."""
        logger = get_logger("test_info")
        
        # Should not raise exceptions
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
    
    def test_multiple_loggers(self):
        """Test creating multiple logger instances."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 != logger2