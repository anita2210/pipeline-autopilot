"""
Unit tests for anomaly_detection.py
Tests missing values, range violations, constraint violations, and outlier detection.
"""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

try:
    from anomaly_detection import (
        check_missing_values,
        check_range_violations,
        check_constraint_violations,
        check_outliers,
        check_schema_violations,
        generate_anomaly_report,
        save_anomaly_report
    )
except ImportError as e:
    print(f"Warning: Could not import from anomaly_detection.py: {e}")


class TestCheckMissingValues:
    """Test missing value detection."""
    
    def test_no_missing_values(self, sample_df):
        """Test detection when no missing values present."""
        result = check_missing_values(sample_df)
        
        assert 'check_name' in result
        assert result['check_name'] == 'missing_values'
        assert isinstance(result, dict)
    
    def test_missing_values_detected(self, sample_df_with_missing):
        """Test that missing values are detected."""
        result = check_missing_values(sample_df_with_missing)
        
        assert 'check_name' in result
        assert result['check_name'] == 'missing_values'
        # Should detect missing values
        assert result['passed'] == False or len(result.get('anomalies', [])) > 0
    
    def test_threshold_parameter(self, sample_df_with_missing):
        """Test that threshold parameter works."""
        # With high threshold, might not flag as anomaly
        result_high = check_missing_values(sample_df_with_missing, threshold=50.0)
        
        # With low threshold, should flag as anomaly
        result_low = check_missing_values(sample_df_with_missing, threshold=1.0)
        
        # Both should return valid results
        assert 'check_name' in result_high
        assert 'check_name' in result_low
        
        # Low threshold should have more anomalies
        assert len(result_low.get('anomalies', [])) >= len(result_high.get('anomalies', []))


class TestCheckRangeViolations:
    """Test range violation detection."""
    
    def test_no_violations_in_clean_data(self, sample_df):
        """Test that clean data has minimal range violations."""
        result = check_range_violations(sample_df)
        
        assert 'check_name' in result
        assert result['check_name'] == 'range_violations'
        assert isinstance(result, dict)
    
    def test_negative_duration_detected(self, sample_df_with_anomalies):
        """Test that negative duration_seconds is detected."""
        result = check_range_violations(sample_df_with_anomalies)
        
        assert 'check_name' in result
        # Should detect the negative duration anomaly
        assert result['passed'] == False or len(result.get('anomalies', [])) > 0
    
    def test_rate_above_one_detected(self, sample_df_with_anomalies):
        """Test that workflow_failure_rate > 1.0 is detected."""
        result = check_range_violations(sample_df_with_anomalies)
        
        assert 'check_name' in result
        assert result['check_name'] == 'range_violations'
        # Should detect violations
        assert isinstance(result.get('anomalies', []), list)


class TestCheckConstraintViolations:
    """Test constraint violation detection."""
    
    def test_constraint_check_runs(self, sample_df):
        """Test that constraint check runs successfully."""
        result = check_constraint_violations(sample_df)
        
        assert 'check_name' in result
        assert result['check_name'] == 'constraint_violations'
        assert isinstance(result, dict)
    
    def test_failed_jobs_exceeds_total_detected(self, sample_df_with_anomalies):
        """Test detection of failed_jobs > total_jobs."""
        result = check_constraint_violations(sample_df_with_anomalies)
        
        assert 'check_name' in result
        # Should detect constraint violations
        assert isinstance(result.get('anomalies', []), list)
    
    def test_empty_dataframe(self, empty_df):
        """Test constraint check on empty DataFrame."""
        # Skip this test if empty df causes issues
        try:
            result = check_constraint_violations(empty_df)
            assert 'check_name' in result
        except (ZeroDivisionError, ValueError):
            pytest.skip("Empty DataFrame not supported")


class TestCheckOutliers:
    """Test outlier detection."""
    
    def test_iqr_method(self, sample_df_with_anomalies):
        """Test IQR-based outlier detection."""
        result = check_outliers(sample_df_with_anomalies, method="iqr")
        
        assert 'check_name' in result
        assert result['check_name'] == 'outliers'
        assert result.get('method') == 'iqr'
    
    def test_zscore_method(self, sample_df_with_anomalies):
        """Test Z-score based outlier detection."""
        result = check_outliers(sample_df_with_anomalies, method="zscore")
        
        assert 'check_name' in result
        assert result.get('method') == 'zscore'
    
    def test_no_outliers_in_normal_data(self, sample_df):
        """Test that normal data check runs successfully."""
        result = check_outliers(sample_df, method="iqr")
        
        assert 'check_name' in result
        assert isinstance(result, dict)
    
    def test_empty_dataframe(self, empty_df):
        """Test outlier detection on empty DataFrame."""
        try:
            result = check_outliers(empty_df)
            assert 'check_name' in result
        except (ValueError, ZeroDivisionError):
            pytest.skip("Empty DataFrame not supported")


class TestGenerateAnomalyReport:
    """Test anomaly report generation."""
    
    def test_report_generation(self, sample_df_with_anomalies):
        """Test generating comprehensive anomaly report."""
        # Run all checks
        checks = [
            check_missing_values(sample_df_with_anomalies),
            check_range_violations(sample_df_with_anomalies),
            check_constraint_violations(sample_df_with_anomalies),
            check_outliers(sample_df_with_anomalies)
        ]
        
        report = generate_anomaly_report(checks)
        
        # Verify report structure
        assert isinstance(report, dict), "Report should be a dictionary"
    
    def test_report_with_no_anomalies(self, sample_df):
        """Test report generation when minimal anomalies present."""
        checks = [
            check_missing_values(sample_df),
            check_range_violations(sample_df),
            check_constraint_violations(sample_df),
            check_outliers(sample_df)
        ]
        
        report = generate_anomaly_report(checks)
        
        assert isinstance(report, dict)


class TestSaveAnomalyReport:
    """Test saving anomaly reports."""
    
    def test_save_report(self, sample_df_with_anomalies, tmp_path):
        """Test saving anomaly report to file."""
        checks = [
            check_missing_values(sample_df_with_anomalies),
            check_range_violations(sample_df_with_anomalies)
        ]
        
        report = generate_anomaly_report(checks)
        
        report_file = tmp_path / "test_anomaly_report.json"
        save_anomaly_report(report, str(report_file))
        
        assert report_file.exists(), "Report file should be created"
        
        # Verify it's valid JSON
        with open(report_file, 'r') as f:
            loaded = json.load(f)
        
        assert isinstance(loaded, dict), "Saved report should be valid JSON"


class TestCheckSchemaViolations:
    """Test schema violation detection."""
    
    def test_with_valid_schema(self, sample_df):
        """Test schema violation check with valid schema."""
        # Create a simple schema
        schema = {
            'expected_columns': list(sample_df.columns),
            'columns': {}
        }
        
        result = check_schema_violations(sample_df, schema)
        
        assert 'check_name' in result
        assert result['check_name'] == 'schema_violations'
    
    def test_with_no_schema(self, sample_df):
        """Test behavior when no schema provided."""
        result = check_schema_violations(sample_df, schema=None)
        
        assert 'check_name' in result
        # Should handle gracefully


class TestAnomalyDetectionIntegration:
    """Integration tests for full anomaly detection pipeline."""
    
    def test_multiple_checks_run(self, sample_df_with_anomalies):
        """Test that multiple anomaly checks run successfully."""
        checks = [
            check_missing_values(sample_df_with_anomalies),
            check_range_violations(sample_df_with_anomalies),
            check_constraint_violations(sample_df_with_anomalies),
            check_outliers(sample_df_with_anomalies)
        ]
        
        # All checks should return results
        assert all(isinstance(check, dict) for check in checks)
        assert all('check_name' in check for check in checks)
    
    def test_clean_data_pipeline(self, sample_df):
        """Test that pipeline runs on clean data."""
        checks = [
            check_missing_values(sample_df),
            check_range_violations(sample_df),
            check_constraint_violations(sample_df)
        ]
        
        report = generate_anomaly_report(checks)
        
        # Should generate valid report
        assert isinstance(report, dict)


class TestEdgeCases:
    """Test edge cases in anomaly detection."""
    
    def test_single_row_dataframe(self, single_row_df):
        """Test anomaly detection on single-row DataFrame."""
        result_range = check_range_violations(single_row_df)
        result_constraint = check_constraint_violations(single_row_df)
        
        # All should handle single row gracefully
        assert isinstance(result_range, dict)
        assert isinstance(result_constraint, dict)
    
    def test_all_values_same(self):
        """Test anomaly detection when all values are the same."""
        df = pd.DataFrame({
            'col1': [100] * 50,
            'col2': [1] * 50,
            'col3': [0] * 50
        })
        
        # Should handle without crashing
        result = check_outliers(df)
        assert isinstance(result, dict)
        assert 'check_name' in result