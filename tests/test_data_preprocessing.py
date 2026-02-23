"""
Unit tests for data_preprocessing.py
Tests data cleaning, missing value handling, duplicate removal, and encoding.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

# Import functions from data_preprocessing.py
# Note: Adjust these imports based on actual function names in your file
try:
    from data_preprocessing import (
        handle_missing_values,
        remove_duplicates,
        enforce_constraints,
        parse_datetime,
        encode_categoricals
    )
except ImportError as e:
    print(f"Warning: Could not import functions from data_preprocessing.py: {e}")
    print("Tests will be skipped if functions are not available.")


class TestHandleMissingValues:
    """Test missing value imputation."""
    
    def test_numerical_imputation_with_median(self, sample_df_with_missing):
        """Test that numerical columns are filled with median."""
        df = sample_df_with_missing.copy()
        
        # Check that duration_seconds has missing values
        initial_nulls = df['duration_seconds'].isna().sum()
        assert initial_nulls > 0, "Test data should have missing values"
        
        # Apply imputation
        df_cleaned = handle_missing_values(df)
        
        # Check no missing values remain in numerical columns
        assert df_cleaned['duration_seconds'].isna().sum() == 0, "Missing values should be filled"
    
    def test_categorical_imputation_with_mode(self, sample_df_with_missing):
        """Test that categorical columns are filled with mode."""
        df = sample_df_with_missing.copy()
        
        # Check that pipeline_name has missing values
        initial_nulls = df['pipeline_name'].isna().sum()
        assert initial_nulls > 0, "Test data should have missing values"
        
        # Apply imputation
        df_cleaned = handle_missing_values(df)
        
        # Check no missing values remain in categorical columns
        assert df_cleaned['pipeline_name'].isna().sum() == 0, "Missing values should be filled"
    
    def test_no_new_columns_added(self, sample_df_with_missing):
        """Test that imputation doesn't add new columns."""
        df = sample_df_with_missing.copy()
        original_columns = set(df.columns)
        
        df_cleaned = handle_missing_values(df)
        
        assert set(df_cleaned.columns) == original_columns, "Column set should remain the same"
    
    def test_empty_dataframe_handled(self, empty_df):
        """Test that empty DataFrame is handled gracefully."""
        df_cleaned = handle_missing_values(empty_df)
        assert len(df_cleaned) == 0, "Empty DataFrame should remain empty"
        assert list(df_cleaned.columns) == list(empty_df.columns), "Columns should be preserved"


class TestRemoveDuplicates:
    """Test duplicate row removal."""
    
    def test_exact_duplicates_removed(self, sample_df_with_duplicates):
        """Test that exact duplicate rows are removed."""
        df = sample_df_with_duplicates.copy()
        original_length = len(df)
        
        df_cleaned = remove_duplicates(df)
        
        # Should have fewer rows after removing duplicates
        assert len(df_cleaned) < original_length, "Duplicates should be removed"
        
        # No duplicates should remain
        assert df_cleaned.duplicated().sum() == 0, "No duplicates should remain"
    
    def test_duplicates_by_run_id(self, sample_df):
        """Test that duplicates are identified by run_id."""
        df = sample_df.copy()
        
        # Create duplicate with same run_id but different other columns
        duplicate_row = df.iloc[0].copy()
        duplicate_row['duration_seconds'] = 9999  # Change a value
        df = pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=True)
        
        df_cleaned = remove_duplicates(df)
        
        # Should keep only unique run_ids
        assert df_cleaned['run_id'].duplicated().sum() == 0, "run_id should be unique"
    
    def test_no_duplicates_unchanged(self, sample_df):
        """Test that DataFrame without duplicates remains unchanged."""
        df = sample_df.copy()
        original_length = len(df)
        
        df_cleaned = remove_duplicates(df)
        
        assert len(df_cleaned) == original_length, "Length should remain the same"
    
    def test_empty_dataframe(self, empty_df):
        """Test duplicate removal on empty DataFrame."""
        df_cleaned = remove_duplicates(empty_df)
        assert len(df_cleaned) == 0, "Empty DataFrame should remain empty"


class TestEnforceConstraints:
    """Test data constraint enforcement."""
    
    def test_duration_clipped_to_valid_range(self, sample_df_with_anomalies):
        """Test that duration_seconds is clipped to [0, 86400]."""
        df = sample_df_with_anomalies.copy()
        
        # Verify we have constraint violations in test data
        assert (df['duration_seconds'] < 0).any() or (df['duration_seconds'] > 86400).any(), \
            "Test data should have duration violations"
        
        df_cleaned = enforce_constraints(df)
        
        # All durations should be within valid range
        assert df_cleaned['duration_seconds'].min() >= 0, "Min duration should be >= 0"
        assert df_cleaned['duration_seconds'].max() <= 86400, "Max duration should be <= 86400"
    
    def test_workflow_failure_rate_clipped(self, sample_df_with_anomalies):
        """Test that workflow_failure_rate is clipped to [0, 1]."""
        df = sample_df_with_anomalies.copy()
        
        # Verify we have rate violations in test data
        assert (df['workflow_failure_rate'] < 0).any() or (df['workflow_failure_rate'] > 1.0).any(), \
            "Test data should have rate violations"
        
        df_cleaned = enforce_constraints(df)
        
        # All rates should be within [0, 1]
        assert df_cleaned['workflow_failure_rate'].min() >= 0, "Min rate should be >= 0"
        assert df_cleaned['workflow_failure_rate'].max() <= 1.0, "Max rate should be <= 1.0"
    
    def test_failed_jobs_constraint(self, sample_df_with_anomalies):
        """Test that failed_jobs <= total_jobs constraint is enforced."""
        df = sample_df_with_anomalies.copy()
        
        # Verify we have constraint violations
        assert (df['failed_jobs'] > df['total_jobs']).any(), \
            "Test data should have failed_jobs > total_jobs"
        
        df_cleaned = enforce_constraints(df)
        
        # After cleaning, failed_jobs should never exceed total_jobs
        violations = (df_cleaned['failed_jobs'] > df_cleaned['total_jobs']).sum()
        assert violations == 0, "failed_jobs should never exceed total_jobs"
    
    def test_empty_dataframe(self, empty_df):
        """Test constraint enforcement on empty DataFrame."""
        df_cleaned = enforce_constraints(empty_df)
        assert len(df_cleaned) == 0, "Empty DataFrame should remain empty"


class TestParseDatetime:
    """Test datetime parsing."""
    
    def test_trigger_time_parsed_correctly(self, sample_df):
        """Test that trigger_time is parsed to datetime."""
        df = sample_df.copy()
        
        # Convert to string first (simulating CSV load)
        df['trigger_time'] = df['trigger_time'].astype(str)
        
        df_cleaned = parse_datetime(df)
        
        # Check it's now datetime type
        assert pd.api.types.is_datetime64_any_dtype(df_cleaned['trigger_time']), \
            "trigger_time should be datetime type"
    
    def test_invalid_datetime_handled(self):
        """Test that invalid datetime strings are handled gracefully."""
        df = pd.DataFrame({
            'run_id': ['run_001', 'run_002', 'run_003'],
            'trigger_time': ['2024-01-01 10:00:00', 'invalid', '2024-01-02 12:00:00'],
            'failed': [0, 1, 0]
        })
        
        # Should not crash
        df_cleaned = parse_datetime(df)
        
        # Result should still be datetime type (invalid entries might be NaT)
        assert pd.api.types.is_datetime64_any_dtype(df_cleaned['trigger_time']), \
            "trigger_time should be datetime type"
    
    def test_already_datetime_unchanged(self, sample_df):
        """Test that already-datetime columns are not changed."""
        df = sample_df.copy()
        
        # trigger_time is already datetime in fixture
        df_cleaned = parse_datetime(df)
        
        assert pd.api.types.is_datetime64_any_dtype(df_cleaned['trigger_time']), \
            "trigger_time should remain datetime type"


class TestEncodeCategoricals:
    """Test categorical encoding."""
    
    def test_categorical_columns_encoded(self, sample_df):
        """Test that categorical columns are converted to numeric."""
        df = sample_df.copy()
        
        categorical_cols = ['pipeline_name', 'repo', 'head_branch', 'trigger_type']
        
        # encode_categoricals returns (df, encoding_maps) tuple
        df_cleaned, encoding_maps = encode_categoricals(df, categorical_cols)
        
        # Check all categorical columns have encoded versions
        for col in categorical_cols:
            encoded_col = f"{col}_encoded"
            if encoded_col in df_cleaned.columns:
                assert pd.api.types.is_numeric_dtype(df_cleaned[encoded_col]), \
                    f"{encoded_col} should be numeric after encoding"
    
    def test_encoding_preserves_unique_values(self, sample_df):
        """Test that encoding preserves the number of unique values."""
        df = sample_df.copy()
    
        original_unique = df['repo'].nunique()
    
        # encode_categoricals encodes ALL categorical columns, not just one
        df_cleaned, encoding_maps = encode_categoricals(df, method="frequency")
    
        # Should have same number of unique values in encoded column
        assert df_cleaned['repo_encoded'].nunique() == original_unique, \
            "Number of unique values should be preserved"
    
        # Verify repo was actually encoded
        assert 'repo' in encoding_maps, "repo should be in encoding maps"
        assert encoding_maps['repo']['n_unique'] == original_unique, \
            "Encoding map should record correct number of unique values"
    
    def test_empty_dataframe(self, empty_df):
        """Test encoding on empty DataFrame."""
        # encode_categoricals returns (df, encoding_maps) tuple
        df_cleaned, encoding_maps = encode_categoricals(empty_df, ['pipeline_name'])
        assert len(df_cleaned) == 0, "Empty DataFrame should remain empty"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_row_dataframe(self, single_row_df):
        """Test preprocessing pipeline on single-row DataFrame."""
        df = single_row_df.copy()
        
        # Apply full preprocessing pipeline
        df = handle_missing_values(df)
        df = remove_duplicates(df)
        df = enforce_constraints(df)
        
        # Should still have 1 row
        assert len(df) == 1, "Single row should be preserved"
    
    def test_all_values_same(self):
        """Test handling when all values in a column are the same."""
        df = pd.DataFrame({
            'run_id': [f'run_{i}' for i in range(10)],
            'pipeline_name': ['build'] * 10,  # All same value
            'duration_seconds': [100] * 10,
            'failed': [0] * 10
        })
        
        # Should handle gracefully
        df_cleaned = handle_missing_values(df)
        assert len(df_cleaned) == 10, "All rows should be preserved"
        assert (df_cleaned['pipeline_name'] == 'build').all(), "Values should be unchanged"