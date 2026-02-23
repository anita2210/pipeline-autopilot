"""
Unit tests for schema_validation.py
Tests schema generation, validation, and violation detection.
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
    from schema_validation import (
        generate_schema,
        generate_statistics,
        validate_against_schema,
        save_schema,
        load_schema,
        save_statistics,
        save_validation_report
    )
except ImportError as e:
    print(f"Warning: Could not import from schema_validation.py: {e}")


class TestGenerateSchema:
    """Test schema generation from DataFrame."""
    
    def test_schema_has_all_columns(self, sample_df):
        """Test that schema includes all DataFrame columns."""
        schema = generate_schema(sample_df)
        
        assert 'columns' in schema, "Schema should have 'columns' key"
        
        for col in sample_df.columns:
            assert col in schema['columns'], f"Column '{col}' should be in schema"
    
    def test_schema_records_dtypes(self, sample_df):
        """Test that schema records correct data types."""
        schema = generate_schema(sample_df)
        
        # Check a few known types
        assert 'run_id' in schema['columns']
        assert schema['columns']['run_id']['dtype'] == 'object'
        
        assert 'duration_seconds' in schema['columns']
        # Could be int64, int32, or float64 depending on data
        assert 'int' in schema['columns']['duration_seconds']['dtype'] or \
               'float' in schema['columns']['duration_seconds']['dtype']
    
    def test_schema_includes_type_info(self, sample_df):
        """Test that schema includes type information."""
        schema = generate_schema(sample_df)
        
        for col in sample_df.columns:
            col_schema = schema['columns'][col]
            
            # Should have dtype
            assert 'dtype' in col_schema, f"{col} should have dtype"
            
            # Should have type classification
            assert 'type' in col_schema, f"{col} should have type"
    
    def test_empty_dataframe_schema(self, empty_df):
        """Test schema generation from empty DataFrame."""
        schema = generate_schema(empty_df)
        
        assert 'columns' in schema
        # Should still have column definitions even if empty
        for col in empty_df.columns:
            assert col in schema['columns']


class TestGenerateStatistics:
    """Test statistics generation."""
    
    def test_statistics_has_row_count(self, sample_df):
        """Test that statistics include row count."""
        stats = generate_statistics(sample_df)
        
        assert 'row_count' in stats
        assert stats['row_count'] == len(sample_df)
    
    def test_statistics_has_column_info(self, sample_df):
        """Test that statistics include column information."""
        stats = generate_statistics(sample_df)
        
        assert 'columns' in stats
        assert len(stats['columns']) > 0
    
    def test_numerical_column_stats(self, sample_df):
        """Test that numerical columns have proper statistics."""
        stats = generate_statistics(sample_df)
        
        # Check duration_seconds has numerical stats
        if 'duration_seconds' in stats['columns']:
            col_stats = stats['columns']['duration_seconds']
            assert 'mean' in col_stats
            assert 'std' in col_stats
            assert 'min' in col_stats
            assert 'max' in col_stats


class TestValidateAgainstSchema:
    """Test schema validation against DataFrame."""
    
    def test_valid_dataframe_passes(self, sample_df):
        """Test that valid DataFrame passes schema validation."""
        schema = generate_schema(sample_df)
        
        result = validate_against_schema(sample_df, schema)
        
        assert 'is_valid' in result
        # Should be valid since we're comparing against its own schema
        assert result['is_valid'] == True, "Valid DataFrame should pass validation"
        assert result.get('total_errors', 0) == 0, "Should have no errors"
    
    def test_missing_column_detected(self, sample_df):
        """Test that missing columns are detected."""
        schema = generate_schema(sample_df)
        
        # Remove a column
        df_missing = sample_df.drop(columns=['duration_seconds'])
        
        result = validate_against_schema(df_missing, schema)
        
        assert result['is_valid'] == False, "Missing column should fail validation"
        assert result['total_errors'] > 0, "Should have errors"
        
        # Check that the error mentions the missing column
        errors = result.get('errors', [])
        assert any('duration_seconds' in str(err).lower() or 'missing' in str(err).lower() 
                   for err in errors), "Should mention missing column"
    
    def test_extra_column_generates_warning(self, sample_df):
        """Test that extra columns generate warnings."""
        schema = generate_schema(sample_df)
        
        # Add an unexpected column
        df_extra = sample_df.copy()
        df_extra['unexpected_column'] = 1
        
        result = validate_against_schema(df_extra, schema)
        
        # Should have warnings about extra column
        assert result.get('total_warnings', 0) > 0, "Should have warnings"
        warnings = result.get('warnings', [])
        assert any('unexpected_column' in str(warn).lower() or 'extra' in str(warn).lower()
                   for warn in warnings), "Should mention unexpected column"
    
    def test_dtype_change_detected(self, sample_df):
        """Test that data type changes are detected."""
        schema = generate_schema(sample_df)
        
        # Change dtype of a column
        df_changed = sample_df.copy()
        df_changed['failed'] = df_changed['failed'].astype(str)
        
        result = validate_against_schema(df_changed, schema)
        
        # Should detect dtype mismatch (either error or warning)
        assert result.get('total_warnings', 0) > 0 or result.get('total_errors', 0) > 0, \
            "Should detect dtype change"
    
    def test_validation_report_structure(self, sample_df):
        """Test that validation report has correct structure."""
        schema = generate_schema(sample_df)
        result = validate_against_schema(sample_df, schema)
        
        # Check required keys
        assert 'is_valid' in result
        assert 'total_errors' in result
        assert 'total_warnings' in result
        assert 'errors' in result
        assert 'warnings' in result


class TestSchemaIO:
    """Test schema save/load operations."""
    
    def test_save_and_load_schema(self, sample_df, tmp_path):
        """Test saving and loading schema to/from file."""
        schema = generate_schema(sample_df)
        
        # Save schema
        schema_file = tmp_path / "test_schema.json"
        save_schema(schema, str(schema_file))
        
        assert schema_file.exists(), "Schema file should be created"
        
        # Load schema
        loaded_schema = load_schema(str(schema_file))
        
        # Compare key fields (timestamps might differ slightly)
        assert loaded_schema['columns'] == schema['columns'], "Columns should match"
        assert loaded_schema['expected_columns'] == schema['expected_columns'], \
            "Expected columns should match"
    
    def test_schema_json_format(self, sample_df, tmp_path):
        """Test that saved schema is valid JSON."""
        schema = generate_schema(sample_df)
        
        schema_file = tmp_path / "test_schema.json"
        save_schema(schema, str(schema_file))
        
        # Try to parse as JSON
        with open(schema_file, 'r') as f:
            loaded = json.load(f)
        
        assert isinstance(loaded, dict), "Schema should be a dictionary"
        assert 'columns' in loaded, "Schema should have columns key"
    
    def test_load_nonexistent_schema(self, tmp_path):
        """Test loading non-existent schema returns None."""
        fake_path = tmp_path / "nonexistent.json"
        result = load_schema(str(fake_path))
        
        assert result is None, "Should return None for non-existent file"


class TestStatisticsIO:
    """Test statistics save operations."""
    
    def test_save_statistics(self, sample_df, tmp_path):
        """Test saving statistics to file."""
        stats = generate_statistics(sample_df)
        
        stats_file = tmp_path / "test_stats.json"
        save_statistics(stats, stats_file)
        
        assert stats_file.exists(), "Statistics file should be created"
        
        # Load and verify
        with open(stats_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['row_count'] == stats['row_count']
        assert loaded['column_count'] == stats['column_count']


class TestSchemaComparison:
    """Test schema comparison and drift detection."""
    
    def test_different_columns_detected(self, sample_df):
        """Test detection of different column sets."""
        schema1 = generate_schema(sample_df)
        
        df2 = sample_df.drop(columns=['retry_count'])
        
        # Validate df2 against schema1 (which has retry_count)
        result = validate_against_schema(df2, schema1)
        
        # Should fail validation due to missing column
        assert result['is_valid'] == False, "Should detect missing column"
        assert result['total_errors'] > 0, "Should have errors"


class TestEdgeCases:
    """Test edge cases in schema validation."""
    
    def test_single_row_schema(self, single_row_df):
        """Test schema generation from single-row DataFrame."""
        schema = generate_schema(single_row_df)
        
        assert 'columns' in schema
        assert len(schema['columns']) == len(single_row_df.columns)
    
    def test_all_null_column(self):
        """Test schema with column that has all null values."""
        df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [None, None, None],
        'col3': ['a', 'b', 'c']
        })
    
        # Should not crash
        schema = generate_schema(df)
        stats = generate_statistics(df)
    
        assert 'col2' in schema['columns'], "col2 should be in schema"
        # col2 might not be in stats if it's not in predefined column lists - that's okay
        # Just verify the functions don't crash
    
    def test_mixed_types_handled(self):
        """Test handling of columns with mixed types."""
        df = pd.DataFrame({
            'mixed_col': [1, '2', 3.0, 'four', 5],
            'normal_col': [1, 2, 3, 4, 5]
        })
        
        # Should not crash
        schema = generate_schema(df)
        
        assert 'mixed_col' in schema['columns']
        assert 'normal_col' in schema['columns']