"""
Shared pytest fixtures for Pipeline Autopilot testing.
All test files can use these fixtures by simply including them as function arguments.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_df():
    """
    Create a clean sample DataFrame with 50 rows for basic testing.
    Matches the Pipeline Autopilot schema (27 features).
    """
    np.random.seed(42)
    n_rows = 50
    
    base_time = datetime(2024, 1, 1)
    
    data = {
        'run_id': [f'run_{i:05d}' for i in range(n_rows)],
        'trigger_time': [base_time + timedelta(hours=i) for i in range(n_rows)],
        'pipeline_name': np.random.choice(['build', 'test', 'deploy'], n_rows),
        'repo': np.random.choice(['airflow', 'spark', 'kubernetes'], n_rows),
        'head_branch': np.random.choice(['main', 'develop', 'feature/x'], n_rows),
        'trigger_type': np.random.choice(['push', 'pull_request', 'schedule', 'workflow_dispatch'], n_rows),
        'failure_type': np.random.choice(['test_failure', 'build_failure', 'timeout', 'dependency_failure', None], n_rows),
        'error_message': [f'Error {i}' if i % 5 == 0 else None for i in range(n_rows)],
        'day_of_week': np.random.randint(0, 7, n_rows),
        'hour': np.random.randint(0, 24, n_rows),
        'duration_seconds': np.random.randint(60, 3600, n_rows),
        'avg_duration_7_runs': np.random.randint(100, 3000, n_rows),
        'duration_deviation': np.random.uniform(-500, 500, n_rows),
        'prev_run_status': np.random.choice([0, 1], n_rows),
        'failures_last_7_runs': np.random.randint(0, 7, n_rows),
        'workflow_failure_rate': np.random.uniform(0, 1, n_rows),
        'hours_since_last_run': np.random.uniform(0.5, 48, n_rows),
        'total_jobs': np.random.randint(1, 10, n_rows),
        'failed_jobs': np.random.randint(0, 5, n_rows),
        'retry_count': np.random.randint(0, 3, n_rows),
        'concurrent_runs': np.random.randint(1, 5, n_rows),
        'is_weekend': np.random.choice([0, 1], n_rows),
        'is_main_branch': np.random.choice([0, 1], n_rows),
        'is_first_run': np.random.choice([0, 1], n_rows),
        'is_bot_triggered': np.random.choice([0, 1], n_rows),
        'failed': np.random.choice([0, 1], n_rows, p=[0.7, 0.3])  # 30% failure rate
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_with_missing():
    """
    DataFrame with intentional missing values for testing missing data handling.
    """
    np.random.seed(42)
    n_rows = 50
    
    base_time = datetime(2024, 1, 1)
    
    data = {
        'run_id': [f'run_{i:05d}' for i in range(n_rows)],
        'trigger_time': [base_time + timedelta(hours=i) for i in range(n_rows)],
        'pipeline_name': np.random.choice(['build', 'test', 'deploy', None], n_rows),
        'repo': np.random.choice(['airflow', 'spark', None], n_rows),
        'duration_seconds': [np.nan if i % 10 == 0 else np.random.randint(60, 3600) for i in range(n_rows)],
        'workflow_failure_rate': [np.nan if i % 8 == 0 else np.random.uniform(0, 1) for i in range(n_rows)],
        'trigger_type': np.random.choice(['push', 'pull_request', None], n_rows),
        'avg_duration_7_runs': [np.nan if i % 12 == 0 else np.random.randint(100, 3000) for i in range(n_rows)],
        'failed': np.random.choice([0, 1], n_rows)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_with_anomalies():
    """
    DataFrame with intentional anomalies for anomaly detection testing.
    """
    np.random.seed(42)
    n_rows = 50
    
    base_time = datetime(2024, 1, 1)
    
    data = {
        'run_id': [f'run_{i:05d}' for i in range(n_rows)],
        'trigger_time': [base_time + timedelta(hours=i) for i in range(n_rows)],
        'pipeline_name': np.random.choice(['build', 'test', 'deploy'], n_rows),
        'repo': np.random.choice(['airflow', 'spark'], n_rows),
        'duration_seconds': np.random.randint(60, 3600, n_rows),
        'workflow_failure_rate': np.random.uniform(0, 1, n_rows),
        'total_jobs': np.random.randint(1, 10, n_rows),
        'failed_jobs': np.random.randint(0, 5, n_rows),
        'failed': np.random.choice([0, 1], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Inject anomalies
    df.loc[0, 'duration_seconds'] = -100  # Negative duration (ANOMALY)
    df.loc[1, 'workflow_failure_rate'] = 1.5  # Rate > 1 (ANOMALY)
    df.loc[2, 'failed_jobs'] = 15  # failed_jobs > total_jobs (ANOMALY)
    df.loc[2, 'total_jobs'] = 5
    df.loc[5, 'duration_seconds'] = 100000  # Extreme outlier (ANOMALY)
    df.loc[10, 'duration_seconds'] = np.nan  # Missing value (ANOMALY)
    
    return df


@pytest.fixture
def sample_df_with_duplicates(sample_df):
    """
    DataFrame with duplicate rows for duplicate removal testing.
    """
    df = sample_df.copy()
    # Add 5 exact duplicates
    duplicates = df.iloc[:5].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    return df


@pytest.fixture
def empty_df():
    """
    Empty DataFrame with correct columns for edge case testing.
    """
    columns = ['run_id', 'trigger_time', 'pipeline_name', 'repo', 'duration_seconds', 
               'workflow_failure_rate', 'failed']
    return pd.DataFrame(columns=columns)


@pytest.fixture
def single_row_df():
    """
    DataFrame with a single row for edge case testing.
    """
    data = {
        'run_id': ['run_00001'],
        'trigger_time': [datetime(2024, 1, 1)],
        'pipeline_name': ['build'],
        'repo': ['airflow'],
        'duration_seconds': [120],
        'workflow_failure_rate': [0.3],
        'trigger_type': ['push'],
        'is_bot_triggered': [0],
        'total_jobs': [5],
        'failed_jobs': [2],
        'failed': [1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_schema():
    """
    Sample schema dictionary for schema validation testing.
    """
    return {
        'columns': {
            'run_id': {'dtype': 'object', 'nullable': False},
            'trigger_time': {'dtype': 'datetime64[ns]', 'nullable': False},
            'pipeline_name': {'dtype': 'object', 'nullable': True},
            'duration_seconds': {'dtype': 'int64', 'nullable': False, 'min': 0, 'max': 86400},
            'workflow_failure_rate': {'dtype': 'float64', 'nullable': False, 'min': 0.0, 'max': 1.0},
            'failed': {'dtype': 'int64', 'nullable': False}
        },
        'required_columns': ['run_id', 'trigger_time', 'duration_seconds', 'failed']
    }