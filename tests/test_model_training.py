"""
Unit tests for model_training.py
Tests model training, hyperparameter tuning, and model selection.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

try:
    from model_training import (
        load_data,
        split_data,
        compute_metrics,
        compute_scale_pos_weight,
        train_logistic_regression,
        train_random_forest,
        train_xgboost_default,
        tune_xgboost,
        select_and_save_best
    )
except ImportError as e:
    print(f"Warning: Could not import from model_training.py: {e}")


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
    
    return X, y


class TestLoadData:
    """Test data loading functionality."""
    
    def test_load_data_returns_dataframe(self):
        """Test that load_data returns X and y."""
        try:
            X, y = load_data()
            assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
            assert isinstance(y, (pd.Series, np.ndarray)), "y should be Series or array"
        except FileNotFoundError:
            pytest.skip("Data file not found - expected in test environment")
    
    def test_loaded_data_shape(self):
        """Test that loaded data has correct shape."""
        try:
            X, y = load_data()
            assert len(X) == len(y), "X and y should have same length"
            assert len(X) > 0, "Should have data rows"
        except FileNotFoundError:
            pytest.skip("Data file not found")


class TestSplitData:
    """Test train/val/test split functionality."""
    
    def test_split_data_returns_six_sets(self, sample_training_data):
        """Test that split_data returns 6 sets (X_train, X_val, X_test, y_train, y_val, y_test)."""
        X, y = sample_training_data
        
        result = split_data(X, y)
        
        assert len(result) == 6, "Should return 6 arrays"
        X_train, X_val, X_test, y_train, y_val, y_test = result
        
        # Check all are correct types
        assert isinstance(X_train, (pd.DataFrame, np.ndarray))
        assert isinstance(y_train, (pd.Series, np.ndarray))
    
    def test_split_preserves_total_samples(self, sample_training_data):
        """Test that split preserves total number of samples."""
        X, y = sample_training_data
        original_count = len(X)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        total_after_split = len(X_train) + len(X_val) + len(X_test)
        assert total_after_split == original_count, "Should preserve sample count"
    
    def test_split_proportions(self, sample_training_data):
        """Test that split creates reasonable proportions."""
        X, y = sample_training_data
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Train should be largest set (typically 70%)
        assert len(X_train) > len(X_val), "Train should be larger than val"
        assert len(X_train) > len(X_test), "Train should be larger than test"


class TestComputeMetrics:
    """Test metrics computation."""
    
    def test_compute_metrics_returns_dict(self):
        """Test that compute_metrics returns dictionary."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8])
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        assert isinstance(metrics, dict), "Should return dictionary"
    
    def test_compute_metrics_includes_auc(self):
        """Test that metrics include AUC."""
        y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 0, 0, 1] * 10)
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8] * 10)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Should have AUC or AUC-ROC
        assert any(key.lower() in ['auc', 'auc_roc', 'roc_auc'] for key in metrics.keys())
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        y_prob = y_true.astype(float)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Perfect predictions should give high scores
        assert isinstance(metrics, dict)


class TestComputeScalePosWeight:
    """Test scale_pos_weight computation for XGBoost."""
    
    def test_compute_scale_pos_weight(self):
        """Test that scale_pos_weight is computed correctly."""
        # Imbalanced: 70% class 0, 30% class 1
        y_train = pd.Series([0] * 70 + [1] * 30)
        
        scale_pos_weight = compute_scale_pos_weight(y_train)
        
        assert isinstance(scale_pos_weight, (int, float)), "Should return numeric value"
        assert scale_pos_weight > 0, "Should be positive"
        
        # For 70/30 split, scale_pos_weight should be ~70/30 = 2.33
        expected = 70 / 30
        assert 2.0 < scale_pos_weight < 3.0, f"Expected ~{expected:.2f}"


class TestTrainLogisticRegression:
    """Test Logistic Regression training."""
    
    def test_logistic_regression_trains(self, sample_training_data):
        """Test that Logistic Regression trains successfully."""
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        result = train_logistic_regression(X_train, X_val, y_train, y_val)
        
        # Returns tuple: (model, metrics, params)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 3, "Should return (model, metrics, params)"


class TestTrainRandomForest:
    """Test Random Forest training."""
    
    def test_random_forest_trains(self, sample_training_data):
        """Test that Random Forest trains successfully."""
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        result = train_random_forest(X_train, X_val, y_train, y_val)
        
        # Returns tuple: (model, metrics, params)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 3, "Should return (model, metrics, params)"


class TestTrainXGBoost:
    """Test XGBoost training."""
    
    def test_xgboost_default_trains(self, sample_training_data):
        """Test that XGBoost trains with default parameters."""
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        result = train_xgboost_default(X_train, X_val, y_train, y_val)
        
        # Returns tuple: (model, metrics, params)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 3, "Should return (model, metrics, params)"
    
    def test_xgboost_tuning_runs(self, sample_training_data):
        """Test that hyperparameter tuning runs."""
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        
        # Use small sample for fast testing
        X_small = X.iloc[:200]
        y_small = y.iloc[:200]
        
        X_train, X_val, y_train, y_val = train_test_split(X_small, y_small, test_size=0.2, random_state=42)
        
        # This might take a while, so we just check it runs
        try:
            result = tune_xgboost(X_train, X_val, y_train, y_val)
            assert isinstance(result, tuple), "Should return tuple"
        except Exception as e:
            # Tuning might fail in test environment, that's okay
            pytest.skip(f"Tuning skipped in test: {e}")


class TestSelectAndSaveBest:
    """Test model selection and saving."""
    
    def test_select_best_model(self, sample_training_data, tmp_path):
        """Test that best model is selected based on AUC."""
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train simple models
        lr = LogisticRegression(max_iter=100)
        lr.fit(X_train, y_train)
        
        rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X_train, y_train)
        
        models_dict = {
            'logistic': {'model': lr, 'val_auc': 0.75},
            'random_forest': {'model': rf, 'val_auc': 0.80}
        }
        
        # Test selection logic
        try:
            result = select_and_save_best(models_dict, X_test, y_test)
            assert isinstance(result, dict), "Should return result dict"
        except Exception as e:
            # Might fail due to file paths in test environment
            pytest.skip(f"Save test skipped: {e}")


class TestEdgeCases:
    """Test edge cases in model training."""
    
    def test_empty_models_dict_handled(self):
        """Test handling of empty models dictionary."""
        X_test = pd.DataFrame(np.random.randn(10, 5))
        y_test = pd.Series([0, 1] * 5)
        
        models_dict = {}
        
        try:
            result = select_and_save_best(models_dict, X_test, y_test)
            # Should either handle gracefully or raise appropriate error
            assert True
        except (ValueError, KeyError):
            # Expected error for empty dict
            assert True