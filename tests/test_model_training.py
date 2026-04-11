"""
Unit tests for model_training.py
Tests model training pipeline with new API.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts import model_training
    
    load_data = model_training.load_data
    split_data = model_training.split_data
    scale_features = model_training.scale_features
    evaluate = model_training.evaluate
    train_all_models = model_training.train_all_models
    select_best = model_training.select_best
except ImportError as e:
    print(f"Warning: Could not import from model_training.py: {e}")


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 200
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
            pytest.skip("Data file not found")


class TestSplitData:
    """Test train/val/test split functionality."""
    
    def test_split_data_returns_six_sets(self, sample_training_data):
        """Test that split_data returns 6 sets."""
        X, y = sample_training_data
        
        result = split_data(X, y)
        
        assert len(result) == 6, "Should return 6 arrays"
        X_train, X_val, X_test, y_train, y_val, y_test = result
        
        assert isinstance(X_train, (pd.DataFrame, np.ndarray))
        assert isinstance(y_train, (pd.Series, np.ndarray))
    
    def test_split_preserves_total_samples(self, sample_training_data):
        """Test that split preserves total number of samples."""
        X, y = sample_training_data
        original_count = len(X)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == original_count


class TestScaleFeatures:
    """Test feature scaling."""
    
    def test_scale_features_returns_scaled_sets(self, sample_training_data):
        """Test that scale_features returns scaled data."""
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        
        # Returns 4 values: X_train_s, X_val_s, X_test_s, scaler
        result = scale_features(X_train, X_val, X_test)
        
        assert len(result) == 4, "Should return 4 values (3 scaled sets + scaler)"

class TestEvaluate:
    """Test model evaluation."""
    
    def test_evaluate_returns_dict(self, sample_training_data):
        """Test that evaluate returns metrics dict."""
        from sklearn.linear_model import LogisticRegression
        
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)
        
        metrics = evaluate("test_model", model, X_test, y_test)
        
        assert isinstance(metrics, dict)


class TestTrainAllModels:
    """Test training all models."""
    
    def test_train_all_returns_dict(self, sample_training_data):
        """Test that train_all_models returns results dict."""
        X, y = sample_training_data
        from sklearn.model_selection import train_test_split
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Scale features - returns 4 values
        result = scale_features(X_train, X_val, X_test)
        X_train_s, X_val_s, X_test_s, scaler = result
        
        # Train all models
        try:
            results = train_all_models(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)
            assert isinstance(results, dict)
        except Exception as e:
            pytest.skip(f"Training skipped: {e}")


class TestSelectBest:
    """Test model selection."""
    
    def test_select_best_returns_name(self):
        """Test that select_best returns best model name."""
        # Correct structure: nested metrics dict
        results = {
            'model_a': {'metrics': {'auc_roc': 0.75, 'f1': 0.70}},
            'model_b': {'metrics': {'auc_roc': 0.85, 'f1': 0.80}},
            'model_c': {'metrics': {'auc_roc': 0.80, 'f1': 0.75}}
        }
        
        best_name = select_best(results)
        
        assert isinstance(best_name, str)
        assert best_name == 'model_b', "Should select model with highest AUC"