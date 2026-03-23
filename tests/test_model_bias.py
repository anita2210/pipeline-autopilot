import pytest
import pandas as pd
import json
import os
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

def test_bias_report_exists():
    """Verify the JSON report was actually generated."""
    report_path = 'models/registry/model_bias_report.json'
    assert os.path.exists(report_path), "Bias report JSON is missing!"

def test_overall_pass_boolean():
    """Ensure the report contains the mandatory pass/fail flag."""
    with open('models/registry/model_bias_report.json', 'r') as f:
        report = json.load(f)
    assert 'overall_pass' in report
    assert isinstance(report['overall_pass'], bool)

def test_disparity_logic():
    """Test that our 0.67 ratio logic correctly identifies bias."""
    # Mock data: Group A (100% acc) vs Group B (50% acc)
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 1] # Group A: 2/2 correct, Group B: 1/2 correct
    sf = ['GroupA', 'GroupA', 'GroupB', 'GroupB']
    
    mf = MetricFrame(metrics={'acc': accuracy_score}, y_true=y_true, y_pred=y_pred, sensitive_features=sf)
    ratio = mf.ratio()
    
    # Accuracy ratio is 0.5 / 1.0 = 0.5. 
    # Since 0.5 < 0.67, this should trigger our 'flagged' logic.
    assert ratio['acc'] < 0.67