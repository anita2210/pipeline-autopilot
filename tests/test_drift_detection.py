"""
test_drift_detection.py
-----------------------
Unit tests for monitoring/drift_detection.py, monitoring/performance_monitor.py,
and monitoring/retrain_trigger.py.

Author  : Member 4 (MLOps Monitor)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_reference_df():
    """Generate a sample reference DataFrame with monitored features."""
    np.random.seed(42)
    return pd.DataFrame({
        "retry_count": np.random.randint(0, 5, 200),
        "duration_deviation": np.random.normal(0, 1, 200),
        "failures_last_7_runs": np.random.randint(0, 7, 200),
        "workflow_failure_rate": np.random.uniform(0, 0.5, 200),
        "concurrent_runs": np.random.randint(1, 10, 200),
    })


@pytest.fixture
def sample_current_df():
    """Generate current data similar to reference (no drift)."""
    np.random.seed(99)
    return pd.DataFrame({
        "retry_count": np.random.randint(0, 5, 200),
        "duration_deviation": np.random.normal(0, 1, 200),
        "failures_last_7_runs": np.random.randint(0, 7, 200),
        "workflow_failure_rate": np.random.uniform(0, 0.5, 200),
        "concurrent_runs": np.random.randint(1, 10, 200),
    })


@pytest.fixture
def sample_drifted_df():
    """Generate current data with clear drift from reference."""
    np.random.seed(7)
    return pd.DataFrame({
        "retry_count": np.random.randint(5, 15, 200),
        "duration_deviation": np.random.normal(5, 2, 200),
        "failures_last_7_runs": np.random.randint(5, 14, 200),
        "workflow_failure_rate": np.random.uniform(0.6, 1.0, 200),
        "concurrent_runs": np.random.randint(10, 20, 200),
    })


@pytest.fixture
def sample_predictions_csv(tmp_path):
    """Create a sample predictions log CSV with labeled predictions."""
    np.random.seed(42)
    n = 150
    actual = np.random.randint(0, 2, n)
    probability = np.where(actual == 1,
                           np.random.beta(5, 2, n),
                           np.random.beta(2, 5, n))
    prediction = (probability >= 0.5).astype(int)

    df = pd.DataFrame({
        "timestamp": ["2026-04-09T00:00:00"] * n,
        "run_id": [str(i) for i in range(n)],
        "probability": probability,
        "prediction": prediction,
        "actual": actual,
        "risk_level": ["HIGH" if p >= 0.75 else "LOW" for p in probability],
        "retry_count": np.random.randint(0, 5, n),
        "duration_deviation": np.random.normal(0, 1, n),
        "failures_last_7_runs": np.random.randint(0, 7, n),
        "workflow_failure_rate": np.random.uniform(0, 0.5, n),
        "concurrent_runs": np.random.randint(1, 10, n),
    })

    log_path = tmp_path / "predictions_log.csv"
    df.to_csv(log_path, index=False)
    return log_path


@pytest.fixture
def drift_summary_json(tmp_path):
    """Create a sample drift_summary.json with no drift."""
    summary = {
        "timestamp": "2026-04-09T00:00:00",
        "monitored_features": ["retry_count", "duration_deviation",
                               "failures_last_7_runs", "workflow_failure_rate",
                               "concurrent_runs"],
        "drift_threshold": 0.3,
        "overall_drift_score": 0.05,
        "drift_detected": False,
        "drifted_features": [],
        "per_feature_scores": {
            "retry_count": 0.04,
            "duration_deviation": 0.03,
            "failures_last_7_runs": 0.06,
            "workflow_failure_rate": 0.07,
            "concurrent_runs": 0.05,
        },
        "action": "NO_ACTION",
    }
    path = tmp_path / "drift_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f)
    return path


@pytest.fixture
def drift_summary_json_flagged(tmp_path):
    """Create a sample drift_summary.json with drift detected."""
    summary = {
        "timestamp": "2026-04-09T00:00:00",
        "monitored_features": ["retry_count", "duration_deviation",
                               "failures_last_7_runs", "workflow_failure_rate",
                               "concurrent_runs"],
        "drift_threshold": 0.3,
        "overall_drift_score": 0.47,
        "drift_detected": True,
        "drifted_features": ["retry_count", "duration_deviation"],
        "per_feature_scores": {
            "retry_count": 0.55,
            "duration_deviation": 0.48,
            "failures_last_7_runs": 0.42,
            "workflow_failure_rate": 0.45,
            "concurrent_runs": 0.44,
        },
        "action": "RETRAIN",
    }
    path = tmp_path / "drift_summary_flagged.json"
    with open(path, "w") as f:
        json.dump(summary, f)
    return path


@pytest.fixture
def performance_report_json(tmp_path):
    """Create a sample performance_report.json with good AUC."""
    report = {
        "timestamp": "2026-04-09T00:00:00",
        "rolling_window": 100,
        "auc_threshold": 0.85,
        "auc": 0.9946,
        "f1": 0.6154,
        "n_samples": 100,
        "auc_flagged": False,
        "action": "NO_ACTION",
    }
    path = tmp_path / "performance_report.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return path


@pytest.fixture
def performance_report_json_flagged(tmp_path):
    """Create a sample performance_report.json with degraded AUC."""
    report = {
        "timestamp": "2026-04-09T00:00:00",
        "rolling_window": 100,
        "auc_threshold": 0.85,
        "auc": 0.4932,
        "f1": 0.1053,
        "n_samples": 100,
        "auc_flagged": True,
        "action": "RETRAIN",
    }
    path = tmp_path / "performance_report_flagged.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return path


# ===========================================================================
# Tests: drift_detection.py
# ===========================================================================

class TestGenerateSyntheticCurrent:
    """Tests for generate_synthetic_current()."""

    def test_no_drift_same_shape(self, sample_reference_df):
        """Synthetic data should have same shape as reference."""
        from monitoring.drift_detection import generate_synthetic_current
        result = generate_synthetic_current(sample_reference_df, drift=False)
        assert result.shape == sample_reference_df.shape

    def test_drift_injected_values_shifted(self, sample_reference_df):
        """Drift injection should shift values higher than reference."""
        from monitoring.drift_detection import generate_synthetic_current
        drifted = generate_synthetic_current(sample_reference_df, drift=True)
        for col in sample_reference_df.columns:
            assert drifted[col].mean() > sample_reference_df[col].mean()

    def test_output_is_dataframe(self, sample_reference_df):
        """Output should always be a DataFrame."""
        from monitoring.drift_detection import generate_synthetic_current
        result = generate_synthetic_current(sample_reference_df, drift=False)
        assert isinstance(result, pd.DataFrame)


class TestExtractDriftScore:
    """Tests for extract_drift_score()."""

    def test_returns_required_keys(self, sample_reference_df, sample_current_df, tmp_path):
        """Drift result should contain all required keys."""
        from monitoring.drift_detection import run_evidently_report, extract_drift_score
        report = run_evidently_report(sample_reference_df, sample_current_df, tmp_path)
        result = extract_drift_score(report)
        assert "overall_drift_score" in result
        assert "per_feature" in result
        assert "drifted_features" in result
        assert "drift_detected" in result
        assert "threshold" in result

    def test_no_drift_score_is_float(self, sample_reference_df, sample_current_df, tmp_path):
        """Drift score should be a valid float between 0 and 1."""
        from monitoring.drift_detection import run_evidently_report, extract_drift_score
        report = run_evidently_report(sample_reference_df, sample_current_df, tmp_path)
        result = extract_drift_score(report)
        assert isinstance(result["overall_drift_score"], float)
        assert 0.0 <= result["overall_drift_score"] <= 1.0

    
    def test_per_feature_scores_all_present(self, sample_reference_df, sample_current_df, tmp_path):
        """Per-feature scores should include all 5 monitored features."""
        from monitoring.drift_detection import (
            run_evidently_report, extract_drift_score, MONITORED_FEATURES
        )
        report = run_evidently_report(sample_reference_df, sample_current_df, tmp_path)
        result = extract_drift_score(report)
        for feature in MONITORED_FEATURES:
            assert feature in result["per_feature"]


class TestSaveDriftSummary:
    """Tests for save_drift_summary()."""

    def test_json_saved(self, tmp_path):
        """Drift summary should be saved as valid JSON."""
        from monitoring.drift_detection import save_drift_summary
        drift_result = {
            "overall_drift_score": 0.05,
            "drift_detected": False,
            "drifted_features": [],
            "per_feature": {"retry_count": 0.05},
            "threshold": 0.3,
        }
        path = tmp_path / "drift_summary.json"
        save_drift_summary(drift_result, summary_path=path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "overall_drift_score" in data
        assert "action" in data

    def test_action_retrain_when_flagged(self, tmp_path):
        """Action should be RETRAIN when drift is detected."""
        from monitoring.drift_detection import save_drift_summary
        drift_result = {
            "overall_drift_score": 0.47,
            "drift_detected": True,
            "drifted_features": ["retry_count"],
            "per_feature": {"retry_count": 0.47},
            "threshold": 0.3,
        }
        path = tmp_path / "drift_summary.json"
        summary = save_drift_summary(drift_result, summary_path=path)
        assert summary["action"] == "RETRAIN"

    def test_action_no_action_when_clean(self, tmp_path):
        """Action should be NO_ACTION when no drift detected."""
        from monitoring.drift_detection import save_drift_summary
        drift_result = {
            "overall_drift_score": 0.05,
            "drift_detected": False,
            "drifted_features": [],
            "per_feature": {"retry_count": 0.05},
            "threshold": 0.3,
        }
        path = tmp_path / "drift_summary.json"
        summary = save_drift_summary(drift_result, summary_path=path)
        assert summary["action"] == "NO_ACTION"


# ===========================================================================
# Tests: performance_monitor.py
# ===========================================================================

class TestComputeRollingMetrics:
    """Tests for compute_rolling_metrics()."""

    def test_returns_required_keys(self, sample_predictions_csv):
        """Metrics dict should contain all required keys."""
        from monitoring.performance_monitor import compute_rolling_metrics
        result = compute_rolling_metrics(log_path=sample_predictions_csv)
        assert "auc" in result
        assert "f1" in result
        assert "n_samples" in result
        assert "auc_flagged" in result
        assert "threshold" in result

    def test_auc_in_valid_range(self, sample_predictions_csv):
        """AUC should be between 0 and 1."""
        from monitoring.performance_monitor import compute_rolling_metrics
        result = compute_rolling_metrics(log_path=sample_predictions_csv)
        assert 0.0 <= result["auc"] <= 1.0

    def test_good_predictions_not_flagged(self, sample_predictions_csv):
        """Good predictions (correlated with actuals) should not flag AUC."""
        from monitoring.performance_monitor import compute_rolling_metrics
        result = compute_rolling_metrics(log_path=sample_predictions_csv)
        assert result["auc_flagged"] is False

    def test_missing_log_raises_error(self, tmp_path):
        """Missing predictions log should raise FileNotFoundError."""
        from monitoring.performance_monitor import compute_rolling_metrics
        with pytest.raises(FileNotFoundError):
            compute_rolling_metrics(log_path=tmp_path / "nonexistent.csv")

    def test_auc_flagged_when_random_predictions(self, tmp_path):
        """Random predictions should produce low AUC and flag it."""
        from monitoring.performance_monitor import compute_rolling_metrics
        np.random.seed(0)
        n = 150
        actual = np.random.randint(0, 2, n)
        probability = np.random.uniform(0, 1, n)
        prediction = (probability >= 0.5).astype(int)

        df = pd.DataFrame({
            "timestamp": ["2026-04-09"] * n,
            "run_id": [str(i) for i in range(n)],
            "probability": probability,
            "prediction": prediction,
            "actual": actual,
            "risk_level": ["LOW"] * n,
            "retry_count": [0] * n,
            "duration_deviation": [0.0] * n,
            "failures_last_7_runs": [0] * n,
            "workflow_failure_rate": [0.0] * n,
            "concurrent_runs": [1] * n,
        })
        log_path = tmp_path / "bad_predictions.csv"
        df.to_csv(log_path, index=False)

        result = compute_rolling_metrics(log_path=log_path)
        assert result["auc_flagged"] is True


class TestLogPrediction:
    """Tests for log_prediction()."""

    def test_creates_log_file(self, tmp_path):
        """log_prediction should create the CSV file if it does not exist."""
        from monitoring.performance_monitor import log_prediction
        log_path = tmp_path / "predictions_log.csv"
        log_prediction(
            run_id="test_001",
            probability=0.8,
            prediction=1,
            actual=1,
            features={"retry_count": 2, "duration_deviation": 1.5,
                      "failures_last_7_runs": 3, "workflow_failure_rate": 0.3,
                      "concurrent_runs": 2},
            log_path=log_path,
        )
        assert log_path.exists()

    def test_appends_multiple_rows(self, tmp_path):
        """Multiple log_prediction calls should append rows."""
        from monitoring.performance_monitor import log_prediction
        log_path = tmp_path / "predictions_log.csv"
        features = {"retry_count": 1, "duration_deviation": 0.5,
                    "failures_last_7_runs": 1, "workflow_failure_rate": 0.1,
                    "concurrent_runs": 1}
        for i in range(5):
            log_prediction(
                run_id=f"run_{i}",
                probability=0.6,
                prediction=1,
                actual=1,
                features=features,
                log_path=log_path,
            )
        df = pd.read_csv(log_path)
        assert len(df) == 5

    def test_risk_level_high_for_high_probability(self, tmp_path):
        """Probability >= 0.75 should be tagged as HIGH risk."""
        from monitoring.performance_monitor import log_prediction
        log_path = tmp_path / "predictions_log.csv"
        log_prediction(
            run_id="high_risk_run",
            probability=0.9,
            prediction=1,
            actual=None,
            features={"retry_count": 3, "duration_deviation": 2.0,
                      "failures_last_7_runs": 5, "workflow_failure_rate": 0.4,
                      "concurrent_runs": 3},
            log_path=log_path,
        )
        df = pd.read_csv(log_path)
        assert df.iloc[0]["risk_level"] == "HIGH"


# ===========================================================================
# Tests: retrain_trigger.py
# ===========================================================================

class TestShouldRetrain:
    """Tests for should_retrain() decision logic."""

    def test_no_retrain_when_healthy(self):
        """No retraining when both drift and AUC are within thresholds."""
        from monitoring.retrain_trigger import should_retrain
        drift = {"overall_drift_score": 0.05, "drift_detected": False}
        perf = {"auc": 0.95, "auc_flagged": False}
        triggered, reason = should_retrain(drift, perf)
        assert triggered is False
        assert "No retraining needed" in reason

    def test_retrain_when_drift_detected(self):
        """Retraining should trigger when drift is detected."""
        from monitoring.retrain_trigger import should_retrain
        drift = {"overall_drift_score": 0.47, "drift_detected": True}
        perf = {"auc": 0.95, "auc_flagged": False}
        triggered, reason = should_retrain(drift, perf)
        assert triggered is True
        assert "drift" in reason.lower()

    def test_retrain_when_auc_drops(self):
        """Retraining should trigger when AUC drops below threshold."""
        from monitoring.retrain_trigger import should_retrain
        drift = {"overall_drift_score": 0.05, "drift_detected": False}
        perf = {"auc": 0.49, "auc_flagged": True}
        triggered, reason = should_retrain(drift, perf)
        assert triggered is True
        assert "AUC" in reason

    def test_retrain_when_both_flagged(self):
        """Both drift and AUC drop should appear in reason string."""
        from monitoring.retrain_trigger import should_retrain
        drift = {"overall_drift_score": 0.47, "drift_detected": True}
        perf = {"auc": 0.49, "auc_flagged": True}
        triggered, reason = should_retrain(drift, perf)
        assert triggered is True
        assert "drift" in reason.lower()
        assert "AUC" in reason


class TestReadDriftSummary:
    """Tests for read_drift_summary()."""

    def test_loads_correctly(self, drift_summary_json):
        """Should load drift summary JSON without errors."""
        from monitoring.retrain_trigger import read_drift_summary
        result = read_drift_summary(drift_summary_json)
        assert "overall_drift_score" in result
        assert "drift_detected" in result

    def test_missing_file_raises_error(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        from monitoring.retrain_trigger import read_drift_summary
        with pytest.raises(FileNotFoundError):
            read_drift_summary(tmp_path / "nonexistent.json")


class TestReadPerformanceReport:
    """Tests for read_performance_report()."""

    def test_loads_correctly(self, performance_report_json):
        """Should load performance report JSON without errors."""
        from monitoring.retrain_trigger import read_performance_report
        result = read_performance_report(performance_report_json)
        assert "auc" in result
        assert "auc_flagged" in result

    def test_missing_file_raises_error(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        from monitoring.retrain_trigger import read_performance_report
        with pytest.raises(FileNotFoundError):
            read_performance_report(tmp_path / "nonexistent.json")


class TestSaveRetrainLog:
    """Tests for save_retrain_log()."""

    def test_creates_log_file(self, tmp_path):
        """Retrain log should be created as valid JSON."""
        from monitoring.retrain_trigger import save_retrain_log
        path = tmp_path / "retrain_log.json"
        entry = save_retrain_log(
            triggered=True,
            reason="Test reason",
            dag_triggered=False,
            drift_summary={"overall_drift_score": 0.47, "drift_detected": True,
                           "drifted_features": ["retry_count"]},
            performance_report={"auc": 0.49, "auc_flagged": True},
            log_path=path,
        )
        assert path.exists()
        assert entry["retrain_triggered"] is True

    def test_appends_multiple_entries(self, tmp_path):
        """Multiple calls should append entries to the log."""
        from monitoring.retrain_trigger import save_retrain_log
        path = tmp_path / "retrain_log.json"
        for i in range(3):
            save_retrain_log(
                triggered=False,
                reason="No retraining needed",
                dag_triggered=False,
                drift_summary={"overall_drift_score": 0.05, "drift_detected": False,
                               "drifted_features": []},
                performance_report={"auc": 0.99, "auc_flagged": False},
                log_path=path,
            )
        with open(path) as f:
            log = json.load(f)
        assert len(log) == 3