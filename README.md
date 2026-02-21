# Pipeline Autopilot 🚀

**AI-Powered CI/CD Pipeline Failure Prediction System**

Predict pipeline failures before they happen using machine learning, with RAG-powered explanations for root cause analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Team Contributions](#team-contributions)
- [Timeline](#timeline)

---

## 🎯 Overview

Pipeline Autopilot is an MLOps system that predicts CI/CD pipeline failures using XGBoost and provides intelligent explanations through RAG (Retrieval-Augmented Generation). The system analyzes pipeline metadata, historical patterns, and error messages to warn engineers before executing pipelines that are likely to fail.

**Problem Solved:**
- Data pipelines fail unexpectedly → engineers waste hours debugging
- Netflix Maestro faces similar issues: stuck workflows, late-arriving data, wasted compute
- **Our solution:** Don't run a pipeline if it's likely to fail. Warn first, suggest fixes.

---

## ✨ Features

- **Failure Prediction**: XGBoost model predicts pipeline failures with 27 engineered features
- **RAG Explanations**: Vertex AI Gemini provides root cause analysis and fix suggestions
- **Real-time Monitoring**: Detects anomalies, schema violations, and data quality issues
- **Comprehensive Testing**: 70+ unit tests ensuring code quality
- **Automated Orchestration**: Cloud Composer (Apache Airflow) for pipeline automation
- **Interactive Dashboard**: Streamlit interface for predictions and insights

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  GitHub Actions API  →  100K real pipeline runs (50 repos)      │
│  Simulated Data      →  50K augmented runs (on-prem pipelines)  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DATA PROCESSING LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  • Data Preprocessing  (handle nulls, duplicates, constraints)  │
│  • Schema Validation   (detect drift, violations)               │
│  • Anomaly Detection   (outliers, range violations)             │
│  • Feature Engineering (27 features: temporal, perf, risk)      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ML PREDICTION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  • XGBoost Model       (binary classification: fail/pass)       │
│  • MLflow Tracking     (experiments, hyperparameters, metrics)  │
│  • Model Registry      (versioning, deployment)                 │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RAG EXPLANATION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  • Vector Search       (index error messages, failure patterns) │
│  • Vertex AI Gemini    (generate explanations + fixes)          │
│  • Context Retrieval   (pull relevant historical failures)      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  • Cloud Run          (Streamlit dashboard)                     │
│  • Cloud Composer     (Airflow DAGs for orchestration)          │
│  • BigQuery           (data warehouse)                          │
│  • Cloud Storage      (model artifacts, logs)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Total: 150,000 pipeline runs**

### Real Data (100K rows)
- **Source**: GitHub Actions API (parallel scraping, 15 threads)
- **Repositories**: 50 open-source projects
  - Apache Airflow, Spark, Kubernetes, TensorFlow, dbt, Kafka, etc.
- **Features**: Real failures, error messages, execution patterns

### Augmented Data (50K rows)
- **Source**: 5 on-premises Jupyter pipelines
- **Method**: Generated based on real data distributions
- **Purpose**: Expand training set with diverse failure scenarios

### Feature Set (27 features)

| Category | Features |
|----------|----------|
| **Temporal** | trigger_time, day_of_week, hour, is_weekend |
| **Performance** | duration_seconds, avg_duration_7_runs, duration_deviation |
| **Historical** | prev_run_status, failures_last_7_runs, workflow_failure_rate, hours_since_last_run |
| **Complexity** | total_jobs, failed_jobs, retry_count, concurrent_runs |
| **Risk** | head_branch, is_main_branch, is_first_run, is_bot_triggered, trigger_type |
| **Context** | failure_type, error_message (for RAG) |
| **Target** | failed (0 or 1) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | XGBoost, LightGBM, scikit-learn |
| **Experiment Tracking** | MLflow |
| **RAG** | Vertex AI Gemini, Vector Search |
| **Orchestration** | Cloud Composer (Apache Airflow) |
| **Data Storage** | BigQuery, Cloud Storage |
| **Dashboard** | Streamlit (Cloud Run) |
| **Testing** | pytest (70+ tests) |
| **Version Control** | Git, GitHub |
| **Language** | Python 3.13 |

---

## 📁 Project Structure
```
pipeline-autopilot/
├── dags/                          # Airflow DAGs
│   └── pipeline_orchestration.py
├── data/                          # Raw and processed data
│   ├── final_dataset.csv
│   └── final_dataset_processed.csv
├── logs/                          # Application logs
│   └── pipeline.log
├── scripts/                       # Core pipeline scripts
│   ├── config.py                  # Configuration
│   ├── data_preprocessing.py      # Data cleaning & feature engineering
│   ├── schema_validation.py       # Schema drift detection
│   ├── anomaly_detection.py       # Anomaly & outlier detection
│   └── logging_config.py          # Shared logging setup
├── tests/                         # Unit tests (70+ tests)
│   ├── conftest.py                # Shared fixtures
│   ├── test_data_preprocessing.py # 20 tests
│   ├── test_schema_validation.py  # 20 tests
│   ├── test_anomaly_detection.py  # 22 tests
│   └── test_logging_config.py     # 8 tests
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.13+
- Google Cloud Platform account (for deployment)
- Git

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/anita2210/pipeline-autopilot.git
   cd pipeline-autopilot
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up directories**
```bash
   mkdir -p logs data reports models
```

---

## 🚀 Usage

### Run Data Preprocessing
```bash
python scripts/data_preprocessing.py
```

**Output:**
- Processed dataset: `data/final_dataset_processed.csv`
- Summary report: `preprocessing_summary.json`

### Run Schema Validation
```bash
python scripts/schema_validation.py
```

**Output:**
- Schema definition: `schema/schema.json`
- Validation report: `reports/validation_report.json`

### Run Anomaly Detection
```bash
python scripts/anomaly_detection.py
```

**Output:**
- Anomaly report: `reports/anomaly_report.json`
- Alerts sent (if configured)

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
# Data preprocessing tests (20 tests)
pytest tests/test_data_preprocessing.py -v

# Schema validation tests (20 tests)
pytest tests/test_schema_validation.py -v

# Anomaly detection tests (22 tests)
pytest tests/test_anomaly_detection.py -v

# Logging tests (8 tests)
pytest tests/test_logging_config.py -v
```

### Test Coverage
```bash
pytest tests/ --cov=scripts --cov-report=html
```

**Current Test Coverage: 70 tests, 100% passing** ✅

---

## 👥 Team Contributions

| Member | Role | Responsibilities |
|--------|------|------------------|
| **Member 1** | Data Engineer | Data scraping, augmentation, BigQuery integration |
| **Member 2** | ML Engineer | Model training, hyperparameter tuning, MLflow tracking |
| **Member 3** | Feature Engineer | Feature engineering, preprocessing pipeline |
| **Member 4** | Data Quality Engineer | Schema validation, statistics generation |
| **Member 5** | Fairness Analyst | Bias detection, mitigation strategies |
| **Member 6** | QA & DevOps | **Testing (70+ tests), logging setup, documentation** |

---

## 📅 Timeline

| Milestone | Date | Status |
|-----------|------|--------|
| Project Kickoff | Feb 10, 2026 | ✅ Complete |
| Data Collection | Feb 15, 2026 | ✅ Complete |
| **Data Pipeline Submission** | **Feb 26, 2026** | 🚧 In Progress |
| Model Training | Mar 5, 2026 | ⏳ Upcoming |
| RAG Integration | Mar 12, 2026 | ⏳ Upcoming |
| Dashboard Deployment | Mar 19, 2026 | ⏳ Upcoming |
| Final Presentation | Mar 26, 2026 | ⏳ Upcoming |

---

## 📄 License

This project is part of the MLOps course at Northeastern University (Spring 2026).

---

## 🙏 Acknowledgments

- **Datasets**: GitHub Actions API, Open-source repositories
- **Inspiration**: Netflix Maestro, Airflow Best Practices
- **Course**: Northeastern University MLOps (INFO 7390)

---

**Built with ❤️ by the Pipeline Autopilot Team**