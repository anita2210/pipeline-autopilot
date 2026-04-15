# Pipeline Autopilot — Kairos Pulse

**MLOps CI/CD Pipeline Failure Prediction System**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Airflow 2.8.1](https://img.shields.io/badge/airflow-2.8.1-green.svg)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-orange.svg)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-model-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-bias%20detection-purple.svg)](https://fairlearn.org/)
[![Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4.svg)](https://cloud.google.com/run)
[![Gemini](https://img.shields.io/badge/Gemini-RAG-blueviolet.svg)](https://deepmind.google/technologies/gemini/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Live Deployment](#live-deployment)
3. [How to Replicate](#how-to-replicate-step-by-step-setup)
4. [How to Run the Data Pipeline](#how-to-run-the-data-pipeline)
5. [How to Run the Model Pipeline](#how-to-run-the-model-pipeline)
6. [How to Run Tests](#how-to-run-tests)
7. [Project Structure](#project-structure)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Dataset Information](#dataset-information)
10. [Model Development](#model-development)
11. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
12. [Model Validation](#model-validation)
13. [Model Bias Detection (Fairlearn)](#model-bias-detection-fairlearn)
14. [Sensitivity Analysis (SHAP)](#sensitivity-analysis-shap)
15. [FastAPI Backend](#fastapi-backend)
16. [RAG Chatbot (Gemini)](#rag-chatbot-gemini)
17. [Kairos Pulse — Streamlit Dashboard](#kairos-pulse--streamlit-dashboard)
18. [Live GitHub Actions Streaming](#live-github-actions-streaming)
19. [Email Alert System](#email-alert-system)
20. [CI/CD Pipeline Automation](#cicd-pipeline-automation)
21. [Model Registry & Deployment](#model-registry--deployment)
22. [Data Versioning with DVC](#data-versioning-with-dvc)
23. [Team Members](#team-members)

---

## Project Overview

**Pipeline Autopilot** is an end-to-end MLOps system that predicts CI/CD pipeline failures before they happen using Machine Learning, and explains root causes using RAG (Retrieval-Augmented Generation). The system is deployed on Google Cloud Platform and branded as **Kairos Pulse** for the Google Cambridge Showcase — April 15, 2026.

### Problem Statement

- Data pipelines fail unexpectedly, causing engineers to waste hours debugging
- Manual monitoring is inefficient and entirely reactive
- No proactive failure prevention mechanism exists in standard CI/CD tools

### Solution

- **Predict** pipeline failures before execution using an XGBoost model trained on 150K real GitHub Actions runs
- **Score** every incoming run in under 200ms via a Cloud Run API
- **Explain** root causes using a Gemini-powered RAG chatbot with FAISS vector search
- **Alert** the team automatically via Gmail when a HIGH risk run is detected
- **Block** high-risk runs and provide actionable fix recommendations

### Key Features

- Automated data acquisition and preprocessing
- Schema validation and statistics generation
- Anomaly detection with alerts
- Data-level and model-level bias detection with Fairlearn
- Data versioning with DVC
- Full pipeline orchestration with Apache Airflow
- ML model training with XGBoost and hyperparameter tuning
- Experiment tracking with MLflow
- Model validation with threshold analysis and rollback mechanism
- Sensitivity analysis with SHAP
- CI/CD automation with GitHub Actions
- Model registry push to GCP Artifact Registry
- FastAPI REST backend deployed on Cloud Run
- Gemini RAG chatbot for root cause diagnosis
- Streamlit dashboard (Kairos Pulse) deployed on Cloud Run
- Live GitHub Actions streaming with real-time scoring
- Automated email alerts for high-risk pipeline runs

---

## Live Deployment

Both services are deployed on GCP Cloud Run and publicly accessible.

| Service | URL |
|---------|-----|
| **Kairos Pulse — Streamlit Dashboard** | https://pipeline-autopilot-frontend-60271318606.us-central1.run.app |
| **FastAPI Prediction Backend** | https://pipeline-autopilot-60271318606.us-central1.run.app |

### Quick Health Check

```bash
curl https://pipeline-autopilot-60271318606.us-central1.run.app/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_name": "XGBoost Tuned",
  "auc_roc": 0.9808
}
```

---

## How to Replicate (Step-by-Step Setup)

### Prerequisites

| Software | Version | Download Link |
|----------|---------|---------------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Docker Desktop | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Git | Latest | [git-scm.com](https://git-scm.com/downloads) |

### Step 1: Clone the Repository

```bash
git clone https://github.com/anita2210/pipeline-autopilot.git
cd pipeline-autopilot
```

### Step 2: Create Environment File

```bash
# Windows PowerShell
echo "AIRFLOW_UID=50000" > .env

# Mac/Linux
echo "AIRFLOW_UID=$(id -u)" > .env
```

### Step 3: Install Python Dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 4: Verify Setup

```bash
python scripts/config.py
```

Expected output:

```
============================================================
PIPELINE AUTOPILOT CONFIGURATION
============================================================
All directories verified/created!
Raw dataset found: .../data/raw/final_dataset.csv
```

---

## How to Run the Data Pipeline

### Step 1: Start Docker Desktop

Ensure Docker Desktop is running and the Engine shows as healthy.

### Step 2: Start Airflow

```bash
docker-compose up -d
```

Wait 2–3 minutes for all containers to initialise.

### Step 3: Verify Containers

```bash
docker-compose ps
```

All four services should show `Up (healthy)`:

```
pipeline_autopilot_postgres
pipeline_autopilot_scheduler
pipeline_autopilot_triggerer
pipeline_autopilot_webserver
```

### Step 4: Access Airflow Web UI

Open `http://localhost:8080` in your browser.

- **Username:** `admin`
- **Password:** `admin`

### Step 5: Run the Data DAG

1. Locate DAG: `pipeline_autopilot_data_pipeline`
2. Enable the DAG using the toggle switch
3. Click **Trigger DAG**
4. Navigate to the **Graph** tab to monitor execution

### Step 6: Monitor Execution

All seven tasks should complete with a green status:

```
data_acquisition
data_preprocessing
schema_validation    (parallel)
bias_detection       (parallel)
anomaly_detection
dvc_versioning
pipeline_complete
```

### Step 7: Stop Airflow

```bash
docker-compose down
```

---

## How to Run the Model Pipeline

### Option 1: Via Airflow (Model DAG)

1. Start Airflow using the steps above
2. Locate DAG: `pipeline_autopilot_model_pipeline`
3. Enable and trigger the DAG
4. Monitor all nine tasks:

```
load_processed_data
train_models
select_best_model
validate_model
model_bias_detection    (parallel)
sensitivity_analysis    (parallel)
validation_gate
push_to_registry
model_pipeline_complete
```

### Option 2: Run Scripts Individually

```bash
python scripts/model_training.py
python scripts/experiment_tracking.py
python scripts/model_validation.py
python scripts/model_bias_detection.py
python scripts/model_sensitivity.py
python scripts/model_registry.py
```

### Option 3: Via Docker

```bash
docker build -f Dockerfile.model -t pipeline-autopilot-model .
docker run pipeline-autopilot-model
```

### View MLflow Results

```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

---

## How to Run Tests

```bash
# All tests
pytest tests/ -v

# Data pipeline tests only
pytest tests/test_data_preprocessing.py tests/test_schema_validation.py tests/test_anomaly_detection.py tests/test_logging_config.py -v

# Model pipeline tests only
pytest tests/test_model_training.py tests/test_model_validation.py tests/test_model_bias.py -v

# With coverage report
pytest tests/ -v --cov=scripts --cov-report=html
```

---

## Project Structure

```
pipeline-autopilot/
│
├── app/
│   ├── main.py                      # FastAPI backend (4 endpoints)
│   ├── rag_chatbot.py               # Gemini RAG diagnosis engine
│   ├── live_poller.py               # GitHub Actions live poller
│   ├── alert_system.py              # Email alert dispatcher
│   └── __init__.py
│
├── frontend/
│   └── streamlit_app.py             # Kairos Pulse dashboard (5 pages)
│
├── dags/
│   ├── pipeline_dag.py              # Data pipeline Airflow DAG (7 tasks)
│   └── model_dag.py                 # Model pipeline Airflow DAG (9 tasks)
│
├── scripts/
│   ├── config.py
│   ├── data_acquisition.py
│   ├── data_preprocessing.py
│   ├── schema_validation.py
│   ├── anomaly_detection.py
│   ├── bias_detection.py
│   ├── dvc_versioning.py
│   ├── logging_config.py
│   ├── model_training.py
│   ├── experiment_tracking.py
│   ├── model_validation.py
│   ├── model_bias_detection.py
│   ├── model_sensitivity.py
│   └── model_registry.py
│
├── models/
│   ├── trained/
│   │   ├── best_model.joblib
│   │   ├── scaler.joblib
│   │   ├── model_metadata.json
│   │   └── feature_names.json
│   ├── registry/
│   │   ├── registry_manifest.json
│   │   └── model_bias_report.json
│   └── sensitivity/
│       ├── shap_summary.png
│       ├── auc_comparison.png
│       └── feature_importance_comparison.png
│
├── knowledge_base/
│   ├── daily_stats.json
│   ├── error_stats.json
│   ├── global_stats.json
│   ├── repo_stats.json
│   └── similar_runs_index.pkl       # FAISS vector index
│
├── monitoring/
│   ├── drift_detection.py
│   ├── performance_monitor.py
│   └── retrain_trigger.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── schema/
│   └── reports/
│
├── tests/
│   ├── conftest.py
│   ├── test_data_preprocessing.py
│   ├── test_schema_validation.py
│   ├── test_anomaly_detection.py
│   ├── test_logging_config.py
│   ├── test_model_training.py
│   ├── test_model_validation.py
│   └── test_model_bias.py
│
├── deploy/
│   ├── deploy_cloudrun.sh           # FastAPI deployment script
│   └── deploy_streamlit.sh          # Streamlit deployment script
│
├── .github/
│   └── workflows/
│       ├── ml_pipeline.yml
│       └── deploy.yml               # Auto-deploy on push to main
│
├── Dockerfile.app                   # FastAPI container
├── Dockerfile.streamlit             # Streamlit container
├── Dockerfile.model                 # Model training container
├── docker-compose.yaml
├── cloudbuild.yaml
├── requirements.txt
├── dvc.yaml
└── README.md
```

---

## Pipeline Architecture

### Data Pipeline DAG

```
data_acquisition
       |
data_preprocessing
       |
  _____|_____
 |           |
schema     bias          <- parallel
validation detection
 |___________|
       |
anomaly_detection
       |
dvc_versioning
       |
pipeline_complete
```

### Model Pipeline DAG

```
load_processed_data
       |
train_models  (Logistic Regression, Random Forest, XGBoost + tuning)
       |
select_best_model  (by AUC-ROC)
       |
validate_model  (hold-out + threshold analysis)
       |
  _____|_____
 |           |
model_bias  sensitivity   <- parallel
detection   analysis
 |___________|
       |
validation_gate  (AUC > 0.85, no bias flags)
       |
push_to_registry  (GCP Artifact Registry)
       |
model_pipeline_complete
```

### Pipeline Execution Screenshots

#### Pipeline Status and Task History

![Pipeline Status](images/pipeline_status.png)

#### Graph View — DAG Structure

![Pipeline Graph](images/pipeline_graph.png)

#### Gantt Chart — Execution Timeline

![Pipeline Gantt](images/pipeline_gantt.png)

---

## Dataset Information

| Property | Value |
|----------|-------|
| Raw file | `data/raw/final_dataset.csv` |
| Processed file | `data/processed/final_dataset_processed.csv` |
| Total rows | 149,967 |
| Total columns | 32 |
| Target variable | `failed` (binary: 0 / 1) |
| Failure rate | ~11.33% |

### Data Sources

- **100,000 rows** — real GitHub Actions runs scraped from 50 open-source repositories (Airflow, Spark, Kubernetes, TensorFlow, dbt, Kafka, and others) using parallel processing with 15 threads
- **50,000 rows** — augmented data generated from on-premises Jupyter pipelines, following the same distribution and feature relationships as the real data

### Column Categories

| Category | Columns |
|----------|---------|
| ID | run_id |
| Datetime | trigger_time |
| Temporal | day_of_week, hour, is_weekend |
| Performance | duration_seconds, avg_duration_7_runs, duration_deviation |
| Historical | prev_run_status, failures_last_7_runs, workflow_failure_rate, hours_since_last_run |
| Complexity | total_jobs, failed_jobs, retry_count, concurrent_runs |
| Risk | head_branch, is_main_branch, is_first_run, is_bot_triggered, trigger_type |
| Categorical | pipeline_name, repo, failure_type, error_message |
| Target | failed |

### Preprocessing Applied

| Step | Description |
|------|-------------|
| Missing values | Median for numerical, mode for categorical |
| Duplicates | 33 removed based on `run_id` |
| Datetime parsing | `trigger_time` converted to datetime |
| Categorical encoding | Frequency encoding for high-cardinality, label encoding for low-cardinality |
| Outlier capping | IQR method (1.5x multiplier) |
| Constraint validation | `failed_jobs <= total_jobs`, `workflow_failure_rate` in range 0–1 |

---

## Model Development

### Data Split

| Set | Split | Purpose |
|-----|-------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final hold-out evaluation |

All splits are stratified on `failed` to preserve the ~11.33% failure rate.

### Models Trained

| Model | Class Imbalance Handling |
|-------|--------------------------|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost Default | `scale_pos_weight` (~7.8x) |
| **XGBoost Tuned** | `scale_pos_weight` + RandomizedSearchCV |
| MLP Neural Network | Class weighting |

### Best Model — XGBoost Tuned

| Metric | Score |
|--------|-------|
| Train Accuracy | 96.46% |
| Test Accuracy | 93.39% |
| AUC-ROC | 98.08% |
| AUC-PR | 87.12% |
| Precision | 65.21% |
| Recall | 89.37% |
| F1 Score | 75.40% |

The decision threshold is set at **0.75**, optimised for the precision-recall balance required in a CI/CD environment where false positives (blocking a safe run) carry a meaningful cost.

**Script:** `scripts/model_training.py`

---

## Experiment Tracking with MLflow

All training runs are tracked under the experiment name `pipelineguard-model-dev`.

### What is Logged

| Category | Details |
|----------|---------|
| Parameters | All hyperparameters per model |
| Metrics | AUC-ROC, F1, Precision, Recall, Accuracy |
| Artifacts | Confusion matrices, comparison bar plots |
| Model versions | Best model registered in MLflow Model Registry (Staging → Production) |

```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

**Script:** `scripts/experiment_tracking.py`

---

## Model Validation

### Hold-Out Evaluation

The best model is evaluated on the 15% test set that was never used during training or tuning.

### Threshold Analysis

Decision thresholds are varied from 0.1 to 0.9 to identify the optimal threshold by F1-score.

### Validation Gate

The model must pass all of the following checks before it is allowed into the registry:

- AUC-ROC > 0.85
- No critical bias flags from Fairlearn
- Performance equal to or better than the previous production model

### Rollback Mechanism

If the newly trained model performs worse than the previously registered model (compared by AUC-ROC), the system rejects the new model and retains `previous_model.joblib` as the active production model.

**Script:** `scripts/model_validation.py`

---

## Model Bias Detection (Fairlearn)

### Data-Level Bias

Data slicing is used to analyse failure rate distributions across subgroups:

| Feature | Slices |
|---------|--------|
| `repo` | 50 repositories |
| `pipeline_name` | All pipeline types |
| `trigger_type` | push, pull_request, schedule, workflow_dispatch |
| `is_weekend` | Weekend vs Weekday |
| `is_bot_triggered` | Bot vs Human |

### Model-Level Bias

Fairlearn `MetricFrame` evaluates whether the trained model predicts fairly across subgroups. Disparity metrics tracked:

- Demographic Parity Difference
- Equalized Odds Difference

If disparity exceeds 1.5x between any two groups, `ThresholdOptimizer` from Fairlearn adjusts decision thresholds per group.

Bias report is saved to: `models/registry/model_bias_report.json`

**Script:** `scripts/model_bias_detection.py`

---

## Sensitivity Analysis (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain which features drive each prediction.

### Outputs

| File | Description |
|------|-------------|
| `models/sensitivity/shap_summary.png` | Beeswarm plot — feature impact distribution |
| `models/sensitivity/shap_bar.png` | Mean absolute SHAP values |
| `models/sensitivity/feature_importance_comparison.png` | SHAP vs XGBoost built-in importance |
| `models/sensitivity/auc_comparison.png` | AUC across all trained models |
| `models/sensitivity/hyperparameter_sensitivity/` | AUC vs each hyperparameter value |

**Script:** `scripts/model_sensitivity.py`

---

## FastAPI Backend

The prediction backend is built with FastAPI and deployed on GCP Cloud Run.

**Base URL:** `https://pipeline-autopilot-60271318606.us-central1.run.app`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns model status, name, version, and AUC-ROC |
| `POST` | `/predict` | Accepts 21 pipeline features, returns failure probability, risk level, and top SHAP features |
| `POST` | `/explain` | Accepts run features and a user message, returns a Gemini RAG diagnosis |
| `GET` | `/metrics` | Returns a summary of the last 100 predictions |

### Example — /predict

```bash
curl -X POST https://pipeline-autopilot-60271318606.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"workflow_failure_rate": 0.62, "failures_last_7_runs": 4, "prev_run_status": 1, "retry_count": 2}'
```

Response:

```json
{
  "probability": 0.847,
  "prediction": 1,
  "risk_level": "HIGH",
  "top_shap_features": [
    {"feature": "workflow_failure_rate", "shap_value": 0.312, "direction": "increases_risk"},
    {"feature": "failures_last_7_runs",  "shap_value": 0.198, "direction": "increases_risk"}
  ]
}
```

### Risk Thresholds

| Risk Level | Probability |
|------------|-------------|
| HIGH | >= 0.75 |
| MEDIUM | >= 0.40 |
| LOW | < 0.40 |

**File:** `app/main.py`

### Deployment

```bash
bash deploy/deploy_cloudrun.sh
```

---

## RAG Chatbot (Gemini)

The `/explain` endpoint powers an AI chatbot that diagnoses why a pipeline is at risk and recommends specific remediation steps.

### Architecture

```
User question
      |
POST /explain  (FastAPI)
      |
FAISS vector search  (knowledge_base/similar_runs_index.pkl)
      |
Retrieve similar past failure records
      |
Gemini generates diagnosis using retrieved context
      |
Response returned to Streamlit UI
```

### LLM Type

The system uses **Retrieval-Augmented Generation (RAG)**. Gemini does not answer purely from its training data — it first retrieves relevant failure history from the knowledge base, then generates a contextualised diagnosis grounded in that data.

### Knowledge Base Files

| File | Contents |
|------|----------|
| `knowledge_base/similar_runs_index.pkl` | FAISS vector index of historical runs |
| `knowledge_base/global_stats.json` | Aggregate failure statistics |
| `knowledge_base/daily_stats.json` | Failure patterns by day and hour |
| `knowledge_base/error_stats.json` | Failure type distributions |
| `knowledge_base/repo_stats.json` | Per-repository failure profiles |

**File:** `app/rag_chatbot.py`

---

## Kairos Pulse — Streamlit Dashboard

The frontend is a single-file Streamlit application branded as **Kairos Pulse** and deployed on GCP Cloud Run.

**Live URL:** `https://pipeline-autopilot-frontend-60271318606.us-central1.run.app`

### Pages

| Page | Description |
|------|-------------|
| Overview | Landing page with animated pipeline canvas, feature cards, and product description |
| Pipeline Monitor | Live scoring dashboard — streams GitHub Actions runs, shows risk scores, surfaces HIGH risk runs with fix recommendations |
| Root Cause Analysis | Gemini RAG chatbot — SHAP gauge, feature impact bar chart, and conversational diagnosis |
| Audit Log | Full scored-run log, risk breakdown pie chart, probability scatter chart, compute-saved metrics |
| Incidents | Active and resolved HIGH risk alerts with top offending pipelines |

### Run Locally

```bash
python -m streamlit run frontend/streamlit_app.py
```

### Redeploy to Cloud Run

```bash
# Step 1: Rebuild the image
docker build -t gcr.io/datapipeline-autopilot/pipeline-autopilot-frontend:latest -f Dockerfile.streamlit .

# Step 2: Push to GCR
docker push gcr.io/datapipeline-autopilot/pipeline-autopilot-frontend:latest

# Step 3: Deploy
gcloud run deploy pipeline-autopilot-frontend \
  --image=gcr.io/datapipeline-autopilot/pipeline-autopilot-frontend:latest \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --set-env-vars="CLOUD_RUN_URL=https://pipeline-autopilot-60271318606.us-central1.run.app" \
  --quiet
```

Or use the deployment script:

```bash
bash deploy/deploy_streamlit.sh
```

**File:** `frontend/streamlit_app.py`

---

## Live GitHub Actions Streaming

The Pipeline Monitor page can stream real GitHub Actions workflow runs directly from the GitHub API and score them in real time via the Cloud Run backend.

### How It Works

1. User provides a GitHub Personal Access Token (PAT) with `repo` and `workflow` read scopes
2. The app fetches recent workflow runs from the configured repositories via the GitHub Actions API
3. Job-level details are retrieved to extract `failed_jobs`, `total_jobs`, and failure rate
4. Each run is sent to `/predict` on the Cloud Run backend
5. Results are displayed row by row as scoring completes

### Repositories Monitored

```python
REPOS = [
    "ClickHouse/ClickHouse",
    "ClickHouse/clickhouse-java",
    "ClickHouse/clickhouse-go",
    "ClickHouse/dbt-clickhouse",
]
```

These can be updated directly in `frontend/streamlit_app.py`.

### Fallback Behaviour

If no GitHub token is provided, the app falls back to a set of representative hardcoded demo runs and scores them through the same Cloud Run endpoint, demonstrating the full prediction pipeline without requiring API access.

---

## Email Alert System

When any scored pipeline run reaches a HIGH risk classification (probability >= 0.75), an automated HTML email alert is dispatched to the designated recipient.

### Alert Contents

- Run ID, pipeline name, repository, and branch
- Failure probability and risk classification
- Workflow failure rate, failures in last 7 runs, previous run status
- Ordered list of recommended fix steps generated from the pipeline's history

### Configuration

| Variable | Value |
|----------|-------|
| Sender | `pipelineguard.alerts@gmail.com` |
| Recipient | `varupandi@gmail.com` |
| Transport | Gmail SMTP — STARTTLS port 587, SSL port 465 fallback |

To change the recipient, update `ALERT_RECIPIENT` in `frontend/streamlit_app.py` or set it as an environment variable.

**File:** `frontend/streamlit_app.py` — `send_gmail_alert()`

---

## CI/CD Pipeline Automation

### GitHub Actions Workflows

Two workflows are defined under `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ml_pipeline.yml` | Push to `model-dev` branch | Train, validate, bias-check, and push model to registry |
| `deploy.yml` | Push to `main` branch | Build Docker image, push to GCR, deploy to Cloud Run |

### ML Pipeline Flow

```
Push to model-dev
       |
Install dependencies
       |
Run model training
       |
Run model validation
       |
Run bias detection
       |
Validation gate  (AUC > 0.85, no critical bias)
       |
  _____|_____
 |           |
PASS        FAIL
 |           |
Push to    Block + notify
registry
```

### Deploy Flow

```
Push to main
       |
Build Docker image  (Dockerfile.app)
       |
Push to Google Container Registry
       |
Deploy to Cloud Run  (us-central1)
       |
Get live URL
```

---

## Model Registry & Deployment

Once the model passes validation and bias checks, it is pushed to **GCP Artifact Registry** with a GCS bucket as fallback.

### Registry Metadata

Each version is tagged with:

- Version ID in format `v{date}-{model_hash}`
- AUC-ROC, F1, Precision, Recall metrics
- Bias check status
- Git commit hash

Registry manifest: `models/registry/registry_manifest.json`

**Script:** `scripts/model_registry.py`

---

## Data Versioning with DVC

```bash
# Track data files
dvc add data/raw/final_dataset.csv
dvc add data/processed/final_dataset_processed.csv

# Configure remote
dvc remote add -d gcs_remote gs://your-bucket-name

# Push data
dvc push

# Pull data on another machine
dvc pull

# View version differences
dvc diff
```

---

## Team Members

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Member 1 | Pipeline Architect / ML Engineer | Folder structure, config.py, Airflow DAGs, Docker setup, model training and selection |
| Member 2 | Data Engineer / MLOps Engineer | Data acquisition scripts, FastAPI backend, Cloud Run deployment, Kairos Pulse Streamlit dashboard |
| Member 3 | Data Scientist | Data preprocessing, feature engineering, model validation, threshold analysis |
| Member 4 | Quality Engineer / Fairness Analyst | Schema validation, anomaly detection, model-level bias detection with Fairlearn |
| Member 5 | MLOps Engineer | DVC versioning, data bias detection, SHAP sensitivity analysis, GCP registry push |
| Member 6 | Test Engineer / DevOps | Unit tests, logging configuration, CI/CD pipeline (GitHub Actions), Dockerfiles, model tests |

---

## Troubleshooting

**Docker containers not starting**

```bash
docker-compose down
docker-compose up -d
```

**Airflow UI not accessible**

Wait 2–3 minutes after starting containers. Verify all services show `healthy` with `docker-compose ps`. Restart the scheduler with `docker-compose restart airflow-scheduler`.

**DAG not visible in Airflow**

Check for syntax errors: `python dags/pipeline_dag.py`. Restart the scheduler after fixing any errors.

**Tests failing**

Ensure the virtual environment is activated and all dependencies are installed: `pip install -r requirements.txt`.

**MLflow UI not loading**

Ensure at least one training experiment has been run. Check that `mlruns/` contains experiment folders. Start with: `mlflow ui --host 0.0.0.0 --port 5000`.

**Model pipeline failing at validation gate**

Review the default AUC threshold (0.85) in `scripts/config.py`. Check the bias report at `models/registry/model_bias_report.json`. If a rollback was triggered, inspect `models/trained/previous_metrics.json`.

**Cloud Run returning 503**

The service scales to zero when idle. The first request after a cold start may take 10–15 seconds. Subsequent requests will be fast.

**Live stream showing no data**

Verify the GitHub PAT has `repo` and `workflow` read scopes. The app will automatically fall back to demo data if the token is missing or invalid.

---

## Links

- **GitHub Repository:** https://github.com/anita2210/pipeline-autopilot
- **Kairos Pulse Dashboard:** https://pipeline-autopilot-frontend-60271318606.us-central1.run.app
- **FastAPI Backend:** https://pipeline-autopilot-60271318606.us-central1.run.app
- **Airflow UI:** http://localhost:8080 (local only)
- **MLflow UI:** http://localhost:5000 (local only)

---

*Pipeline Autopilot — MLOps Course Project, Google Cambridge Showcase, April 2026*