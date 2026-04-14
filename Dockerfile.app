FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    scikit-learn xgboost shap \
    pandas numpy joblib pydantic \
    faiss-cpu==1.7.4 \
    langchain langchain-community \
    google-genai

# Copy knowledge base
COPY knowledge_base/ knowledge_base/

# Copy model artifacts
COPY models/trained/best_model.joblib   models/trained/best_model.joblib
COPY models/trained/scaler.joblib       models/trained/scaler.joblib
COPY models/trained/model_metadata.json models/trained/model_metadata.json
COPY models/trained/feature_names.json  models/trained/feature_names.json

# Copy app
COPY app/ app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]