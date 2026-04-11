#!/bin/bash
# deploy/deploy_cloudrun.sh
# One-click deploy script for Pipeline Autopilot to GCP Cloud Run
# Usage: bash deploy/deploy_cloudrun.sh

set -e

PROJECT_ID="datapipeline-autopilot"
REGION="us-central1"
SERVICE_NAME="pipeline-autopilot"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "=============================================="
echo " PipelineGuard — Cloud Run Deploy"
echo " Project : $PROJECT_ID"
echo " Region  : $REGION"
echo "=============================================="

# 1. Check model artifacts
echo ""
echo "▶ [1/5] Checking model artifacts..."
for f in "models/trained/best_model.joblib" \
          "models/trained/scaler.joblib" \
          "models/trained/model_metadata.json" \
          "models/trained/feature_names.json"; do
  if [ ! -f "$f" ]; then
    echo "❌ Missing: $f — run python scripts/model_training.py first"
    exit 1
  fi
done
echo "   ✅ All model artifacts present."

# 2. GCP auth
echo ""
echo "▶ [2/5] Checking GCP authentication..."
gcloud config set project $PROJECT_ID
gcloud auth configure-docker --quiet
echo "   ✅ GCP auth OK."

# 3. Build
echo ""
echo "▶ [3/5] Building Docker image..."
TAG=$(date +%Y%m%d-%H%M%S)
docker build -t "$IMAGE:$TAG" -t "$IMAGE:latest" -f Dockerfile.app .
echo "   ✅ Image built: $IMAGE:$TAG"

# 4. Push
echo ""
echo "▶ [4/5] Pushing to GCR..."
docker push "$IMAGE:$TAG"
docker push "$IMAGE:latest"
echo "   ✅ Image pushed."

# 5. Deploy
echo ""
echo "▶ [5/5] Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image="$IMAGE:$TAG" \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=60 \
  --quiet

URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format='value(status.url)')

echo ""
echo "=============================================="
echo " ✅ DEPLOYMENT COMPLETE"
echo " Live URL : $URL"
echo " Health   : $URL/health"
echo " Predict  : $URL/predict"
echo " Metrics  : $URL/metrics"
echo "=============================================="

# Smoke test
sleep 5
curl -s "$URL/health" | python3 -m json.tool