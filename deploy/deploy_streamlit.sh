#!/bin/bash
# deploy/deploy_streamlit.sh
# Deploys the Streamlit frontend to GCP Cloud Run
# Usage: bash deploy/deploy_streamlit.sh

set -e

PROJECT_ID="datapipeline-autopilot"
REGION="us-central1"
SERVICE_NAME="pipeline-autopilot-frontend"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"
BACKEND_URL="https://pipeline-autopilot-60271318606.us-central1.run.app"

echo "=============================================="
echo " PipelineGuard — Streamlit Frontend Deploy"
echo " Project : $PROJECT_ID"
echo " Region  : $REGION"
echo "=============================================="

# 1. Check frontend file exists
echo ""
echo "▶ [1/5] Checking frontend file..."
if [ ! -f "frontend/streamlit_app.py" ]; then
  echo "❌ Missing: frontend/streamlit_app.py"
  exit 1
fi
echo "   ✅ frontend/streamlit_app.py found."

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
docker build -t "$IMAGE:$TAG" -t "$IMAGE:latest" -f Dockerfile.streamlit .
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
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=60 \
  --set-env-vars="CLOUD_RUN_URL=$BACKEND_URL" \
  --quiet

URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format='value(status.url)')

echo ""
echo "=============================================="
echo " ✅ STREAMLIT DEPLOYMENT COMPLETE"
echo " Live URL : $URL"
echo "=============================================="

# Smoke test
sleep 5
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" "$URL"