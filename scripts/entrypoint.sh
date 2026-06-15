#!/bin/bash

# Download model from MLflow/Dagshub if not present
python -c "from scripts.download_model import download_production_model; download_production_model()" || echo "Model download failed, continuing..."

# Pull test data from DVC if not present
if [ ! -f "data/raw/fraudTest.csv" ] || [ ! -f "data/raw/fraudTrain.csv" ]; then
  echo "Pulling fraudTest.csv from DVC..."
  mkdir -p data/raw
  git init 2>/dev/null || true
  # DVC S3 credentials = DAGSHUB_TOKEN
  TOKEN="${DAGSHUB_TOKEN:-}"
  if [ -n "$TOKEN" ]; then
    dvc remote modify origin access_key_id "$TOKEN" 2>/dev/null || true
    dvc remote modify origin secret_access_key "$TOKEN" 2>/dev/null || true
    dvc remote modify origin endpointurl "https://dagshub.com/${DAGSHUB_REPO_OWNER:-NullBitZer0}/${DAGSHUB_REPO_NAME:-real-time-fraud-detection}.s3" 2>/dev/null || true
  fi
  dvc pull data/raw/fraudTest.csv data/raw/fraudTrain.csv || echo "DVC pull failed, demo endpoint unavailable"
fi

# Start API
exec uvicorn api.app:app --host 0.0.0.0 --port 8888 --workers 1
