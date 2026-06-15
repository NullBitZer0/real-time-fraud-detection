#!/bin/bash

# Download model from MLflow/Dagshub if not present
python -c "from scripts.download_model import download_production_model; download_production_model()" || echo "Model download failed, continuing..."

# Pull test data from DVC if not present
if [ ! -f "data/raw/fraudTest.csv" ]; then
  echo "Pulling fraudTest.csv from DVC..."
  mkdir -p data/raw
  dvc pull data/raw/fraudTest.csv || echo "DVC pull failed, demo endpoint unavailable"
fi

# Start API
exec uvicorn api.app:app --host 0.0.0.0 --port 8888 --workers 1
