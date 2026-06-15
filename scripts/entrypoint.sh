#!/bin/bash
set -e

# Download model from MLflow/Dagshub if not present
python -c "from scripts.download_model import download_production_model; download_production_model()"

# Pull test data from DVC if not present
if [ ! -f "data/raw/fraudTest.csv" ]; then
  echo "Pulling fraudTest.csv from DVC..."
  mkdir -p data/raw
  dvc pull data/raw/fraudTest.csv
fi

# Start API
exec uvicorn api.app:app --host 0.0.0.0 --port 8888 --workers 1
