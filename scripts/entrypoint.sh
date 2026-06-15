#!/bin/bash

# Download model from MLflow/Dagshub if not present
python -c "from scripts.download_model import download_production_model; download_production_model()" || echo "Model download failed, continuing..."

# Pull test data if not present
if [ ! -f "data/raw/fraudTest.csv" ]; then
  echo "Downloading fraudTest.csv..."
  python -c "from scripts.download_model import download_test_data; download_test_data()" || echo "Data download failed, demo endpoint unavailable"
fi

# Start API
exec uvicorn api.app:app --host 0.0.0.0 --port 8888 --workers 1
