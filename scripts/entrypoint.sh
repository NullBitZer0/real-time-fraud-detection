#!/bin/bash
set -e

# Download model from MLflow/Dagshub if not present
python -c "from scripts.download_model import download_production_model; download_production_model()"

# Start API
exec uvicorn api.app:app --host 0.0.0.0 --port 8888 --workers 1
