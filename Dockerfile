FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_FORMAT=json

WORKDIR /app

# System deps (psycopg2 + catboost + feasting build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — split into a "requirements" layer for caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY src/        src/
COPY pipelines/  pipelines/
COPY api/        api/
COPY configs/    configs/
COPY models/     models/
COPY feast/feature_repo/feature_store.yaml feast/feature_repo/feature_store.yaml
COPY params.yaml .
COPY dvc.yaml    dvc.yaml
COPY .env        .env

# Expose API + Prometheus port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/readyz | grep -q '"ready":true' || exit 1

# Default: production-style JSON logs, 1 uvicorn worker (swap to gunicorn for prod)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
