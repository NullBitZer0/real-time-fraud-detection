FROM node:20-alpine AS frontend-build
WORKDIR /app
ARG VITE_API_URL=""
ARG VITE_WS_URL=""
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WS_URL=$VITE_WS_URL
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

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
COPY params.yaml params.yaml
COPY models/     models/
COPY scripts/    scripts/
COPY feast/feature_repo/feature_store.yaml feast/feature_repo/feature_store.yaml
COPY .dvc/config .dvc/config
COPY .dvc/.gitignore .dvc/.gitignore
COPY data/raw/fraudTest.csv.dvc data/raw/fraudTest.csv.dvc

# React frontend (built in first stage)
COPY --from=frontend-build /app/dist /app/static/dashboard

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8888/readyz | grep -q '"ready":true' || exit 1

RUN chmod +x scripts/entrypoint.sh
ENTRYPOINT ["scripts/entrypoint.sh"]
