# Airflow image for the fraud-detection project.
# Extends the official Airflow image (Python 3.12) and adds:
#   • Project source code at /opt/airflow/project
#   • Project Python dependencies
#   • DAGSHUB / MLflow / DVC env vars (loaded from .env at runtime)
#
# Build:
#   docker build -f airflow/Dockerfile -t fraud-airflow:latest .
#
# Run (via docker-compose):
#   docker compose -f airflow/docker-compose.yml up
#
# Local dev (Airflow on the host, not in Docker):
#   export PROJECT_ROOT=$(pwd)
#   airflow db init
#   airflow scheduler &
#   airflow webserver --port 8080
#
#   PROJECT_ROOT env var is read by every task so it works either way.

FROM apache/airflow:2.10.4-python3.12

USER root

# System packages needed by psycopg2 / feast / catboost
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libpq-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install project dependencies. We exclude apache-airflow itself from
# requirements.txt? — no, we pin it to the base image version in requirements,
# and pip will no-op. Other deps are installed cleanly.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        apache-airflow==2.10.4 \
        apache-airflow-providers-postgres==5.13.0 \
        mlflow==3.11.1 \
        dagshub==0.7.0 \
        catboost \
        "feast[postgres,redis]" \
        psycopg2-binary \
        redis \
        dvc \
        dvc-s3 \
        pandera==0.31.1 \
        evidently \
        prometheus-client==0.24.1 \
        python-dotenv>=1.0.0 \
        hydra-core \
        optuna

# Copy the entire project into the image so DAGs and source are available.
# In production you'd bind-mount the project instead; this is a self-contained
# demo image.
COPY . /opt/airflow/project

# Make project source importable from any DAG task.
ENV PYTHONPATH="/opt/airflow/project:${PYTHONPATH}"
ENV PROJECT_ROOT="/opt/airflow/project"
ENV AIRFLOW__CORE__LOAD_EXAMPLES="False"
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION="False"

# Healthcheck: airflow is healthy if "airflow jobs check --job-type SchedulerJob"
# returns 0. Not a perfect probe but good enough for demo.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD airflow jobs check --job-type SchedulerJob 2>/dev/null || exit 1

WORKDIR /opt/airflow/project
