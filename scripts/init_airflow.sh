#!/usr/bin/env bash
# Initialize Airflow for the fraud-detection project.
#
# Two modes:
#   1) Docker stack:  ./scripts/init_airflow.sh docker
#      - Just runs `docker compose -f airflow/docker-compose.yml up -d`
#      - The airflow-init service creates the admin user automatically
#
#   2) Local dev:    ./scripts/init_airflow.sh local
#      - Runs `airflow db init` + creates admin user
#      - Caller must have pip-installed apache-airflow
#
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-docker}"

case "$MODE" in
    docker)
        echo "🐳 Starting Airflow stack via Docker..."

        # Load .env FIRST so docker compose sees DAGSHUB_*, MLFLOW_*, etc.
        # Without this, DAGsHub/MLflow tasks will fail with 401 Unauthorized.
        if [[ -f .env ]]; then
            set -a
            # shellcheck disable=SC1091
            source .env
            set +a
            echo "✓ Loaded .env (DAGSHUB_*, MLFLOW_TRACKING_URI)"
        else
            echo "⚠️  No .env file found — DAGsHub / MLflow will not be reachable"
            echo "   (this is fine for Airflow itself; only matters for DAG tasks)"
        fi

        # Run with --env-file as belt-and-braces in case any var was missed
        docker compose \
            --env-file .env \
            -f airflow/docker-compose.yml \
            up -d 2>&1 | grep -v "WARN\[0000\]" || true
        echo ""
        echo "✓ Airflow is running at http://localhost:8080"
        echo "  user: admin"
        echo "  pass: admin"
        echo ""
        echo "Useful commands:"
        echo "  docker compose -f airflow/docker-compose.yml ps"
        echo "  docker compose -f airflow/docker-compose.yml logs -f airflow-scheduler"
        echo "  docker compose -f airflow/docker-compose.yml down"
        ;;

    local)
        echo "🖥️  Initialising Airflow in local mode..."
        export AIRFLOW_HOME="${AIRFLOW_HOME:-$(pwd)/.airflow_home}"
        export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
        export PROJECT_ROOT="$(pwd)"

        echo "  AIRFLOW_HOME = ${AIRFLOW_HOME}"
        echo "  DAGS_FOLDER  = ${AIRFLOW__CORE__DAGS_FOLDER}"
        echo ""

        # Required env (project .env must be sourced for DAGsHub + MLflow)
        if [[ -f .env ]]; then
            set -a; source .env; set +a
            echo "  ✓ Loaded .env"
        else
            echo "  ⚠️  .env not found — DAGsHub / MLflow will not be reachable"
        fi

        # DB init
        airflow db init

        # Admin user (idempotent)
        airflow users create \
            --username admin \
            --firstname Admin \
            --lastname  User \
            --role      Admin \
            --email     admin@example.com \
            --password  admin || echo "  (admin user already exists)"

        echo ""
        echo "✓ Airflow initialised. Start scheduler + webserver in separate terminals:"
        echo "  airflow scheduler --port 8080"
        echo "  airflow webserver"
        ;;

    *)
        echo "Usage: $0 [docker|local]"
        exit 1
        ;;
esac
