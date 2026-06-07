# Airflow for the Fraud Detection Project

This stack orchestrates the **weekly + drift-triggered retraining** of the
fraud detection model. It replaces the old `dvc.yaml` pipeline (which is now
empty — DVC is only used for data versioning, not for orchestration).

## Architecture

```
       ┌─────────────────────┐
       │  fraud_drift_check  │  daily 06:00 UTC
       │  (Drift watcher)    │  runs Evidently, decides
       └──────────┬──────────┘
                  │ TriggerDagRunOperator (only if drift)
                  ▼
       ┌─────────────────────┐         ┌────────────────────┐
       │  fraud_retraining   │ ◀────── │ weekly Mon 00:00   │
       │  (Main retrain DAG) │         │ schedule           │
       └──────────┬──────────┘         └────────────────────┘
                  │
                  ├── generate_drift_report
                  ├── decide_path (branch)
                  ├── retrain (task group)
                  │     ├── pull_data        (dvc pull)
                  │     ├── train_model      (python -m pipelines.training_pipeline)
                  │     ├── metric_gate      (refuses if test_pr_auc < 0.78)
                  │     ├── promote          (python -m src.components.mlflow_promote)
                  │     └── refresh_feast    (feast apply + materialize)
                  └── notify
```

## Quick start

```bash
# 1. Build the Airflow image (one-time, takes a few minutes)
docker compose -f airflow/docker-compose.yml build

# 2. Start the stack
docker compose -f airflow/docker-compose.yml up -d

# 3. Open the UI
#    http://localhost:8080
#    user: admin
#    pass: admin
```

## DAGs

| DAG id                | Schedule           | Purpose                                        |
|-----------------------|--------------------|------------------------------------------------|
| `fraud_retraining`    | `0 0 * * 1` (Mon)  | Full retrain + promotion                       |
| `fraud_drift_check`   | `0 6 * * *`        | Daily drift check, triggers `fraud_retraining` |

You can also trigger `fraud_retraining` manually from the UI (or via REST):

```bash
curl -X POST http://localhost:8080/api/v1/dags/fraud_retraining/dagRuns \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{"conf": {"trigger_source": "manual"}}'
```

## Tunable variables

Both DAGs read from Airflow Variables (set in the UI under Admin → Variables):

| Variable name             | Default | Meaning                                          |
|---------------------------|---------|--------------------------------------------------|
| `drift_feature_threshold` | `5`     | # of drifted features that trigger a retrain     |
| `promotion_metric_floor`  | `0.78`  | Min test PR-AUC to allow a model into Production |

## Local dev (Airflow on the host, not Docker)

```bash
# 1. Install Airflow
pip install apache-airflow==2.10.4
export AIRFLOW_HOME=$(pwd)/.airflow_home

# 2. Point Airflow at this project's DAGs
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
export PROJECT_ROOT=$(pwd)

# 3. Init
airflow db init
airflow users create --username admin --role Admin --password admin \
  --email admin@example.com --firstname A --lastname B

# 4. Run
airflow scheduler &
airflow webserver --port 8080
```

## Files

- `dags/fraud_retraining.py` — main retraining DAG (weekly + triggered)
- `dags/drift_check.py` — daily drift watcher that triggers retraining
- `Dockerfile` — extends `apache/airflow:2.10.4-python3.12`
- `docker-compose.yml` — full stack (Postgres + webserver + scheduler)
