# Real-Time Fraud Detection — Sparkov + CatBoost + Feast + Kafka

[![CI](https://github.com/NullBitZer0/real-time-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/NullBitZer0/real-time-fraud-detection/actions/workflows/ci.yml)
[![MLflow](https://img.shields.io/badge/MLflow-DAGsHub-blue?logo=mlflow)](https://dagshub.com/NullBitZer0/real-time-fraud-detection.mlflow)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-Vite-61dafb?logo=react)](https://vitejs.dev/)

End-to-end production-grade real-time fraud-detection system. It processes credit card transaction streams using **Kafka (KRaft)**, performs low-latency feature lookups using **Feast** (with **PostgreSQL** as the offline historical store and **Redis** as the online store), and evaluates transactions using a **CatBoost** classifier with a custom **3-tier action framework**. 

All configurations are managed simply through `params.yaml`. System orchestration is handled via **Apache Airflow**, experiment tracking is managed via **DAGsHub MLflow**, and deployment validation is automated through **GitHub Actions CI/CD**.

---

## Architecture Diagram

![Fraud Detection Architecture](images/diagram.jpeg)


## Retraining & Ingestion Architecture

Pipeline orchestration has transitioned from DVC stages (`dvc repro`) to **Apache Airflow** DAGs under `airflow/dags/`. DVC is used **exclusively for data versioning** of raw CSV files.

### Orchestration DAGs

| DAG ID | Schedule | Description |
| :--- | :--- | :--- |
| `fraud_retraining` | `0 0 * * 1` (Monday) | Performs time-split feature loading, trains CatBoost, performs metric verification (gate), registers the model to MLflow (Staging), and updates Feast. |
| `fraud_drift_check` | `0 6 * * *` (Daily) | Computes population drift via Evidently and triggers `fraud_retraining` if drift occurs on 5 or more features. |

---

## Getting Started

### Complete One-Command Demo Launch
Bring up all services (Ingest, Feast, Kafka, ML, APIs, Dashboard):
```bash
bash scripts/demo.sh
```

### Manual Component Initialization

```bash
# 1. Start all infrastructure services
docker compose up -d postgres redis kafka prometheus grafana

# 2. Seed Feast offline (PostgreSQL) and online (Redis) databases
python feast/seed.py

# 3. Start Apache Airflow orchestrator
bash scripts/init_airflow.sh docker

# 4. Run the API scorer
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# 5. Run Vite React application
cd frontend && npm install && npm run dev
```

To shut down everything: `bash scripts/demo-stop.sh`

---

## Observability Port Guide

- **Vite React Dashboard**: [http://localhost:3000](http://localhost:3000) (7 Observability Tabs)
- **FastAPI OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Apache Airflow Dashboard**: [http://localhost:8080](http://localhost:8080) (Default credentials: `admin` / `admin`)
- **Grafana Metrics View**: [http://localhost:3001](http://localhost:3001)
- **Prometheus Scraper**: [http://localhost:9090](http://localhost:9090)
- **RedisInsight Console**: [http://localhost:5540](http://localhost:5540)
- **pgAdmin Panel**: [http://localhost:5050](http://localhost:5050) (Credentials: `admin@admin.com` / `admin`)

---

## 3-Tier Decision Actions

The prediction pipeline scores transaction records and flags them according to the following probability thresholds defined in `params.yaml`:

| Tier | Probability Bounds | System Action | Action Details |
| :---: | :--- | :--- | :--- |
| **0** | `p < 0.0040` | `approve` | Low risk transaction: automatically processed. |
| **3** | `0.0040 <= p < 0.1198` | `soft_signal` | Moderate risk: processed but flagged for downstream auditing. |
| **2** | `0.1198 <= p < 0.5613` | `review_queue` | High risk: forwarded to queues for manual operator review. |
| **1** | `p >= 0.5613` | `auto_block` | Critical risk: transaction declined instantly. |

---

## Project Layout

```
.
├── api/                    # FastAPI microservice serving predictions and websocket metrics
├── airflow/                # Apache Airflow orchestrators, DAGs and Compose configuration
├── data/
│   └── raw/                # Raw CSV records and DVC tracking files (.csv.dvc)
├── feast/                  # Feast Feature Store definitions and config yaml
├── frontend/               # React (Vite) Unified Dashboard UI
├── kf/                     # Kafka message producers and consumer logic
├── pipelines/
│   ├── training_pipeline.py   # Main ML training pipeline (loads from Feast, trains CatBoost)
│   └── prediction_pipeline.py # Inference pipeline serving real-time predictions
├── src/                    # Source package (features, validation, training components, logger)
├── scripts/                # Startup, shutdown, and testing utility scripts
├── tests/                  # Integration tests, parity verification scripts
├── params.yaml             # Consolidated, single-point parameter configuration
└── requirements.txt        # Python dependency manifest
```

---

## Common Tasks

### 1. Trigger Model Retraining (Airflow CLI/REST)
```bash
curl -X POST http://localhost:8080/api/v1/dags/fraud_retraining/dagRuns \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{"conf": {"trigger_source": "manual"}}'
```

### 2. Run Standalone Offline Training
```bash
python -m pipelines.training_pipeline
```

### 3. Re-Apply Feast Feature Schema Changes
```bash
cd feast/feature_repo
feast apply
feast materialize "2019-01-01" "2021-01-01"
```

### 4. Run Kafka 100-Transaction Demo Stream
```bash
python -m kf.test_100 --broker localhost:9094
```