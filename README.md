# Real-Time Fraud Detection — Sparkov + CatBoost + Feast + Kafka

[![CI](https://github.com/NullBitZer0/real-time-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/NullBitZer0/real-time-fraud-detection/actions/workflows/ci.yml)
[![MLflow](https://img.shields.io/badge/MLflow-DAGsHub-blue?logo=mlflow)](https://dagshub.com/NullBitZer0/real-time-fraud-detection.mlflow)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-Vite-61dafb?logo=react)](https://vitejs.dev/)

End-to-end demo: trains a fraud-detection model on the **Sparkov** dataset
(`fraudTrain.csv` / `fraudTest.csv`), serves it via a **FastAPI** + **CatBoost**
inference pipeline, fetches online features from a **Feast** feature store
(Redis online + file/DuckDB offline), and streams 100-test demo runs
through **Kafka** (KRaft). Experiments + model registry
are tracked on **DAGsHub MLflow**; CI/CD runs on **GitHub Actions**.

> **Next Updates.** Add airflow as a orchastrator —  periodic retraining , production-grade monitoring woth promethus and grafana.also add drift detection with evidently.ETL also will with spark

---

## Architecture

```
                ┌────────────────────────────────────────────────────────────┐
                │                     Sparkov dataset                        │
                │  data/raw/fraudTrain.csv  +  data/raw/fraudTest.csv        │
                └────────────┬───────────────────────────────────────────────┘
                             │  DVC: ingest_postgres
                             ▼
   ┌─────────────────────  PostgreSQL  ─────────────────────┐
   │  fraud_detection.raw_transactions  (1.85M rows)         │
   │  (also used as Feast offline store when configured)     │
   └────────────────────┬────────────────────────────────────┘
                        │  DVC: feast_ingest
                        ▼
   ┌─────────────  Parquet sources  ─────────────┐
   │  data/sparkov_{transaction,cc,merchant}_features.parquet
   └────┬──────────────────┬────────────────────┘
        │                  │  DVC: feast_apply
        │                  ▼
        │         ┌──────────────────┐
        │         │  Feast registry  │
        │         └────────┬─────────┘
        │                  │  DVC: feast_materialize
        │                  ▼
        │         ┌──────────────────┐
        │         │      Redis       │ ◄───── online features
        │         └────────┬─────────┘   (per-cc_num + per-merchant)
        │                  │
        │  DVC: train      │
        ▼                  ▼
   ┌──────────────────────────┐
   │   CatBoost (32 features) │ ──► models/catboost.cbm
   │   + 3-tier thresholds    │ ──► models/metadata.json
   └────────────┬─────────────┘ ──► models/feature_engineering.pkl
                │
                ▼
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │   FastAPI  (api/app.py)  │  ───►   │ React (port 5173)        │
   │   /predict /demo/run-100 │         │  • Live transaction feed │
   │   /metrics /ws           │         │  • 100-test demo results │
   └──────────────────────────┘         └──────────────────────────┘

   ┌──────────────────────────┐
   │   Kafka (KRaft, 9094)    │
   │   topic: fraud-transactions
   │   topic: fraud-decisions
   │   kf/producer.py + kf/consumer.py
   └──────────────────────────┘
```

---

## Quick start

```bash
# 1. Start all infrastructure
docker compose up -d postgres redis kafka

# 2. Run the full DVC pipeline (ingest → feast → train)
dvc repro

# 3. Start the API
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# 4. Start the React dashboard
cd frontend && npm install && npm run dev

# 5. (Optional) Run the 100-test Kafka end-to-end demo
python -m kf.test_100 --broker localhost:9094
```

Then open:
- `http://localhost:5173` — React dashboard with "▶ Run 100 Tests" button
- `http://localhost:8000/docs` — FastAPI Swagger
- `http://localhost:5540` — RedisInsight (Redis UI)
- `http://localhost:5050` — pgAdmin (Postgres UI, login: `admin@admin.com` / `admin`)

---

## DVC pipeline

```
ingest_postgres  ─►  feast_ingest  ─►  train  ─►  metrics/train_metrics.json
                       │
                       └────────►  optuna_tune  (run separately)
                       │
                       └────────►  feast_apply  ─►  feast_materialize  (Redis)
```

| Stage | Command | Description |
|---|---|---|
| `ingest_postgres` | `python -m src.components.data_ingestion_postgres` | CSVs → `fraud_detection.raw_transactions` (COPY, ~30s) |
| `feast_ingest` | `python -m src.components.data_ingestion_feast` | Build `data/sparkov_*.parquet` (32 features) |
| `feast_apply` | `cd feast/feature_repo && feast apply` | Register entities + feature views |
| `feast_materialize` | `feast materialize -v cc_num_features -v merchant_features ...` | Postgres → Redis (999 cards + 693 merchants) |
| `train` | `python -m pipelines.training_pipeline` | CatBoost baseline + 3-tier analysis |
| `optuna_tune` | `python -m src.components.optuna_tuning` | Optional: hyperparameter search (run separately) |

Useful commands:
```bash
dvc repro                     # run all stages
dvc repro train               # run only train
dvc repro drift_report        # Evidently HTML drift report
dvc repro promote             # Staging → Production (with metric gate)
dvc repro optuna_tune -f      # force Optuna tuning
dvc metrics show              # show test_pr_auc, f1@T2, etc.
dvc dag                       # show pipeline graph
dvc push                      # push parquet + model artifacts to DAGsHub S3
```

The full DVC DAG:

```
ingest_postgres ─► feast_ingest ─► train ─► promote
                       │                            ▲
                       ├─► drift_report             │ (metric gate)
                       └─► feast_apply ─► feast_materialize
```

---

## Production-style features

This isn't a notebook prototype — it's structured to demonstrate the
operating concerns a real ML platform needs:

| Concern | Implementation | Where to look |
|---|---|---|
| **Versioned data** | DVC remote = DAGsHub S3 (`s3://dvc` → `dagshub.com/.../s3`) | `.dvc/config`, `dvc push` |
| **Versioned model** | MLflow Model Registry on DAGsHub (Staging → Production) | `src/components/mlflow_tracking.py`, `mlflow_promote.py` |
| **Reproducible training** | Tuned params pinned in `params.yaml`, `random_seed=42`, locked deps | `params.yaml`, `requirements.txt` |
| **Schema contracts** | Pandera validates the 14-field input and 32-feature batch | `src/components/schema_validation.py` |
| **Audit log** | Every `/predict` writes to `fraud_detection.decision_log` (Postgres) | `src/components/audit_log.py` |
| **Drift monitoring** | Evidently HTML + JSON report (train vs test) | `src/components/drift_report.py` → `metrics/drift_report.html` |
| **Metric-gated promotion** | CD fails if `test_pr_auc < 0.78` | `.github/workflows/cd.yml` |
| **Rollback** | Manual workflow that moves Staging → Production | `.github/workflows/rollback.yml` |
| **Readiness probe** | `/readyz` checks model + Redis + Postgres + Feast | `api/app.py:readyz` |
| **Observability** | Prometheus `/metrics/prom` + Grafana dashboard | `api/app.py`, `monitoring/grafana_dashboard.json` |
| **Structured logs** | JSON logs (timestamp, level, name, run_id) — drop-in for ELK | `src/utils/logger.py` |
| **Containerized** | Multi-stage `Dockerfile`, served via `docker compose up api` | `Dockerfile`, `docker-compose.yml` |

---

## MLflow / DAGsHub tracking

Every `dvc repro train` run is logged to DAGsHub MLflow — including
hyperparameters, PR-AUC / ROC-AUC, 3-tier metrics, the trained model, and
the feature-engineering pickle.

| Field | Value |
|---|---|
| Tracking URI | `https://dagshub.com/NullBitZer0/real-time-fraud-detection.mlflow` |
| Experiment | `sparkov-fraud` |
| Registry | `FraudDetectionCatBoost` (new runs → `Staging` stage) |
| DAGsHub repo | https://dagshub.com/NullBitZer0/real-time-fraud-detection |

**Local dev (falls back to `mlruns/`):**
```bash
mlflow ui  # open http://localhost:5000
```

**Required env vars (set in `.env` or GitHub Actions secrets):**
- `DAGSHUB_REPO_OWNER` — `NullBitZer0`
- `DAGSHUB_REPO_NAME`  — `real-time-fraud-detection`
- `DAGSHUB_TOKEN`      — DAGsHub personal access token

---

## CI/CD (GitHub Actions)

| Workflow | File | Trigger | Does |
|---|---|---|---|
| **CI** | `.github/workflows/ci.yml` | push / PR to `main` | lint Python + compile-check + frontend build |
| **CD** | `.github/workflows/cd.yml` | push to `main` | retrain + drift report + metric gate (PR-AUC ≥ 0.78) + promote Staging→Production + upload artifact |
| **Rollback** | `.github/workflows/rollback.yml` | manual (`workflow_dispatch`) | moves Staging → Production, archives the old version, writes a `rollback_log.json` audit record |
| **Dependabot** | `.github/dependabot.yml` | weekly | bumps Python, npm, and GitHub-Actions versions |

Docker images are intentionally **not** built/pushed in CD (per repo policy).
The trained model + drift report are uploaded as a GitHub Actions artifact
(30-day retention).

Required **GitHub repo secrets**: `DAGSHUB_REPO_OWNER`, `DAGSHUB_REPO_NAME`,
`DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`.

---

## Observability

- **Structured JSON logs**: every record is one JSON line (timestamp, level, name, message, run_id). Set `LOG_FORMAT=plain` for human-readable dev output.
- **Prometheus metrics**: `GET /metrics/prom` returns:
  - `fraud_predictions_total` (counter)
  - `fraud_predictions_fraud_total` (counter — tier ≥ 1)
  - `fraud_prediction_latency_ms` (histogram)
- **Grafana dashboard**: `monitoring/grafana_dashboard.json` (4 panels: totals, fraud rate, p50/p95/p99 latency, throughput). Mounted automatically by the Grafana container.
- **Readiness probe**: `GET /readyz` returns 200 only if model + Postgres are reachable (Redis + Feast are reported but not blocking).

Spin up the full observability stack:

```bash
docker compose up -d prometheus grafana
# open http://localhost:3001 (admin/admin) → Dashboards → Real-Time Fraud Detection
```

---

## Audit log

Every `/predict` call writes a row to `fraud_detection.decision_log` (Postgres).
Schema:

| Column | Type | Description |
|---|---|---|
| `trans_num` | TEXT | Transaction ID |
| `fraud_probability` | DOUBLE PRECISION | Model output (0–1) |
| `tier` | SMALLINT | 0=approve, 3=soft_signal, 2=review_queue, 1=auto_block |
| `action` | TEXT | "approve" / "soft_signal" / "review_queue" / "auto_block" |
| `threshold_used` | DOUBLE PRECISION | Tier boundary hit |
| `is_fraud_ground_truth` | SMALLINT NULL | Backfilled from `raw_transactions` for the 100-test demo |
| `model_run_id` | TEXT | MLflow run that produced the model |
| `latency_ms` | DOUBLE PRECISION | End-to-end `/predict` latency |
| `ingested_at` | TIMESTAMPTZ | When the decision was made |
| `request_ip`, `user_agent` | TEXT | Caller metadata |

Useful queries:

```sql
-- How many decisions did each tier in the last hour?
SELECT tier, COUNT(*) FROM fraud_detection.decision_log
WHERE ingested_at > now() - interval '1 hour'
GROUP BY tier ORDER BY tier;

-- Backfill ground truth after a batch test
UPDATE fraud_detection.decision_log d
SET    is_fraud_ground_truth = r.is_fraud
FROM   fraud_detection.raw_transactions r
WHERE  d.trans_num = r.trans_num
  AND  d.is_fraud_ground_truth IS NULL;
```

---

## 3-tier decision logic

| Tier | Threshold | Action | Description |
|---|---|---|---|
| 0 | < 0.0040 | **approve** | Low probability — auto-approve |
| 3 | 0.0040 ≤ p < 0.1198 | **soft_signal** | Possible fraud — flag for monitoring |
| 2 | 0.1198 ≤ p < 0.5613 | **review_queue** | Likely fraud — manual review |
| 1 | ≥ 0.5613 | **auto_block** | High-confidence fraud — auto-decline |

Thresholds come from the OOF run in `notebooks/experiments.ipynb`
(test PR-AUC 0.8665). For `fraud_prediction` (binary), tier ≥ 1 = fraud.

---

## Model performance

| Metric | Val | Test |
|---|---|---|
| PR-AUC | 0.8829 | 0.7903 |
| ROC-AUC | 0.9951 | 0.9894 |
| Macro F1 (T2 threshold) | 0.7305 | 0.6995 |

End-to-end 100-test demo: **Macro F1 = 0.8897** (TP=42, FP=3, FN=8, TN=47).

---

## Feature store (Feast)

Three entities + three feature views:

| Entity | Feature view | Online? | Features |
|---|---|---|---|
| `cc_num` | `cc_num_features` | ✅ Redis | velocity (txn_last_*, amt_sum_last_*), amt stats, FE |
| `merchant` | `merchant_features` | ✅ Redis | merchant_FE, merchant_te, amt_per_merchant stats |
| `trans_num` | `transaction_features` | ❌ | per-row features (hour, amt, distance, ...) |

At inference, the Kafka consumer / API fetches online features from
Redis via `FeatureStoreClient.get_online_features_for_batch()` and
joins them with the row's own transaction features before scoring.

---

## Project layout

```
.
├── api/                    FastAPI app
│   ├── app.py              /health, /predict, /demo/run-100-tests, /metrics, /ws
│   └── schema.py           Pydantic models
├── data/
│   ├── raw/                fraudTrain.csv, fraudTest.csv
│   └── sparkov_*.parquet   Feast offline store sources
├── feast/feature_repo/
│   ├── feature_store.yaml  Redis (online) + file (offline) + Postgres
│   └── features.py         3 entities, 3 feature views
├── frontend/               React + Vite + Recharts dashboard
├── kf/                     Kafka producer/consumer + 100-test demo
│   ├── producer.py         JSON-encode Sparkov rows → fraud-transactions
│   ├── consumer.py         score → 3-tier → fraud-decisions
│   ├── test_100.py         end-to-end 100-test demo
│   └── state.py            static velocity cache (Phase 5, optional)
├── pipelines/
│   ├── training_pipeline.py    hydra-driven training
│   └── prediction_pipeline.py  inference (Feast + CatBoost)
├── src/
│   ├── components/             data_ingestion*, feature_engineering, model_*, feature_store
│   └── entity/                 config_entity, artifact_entity
├── tests/test_parity.py    production model sanity check
├── configs/                Hydra config (model + data + optuna)
├── models/                 catboost.cbm + metadata.json + feature_engineering.pkl
├── metrics/                DVC metrics (train_metrics.json)
├── dvc.yaml                DVC pipeline definition
├── docker-compose.yml      postgres + redis + kafka (KRaft) + UI tools
└── params.yaml             DVC-tracked params (model.iterations, optuna.n_trials, ...)
```

---

## Common tasks

```bash
# Re-train only
dvc repro train

# Tune hyperparameters (separate stage)
dvc repro optuna_tune -f --set-param optuna.n_trials=50

# Re-apply + re-materialize Feast (after editing features.py)
cd feast/feature_repo && feast apply
cd feast/feature_repo && feast materialize -v cc_num_features -v merchant_features "2019-01-01" "2021-01-01"

# Test the model directly
python -m pipelines.prediction_pipeline          # 5-row sanity check
python -m tests.test_parity --n 2000             # full parity test
python -m kf.test_100 --broker localhost:9094    # 100-test Kafka demo

# Inspect Feast online store
redis-cli -h localhost KEYS '*cc_num*' | head -3

# Stop everything
docker compose down
```
