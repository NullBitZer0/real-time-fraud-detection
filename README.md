# Real-Time Fraud Detection — Sparkov + CatBoost + Feast + Kafka

[![CI](https://github.com/NullBitZer0/real-time-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/NullBitZer0/real-time-fraud-detection/actions/workflows/ci.yml)
[![MLflow](https://img.shields.io/badge/MLflow-DAGsHub-blue?logo=mlflow)](https://dagshub.com/NullBitZer0/real-time-fraud-detection.mlflow)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-Vite-61dafb?logo=react)](https://vitejs.dev/)

End-to-end production-grade real-time fraud-detection system. It processes credit card transaction streams using **Kafka (KRaft)**, performs low-latency feature lookups using **Feast** (with **PostgreSQL** as the offline historical store and **Redis** as the online store), and evaluates transactions using a **CatBoost** classifier with a custom **3-tier action framework**. 

All configurations are managed simply through `params.yaml`. System orchestration is handled via **Apache Airflow**, experiment tracking is managed via **DAGsHub MLflow**, and deployment validation is automated through **GitHub Actions CI/CD**.

---

## Architecture Diagram (Draw.io XML)

To render the system architecture dynamically and beautifully in [Draw.io](https://app.diagrams.net/):
1. Copy the XML code block below.
2. Open Draw.io, click **File** -> **Import from** -> **Text...** (or **Device...**).
3. Paste the XML code and click **Import**.

```xml
<mxfile host="65bd71144e">
  <diagram id="RealTimeFraudDetectionArchitecture" name="System Architecture">
    <mxGraphModel dx="1422" dy="829" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="850" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Group: Data Ingestion & Features -->
        <mxCell id="GroupData" value="Data &amp; Feature Store Infrastructure" style="swimlane;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#BDC3C7;fontColor=#2C3E50;fontSize=14;fontStyle=1;align=left;spacingLeft=10;" vertex="1" parent="1">
          <mxGeometry x="40" y="80" width="300" height="420" as="geometry" />
        </mxCell>
        <mxCell id="CSV" value="&lt;b&gt;Raw Dataset&lt;/b&gt;&lt;br&gt;fraudTrain.csv / fraudTest.csv&lt;br&gt;&lt;i&gt;(DVC Tracked)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F0FE;strokeColor=#4285F4;fontColor=#1967D2;" vertex="1" parent="GroupData">
          <mxGeometry x="30" y="50" width="240" height="60" as="geometry" />
        </mxCell>
        <mxCell id="Postgres" value="&lt;b&gt;PostgreSQL&lt;/b&gt;&lt;br&gt;Offline Store (1.85M Rows)&lt;br&gt;fraud_detection.transaction_features" style="shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=15;whiteSpace=wrap;html=1;fillColor=#E6F4EA;strokeColor=#34A853;fontColor=#137333;" vertex="1" parent="GroupData">
          <mxGeometry x="30" y="170" width="240" height="80" as="geometry" />
        </mxCell>
        <mxCell id="Redis" value="&lt;b&gt;Redis Online Store&lt;/b&gt;&lt;br&gt;Low-Latency Features&lt;br&gt;(cc_num / merchant keys)" style="shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=15;whiteSpace=wrap;html=1;fillColor=#FCE8E6;strokeColor=#EA4335;fontColor=#C5221F;" vertex="1" parent="GroupData">
          <mxGeometry x="30" y="310" width="240" height="80" as="geometry" />
        </mxCell>

        <!-- Group: Streaming & Inference -->
        <mxCell id="GroupInference" value="Streaming Ingestion &amp; Real-Time Scoring" style="swimlane;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#BDC3C7;fontColor=#2C3E50;fontSize=14;fontStyle=1;align=left;spacingLeft=10;" vertex="1" parent="1">
          <mxGeometry x="410" y="80" width="320" height="420" as="geometry" />
        </mxCell>
        <mxCell id="KafkaTopicIn" value="&lt;b&gt;Kafka Topic: fraud-transactions&lt;/b&gt;&lt;br&gt;Streaming Raw Transaction JSON" style="shape=mxgraph.flowchart.direct_data;whiteSpace=wrap;html=1;fillColor=#FEF7E0;strokeColor=#FBBC05;fontColor=#B06000;" vertex="1" parent="GroupInference">
          <mxGeometry x="25" y="50" width="270" height="60" as="geometry" />
        </mxCell>
        <mxCell id="Consumer" value="&lt;b&gt;Kafka Consumer &amp; Scorer&lt;/b&gt;&lt;br&gt;scores batches via CatBoost&lt;br&gt;&lt;i&gt;(kf/consumer.py)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F0FE;strokeColor=#4285F4;fontColor=#1967D2;" vertex="1" parent="GroupInference">
          <mxGeometry x="25" y="180" width="270" height="70" as="geometry" />
        </mxCell>
        <mxCell id="KafkaTopicOut" value="&lt;b&gt;Kafka Topic: fraud-decisions&lt;/b&gt;&lt;br&gt;Decisions &amp;amp; 3-Tier Action Envelopes" style="shape=mxgraph.flowchart.direct_data;whiteSpace=wrap;html=1;fillColor=#FEF7E0;strokeColor=#FBBC05;fontColor=#B06000;" vertex="1" parent="GroupInference">
          <mxGeometry x="25" y="320" width="270" height="60" as="geometry" />
        </mxCell>

        <!-- Group: App, UI & Monitoring -->
        <mxCell id="GroupObservability" value="Application API &amp; Observability" style="swimlane;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#BDC3C7;fontColor=#2C3E50;fontSize=14;fontStyle=1;align=left;spacingLeft=10;" vertex="1" parent="1">
          <mxGeometry x="800" y="80" width="260" height="420" as="geometry" />
        </mxCell>
        <mxCell id="FastAPI" value="&lt;b&gt;FastAPI Application&lt;/b&gt;&lt;br&gt;api/app.py&lt;br&gt;/predict | /ws | /metrics" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F0FE;strokeColor=#4285F4;fontColor=#1967D2;" vertex="1" parent="GroupObservability">
          <mxGeometry x="20" y="50" width="220" height="60" as="geometry" />
        </mxCell>
        <mxCell id="ReactDashboard" value="&lt;b&gt;React Unified Dashboard&lt;/b&gt;&lt;br&gt;Live Streams | Audits | Drift&lt;br&gt;&lt;i&gt;(Vite App)&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EAF2FF;strokeColor=#2B579A;fontColor=#1E3A8A;" vertex="1" parent="GroupObservability">
          <mxGeometry x="20" y="160" width="220" height="60" as="geometry" />
        </mxCell>
        <mxCell id="PromGrafana" value="&lt;b&gt;Prometheus &amp;amp; Grafana&lt;/b&gt;&lt;br&gt;Throughput, P99 Latency,&lt;br&gt;and Fraud Metrics" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF0E6;strokeColor=#FF6600;fontColor=#CC5200;" vertex="1" parent="GroupObservability">
          <mxGeometry x="20" y="270" width="220" height="70" as="geometry" />
        </mxCell>

        <!-- Group: Orchestration & MLOps -->
        <mxCell id="GroupOrchestration" value="Orchestration &amp;amp; MLOps Lifecycle" style="swimlane;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#BDC3C7;fontColor=#2C3E50;fontSize=14;fontStyle=1;align=left;spacingLeft=10;" vertex="1" parent="1">
          <mxGeometry x="40" y="550" width="1020" height="200" as="geometry" />
        </mxCell>
        <mxCell id="Airflow" value="&lt;b&gt;Apache Airflow&lt;/b&gt;&lt;br&gt;Drift Detection (Daily Evidently Report) &amp;amp; Retraining DAGs (Weekly)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EBF3FB;strokeColor=#017CE4;fontColor=#004080;" vertex="1" parent="GroupOrchestration">
          <mxGeometry x="30" y="50" width="450" height="80" as="geometry" />
        </mxCell>
        <mxCell id="MLflow" value="&lt;b&gt;MLflow (DAGsHub)&lt;/b&gt;&lt;br&gt;Experiment Tracking, Parameters, Metrics &amp;amp; Model Registry" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F0FE;strokeColor=#4285F4;fontColor=#1967D2;" vertex="1" parent="GroupOrchestration">
          <mxGeometry x="540" y="50" width="450" height="80" as="geometry" />
        </mxCell>

        <!-- Connectors -->
        <mxCell id="c1" edge="1" parent="1" source="CSV" target="Postgres" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7F8C8D;strokeWidth=2;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="c2" edge="1" parent="1" source="Postgres" target="Redis" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7F8C8D;strokeWidth=2;labelBackgroundColor=none;">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="190" y="390" as="offset" />
          </mxGeometry>
        </mxCell>
        <mxCell id="c3" edge="1" parent="1" source="KafkaTopicIn" target="Consumer" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7F8C8D;strokeWidth=2;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="c4" edge="1" parent="1" source="Consumer" target="KafkaTopicOut" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7F8C8D;strokeWidth=2;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="c5" edge="1" parent="1" source="Redis" target="Consumer" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2980B9;strokeWidth=2;exitX=1;exitY=0.5;exitDx=0;exitDy=0;exitPerimeter=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="360" y="300" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="c6" edge="1" parent="1" source="Consumer" target="FastAPI" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#27AE60;strokeWidth=2;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="c7" edge="1" parent="1" source="FastAPI" target="ReactDashboard" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7F8C8D;strokeWidth=2;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="c8" edge="1" parent="1" source="FastAPI" target="PromGrafana" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7F8C8D;strokeWidth=2;exitX=0.75;exitY=1;exitDx=0;exitDy=0;entryX=0.75;entryY=0;entryDx=0;entryDy=0;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="c9" edge="1" parent="1" source="Airflow" target="MLflow" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2980B9;strokeWidth=2;dashed=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="c10" edge="1" parent="1" source="Airflow" target="Postgres" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#017CE4;strokeWidth=1.5;dashed=1;exitX=0.25;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;entryPerimeter=0;">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="140" y="510" as="targetPoint" />
          </mxGeometry>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

---

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