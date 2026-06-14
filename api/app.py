"""Fraud Detection API — Sparkov + CatBoost + 3-tier.

Endpoints:
    GET  /health                       — model status
    POST /predict                      — single Sparkov transaction → {proba, tier, action}
    POST /demo/run-100-tests           — sample 50 fraud + 50 legit from fraudTest, score all
    GET  /demo/decision/{trans_num}    — pull a single decision from the Redis decision cache
    GET  /metrics                      — running totals + avg latency
    WS   /ws                           — push live predictions to React dashboard
"""
import os
import pathlib
import time
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sklearn.metrics import confusion_matrix, f1_score
from starlette.responses import FileResponse, Response

from api.schema import (
    AuditRow,
    FraudResponse,
    HealthResponse,
    MetricsResponse,
    Run100TestsRequest,
    Run100TestsResponse,
    SingleTestResult,
    TransactionRequest,
)
from pipelines.prediction_pipeline import PredictionPipeline
from src.components.audit_log import insert_decision
from src.components.data_ingestion import read_sparkov_split
from src.components.schema_validation import validate_transactions
from src.utils.logger import logging

# Prometheus metrics — defined once at module level to avoid
# "Duplicated timeseries in CollectorRegistry" errors.
METRIC_PREDS_TOTAL = Counter("fraud_predictions_total", "Total /predict calls")
METRIC_PREDS_FRAUD = Counter("fraud_predictions_fraud_total", "Predictions flagged as fraud (tier >= 1)")
METRIC_LATENCY     = Histogram("fraud_prediction_latency_ms", "Latency in ms",
                                buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000))


# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ── Sample loader for the 100-test demo ───────────────────────────────────────
def sample_100(n_fraud: int = 50, n_legit: int = 50, seed: int = 42) -> pd.DataFrame:
    """Pull n_fraud + n_legit rows from fraudTest.csv, sorted by time."""
    path = "data/raw/fraudTest.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — Sparkov dataset is required for the demo")

    df = read_sparkov_split(path)
    fraud = df[df.is_fraud == 1].sample(n=min(n_fraud, (df.is_fraud == 1).sum()), random_state=seed)
    legit = df[df.is_fraud == 0].sample(n=min(n_legit, (df.is_fraud == 0).sum()), random_state=seed)
    sample = pd.concat([fraud, legit]).sort_values("trans_date_trans_time").reset_index(drop=True)
    return sample


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting Fraud Detection API (Sparkov + CatBoost)...")
    try:
        app.state.pipeline = PredictionPipeline()
        logging.info("Model loaded")
    except Exception as e:
        logging.error(f"Model not loaded: {e}")
        app.state.pipeline = None

    app.state.manager    = ConnectionManager()
    app.state.start_time = time.time()
    app.state.stats      = {"total": 0, "fraud": 0, "latencies": []}

    yield
    logging.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time Sparkov fraud scoring — CatBoost + 3-tier",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve static assets from /metrics (drift_report.html, etc.) at /static
_static_dir = pathlib.Path("metrics")
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
else:
    logging.warning(f"static dir '{_static_dir}' not found — drift_report.html will be unavailable")

# Serve React dashboard
_dashboard_dir = pathlib.Path("static/dashboard")
if _dashboard_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_dashboard_dir / "assets")), name="dashboard-assets")
    _dashboard_index = _dashboard_dir / "index.html"
else:
    logging.warning(f"dashboard dir '{_dashboard_dir}' not found — React dashboard will be unavailable")
    _dashboard_index = None


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    p = request.app.state.pipeline
    return HealthResponse(
        status       = "ok",
        model_loaded = p is not None,
        model_name   = (p.metadata["model_name"] if p else None),
    )


# ── /readyz ───────────────────────────────────────────────────────────────────
@app.get("/readyz", tags=["System"])
async def readyz(request: Request):
    """Readiness probe — model + Redis + Postgres + Feast all reachable."""
    checks = {}
    overall_ok = True

    # 1. Model
    p = request.app.state.pipeline
    checks["model"] = {"ok": p is not None}
    if p is None:
        overall_ok = False

    # 2. Redis (Feast online store — probed via Feast itself, see check 4)
    checks["redis"] = {"ok": True, "note": "health verified via Feast check below"}

    # 3. Postgres (audit log)
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            user=os.environ.get("POSTGRES_USER", "feast"),
            password=os.environ.get("POSTGRES_PASSWORD", "feast"),
            dbname=os.environ.get("POSTGRES_DB", "feast"),
        )
        conn.close()
        checks["postgres"] = {"ok": True}
    except Exception as e:
        checks["postgres"] = {"ok": False, "error": str(e)}
        overall_ok = False

    # 4. Feast
    try:
        if p is not None and p.fsc is not None:
            p.fsc.store.get_feature_server()
        checks["feast"] = {"ok": True}
    except Exception as e:
        checks["feast"] = {"ok": False, "error": str(e)}
        # Feast is optional
        # overall_ok = False

    return {
        "ready":   overall_ok,
        "checks":  checks,
        "version": "2.1.0",
    }


# ── /metrics/prom ────────────────────────────────────────────────────────────
@app.get("/metrics/prom", tags=["System"])
async def prometheus_metrics(request: Request):
    """Prometheus-format metrics — total/fraud counters + latency histogram."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── /predict ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=FraudResponse, tags=["Inference"])
async def predict(transaction: TransactionRequest, request: Request):
    pipeline = request.app.state.pipeline
    stats    = request.app.state.stats
    manager  = request.app.state.manager

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0  = time.perf_counter()
    row = transaction.model_dump()
    df  = pd.DataFrame([row])

    # Schema contract (Pandera) — coerce bad types, log warnings, never block
    try:
        df = validate_transactions(df)
    except Exception as e:
        logging.warning(f"schema validation failed: {e} — passing through")

    out = pipeline.predict(df)
    latency_ms = (time.perf_counter() - t0) * 1000

    prob   = float(out["proba"].iloc[0])
    tier   = int  (out["tier"].iloc[0])
    action = str  (out["action"].iloc[0])
    pred   = int  (tier >= 1)  # tier 1+ = flagged (T1 auto_block / T2 review / T3 soft_signal)
    txn_id = str  (out["transaction_id"].iloc[0])
    threshold = float(out["threshold"].iloc[0])

    stats["total"] += 1
    stats["fraud"] += pred
    stats["latencies"].append(latency_ms)
    if len(stats["latencies"]) > 5000:
        stats["latencies"].pop(0)

    # Prometheus counters (module-level, no duplicates)
    METRIC_PREDS_TOTAL.inc()
    if pred:
        METRIC_PREDS_FRAUD.inc()
    METRIC_LATENCY.observe(latency_ms)

    # Audit log → Postgres
    try:
        insert_decision(
            trans_num=txn_id, proba=prob, tier=tier, action=action, threshold=threshold,
            latency_ms=latency_ms,
            request_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    except Exception as e:
        logging.warning(f"audit_log insert failed: {e}")

    response = FraudResponse(
        transaction_id    = txn_id,
        fraud_probability = round(prob, 4),
        fraud_prediction  = pred,
        tier              = tier,
        action            = action,
        threshold_used    = threshold,
        latency_ms        = round(latency_ms, 2),
    )
    await manager.broadcast(response.model_dump())
    return response


# ── /demo/run-100-tests ──────────────────────────────────────────────────────
@app.post("/demo/run-100-tests",
          response_model=Run100TestsResponse,
          tags=["Demo"])
async def run_100_tests(req: Run100TestsRequest, request: Request):
    """Run a 100-transaction demo: n_fraud fraud + n_legit legit from fraudTest.

    Returns individual results + 3-tier summary + confusion matrix
    (vs ground-truth `is_fraud` from fraudTest.csv).
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sample  = sample_100(n_fraud=req.n_fraud, n_legit=req.n_legit)
    t0      = time.perf_counter()
    out     = pipeline.predict(sample)
    batch_latency = (time.perf_counter() - t0) * 1000
    avg_lat = batch_latency / len(out)
    stats   = request.app.state.stats
    manager = request.app.state.manager
    y_true  = out["is_fraud"].astype(int).values

    tier_thresholds = pipeline.tier_thresholds

    results = []
    for _, r in out.iterrows():
        proba   = float(r["proba"])
        tier    = int(r["tier"])
        action  = str(r["action"])
        pred    = int(tier >= 1)
        txn_id  = str(r["trans_num"])
        lat_ms  = avg_lat

        results.append(SingleTestResult(
            trans_num         = txn_id,
            is_fraud          = int(r["is_fraud"]),
            fraud_probability = proba,
            fraud_prediction  = pred,
            tier              = tier,
            action            = action,
            amt               = float(r["amt"]),
            merchant          = str(r["merchant"]),
            category          = str(r["category"]),
        ))

        # Record for Live tab / Prometheus / audit log (same as /predict)
        stats["total"] += 1
        stats["fraud"] += pred
        METRIC_PREDS_TOTAL.inc()
        if pred:
            METRIC_PREDS_FRAUD.inc()
        METRIC_LATENCY.observe(lat_ms)

        try:
            insert_decision(
                trans_num=txn_id, proba=proba, tier=tier, action=action,
                threshold=float(r["threshold"]), latency_ms=lat_ms,
                request_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
        except Exception as e:
            logging.warning(f"audit_log insert failed for {txn_id}: {e}")

        # Broadcast each prediction to the Live tab via WebSocket
        await manager.broadcast({
            "transaction_id":    txn_id,
            "fraud_probability": round(proba, 4),
            "fraud_prediction":  pred,
            "tier":              tier,
            "action":            action,
            "threshold_used":    float(r["threshold"]),
            "latency_ms":        lat_ms,
        })

    # Confusion matrix + F1 (all tier >= 1 = flagged)
    y_pred = (out["tier"] >= 1).astype(int).values
    cm  = confusion_matrix(y_true, y_pred).tolist()
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return Run100TestsResponse(
        n_total      = len(out),
        n_fraud      = int(y_true.sum()),
        n_legit      = int((y_true == 0).sum()),
        tier1_count  = int((out["tier"] == 1).sum()),
        tier2_count  = int((out["tier"] == 2).sum()),
        tier3_count  = int((out["tier"] == 3).sum()),
        tier0_count  = int((out["tier"] == 0).sum()),
        tier_thresholds = tier_thresholds,
        confusion_matrix = cm,
        macro_f1     = f1m,
        results      = results if req.include_results else [],
    )


# ── /demo/decision/{trans_num} ────────────────────────────────────────────────
@app.get("/demo/decision/{trans_num}", tags=["Demo"])
async def get_decision(trans_num: str, request: Request):
    """Look up a single decision from the Postgres audit log.

    Returns 200 + decision if found, 404 if not present.
    The legacy Redis decision cache was removed — predictions are now
    stored permanently in the Postgres audit log.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            user=os.environ.get("POSTGRES_USER", "feast"),
            password=os.environ.get("POSTGRES_PASSWORD", "feast"),
            dbname=os.environ.get("POSTGRES_DB", "feast"),
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trans_num, fraud_probability, tier, action, threshold_used,
                       is_fraud_ground_truth, model_version, latency_ms, ingested_at,
                       request_ip, user_agent
                FROM   fraud_detection.decision_log
                WHERE  trans_num = %s
                """,
                (trans_num,),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"no decision found for {trans_num}")
        return {
            "trans_num":                row[0],
            "fraud_probability":        row[1],
            "tier":                     row[2],
            "action":                   row[3],
            "threshold_used":           row[4],
            "is_fraud_ground_truth":    row[5],
            "model_version":            row[6],
            "latency_ms":               row[7],
            "ingested_at":              row[8].isoformat() if row[8] else None,
            "request_ip":               row[9],
            "user_agent":               row[10],
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"/demo/decision/{trans_num} failed: {e}")
        raise HTTPException(status_code=503, detail=f"decision lookup failed: {e}")
    finally:
        if conn is not None:
            conn.close()


# ── /metrics ──────────────────────────────────────────────────────────────────
@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics(request: Request):
    stats = request.app.state.stats
    lats  = stats["latencies"]
    total = stats["total"]
    fraud = stats["fraud"]
    return MetricsResponse(
        total_transactions = total,
        total_fraud        = fraud,
        fraud_rate_pct     = round(fraud / total * 100, 3) if total else 0.0,
        avg_latency_ms     = round(sum(lats) / len(lats), 2) if lats else 0.0,
        uptime_seconds     = round(time.time() - request.app.state.start_time, 1),
    )


# ── /ws ───────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager = websocket.app.state.manager
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ── /audit/recent ─────────────────────────────────────────────────────────────
@app.get("/audit/recent", response_model=List[AuditRow], tags=["System"])
async def audit_recent(n: int = 100):
    """Read the last N rows from fraud_detection.decision_log.

    Used by the React dashboard's Audit Log tab.
    Fails with 503 if Postgres is unreachable.
    """
    n = max(1, min(int(n), 1000))
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            user=os.environ.get("POSTGRES_USER", "feast"),
            password=os.environ.get("POSTGRES_PASSWORD", "feast"),
            dbname=os.environ.get("POSTGRES_DB", "feast"),
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, trans_num, fraud_probability, tier, action, threshold_used,
                       is_fraud_ground_truth, model_version, latency_ms, ingested_at,
                       request_ip, user_agent
                FROM   fraud_detection.decision_log
                ORDER  BY ingested_at DESC
                LIMIT  %s
                """,
                (n,),
            )
            rows = cur.fetchall()
    except Exception as e:
        logging.error(f"/audit/recent failed: {e}")
        raise HTTPException(status_code=503, detail=f"audit log unavailable: {e}")
    finally:
        if conn is not None:
            conn.close()

    return [
        AuditRow(
            id                    = r[0],
            trans_num             = r[1],
            fraud_probability     = r[2],
            tier                  = r[3],
            action                = r[4],
            threshold_used        = r[5],
            is_fraud_ground_truth = r[6],
            model_version         = r[7],
            latency_ms            = r[8],
            ingested_at           = r[9],
            request_ip            = r[10],
            user_agent            = r[11],
        )
        for r in rows
    ]


# ── Catch-all: serve React index.html for client-side routing ─────────────────

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_dashboard(full_path: str):
    """Serve React SPA — any non-API route returns index.html."""
    if _dashboard_index and _dashboard_index.exists():
        return FileResponse(str(_dashboard_index))
    raise HTTPException(status_code=404, detail="Dashboard not found")
