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
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import confusion_matrix, f1_score

from api.schema import (
    TransactionRequest, FraudResponse,
    Run100TestsRequest, Run100TestsResponse, SingleTestResult,
    MetricsResponse, HealthResponse,
)
from pipelines.prediction_pipeline import PredictionPipeline
from src.components.data_ingestion import read_sparkov_split
from src.utils.logger import logging


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


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    p = request.app.state.pipeline
    return HealthResponse(
        status       = "ok",
        model_loaded = p is not None,
        model_name   = (p.metadata["model_name"] if p else None),
    )


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
    out = pipeline.predict(df)
    latency_ms = (time.perf_counter() - t0) * 1000

    prob   = float(out["proba"].iloc[0])
    tier   = int  (out["tier"].iloc[0])
    action = str  (out["action"].iloc[0])
    pred   = int  (tier >= 1)  # tier 1+ = flagged (T1 auto_block / T2 review / T3 soft_signal)
    txn_id = str  (out["transaction_id"].iloc[0])

    stats["total"] += 1
    stats["fraud"] += pred
    stats["latencies"].append(latency_ms)
    if len(stats["latencies"]) > 5000:
        stats["latencies"].pop(0)

    response = FraudResponse(
        transaction_id    = txn_id,
        fraud_probability = round(prob, 4),
        fraud_prediction  = pred,
        tier              = tier,
        action            = action,
        threshold_used    = float(out["threshold"].iloc[0]),
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

    sample = sample_100(n_fraud=req.n_fraud, n_legit=req.n_legit)
    out    = pipeline.predict(sample)
    y_true = out["is_fraud"].astype(int).values

    # Predictions for the confusion matrix: any tier >= 1 = flagged
    # (T1 auto_block, T2 review_queue, T3 soft_signal all count as "fraud detected")
    y_pred = (out["tier"] >= 1).astype(int).values

    tier_thresholds = pipeline.tier_thresholds
    cm  = confusion_matrix(y_true, y_pred).tolist()
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    results = [
        SingleTestResult(
            trans_num         = str(r["trans_num"]),
            is_fraud          = int (r["is_fraud"]),
            fraud_probability = float(r["proba"]),
            fraud_prediction  = int (r["tier"] >= 1),
            tier              = int (r["tier"]),
            action            = str (r["action"]),
            amt               = float(r["amt"]),
            merchant          = str (r["merchant"]),
            category          = str (r["category"]),
        )
        for _, r in out.iterrows()
    ]

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
        results      = results,
    )


# ── /demo/decision/{trans_num} ────────────────────────────────────────────────
@app.get("/demo/decision/{trans_num}", tags=["Demo"])
async def get_decision(trans_num: str, request: Request):
    """Read a single decision from the Redis decision cache (fraud:decision:{trans_num}).

    Returns 200 + decision if cached, 404 if not present.
    Used by the React dashboard to poll for results after a /predict call.
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cached = pipeline.get_cached_decision(trans_num)
    if not cached:
        raise HTTPException(status_code=404, detail=f"no cached decision for {trans_num}")
    return cached


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
