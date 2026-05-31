import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schema import (
    TransactionRequest, FraudResponse,
    MetricsResponse, HealthResponse,
)
from pipelines.prediction_pipeline import PredictionPipeline


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


def get_risk_level(prob: float) -> str:
    if prob < 0.30:  return "LOW"
    if prob < 0.50:  return "MEDIUM"
    if prob < 0.75:  return "HIGH"
    return "CRITICAL"


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Fraud Detection API...")
    try:
        app.state.pipeline = PredictionPipeline()
        print(f"Model loaded — threshold={app.state.pipeline.threshold}")
    except Exception as e:
        print(f"Model not loaded: {e}")
        app.state.pipeline = None

    app.state.manager    = ConnectionManager()
    app.state.start_time = time.time()
    app.state.stats      = {"total": 0, "fraud": 0, "latencies": []}

    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time credit card fraud scoring — LightGBM + Feast + Kafka",
    version     = "1.0.0",
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
    return HealthResponse(
        status       = "ok",
        model_loaded = request.app.state.pipeline is not None,
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
    df  = pd.DataFrame([transaction.model_dump()])
    out = pipeline.predict(df)
    latency_ms = (time.perf_counter() - t0) * 1000

    prob   = float(out["fraud_probability"].iloc[0])
    pred   = int(out["fraud_prediction"].iloc[0])
    txn_id = str(uuid.uuid4())

    # Update stats
    stats["total"] += 1
    stats["fraud"] += pred
    stats["latencies"].append(latency_ms)
    if len(stats["latencies"]) > 5000:
        stats["latencies"].pop(0)

    response = FraudResponse(
        transaction_id    = txn_id,
        fraud_probability = round(prob, 4),
        fraud_prediction  = pred,
        threshold_used    = pipeline.threshold,
        risk_level        = get_risk_level(prob),
        latency_ms        = round(latency_ms, 2),
    )

    # Push to all connected WebSocket clients (React dashboard)
    await manager.broadcast(response.model_dump())

    return response


# ── /metrics ──────────────────────────────────────────────────────────────────
@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics(request: Request):
    stats  = request.app.state.stats
    lats   = stats["latencies"]
    total  = stats["total"]
    fraud  = stats["fraud"]

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
        # Wait for client disconnect — receive() raises WebSocketDisconnect on close
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
