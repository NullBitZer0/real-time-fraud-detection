import uuid
import time
import asyncio
import pandas as pd

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from api.schema import (
    TransactionRequest,
    FraudResponse,
    MetricsResponse,
    HealthResponse,
)

router = APIRouter()


def get_risk_level(prob: float) -> str:
    if prob < 0.3:   return "LOW"
    if prob < 0.5:   return "MEDIUM"
    if prob < 0.75:  return "HIGH"
    return "CRITICAL"


# ── /health ───────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request):
    pipeline = request.app.state.pipeline
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
    )


# ── /predict ──────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=FraudResponse, tags=["Inference"])
async def predict(transaction: TransactionRequest, request):
    pipeline = request.app.state.pipeline
    stats    = request.app.state.stats

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    df    = pd.DataFrame([transaction.model_dump()])
    result = pipeline.predict(df)

    latency_ms = (time.perf_counter() - t0) * 1000
    prob       = float(result["fraud_probability"].iloc[0])
    pred       = int(result["fraud_prediction"].iloc[0])
    txn_id     = str(uuid.uuid4())

    # Update live stats
    stats["total"] += 1
    stats["fraud"] += pred
    stats["latencies"].append(latency_ms)
    if len(stats["latencies"]) > 1000:
        stats["latencies"].pop(0)

    response = FraudResponse(
        transaction_id    = txn_id,
        fraud_probability = round(prob, 4),
        fraud_prediction  = pred,
        threshold_used    = pipeline.threshold,
        risk_level        = get_risk_level(prob),
        latency_ms        = round(latency_ms, 2),
    )

    # Broadcast to all WebSocket clients
    manager = request.app.state.ws_manager
    await manager.broadcast(response.model_dump())

    return response


# ── /metrics ──────────────────────────────────────────────────────────────────
@router.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics(request):
    stats   = request.app.state.stats
    uptime  = time.time() - request.app.state.start_time
    lats    = stats["latencies"]
    avg_lat = round(sum(lats) / len(lats), 2) if lats else 0.0
    total   = stats["total"]
    fraud   = stats["fraud"]

    return MetricsResponse(
        total_transactions = total,
        total_fraud        = fraud,
        fraud_rate_pct     = round(fraud / total * 100, 3) if total else 0.0,
        avg_latency_ms     = avg_lat,
        uptime_seconds     = round(uptime, 1),
    )


# ── /ws ───────────────────────────────────────────────────────────────────────
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, request):
    manager = websocket.app.state.ws_manager
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive — actual data pushed from /predict
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
