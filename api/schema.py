from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """Raw transaction features — mirrors creditcard.csv schema."""

    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in EUR")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 50000, "Amount": 2847.50,
                "V1": -2.3, "V2": 1.7, "V3": -2.0, "V4": 0.5,
                "V5": -0.9, "V6": -0.3, "V7": -0.7, "V8": 0.1,
                "V9": -0.4, "V10": -2.1, "V11": 1.2, "V12": -2.5,
                "V13": 0.3, "V14": -3.1, "V15": 0.2, "V16": -0.8,
                "V17": -1.9, "V18": -0.5, "V19": 0.1, "V20": 0.2,
                "V21": 0.3, "V22": 0.1, "V23": -0.1, "V24": 0.2,
                "V25": 0.1, "V26": 0.3, "V27": 0.1, "V28": 0.05,
            }
        }


class FraudResponse(BaseModel):
    """Prediction result returned by /predict."""
    transaction_id:    str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    fraud_prediction:  int   = Field(..., ge=0, le=1)
    threshold_used:    float
    risk_level:        str   # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    latency_ms:        float


class MetricsResponse(BaseModel):
    """Aggregate statistics returned by /metrics."""
    total_transactions: int
    total_fraud:        int
    fraud_rate_pct:     float
    avg_latency_ms:     float
    uptime_seconds:     float


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    version:     str = "1.0.0"
