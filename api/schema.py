"""Pydantic schemas for the Sparkov fraud-detection API.

Two request shapes are supported:
  - TransactionRequest : single Sparkov transaction → /predict
  - Run100TestsRequest : no body, just trigger the 100-test demo run

Two response shapes for /predict and /demo/run-100-tests.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Single-transaction prediction ─────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """Single Sparkov transaction — the 14 features the model needs."""

    trans_date_trans_time: datetime = Field(..., description="Transaction timestamp")
    cc_num:                float   = Field(..., description="Card number (hashed)")
    merchant:              str     = Field(..., description="Merchant name")
    category:              str     = Field(..., description="Merchant category")
    amt:                   float   = Field(..., ge=0, description="Amount in USD")
    lat:                   float
    long:                  float
    merch_lat:             float
    merch_long:            float
    dob:                   str     = Field(..., description="Cardholder DOB (YYYY-MM-DD)")
    city:                  str
    state:                 str
    job:                   str
    zip:                   int

    class Config:
        json_schema_extra = {
            "example": {
                "trans_date_trans_time": "2019-01-01 00:00:18",
                "cc_num": 4.042200e+17,
                "merchant": "Hudson, Davis and Copeland",
                "category": "grocery_pos",
                "amt": 4.07,
                "lat": 39.9658, "long": -82.9743,
                "merch_lat": 39.8856, "merch_long": -82.9393,
                "dob": "1977-08-12",
                "city": "Columbus", "state": "OH",
                "job": "Public attorney", "zip": 43201,
            }
        }


class FraudResponse(BaseModel):
    """Returned by /predict."""
    transaction_id:    str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    fraud_prediction:  int   = Field(..., ge=0, le=1)
    tier:              int   = Field(..., ge=0, le=3)
    action:            str   # "approve" | "soft_signal" | "review_queue" | "auto_block"
    threshold_used:    float
    latency_ms:        float


# ── 100-test demo run ─────────────────────────────────────────────────────────
class Run100TestsRequest(BaseModel):
    """Body for POST /demo/run-100-tests — no fields, just triggers the run."""
    n_fraud: int = Field(50, ge=0, le=100, description="How many fraud rows to include (0-100)")
    n_legit: int = Field(50, ge=0, le=100, description="How many legit rows to include (0-100)")


class SingleTestResult(BaseModel):
    trans_num:         str
    is_fraud:          int
    fraud_probability: float
    fraud_prediction:  int
    tier:              int
    action:            str
    amt:               float
    merchant:          str
    category:          str


class Run100TestsResponse(BaseModel):
    n_total:    int
    n_fraud:    int
    n_legit:    int
    tier1_count: int   # auto_block
    tier2_count: int   # review_queue
    tier3_count: int   # soft_signal
    tier0_count: int   # approve
    tier_thresholds: dict
    confusion_matrix: List[List[int]]   # [[TN, FP], [FN, TP]]
    macro_f1:        float
    results:         List[SingleTestResult]


# ── System endpoints ──────────────────────────────────────────────────────────
class MetricsResponse(BaseModel):
    total_transactions: int
    total_fraud:        int
    fraud_rate_pct:     float
    avg_latency_ms:     float
    uptime_seconds:     float


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    model_name:  Optional[str] = None
    version:     str = "2.0.0"   # Sparkov + CatBoost + 3-tier


class AuditRow(BaseModel):
    """One row from fraud_detection.decision_log — exposed via /audit/recent."""
    id:                    int
    trans_num:             str
    fraud_probability:     float
    tier:                  int
    action:                str
    threshold_used:        float
    is_fraud_ground_truth: Optional[int]   = None
    model_version:         Optional[str]   = None
    latency_ms:            Optional[float] = None
    ingested_at:           datetime
    request_ip:            Optional[str]   = None
    user_agent:            Optional[str]   = None
