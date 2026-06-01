from datetime import timedelta
import pandas as pd
import numpy as np
from feast import Entity, FeatureView, Field, PushSource, FileSource
from feast.types import Float32
from feast.value_type import ValueType

# ── Entity ─────────────────────────────────────────────────────────
transaction = Entity(
    name="transaction_id",
    description="Unique ID per transaction",
    value_type=ValueType.INT64,
)

# ── Push Source ────────────────────────────────────────────────────
batch_source = FileSource(
    path="feast/feature_repo/data/batch.parquet",
    timestamp_field="Time",
)

fraud_push_source = PushSource(
    name="fraud_push_source",
    batch_source=batch_source,
)

# ── Feature View ───────────────────────────────────────────────────
V_FEATURES = [
    Field(name=f"V{i}", dtype=Float32) for i in range(1, 29)
]

ENG_FEATURES = [
    Field(name="Amount_log", dtype=Float32),
    Field(name="Hour_sin",  dtype=Float32),
    Field(name="Hour_cos",  dtype=Float32),
]

RAW_FEATURES = [
    Field(name="Amount", dtype=Float32),
    Field(name="Time",   dtype=Float32),
]

fraud_features = FeatureView(
    name="fraud_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    source=fraud_push_source,
    schema=V_FEATURES + ENG_FEATURES + RAW_FEATURES,
)

# ── Helper: compute features from raw row ──────────────────────────
def compute_features(row: dict) -> dict:
    """Take a raw transaction dict, return features dict with transaction_id."""
    result = {"transaction_id": row["transaction_id"]}

    # Pass through V features
    for i in range(1, 29):
        result[f"V{i}"] = row.get(f"V{i}", 0.0)

    # Pass through raw fields
    result["Amount"] = row.get("Amount", 0.0)
    result["Time"] = row.get("Time", 0.0)

    # Engineered
    amount = row.get("Amount", 0.0)
    result["Amount_log"] = float(np.log1p(max(amount, 0.0)))

    time_sec = row.get("Time", 0.0)
    hour = (time_sec // 3600) % 24
    result["Hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    result["Hour_cos"] = float(np.cos(2 * np.pi * hour / 24))

    return result


def compute_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Take a raw transaction DataFrame, return features DataFrame."""
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"].clip(lower=0))
    hour = (out["Time"] // 3600) % 24
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out
