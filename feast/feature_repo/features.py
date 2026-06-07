"""Feast feature definitions for the Sparkov fraud-detection project.

Three entities (card, merchant, transaction) and three feature views.
Features are computed in src/components/feature_engineering.py and
written to Postgres tables by src/components/data_ingestion_feast.py.

Offline store (Postgres) is read by training via Feast's
`get_historical_features()`. Online store (Redis) is read by the
prediction pipeline / Kafka consumer via `get_online_features()`.
"""
from datetime import timedelta
import pandas as pd
import numpy as np

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


# ── Entities ────────────────────────────────────────────────────────────────
cc_num = Entity(
    name="cc_num",
    description="Card number (Sparkov hashed card ID, INT64)",
    value_type=ValueType.INT64,
)

merchant = Entity(
    name="merchant",
    description="Merchant name (Sparkov merchant string)",
    value_type=ValueType.STRING,
)

transaction_id = Entity(
    name="trans_num",
    description="Unique transaction ID (Sparkov trans_num string)",
    value_type=ValueType.STRING,
)


# ── Data sources (point at the Postgres tables we'll create) ────────────────
# Feast requires a FileSource as a stand-in for the offline store table.
# At apply-time, Feast uses this to discover the schema; at query-time it
# rewrites the path to the configured offline store. The CSV paths below
# are also used by the ingestion script to populate the Postgres tables.
cc_num_source = FileSource(
    name="cc_num_source",
    path="/home/NullbitZer0/projects/real-time-fraud-detection/data/sparkov_cc_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

merchant_source = FileSource(
    name="merchant_source",
    path="/home/NullbitZer0/projects/real-time-fraud-detection/data/sparkov_merchant_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

transaction_source = FileSource(
    name="transaction_source",
    path="/home/NullbitZer0/projects/real-time-fraud-detection/data/sparkov_transaction_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)


# ── Feature View: per-card aggregates ───────────────────────────────────────
cc_num_features = FeatureView(
    name="cc_num_features",
    entities=[cc_num],
    ttl=timedelta(days=365),
    source=cc_num_source,
    schema=[
        Field(name="cc_num_FE",            dtype=Float32),
        Field(name="txn_last_1h",          dtype=Float32),
        Field(name="txn_last_24h",         dtype=Float32),
        Field(name="txn_last_168h",        dtype=Float32),
        Field(name="amt_sum_last_1h",      dtype=Float32),
        Field(name="amt_sum_last_24h",     dtype=Float32),
        Field(name="amt_sum_last_168h",    dtype=Float32),
        Field(name="amt_per_cc_num_mean",  dtype=Float32),
        Field(name="amt_per_cc_num_std",   dtype=Float32),
    ],
    online=True,
)


# ── Feature View: per-merchant aggregates ───────────────────────────────────
merchant_features = FeatureView(
    name="merchant_features",
    entities=[merchant],
    ttl=timedelta(days=365),
    source=merchant_source,
    schema=[
        Field(name="merchant_FE",             dtype=Float32),
        Field(name="merchant_te",             dtype=Float32),
        Field(name="amt_per_merchant_mean",   dtype=Float32),
        Field(name="amt_per_merchant_std",    dtype=Float32),
    ],
    online=True,
)


# ── Feature View: per-transaction features ──────────────────────────────────
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction_id],
    ttl=timedelta(days=365),
    source=transaction_source,
    schema=[
        # Time / location
        Field(name="hour",         dtype=Float32),
        Field(name="dow",          dtype=Float32),
        Field(name="month",        dtype=Float32),
        Field(name="is_night",     dtype=Float32),
        Field(name="age",          dtype=Float32),
        Field(name="distance_km",  dtype=Float32),
        # Amount
        Field(name="amt",          dtype=Float32),
        Field(name="amt_log",      dtype=Float32),
        Field(name="amt_is_round", dtype=Float32),
        # Frequency encodings
        Field(name="category_FE",  dtype=Float32),
        Field(name="city_FE",      dtype=Float32),
        Field(name="state_FE",     dtype=Float32),
        Field(name="job_FE",       dtype=Float32),
        Field(name="zip_FE",       dtype=Float32),
        # Target encodings
        Field(name="category_te",  dtype=Float32),
        Field(name="city_te",      dtype=Float32),
        Field(name="state_te",     dtype=Float32),
        Field(name="job_te",       dtype=Float32),
        # Amount stats
        Field(name="amt_per_category_mean",  dtype=Float32),
        Field(name="amt_per_category_std",   dtype=Float32),
        # Label (only for training — pulled from fraudTrain.csv directly)
        Field(name="is_fraud",     dtype=Float32),
    ],
    # transaction_features is per-transaction (trans_num is unique per row),
    # so there's no point materializing it to Redis. Training reads it via
    # get_historical_features from the offline store (Postgres).
    online=False,
)
