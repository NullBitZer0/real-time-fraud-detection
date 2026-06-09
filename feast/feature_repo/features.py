"""Feast feature definitions for the Sparkov fraud-detection project.

Three entities (card, merchant, transaction) and three feature views.
Features match the FeatureEngineering pipeline in src/components/feature_engineering.py.

Offline store (Postgres) feeds training via get_historical_features().
Online store (Redis) feeds inference via get_online_features().

Usage:
    feast apply        # register definitions + validate schemas
    feast materialize  # copy offline -> online (Postgres -> Redis)
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Int64, String


# ── Entities ──────────────────────────────────────────────────────────────────
cc_num = Entity(
    name="cc_num",
    description="Card number (Sparkov hashed card ID, INT64)",
    join_keys=["cc_num"],
)

merchant = Entity(
    name="merchant",
    description="Merchant name (Sparkov merchant string)",
    join_keys=["merchant"],
)

trans_num = Entity(
    name="trans_num",
    description="Unique transaction ID (Sparkov trans_num string)",
    join_keys=["trans_num"],
)


# ── Data sources (Postgres tables populated by seed.py) ──────────────────────
cc_num_source = PostgreSQLSource(
    name="cc_num_source",
    table="fraud_detection.cc_num_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

merchant_source = PostgreSQLSource(
    name="merchant_source",
    table="fraud_detection.merchant_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

transaction_source = PostgreSQLSource(
    name="transaction_source",
    table="fraud_detection.transaction_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)


# ── Feature View: per-card aggregates (online=True → Redis) ──────────────────
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

# ── Feature View: per-merchant aggregates (online=True → Redis) ──────────────
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

# ── Feature View: per-transaction features (online=False → Postgres only) ────
# These are one-per-transaction so there's no need to cache them in Redis.
# Training reads them via get_historical_features() from Postgres.
transaction_features = FeatureView(
    name="transaction_features",
    entities=[trans_num],
    ttl=timedelta(days=365),
    source=transaction_source,
    schema=[
        Field(name="hour",               dtype=Float32),
        Field(name="dow",                dtype=Float32),
        Field(name="month",              dtype=Float32),
        Field(name="is_night",           dtype=Float32),
        Field(name="age",                dtype=Float32),
        Field(name="distance_km",         dtype=Float32),
        Field(name="amt",                dtype=Float32),
        Field(name="amt_log",            dtype=Float32),
        Field(name="amt_is_round",       dtype=Float32),
        Field(name="category_FE",        dtype=Float32),
        Field(name="city_FE",            dtype=Float32),
        Field(name="state_FE",           dtype=Float32),
        Field(name="job_FE",             dtype=Float32),
        Field(name="zip_FE",             dtype=Float32),
        Field(name="category_te",        dtype=Float32),
        Field(name="city_te",            dtype=Float32),
        Field(name="state_te",           dtype=Float32),
        Field(name="job_te",             dtype=Float32),
        Field(name="amt_per_category_mean", dtype=Float32),
        Field(name="amt_per_category_std",  dtype=Float32),
        Field(name="is_fraud",           dtype=Float32),
    ],
    online=False,
)
