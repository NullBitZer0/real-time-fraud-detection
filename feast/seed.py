"""Seed Feast offline store (Postgres) + online store (Redis) with Sparkov features.

Steps:
  1. Load fraudTrain.csv + fraudTest.csv
  2. Run FeatureEngineering (same pipeline as training)
  3. Build three feature tables: cc_num / merchant / transaction
  4. Write each table to Postgres (offline store)
  5. `feast apply` — register definitions
  6. `feast materialize` — copy offline -> online (Postgres -> Redis)
  7. Verify by fetching from online store

Usage:
    python feast/seed.py                     # full dataset
    python feast/seed.py --nrows 5000        # small test
    python feast/seed.py --skip-materialize  # table-only, no Redis
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

# Ensure project root is on sys.path for src.* imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEAST_REPO = os.environ.get(
    "FEAST_REPO_PATH",
    os.path.join(os.path.dirname(__file__), "feature_repo"),
)

PG_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", 5432)),
    "user": os.environ.get("POSTGRES_USER", "feast"),
    "password": os.environ.get("POSTGRES_PASSWORD", "feast"),
    "dbname": os.environ.get("POSTGRES_DB", "feast"),
}

# Column groups matching the three feature views
CC_NUM_COLS = [
    "cc_num", "cc_num_FE", "txn_last_1h", "txn_last_24h", "txn_last_168h",
    "amt_sum_last_1h", "amt_sum_last_24h", "amt_sum_last_168h",
    "amt_per_cc_num_mean", "amt_per_cc_num_std",
]
MERCHANT_COLS = [
    "merchant", "merchant_FE", "merchant_te",
    "amt_per_merchant_mean", "amt_per_merchant_std",
]
TRANSACTION_COLS = [
    "trans_num", "hour", "dow", "month", "is_night", "age", "distance_km",
    "amt", "amt_log", "amt_is_round",
    "category_FE", "city_FE", "state_FE", "job_FE", "zip_FE",
    "category_te", "city_te", "state_te", "job_te",
    "amt_per_category_mean", "amt_per_category_std",
    "is_fraud",
]


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def _pg_conn():
    return psycopg2.connect(**PG_CONFIG)


def _create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS fraud_detection;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_detection.cc_num_features (
                event_timestamp TIMESTAMP,
                created         TIMESTAMP,
                cc_num          BIGINT,
                cc_num_FE       FLOAT,
                txn_last_1h     FLOAT,
                txn_last_24h    FLOAT,
                txn_last_168h   FLOAT,
                amt_sum_last_1h FLOAT,
                amt_sum_last_24h FLOAT,
                amt_sum_last_168h FLOAT,
                amt_per_cc_num_mean FLOAT,
                amt_per_cc_num_std  FLOAT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_detection.merchant_features (
                event_timestamp TIMESTAMP,
                created         TIMESTAMP,
                merchant        VARCHAR(255),
                merchant_FE     FLOAT,
                merchant_te     FLOAT,
                amt_per_merchant_mean FLOAT,
                amt_per_merchant_std  FLOAT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_detection.transaction_features (
                event_timestamp TIMESTAMP,
                created         TIMESTAMP,
                trans_num       VARCHAR(255),
                hour            FLOAT,
                dow             FLOAT,
                month           FLOAT,
                is_night        FLOAT,
                age             FLOAT,
                distance_km     FLOAT,
                amt             FLOAT,
                amt_log         FLOAT,
                amt_is_round    FLOAT,
                category_FE     FLOAT,
                city_FE         FLOAT,
                state_FE        FLOAT,
                job_FE          FLOAT,
                zip_FE          FLOAT,
                category_te     FLOAT,
                city_te         FLOAT,
                state_te        FLOAT,
                job_te          FLOAT,
                amt_per_category_mean FLOAT,
                amt_per_category_std  FLOAT,
                is_fraud        FLOAT
            );
        """)
    conn.commit()


def _write_table(conn, table: str, df: pd.DataFrame):
    if df.empty:
        return
    cols = list(df.columns)
    placeholders = ",".join(["%s"] * len(cols))
    col_names = ",".join(cols)
    sql = f"INSERT INTO fraud_detection.{table} ({col_names}) VALUES ({placeholders})"
    rows = [tuple(None if pd.isna(v) else v for v in row) for row in df[cols].to_numpy()]
    with conn.cursor() as cur:
        for batch_start in range(0, len(rows), 5000):
            batch = rows[batch_start:batch_start + 5000]
            cur.executemany(sql, batch)
    conn.commit()


def seed(nrows: int | None = None, skip_materialize: bool = False):
    sys.path.insert(0, FEAST_REPO)
    from feast import FeatureStore
    from features import (
        cc_num, merchant, trans_num,
        cc_num_features, merchant_features, transaction_features,
    )
    from src.components.feature_engineering import FeatureEngineering
    from src.utils.logger import logging

    logging.info("=" * 60)
    logging.info("Feast seed — Sparkov fraud detection")
    logging.info("=" * 60)

    try:
        # ── 1. Load CSV ──────────────────────────────────────────────────
        train_path = "data/raw/fraudTrain.csv"
        test_path  = "data/raw/fraudTest.csv"
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logging.error(f"Sparkov CSVs not found at {train_path} / {test_path}")
            return

        train_df = _read_csv(train_path)
        test_df  = _read_csv(test_path)
        if nrows:
            train_df = train_df.iloc[:max(nrows // 2, 1)]
            test_df  = test_df.iloc[:max(nrows // 2, 1)]
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined = combined.sort_values("trans_date_trans_time").reset_index(drop=True)
        logging.info(f"Loaded train={len(train_df):,}  test={len(test_df):,}  combined={len(combined):,}")

        # ── 2. Compute features ──────────────────────────────────────────
        fe = FeatureEngineering()
        features_df = fe.fit_transform(combined, compute_velocity=True)
        features_df["trans_num"] = features_df["trans_num"].astype(str)
        features_df["is_fraud"]  = features_df["is_fraud"].astype(float)
        logging.info(f"Computed {len(features_df.columns)} feature columns over {len(features_df)} rows")

        features_df["event_timestamp"] = features_df["trans_date_trans_time"]
        features_df["created"]         = features_df["trans_date_trans_time"]

        # ── 3. Build three feature tables ────────────────────────────────
        cc_view = features_df[["event_timestamp", "created"] + CC_NUM_COLS].copy()
        cc_view["cc_num"] = cc_view["cc_num"].astype("int64")
        cc_view = cc_view.sort_values("event_timestamp").drop_duplicates("cc_num", keep="last")

        mer_view = features_df[["event_timestamp", "created"] + MERCHANT_COLS].copy()
        mer_view["merchant"] = mer_view["merchant"].astype(str)
        mer_view = mer_view.sort_values("event_timestamp").drop_duplicates("merchant", keep="last")

        txn_view = features_df[["event_timestamp", "created"] + TRANSACTION_COLS].copy()
        txn_view["trans_num"] = txn_view["trans_num"].astype(str)

        logging.info(f"cc_num_features: {len(cc_view):,}  merchant_features: {len(mer_view):,}  transaction_features: {len(txn_view):,}")

        # ── 4. Write to Postgres ─────────────────────────────────────────
        logging.info("Writing to Postgres...")
        conn = _pg_conn()
        try:
            _create_tables(conn)
            for table, view in [("cc_num_features", cc_view),
                                 ("merchant_features", mer_view),
                                 ("transaction_features", txn_view)]:
                with conn.cursor() as cur:
                    cur.execute(f"TRUNCATE fraud_detection.{table};")
                conn.commit()
                _write_table(conn, table, view)
                logging.info(f"  fraud_detection.{table}: {len(view):,} rows")
        finally:
            conn.close()

        # ── 5. feast apply ───────────────────────────────────────────────
        logging.info("Running feast apply...")
        store = FeatureStore(repo_path=FEAST_REPO)
        store.apply([cc_num, merchant, trans_num,
                      cc_num_features, merchant_features, transaction_features])
        logging.info("feast apply complete")

        # ── 6. feast materialize ─────────────────────────────────────────
        if skip_materialize:
            logging.info("Skipping materialize (--skip-materialize)")
        else:
            logging.info("Running feast materialize (Postgres -> Redis)...")
            start_dt = features_df["event_timestamp"].min().to_pydatetime()
            end_dt   = features_df["event_timestamp"].max().to_pydatetime()
            store.materialize(start_dt, end_dt)
            logging.info(f"Materialized {start_dt} -> {end_dt}")

        # ── 7. Verify ────────────────────────────────────────────────────
        first_cc  = int(cc_view.iloc[0]["cc_num"])
        first_mer = str(mer_view.iloc[0]["merchant"])
        entity_rows = [{"cc_num": first_cc, "merchant": first_mer}]
        feature_refs = [
            "cc_num_features:txn_last_1h",
            "cc_num_features:amt_per_cc_num_mean",
            "merchant_features:merchant_FE",
            "merchant_features:amt_per_merchant_mean",
        ]
        response = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )
        online = response.to_dict()
        for k, v in online.items():
            logging.info(f"  {k}: {v}")
        logging.info("Feast seed complete ✓")

    except Exception as e:
        logging.exception(f"Feast seed failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Feast stores with Sparkov features")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--skip-materialize", action="store_true",
                        help="Skip materialization to Redis (offline-only)")
    args = parser.parse_args()
    seed(nrows=args.nrows, skip_materialize=args.skip_materialize)
