"""Build Feast feature tables and write them to Postgres (offline store).

Loads fraudTrain.csv + fraudTest.csv, runs FeatureEngineering (same as the
production pipeline), splits features into three tables matching the Feast
feature views, and writes them to Postgres.

Run this BEFORE `feast apply` + `feast materialize` to populate the offline store.

Usage:
    python -m src.components.data_ingestion_feast
    python -m src.components.data_ingestion_feast --nrows 5000
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import psycopg2

from src.components.feature_engineering import FeatureEngineering
from src.utils.logger import logging
from src.utils.exception import CustomException

PG_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", 5432)),
    "user": os.environ.get("POSTGRES_USER", "feast"),
    "password": os.environ.get("POSTGRES_PASSWORD", "feast"),
    "dbname": os.environ.get("POSTGRES_DB", "feast"),
}

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


def _read_sparkov(path: str) -> pd.DataFrame:
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
                event_timestamp TIMESTAMP, created TIMESTAMP,
                cc_num BIGINT,
                cc_num_FE FLOAT, txn_last_1h FLOAT, txn_last_24h FLOAT,
                txn_last_168h FLOAT, amt_sum_last_1h FLOAT,
                amt_sum_last_24h FLOAT, amt_sum_last_168h FLOAT,
                amt_per_cc_num_mean FLOAT, amt_per_cc_num_std FLOAT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_detection.merchant_features (
                event_timestamp TIMESTAMP, created TIMESTAMP,
                merchant VARCHAR(255),
                merchant_FE FLOAT, merchant_te FLOAT,
                amt_per_merchant_mean FLOAT, amt_per_merchant_std FLOAT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_detection.transaction_features (
                event_timestamp TIMESTAMP, created TIMESTAMP,
                trans_num VARCHAR(255),
                hour FLOAT, dow FLOAT, month FLOAT, is_night FLOAT,
                age FLOAT, distance_km FLOAT, amt FLOAT, amt_log FLOAT,
                amt_is_round FLOAT, category_FE FLOAT, city_FE FLOAT,
                state_FE FLOAT, job_FE FLOAT, zip_FE FLOAT, category_te FLOAT,
                city_te FLOAT, state_te FLOAT, job_te FLOAT,
                amt_per_category_mean FLOAT, amt_per_category_std FLOAT,
                is_fraud FLOAT
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
        for i in range(0, len(rows), 5000):
            cur.executemany(sql, rows[i:i + 5000])
    conn.commit()


def build_feature_tables(train_path: str = "data/raw/fraudTrain.csv",
                          test_path:  str = "data/raw/fraudTest.csv",
                          nrows: int | None = None) -> dict:
    """Build three feature tables and write to Postgres.

    Returns: dict of {table_name: row_count}
    """
    logging.info("Feast data ingestion started")
    try:
        train_df = _read_sparkov(train_path)
        test_df  = _read_sparkov(test_path)
        if nrows:
            train_df = train_df.iloc[:max(nrows // 2, 1)]
            test_df  = test_df.iloc[:max(nrows // 2, 1)]
        logging.info(f"Loaded train={len(train_df):,}  test={len(test_df):,}")

        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined = combined.sort_values("trans_date_trans_time").reset_index(drop=True)

        fe = FeatureEngineering()
        combined_fe = fe.fit_transform(combined, compute_velocity=True)
        combined_fe["trans_num"] = combined_fe["trans_num"].astype(str)
        combined_fe["is_fraud"]  = combined_fe["is_fraud"].astype(float)
        combined_fe["event_timestamp"] = combined_fe["trans_date_trans_time"]
        combined_fe["created"]         = combined_fe["trans_date_trans_time"]
        logging.info(f"Feature engineering: {len(combined_fe.columns)} columns")

        cc_view = combined_fe[["event_timestamp", "created"] + CC_NUM_COLS].copy()
        cc_view["cc_num"] = cc_view["cc_num"].astype("int64")
        cc_view = cc_view.sort_values("event_timestamp").drop_duplicates("cc_num", keep="last")

        mer_view = combined_fe[["event_timestamp", "created"] + MERCHANT_COLS].copy()
        mer_view["merchant"] = mer_view["merchant"].astype(str)
        mer_view = mer_view.sort_values("event_timestamp").drop_duplicates("merchant", keep="last")

        txn_view = combined_fe[["event_timestamp", "created"] + TRANSACTION_COLS].copy()
        txn_view["trans_num"] = txn_view["trans_num"].astype(str)

        conn = _pg_conn()
        try:
            _create_tables(conn)
            tables = {
                "cc_num_features": cc_view,
                "merchant_features": mer_view,
                "transaction_features": txn_view,
            }
            for name, view in tables.items():
                with conn.cursor() as cur:
                    cur.execute(f"TRUNCATE fraud_detection.{name};")
                conn.commit()
                _write_table(conn, name, view)
                logging.info(f"Wrote {len(view):,} rows to fraud_detection.{name}")
            return {k: len(v) for k, v in tables.items()}
        finally:
            conn.close()
    except Exception as e:
        raise CustomException(e, sys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit rows for testing")
    args = parser.parse_args()
    counts = build_feature_tables(nrows=args.nrows)
    print("Feast Postgres tables ready:")
    for table, n in counts.items():
        print(f"  fraud_detection.{table:<25} {n:>8,} rows")


if __name__ == "__main__":
    main()
