"""Load Sparkov CSVs into Postgres for training-data access.

Creates a `fraud_detection.raw_transactions` table with the 22 raw
columns from fraudTrain + fraudTest. The training pipeline can read
from this table via SQL instead of CSVs.

Usage:
    python -m src.components.data_ingestion_postgres
    python -m src.components.data_ingestion_postgres --train /path/to/fraudTrain.csv
"""
import argparse
import io
import sys

import pandas as pd
import psycopg2

from src.utils.exception import CustomException
from src.utils.logger import logging

DB_KWARGS = dict(host="localhost", port=5432, user="feast", password="feast", dbname="feast")
TABLE     = "fraud_detection.raw_transactions"
SCHEMA    = """
CREATE SCHEMA IF NOT EXISTS fraud_detection;
DROP TABLE IF EXISTS fraud_detection.raw_transactions;
CREATE TABLE fraud_detection.raw_transactions (
    trans_num              TEXT PRIMARY KEY,
    trans_date_trans_time  TIMESTAMP NOT NULL,
    cc_num                 BIGINT NOT NULL,
    merchant               TEXT,
    category               TEXT,
    amt                    DOUBLE PRECISION,
    first                  TEXT,
    last                   TEXT,
    gender                 TEXT,
    street                 TEXT,
    city                   TEXT,
    state                  TEXT,
    zip                    BIGINT,
    lat                    DOUBLE PRECISION,
    long                   DOUBLE PRECISION,
    city_pop               BIGINT,
    job                    TEXT,
    dob                    DATE,
    merch_lat              DOUBLE PRECISION,
    merch_long             DOUBLE PRECISION,
    is_fraud               INT,
    unix_time              BIGINT
);
CREATE INDEX idx_raw_trans_date ON fraud_detection.raw_transactions (trans_date_trans_time);
CREATE INDEX idx_raw_cc_num     ON fraud_detection.raw_transactions (cc_num);
CREATE INDEX idx_raw_merchant   ON fraud_detection.raw_transactions (merchant);
"""


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    return df


def _copy_df_to_pg(cur, df: pd.DataFrame) -> None:
    """Use Postgres COPY FROM STDIN (fastest path) to load a DataFrame."""
    cols = [
        "trans_num", "trans_date_trans_time", "cc_num", "merchant", "category",
        "amt", "first", "last", "gender", "street", "city", "state", "zip",
        "lat", "long", "city_pop", "job", "dob", "merch_lat", "merch_long",
        "is_fraud", "unix_time",
    ]
    df = df[cols].copy()
    # format dates and NaNs for COPY
    df["trans_date_trans_time"] = df["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["dob"]                    = df["dob"].dt.strftime("%Y-%m-%d")
    df = df.where(pd.notna(df), None)

    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    col_list = ",".join(cols)
    cur.copy_expert(f"COPY {TABLE} ({col_list}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')", buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/fraudTrain.csv")
    parser.add_argument("--test",  default="data/raw/fraudTest.csv")
    args = parser.parse_args()

    logging.info(f"Postgres ingestion: {args.train} + {args.test} → {TABLE}")
    train_df = _read_csv(args.train)
    test_df  = _read_csv(args.test)
    logging.info(f"Loaded train={len(train_df):,}  test={len(test_df):,}")

    try:
        with psycopg2.connect(**DB_KWARGS) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA)
                conn.commit()
                logging.info("Schema + indexes created")

                for label, df in [("fraudTrain", train_df), ("fraudTest", test_df)]:
                    _copy_df_to_pg(cur, df)
                    conn.commit()
                    logging.info(f"{label}: inserted {len(df):,} rows via COPY")

                cur.execute(f"SELECT COUNT(*), SUM(is_fraud) FROM {TABLE}")
                n, fraud = cur.fetchone()
                logging.info(f"Total: {n:,} rows, {fraud:,} fraud ({fraud/n:.4%})")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

