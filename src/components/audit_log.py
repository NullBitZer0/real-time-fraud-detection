"""Postgres audit log for /predict decisions.

Writes every prediction to a `fraud_detection.decision_log` table — used for:
- Compliance / regulator audits
- Post-hoc model performance analysis
- Drift detection (compare predicted tier distribution vs labels in batch)

Schema (one row per /predict call):
    trans_num          TEXT
    fraud_probability  DOUBLE PRECISION
    tier               SMALLINT
    action             TEXT
    threshold_used     DOUBLE PRECISION
    is_fraud_ground_truth SMALLINT NULL  -- backfilled by the 100-test demo
    model_run_id       TEXT
    model_version      TEXT
    latency_ms         DOUBLE PRECISION
    ingested_at        TIMESTAMPTZ DEFAULT now()
    request_ip         TEXT
    user_agent         TEXT

Connection uses the same env vars as the rest of the project:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
"""
import os
import sys
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import execute_values

from src.utils.exception import CustomException
from src.utils.logger import logging

DDL = """
CREATE TABLE IF NOT EXISTS fraud_detection.decision_log (
    id                   BIGSERIAL PRIMARY KEY,
    trans_num            TEXT                     NOT NULL,
    fraud_probability    DOUBLE PRECISION         NOT NULL,
    tier                 SMALLINT                 NOT NULL,
    action               TEXT                     NOT NULL,
    threshold_used       DOUBLE PRECISION         NOT NULL,
    is_fraud_ground_truth SMALLINT,
    model_run_id         TEXT,
    model_version        TEXT,
    latency_ms           DOUBLE PRECISION,
    ingested_at          TIMESTAMPTZ DEFAULT now() NOT NULL,
    request_ip           TEXT,
    user_agent           TEXT
);
CREATE INDEX IF NOT EXISTS decision_log_trans_num_idx
    ON fraud_detection.decision_log (trans_num);
CREATE INDEX IF NOT EXISTS decision_log_ingested_at_idx
    ON fraud_detection.decision_log (ingested_at DESC);
"""


def _conn_str() -> str:
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', 5432)} "
        f"user={os.environ.get('POSTGRES_USER', 'feast')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'feast')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'feast')}"
    )


@contextmanager
def _connect():
    conn = psycopg2.connect(_conn_str())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_table() -> None:
    """Create the decision_log table if it doesn't exist."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(DDL)
        logging.info("audit_log: decision_log table ensured")
    except Exception as e:
        logging.warning(f"audit_log: ensure_table failed ({e})")
        raise CustomException(e, sys)


def insert_decision(*, trans_num: str, proba: float, tier: int, action: str,
                     threshold: float, model_run_id: str = None,
                     model_version: str = None, latency_ms: float = None,
                     request_ip: str = None, user_agent: str = None) -> None:
    """Insert a single /predict decision. Fails silently if Postgres is down."""
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO fraud_detection.decision_log
                        (trans_num, fraud_probability, tier, action, threshold_used,
                         model_run_id, model_version, latency_ms, request_ip, user_agent)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (trans_num, proba, tier, action, threshold,
                     model_run_id, model_version, latency_ms, request_ip, user_agent),
                )
    except Exception as e:
        # Don't fail /predict just because the audit log is down
        logging.warning(f"audit_log: insert_decision failed for {trans_num} ({e})")


def insert_batch(rows: list[dict]) -> int:
    """Bulk-insert a list of decision dicts. Returns the row count."""
    if not rows:
        return 0
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO fraud_detection.decision_log
                        (trans_num, fraud_probability, tier, action, threshold_used,
                         model_run_id, model_version, latency_ms)
                    VALUES %s
                    """,
                    [(
                        r["trans_num"], r["proba"], r["tier"], r["action"],
                        r["threshold"], r.get("model_run_id"),
                        r.get("model_version"), r.get("latency_ms"),
                    ) for r in rows],
                )
        logging.info(f"audit_log: bulk-inserted {len(rows)} decisions")
        return len(rows)
    except Exception as e:
        logging.warning(f"audit_log: insert_batch failed ({e})")
        return 0


def backfill_ground_truth() -> int:
    """Match the decision_log against raw_transactions.is_fraud and backfill.

    Useful after the 100-test demo: every demo row has a trans_num that
    exists in raw_transactions, so we can backfill the is_fraud column
    for offline analysis.
    """
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE fraud_detection.decision_log d
                    SET    is_fraud_ground_truth = r.is_fraud
                    FROM   fraud_detection.raw_transactions r
                    WHERE  d.trans_num = r.trans_num
                      AND  d.is_fraud_ground_truth IS NULL
                """)
                n = cur.rowcount
        logging.info(f"audit_log: backfilled ground truth for {n} rows")
        return n
    except Exception as e:
        logging.warning(f"audit_log: backfill_ground_truth failed ({e})")
        return 0


if __name__ == "__main__":
    ensure_table()
    backfill_ground_truth()
