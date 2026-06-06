"""Kafka producer for the Sparkov fraud-detection pipeline.

Reads transactions from a Sparkov CSV (fraudTrain.csv or fraudTest.csv) and
publishes them as JSON to the `fraud-transactions` topic. Used by
`kafka/test_100.py` for the demo.

Topics:
    fraud-transactions  — input  (raw Sparkov transactions, JSON-encoded rows)
    fraud-decisions     — output (model predictions + 3-tier action)

Run directly to publish a sample:
    python -m kafka.producer --n 100 --csv data/raw/fraudTest.csv
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

from src.utils.logger import logging
from src.utils.exception import CustomException


DEFAULT_BROKER = "localhost:9092"
DEFAULT_TOPIC  = "fraud-transactions"


def _serialize_value(v):
    """Convert a pandas value to a JSON-serializable Python type."""
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def build_producer(broker: str = DEFAULT_BROKER) -> KafkaProducer:
    """Create a JSON-serializing KafkaProducer with retry on connect."""
    for attempt in range(1, 6):
        try:
            return KafkaProducer(
                bootstrap_servers=broker,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer   =lambda k: (k or "").encode("utf-8") if isinstance(k, str) else k,
                acks="all",
                linger_ms=10,
                retries=3,
            )
        except NoBrokersAvailable:
            wait = min(2 ** attempt, 10)
            logging.warning(f"Kafka not ready (attempt {attempt}/5) — waiting {wait}s")
            time.sleep(wait)
    raise ConnectionError(f"Could not connect to Kafka broker at {broker}")


def publish_csv(producer: KafkaProducer,
                csv_path: str,
                topic: str = DEFAULT_TOPIC,
                n: int = 100,
                fraud_ratio: float = 0.5,
                delay_ms: int = 0,
                seed: int = 42) -> int:
    """Sample n rows from a Sparkov CSV and publish to `topic`.

    Args:
        producer     : KafkaProducer instance
        csv_path     : path to fraudTest.csv or fraudTrain.csv
        topic        : Kafka topic to publish to
        n            : total rows to publish
        fraud_ratio  : fraction of fraud rows in the sample (rest is legit)
        delay_ms     : per-row sleep (simulate streaming; 0 = batch)
        seed         : random seed for reproducibility
    """
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    n_fraud = min(int(n * fraud_ratio), (df.is_fraud == 1).sum())
    n_legit = min(n - n_fraud, (df.is_fraud == 0).sum())
    fraud = df[df.is_fraud == 1].sample(n=n_fraud, random_state=seed)
    legit = df[df.is_fraud == 0].sample(n=n_legit, random_state=seed)
    sample = pd.concat([fraud, legit]).sort_values("trans_date_trans_time").reset_index(drop=True)

    sent = 0
    for _, row in sample.iterrows():
        record = {c: _serialize_value(row[c]) for c in sample.columns}
        future = producer.send(topic, key=record.get("trans_num"), value=record)
        try:
            future.get(timeout=10)
            sent += 1
        except KafkaError as e:
            logging.error(f"Send failed: {e}")
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
    producer.flush()
    return sent


def main():
    parser = argparse.ArgumentParser(description="Sparkov Kafka producer")
    parser.add_argument("--broker", default=os.environ.get("KAFKA_BROKER", DEFAULT_BROKER))
    parser.add_argument("--topic",  default=DEFAULT_TOPIC)
    parser.add_argument("--csv",    default="data/raw/fraudTest.csv")
    parser.add_argument("--n",      type=int, default=100)
    parser.add_argument("--fraud-ratio", type=float, default=0.5)
    parser.add_argument("--delay-ms", type=int, default=0)
    args = parser.parse_args()

    logging.info(f"Producer → {args.broker} topic={args.topic}  n={args.n}")
    producer = build_producer(args.broker)
    sent = publish_csv(
        producer, args.csv, args.topic,
        n=args.n, fraud_ratio=args.fraud_ratio, delay_ms=args.delay_ms,
    )
    logging.info(f"Published {sent}/{args.n} messages")
    producer.close()


if __name__ == "__main__":
    main()
