"""End-to-end Kafka demo: 100 transactions through the full pipeline.

Flow:
  1. Producer samples 50 fraud + 50 legit from fraudTest.csv
  2. Publishes to `fraud-transactions` topic
  3. Consumer reads, scores with CatBoost, applies 3-tier
  4. Publishes decisions to `fraud-decisions` topic
  5. This script reads the decisions, computes the confusion matrix,
     prints a 3-tier summary, and exits.

Usage:
    python -m kafka.test_100                       # uses localhost:9092
    python -m kafka.test_100 --broker host:9092    # custom broker
"""
import os
import sys
import time
import json
import argparse
import threading
from collections import Counter

import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from sklearn.metrics import confusion_matrix, f1_score

from kf.producer import build_producer, publish_csv
from src.utils.logger import logging


DEFAULT_BROKER     = "localhost:9092"
IN_TOPIC           = "fraud-transactions"
OUT_TOPIC          = "fraud-decisions"
CONSUMER_GROUP     = "fraud-demo-runner"


def consume_decisions(broker: str, expected: int, timeout_s: int = 60) -> list:
    """Read `expected` messages from `fraud-decisions` with a timeout."""
    consumer = KafkaConsumer(
        OUT_TOPIC,
        bootstrap_servers=broker,
        group_id=f"{CONSUMER_GROUP}-{int(time.time())}",  # fresh group per run
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=2000,
    )
    decisions = []
    deadline = time.time() + timeout_s
    while len(decisions) < expected and time.time() < deadline:
        records = consumer.poll(timeout_ms=1000, max_records=expected - len(decisions))
        for tp, batch in records.items():
            decisions.extend([r.value for r in batch])
        if not records:
            # nudge the broker
            time.sleep(0.1)
    consumer.close()
    return decisions


def main():
    parser = argparse.ArgumentParser(description="Run the 100-test Kafka demo")
    parser.add_argument("--broker",    default=os.environ.get("KAFKA_BROKER", DEFAULT_BROKER))
    parser.add_argument("--csv",       default="data/raw/fraudTest.csv")
    parser.add_argument("--n-fraud",   type=int, default=50)
    parser.add_argument("--n-legit",   type=int, default=50)
    parser.add_argument("--timeout-s", type=int, default=60)
    args = parser.parse_args()

    n_total = args.n_fraud + args.n_legit

    # ── Start the consumer in a background thread ────────────────────────────
    # The consumer also needs to be running so the broker has a group to
    # register the topic; we use kafka.consumer.run via a thread.
    from kf.consumer import run
    consumer_thread = threading.Thread(
        target=run,
        kwargs=dict(
            broker=args.broker,
            in_topic=IN_TOPIC,
            out_topic=OUT_TOPIC,
            group_id=f"fraud-test-{int(time.time())}",
            max_messages=n_total,
        ),
        daemon=True,
    )
    consumer_thread.start()

    # Give the consumer time to subscribe + join its group
    logging.info("Waiting 3s for consumer to subscribe...")
    time.sleep(3)

    # ── Produce 100 transactions ─────────────────────────────────────────────
    logging.info(f"Producing {n_total} transactions → {IN_TOPIC}")
    producer = build_producer(args.broker)
    sent = publish_csv(
        producer, args.csv, IN_TOPIC,
        n=n_total, fraud_ratio=args.n_fraud / n_total,
    )
    producer.close()
    logging.info(f"Published {sent} messages")

    # ── Wait for the consumer to process them ────────────────────────────────
    logging.info("Waiting for consumer to process + publish decisions...")
    decisions = consume_decisions(args.broker, expected=n_total, timeout_s=args.timeout_s)

    consumer_thread.join(timeout=10)

    if len(decisions) != n_total:
        logging.warning(f"Expected {n_total} decisions, got {len(decisions)}")

    # ── 3-tier summary + confusion matrix ────────────────────────────────────
    y_true = [d.get("is_fraud_ground_truth", -1) for d in decisions]
    y_pred = [1 if d["tier"] >= 1 else 0 for d in decisions]
    tiers  = Counter(d["tier"] for d in decisions)

    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    tn, fp = cm[0]; fn, tp = cm[1]

    print()
    print("=" * 60)
    print(f"  100-TEST KAFKA DEMO RESULTS")
    print("=" * 60)
    print(f"  Produced         : {sent}")
    print(f"  Decisions received: {len(decisions)}")
    print(f"  Tier distribution: T0={tiers[0]} T1={tiers[1]} T2={tiers[2]} T3={tiers[3]}")
    print(f"  Confusion [[TN,FP],[FN,TP]]: {cm}")
    print(f"  Macro F1         : {f1m:.4f}")
    print("=" * 60)

    return decisions


if __name__ == "__main__":
    main()
