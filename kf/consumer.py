"""Kafka consumer for the Sparkov fraud-detection pipeline.

Reads JSON-encoded Sparkov transactions from the `fraud-transactions` topic,
scores each with the trained CatBoost model + 3-tier, and publishes the
result to the `fraud-decisions` topic.

Topics:
    fraud-transactions  — input  (raw Sparkov transactions)
    fraud-decisions     — output (model predictions + 3-tier action)

Run directly:
    python -m kafka.consumer
    python -m kafka.consumer --broker localhost:9092 --timeout-s 30
"""
import os
import sys
import json
import time
import signal
import threading
import argparse
import pandas as pd

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from pipelines.prediction_pipeline import PredictionPipeline
from src.utils.logger import logging
from src.utils.exception import CustomException


DEFAULT_BROKER        = "localhost:9092"
DEFAULT_IN_TOPIC      = "fraud-transactions"
DEFAULT_OUT_TOPIC     = "fraud-decisions"
DEFAULT_GROUP_ID      = "fraud-consumer-v1"
SHUTDOWN_FLAG         = {"stop": False}


def _build_consumer(broker: str, in_topic: str, group_id: str) -> KafkaConsumer:
    """Create a KafkaConsumer with retry on connect."""
    for attempt in range(1, 6):
        try:
            return KafkaConsumer(
                in_topic,
                bootstrap_servers=broker,
                group_id=group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                key_deserializer   =lambda b: b.decode("utf-8") if b else None,
                consumer_timeout_ms=1000,  # poll loop checks shutdown flag between batches
            )
        except NoBrokersAvailable:
            wait = min(2 ** attempt, 10)
            logging.warning(f"Kafka not ready (attempt {attempt}/5) — waiting {wait}s")
            time.sleep(wait)
    raise ConnectionError(f"Could not connect to Kafka broker at {broker}")


def _build_producer(broker: str) -> KafkaProducer:
    for attempt in range(1, 6):
        try:
            return KafkaProducer(
                bootstrap_servers=broker,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer   =lambda k: (k or "").encode("utf-8") if isinstance(k, str) else k,
                acks="all",
                linger_ms=10,
            )
        except NoBrokersAvailable:
            time.sleep(min(2 ** attempt, 10))
    raise ConnectionError(f"Could not connect to Kafka broker at {broker}")


def _install_signal_handlers():
    """Install SIGINT/SIGTERM handlers. Only works in the main thread;
    skip silently when called from a background thread (e.g. test_100)."""
    if threading.current_thread() is not threading.main_thread():
        return
    def _handler(sig, frame):
        logging.info(f"Received signal {sig} — shutting down consumer")
        SHUTDOWN_FLAG["stop"] = True
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


def run(broker: str = DEFAULT_BROKER,
        in_topic: str = DEFAULT_IN_TOPIC,
        out_topic: str = DEFAULT_OUT_TOPIC,
        group_id: str = DEFAULT_GROUP_ID,
        max_messages: int = 0):
    """Consume, score, and publish. Set max_messages > 0 to exit after N messages."""
    _install_signal_handlers()
    logging.info("Loading prediction pipeline...")
    pipeline = PredictionPipeline()

    consumer = _build_consumer(broker, in_topic, group_id)
    producer = _build_producer(broker)
    logging.info(f"Consumer ready: {broker} ← {in_topic} → {out_topic}")

    n_processed = 0
    try:
        while not SHUTDOWN_FLAG["stop"]:
            batch = consumer.poll(timeout_ms=500, max_records=100)
            if not batch:
                continue
            for tp, records in batch.items():
                if not records:
                    continue
                # Batch-scoring is much faster than per-row
                df = pd.DataFrame([r.value for r in records])
                df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
                out = pipeline.predict(df)

                for src, pred in zip(records, out.to_dict(orient="records")):
                    decision = {
                        "trans_num":         pred["trans_num"],
                        "proba":             float(pred["proba"]),
                        "tier":              int  (pred["tier"]),
                        "action":            str  (pred["action"]),
                        "threshold":         float(pred["threshold"]),
                        "is_fraud_ground_truth": int(src.value.get("is_fraud", -1)),
                    }
                    producer.send(out_topic, key=decision["trans_num"], value=decision)
                    n_processed += 1
                producer.flush()
                logging.info(
                    f"Batch {len(records)} from {tp.topic}-{tp.partition} "
                    f"@ offsets {records[0].offset}..{records[-1].offset} | "
                    f"total={n_processed}"
                )
            if max_messages and n_processed >= max_messages:
                logging.info(f"Reached max_messages={max_messages} — exiting")
                break
    finally:
        consumer.close()
        producer.close()
        logging.info(f"Consumer stopped. Processed {n_processed} messages.")


def main():
    parser = argparse.ArgumentParser(description="Sparkov Kafka consumer")
    parser.add_argument("--broker",       default=os.environ.get("KAFKA_BROKER", DEFAULT_BROKER))
    parser.add_argument("--in-topic",     default=DEFAULT_IN_TOPIC)
    parser.add_argument("--out-topic",    default=DEFAULT_OUT_TOPIC)
    parser.add_argument("--group-id",     default=DEFAULT_GROUP_ID)
    parser.add_argument("--max-messages", type=int, default=0,
                        help="Exit after N messages (0 = run forever)")
    args = parser.parse_args()

    run(args.broker, args.in_topic, args.out_topic, args.group_id, args.max_messages)


if __name__ == "__main__":
    main()
