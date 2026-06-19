"""Shared configuration helpers used across multiple modules."""
import os

import mlflow

PG_CONFIG = {
    "host":     os.environ.get("POSTGRES_HOST", "localhost"),
    "port":     int(os.environ.get("POSTGRES_PORT", 5432)),
    "user":     os.environ.get("POSTGRES_USER", "feast"),
    "password": os.environ.get("POSTGRES_PASSWORD", "feast"),
    "dbname":   os.environ.get("POSTGRES_DB", "feast"),
}


def init_dagshub() -> None:
    """Initialize DAGsHub-backed MLflow tracking URI + auth.

    Sets MLFLOW_TRACKING_URI and auth env vars from DAGSHUB_REPO_OWNER,
    DAGSHUB_REPO_NAME, and DAGSHUB_TOKEN.
    """
    owner = os.environ.get("DAGSHUB_REPO_OWNER", "NullBitZer0")
    repo  = os.environ.get("DAGSHUB_REPO_NAME",  "real-time-fraud-detection")
    if "DAGSHUB_REPO_OWNER" not in os.environ:
        os.environ["DAGSHUB_REPO_OWNER"] = owner
    if "DAGSHUB_REPO_NAME" not in os.environ:
        os.environ["DAGSHUB_REPO_NAME"]  = repo
    token = os.environ.get("DAGSHUB_TOKEN", "")
    if not token:
        raise RuntimeError("DAGSHUB_TOKEN is required")
    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
