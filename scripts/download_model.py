"""Download the latest Production model from MLflow/DagsHub to a local directory.

Called at container startup when models/ is empty.
"""
import os
from pathlib import Path

import mlflow


def _init_dagshub() -> None:
    owner = os.environ.get("DAGSHUB_REPO_OWNER", "NullBitZer0")
    repo  = os.environ.get("DAGSHUB_REPO_NAME",  "real-time-fraud-detection")
    token = os.environ.get("DAGSHUB_TOKEN", "")
    if not token:
        raise RuntimeError("DAGSHUB_TOKEN is required to download model")
    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token


def download_production_model(model_dir: str = "models") -> None:
    """Download the Production model artifacts from MLflow to model_dir."""
    target = Path(model_dir)
    model_file = target / "catboost.cbm"
    if model_file.exists():
        print(f"Model already exists at {model_file}, skipping download")
        return

    target.mkdir(parents=True, exist_ok=True)

    _init_dagshub()
    client = mlflow.tracking.MlflowClient()

    registered_name = os.environ.get("MLFLOW_MODEL_NAME", "FraudDetectionCatBoost")
    print(f"Fetching Production model for '{registered_name}'...")

    versions = client.search_model_versions(f"name='{registered_name}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if not prod_versions:
        raise RuntimeError(f"No Production version found for '{registered_name}'")

    latest = max(prod_versions, key=lambda v: int(v.version))
    print(f"Found Production v{latest.version} (run_id={latest.run_id})")

    # Download artifacts
    for artifact_name in ["catboost.cbm", "metadata.json", "feature_engineering.pkl"]:
        artifact_path = client.download_artifacts(
            run_id=latest.run_id,
            path=artifact_name,
            dst_path=str(target),
        )
        print(f"  Downloaded {artifact_name} -> {artifact_path}")

    print(f"Model download complete: {target}")


def download_test_data() -> None:
    """Download fraudTest.csv from DVC remote via DagsHub S3 API."""
    import requests

    target = Path("data/raw/fraudTest.csv")
    if target.exists():
        print(f"Test data already exists at {target}, skipping")
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("DAGSHUB_TOKEN", "")
    owner = os.environ.get("DAGSHUB_REPO_OWNER", "NullBitZer0")
    repo  = os.environ.get("DAGSHUB_REPO_NAME",  "real-time-fraud-detection")

    md5 = "692357a2589a1a6fcd44f14b3e1f9d2c"
    url = f"https://dagshub.com/{owner}/{repo}.s3/files/md5/{md5[:2]}/{md5[2:]}"

    print(f"Downloading fraudTest.csv from DVC remote ({md5})...")
    resp = requests.get(url, auth=(token, token), stream=True, timeout=600)
    resp.raise_for_status()

    with open(target, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192 * 1024):
            f.write(chunk)

    print(f"Downloaded fraudTest.csv ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    download_production_model()
