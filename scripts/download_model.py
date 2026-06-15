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


if __name__ == "__main__":
    download_production_model()
