"""Download the latest Production model from MLflow/DagsHub to a local directory.

Called at container startup when models/ is empty.
"""
from pathlib import Path

import mlflow

from src.utils.config import init_dagshub


def download_production_model(model_dir: str = "models") -> None:
    """Download the Production model artifacts from MLflow to model_dir."""
    target = Path(model_dir)
    model_file = target / "catboost.cbm"
    if model_file.exists():
        print(f"Model already exists at {model_file}, skipping download")
        return

    target.mkdir(parents=True, exist_ok=True)

    init_dagshub()
    client = mlflow.tracking.MlflowClient()

    registered_name = "FraudDetectionCatBoost"
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
