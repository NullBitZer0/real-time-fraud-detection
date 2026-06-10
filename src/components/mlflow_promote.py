"""MLflow model promotion: Staging → Production (with a metric gate).

Called by `.github/workflows/cd.yml` after the train stage.
- Reads `metrics/train_metrics.json` and compares `test_pr_auc` against a floor.
- If the new model is better than the current Production version, it gets promoted
  via the Model Registry alias `Production`; the previous Production is moved
  to `Archived`.

Usage:
    python -m src.components.mlflow_promote
    python -m src.components.mlflow_promote --floor 0.78 --name FraudDetectionCatBoost
"""
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow

# Load .env early — must happen BEFORE the logger is imported, because
# some env vars (e.g. DAGSHUB_TOKEN) are read below
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

from src.utils.exception import CustomException
from src.utils.logger import logging

# Default model registry name — must match `params.yaml:mlflow.registered_name`
DEFAULT_NAME = "FraudDetectionCatBoost"
# Floor: refuse to promote a model with worse test PR-AUC than this
DEFAULT_FLOOR = 0.78


def _init_dagshub() -> None:
    """Initialize the DAGsHub-backed MLflow tracking URI + auth."""
    owner = os.environ.get("DAGSHUB_REPO_OWNER", "NullBitZer0")
    repo  = os.environ.get("DAGSHUB_REPO_NAME",  "real-time-fraud-detection")
    if "DAGSHUB_REPO_OWNER" not in os.environ:
        os.environ["DAGSHUB_REPO_OWNER"] = owner
    if "DAGSHUB_REPO_NAME" not in os.environ:
        os.environ["DAGSHUB_REPO_NAME"]  = repo
    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        raise RuntimeError("DAGSHUB_TOKEN env var is required for promotion")
    # Configure MLflow tracking URI + auth directly instead of using
    # dagshub.init() which can trigger a fragile OAuth flow.
    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token


def get_production_metric(client, name: str) -> float | None:
    """Return the test_pr_auc of the current Production model, or None if none."""
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:
        return None
    for v in versions:
        # DAGsHub supports the legacy stage API but aliases may not persist
        if v.current_stage == "Production":
            run = client.get_run(v.run_id)
            return float(run.data.metrics.get("test_pr_auc", 0.0))
    return None


def get_latest_staging_metric(client, name: str) -> tuple[str | None, float | None]:
    """Return (version, test_pr_auc) of the most recent Staging version, or (None, None)."""
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:
        return None, None
    staging = [v for v in versions if v.current_stage == "Staging"]
    if not staging:
        return None, None
    latest = max(staging, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)
    return latest.version, float(run.data.metrics.get("test_pr_auc", 0.0))


def promote_to_production(client, name: str, version: str) -> None:
    """Move the given version to Production, archiving any previous Production.

    Uses the legacy stage-transition API because DAGsHub's MLflow backend
    supports it but the newer alias API may not persist changes.
    """
    client.transition_model_version_stage(
        name=name, version=version, stage="Production",
        archive_existing_versions=True,
    )
    logging.info(f"Promoted v{version} of '{name}' to @Production")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",  default=DEFAULT_NAME, help="Registered model name")
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                        help="Minimum acceptable test_pr_auc (default 0.78)")
    parser.add_argument("--metrics", default="metrics/train_metrics.json",
                        help="Path to the train-metrics JSON (sanity check)")
    args = parser.parse_args()

    try:
        # Sanity: make sure the local train-metrics file exists & meets the floor
        if not os.path.exists(args.metrics):
            raise FileNotFoundError(f"{args.metrics} not found — did `dvc repro train` run?")
        with open(args.metrics) as f:
            train_metrics = json.load(f)
        new_pr = float(train_metrics.get("test_pr_auc", 0.0))
        logging.info(f"Local train metrics: test_pr_auc = {new_pr:.4f}  (floor = {args.floor})")
        if new_pr < args.floor:
            raise RuntimeError(
                f"REJECTED: test_pr_auc {new_pr:.4f} < floor {args.floor:.4f} — "
                f"not promoting to Production"
            )

        _init_dagshub()
        client = mlflow.tracking.MlflowClient()

        # Get the current Production model (if any)
        old_prod_pr = get_production_metric(client, args.name)
        logging.info(f"Current Production test_pr_auc: {old_prod_pr}")

        # Get the latest Staging version
        version, staging_pr = get_latest_staging_metric(client, args.name)
        if version is None:
            raise RuntimeError(f"No Staging version found for '{args.name}'")
        logging.info(f"Latest Staging: v{version} test_pr_auc = {staging_pr:.4f}")

        # Decide whether to promote
        if old_prod_pr is not None and staging_pr is not None and staging_pr <= old_prod_pr:
            logging.warning(
                f"NOT PROMOTING: Staging v{version} ({staging_pr:.4f}) is not better "
                f"than current Production ({old_prod_pr:.4f})"
            )
            return 0

        promote_to_production(client, args.name, version)
        logging.info("✓ Promotion complete")
        return 0
    except Exception as e:
        logging.error(f"Promotion failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    sys.exit(main())
