import os
import sys
import shutil

import joblib
import mlflow
from mlflow import MlflowClient

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.mlflow_tracking import *


PUSH_DIR = "models"   # local directory where deployed model artifacts live


class ModelPusher:
    """
    Loads the champion model from the MLflow Model Registry and pushes it
    to the local serving directory (models/).

    After pushing, the prediction pipeline automatically picks up the new
    model on its next load because it reads from models/model.pkl.

    Workflow:
        1. Pull the Production (or Staging) version from the Registry.
        2. Download model + artifacts to a temp location via MLflow.
        3. Copy model.pkl + companion artifacts (scaler, imputer,
           label_encoders) into models/ — atomically overwriting the old ones.
        4. Log the push event back to MLflow for auditability.

    Usage:
        pusher = ModelPusher()
        result = pusher.push_model(
            model_name="FraudDetectionModel",
            stage="Production"          # or "Staging"
        )
    """

    def __init__(self):
        self.client   = MlflowClient()
        self.push_dir = PUSH_DIR
        os.makedirs(self.push_dir, exist_ok=True)

    def _get_latest_version(self, model_name: str, stage: str):
        """Return the latest ModelVersion object for the given stage."""

        versions = self.client.get_latest_versions(
            model_name, stages=[stage]
        )

        if not versions:
            raise ValueError(
                f"No model version found in stage '{stage}' "
                f"for registered model '{model_name}'. "
                "Register and promote a model first."
            )

        return versions[0]

    def push_model(
        self,
        model_name: str = "FraudDetectionModel",
        stage:      str = "Production",
        dst_name:   str = "model.pkl"
    ) -> dict:
        """
        Pull the champion model from the Registry and deploy it locally.

        Args:
            model_name : registered model name in MLflow
            stage      : "Production" or "Staging"
            dst_name   : filename to save the model as in models/

        Returns:
            dict with version, run_id, stage, dst_path
        """

        try:

            logging.info(
                f"Pushing '{model_name}' [{stage}] → {self.push_dir}/{dst_name}"
            )

            # ── 1. Resolve version ────────────────────────────────────────────
            mv      = self._get_latest_version(model_name, stage)
            run_id  = mv.run_id
            version = mv.version

            logging.info(
                f"Found v{version} | run_id={run_id} | stage={stage}"
            )

            # ── 2. Download model artifact from MLflow ────────────────────────
            model_uri  = f"models:/{model_name}/{stage}"
            local_path = mlflow.sklearn.load_model(model_uri)

            # mlflow.sklearn.load_model returns the loaded model object directly
            dst_path = os.path.join(self.push_dir, dst_name)
            joblib.dump(local_path, dst_path)

            logging.info(f"Model saved → {dst_path}")

            # ── 3. Log push event to MLflow ───────────────────────────────────
            with mlflow.start_run(run_name="model_push"):
                mlflow.log_param("pushed_model_name",    model_name)
                mlflow.log_param("pushed_model_version", version)
                mlflow.log_param("pushed_stage",         stage)
                mlflow.log_param("pushed_to",            dst_path)
                mlflow.set_tag("event", "model_push")

            result = {
                "model_name":    model_name,
                "model_version": version,
                "run_id":        run_id,
                "stage":         stage,
                "dst_path":      dst_path,
            }

            logging.info(f"Push completed: {result}")

            return result

        except Exception as e:
            raise CustomException(e, sys)

    def push_all_artifacts(
        self,
        model_name: str = "FraudDetectionModel",
        stage:      str = "Production"
    ) -> dict:
        """
        Push model + companion artifacts (scaler, imputer, label_encoders)
        from the best run associated with the registered version.

        Use this when the companion artifacts were logged as MLflow artifacts
        (e.g., mlflow.log_artifact("models/scaler.pkl")).

        Args:
            model_name : registered model name
            stage      : "Production" or "Staging"

        Returns:
            dict with version, run_id, stage, downloaded artifacts list
        """

        try:

            mv     = self._get_latest_version(model_name, stage)
            run_id = mv.run_id

            logging.info(
                f"Downloading all artifacts for run {run_id} → {self.push_dir}/"
            )

            # Download all artifacts logged in that run
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                dst_path=self.push_dir
            )

            logging.info(f"All artifacts downloaded to: {artifact_path}")

            result = {
                "model_name":    model_name,
                "model_version": mv.version,
                "run_id":        run_id,
                "stage":         stage,
                "artifact_path": artifact_path,
            }

            logging.info(f"Artifact push completed: {result}")

            return result

        except Exception as e:
            raise CustomException(e, sys)

    def rollback(
        self,
        model_name: str = "FraudDetectionModel",
        version:    str = None
    ) -> None:
        """
        Roll back to a specific version by transitioning it back to Production.

        Args:
            model_name : registered model name
            version    : version number as string to roll back to
        """

        try:

            if version is None:
                raise ValueError(
                    "Specify a version to roll back to. "
                    "Use ModelRegistry.list_versions() to see all versions."
                )

            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )

            logging.info(
                f"Rolled back '{model_name}' to v{version} → Production."
            )

        except Exception as e:
            raise CustomException(e, sys)
