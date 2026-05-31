import sys

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.mlflow_tracking import *


class ModelRegistry:
    """
    Registers the best trained model to the MLflow Model Registry.

    Workflow:
        1. Find the best run in the active experiment ranked by PR-AUC
           (primary metric for imbalanced fraud detection).
        2. Register that run's model under a versioned model name.
        3. Transition the new version to the target stage
           (Staging by default; pass stage="Production" to promote directly).

    Usage:
        registry = ModelRegistry()
        info = registry.register_best_model(
            model_name="FraudDetectionModel",
            metric="eval_pr_auc",         # logged by ModelEvaluation
            stage="Staging"
        )
    """

    def __init__(self):
        self.client = MlflowClient()

    def _get_best_run(self, experiment_name: str, metric: str) -> mlflow.entities.Run:
        """Return the run with the highest value of `metric`."""

        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise ValueError(
                f"Experiment '{experiment_name}' not found in MLflow. "
                "Run the training pipeline first."
            )

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"metrics.{metric} > 0",
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )

        if not runs:
            raise ValueError(
                f"No runs found with metric '{metric}' "
                f"in experiment '{experiment_name}'."
            )

        best_run = runs[0]

        logging.info(
            f"Best run found — ID: {best_run.info.run_id} | "
            f"{metric}: {best_run.data.metrics[metric]:.4f}"
        )

        return best_run

    def register_best_model(
        self,
        model_name:      str   = "FraudDetectionModel",
        experiment_name: str   = "fraud_detection_experiment",
        metric:          str   = "eval_pr_auc",
        stage:           str   = "Staging"
    ) -> dict:
        """
        Find the best run and register its model.

        Args:
            model_name      : name under which to register in the registry
            experiment_name : MLflow experiment to search
            metric          : metric to rank runs by (higher = better)
            stage           : target stage — "Staging" or "Production"

        Returns:
            dict with run_id, model_version, stage, metric_value
        """

        try:

            logging.info(
                f"Searching for best run in '{experiment_name}' "
                f"ranked by '{metric}' ..."
            )

            best_run = self._get_best_run(experiment_name, metric)
            run_id   = best_run.info.run_id
            metric_value = best_run.data.metrics[metric]

            # ── Register model ────────────────────────────────────────────────
            model_uri = f"runs:/{run_id}/model"

            registered = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            version = registered.version

            logging.info(
                f"Registered '{model_name}' version {version} "
                f"from run {run_id}"
            )

            # ── Transition to target stage ────────────────────────────────────
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production")
                # archive old Production versions when promoting a new one
            )

            logging.info(
                f"Model '{model_name}' v{version} → stage='{stage}'"
            )

            result = {
                "model_name":    model_name,
                "model_version": version,
                "run_id":        run_id,
                "stage":         stage,
                metric:          metric_value,
            }

            logging.info(f"Registry result: {result}")

            return result

        except Exception as e:
            raise CustomException(e, sys)

    def promote_to_production(
        self,
        model_name: str = "FraudDetectionModel",
        version:    str = None
    ) -> None:
        """
        Promote a specific version (or the latest Staging version)
        to Production, archiving the existing Production version.

        Args:
            model_name : registered model name
            version    : version number as string; if None, uses latest Staging
        """

        try:

            if version is None:
                # Find the latest version currently in Staging
                staging_versions = self.client.get_latest_versions(
                    model_name, stages=["Staging"]
                )

                if not staging_versions:
                    raise ValueError(
                        f"No model version in 'Staging' for '{model_name}'. "
                        "Register a model first."
                    )

                version = staging_versions[0].version

            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )

            logging.info(
                f"'{model_name}' v{version} promoted to Production. "
                "Previous Production version archived."
            )

        except Exception as e:
            raise CustomException(e, sys)

    def list_versions(self, model_name: str = "FraudDetectionModel") -> None:
        """Print all registered versions and their stages."""

        try:

            versions = self.client.search_model_versions(
                f"name='{model_name}'"
            )

            if not versions:
                logging.info(f"No versions found for '{model_name}'.")
                return

            logging.info(f"All versions of '{model_name}':")

            for v in versions:
                logging.info(
                    f"  v{v.version} | stage={v.current_stage} "
                    f"| run_id={v.run_id}"
                )

        except Exception as e:
            raise CustomException(e, sys)
