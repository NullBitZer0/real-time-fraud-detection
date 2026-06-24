"""MLflowTracker — DAGsHub-backed MLflow wrapper.

Initializes the MLflow tracking URI to the DAGsHub-hosted MLflow server
(`https://dagshub.com/<owner>/<repo>.mlflow`) and exposes a simple
context-manager API used by the training pipeline.

If DAGsHub env vars are missing or unreachable, the tracker falls back
to a local `mlruns/` file store so dev work still works offline.

Usage:
    with MLflowTracker(cfg) as tracker:
        tracker.log_params({...})
        tracker.log_metrics({...})
        tracker.log_model(catboost_model, "catboost")
"""
import os
import re

import mlflow

from src.utils.logger import logging


class MLflowTracker:

    def __init__(self, cfg_mlflow: dict, run_name: str = None):
        self.cfg = cfg_mlflow
        self.run_name = run_name or cfg_mlflow.get("run_name", "train")
        self.experiment = cfg_mlflow.get("experiment_name", "default")
        self.tracking_uri = self._resolve_tracking_uri()
        self.run = None
        self._using_dagshub = False

    def _resolve_tracking_uri(self) -> str:
        """Build the tracking URI from config. Supports ${oc.env:VAR} placeholders."""
        uri = self.cfg.get("tracking_uri", "")
        # Resolve ${oc.env:VAR} placeholders manually (Hydra already does this,
        # but the tracker may be called outside a Hydra context)
        for m in re.findall(r"\$\{oc\.env:([A-Z_]+)\}", uri):
            val = os.environ.get(m, "")
            if val:
                uri = uri.replace(f"${{oc.env:{m}}}", val)
        return uri

    def _init_dagshub(self) -> bool:
        """Try to init DAGsHub-backed tracking. Returns True on success."""
        if "dagshub.com" not in self.tracking_uri:
            return False
        try:
            import dagshub
            # dagshub.init reads DAGSHUB_REPO_OWNER + DAGSHUB_REPO_NAME from env
            # and sets MLflow tracking URI + auth
            m = __import__("re").search(r"dagshub\.com/([^/]+)/([^/.]+)", self.tracking_uri)
            if m:
                os.environ.setdefault("DAGSHUB_REPO_OWNER", m.group(1))
                os.environ.setdefault("DAGSHUB_REPO_NAME",  m.group(2))
            dagshub.init(
                repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
                repo_name =os.environ["DAGSHUB_REPO_NAME"],
                mlflow=True,
            )
            self._using_dagshub = True
            return True
        except Exception as e:
            logging.warning(f"DAGsHub init failed ({e}) — falling back to local mlruns/")
            return False

    def start(self):
        """Begin a new MLflow run. Call end() or use as context manager."""
        self._init_dagshub()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        self.run = mlflow.start_run(run_name=self.run_name)
        logging.info(
            f"MLflow run started — uri={self.tracking_uri} "
            f"experiment={self.experiment} run={self.run_name}"
        )
        return self

    def end(self):
        if self.run is not None:
            mlflow.end_run()
            self.run = None
            logging.info("MLflow run ended")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_artifacts(self, paths: list):
        for p in paths:
            if os.path.exists(p):
                mlflow.log_artifact(p)

    def log_model(self, model, name: str = "model"):
        """Log a CatBoost model. Uses mlflow.catboost if available."""
        try:
            import mlflow.catboost
            mlflow.catboost.log_model(model, artifact_path=name)
            logging.info(f"MLflow: logged CatBoost model as '{name}'")
        except Exception as e:
            logging.warning(f"mlflow.catboost.log_model failed ({e}); using pyfunc fallback")
            try:
                import mlflow.pyfunc
                mlflow.pyfunc.log_model(
                    artifact_path=name,
                    python_model=_CatBoostPyfunc(model),
                )
            except Exception as e2:
                logging.error(f"Model logging failed entirely: {e2}")

    def register_model(self, model_uri: str, registered_name: str, stage: str = "Staging"):
        """Push a logged model to the MLflow Model Registry.

        Uses the legacy stage-transition API because DAGsHub's MLflow backend
        supports it but the newer `set_registered_model_alias` may not persist.
        """
        if not self.cfg.get("register_model", False):
            return
        try:
            import mlflow.tracking
            client = mlflow.tracking.MlflowClient()
            # Create the registered model if it doesn't exist yet
            try:
                client.create_registered_model(registered_name)
                logging.info(f"MLflow: created registered model '{registered_name}'")
            except Exception:
                pass  # already exists
            mv = client.create_model_version(
                name=registered_name,
                source=model_uri,
                run_id=self.run.info.run_id,
            )
            # DAGsHub: legacy stage transition is the reliable API
            client.transition_model_version_stage(
                name=registered_name, version=mv.version, stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
            logging.info(f"MLflow: registered v{mv.version} of '{registered_name}' @ {stage}")
        except Exception as e:
            logging.warning(f"Model registration failed: {e}")


class _CatBoostPyfunc(mlflow.pyfunc.PythonModel):
    """Minimal pyfunc wrapper around a CatBoost model for MLflow logging fallback."""
    def __init__(self, catboost_model):
        self.model = catboost_model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


