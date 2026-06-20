"""Monthly Optuna hyperparameter tuning DAG.

Runs on the 1st of every month to search for the best CatBoost
hyperparameters using the latest data. Saves results to
models/optuna_best.json, which the weekly retrain pipeline
(fraud_retraining.py → training_pipeline.py) auto-loads.

Enable/disable via Airflow Variable `optuna_enabled` (default: "true")
or env var `OPTUNA_ENABLED`. Adjust trial count via
`optuna_n_trials` (default: 50).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable

from utils import PROJECT_ROOT, run_cmd, slack_dag_callback

OPTUNA_ENABLED = os.environ.get("OPTUNA_ENABLED", "").lower() in ("1", "true", "") or \
                 Variable.get("optuna_enabled", default_var="true").lower() == "true"
N_TRIALS = int(Variable.get("optuna_n_trials", default_var="50"))


@dag(
    dag_id="fraud_optuna_monthly",
    description="Monthly Optuna hyperparameter search for CatBoost",
    schedule="0 0 1 * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner":            "mlops",
        "depends_on_past":  False,
        "retries":          1,
        "retry_delay":      timedelta(minutes=5),
        "execution_timeout": timedelta(hours=4),
        "on_failure_callback": slack_dag_callback,
        "on_success_callback": slack_dag_callback,
    },
    tags=["fraud", "ml", "optuna", "tuning", "production"],
)
def fraud_optuna_monthly_dag():

    @task
    def run_optuna_search() -> dict:
        if not OPTUNA_ENABLED:
            print("Optuna disabled via config — skipping")
            return {"status": "skipped", "reason": "disabled"}

        print(f"Starting Optuna search — {N_TRIALS} trials")
        result = run_cmd(
            f"python -m src.components.optuna_tuning "
            f"--n-trials {N_TRIALS} --mlflow",
            timeout=7200,
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"Optuna search failed:\n{result['stderr']}")

        optuna_path = PROJECT_ROOT / "models" / "optuna_best.json"
        if optuna_path.exists():
            with open(optuna_path) as f:
                data = json.load(f)
            return {
                "status":      "completed",
                "n_trials":    N_TRIALS,
                "best_value":  data.get("best_value"),
                "best_params": data.get("best_params"),
            }
        return {"status": "completed", "n_trials": N_TRIALS, "best_value": None}

    @task
    def log_result(result: dict) -> None:
        print(f"Optuna monthly result: {json.dumps(result, indent=2)}")

    log_result(run_optuna_search())


fraud_optuna_monthly_dag()
