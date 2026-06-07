"""Daily drift watcher DAG.

Runs once a day. Generates the Evidently drift report and, if drift is
significant, triggers the `fraud_retraining` DAG via the TriggerDagRunOperator.

This is the "cheap and frequent" half of the architecture:
- fraud_drift_check   — daily, fast, only reads + generates a report
- fraud_retraining    — weekly + triggered, expensive, retrains a model

The two-DAG split keeps the daily check cheap and the retraining DAG
focused on its single responsibility.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import Variable


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
METRICS_DIR  = PROJECT_ROOT / "metrics"
DRIFT_JSON   = METRICS_DIR / "drift_report.json"

DRIFT_FEATURE_THRESHOLD = int(Variable.get(
    "drift_feature_threshold", default_var="5"
))


def _run(cmd: str) -> dict:
    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=600,
    )
    return {"returncode": result.returncode, "stderr": result.stderr[-2000:]}


@task
def generate_drift_report() -> dict:
    """Generate Evidently report (HTML + JSON)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result = _run("python -m src.components.drift_report")
    if result["returncode"] != 0:
        raise RuntimeError(f"drift_report failed: {result['stderr']}")
    return {"report_path": str(DRIFT_JSON)}


@task
def should_trigger_retrain() -> bool:
    """Parse the drift JSON and decide if a retrain is warranted."""
    if not DRIFT_JSON.exists():
        return True
    with open(DRIFT_JSON) as f:
        report = json.load(f)
    n_drifted = 0
    n_total = 0
    for m in report.get("metrics", []):
        if "ValueDrift" in m.get("metric_id", ""):
            n_total += 1
            if isinstance(m.get("value"), dict) and m["value"].get("drift_detected"):
                n_drifted += 1
    print(
        f"  Drift summary: {n_drifted} / {n_total} features drifted "
        f"(threshold = {DRIFT_FEATURE_THRESHOLD})"
    )
    return n_drifted >= DRIFT_FEATURE_THRESHOLD


@dag(
    dag_id="fraud_drift_check",
    description="Daily drift check. Triggers fraud_retraining if drift > threshold.",
    schedule="0 6 * * *",            # daily at 06:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["fraud", "drift", "monitoring"],
)
def fraud_drift_check_dag():

    report    = generate_drift_report()
    triggered = should_trigger_retrain()

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="fraud_retraining",
        conf={"trigger_source": "drift-check"},
        reset_dag_run=True,
        # Only run if should_trigger_retrain returned True
        # (Airflow handles the conditional via the trigger_rule below)
    )

    # If drift was NOT detected, skip the trigger
    # (we use a no-op skip task to make the dependency explicit)
    @task
    def no_op():
        print("  No drift → no retrain triggered")

    skip = no_op()

    # Branch based on the boolean
    from airflow.operators.python import BranchPythonOperator
    branch = BranchPythonOperator(
        task_id="branch",
        python_callable=lambda: "trigger_retrain" if triggered else "no_op",
    )

    report >> branch
    branch >> [trigger_retrain, skip]


fraud_drift_check_dag()
