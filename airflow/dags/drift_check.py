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

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from utils import METRICS_DIR, read_drift_summary, run_cmd, slack_dag_callback

DRIFT_FEATURE_THRESHOLD = int(Variable.get(
    "drift_feature_threshold", default_var="5"
))


@task
def generate_drift_report() -> dict:
    """Generate Evidently report (HTML + JSON)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_cmd("python -m src.components.drift_report")
    if result["returncode"] != 0:
        raise RuntimeError(f"drift_report failed: {result['stderr']}")
    return {"report_path": str(METRICS_DIR / "drift_report.json")}


@task
def should_trigger_retrain() -> bool:
    """Parse the drift JSON and decide if a retrain is warranted."""
    summary = read_drift_summary(DRIFT_FEATURE_THRESHOLD)
    if not summary["available"]:
        return True
    n_drifted = summary["n_drifted_features"]
    print(
        f"  Drift summary: {n_drifted} / {summary['n_total_features']} features drifted "
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
        "on_failure_callback": slack_dag_callback,
        "on_success_callback": slack_dag_callback,
    },
    tags=["fraud", "drift", "monitoring"],
)
def fraud_drift_check_dag():

    report    = generate_drift_report()
    should_retrain = should_trigger_retrain()

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="fraud_retraining",
        conf={"trigger_source": "drift-check"},
        reset_dag_run=True,
    )

    @task
    def no_op():
        print("  No drift → no retrain triggered")

    skip = no_op()

    @task.branch
    def branch_on_drift(should_retrain: bool) -> str:
        return "trigger_retrain" if should_retrain else "no_op"

    branch = branch_on_drift(should_retrain)

    report >> branch
    branch >> [trigger_retrain, skip]


fraud_drift_check_dag()
