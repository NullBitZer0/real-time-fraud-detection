"""Main retraining DAG for the fraud detection model.

This DAG replaces the old DVC pipeline (`dvc repro`). It runs on a weekly
schedule (every Monday at 00:00) and can also be triggered manually.

The DAG is a drift-aware branch:
  1. Generate the Evidently drift report (train vs latest test window)
  2. Check if drift is significant
  3. If drift detected → retrain (pull data, train, promote)
     If no drift   → log "no-op" and exit
  4. Notify the team either way

External triggers (e.g. a webhook from the React dashboard, or a separate
daily drift-check DAG) can fire this DAG via the Airflow REST API:

    POST /api/v1/dags/fraud_retraining/dagRuns
    {"conf": {"trigger_source": "drift-check"}}
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow.decorators import dag, task, task_group
from airflow.models import Variable
from airflow.operators.python import get_current_context

from utils import METRICS_DIR, PROJECT_ROOT, read_drift_summary, run_cmd, slack_dag_callback, slack_post

# ── Thresholds ───────────────────────────────────────────────────────────────
DRIFT_FEATURE_THRESHOLD = int(Variable.get(
    "drift_feature_threshold", default_var="5"
))
PROMOTION_FLOOR = float(Variable.get(
    "promotion_metric_floor", default_var="0.78"
))


# ── Tasks ────────────────────────────────────────────────────────────────────
@task
def generate_drift_report() -> dict:
    """Run src/components/drift_report.py → metrics/drift_report.{html,json}."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_cmd("python -m src.components.drift_report", timeout=1800)
    if result["returncode"] != 0:
        raise RuntimeError(f"drift_report failed:\n{result['stderr']}")
    return read_drift_summary(DRIFT_FEATURE_THRESHOLD)


@task.branch
def decide_path(drift_summary: dict) -> str:
    """Branch: if drift > threshold → retrain, else → skip."""
    if not drift_summary.get("available"):
        # No drift report exists yet — always retrain on first run
        return "retrain"
    n_drifted = drift_summary["n_drifted_features"]
    triggered = n_drifted >= DRIFT_FEATURE_THRESHOLD
    print(
        f"  Drifted features: {n_drifted} / {drift_summary['n_total_features']}  "
        f"(threshold = {DRIFT_FEATURE_THRESHOLD})"
    )
    print(f"  → {'RETRAIN' if triggered else 'SKIP'}")
    return "retrain.retrain_model" if triggered else "retrain.skip"


@task_group(group_id="retrain")
def retrain():
    """The full retraining pipeline — only runs if drift was detected."""

    @task
    def pull_data() -> dict:
        """Pull the latest Sparkov CSVs from DVC remote (DAGsHub S3)."""
        result = run_cmd(
            "dvc pull data/raw/fraudTest.csv data/raw/fraudTrain.csv --force"
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"dvc pull failed:\n{result['stderr']}")
        return {"pulled_bytes": sum(
            (PROJECT_ROOT / "data/raw").glob("*.csv")
        ) and (PROJECT_ROOT / "data/raw/fraudTest.csv").stat().st_size
                  + (PROJECT_ROOT / "data/raw/fraudTrain.csv").stat().st_size}

    @task
    def seed_feast() -> dict:
        """Compute features, write to Postgres, apply Feast definitions."""
        result = run_cmd("python feast/seed.py --skip-materialize")
        if result["returncode"] != 0:
            raise RuntimeError(f"feast seed failed:\n{result['stderr']}")
        return {"postgres_seeded": True}

    @task
    def train_model() -> dict:
        """Run pipelines.training_pipeline (reads from Feast/Postgres) → MLflow."""
        result = run_cmd("python -m pipelines.training_pipeline")
        if result["returncode"] != 0:
            raise RuntimeError(f"training failed:\n{result['stderr']}")
        metrics_file = METRICS_DIR / "train_metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)
        return {
            "test_pr_auc":  metrics.get("test_pr_auc", 0.0),
            "val_pr_auc":   metrics.get("val_pr_auc", 0.0),
            "model_path":   str(PROJECT_ROOT / "models" / "catboost.cbm"),
        }

    @task
    def metric_gate(metrics: dict) -> dict:
        """Refuse to promote if PR-AUC < floor."""
        pr = metrics["test_pr_auc"]
        if pr < PROMOTION_FLOOR:
            raise ValueError(
                f"Metric gate FAILED: test_pr_auc={pr:.4f} < floor={PROMOTION_FLOOR}"
            )
        return {"passed": True, "test_pr_auc": pr, "floor": PROMOTION_FLOOR}

    @task
    def promote(metrics: dict) -> dict:
        """Promote the new model in MLflow registry (gated by comparison to
        current Production)."""
        result = run_cmd(
            f"python -m src.components.mlflow_promote --floor {PROMOTION_FLOOR}"
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"mlflow_promote failed:\n{result['stderr']}")
        return {"promoted": True, "test_pr_auc": metrics["test_pr_auc"]}

    @task
    def materialize_online() -> dict:
        """Materialize Postgres → Redis with the latest feature data."""
        result = run_cmd(
            "python -c \"from datetime import datetime; "
            "from feast import FeatureStore; "
            "s = FeatureStore(repo_path='feast/feature_repo'); "
            "s.materialize(datetime(2019,1,1), datetime(2025,1,1))\""
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"feast materialize failed:\n{result['stderr']}")
        return {"redis_materialized": True}

    model_result = train_model()
    chain = (
        pull_data()
        >> seed_feast()
        >> model_result
        >> metric_gate(model_result)
        >> promote(model_result)
        >> materialize_online()
    )
    return chain


@task
def skip() -> dict:
    """No-op branch when drift is below threshold."""
    print("✓ No significant drift detected — skipping retrain")
    return {"retrained": False, "reason": "no_drift"}


@task
def notify(drift_summary: dict, retrain_result: dict | None = None) -> dict:
    """Log a final summary and post to Slack."""
    ctx = get_current_context()
    triggered_by = ctx.get("dag_run").conf.get("trigger_source", "schedule")
    retrain_info = retrain_result or {"retrained": False}
    summary = {
        "ts":              datetime.utcnow().isoformat() + "Z",
        "dag_run_id":      ctx.get("run_id"),
        "triggered_by":    triggered_by,
        "drift_summary":   drift_summary,
        "retrain_result":  retrain_info,
    }
    print(f"📢 Final summary: {json.dumps(summary, indent=2)}")

    drifted = drift_summary.get("n_drifted_features", 0)
    total = drift_summary.get("n_total_features", 0)
    retrained = retrain_info.get("retrained", False)
    model_pr = retrain_info.get("new_pr_auc") or retrain_info.get("pr_auc")

    lines = [
        f"*Fraud Retrain DAG* — `{ctx.get('run_id')}`",
        f"Triggered by: {triggered_by}",
        f"Drift: {drifted}/{total} features",
        f"Retrained: {'Yes' if retrained else 'No'}",
    ]
    if model_pr:
        lines.append(f"New PR-AUC: {model_pr:.4f}")

    color = "success" if retrained else "running"
    slack_post("\n".join(lines), color=color)
    return summary


# ── DAG definition ───────────────────────────────────────────────────────────
@dag(
    dag_id="fraud_retraining",
    description=(
        "Weekly retraining of the fraud detection model. "
        "Branches on drift: only retrains if Evidently reports drift on >= "
        f"{DRIFT_FEATURE_THRESHOLD} features."
    ),
    schedule="0 0 * * 1",           # every Monday at 00:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,                  # don't backfill missed runs
    max_active_runs=1,              # one retrain at a time
    default_args={
        "owner":            "mlops",
        "depends_on_past":  False,
        "retries":          1,
        "retry_delay":      timedelta(minutes=5),
        "execution_timeout": timedelta(hours=2),
        "on_failure_callback": slack_dag_callback,
        "on_success_callback": slack_dag_callback,
    },
    tags=["fraud", "ml", "retraining", "production"],
)
def fraud_retraining_dag():

    drift = generate_drift_report()
    decision = decide_path(drift)
    skip_task = skip()

    # After retrain runs, attach its result to the notify call
    @task
    def notify_after_retrain(drift_summary, retrain_out):
        return notify.function(drift_summary, retrain_out)

    # Wire up: drift → decision → [retrain group | skip] → notify
    decision >> retrain() >> notify_after_retrain(drift, retrain())
    decision >> skip_task >> notify(drift, retrain_result=None)


fraud_retraining_dag()
