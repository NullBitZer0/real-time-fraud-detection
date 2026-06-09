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
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task, task_group
from airflow.operators.python import get_current_context
from airflow.models import Variable


# ── Paths ────────────────────────────────────────────────────────────────────
# The project root is mounted at /opt/airflow/project in the Docker image
# (see airflow/Dockerfile). When running Airflow on the host, set
# AIRFLOW__CORE__DAGS_FOLDER or use the env var PROJECT_ROOT.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
METRICS_DIR  = PROJECT_ROOT / "metrics"
DRIFT_JSON   = METRICS_DIR / "drift_report.json"

# ── Thresholds ───────────────────────────────────────────────────────────────
# Number of features with statistical drift (KS test p < 0.05) that triggers
# a retrain. Evidently's report covers ~32 features; >5 drifted features is
# a real signal. Tweakable via Airflow Variable `drift_feature_threshold`.
DRIFT_FEATURE_THRESHOLD = int(Variable.get(
    "drift_feature_threshold", default_var="5"
))
PROMOTION_FLOOR = float(Variable.get(
    "promotion_metric_floor", default_var="0.78"
))


# ── Helpers ──────────────────────────────────────────────────────────────────
def _run(cmd: str, cwd: Path = PROJECT_ROOT) -> dict:
    """Run a shell command, return dict with returncode/stdout/stderr."""
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd),
        capture_output=True, text=True, timeout=1800,
    )
    return {
        "cmd":     cmd,
        "returncode": result.returncode,
        "stdout":  result.stdout[-4000:],   # cap for XCom
        "stderr":  result.stderr[-4000:],
    }


def _read_drift_summary() -> dict:
    """Parse metrics/drift_report.json → {n_drifted_features, drift_share, ...}."""
    if not DRIFT_JSON.exists():
        return {"n_drifted_features": 999, "drift_share": 1.0, "available": False}
    with open(DRIFT_JSON) as f:
        report = json.load(f)
    # Evidently 0.4 schema: metrics keyed by id
    metrics = report.get("metrics", [])
    drift_count = 0
    drift_total = 0
    for m in metrics:
        metric_id = m.get("metric_id", "")
        if "ValueDrift" in metric_id:
            drift_total += 1
            value = m.get("value", {})
            if isinstance(value, dict) and value.get("drift_detected", False):
                drift_count += 1
    return {
        "n_drifted_features": drift_count,
        "n_total_features":   drift_total,
        "drift_share":        drift_count / max(drift_total, 1),
        "available":          True,
    }


# ── Tasks ────────────────────────────────────────────────────────────────────
@task
def generate_drift_report() -> dict:
    """Run src/components/drift_report.py → metrics/drift_report.{html,json}."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result = _run("python -m src.components.drift_report")
    if result["returncode"] != 0:
        raise RuntimeError(f"drift_report failed:\n{result['stderr']}")
    return _read_drift_summary()


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
        result = _run(
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
        result = _run("python feast/seed.py --skip-materialize")
        if result["returncode"] != 0:
            raise RuntimeError(f"feast seed failed:\n{result['stderr']}")
        return {"postgres_seeded": True}

    @task
    def train_model() -> dict:
        """Run pipelines.training_pipeline (reads from Feast/Postgres) → MLflow."""
        result = _run("python -m pipelines.training_pipeline")
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
        result = _run(
            f"python -m src.components.mlflow_promote --floor {PROMOTION_FLOOR}"
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"mlflow_promote failed:\n{result['stderr']}")
        return {"promoted": True, "test_pr_auc": metrics["test_pr_auc"]}

    @task
    def materialize_online() -> dict:
        """Materialize Postgres → Redis with the latest feature data."""
        result = _run(
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
    """Log a final summary. In production, wire this to Slack/email/PagerDuty."""
    ctx = get_current_context()
    triggered_by = ctx.get("dag_run").conf.get("trigger_source", "schedule")
    summary = {
        "ts":              datetime.utcnow().isoformat() + "Z",
        "dag_run_id":      ctx.get("run_id"),
        "triggered_by":    triggered_by,
        "drift_summary":   drift_summary,
        "retrain_result":  retrain_result or {"retrained": False},
    }
    # In a real system: requests.post(SLACK_WEBHOOK, json=summary)
    print(f"📢 Final summary: {json.dumps(summary, indent=2)}")
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
    },
    tags=["fraud", "ml", "retraining", "production"],
)
def fraud_retraining_dag():

    drift = generate_drift_report()
    decision = decide_path(drift)
    skip_task = skip()
    final = notify(drift, retrain_result=None)

    # After retrain runs, attach its result to the notify call
    @task
    def notify_after_retrain(drift_summary, retrain_out):
        return notify.function(drift_summary, retrain_out)

    # Wire up: drift → decision → [retrain group | skip] → notify
    decision >> retrain() >> notify_after_retrain(drift, retrain())
    decision >> skip_task >> notify(drift, retrain_result=None)


fraud_retraining_dag()
