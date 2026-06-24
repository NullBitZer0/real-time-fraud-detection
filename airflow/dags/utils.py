"""Shared utilities for Airflow DAGs."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import requests

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))
METRICS_DIR = PROJECT_ROOT / "metrics"
DRIFT_JSON = METRICS_DIR / "drift_report.json"

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def run_cmd(cmd: str, cwd: Path = PROJECT_ROOT, timeout: int = 600) -> dict:
    """Run a shell command and return {returncode, stdout, stderr}."""
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd),
        capture_output=True, text=True, timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def read_drift_summary(threshold: int = 5) -> dict:
    """Parse metrics/drift_report.json and return drift summary."""
    if not DRIFT_JSON.exists():
        return {"n_drifted_features": 999, "n_total_features": 0, "drift_share": 1.0, "available": False}
    with open(DRIFT_JSON) as f:
        report = json.load(f)
    n_drifted = 0
    n_total = 0
    for m in report.get("metrics", []):
        if "ValueDrift" in m.get("metric_id", ""):
            n_total += 1
            value = m.get("value", {})
            if isinstance(value, dict) and value.get("drift_detected", False):
                n_drifted += 1
    return {
        "n_drifted_features": n_drifted,
        "n_total_features": n_total,
        "drift_share": n_drifted / max(n_total, 1),
        "available": True,
    }


# ── Slack helpers ──────────────────────────────────────────────────────────────
_COLOR_MAP = {
    "success": "#2ecc71",
    "failure": "#e74c3c",
    "running": "#3498db",
}


def slack_post(text: str, color: str = "running") -> None:
    """Post a message to Slack via incoming webhook. Silently no-ops if URL is unset."""
    if not SLACK_WEBHOOK_URL:
        return
    payload = {
        "attachments": [{
            "color": _COLOR_MAP.get(color, "#95a5a6"),
            "text": text,
            "ts": int(__import__("time").time()),
        }],
    }
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Slack post failed (non-blocking): {e}")


def _task_status_emoji(state: str) -> str:
    return {"success": "✅", "failed": "❌", "running": "🔄"}.get(state, "⚪")


def slack_task_callback(context, color: str | None = None) -> None:
    """Airflow on_success / on_failure callback — posts task result to Slack."""
    ti = context.get("ti")
    task_id = ti.task_id if ti else "?"
    dag_id = context.get("dag").dag_id if context.get("dag") else "?"
    state = context.get("state") or ("failure" if context.get("exception") else "success")
    color = color or state
    emoji = _task_status_emoji(state)
    exc = context.get("exception")
    exc_line = f"\n`{exc}`" if exc else ""

    slack_post(
        f"{emoji} *{dag_id}* → `{task_id}` {state.upper()}{exc_line}",
        color=color,
    )


def slack_dag_callback(context, color: str | None = None) -> None:
    """Airflow on_success_callback / on_failure_callback for DAG-level default_args."""
    state = context.get("state") or ("failure" if context.get("exception") else "success")
    color = color or state
    emoji = _task_status_emoji(state)
    dag_id = context.get("dag").dag_id if context.get("dag") else "?"
    run_id = context.get("run_id", "?")
    exc = context.get("exception")
    exc_line = f"\n`{exc}`" if exc else ""

    slack_post(
        f"{emoji} *DAG {dag_id}* [{run_id}] {state.upper()}{exc_line}",
        color=color,
    )
