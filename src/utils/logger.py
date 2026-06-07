"""Structured JSON logger.

Drop-in replacement for the default Python logger — every record becomes a
JSON object with timestamp, level, name, message, and any extra fields
attached via `logger.info("msg", extra={...})`. The format is friendly to
ELK / Loki / Cloud Logging (one JSON object per line).

If `LOG_FORMAT=plain` is set in the env, the legacy human-readable format
is used (handy for local dev).

Usage:
    from src.utils.logger import logging
    logging.info("training started")
    logging.info("model promoted", extra={"version": 1, "stage": "Production"})
"""
import os
import sys
import json
import socket
import logging
from datetime import datetime, timezone


# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR  = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
os.makedirs(LOG_DIR, exist_ok=True)

# Set to "plain" to get the human-readable format (default = "json")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "json").lower()

# Hostname (for multi-instance log aggregation)
HOSTNAME = socket.gethostname()


# ── JSON formatter ────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """Render every LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.now(tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
            "host":    HOSTNAME,
            "pid":     record.process,
        }
        # Capture any `extra={...}` fields the caller attached
        for key, val in record.__dict__.items():
            if key in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = str(val)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# ── Setup ─────────────────────────────────────────────────────────────────────
def _setup() -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Wipe any previously-attached handlers (avoid duplicate logs on re-import)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = (
        "[ %(asctime)s ] %(levelname)s %(name)s - %(message)s"
        if LOG_FORMAT == "plain"
        else None
    )

    # File handler — always JSON (machine-readable, for ELK)
    fh = logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE))
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)

    # Stdout handler — JSON or plain depending on LOG_FORMAT
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(JsonFormatter() if LOG_FORMAT == "json" else logging.Formatter(fmt))
    root.addHandler(sh)
    return root


logging.getLogger().info  # touch attribute
_setup()


# ── Convenience: named logger getter ──────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# Backwards-compat shim: `from src.utils.logger import logging`
logging = logging
