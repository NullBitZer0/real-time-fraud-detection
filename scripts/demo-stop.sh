#!/usr/bin/env bash
# scripts/demo-stop.sh — stop the demo (uvicorn + vite).
#
# Usage:
#   bash scripts/demo-stop.sh

set -e
echo "▶ Stopping uvicorn + vite…"
pkill -f 'uvicorn api.app:app' 2>/dev/null || true
pkill -f 'vite' 2>/dev/null || true
sleep 1
if pgrep -f 'uvicorn|vite' > /dev/null 2>&1; then
  echo "  ⚠ some processes still running:"
  pgrep -af 'uvicorn|vite' || true
  echo "  Run: pkill -9 -f 'uvicorn|vite'"
else
  echo "  ✓ all demo processes stopped"
fi
