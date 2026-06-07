#!/usr/bin/env bash
# scripts/demo.sh — bring up the full fraud-detection demo with one command.
#
# What it starts:
#   1. Docker infra:  postgres, redis, kafka, prometheus, grafana
#   2. FastAPI:       uvicorn api.app:app on :8000
#   3. React:         vite dev server on :3000
#
# Stop with:   pkill -f 'uvicorn|vite'
# Logs:        /tmp/api.log  +  /tmp/frontend.log
#
# Usage:
#   bash scripts/demo.sh
#
set -e

# Resolve project root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "═══ Fraud Detection Demo Launcher ═══"
echo "  Project root: $ROOT"
echo ""

# ── 1. Docker infrastructure (idempotent — no-op if already up) ───────────────
echo "▶ Starting docker infrastructure (postgres, redis, kafka, prometheus, grafana)…"
# Pipe to tail so non-zero exit (e.g. port conflict) doesn't abort the script
docker compose up -d postgres redis kafka prometheus grafana 2>&1 | tail -5 || true

# ── 2. Load secrets (DAGsHub, Postgres, Redis) ────────────────────────────────
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "✓ Loaded .env (DAGSHUB_TOKEN, POSTGRES_*, REDIS_*)"
else
  echo "⚠ No .env file found — DAGsHub + audit log will not work"
fi

# ── 3. Start FastAPI in background ────────────────────────────────────────────
echo "▶ Starting FastAPI (uvicorn on :8000)…"
mkdir -p /tmp
setsid python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --log-level info \
  > /tmp/api.log 2>&1 < /dev/null &
API_PID=$!
disown

# ── 4. Start React dev server in background ───────────────────────────────────
echo "▶ Starting React dashboard (vite on :3000)…"
cd "$ROOT/frontend"
setsid npm run dev > /tmp/frontend.log 2>&1 < /dev/null &
FRONTEND_PID=$!
disown
cd "$ROOT"

# ── 5. Wait for services to be ready ──────────────────────────────────────────
echo ""
echo "⏳ Waiting for API to come up (max 30s)…"
for i in {1..30}; do
  if curl -fsS http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✓ API up after ${i}s"
    break
  fi
  sleep 1
done

echo "⏳ Waiting for frontend to come up (max 30s)…"
for i in {1..30}; do
  if curl -fsS http://localhost:3000/ > /dev/null 2>&1; then
    echo "  ✓ Frontend up after ${i}s"
    break
  fi
  sleep 1
done

# ── 6. Print the URL cheat sheet ──────────────────────────────────────────────
cat <<'EOF'

╔══════════════════════════════════════════════════════════════╗
║                ✅  DEMO IS READY                             ║
╠══════════════════════════════════════════════════════════════╣
║  🌐  React dashboard (7 tabs)   http://localhost:3000        ║
║  📡  FastAPI + Swagger          http://localhost:8000/docs   ║
║  ❤️  Health / readiness probe    http://localhost:8000/readyz ║
║  📈  Prometheus metrics         http://localhost:8000/metrics║
║  📊  Grafana monitoring         http://localhost:3001        ║
║                                (admin / admin)               ║
║  🔥  Prometheus raw             http://localhost:9090        ║
║  🗄️   pgAdmin (Postgres)        http://localhost:5050        ║
║                                (admin@admin.com / admin)     ║
║  📦  RedisInsight (Redis)       http://localhost:5540        ║
╚══════════════════════════════════════════════════════════════╝

📋  Watch logs:    tail -f /tmp/api.log /tmp/frontend.log
🛑  Stop demo:     pkill -f 'uvicorn|vite'
📊  Re-train:      dvc repro
🔗  MLflow runs:   https://dagshub.com/NullBitZer0/real-time-fraud-detection.mlflow
EOF
