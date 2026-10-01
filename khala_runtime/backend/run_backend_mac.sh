#!/usr/bin/env bash
# Launch the Khala web stack on Apple Silicon (MPS) using the vanilla pipeline:
# one keep_loaded worker + the API gateway. The frontend (vite) is started separately.
#
#   bash backend/run_backend_mac.sh            # MPS
#   bash backend/run_backend_mac.sh --device cpu
set -euo pipefail

DEVICE="mps"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) echo "usage: run_backend_mac.sh [--device mps|cpu]"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PY="$PROJECT_ROOT/.venv-mac/bin/python"
LOG_DIR="$PROJECT_ROOT/backend/logs"
mkdir -p "$LOG_DIR"

WORKER_PORT=8001
API_PORT=8889

export KHALA_BACKEND="vanilla"
export KHALA_DEVICE="$DEVICE"
export KHALA_VANILLA_WEIGHTS="$PROJECT_ROOT/_cuda_artifacts"
export KHALA_TRACKS_PER_JOB="1"
export PYTHONPATH="$PROJECT_ROOT"

echo "=== Khala (Mac/vanilla) — device=$DEVICE ==="

echo "[1/2] Starting worker on :$WORKER_PORT (keep_loaded; preloads ~8GB, ~1-2 min)..."
nohup "$PY" "$PROJECT_ROOT/backend/backend_worker.py" \
  --worker-port "$WORKER_PORT" --runtime-mode keep_loaded \
  > "$LOG_DIR/worker_mac.log" 2>&1 &
WORKER_PID=$!
echo "  worker pid=$WORKER_PID  log=$LOG_DIR/worker_mac.log"

if ! "$PY" "$PROJECT_ROOT/tools/check_worker_health.py" \
      "http://127.0.0.1:$WORKER_PORT/health" idle 360; then
  echo "ERROR: worker did not reach 'idle'. Last log lines:"
  tail -20 "$LOG_DIR/worker_mac.log"
  kill "$WORKER_PID" 2>/dev/null || true
  exit 1
fi
echo "  worker idle."

echo "[2/2] Starting API on :$API_PORT ..."
nohup "$PY" "$PROJECT_ROOT/backend/backend_api.py" \
  --port "$API_PORT" --num-workers 1 --worker-base-port "$WORKER_PORT" \
  > "$LOG_DIR/api_mac.log" 2>&1 &
API_PID=$!
echo "  api pid=$API_PID  log=$LOG_DIR/api_mac.log"

echo "$WORKER_PID $API_PID" > "$LOG_DIR/mac_pids.txt"
sleep 2
echo
echo "Backend up. Worker pid=$WORKER_PID, API pid=$API_PID (saved to $LOG_DIR/mac_pids.txt)."
echo "Frontend:  cd frontend && npm install && npm run dev   then open http://localhost:30869"
echo "Stop:      kill \$(cat $LOG_DIR/mac_pids.txt)"
