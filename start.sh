#!/usr/bin/env bash
# Start backend (default http://localhost:8000) and frontend (default http://localhost:5173) in the background.
# Logs and pid files go to .run/. Stop with ./stop.sh. Windows: use start.ps1.
# Other ports:  BACKEND_PORT=8010 FRONTEND_PORT=5180 ./start.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/.run"
VPY="$ROOT/.venv/bin/python"
VITE="$ROOT/frontend/node_modules/vite/bin/vite.js"
BP="${BACKEND_PORT:-8000}"
FP="${FRONTEND_PORT:-5173}"
export SLR_BACKEND="http://127.0.0.1:$BP" SLR_FRONTEND_ORIGIN="http://localhost:$FP"
mkdir -p "$RUN"
[ -x "$VPY" ] || { echo "Python environment not found. Run ./setup.sh first." >&2; exit 1; }
[ -f "$VITE" ] || { echo "Frontend packages not installed. Run ./setup.sh first." >&2; exit 1; }
[ -f "$ROOT/assets/mediapipe/holistic_landmarker.task" ] || "$VPY" "$ROOT/scripts/fetch_models.py"

for name in backend frontend; do
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
    echo "The application already appears to be running. Use ./stop.sh first." >&2; exit 1
  fi
done
for port in "$BP" "$FP"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Port $port is already in use by another program (choose others with BACKEND_PORT=... FRONTEND_PORT=...)." >&2; exit 1
  fi
done

echo "Starting Sign Intelligence Studio..."
(cd "$ROOT/backend" && { nohup "$VPY" -m uvicorn app.main:app --host 127.0.0.1 --port "$BP" >"$RUN/backend.log" 2>&1 </dev/null & echo $! >"$RUN/backend.pid"; })
(cd "$ROOT/frontend" && { nohup node "$VITE" --port "$FP" --strictPort >"$RUN/frontend.log" 2>&1 </dev/null & echo $! >"$RUN/frontend.pid"; })

wait_url() {
  for _ in $(seq 1 "$2"); do
    if command -v curl >/dev/null 2>&1; then curl -sf "$1" >/dev/null && return 0
    else "$VPY" -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=2)" "$1" 2>/dev/null && return 0; fi
    sleep 1
  done
  return 1
}
if ! wait_url "http://127.0.0.1:$BP/api/health" 90; then
  echo "Backend did not become ready. Last log lines:" >&2; tail -n 25 "$RUN/backend.log" >&2; "$ROOT/stop.sh"; exit 1
fi
if ! wait_url "http://localhost:$FP/" 60; then
  echo "Frontend did not become ready. Last log lines:" >&2; tail -n 25 "$RUN/frontend.log" >&2; "$ROOT/stop.sh"; exit 1
fi
cat <<MSG

  Frontend   http://localhost:$FP
  Backend    http://localhost:$BP
  API docs   http://localhost:$BP/docs

Stop with:  ./stop.sh
MSG
