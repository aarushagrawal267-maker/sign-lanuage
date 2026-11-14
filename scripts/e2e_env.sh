#!/usr/bin/env bash
# Start an isolated backend (synthetic test data + small trained model) and the
# Vite dev server for Playwright e2e tests on Linux/CI. Usage:
#   scripts/e2e_env.sh            # start; prints the data dir
#   cd frontend && npx playwright test
#   pkill -f "uvicorn app.main:app"; pkill -f "vite"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
TMP="$(mktemp -d -t slr-e2e-XXXX)"
export SLR_DATA_DIR="$TMP/data" SLR_ARTIFACTS_DIR="$TMP/artifacts"
cd "$ROOT/backend"
"$PY" -m app.cli seed-synthetic >/dev/null
"$PY" -m app.cli train --split signer-holdout --epochs 15 --hidden 64 --layers 1 --seq-len 32 --activate >/dev/null
nohup "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >"$TMP/backend.log" 2>&1 &
cd "$ROOT/frontend"
nohup node node_modules/vite/bin/vite.js --port 5173 --strictPort >"$TMP/frontend.log" 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8000/api/health >/dev/null && curl -sf http://localhost:5173/ >/dev/null && break; sleep 1; done
echo "e2e environment ready (data: $TMP)"
