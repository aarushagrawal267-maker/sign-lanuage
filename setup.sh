#!/usr/bin/env bash
# One-time setup for Sign Intelligence Studio on Linux / macOS (Windows: use setup.ps1).
#   ./setup.sh          CPU PyTorch on Linux (CUDA build if nvidia-smi finds a GPU), default build on macOS
#   ./setup.sh --cpu    force the CPU-only PyTorch build
#   ./setup.sh --dev    also install test tools (pytest)
# Creates .venv, installs backend + frontend dependencies, downloads the 13 MB MediaPipe model
# and initialises the SQLite database. It does NOT download WLASL or start training; the
# pretrained WLASL100 model in models/ is registered automatically on first start.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
CPU=0; DEV=0
for a in "$@"; do
  case "$a" in
    --cpu) CPU=1 ;;
    --dev) DEV=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
step() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

step "Looking for Python 3.12 / 3.11"
PY=""
for c in python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$v" = "3.12" ] || [ "$v" = "3.11" ]; then PY="$c"; echo "    using $c (Python $v)"; break; fi
  fi
done
[ -n "$PY" ] || fail "Python 3.12 or 3.11 is required (MediaPipe wheels). Install it (e.g. 'sudo apt install python3.12 python3.12-venv' or 'brew install python@3.12') and re-run."

step "Checking Node.js"
command -v node >/dev/null 2>&1 || fail "Node.js 20+ is required (https://nodejs.org)."
NODE_MAJOR="$(node --version | sed 's/^v//; s/\..*//')"
[ "$NODE_MAJOR" -ge 20 ] || fail "Node.js 20+ is required (found $(node --version))."
echo "    Node $(node --version)"

step "Creating virtual environment (.venv)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip install --upgrade pip --quiet

step "Installing PyTorch"
if "$VPY" -c "import torch" 2>/dev/null; then
  echo "    already installed"
elif [ "$(uname)" = "Darwin" ]; then
  "$VPY" -m pip install torch --quiet
elif [ "$CPU" = 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "    NVIDIA GPU detected - installing the CUDA 12.6 build"
  "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --quiet \
    || "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
else
  echo "    installing the CPU build"
  "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi

step "Installing backend requirements"
if [ "$DEV" = 1 ]; then REQ=backend/requirements-dev.txt; else REQ=backend/requirements.txt; fi
"$VPY" -m pip install -r "$REQ" --quiet

step "Installing frontend packages (npm)"
(cd frontend && if [ -f package-lock.json ]; then npm ci --no-fund --no-audit; else npm install --no-fund --no-audit; fi)

step "Downloading MediaPipe Holistic model"
"$VPY" scripts/fetch_models.py

step "Initialising the local database"
(cd backend && "$VPY" -c "from app.db.session import get_engine; get_engine(); print('    data/app.db ready')")

command -v ffmpeg >/dev/null 2>&1 || echo "    note: ffmpeg not found - only needed to convert ASLPro .swf videos"
printf '\nSetup complete. Start the application with:  ./start.sh\n'
