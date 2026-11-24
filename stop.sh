#!/usr/bin/env bash
# Stop the processes recorded by start.sh (and only those).
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/.run"
stopped=0
for name in frontend backend; do
  f="$RUN/$name.pid"
  [ -f "$f" ] || continue
  pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    echo "stopped $name (pid $pid)"; stopped=$((stopped + 1))
  fi
  rm -f "$f"
done
[ "$stopped" -gt 0 ] && echo "Sign Intelligence Studio stopped." || echo "Nothing to stop."
