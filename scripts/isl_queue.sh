#!/usr/bin/env bash
# Queue for the ISL video/image sources that need our own MediaPipe extraction (scripts/ingest_isl_video.py).
#
# Runs the sources one after another. Every source is resumable (manifest per source) and runs in chunks of
# CHUNK_MIN minutes so the number of workers is re-chosen between chunks ("--workers auto": 1 while the INCLUDE
# and ISLRTC jobs of scripts/ingest_isl.py run, 3 of the 4 CPUs once both have finished). Everything runs with
# nice 10. Before every chunk the free disk must be above MIN_FREE_GB (default 10 GB), otherwise it waits.
# CISLR (gated on Hugging Face) is probed with a 1-byte request for a data file between sources and every
# CISLR_POLL_H hours after the list is done (for CISLR_POLL_DAYS days); it is ingested as soon as it answers 200/206.
#
# Start (detached, from the repository root):
#   setsid nohup bash scripts/isl_queue.sh >> data/isl/logs/queue.log 2>&1 < /dev/null &
# Progress: data/isl/logs/queue.log (one line per chunk), data/isl/logs/video_<source>.log (per item batches),
#           python scripts/ingest_isl_video.py status
# Stop: kill the queue process (its PID is in data/isl/logs/queue.pid); a restart resumes every source.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
LOGDIR="$ROOT/data/isl/logs"
LOG="$LOGDIR/queue.log"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
CHUNK_MIN="${CHUNK_MIN:-45}"
CISLR_POLL_H="${CISLR_POLL_H:-6}"
CISLR_POLL_DAYS="${CISLR_POLL_DAYS:-14}"
# Order: tiny ISLRTC re-run, then the most useful sources first (most labelled words, letters), unlabelled last.
SOURCES="${SOURCES:-islrtc-full wslp isl_fingerspelling mendeley_static isl_csltr mendeley_agri mendeley_emergency isl_csltr_words hf_akrit_alphabet hf_hemg_alphabet wslp_unlabelled}"
export SLR_TORCH_THREADS="${SLR_TORCH_THREADS:-2}" GLOG_minloglevel=2 PYTHONUNBUFFERED=1

mkdir -p "$LOGDIR"
exec 9> "$LOGDIR/queue.lock"
if ! flock -n 9; then
  echo "$(date -u '+%F %T') another isl_queue.sh is running; exiting" >> "$LOG"
  exit 0
fi
echo $$ > "$LOGDIR/queue.pid"

log() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
free_gb() { df -BG --output=avail "$ROOT/data/isl" | tail -1 | tr -dc '0-9'; }

wait_disk() {
  while [ "$(free_gb)" -le "$MIN_FREE_GB" ]; do
    log "free disk $(free_gb) GB <= ${MIN_FREE_GB} GB; waiting 10 min"
    sleep 600
  done
}

CISLR_DONE=0

run_source() {  # $1 = source name; returns 0 done, 2 gave up, 3 gated
  local src="$1" tries=0 rc
  while true; do
    wait_disk
    log "$src: start chunk (free disk $(free_gb) GB)"
    if [ "$src" = "islrtc-full" ]; then
      nice -n 10 "$PY" scripts/ingest_isl_video.py islrtc-full --workers 1 >> "$LOGDIR/video_${src}.out" 2>&1
      rc=$?
    else
      nice -n 10 "$PY" scripts/ingest_isl_video.py run "$src" --workers auto --max-minutes "$CHUNK_MIN" \
        --min-free-gb "$MIN_FREE_GB" >> "$LOGDIR/video_${src}.out" 2>&1
      rc=$?
    fi
    case "$rc" in
      0) log "$src: finished"; return 0 ;;
      10) log "$src: chunk done, more to do" ;;
      4) log "$src: stopped for low disk; waiting 10 min"; sleep 600 ;;
      3) log "$src: gated dataset not accessible yet"; return 3 ;;
      *) tries=$((tries + 1)); log "$src: error rc=$rc (try $tries of 5, see $LOGDIR/video_${src}.out)"
         if [ "$tries" -ge 5 ]; then log "$src: giving up for now"; return 2; fi
         sleep $((300 * tries)) ;;
    esac
  done
}

try_cislr() {
  [ "$CISLR_DONE" = 1 ] && return 0
  if "$PY" scripts/ingest_isl_video.py probe cislr >> "$LOGDIR/video_cislr.out" 2>&1; then
    log "cislr: access granted, ingesting"
    run_source cislr && CISLR_DONE=1
  fi
  return 0
}

log "queue start (pid $$): $SOURCES"
for s in $SOURCES; do
  try_cislr
  run_source "$s"
done
log "all listed sources processed; status:"
"$PY" scripts/ingest_isl_video.py status >> "$LOG" 2>&1

end=$(( $(date +%s) + CISLR_POLL_DAYS * 86400 ))
while [ "$CISLR_DONE" = 0 ] && [ "$(date +%s)" -lt "$end" ]; do
  try_cislr
  [ "$CISLR_DONE" = 1 ] && break
  log "cislr: still pending; next probe in ${CISLR_POLL_H} h"
  sleep $((CISLR_POLL_H * 3600))
done
log "queue end"
