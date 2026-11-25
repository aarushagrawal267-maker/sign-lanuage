#!/usr/bin/env bash
# Queue for the ISL video/image sources that need our own MediaPipe extraction (scripts/ingest_isl_video.py).
#
# Runs the sources one after another. Every source is resumable (manifest per source) and runs in chunks of
# CHUNK_MIN minutes so the number of workers is re-chosen between chunks ("--workers auto": 1 while the INCLUDE
# and ISLRTC jobs of scripts/ingest_isl.py run, 3 of the 4 CPUs once both have finished). Everything runs with
# nice 10. Before every chunk the free disk must be above MIN_FREE_GB (default 10 GB), otherwise it waits.
#
# Gated datasets, checked every queue cycle (before a chunk, at most every GATED_PROBE_MIN minutes, then every
# CISLR_POLL_H hours for CISLR_POLL_DAYS days after the list is done) with a 1-byte request for a DATA file
# (`ingest_isl_gated.py --check`: CISLR dataset.csv, iSign iSign_v1.1.csv; the dataset cards answer even without
# access):
#   * CISLR  -> `ingest_isl_video.py run cislr`, the only CISLR path (ingest_isl_gated.py cislr delegates to it;
#               both hold manifests/cislr.lock).
#   * iSign  -> `ingest_isl_gated.py isign` (research only, read from the remote archive, never stored); uids that
#               GloSL already provides (data/isl/derived/glosl_isign_uids.txt) are skipped.
#
# Exit codes handled per chunk: 0 done, 10 more to do, 4 low disk (wait), 3 gated, 5 stopped for memory
# (--max-rss-gb, default MAX_RSS_GB=1.5 per process: restarted in fresh processes after 1 min, up to
# MEM_RESTARTS times per source), 6 another run of that manifest is active (wait 5 min), other = error (5 tries).
# Child processes do not inherit the queue lock (fd 9 is closed for them), so a chunk left running by a stopped
# queue cannot block the next queue.
#
# Start (detached, from the repository root):
#   setsid nohup bash scripts/isl_queue.sh >> data/isl/logs/queue.log 2>&1 < /dev/null &
# Progress: data/isl/logs/queue.log (one line per chunk), data/isl/logs/video_<source>.log (per item batches),
#           python scripts/ingest_isl_video.py status
# Stop: send SIGTERM to the queue process (its PID is in data/isl/logs/queue.pid); a restart resumes every source.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PY:-$ROOT/.venv/bin/python}"
LOGDIR="$ROOT/data/isl/logs"
LOG="$LOGDIR/queue.log"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
CHUNK_MIN="${CHUNK_MIN:-45}"
MAX_RSS_GB="${MAX_RSS_GB:-1.5}"
MEM_RESTARTS="${MEM_RESTARTS:-20}"
GATED_PROBE_MIN="${GATED_PROBE_MIN:-60}"
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
ISIGN_DONE=0
IN_GATED=0
LAST_PROBE=0

run_source() {  # $1 = source name; returns 0 done, 2 gave up, 3 gated
  local src="$1" tries=0 mem=0 rc
  while true; do
    maybe_gated
    wait_disk
    log "$src: start chunk (free disk $(free_gb) GB)"
    if [ "$src" = "islrtc-full" ]; then
      nice -n 10 "$PY" scripts/ingest_isl_video.py islrtc-full --workers 1 >> "$LOGDIR/video_${src}.out" 2>&1 9>&-
      rc=$?
    else
      nice -n 10 "$PY" scripts/ingest_isl_video.py run "$src" --workers auto --max-minutes "$CHUNK_MIN" \
        --min-free-gb "$MIN_FREE_GB" --max-rss-gb "$MAX_RSS_GB" >> "$LOGDIR/video_${src}.out" 2>&1 9>&-
      rc=$?
    fi
    case "$rc" in
      0) log "$src: finished"; return 0 ;;
      10) log "$src: chunk done, more to do" ;;
      4) log "$src: stopped for low disk; waiting 10 min"; sleep 600 ;;
      3) log "$src: gated dataset not accessible yet"; return 3 ;;
      5) mem=$((mem + 1))
         log "$src: stopped for memory (RSS above ${MAX_RSS_GB} GB; $mem of $MEM_RESTARTS); restarting in 1 min"
         if [ "$mem" -ge "$MEM_RESTARTS" ]; then log "$src: too many memory stops; giving up for now"; return 2; fi
         sleep 60 ;;
      6) log "$src: another run of this manifest is active; waiting 5 min"; sleep 300 ;;
      *) tries=$((tries + 1)); log "$src: error rc=$rc (try $tries of 5, see $LOGDIR/video_${src}.out)"
         if [ "$tries" -ge 5 ]; then log "$src: giving up for now"; return 2; fi
         sleep $((300 * tries)) ;;
    esac
  done
}

run_isign() {  # 0 done, 2 gave up
  local tries=0 mode=zip rc
  while true; do
    wait_disk
    log "isign: start (mode $mode, uids already in GloSL are skipped)"
    nice -n 10 "$PY" scripts/ingest_isl_gated.py isign --mode "$mode" >> "$LOGDIR/isign.out" 2>&1 9>&-
    rc=$?
    case "$rc" in
      0) log "isign: finished"; return 0 ;;
      2) if [ "$mode" = zip ]; then mode=stream; log "isign: archive is not a zip; retrying in stream mode"; continue; fi
         tries=$((tries + 1)) ;;
      3) log "isign: access withdrawn"; return 2 ;;
      6) log "isign: another iSign run is active; waiting 5 min"; sleep 300; continue ;;
      *) tries=$((tries + 1)) ;;
    esac
    log "isign: error rc=$rc (try $tries of 5, see $LOGDIR/isign.out)"
    if [ "$tries" -ge 5 ]; then log "isign: giving up for now"; return 2; fi
    sleep $((300 * tries))
  done
}

maybe_gated() {  # probe CISLR / iSign (data files) at most every GATED_PROBE_MIN minutes; ingest when granted
  [ "$IN_GATED" = 1 ] && return 0
  local now
  now=$(date +%s)
  [ $((now - LAST_PROBE)) -lt $((GATED_PROBE_MIN * 60)) ] && return 0
  LAST_PROBE=$now
  IN_GATED=1
  if [ "$CISLR_DONE" = 0 ] && "$PY" scripts/ingest_isl_gated.py --check cislr >> "$LOGDIR/video_cislr.out" 2>&1 9>&-; then
    log "cislr: dataset.csv answers (access granted); ingesting"
    run_source cislr && CISLR_DONE=1
  fi
  if [ "$ISIGN_DONE" = 0 ] && "$PY" scripts/ingest_isl_gated.py --check isign >> "$LOGDIR/isign.out" 2>&1 9>&-; then
    log "isign: iSign_v1.1.csv answers (access granted); ingesting the uids GloSL lacks"
    run_isign && ISIGN_DONE=1
  fi
  IN_GATED=0
}

log "queue start (pid $$): $SOURCES"
for s in $SOURCES; do
  run_source "$s"
done
log "all listed sources processed; status:"
"$PY" scripts/ingest_isl_video.py status >> "$LOG" 2>&1 9>&-

end=$(( $(date +%s) + CISLR_POLL_DAYS * 86400 ))
while { [ "$CISLR_DONE" = 0 ] || [ "$ISIGN_DONE" = 0 ]; } && [ "$(date +%s)" -lt "$end" ]; do
  LAST_PROBE=0
  maybe_gated
  { [ "$CISLR_DONE" = 1 ] && [ "$ISIGN_DONE" = 1 ]; } && break
  log "gated: CISLR done=$CISLR_DONE, iSign done=$ISIGN_DONE; next probe in ${CISLR_POLL_H} h"
  sleep $((CISLR_POLL_H * 3600))
done
log "queue end"
