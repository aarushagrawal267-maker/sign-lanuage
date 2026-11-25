#!/usr/bin/env bash
# Phase-2 ISL training driver: one resumable command per data profile (docs/ISL_DESIGN.md 15.2).
#
#   scripts/phase2_train.sh --profile research            # full run (many hours; see the time budget in 15.2)
#   scripts/phase2_train.sh --profile commercial-safe
#   scripts/phase2_train.sh --profile both                # commercial-safe, then research
#   scripts/phase2_train.sh --profile research --smoke    # tiny settings on whatever data is on disk
#
# Steps (each one writes a stamp file when it succeeds; a re-run skips stamped steps, so the driver can be
# restarted at any time; training steps also resume inside an epoch from their *.state files):
#   trim dedup vocab splits link bench pretrain encoder oneshot classifier cv e1a fingerspell
#   index fusion evaluate index_final pack
# Options:
#   --from STEP      ignore stamps from STEP on (re-run it and everything after it)
#   --only STEP      run just this step (ignores its stamp)
#   --force STEP     forget this step's stamp and training state, then run normally
#   --threads N      torch threads (default 2 while other jobs run; 4 when the machine is free)
#   --max-parts N    how many times a training step stopped by its time budget is re-run (default 1000)
#   --step-minutes M time budget per training call (0 = none); the step is re-run with --resume until done
#   --list           print the steps and their stamps
# Environment overrides for the full settings: PRE_EPOCHS PRE_STEPS PRE_BATCH HIDDEN LAYERS EMB ENC_EPOCHS
# ENC_STEPS CLS_EPOCHS CLS_STEPS E1A_STEPS CV_FOLDS (empty = skip cv) ONESHOT_FOLDS ONESHOT_EPOCHS LETTER_EPOCHS.
#
# Outputs: full -> artifacts/isl/<profile>/ (work) and artifacts/packs/<isl|isl-research>/ (pack the app loads);
# smoke -> artifacts/isl/phase2-smoke/<profile>/ (work, pack/ inside it; never loaded by the app).
# Logs: data/isl/logs/phase2_<profile>[_smoke].log. Everything runs under nice -n 10.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
PROFILE="research"
SMOKE=0
FROM=""
ONLY=""
FORCE=""
THREADS="${THREADS:-2}"
MAX_PARTS=1000
STEP_MIN=0
LIST=0
STEPS=(trim dedup vocab splits link bench pretrain encoder oneshot classifier cv e1a fingerspell index fusion evaluate index_final pack)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --max-parts) MAX_PARTS="$2"; shift 2 ;;
    --step-minutes) STEP_MIN="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if [[ "$PROFILE" == "both" ]]; then
  args=("$@")
  rc=0
  for p in commercial-safe research; do
    "$0" --profile "$p" $([[ $SMOKE == 1 ]] && echo --smoke) ${FROM:+--from "$FROM"} ${ONLY:+--only "$ONLY"} \
      ${FORCE:+--force "$FORCE"} --threads "$THREADS" --max-parts "$MAX_PARTS" --step-minutes "$STEP_MIN" || rc=$?
  done
  exit $rc
fi
case "$PROFILE" in commercial-safe|research) ;; *) echo "profile must be commercial-safe, research or both" >&2; exit 2 ;; esac

TAG="$PROFILE$([[ $SMOKE == 1 ]] && echo _smoke)"
LOGDIR="$REPO/data/isl/logs"
STAMPS="$LOGDIR/phase2_${TAG}.stamps"
LOG="$LOGDIR/phase2_${TAG}.log"
mkdir -p "$LOGDIR" "$STAMPS"

ROOTARG=""
if [[ $SMOKE == 1 ]]; then
  export SLR_ISL_WORKDIR="$REPO/artifacts/isl/phase2-smoke"
  WORK="$SLR_ISL_WORKDIR/$PROFILE"
  # a private ISL root: real manifests / landmarks (read-only symlinks), own derived/ and splits/, so the data
  # refresh steps run for real without touching the shared derived files
  SROOT="$SLR_ISL_WORKDIR/isl_root_$PROFILE"
  if [[ ! -d "$SROOT" ]]; then
    mkdir -p "$SROOT/splits"
    for d in manifests landmarks raw nltk_data; do ln -sfn "$REPO/data/isl/$d" "$SROOT/$d"; done
    cp -r "$REPO/data/isl/derived" "$SROOT/derived" 2>/dev/null || mkdir -p "$SROOT/derived"
    cp -r "$REPO/data/isl/splits/official" "$SROOT/splits/official" 2>/dev/null || true
  fi
  ROOTARG="--root $SROOT"
  PACK_OUT="$WORK/pack"
  SM="--smoke"
  CAP="--max-per-source ${SMOKE_PER_SOURCE:-300} --max-islrtc ${SMOKE_ISLRTC:-400}"
  PRE="--hidden 32 --layers 1 --emb-dim 32 --epochs 1 --batches 4 --batch-size 8 --max-clips 200"
  ENC="--hidden 32 --layers 1 --emb-dim 32 --epochs 1 --batches 4 --p 8 --k 2 --batch-size 16"
  ONE="--hidden 32 --layers 1 --emb-dim 32 --epochs 1 --batches 3 --p 8 --k 2 --batch-size 16 --folds 0"
  CLS="--hidden 32 --layers 1 --epochs 2 --batches 6 --batch-size 16"
  LET="--epochs 3"
else
  unset SLR_ISL_WORKDIR
  WORK="$REPO/artifacts/isl/$PROFILE"
  PACK_OUT=""
  SM=""
  CAP=""
  H="${HIDDEN:-192}"; L="${LAYERS:-2}"; E="${EMB:-256}"
  PRE="--hidden $H --layers $L --emb-dim $E --epochs ${PRE_EPOCHS:-20} --batches ${PRE_STEPS:-400} --batch-size ${PRE_BATCH:-48}"
  ENC="--hidden $H --layers $L --emb-dim $E --epochs ${ENC_EPOCHS:-40} --batches ${ENC_STEPS:-400} --p 32 --k 2 --patience 10"
  ONE="--hidden $H --layers $L --emb-dim $E --epochs ${ONESHOT_EPOCHS:-20} --batches ${ENC_STEPS:-300} --p 32 --k 2 --patience 6 --folds ${ONESHOT_FOLDS:-0 1}"
  CLS="--hidden $H --layers $L --epochs ${CLS_EPOCHS:-40} --batches ${CLS_STEPS:-300} --patience 10"
  LET="--epochs ${LETTER_EPOCHS:-40}"
fi
mkdir -p "$WORK"
VOCAB="$WORK/vocabulary.jsonl"
COMMON="--profile $PROFILE --threads $THREADS --vocab $VOCAB $SM"
export SLR_TORCH_THREADS="$THREADS"
cd "$REPO/backend" || exit 1

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
cli() { nice -n 10 "$PY" -m app.isl.cli $ROOTARG "$@" >> "$LOG" 2>&1; }

# a training call that may stop early (exit 75 = time budget, state saved): re-run with --resume until done
train() {
  local n=0 rc
  while :; do
    cli "$@" --resume $([[ "$STEP_MIN" != 0 ]] && echo --max-minutes "$STEP_MIN"); rc=$?
    [[ $rc -ne 75 ]] && return $rc
    n=$((n + 1)); [[ $n -ge $MAX_PARTS ]] && return 75
    log "  part $n done, continuing"
  done
}

state_files() {  # training state of a step (removed by --force)
  case "$1" in
    pretrain) echo "$WORK/pretrained_encoder.state.pt" ;;
    encoder) echo "$WORK/encoder.pt.state" ;;
    oneshot) ls "$WORK"/oneshot/fold*/encoder.pt.state 2>/dev/null ;;
    classifier) echo "$WORK/classifier.pt.state" ;;
    e1a) echo "$WORK/e1a/classifier.pt.state" ;;
    cv) ls "$WORK"/cv/fold*/classifier.pt.state 2>/dev/null ;;
  esac
}

run_step() {
  local s="$1"
  case "$s" in
    trim) cli trim ;;  # incremental: only clips without a trim yet
    dedup)
      if [[ $SMOKE == 1 ]]; then cli dedup --sources include wslp mendeley_emergency mendeley_agri; return $?; fi
      cli dedup --sources include islrtc cislr wslp isl500 mendeley_emergency mendeley_agri ;;
    vocab) cli vocab --no-trims --out "$VOCAB" ;;  # to the work folder (commit to models/packs/isl after review)
    splits) cli splits --vocab "$VOCAB" ;;  # fetches the official INCLUDE lists only when missing
    link) cli link-sources $COMMON --all-sources $CAP ;;
    bench) cli bench --threads "$THREADS" --steps $([[ $SMOKE == 1 ]] && echo 1 || echo 5) --hidden-sizes ${HIDDEN:-192} ;;
    pretrain) train pretrain $COMMON $CAP $PRE ;;
    encoder) train train-encoder $COMMON --all-sources $CAP $ENC --init auto ;;
    oneshot) train train-oneshot $COMMON --all-sources $CAP $ONE --init auto ;;
    classifier) train train-classifier $COMMON --all-sources $CAP $CLS ;;
    cv)  # classifiers for the other folds of the primary split: E1b mean / spread and a signer-grouped interval
      local base
      base=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('primary_split',''))" \
             "${SROOT:-$REPO/data/isl}/splits/splits_report.json" 2>/dev/null)
      if [[ -z "$base" ]]; then log "  no splits_report.json: skipped"; return 0; fi
      for k in ${CV_FOLDS-$([[ $SMOKE == 1 ]] && echo 1 || echo 1 2 3 4)}; do
        mkdir -p "$WORK/cv/fold$k"
        train train-classifier $COMMON --all-sources $CAP $CLS --split "$base-fold$k" --ckpt "$WORK/cv/fold$k/classifier.pt" || return $?
      done ;;
    e1a)  # E1a needs its own model trained on the official (not signer-independent) INCLUDE split
      if [[ ! -f "${SROOT:-$REPO/data/isl}/splits/include-official.json" ]]; then log "  no official split: skipped"; return 0; fi
      mkdir -p "$WORK/e1a"
      train train-classifier $COMMON $CAP $CLS --batches ${E1A_STEPS:-$([[ $SMOKE == 1 ]] && echo 6 || echo 120)} --split include-official --ckpt "$WORK/e1a/classifier.pt" ;;
    fingerspell) cli train-fingerspell --profile "$PROFILE" --threads "$THREADS" --vocab "$VOCAB" $SM $LET ;;
    index) cli build-index $COMMON --all-sources $CAP ;;  # training parts only: evaluation stays clean
    fusion) cli fit-fusion $COMMON --all-sources $CAP ;;
    evaluate) cli evaluate --all $COMMON --all-sources $CAP ${PACK_OUT:+--out "$PACK_OUT"} ;;
    index_final) cli build-index $COMMON --all-sources $CAP --all-include ;;  # the shipped index: every reference
    pack) cli build-pack $COMMON ${PACK_OUT:+--out "$PACK_OUT"} ;;
    *) log "unknown step $s"; return 2 ;;
  esac
}

if [[ $LIST == 1 ]]; then
  for s in "${STEPS[@]}"; do printf '%-12s %s\n' "$s" "$(cat "$STAMPS/$s.done" 2>/dev/null || echo -)"; done
  exit 0
fi
free_gb=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
if [[ "$free_gb" -lt 10 ]]; then log "only ${free_gb} GB free (< 10 GB floor): not starting"; exit 3; fi
if [[ -n "$FORCE" ]]; then rm -f "$STAMPS/$FORCE.done"; for f in $(state_files "$FORCE"); do rm -f "$f"; done; fi

log "phase 2 driver: profile=$PROFILE smoke=$SMOKE threads=$THREADS work=$WORK git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
reached_from=0
[[ -z "$FROM" ]] && reached_from=1
for s in "${STEPS[@]}"; do
  [[ "$s" == "$FROM" ]] && reached_from=1
  if [[ -n "$ONLY" && "$s" != "$ONLY" ]]; then continue; fi
  if [[ -z "$ONLY" && $reached_from == 1 && -n "$FROM" ]]; then rm -f "$STAMPS/$s.done"; fi
  if [[ -z "$ONLY" && -f "$STAMPS/$s.done" ]]; then log "skip $s (done $(cat "$STAMPS/$s.done"))"; continue; fi
  free_gb=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
  if [[ "$free_gb" -lt 10 ]]; then log "only ${free_gb} GB free: stopping before $s"; exit 3; fi
  log "step $s ..."
  t0=$(date +%s)
  run_step "$s"; rc=$?
  if [[ $rc -ne 0 ]]; then log "step $s FAILED (exit $rc); fix the cause and run the same command again (it resumes here)"; exit $rc; fi
  echo "$(date -u +%FT%TZ) $(( $(date +%s) - t0 ))s" > "$STAMPS/$s.done"
  log "step $s done in $(( $(date +%s) - t0 )) s"
done
log "phase 2 driver finished: profile=$PROFILE smoke=$SMOKE (report: $WORK/eval/report.md)"
