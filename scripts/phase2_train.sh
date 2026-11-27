#!/usr/bin/env bash
# Phase-2 ISL training driver: one resumable command per data profile (docs/ISL_DESIGN.md 15.2).
#
#   scripts/phase2_train.sh --profile research            # full run (many hours; see the time budget in 15.2)
#   scripts/phase2_train.sh --profile commercial-safe
#   scripts/phase2_train.sh --profile both                # commercial-safe, then research
#   scripts/phase2_train.sh --profile research --smoke    # tiny settings on whatever data is on disk
#
# Data guard (full runs only): the driver refuses to start (exit 4) while any ingestion job runs
# (scripts/ingest_isl.py / ingest_isl_pose.py / ingest_isl_video.py, or isl_queue.sh before its list is done) or
# while any manifest the profile reads is incomplete (python -m app.isl.cli readiness). --allow-partial starts
# anyway and marks the work folder (partial_data.json): every report and pack built from it then says PARTIAL DATA
# (the mark stays until the work folder is rebuilt with --from trim). --check-only runs just this check.
#
# Steps (each one writes a stamp file when it succeeds; a re-run skips stamped steps, so the driver can be
# restarted at any time; training steps also resume inside an epoch from their *.state files):
#   trim dedup vocab splits link bench pretrain_dedup pretrain encoder oneshot classifier cv e1a fingerspell
#   index fusion trim_eval evaluate index_final pack
# (trim uses the learned end model artifacts/isl/trim_boundary.pt when it exists - trim_version
#  trim-2+trim-learned-1 - and rules otherwise; trim_eval runs `cli trim-eval` on that same trimmer into
#  <work>/trim_eval.json, which E8 reads and checks against the trim sidecar's trim_version (stale -> "mismatch");
#  pretrain_dedup checks the OpenHands Sign Library chunks)
# Options:
#   --from STEP      re-run STEP and everything after it: their stamps AND training state files (*.state, incl.
#                    cv/fold*/ and oneshot/fold*/) are removed at once, before the first step runs, so a retrain
#                    really retrains (docs 15.19, 15.20). After a failure, run again WITHOUT --from to resume.
#                    In a scratch run, steps before STEP that have no stamp are seeded, not run (see --tag)
#   --only STEP      run just this step (ignores its stamp; a training step resumes from its state file, and a
#                    finished state returns the old weights: use --force STEP to retrain it)
#   --force S1,S2    forget these steps' stamps and training state, then run normally (comma-separated list)
#   --tag NAME       a scratch run (also env PHASE2_TAG): its own work folder (artifacts/isl/runs/NAME/<profile>
#                    unless PHASE2_WORK_BASE is set), its own stamps / log / readiness file under <work>/driver/,
#                    and its pack and eval report under <work>/pack (never artifacts/packs). Without a tag, a run
#                    with PHASE2_WORK_BASE set also keeps its stamps under <work>/driver/, so it never reads, skips
#                    on or deletes the real run's stamps (data/isl/logs/phase2_<profile>.stamps).
#                    A scratch run never rewrites the shared data/isl (trim, dedup, splits are refused, exit 2):
#                    start it with --from bench (or later); the earlier steps are then seeded: their outputs
#                    (vocabulary.jsonl, encoder.pt, ...; never *.state) are copied from the real work folder
#                    (PHASE2_SEED_BASE/<profile>, default artifacts/isl/<profile>) and stamped "seeded" (docs 15.20)
#   --allow-shared-data  let a scratch run run trim / dedup / splits anyway (rewrites data/isl for every run)
#   --dry-run        print the work folder, the stamp folder, the files that would be removed and the steps that
#                    would run, then exit without writing anything (no data guard, no training)
#   --threads N      torch threads (default 2 while other jobs run; 4 when the machine is free)
#   --max-parts N    how many times a training step stopped by its time budget is re-run (default 1000)
#   --step-minutes M time budget per training call (0 = none); the step is re-run with --resume until done
#   --list           print the steps and their stamps
#   --allow-partial  full run on incomplete data (reports and packs are labelled PARTIAL DATA)
#   --check-only     run the data guard and exit (0 = ready or allowed, 4 = refused)
# Environment overrides for the full settings: PRE_EPOCHS PRE_STEPS PRE_BATCH HIDDEN LAYERS EMB ENC_EPOCHS
# ENC_STEPS CLS_EPOCHS CLS_STEPS E1A_STEPS CV_FOLDS (empty = skip cv) ONESHOT_FOLDS ONESHOT_EPOCHS LETTER_EPOCHS.
# WSLP is a second view of the ISLRTC dictionary (docs 15.32); defaults: research on, commercial-safe off (no WSLP):
#   CLS_DICTIONARY_GROUP=1|0   classifier / cv / e1a count ISLRTC, WSLP and CISLR as ONE signer group (--dictionary-group)
#   ENC_EXCLUDE_SOURCES=wslp   encoder / oneshot leave these sources out of supervision (--exclude-sources; "" = none)
#   INDEX_WSLP_TWINS=auto      index / index_final: keep | drop | auto (drop WSLP rows of concepts with an ISLRTC row
#                              when encoder.pt was trained without WSLP; an older encoder's index is built as before)
# LINK_ALIASES=1|0 (docs 15.34; research 1, commercial-safe 0): the link step strips WSLP's "(N)" numbering
#   (adult(1) -> isl.adult) and turns on the twin alias groups (female / woman-female, ...) for the work folder
#   (link-sources --link-aliases -> <work>/link_settings.json, read by every later step). Only the link step reads it.
# VARIANT_LINKS=auto|none (docs 15.36-37; default auto for both profiles): the link step reads <work>/variant_links.json
#   (the APPROVED copy of the proposal <work>/variant_audit.json that `cli variant-audit` writes; never seeded into a
#   scratch run) when it exists and freezes its relinks and SHA-256 into link_settings.json; none = ignore it. No file
#   (every run so far, every scratch run): nothing changes. VARIANT_ACCEPT_NOISY=1 / VARIANT_ALLOW_SPLIT_CHANGE=1
#   pass --accept-noisy / --allow-split-change to the link step (recorded). --only link is refused when the file's
#   state differs from the folder's frozen one (the later steps would pair new labels with old models: use --from link).
# Classifier sampler and selection (docs 15.38, QA plan 1.4; research on, commercial-safe off):
#   CLS_SOURCE_QUOTA=include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08 | default | none   classifier and cv draw
#     a source group by these shares, then a class inside it, then a clip (--source-quota; renormalised over the sources
#     present; e1a never gets it). Research default: the shares shown (= default); commercial-safe default: none. The
#     quota is checked against the profile's word sources before anything runs (exit 2; docs 15.39).
#   CLS_SELECT=worst_source|macro_f1   classifier and cv select the epoch on the lowest validation macro F1 of INCLUDE and
#     ISL500 and fit the temperature on source-balanced validation (--select-on). Research worst_source, else macro_f1.
#   Full research runs also default to CLS_STEPS=800 and CLS_PATIENCE=8 (commercial-safe 300 and 10; e1a keeps 10).
# EVAL_COMPARE_TO=<report.json>: the evaluate step runs `evaluate --all --compare-to <report.json>` (E1b and E4 like
# for like against that report, e.g. artifacts/isl/commercial-safe/eval/report.json; docs 15.27, 15.28, 15.31).
# One profile per run (refused with --profile both); a missing file is refused before anything runs (exit 2).
#
# Outputs: full -> artifacts/isl/<profile>/ (work) and artifacts/packs/<isl|isl-research>/ (pack the app loads);
# smoke -> artifacts/isl/phase2-smoke/<profile>/ (work, pack/ inside it; never loaded by the app).
# Logs: data/isl/logs/phase2_<profile>[_smoke].log for the real run (no tag, default work folder); otherwise
# <work>/driver/phase2_<profile>[_smoke][_<tag>].log. Everything runs under nice -n 10.
# Test hooks: SLR_ISL_DIR (data root), PHASE2_WORK_BASE (instead of artifacts/isl), PHASE2_SEED_BASE (seed folder of
# a scratch run, instead of artifacts/isl), SLR_ISL_PROC_ROOT (process table).
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
ALLOW_PARTIAL=0
CHECK_ONLY=0
DRY_RUN=0
ALLOW_SHARED=0
RUN_TAG="${PHASE2_TAG:-}"
STEPS=(trim dedup vocab splits link bench pretrain_dedup pretrain encoder oneshot classifier cv e1a fingerspell index fusion trim_eval evaluate index_final pack)

# an option that takes a value refuses an empty or missing one (exit 2): `--tag ""` must not mean the real run
need() { [[ -n "${2:-}" && "${2:-}" != --* ]] || { echo "$1 needs a value (see --help)" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) need "$1" "${2:-}"; PROFILE="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --from) need "$1" "${2:-}"; FROM="$2"; shift 2 ;;
    --only) need "$1" "${2:-}"; ONLY="$2"; shift 2 ;;
    --force) need "$1" "${2:-}"; FORCE="$2"; shift 2 ;;
    --threads) need "$1" "${2:-}"; THREADS="$2"; shift 2 ;;
    --max-parts) need "$1" "${2:-}"; MAX_PARTS="$2"; shift 2 ;;
    --step-minutes) need "$1" "${2:-}"; STEP_MIN="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    --allow-partial) ALLOW_PARTIAL=1; shift ;;
    --allow-shared-data) ALLOW_SHARED=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --tag) need "$1" "${2:-}"; RUN_TAG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) awk 'NR > 1 && /^set -uo/ {exit} NR > 1' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if [[ "$PROFILE" == "both" && -n "${EVAL_COMPARE_TO:-}" ]]; then
  # commercial-safe would be compared with its own report, which its evaluate step then replaces (docs 15.31)
  echo "EVAL_COMPARE_TO needs one profile per run (not --profile both): run the two profiles separately" >&2; exit 2
fi
if [[ "$PROFILE" == "both" ]]; then
  args=("$@")
  rc=0
  for p in commercial-safe research; do
    "$0" --profile "$p" $([[ $SMOKE == 1 ]] && echo --smoke) ${FROM:+--from "$FROM"} ${ONLY:+--only "$ONLY"} \
      ${FORCE:+--force "$FORCE"} --threads "$THREADS" --max-parts "$MAX_PARTS" --step-minutes "$STEP_MIN" \
      $([[ $ALLOW_PARTIAL == 1 ]] && echo --allow-partial) $([[ $CHECK_ONLY == 1 ]] && echo --check-only) \
      $([[ $LIST == 1 ]] && echo --list) ${RUN_TAG:+--tag "$RUN_TAG"} $([[ $DRY_RUN == 1 ]] && echo --dry-run) \
      $([[ $ALLOW_SHARED == 1 ]] && echo --allow-shared-data) \
      || rc=$?
  done
  exit $rc
fi
case "$PROFILE" in commercial-safe|research) ;; *) echo "profile must be commercial-safe, research or both" >&2; exit 2 ;; esac

is_step() { local x; for x in "${STEPS[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }
for s in ${FROM:+"$FROM"} ${ONLY:+"$ONLY"}; do is_step "$s" || { echo "unknown step $s (see --help)" >&2; exit 2; }; done
FORCE_STEPS=()
if [[ -n "$FORCE" ]]; then
  IFS=',' read -r -a FORCE_STEPS <<< "$FORCE"
  for s in "${FORCE_STEPS[@]}"; do is_step "$s" || { echo "unknown step '$s' in --force (see --help)" >&2; exit 2; }; done
fi
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--tag / PHASE2_TAG: letters, digits, '.', '_' and '-' only" >&2; exit 2
fi
COMPARE_TO=""  # absolute now: the steps run from backend/
if [[ -n "${EVAL_COMPARE_TO:-}" ]]; then
  COMPARE_TO="$(realpath -m -- "$EVAL_COMPARE_TO")"
  [[ -f "$COMPARE_TO" ]] || { echo "EVAL_COMPARE_TO: $COMPARE_TO is not a file (a report.json of evaluate --all)" >&2; exit 2; }
fi

NAME="$PROFILE$([[ $SMOKE == 1 ]] && echo _smoke)${RUN_TAG:+_$RUN_TAG}"
ISL_ROOT="${SLR_ISL_DIR:-$REPO/data/isl}"
DEFAULT_BASE="$REPO/artifacts/isl"
BASE=""

ROOTARG=""
if [[ $SMOKE == 1 ]]; then
  export SLR_ISL_WORKDIR="$REPO/artifacts/isl/phase2-smoke${RUN_TAG:+/runs/$RUN_TAG}"
  WORK="$SLR_ISL_WORKDIR/$PROFILE"
  # a private ISL root: real manifests / landmarks (read-only symlinks), own derived/ and splits/, so the data
  # refresh steps run for real without touching the shared derived files
  SROOT="$SLR_ISL_WORKDIR/isl_root_$PROFILE"
  if [[ ! -d "$SROOT" ]]; then
    mkdir -p "$SROOT/splits"
    for d in manifests landmarks raw nltk_data logs; do ln -sfn "$ISL_ROOT/$d" "$SROOT/$d"; done
    cp -r "$ISL_ROOT/derived" "$SROOT/derived" 2>/dev/null || mkdir -p "$SROOT/derived"
    cp -r "$ISL_ROOT/splits/official" "$SROOT/splits/official" 2>/dev/null || true
  fi
  ROOTARG="--root $SROOT"
  PACK_OUT="$WORK/pack"
  SM="--smoke"
  CAP="--max-per-source ${SMOKE_PER_SOURCE:-300} --max-islrtc ${SMOKE_ISLRTC:-400}"
  PRE="--hidden 32 --layers 1 --emb-dim 32 --epochs 1 --batches 4 --batch-size 8 --max-clips 200"
  ENC="--hidden 32 --layers 1 --emb-dim 32 --epochs 1 --batches 4 --p 8 --k 2 --batch-size 16"
  ONE="--hidden 32 --layers 1 --emb-dim 32 --epochs 1 --batches 3 --p 8 --k 2 --batch-size 16 --folds 0"
  CLS="--hidden 32 --layers 1 --epochs 2 --batches 6 --batch-size 16"
  E1A_PAT=""
  LET="--epochs 3"
  PDD="--max-chunks ${SMOKE_SL_CHUNKS:-600} --calib ${SMOKE_SL_CALIB:-100}"
else
  BASE="${PHASE2_WORK_BASE:-}"
  [[ -z "$BASE" && -n "$RUN_TAG" ]] && BASE="$DEFAULT_BASE/runs/$RUN_TAG"  # a tagged run never shares the real work
  if [[ -n "$BASE" && "$(realpath -m "$BASE")" == "$(realpath -m "$DEFAULT_BASE")" ]]; then
    if [[ -n "$RUN_TAG" ]]; then
      echo "--tag $RUN_TAG with PHASE2_WORK_BASE = $DEFAULT_BASE: a scratch run must not use the real work folder" >&2
      exit 2
    fi
    BASE=""  # the real run, named explicitly
  fi
  if [[ -n "$BASE" ]]; then export SLR_ISL_WORKDIR="$BASE"; else unset SLR_ISL_WORKDIR; fi
  WORK="${BASE:-$DEFAULT_BASE}/$PROFILE"
  PACK_OUT=""
  [[ -n "$BASE" ]] && PACK_OUT="$WORK/pack"  # never over the shipped artifacts/packs/<id> (pack + eval report)
  SM=""
  CAP=""
  H="${HIDDEN:-192}"; L="${LAYERS:-2}"; E="${EMB:-256}"
  PRE="--hidden $H --layers $L --emb-dim $E --epochs ${PRE_EPOCHS:-20} --batches ${PRE_STEPS:-400} --batch-size ${PRE_BATCH:-48}"
  ENC="--hidden $H --layers $L --emb-dim $E --epochs ${ENC_EPOCHS:-40} --batches ${ENC_STEPS:-400} --p 32 --k 2 --patience 10"
  ONE="--hidden $H --layers $L --emb-dim $E --epochs ${ONESHOT_EPOCHS:-20} --batches ${ENC_STEPS:-300} --p 32 --k 2 --patience 6 --folds ${ONESHOT_FOLDS:-0 1}"
  # research: more steps and a shorter patience on the worst-source key (docs 15.38); commercial-safe as before
  if [[ "$PROFILE" == "research" ]]; then CS_DEF=800; CP_DEF=8; else CS_DEF=300; CP_DEF=10; fi
  CLS="--hidden $H --layers $L --epochs ${CLS_EPOCHS:-40} --batches ${CLS_STEPS:-$CS_DEF} --patience ${CLS_PATIENCE:-$CP_DEF}"
  E1A_PAT="--patience 10"  # e1a keeps its old patience (it gets no quota either)
  LET="--epochs ${LETTER_EPOCHS:-40}"
  PDD=""
fi
# WSLP / dictionary options (docs 15.32): research defaults on; commercial-safe has no WSLP and passes no new flag
if [[ "$PROFILE" == "research" ]]; then
  DG_DEFAULT=1; XS_DEFAULT="wslp"; TW_DEFAULT="auto"
else
  DG_DEFAULT=0; XS_DEFAULT=""; TW_DEFAULT="keep"
fi
DICT_GROUP="${CLS_DICTIONARY_GROUP:-$DG_DEFAULT}"
# classifier source quota and selection (docs 15.38): research on, commercial-safe off; "" or none = off
CQ_RESEARCH="include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08"  # = source_quota.DEFAULT_SOURCE_QUOTA
if [[ "$PROFILE" == "research" ]]; then
  CQ_DEFAULT="$CQ_RESEARCH"; CSEL_DEFAULT="worst_source"
else
  CQ_DEFAULT="none"; CSEL_DEFAULT="macro_f1"
fi
CLS_QUOTA="${CLS_SOURCE_QUOTA-$CQ_DEFAULT}"
[[ -z "$CLS_QUOTA" ]] && CLS_QUOTA="none"
[[ "$CLS_QUOTA" == default ]] && CLS_QUOTA="$CQ_RESEARCH"  # as the cli's --source-quota default
CLS_SEL="${CLS_SELECT:-$CSEL_DEFAULT}"
case "$CLS_SEL" in macro_f1|worst_source) ;; *) echo "CLS_SELECT must be macro_f1 or worst_source" >&2; exit 2 ;; esac
if [[ "$CLS_QUOTA" != none && ! "$CLS_QUOTA" =~ ^[a-z0-9_]+=[0-9.]+(,[a-z0-9_]+=[0-9.]+)*$ ]]; then
  echo "CLS_SOURCE_QUOTA: name=share,... (e.g. $CQ_RESEARCH), default or none" >&2; exit 2
fi
CQ_ARG=""; [[ "$CLS_QUOTA" != none ]] && CQ_ARG="--source-quota $CLS_QUOTA"
[[ "$CLS_SEL" != macro_f1 ]] && CQ_ARG="$CQ_ARG --select-on $CLS_SEL"
ENC_XSRC="${ENC_EXCLUDE_SOURCES-$XS_DEFAULT}"
WSLP_TWINS="${INDEX_WSLP_TWINS:-$TW_DEFAULT}"
# linker aliases (docs 15.34): research on, commercial-safe off; only the link step passes it (later steps read the
# work folder's link_settings.json; the start line prints the folder's real state when link is not run, docs 15.35)
LINK_ALIASES="${LINK_ALIASES:-$([[ "$PROFILE" == "research" ]] && echo 1 || echo 0)}"
case "$LINK_ALIASES" in 0|1) ;; *) echo "LINK_ALIASES must be 0 or 1" >&2; exit 2 ;; esac
LA_ARG=""; [[ $LINK_ALIASES == 1 ]] && LA_ARG="--link-aliases"
# variant forms (docs 15.36): the link step reads this work folder's variant_links.json when present (auto)
VARIANT_LINKS="${VARIANT_LINKS:-auto}"
case "$VARIANT_LINKS" in auto|none) ;; *) echo "VARIANT_LINKS must be auto or none" >&2; exit 2 ;; esac
VL_ARG=""; [[ $VARIANT_LINKS == none ]] && VL_ARG="--variant-links none"
[[ "${VARIANT_ACCEPT_NOISY:-0}" == 1 ]] && VL_ARG="$VL_ARG --accept-noisy"
[[ "${VARIANT_ALLOW_SPLIT_CHANGE:-0}" == 1 ]] && VL_ARG="$VL_ARG --allow-split-change"
if [[ $VARIANT_LINKS == none ]]; then VL_FILE="off"
elif [[ -f "$WORK/variant_links.json" ]]; then VL_FILE="sha256:$(sha256sum "$WORK/variant_links.json" | cut -c1-16)"
else VL_FILE="none"; fi
case "$DICT_GROUP" in 0|1) ;; *) echo "CLS_DICTIONARY_GROUP must be 0 or 1" >&2; exit 2 ;; esac
case "$WSLP_TWINS" in keep|drop|auto) ;; *) echo "INDEX_WSLP_TWINS must be keep, drop or auto" >&2; exit 2 ;; esac
[[ "$ENC_XSRC" =~ ^[a-z0-9_,]*$ ]] || { echo "ENC_EXCLUDE_SOURCES: comma-separated source names" >&2; exit 2; }
# the core sources are never left out (docs 15.33); train-encoder also refuses an unknown name (exit 2)
[[ ",$ENC_XSRC," =~ ,(include|islrtc), ]] && { echo "ENC_EXCLUDE_SOURCES: include and islrtc are core sources" >&2; exit 2; }
DG_ARG=""; [[ $DICT_GROUP == 1 ]] && DG_ARG="--dictionary-group"
XS_ARG=""; [[ -n "$ENC_XSRC" ]] && XS_ARG="--exclude-sources $ENC_XSRC"
TW_ARG=""; [[ "$WSLP_TWINS" != keep ]] && TW_ARG="--wslp-twins $WSLP_TWINS"
WSLP_SETTINGS="dict_group=$DICT_GROUP enc_exclude=${ENC_XSRC:--} wslp_twins=$WSLP_TWINS link_aliases=$LINK_ALIASES"
# the setting a step ran with, written into its stamp so the log's "skip ... (done ...)" lines show it (docs 15.33)
step_settings() {
  case "$1" in
    classifier|cv) echo " dict_group=$DICT_GROUP quota=$CLS_QUOTA select=$CLS_SEL" ;;
    e1a) echo " dict_group=$DICT_GROUP" ;;
    encoder|oneshot) echo " enc_exclude=${ENC_XSRC:--}" ;;
    index|index_final) echo " wslp_twins=$WSLP_TWINS" ;;
    link) echo " link_aliases=$LINK_ALIASES variant_links=$VL_FILE" ;;
  esac
}
# a skipped classifier / cv step whose stamp has other settings than this run's (e.g. a stamp from before the quota):
# kept as it is (never refused), but said, so a plain resume does not silently keep an old classifier (docs 15.39)
stamp_warn() {
  case "$1" in classifier|cv) ;; *) return 0 ;; esac
  local want
  want="$(step_settings "$1")"
  [[ "$(cat "$STAMPS/$1.done" 2>/dev/null)" == *"$want" ]] && return 0
  log "warning: $1 stamp settings differ from the current ones (${want# }); use --from $1 to retrain"
}
# a scratch run: a full run with a tag or its own work folder (smoke runs have their own private ISL root). It must
# not rewrite the shared data under data/isl (docs 15.20): the steps in SHARED_STEPS are refused there unless
# --allow-shared-data, and with --from the steps before STEP are seeded from SEED_WORK instead of run.
SCRATCH=0
[[ $SMOKE == 0 && -n "$BASE" ]] && SCRATCH=1
SEED_WORK="${PHASE2_SEED_BASE:-$DEFAULT_BASE}/$PROFILE"
SHARED_STEPS=(trim dedup splits)  # write data/isl/derived and data/isl/splits, which every run reads
# stamps / log: the real run (no tag and the default work folder; smoke: no tag) keeps the old place under
# data/isl/logs, so the finished runs stay "done"; every other run keeps them in its own work folder (docs 15.19)
if [[ -z "$RUN_TAG" && -z "$BASE" ]]; then
  LOGDIR="$ISL_ROOT/logs"
else
  LOGDIR="$WORK/driver"
fi
STAMPS="$LOGDIR/phase2_${NAME}.stamps"
LOG="$LOGDIR/phase2_${NAME}.log"

VOCAB="$WORK/vocabulary.jsonl"
COMMON="--profile $PROFILE --threads $THREADS --vocab $VOCAB $SM"
export SLR_TORCH_THREADS="$THREADS"
cd "$REPO/backend" || exit 1

log() { if [[ $DRY_RUN == 1 ]]; then echo "$*"; else echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; fi; }
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

state_files() {  # training state files of a step that exist (removed by --from / --force)
  local f x y
  case "$1" in
    # the clip list too: write_pretrain_list(resume=True) unions with an old list, so a retrain would record
    # clips it never saw in its meta (provenance / sha256)
    pretrain) f=("$WORK/pretrained_encoder.state.pt" "$WORK/pretrain_clips.txt") ;;
    encoder) f=("$WORK/encoder.pt.state") ;;
    oneshot) f=("$WORK"/oneshot/fold*/encoder.pt.state) ;;
    classifier) f=("$WORK/classifier.pt.state") ;;
    e1a) f=("$WORK/e1a/classifier.pt.state") ;;
    cv) f=("$WORK"/cv/fold*/classifier.pt.state) ;;
    *) return 0 ;;
  esac
  for x in "${f[@]}"; do
    # also a half-written atomic save (<state>.tmp<pid>) left by a killed process
    for y in "$x" "$x".tmp*; do [[ -f "$y" ]] && echo "$y"; done
  done
  return 0
}

forget() {  # forget a step (--from, --force): its stamp and its training state
  local s="$1" f
  for f in "$STAMPS/$s.done" $(state_files "$s"); do
    [[ -f "$f" ]] || continue
    if [[ $DRY_RUN == 1 ]]; then echo "would remove $f"; else rm -f -- "${f:?}" && log "  removed $f"; fi
  done
}

in_list() { local x n="$1"; shift; for x in "$@"; do [[ "$x" == "$n" ]] && return 0; done; return 1; }

step_outputs() {  # what a step leaves in the work folder for later steps (a leading ! = required); for seeding
  case "$1" in
    vocab) echo "!vocabulary.jsonl" ;;
    link) echo "extra_concepts.jsonl sources.json multisource.json link_settings.json" ;;
    pretrain_dedup) echo "pretrain_dedup.json" ;;
    pretrain) echo "!pretrained_encoder.pt pretrain_clips.txt" ;;
    encoder) echo "!encoder.pt" ;;
    oneshot) echo "oneshot" ;;
    classifier) echo "!classifier.pt" ;;
    cv) echo "cv" ;;
    e1a) echo "e1a" ;;
    fingerspell) echo "letters.pt" ;;
    index) echo "index" ;;
    index_final) echo "index_final" ;;
    fusion) echo "fusion.json" ;;
    trim_eval) echo "trim_eval.json" ;;
  esac
}

seed_step() {  # scratch run, a step before --from: take its outputs from SEED_WORK (never its state), stamp it
  local s="$1" o f
  for o in $(step_outputs "$s"); do
    f="${o#!}"
    if [[ -e "$WORK/$f" ]]; then continue; fi
    if [[ ! -e "$SEED_WORK/$f" ]]; then
      [[ "$o" == !* ]] && { echo "seed $s: $SEED_WORK/$f is missing" >&2; return 2; }
      continue
    fi
    if [[ $DRY_RUN == 1 ]]; then echo "would copy $SEED_WORK/$f"; continue; fi
    # files only, never training state (*.state, *.state.pt, half-written *.tmp*)
    (cd "$SEED_WORK" && find "$f" -type f ! -name '*.state' ! -name '*.state.pt' ! -name '*.tmp*' -print0 \
       | xargs -0 -r cp -p --parents -t "$WORK") || return 1
    log "  seed $s: copied $SEED_WORK/$f"
  done
  if [[ $DRY_RUN == 1 ]]; then echo "would seed $s"; return 0; fi
  echo "$(date -u +%FT%TZ) seeded from $SEED_WORK (scratch run --from $FROM)" > "$STAMPS/$s.done"
}

run_step() {
  local s="$1"
  case "$s" in
    trim) cli trim ;;  # incremental: clips without a trim, or whose trim_version, learned-model SHA-256 or landmark file changed
    dedup)
      if [[ $SMOKE == 1 ]]; then cli dedup --sources include wslp mendeley_emergency mendeley_agri; return $?; fi
      cli dedup --sources include islrtc cislr wslp isl500 mendeley_emergency mendeley_agri ;;
    vocab) cli vocab --no-trims --out "$VOCAB" ;;  # to the work folder (commit to models/packs/isl after review)
    splits) cli splits --vocab "$VOCAB" ;;  # fetches the official INCLUDE lists only when missing
    link) cli link-sources $COMMON --all-sources $CAP $LA_ARG $VL_ARG ;;
    bench) cli bench --threads "$THREADS" --steps $([[ $SMOKE == 1 ]] && echo 1 || echo 5) --hidden-sizes ${HIDDEN:-192} ;;
    pretrain_dedup) cli pretrain-dedup $COMMON --all-sources $CAP $PDD ;;  # Sign Library copies / eval-clip matches
    pretrain) train pretrain $COMMON $CAP $PRE ;;
    encoder) train train-encoder $COMMON --all-sources $CAP $ENC --init auto $XS_ARG ;;
    oneshot) train train-oneshot $COMMON --all-sources $CAP $ONE --init auto $XS_ARG ;;
    classifier) train train-classifier $COMMON --all-sources $CAP $CLS $DG_ARG $CQ_ARG ;;
    cv)  # classifiers for the other folds of the primary split: E1b mean / spread and a signer-grouped interval
      local base
      base=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('primary_split',''))" \
             "${SROOT:-$REPO/data/isl}/splits/splits_report.json" 2>/dev/null)
      if [[ -z "$base" ]]; then log "  no splits_report.json: skipped"; return 0; fi
      for k in ${CV_FOLDS-$([[ $SMOKE == 1 ]] && echo 1 || echo 1 2 3 4)}; do
        mkdir -p "$WORK/cv/fold$k"
        train train-classifier $COMMON --all-sources $CAP $CLS $DG_ARG $CQ_ARG --split "$base-fold$k" --ckpt "$WORK/cv/fold$k/classifier.pt" || return $?
      done ;;
    e1a)  # E1a needs its own model trained on the official (not signer-independent) INCLUDE split
      if [[ ! -f "${SROOT:-$REPO/data/isl}/splits/include-official.json" ]]; then log "  no official split: skipped"; return 0; fi
      mkdir -p "$WORK/e1a"
      train train-classifier $COMMON $CAP $CLS $E1A_PAT $DG_ARG --batches ${E1A_STEPS:-$([[ $SMOKE == 1 ]] && echo 6 || echo 120)} --split include-official --ckpt "$WORK/e1a/classifier.pt" ;;
    fingerspell) cli train-fingerspell --profile "$PROFILE" --threads "$THREADS" --vocab "$VOCAB" $SM $LET ;;
    # training parts only, into <work>/index: fit-fusion and evaluate read it and refuse anything else (docs 15.21).
    # A rebuilt index makes the later steps that used or must follow it stale: forget their stamps (docs 15.22),
    # so an older index_final (other data / row logic) is never shipped next to it
    index) cli build-index $COMMON --all-sources $CAP $TW_ARG --out "$WORK/index" \
        && rm -f "$STAMPS/fusion.done" "$STAMPS/evaluate.done" "$STAMPS/index_final.done" "$STAMPS/pack.done" ;;
    fusion) cli fit-fusion $COMMON --all-sources $CAP ;;
    trim_eval)  # E8: evaluate exactly the trimmer of the trim step (same learned-model choice) into the work folder
      # a new trim-eval makes an older report / pack stale: forget their stamps so they are rebuilt from it
      cli trim-eval --learned-model auto --out "$WORK/trim_eval.json" $([[ $SMOKE == 1 ]] && echo "--n ${SMOKE_TRIM_EVAL_N:-60}") \
        && rm -f "$STAMPS/evaluate.done" "$STAMPS/pack.done" ;;
    evaluate) cli evaluate --all $COMMON --all-sources $CAP ${PACK_OUT:+--out "$PACK_OUT"} \
        ${COMPARE_TO:+--compare-to "$COMPARE_TO"} ;;
    # the shipped index: every reference (val / test too) into <work>/index_final, never over <work>/index
    index_final) cli build-index $COMMON --all-sources $CAP $TW_ARG --all-include --out "$WORK/index_final" ;;
    pack) cli build-pack $COMMON ${PACK_OUT:+--out "$PACK_OUT"} ;;
    *) log "unknown step $s"; return 2 ;;
  esac
}

if [[ $DRY_RUN == 1 ]]; then
  echo "dry run: profile=$PROFILE smoke=$SMOKE tag=${RUN_TAG:--} work=$WORK"
  echo "stamps: $STAMPS"
  echo "log: $LOG"
  echo "pack: ${PACK_OUT:-default (artifacts/packs/<id>)}"
  [[ -n "$COMPARE_TO" ]] && echo "evaluate compares to: $COMPARE_TO"
  echo "wslp/dictionary: classifier dictionary group=$DICT_GROUP, encoder excludes=${ENC_XSRC:--}, index wslp twins=$WSLP_TWINS"
  echo "linker: aliases and WSLP number stripping=$LINK_ALIASES (link step only)"
  echo "classifier: source quota=$CLS_QUOTA, selection=$CLS_SEL (classifier and cv), settings: $CLS"
  echo "variant links: VARIANT_LINKS=$VARIANT_LINKS, $WORK/variant_links.json: $VL_FILE (link step only)"
fi

if [[ $LIST == 1 ]]; then
  for s in "${STEPS[@]}"; do printf '%-15s %s\n' "$s" "$(cat "$STAMPS/$s.done" 2>/dev/null || echo -)"; done
  [[ -f "$WORK/partial_data.json" ]] && echo "work folder is marked PARTIAL DATA ($WORK/partial_data.json)"
  exit 0
fi

# ---- plan: what --from / --force forget, which steps are seeded, run or skipped ------------------------------
# --from is applied eagerly: STEP and every later step are forgotten in one pass before the first step runs, so a
# failed run is resumed by running again WITHOUT --from (re-running with --from would forget finished steps again)
FORGET=("${FORCE_STEPS[@]}")
SEED=()
if [[ -n "$FROM" && -z "$ONLY" ]]; then
  reached=0
  for s in "${STEPS[@]}"; do
    [[ "$s" == "$FROM" ]] && reached=1
    if [[ $reached == 1 ]]; then in_list "$s" "${FORGET[@]}" || FORGET+=("$s"); continue; fi
    # scratch run: an earlier step without a stamp is seeded from the real work folder, never run (docs 15.20)
    if [[ $SCRATCH == 1 && ! -f "$STAMPS/$s.done" ]] && ! in_list "$s" "${FORGET[@]}"; then SEED+=("$s"); fi
  done
fi
RUN=()
for s in "${STEPS[@]}"; do
  if [[ -n "$ONLY" ]]; then [[ "$s" == "$ONLY" ]] && RUN+=("$s"); continue; fi
  in_list "$s" "${SEED[@]}" && continue
  if in_list "$s" "${FORGET[@]}" || [[ ! -f "$STAMPS/$s.done" ]]; then RUN+=("$s"); fi
done
# the linker setting this run really has (docs 15.35): LINK_ALIASES acts only when the link step runs; otherwise the
# work folder's link_settings.json (written by its link step) decides, and the start line says which
if in_list link "${RUN[@]}"; then
  LA_STATE="$LINK_ALIASES"
else
  la_dir="$WORK"; in_list link "${SEED[@]}" && la_dir="$SEED_WORK"
  if [[ ! -f "$la_dir/link_settings.json" ]]; then la_folder="none"
  elif grep -q '"aliases": true' "$la_dir/link_settings.json"; then la_folder="on"
  else la_folder="off"; fi
  LA_STATE="$LINK_ALIASES (link step not run; folder: $la_folder)"
  if [[ $LINK_ALIASES == 1 && $la_folder != on ]] || [[ $LINK_ALIASES == 0 && $la_folder == on ]]; then
    LA_WARN="LINK_ALIASES=$LINK_ALIASES but the link step is not run and the work folder's link settings are $la_folder: this run links as the folder says (run --from link to change it)"
  fi
fi
WSLP_SETTINGS="${WSLP_SETTINGS% link_aliases=*} link_aliases=$LA_STATE"
# variant links (docs 15.36): the file the link step reads, or the folder's frozen state when link is not run
if in_list link "${RUN[@]}"; then
  VL_STATE="$VL_FILE"
else
  vl_dir="$WORK"; in_list link "${SEED[@]}" && vl_dir="$SEED_WORK"
  if [[ ! -f "$vl_dir/link_settings.json" ]]; then vl_folder="none"
  elif grep -q '"variants"' "$vl_dir/link_settings.json"; then vl_folder="on"
  else vl_folder="off"; fi
  VL_STATE="(link step not run; folder: $vl_folder)"
fi
WSLP_SETTINGS="$WSLP_SETTINGS variant_links=$VL_STATE"
# the classifier's source quota, checked in full before anything runs (docs 15.39): an unknown name, a word source of
# the profile with no share or all shares 0 would otherwise fail only when the classifier step starts, hours into a
# run. Checked against every word source of the profile (stricter than the sources a run really has: fine here).
if [[ "$CLS_QUOTA" != none ]] && { in_list classifier "${RUN[@]}" || in_list cv "${RUN[@]}"; }; then
  # (one line, torch-free imports: quick, and any python the tests put in $PY can run it)
  if ! "$PY" -c 'import sys; from app.isl.profiles import WORD_SOURCES as W; from app.isl.training.source_quota import check_source_quota as c; sys.exit(c(sys.argv[1], W[sys.argv[2]]))' "$CLS_QUOTA" "$PROFILE"; then
    echo "CLS_SOURCE_QUOTA=$CLS_QUOTA refused for profile $PROFILE (nothing was run)" >&2
    exit 2
  fi
fi
# --only link with another variant-links state than the folder's frozen one would leave the later steps' stamps on
# models trained with the old labels (docs 15.37): refused, use --from link
if [[ "$ONLY" == link ]]; then
  if [[ ! -f "$WORK/link_settings.json" ]] || ! grep -q '"variants"' "$WORK/link_settings.json"; then vl_frozen="none"
  else
    vl_frozen="sha256:$(tr -d '\n' < "$WORK/link_settings.json" | grep -o '"variant_links": *{[^}]*}' \
      | grep -o '"sha256": *"[0-9a-f]*"' | head -1 | sed 's/.*"\([0-9a-f]*\)"$/\1/' | cut -c1-16)"
  fi
  vl_now="$VL_FILE"; [[ "$vl_now" == off ]] && vl_now="none"
  if [[ "$vl_now" != "$vl_frozen" ]]; then
    echo "--only link: variant links would change ($vl_frozen -> $vl_now): the later steps were trained with the old" >&2
    echo "  labels. Run --from link instead." >&2
    exit 2
  fi
fi
[[ $DRY_RUN == 1 ]] && echo "linker this run: link_aliases=$LA_STATE variant_links=$VL_STATE"
# (not for --check-only, which runs no step)
if [[ $SCRATCH == 1 && $ALLOW_SHARED == 0 && $CHECK_ONLY == 0 ]]; then
  for s in "${RUN[@]}"; do
    if in_list "$s" "${SHARED_STEPS[@]}"; then
      echo "scratch run: step $s would rewrite the shared data under $ISL_ROOT that the real run reads. Start after" >&2
      echo "  splits (e.g. --from bench: earlier steps are seeded from $SEED_WORK), or pass --allow-shared-data." >&2
      exit 2
    fi
  done
fi
[[ $CHECK_ONLY == 1 ]] && SEED=()
for s in "${SEED[@]}"; do
  for o in $(step_outputs "$s"); do
    if [[ "$o" == !* && ! -e "$WORK/${o#!}" && ! -e "$SEED_WORK/${o#!}" ]]; then
      echo "scratch run --from $FROM: step $s is seeded, but $SEED_WORK/${o#!} is missing (set PHASE2_SEED_BASE," >&2
      echo "  or start from an earlier step)" >&2
      exit 2
    fi
  done
done

# ---- dry run: what would be removed, seeded, run or skipped (writes nothing) --------------------------------
if [[ $DRY_RUN == 1 ]]; then
  if [[ "$FROM" == "${STEPS[0]}" && -z "$ONLY" && -f "$WORK/partial_data.json" ]]; then
    echo "would remove $WORK/partial_data.json"
  fi
  for s in "${FORGET[@]}"; do forget "$s"; done
  for s in "${SEED[@]}"; do seed_step "$s" || exit $?; done
  for s in "${STEPS[@]}"; do
    if in_list "$s" "${RUN[@]}"; then echo "would run $s"
    elif [[ -z "$ONLY" ]] && ! in_list "$s" "${SEED[@]}"; then
      echo "would skip $s (done $(cat "$STAMPS/$s.done"))"; stamp_warn "$s"
    fi
  done
  exit 0
fi

mkdir -p "$LOGDIR" "$STAMPS"  # only now: a refused run (bad option, shared-data step, missing seed) writes nothing

# ---- data guard: full runs never start silently on partial data -------------------------------------------
[[ "$FROM" == "${STEPS[0]}" && -z "$ONLY" ]] && rm -f "$WORK/partial_data.json"  # rebuilt from scratch
guard_args=(readiness --profile "$PROFILE" --out "$WORK/data_readiness.json" --work "$WORK")
if [[ $SMOKE == 1 || $ALLOW_PARTIAL == 1 ]]; then guard_args+=(--allow-partial); fi
nice -n 10 "$PY" -m app.isl.cli $ROOTARG "${guard_args[@]}" > "$LOGDIR/phase2_${NAME}.readiness.out" 2>&1
grc=$?
cat "$LOGDIR/phase2_${NAME}.readiness.out" >> "$LOG"
if [[ $grc -ne 0 ]]; then
  log "data guard: NOT READY - refusing to start a full run (an ingestion job is running or a manifest is incomplete;"
  log "  details: $WORK/data_readiness.json). Wait for ingestion to finish, or pass --allow-partial (results and"
  log "  packs are then labelled PARTIAL DATA)."
  grep -A40 '"reasons"' "$LOGDIR/phase2_${NAME}.readiness.out" | sed -n '1,25p' >&2
  exit 4
fi
if [[ -f "$WORK/partial_data.json" ]]; then
  log "data guard: work folder marked PARTIAL DATA ($WORK/partial_data.json): reports and the pack will say so"
else
  log "data guard: ready (no ingestion job running, every manifest of $PROFILE complete)"
fi
[[ $CHECK_ONLY == 1 ]] && exit 0
mkdir -p "$WORK"
free_gb=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
if [[ "$free_gb" -lt 10 ]]; then log "only ${free_gb} GB free (< 10 GB floor): not starting"; exit 3; fi

log "phase 2 driver: profile=$PROFILE smoke=$SMOKE tag=${RUN_TAG:--} allow_partial=$ALLOW_PARTIAL threads=$THREADS work=$WORK stamps=$STAMPS${COMPARE_TO:+ compare_to=$COMPARE_TO} $WSLP_SETTINGS cls_quota=$CLS_QUOTA cls_select=$CLS_SEL git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
[[ -n "${LA_WARN:-}" ]] && log "warning: $LA_WARN"
if [[ ${#SEED[@]} -gt 0 ]]; then
  for s in "${SEED[@]}"; do seed_step "$s" || { log "seeding $s FAILED"; exit 2; }; done
  if [[ -f "$SEED_WORK/partial_data.json" && ! -f "$WORK/partial_data.json" ]]; then
    cp -p "$SEED_WORK/partial_data.json" "$WORK/" && log "  seeded from a PARTIAL DATA work folder: this run is marked too"
  fi
fi
for s in "${FORGET[@]}"; do forget "$s"; done  # eager: STEP and every later step at once (see the plan above)
for s in "${STEPS[@]}"; do
  if [[ -n "$ONLY" && "$s" != "$ONLY" ]]; then continue; fi
  if [[ -z "$ONLY" && -f "$STAMPS/$s.done" ]]; then log "skip $s (done $(cat "$STAMPS/$s.done"))"; stamp_warn "$s"; continue; fi
  if [[ $SCRATCH == 1 && $ALLOW_SHARED == 0 ]] && in_list "$s" "${SHARED_STEPS[@]}"; then
    log "step $s refused: a scratch run must not rewrite the shared data (--allow-shared-data)"; exit 2
  fi
  free_gb=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
  if [[ "$free_gb" -lt 10 ]]; then log "only ${free_gb} GB free: stopping before $s"; exit 3; fi
  log "step $s ..."
  t0=$(date +%s)
  run_step "$s"; rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ -n "$ONLY" ]]; then
      log "step $s FAILED (exit $rc); fix the cause and run the same --only command again (it resumes from its state)"
    else
      log "step $s FAILED (exit $rc); fix the cause and run again WITHOUT --from/--force to resume here"
    fi
    exit $rc
  fi
  echo "$(date -u +%FT%TZ) $(( $(date +%s) - t0 ))s$(step_settings "$s")" > "$STAMPS/$s.done"
  log "step $s done in $(( $(date +%s) - t0 )) s"
done
log "phase 2 driver finished: profile=$PROFILE smoke=$SMOKE (report: $WORK/eval/report.md)"
