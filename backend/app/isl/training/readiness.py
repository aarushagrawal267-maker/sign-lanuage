"""Is the ISL data complete enough for a full (non-smoke) phase-2 run? (docs/ISL_DESIGN.md 15.3 / 15.4)

Two checks, both required before ``scripts/phase2_train.sh`` starts a full run:

* **no ingestion job is running**: a process whose command line runs ``scripts/ingest_isl.py``,
  ``ingest_isl_pose.py``, ``ingest_isl_video.py`` (except the read-only ``status`` / ``sources`` / ``probe``
  sub-commands), ``ingest_isl_gated.py cislr|isign`` (not its ``--check`` access probe) or ``isl_queue.sh``.
  The queue counts only until its log says "all listed sources processed":
  after that it only polls the gated CISLR dataset every 6 h for 14 days, which must not block training.
* **every manifest the profile reads is complete**: rows (ok or failed) for every expected item. Expected counts
  come from the ingestion item lists (``manifests/<source>_items.json``, ``islrtc_index.json``), the GloSL id list,
  or the published dataset sizes. Gated sources without access (CISLR, iSign) are "unavailable", not incomplete;
  OpenHands raw is budget-limited, so it is complete once its job has ended. A failed row counts as attempted only
  when its failure is permanent: rows whose last record is a download / fetch failure or a ``superseded`` marker are
  retried by the next ingestion run, so they count as missing. A public source with no rows and no item list (its
  listing failed, e.g. Hugging Face 429 / 503, and the queue gave up) is incomplete, not "not available".

``--allow-partial`` overrides the check; the run is then marked with ``<work>/partial_data.json`` (sticky: every
later step, report and pack of that work folder says PARTIAL DATA until the work folder is rebuilt from the first
step). :func:`partial_data_flag` is what ``evaluate`` and ``build-pack`` read.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from app.isl.paths import IslPaths

INGEST_SCRIPTS = ("ingest_isl.py", "ingest_isl_pose.py", "ingest_isl_video.py", "ingest_isl_gated.py",
                  "isl_queue.sh")
# ingest_isl_gated.py without a source, or with --check anywhere, only probes access (the queue runs that every cycle)
GATED_SCRIPT = "ingest_isl_gated.py"
READ_ONLY_SUBCOMMANDS = {"status", "sources", "probe", "-h", "--help"}
QUEUE_DONE_MARK = "all listed sources processed"
QUEUE_START_MARK = "queue start"
PARTIAL_MARKER = "partial_data.json"

# published sizes for sources without an item list on disk
KNOWN_TOTALS = {"include": 4292, "isl500": 86756, "openhands_fs": 172, "glosl": 126846}
# INCLUDE: the zenodo record (4010759) has 44 zips; scripts/ingest_isl.py appends a zip to include_zips_done.txt
# once all its videos have rows. The zips hold 4,284 distinct clip ids, not the published 4,292, so the row count
# alone would never reach the published size (measured 25 Nov 2026, 19:59 UTC, when the job finished).
INCLUDE_ZIPS = 44
GATED = ("cislr", "isign")
BUDGET_LIMITED = ("openhands_raw",)  # ingested up to a disk budget: complete when its job has ended
# items file of a source when it differs from "<source>_items.json" (manifest shared with another source)
ITEMS_FILE = {"wslp_unlabelled": "wslp_unlabelled_items.json"}
MANIFEST_OF = {"wslp_unlabelled": "wslp"}
# failures the next ingestion run retries (video engine "fetch: ...", ingest_isl.py "download failed: ...", rows
# marked "superseded: ... re-extract"): such an item has not really been attempted yet
RETRY_ERROR_PREFIXES = ("fetch:", "download failed", "superseded")


# --------------------------------------------------------------------------------------------- processes

def _argv(pid_dir: Path) -> list[str]:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return []
    return [x.decode("utf-8", "replace") for x in raw.split(b"\0") if x]


def queue_polling_only(queue_log: Path | None) -> bool:
    """True when the queue has processed its whole list (it now only polls CISLR)."""
    if queue_log is None or not queue_log.exists():
        return False
    try:
        with queue_log.open("rb") as fh:
            fh.seek(max(0, queue_log.stat().st_size - 400_000))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return False
    start = text.rfind(QUEUE_START_MARK)
    return text.find(QUEUE_DONE_MARK, max(0, start)) >= 0


def ingestion_processes(proc_root: Path | None = None, queue_log: Path | None = None,
                        self_pid: int | None = None) -> list[dict]:
    """Running ingestion jobs: ``[{pid, script, subcommand, cmd, blocking}]`` (``blocking`` False only for a queue
    that is just polling CISLR)."""
    proc_root = Path(proc_root or os.environ.get("SLR_ISL_PROC_ROOT") or "/proc")
    me = {self_pid or os.getpid(), os.getppid()}
    out = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.name.isdigit() or int(d.name) in me:
            continue
        argv = _argv(d)
        for i, tok in enumerate(argv):
            script = Path(tok).name
            if script not in INGEST_SCRIPTS:
                continue
            sub = argv[i + 1] if i + 1 < len(argv) else ""
            if script != "isl_queue.sh" and sub in READ_ONLY_SUBCOMMANDS:
                break
            if script == GATED_SCRIPT:  # iSign / CISLR conversion; "--check" (anywhere) or no source = probe only
                rest = argv[i + 1:]
                if "--check" in rest or not any(a in ("cislr", "isign") for a in rest):
                    break
                sub = next(a for a in rest if a in ("cislr", "isign"))
            blocking = not (script == "isl_queue.sh" and queue_polling_only(queue_log))
            out.append({"pid": int(d.name), "script": script, "subcommand": sub, "cmd": " ".join(argv)[:200],
                        "blocking": blocking})
            break
    return sorted(out, key=lambda r: r["pid"])


# --------------------------------------------------------------------------------------------- manifests

def _ids_of_items(path: Path) -> set[str] | int | None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(doc, list):
        ids = {x.get("id") for x in doc if isinstance(x, dict) and x.get("id")}
        return ids if ids else len(doc)
    return None


def _manifest_ids(path: Path) -> tuple[set[str], set[str]]:
    """(attempted ids, ok ids) of a manifest (last row per id wins, like ``read_manifest``). An id whose last row
    is a retryable failure (:data:`RETRY_ERROR_PREFIXES`) is not attempted yet: the next ingestion run redoes it."""
    last: dict[str, bool | None] = {}  # True ok, False permanent failure, None retryable failure
    if not path.exists():
        return set(), set()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except ValueError:
                continue  # a line being written right now
            if isinstance(r, dict) and r.get("id"):
                if r.get("ok"):
                    last[r["id"]] = True
                else:
                    err = str(r.get("error") or "").lstrip().lower()
                    last[r["id"]] = None if err.startswith(RETRY_ERROR_PREFIXES) else False
    return {k for k, v in last.items() if v is not None}, {k for k, v in last.items() if v}


def _retry_count(path: Path) -> int:
    """Ids whose last row is a retryable failure (reported so a stuck download is visible)."""
    if not path.exists():
        return 0
    last: dict[str, bool] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("id"):
                last[r["id"]] = (not r.get("ok")) and str(r.get("error") or "").lstrip().lower().startswith(
                    RETRY_ERROR_PREFIXES)
    return sum(last.values())


def expected_items(paths: IslPaths, name: str) -> tuple[set[str] | int | None, str]:
    """Expected item ids (or a count) of a source and where that comes from."""
    items = paths.manifests / ITEMS_FILE.get(name, f"{name}_items.json")
    if items.exists():
        v = _ids_of_items(items)
        if v is not None:
            return v, items.name
    if name == "islrtc" and paths.islrtc_index.exists():
        try:
            return len(json.loads(paths.islrtc_index.read_text())), paths.islrtc_index.name
        except ValueError:
            pass
    if name == "glosl":
        f = paths.derived / "glosl_isign_uids.txt"
        if f.exists():
            return sum(1 for ln in f.open() if ln.strip()), f.name
    if name in KNOWN_TOTALS:
        return KNOWN_TOTALS[name], "published dataset size"
    return None, "unknown"


def source_status(paths: IslPaths, name: str, running_scripts: set[str] | None = None) -> dict:
    man = paths.manifest(MANIFEST_OF.get(name, name))
    ids, ok = _manifest_ids(man)
    exp, how = expected_items(paths, name)
    st = {"source": name, "manifest": man.name, "rows": len(ids), "ok": len(ok), "failed": len(ids) - len(ok),
          "retry_pending": _retry_count(man), "expected_from": how}
    if name == "include":
        zf = paths.manifests / "include_zips_done.txt"
        zdone = len(set(zf.read_text().split())) if zf.exists() else 0
        if zdone >= INCLUDE_ZIPS:
            st.update(expected=len(ids), done=len(ids), complete=bool(ids),
                      expected_from=f"all {INCLUDE_ZIPS} INCLUDE zips processed (include_zips_done.txt)")
            return st
    if name in BUDGET_LIMITED:
        running = bool(running_scripts and "ingest_isl_pose.py" in running_scripts and (paths.manifests /
                                                                                          f"{name}.lock").exists())
        st.update(expected=None, complete=bool(ids) and not running,
                  note="budget-limited: complete once its ingestion job has ended" if ids else "not ingested")
        return st
    if isinstance(exp, set):
        have = len(ids & exp)
        st.update(expected=len(exp), done=have, complete=have >= len(exp))
    elif isinstance(exp, int):
        st.update(expected=exp, done=len(ids), complete=len(ids) >= exp)
    else:
        st.update(expected=None, done=len(ids), complete=None)
    if not ids and name in GATED:
        st.update(complete=None, note="gated: no access yet (unavailable, not incomplete)")
    elif not ids and st.get("expected") is None:
        # a public source: its listing failed (e.g. HF 429 / 503) or it was never run; training without it must be
        # a decision (--allow-partial), not a silent default
        st.update(complete=False, note="nothing ingested and no item list (listing failed or never run)")
    if st["retry_pending"] and st.get("complete"):
        st["complete"] = False
    if st["retry_pending"]:
        st["note"] = (st.get("note", "") + "; " if st.get("note") else "") + \
            f"{st['retry_pending']} download / superseded rows to retry (re-run the source)"
    return st


def profile_source_names(profile: str) -> list[str]:
    from app.isl.profiles import get_profile

    prof = get_profile(profile)
    return list(dict.fromkeys(["include", "islrtc", *prof.sources, *prof.pretraining, *prof.letters]))


def check(paths: IslPaths, profile: str, proc_root: Path | None = None, queue_log: Path | None = None) -> dict:
    """``{"ready": bool, "reasons": [...], "processes": [...], "sources": {name: status}}``."""
    procs = ingestion_processes(proc_root, queue_log if queue_log is not None else paths.root / "logs" / "queue.log")
    blocking = [p for p in procs if p["blocking"]]
    scripts = {p["script"] for p in procs}
    sources = {n: source_status(paths, n, scripts) for n in profile_source_names(profile)}
    reasons = [f"ingestion job running: pid {p['pid']} {p['script']} {p['subcommand']}".strip() for p in blocking]
    for n, st in sources.items():
        if st["complete"] is False:
            reasons.append(f"manifest incomplete: {n} {st.get('done', st['rows'])}/{st.get('expected') or '?'}"
                           + (f" ({st['note']})" if st.get("note") else ""))
    return {"ready": not reasons, "profile": profile, "checked": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reasons": reasons, "processes": procs, "sources": sources}


# --------------------------------------------------------------------------------------------- partial flag

def mark_partial(work: Path, report: dict) -> Path:
    """Write the sticky partial-data marker of a work folder."""
    work.mkdir(parents=True, exist_ok=True)
    f = work / PARTIAL_MARKER
    prev = json.loads(f.read_text()) if f.exists() else {}
    doc = {"partial_data": True, "first_marked": prev.get("first_marked") or report.get("checked"),
           "last_marked": report.get("checked"), "reasons": report.get("reasons", [])[:50]}
    f.write_text(json.dumps(doc, indent=1))
    return f


def partial_data_flag(work: Path | None) -> bool:
    """True when a work folder was (partly) built on partial data, or the environment says so."""
    if os.environ.get("SLR_ISL_PARTIAL_DATA", "") not in ("", "0"):
        return True
    return bool(work is not None and (Path(work) / PARTIAL_MARKER).exists())
