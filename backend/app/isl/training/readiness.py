"""Is the ISL data complete enough for a full (non-smoke) phase-2 run? (docs/ISL_DESIGN.md 15.3 / 15.4)

Two checks, both required before ``scripts/phase2_train.sh`` starts a full run:

* **no ingestion job is running** that writes a source of the profile (a job of a single known source, e.g.
  ``ingest_isl_pose.py openhands-raw`` or ``ingest_isl_video.py list hf_akrit_alphabet``, blocks only the profiles
  that read that source; any job whose source is unknown blocks every profile): a process whose command line runs
  ``scripts/ingest_isl.py``, ``ingest_isl_pose.py``, ``ingest_isl_video.py`` (except the read-only ``status`` /
  ``sources`` / ``probe`` sub-commands), ``ingest_isl_gated.py cislr|isign`` (not its ``--check`` access probe) or
  ``isl_queue.sh``.
  The queue (``scripts/isl_queue.sh``) blocks only the profiles that read a source it still has to process
  (:func:`queue_progress`): its list is the ``queue start (pid N): ...`` line it logged, cross-checked against the
  process's ``SOURCES`` environment variable or the script's default; its progress is the per-source lines after
  that start line (a source is behind the queue once it logged ``finished``, ``giving up`` or ``gated dataset not
  accessible``); its running child (``ingest_isl_video.py run <source>``) is always counted as still to process; and
  while the list runs it may start CISLR / iSign (probed before every chunk), so those count too until they logged
  ``finished``. Anything that cannot be read or does not agree (no start line for that pid, a list that differs from
  the script, an unknown name) blocks every profile (fail closed). Once the log says "all listed sources processed"
  the queue only polls the gated datasets every 6 h for 14 days, which blocks nothing (a CISLR / iSign job it starts
  then is a process of its own and blocks research).
* **every manifest the profile reads is complete**: rows (ok or failed) for every expected item. Expected counts
  come from the ingestion item lists (``manifests/<source>_items.json``, ``islrtc_index.json``), the GloSL id list,
  or the published dataset sizes. Gated sources without access (CISLR, iSign) are "unavailable", not incomplete; a
  gated source whose manifest is partly filled while no job writes it is "partial (gated; job not running)": it does
  not block (both are research-only and access may never be granted again), but it is listed in ``warnings`` of the
  report (``data_readiness.json``) and in the evaluation report's data section (:func:`readiness_summary`);
  OpenHands raw is budget-limited, so it is complete once its job has ended normally (its log ends with "all members
  done", a disk-guard stop or a deliberate SIGINT; a crash keeps it incomplete). A failed row counts as attempted only
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
import re
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
GATED_PARTIAL = "partial (gated; job not running)"
GATED_RUNNING = "partial (gated; job running)"
BUDGET_LIMITED = ("openhands_raw",)  # ingested up to a disk budget: complete when its job has ended normally
# the job's log (``nohup ... >> data/isl/logs/<file>``) and the last lines that mean it ended on purpose: every member
# done, the disk guard (budget / min free) stopped it, or a deliberate SIGINT (ISL_DESIGN 15.6, command 4). Anything
# else (a traceback such as h5py's "bad symbol table node signature", a kill, a memory give-up) is a crash: the source
# stays incomplete until the job is restarted (it resumes) or the run is started with --allow-partial.
BUDGET_LOGS = {"openhands_raw": "ingest_pose_openhands_raw.log"}
BUDGET_CLEAN_END = ("openhands-raw: all members done", "openhands-raw: stopping: ", "KeyboardInterrupt")
BUDGET_LOG_TAIL = 200_000
# items file of a source when it differs from "<source>_items.json" (manifest shared with another source)
ITEMS_FILE = {"wslp_unlabelled": "wslp_unlabelled_items.json"}
MANIFEST_OF = {"wslp_unlabelled": "wslp"}
# failures the next ingestion run retries (video engine "fetch: ...", ingest_isl.py "download failed: ...", rows
# marked "superseded: ... re-extract"): such an item has not really been attempted yet
RETRY_ERROR_PREFIXES = ("fetch:", "download failed", "superseded")


# --------------------------------------------------------------------------------------------- processes

QUEUE_SCRIPT = "isl_queue.sh"
QUEUE_LOG_TAIL = 8_000_000  # bytes of queue.log read (the log grows by ~3 KB per hour of list work)
# queue list names that write another source's rows (``ingest_isl_video.py islrtc-full`` -> ISLRTC)
QUEUE_SOURCE_ALIASES = {"islrtc-full": "islrtc"}
# the queue's run_source() returned for that source: "finished" (0), "giving up for now" (2, also after too many
# memory stops), "gated dataset not accessible yet" (3); the queue then moves to the next listed source
QUEUE_TERMINAL = ("finished", "giving up for now", "gated dataset not accessible yet")
# options of the ingestion scripts that take one value (argparse accepts them before the source / sub-command too:
# ``ingest_isl_pose.py --budget-gb 26 openhands-raw``), so their values are never read as the source
VALUE_OPTIONS = ("--root", "--workers", "--limit", "--tmp", "--max-s", "--out-subdir", "--poll-min", "--max-minutes",
                 "--min-free-gb", "--per-class", "--max-rss-gb", "--dtype", "--target-fps", "--budget-gb",
                 "--convert-workers", "--respawn-on-memory", "--subset", "--skip-uids", "--mode", "--block-mb")
# ``ingest_isl_pose.py`` sources -> the profile source they write (isl500-links writes derived/isl500_links.jsonl,
# which the ISL500 adapter reads)
POSE_SOURCES = {"glosl": "glosl", "isl500": "isl500", "isl500-links": "isl500", "openhands-raw": "openhands_raw",
                "openhands-fs": "openhands_fs"}
_STAMP = r"\d{4}-\d\d-\d\d(?: \d\d:\d\d:\d\d)?"
_LOG_EVENT = re.compile(rf"^{_STAMP} ([A-Za-z0-9_.-]+): (.*)$")
_LOG_START = re.compile(rf"^{_STAMP} queue start \(pid (\d+)\):(.*)$")
_SOURCES_DEFAULT = re.compile(r'^\s*SOURCES="\$\{SOURCES:-([^}"]*)\}"', re.MULTILINE)


def _argv(pid_dir: Path) -> list[str]:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return []
    return [x.decode("utf-8", "replace") for x in raw.split(b"\0") if x]


def _ppid(pid_dir: Path) -> int | None:
    try:
        for ln in (pid_dir / "status").read_text(errors="replace").splitlines():
            if ln.startswith("PPid:"):
                return int(ln.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _environ_var(pid_dir: Path, var: str) -> tuple[bool, str | None]:
    """(environment readable, value of ``var`` or None when unset)."""
    try:
        raw = (pid_dir / "environ").read_bytes()
    except OSError:
        return False, None
    pre = var.encode() + b"="
    for x in raw.split(b"\0"):
        if x.startswith(pre):
            return True, x[len(pre):].decode("utf-8", "replace")
    return True, None


def _tail(path: Path | None, size: int) -> str | None:
    if path is None:
        return None
    try:
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - size))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return None


def _positional(rest: list[str]) -> list[str]:
    """Positional arguments of an ingestion command line (options and the values of :data:`VALUE_OPTIONS` dropped)."""
    out, skip = [], False
    for a in rest:
        if skip:
            skip = False
        elif a in VALUE_OPTIONS:
            skip = True
        elif not a.startswith("-"):
            out.append(a)
    return out


def default_queue_script() -> Path:
    from app.core.config import REPO_ROOT

    return Path(os.environ.get("SLR_ISL_QUEUE_SCRIPT") or REPO_ROOT / "scripts" / QUEUE_SCRIPT)


def queue_default_sources(script: Path | None = None) -> list[str] | None:
    """The ``SOURCES`` default of ``scripts/isl_queue.sh`` (``SOURCES="${SOURCES:-a b c}"``), or None."""
    try:
        text = Path(script or default_queue_script()).read_text(errors="replace")
    except OSError:
        return None
    m = _SOURCES_DEFAULT.findall(text)
    return m[-1].split() if len(m) == 1 and m[0].split() else None


def queue_source_names() -> set[str]:
    """Names the queue may list: the video sources of ``ingest_isl_video.py`` plus :data:`QUEUE_SOURCE_ALIASES`."""
    from app.isl.data.video_sources.catalog import SOURCES

    return set(SOURCES) | set(QUEUE_SOURCE_ALIASES)


def written_sources(name: str) -> set[str]:
    """Profile source names whose data a job of ``name`` writes (its own name and the manifest it shares, e.g.
    ``wslp_unlabelled`` -> ``wslp``; ``islrtc-full`` -> ``islrtc``)."""
    from app.isl.data.video_sources.catalog import SOURCES

    src = QUEUE_SOURCE_ALIASES.get(name, name.replace("-", "_"))
    out = {src, MANIFEST_OF.get(src, src)}
    if src in SOURCES:
        out.add(SOURCES[src].manifest_name)
    return out


def queue_polling_only(queue_log: Path | None) -> bool:
    """True when the queue has processed its whole list (it now only polls CISLR)."""
    text = _tail(queue_log, 400_000) if queue_log is not None and queue_log.exists() else None
    if text is None:
        return False
    start = text.rfind(QUEUE_START_MARK)
    return text.find(QUEUE_DONE_MARK, max(0, start)) >= 0


def _terminal(msg: str | None) -> bool:
    return bool(msg) and any(msg == t or msg.endswith(t) for t in QUEUE_TERMINAL)


def queue_progress(queue_log: Path | None, pid: int, *, script: Path | None = None, env_sources: str | None = None,
                   env_readable: bool = False, child_sources: list[str | None] | tuple = ()) -> dict:
    """What a running ``isl_queue.sh`` (``pid``) still has to process.

    ``sources`` = profile source names it may still write (None = unknown: the caller must block every profile);
    ``polling_only`` = its list is done (it only polls the gated datasets). ``error`` says why ``sources`` is None.
    """
    out: dict = {"pid": pid, "log": str(queue_log) if queue_log else None, "list": None, "list_checked_against": None,
                 "current": None, "remaining": None, "gated_pending": None, "children": list(child_sources),
                 "sources": None, "polling_only": False, "error": None}

    def fail(why: str) -> dict:
        out["error"] = why
        return out

    text = _tail(queue_log, QUEUE_LOG_TAIL)
    if text is None:
        return fail(f"queue log not readable: {queue_log}")
    lines = text.splitlines()
    starts = [(i, _LOG_START.match(ln)) for i, ln in enumerate(lines) if QUEUE_START_MARK in ln]
    starts = [(i, m) for i, m in starts if m]
    if not starts:
        return fail(f"no 'queue start (pid N): ...' line in the last {QUEUE_LOG_TAIL // 1_000_000} MB of the log")
    at, m = starts[-1]
    if int(m.group(1)) != pid:
        return fail(f"the last 'queue start' line of the log is pid {m.group(1)}, not this queue (pid {pid})")
    after = lines[at + 1:]
    if any(QUEUE_DONE_MARK in ln for ln in after):
        out.update(polling_only=True, remaining=[], gated_pending=[], sources=[])
        return out
    listed = m.group(2).split()
    out["list"] = listed
    if not listed or len(set(listed)) != len(listed):
        return fail(f"queue list empty or with repeated names: {listed}")
    if env_readable and env_sources is not None and env_sources.strip():
        expected, how = env_sources.split(), "SOURCES in the queue's environment"
    else:
        script = Path(script or default_queue_script())
        expected, how = queue_default_sources(script), f"SOURCES default of {script.name}"
    out["list_checked_against"] = how
    if expected is None:
        return fail(f"could not read the {how}")
    if expected != listed:
        return fail(f"the logged queue list differs from the {how} ({' '.join(expected)})")
    try:
        known = queue_source_names()
    except Exception as e:  # noqa: BLE001
        return fail(f"source catalog not readable: {e}")
    unknown = [s for s in listed if s not in known]
    if unknown:
        return fail(f"queue lists unknown sources: {unknown}")
    last: dict[str, str] = {}
    for ln in after:
        ev = _LOG_EVENT.match(ln)
        if ev:
            last[ev.group(1)] = ev.group(2).strip()
    started = [i for i, s in enumerate(listed) if s in last]
    if started:
        idx = max(started)
        cur = listed[idx]
        # every earlier source must have returned (the loop is sequential); one that did not still counts (fail closed)
        remaining = [s for s in listed[:idx] if not _terminal(last.get(s))]
        if not _terminal(last[cur]):
            remaining.append(cur)
            out["current"] = cur
        remaining += listed[idx + 1:]
    else:
        remaining = list(listed)  # started, no source yet
    gated = [g for g in GATED if last.get(g) != "finished"]
    out.update(remaining=remaining, gated_pending=gated)
    if any(c is None for c in child_sources):
        return fail("a child job of the queue writes an unknown source")
    names: list[str] = []
    for s in [*remaining, *(c for c in child_sources if c), *gated]:
        for w in sorted(written_sources(s)):
            if w not in names:
                names.append(w)
    out["sources"] = names
    return out


def process_source(script: str, rest: list[str]) -> str | None:
    """The data source an ingestion process writes, or None when it is unknown or several (the queue)."""
    args = _positional(rest)
    if not args or script == QUEUE_SCRIPT:
        return None
    sub = args[0]
    # only known names: anything else (an option value read as a positional, a new sub-command) is unknown and
    # blocks every profile, never "a source no profile reads"
    if script == "ingest_isl.py":
        return {"include": "include", "islrtc": "islrtc", "islrtc-recut": "islrtc"}.get(sub)
    if script == "ingest_isl_video.py":
        if sub == "islrtc-full":
            return "islrtc"
        # "run <source>" ingests, "list <source>" writes the source's item list (manifests/<source>_items.json)
        if sub not in ("run", "list") or len(args) < 2:
            return None
        try:
            from app.isl.data.video_sources.catalog import SOURCES
        except Exception:  # noqa: BLE001 - catalog not importable: unknown (fail closed)
            return None
        src = args[1].replace("-", "_")
        return src if src in SOURCES else None
    if script == "ingest_isl_pose.py":
        return POSE_SOURCES.get(sub)
    if script == GATED_SCRIPT:
        return next((a for a in args if a in GATED), None)
    return None


def ingestion_processes(proc_root: Path | None = None, queue_log: Path | None = None,
                        self_pid: int | None = None, queue_script: Path | None = None) -> list[dict]:
    """Running ingestion jobs: ``[{pid, ppid, script, subcommand, cmd, blocking, source}]``. ``blocking`` is False
    only for a queue that is just polling the gated datasets; a queue also carries ``sources`` (what it may still
    write, None = unknown) and ``queue`` (:func:`queue_progress`). Bash subshells of the queue (same command line,
    parent is the queue) are not listed twice."""
    proc_root = Path(proc_root or os.environ.get("SLR_ISL_PROC_ROOT") or "/proc")
    me = {self_pid or os.getpid(), os.getppid()}
    out = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return out
    queue_pids: set[int] = set()
    for d in entries:
        if not d.name.isdigit() or int(d.name) in me:
            continue
        argv = _argv(d)
        for i, tok in enumerate(argv):
            script = Path(tok).name
            if script not in INGEST_SCRIPTS:
                continue
            pos = _positional(argv[i + 1:])
            sub = pos[0] if pos else ""
            if script != QUEUE_SCRIPT and (sub in READ_ONLY_SUBCOMMANDS
                                           or any(a in ("-h", "--help") for a in argv[i + 1:])):
                break
            if script == GATED_SCRIPT:  # iSign / CISLR conversion; "--check" (anywhere) or no source = probe only
                rest = argv[i + 1:]
                if "--check" in rest or not any(a in GATED for a in rest):
                    break
                sub = next(a for a in rest if a in GATED)
            if script == QUEUE_SCRIPT:
                queue_pids.add(int(d.name))
            out.append({"pid": int(d.name), "ppid": _ppid(d), "script": script, "subcommand": sub,
                        "cmd": " ".join(argv)[:200], "blocking": True,
                        "source": process_source(script, argv[i + 1:]), "_dir": d})
            break
    # a "$(...)" subshell of the queue is a forked bash with the queue's command line: it is the same queue
    out = [p for p in out if not (p["script"] == QUEUE_SCRIPT and p["ppid"] in queue_pids)]
    for p in out:
        d = p.pop("_dir")
        if p["script"] != QUEUE_SCRIPT:
            continue
        readable, env = _environ_var(d, "SOURCES")
        children = [c["source"] for c in out if c["ppid"] == p["pid"] and c["script"] != QUEUE_SCRIPT]
        prog = queue_progress(queue_log, p["pid"], script=queue_script, env_sources=env, env_readable=readable,
                              child_sources=children)
        p.update(blocking=not prog["polling_only"], sources=prog["sources"], queue=prog)
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


def budget_job_end(paths: IslPaths, name: str) -> dict:
    """How the last run of a budget-limited job ended, from the last non-empty line of its log: ``{clean, detail,
    log}``. ``clean`` only for the ends in :data:`BUDGET_CLEAN_END`; a missing log or any other last line is not."""
    log_f = paths.root / "logs" / BUDGET_LOGS.get(name, f"ingest_pose_{name}.log")
    text = _tail(log_f, BUDGET_LOG_TAIL) if log_f.exists() else None
    if text is None:
        return {"clean": False, "log": str(log_f), "detail": f"no readable log {log_f.name} to confirm how it ended"}
    last = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
    clean = any(m in last for m in BUDGET_CLEAN_END)
    return {"clean": clean, "log": str(log_f), "detail": f"last log line: {last[:200]!r}"}


def source_status(paths: IslPaths, name: str, running_scripts: set[str] | None = None,
                  running_sources: set[str] | None = None) -> dict:
    """Completeness of one source's manifest. ``running_sources``: sources a running job writes (only the gated
    CISLR / iSign read it: partly filled + no job = :data:`GATED_PARTIAL`, which does not block)."""
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
        if not ids:
            st.update(expected=None, complete=False, note="not ingested")
        elif running:
            st.update(expected=None, complete=False, note="budget-limited: complete once its ingestion job has ended")
        else:
            end = budget_job_end(paths, name)
            st.update(expected=None, complete=end["clean"], job_end=end,
                      note=("budget-limited; its job ended normally: " if end["clean"] else
                            "budget-limited, but its job did not end normally (restart it: it resumes; or start the "
                            "run with --allow-partial): ") + end["detail"])
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
    if name in GATED and (ids or st["retry_pending"]) and st.get("complete") is not True:
        # access was granted at some point and ingestion stopped part-way (or is still running). A running job blocks
        # (its process does too); a stopped one does not: CISLR / iSign are research-only and gated, access may never
        # come back, so research trains on what is there - but never silently (check() lists it in "warnings")
        of = f"{st.get('done', len(ids))}/{st.get('expected') or '? (no item list)'}"
        if name in (running_sources or set()):
            st.update(complete=False, state=GATED_RUNNING,
                      note=f"{GATED_RUNNING}: {of} rows" + (f"; {st['note']}" if st.get("note") else ""))
        else:
            st.update(complete=None, state=GATED_PARTIAL, partial_gated=True,
                      note=f"{GATED_PARTIAL}: {of} rows, used as they are" +
                      (f"; {st['note']}" if st.get("note") else ""))
    return st


def profile_source_names(profile: str) -> list[str]:
    from app.isl.profiles import get_profile

    prof = get_profile(profile)
    return list(dict.fromkeys(["include", "islrtc", *prof.sources, *prof.pretraining, *prof.letters]))


def check(paths: IslPaths, profile: str, proc_root: Path | None = None, queue_log: Path | None = None,
          queue_script: Path | None = None) -> dict:
    """``{"ready": bool, "reasons": [...], "warnings": [...], "processes": [...], "sources": {name: status}}``.
    ``warnings`` never block (a gated source left partly ingested by a stopped job); ``reasons`` do."""
    procs = ingestion_processes(proc_root, queue_log if queue_log is not None else paths.root / "logs" / "queue.log",
                                queue_script=queue_script)
    names = profile_source_names(profile)
    # a job that writes one known source blocks only the profiles that read it (an OpenHands raw job, research only,
    # must not hold back the commercial-safe run); the queue blocks the profiles that read a source it still has to
    # process; unknown jobs and a queue whose progress cannot be read block every profile (fail closed)
    for p in procs:
        if not p["blocking"]:
            continue
        if p["script"] == QUEUE_SCRIPT:
            todo = p.get("sources")
            if todo is None:
                why = (p.get("queue") or {}).get("error") or "progress unknown"
                p["note"] = f"queue progress unknown ({why}): blocks every profile"
                p["reason_detail"] = f"progress unknown: {why}"
                continue
            hit = [s for s in todo if s in names]
            if not hit:
                p["blocking"] = False
                p["note"] = (f"queue still to process {', '.join(todo) or 'nothing'}; profile {profile} reads none "
                             "of them")
            else:
                p["note"] = f"queue still to process {', '.join(todo)}; profile {profile} reads {', '.join(hit)}"
                p["reason_detail"] = f"still to process for {profile}: {', '.join(hit)}"
        elif p.get("source") is not None and not (written_sources(p["source"]) & set(names)):
            p["blocking"] = False
            p["note"] = f"writes {p['source']}, which profile {profile} does not read"
    blocking = [p for p in procs if p["blocking"]]
    scripts = {p["script"] for p in procs if p.get("source") is None or p["source"] in names}
    running = {w for p in procs if p.get("source") for w in written_sources(p["source"])}
    sources = {n: source_status(paths, n, scripts, running) for n in names}
    reasons = [f"ingestion job running: pid {p['pid']} {p['script']} {p['subcommand']}".strip()
               + (f" ({p['reason_detail']})" if p.get("reason_detail") else "") for p in blocking]
    warnings = []
    for n, st in sources.items():
        if st["complete"] is False:
            reasons.append(f"manifest incomplete: {n} {st.get('done', st['rows'])}/{st.get('expected') or '?'}"
                           + (f" ({st['note']})" if st.get("note") else ""))
        elif st.get("partial_gated"):
            warnings.append(f"{n}: {st['note']} (research-only gated source; not blocking, the run uses the rows "
                            "that exist)")
    return {"ready": not reasons, "profile": profile, "checked": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reasons": reasons, "warnings": warnings, "processes": procs, "sources": sources,
            "partial_gated": {n: {k: st.get(k) for k in ("state", "rows", "done", "expected", "expected_from")}
                              for n, st in sources.items() if st.get("partial_gated")}}


def readiness_summary(work: Path | None) -> dict | None:
    """What the data guard saw when this work folder's run started (``<work>/data_readiness.json``, written by
    ``scripts/phase2_train.sh``), for the evaluation report's data section: ready, reasons (when started with
    ``--allow-partial``), non-blocking warnings and partly ingested gated sources. None without the file."""
    f = Path(work) / "data_readiness.json" if work is not None else None
    if f is None or not f.exists():
        return None
    try:
        doc = json.loads(f.read_text())
    except (OSError, ValueError):
        return {"error": f"{f.name} not readable"}
    return {"checked": doc.get("checked"), "ready": doc.get("ready"), "reasons": (doc.get("reasons") or [])[:50],
            "warnings": doc.get("warnings") or [], "partial_gated": doc.get("partial_gated") or {}}


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
