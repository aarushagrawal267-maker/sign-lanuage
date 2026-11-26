"""Fetch-failure retry passes: ``ingest_isl_video.py run`` exits 11 when a run finished with items that failed to
fetch, and ``scripts/isl_queue.sh`` re-runs such a source up to FETCH_RETRY_PASSES times before moving on.

The queue test runs a copy of the real script in a temp "repository" (ROOT is the script's parent folder, so the
real data/isl/logs, lock and manifests are never touched) with PY pointing at a stub that plays the exit codes."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
QUEUE = REPO_ROOT / "scripts" / "isl_queue.sh"


def _load_script():
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sp = importlib.util.spec_from_file_location("ingest_isl_video_fetch_retry_test",
                                                REPO_ROOT / "scripts" / "ingest_isl_video.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------- CLI exit code

@pytest.fixture()
def run_locked(tmp_path, monkeypatch):
    from app.isl.data.video_sources import engine
    from app.isl.data.video_sources.catalog import get_source
    from app.isl.paths import isl_paths

    m = _load_script()
    spec = get_source("mendeley_static")
    monkeypatch.setattr(m, "build_items", lambda name, paths, per_class=None: (spec, [], object()))
    monkeypatch.setattr(m, "ensure_hand_model", lambda: None)

    def call(stats, limit=0):
        base = {"todo": 4, "ok": 0, "failed": 0, "fetch_failed": 0, "duplicates": 0, "stopped": None}
        eng = SimpleNamespace(choose_fetch_threads=engine.choose_fetch_threads, dump_items=lambda *a: None,
                              run=lambda *a, **kw: dict(base, **stats))
        args = SimpleNamespace(workers="1", source="mendeley_static", per_class=None, limit=limit,
                               max_minutes=0.0, min_free_gb=10.0)
        return m._run_locked(args, isl_paths(tmp_path / "isl"), spec, eng)

    return m, call


def test_exit_11_is_new_and_distinct():
    m = _load_script()
    assert m.EXIT_FETCH_RETRY == 11
    assert m.EXIT_FETCH_RETRY not in {0, m.EXIT_MORE, m.EXIT_DISK, m.EXIT_GATED, m.EXIT_ERR, m.EXIT_MEMORY,
                                      m.EXIT_BUSY}


def test_finished_run_with_fetch_failures_exits_11(run_locked, caplog):
    m, call = run_locked
    caplog.set_level("WARNING", logger="ingest_isl_video")
    assert call({"ok": 24573, "fetch_failed": 140}) == m.EXIT_FETCH_RETRY == 11
    assert "fetch_failed=140" in caplog.text  # the queue reads N from this line
    assert call({"ok": 24573, "fetch_failed": 0}) == 0
    assert call({"ok": 3}) == 0  # older stats without the key
    assert call({"todo": 0}) == 0  # nothing left to do


def test_stops_keep_their_codes_with_fetch_failures(run_locked):
    """Only a run with no stop reason becomes 11; remote (rows API / circuit breaker), error, time, memory and disk
    stops keep their codes even when items failed to fetch."""
    m, call = run_locked
    ff = {"fetch_failed": 40}
    assert call(dict(ff, stopped="remote", remote_permanent=False, ok=5)) == m.EXIT_MORE
    assert call(dict(ff, stopped="remote", remote_permanent=False, ok=0)) == m.EXIT_ERR
    assert call(dict(ff, stopped="remote", remote_permanent=True, ok=5)) == m.EXIT_ERR
    assert call(dict(ff, stopped="error")) == m.EXIT_ERR
    assert call(dict(ff, stopped="time")) == m.EXIT_MORE
    assert call(dict(ff, stopped="memory")) == m.EXIT_MEMORY
    assert call(dict(ff, stopped="disk")) == m.EXIT_DISK
    assert call(dict(ff, todo=5), limit=5) == m.EXIT_MORE  # --limit reached: more to do, not 11


# --------------------------------------------------------------------------------------------- queue retry passes

STUB = """#!/usr/bin/env bash
# fake python: plays the exit codes listed in $STUB_DIR/codes_<source> (one per run, last one repeats)
# (--check SRC answers only when $STUB_DIR/grant_SRC exists)
for a in "$@"; do case "$a" in --check) [ -f "$STUB_DIR/grant_$3" ] && exit 0; exit 3 ;; status) echo '{}'; exit 0 ;; esac; done
src="$3"
f="$STUB_DIR/codes_$src"; n="$STUB_DIR/runs_$src"
k=$(( $(cat "$n" 2>/dev/null || echo 0) + 1 )); echo "$k" > "$n"
rc=$(sed -n "${k}p" "$f"); [ -n "$rc" ] || rc=$(tail -n 1 "$f")
[ "$rc" = 11 ] && echo "WARNING $src: finished this pass with fetch_failed=$((150 - 10 * k)) (exit 11)" >&2
exit "$rc"
"""


def _run_queue(tmp_path, codes: dict[str, list[int]], sources: str | None = None, grant: tuple = (),
               **env_extra: str) -> tuple[str, dict[str, int]]:
    if not shutil.which("flock"):
        pytest.skip("flock not installed")
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "data" / "isl").mkdir(parents=True)
    shutil.copy(QUEUE, root / "scripts" / "isl_queue.sh")
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    py = stub_dir / "python"
    py.write_text(STUB)
    py.chmod(0o755)
    for src, cs in codes.items():
        (stub_dir / f"codes_{src}").write_text("\n".join(map(str, cs)) + "\n")
    for g in grant:
        (stub_dir / f"grant_{g}").write_text("")
    env = dict(os.environ, PY=str(py), STUB_DIR=str(stub_dir), SOURCES=sources or " ".join(codes), MIN_FREE_GB="-1",
               CISLR_POLL_DAYS="0", FETCH_RETRY_WAIT="0", **env_extra)
    env.pop("FETCH_RETRY_PASSES", None)
    p = subprocess.run(["bash", str(root / "scripts" / "isl_queue.sh")], env=env, cwd=tmp_path, timeout=60,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    log = (root / "data" / "isl" / "logs" / "queue.log").read_text()
    runs = {src: int((stub_dir / f"runs_{src}").read_text()) if (stub_dir / f"runs_{src}").exists() else 0
            for src in codes}
    return log, runs


def _msgs(log: str) -> list[str]:
    return [ln.split(" ", 2)[2] for ln in log.splitlines() if len(ln.split(" ", 2)) == 3]


def test_queue_retries_fetch_failures_until_finished(tmp_path):
    log, runs = _run_queue(tmp_path, {"srca": [11, 11, 0]})
    assert runs == {"srca": 3}
    msgs = [m for m in _msgs(log) if m.startswith("srca:") and "start chunk" not in m]
    assert msgs == ["srca: 140 fetch failures; retry pass 1 of 3 in 0 s",
                    "srca: 130 fetch failures; retry pass 2 of 3 in 0 s",
                    "srca: finished"]
    assert "all listed sources processed; status:" in log and "giving up" not in log


def test_queue_moves_on_after_three_retry_passes(tmp_path):
    log, runs = _run_queue(tmp_path, {"srca": [10, 11], "srcb": [0]})
    assert runs == {"srca": 5, "srcb": 1}  # one chunk, the pass that failed, then 3 retry passes; srcb still runs
    msgs = [m for m in _msgs(log) if m.startswith("src") and "start chunk" not in m]
    assert msgs == ["srca: chunk done, more to do",
                    "srca: 130 fetch failures; retry pass 1 of 3 in 0 s",
                    "srca: 120 fetch failures; retry pass 2 of 3 in 0 s",
                    "srca: 110 fetch failures; retry pass 3 of 3 in 0 s",
                    "srca: fetch failures remain after 3 retry passes; moving on",
                    "srcb: finished"]
    assert "all listed sources processed; status:" in log and "giving up" not in log


def test_cislr_that_moved_on_is_not_done(tmp_path):
    """A CISLR run left with fetch failures after its retry passes is not marked done (run_source returns 4, not
    0), so the next gated probe runs it again; before, "moving on" returned 0 and CISLR was never retried."""
    log, runs = _run_queue(tmp_path, {"cislr": [11], "srca": [0], "srcb": [0]}, sources="srca srcb",
                           grant=("cislr",), GATED_PROBE_MIN="0")
    assert runs == {"cislr": 8, "srca": 1, "srcb": 1}  # probed before each list source: 1 + 3 retry passes, twice
    moved = [m for m in _msgs(log) if m == "cislr: fetch failures remain after 3 retry passes; moving on"]
    assert len(moved) == 2 and "srcb: finished" in log
    assert "all listed sources processed; status:" in log and "giving up" not in log
    # the same with CISLR finishing on its second probe: done after that, not run again
    log, runs = _run_queue(tmp_path / "b", {"cislr": [11, 11, 11, 11, 0], "srca": [0], "srcb": [0], "srcc": [0]},
                           sources="srca srcb srcc", grant=("cislr",), GATED_PROBE_MIN="0")
    assert runs == {"cislr": 5, "srca": 1, "srcb": 1, "srcc": 1} and "cislr: finished" in log


def test_queue_script_parses_and_documents_exit_11():
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    q = QUEUE.read_text()
    assert "11) " in q and 'FETCH_RETRY_PASSES="${FETCH_RETRY_PASSES:-3}"' in q
    assert 'FETCH_RETRY_WAIT="${FETCH_RETRY_WAIT:-120}"' in q
    assert 'log "all listed sources processed; status:"' in q  # the watchdog greps for this
