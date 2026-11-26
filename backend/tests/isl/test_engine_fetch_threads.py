"""Concurrent url fetches in the video-source engine: one manifest row per item, correct stats, zip / hfrow stay
single-threaded, the time / disk / remote stops still work with 8 threads, and no temp file is ever left behind."""

from __future__ import annotations

import errno
import importlib.util
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.isl.data.manifest import read_manifest
from app.isl.data.video_sources import engine
from app.isl.data.video_sources.catalog import get_source
from app.isl.data.video_sources.engine import choose_fetch_threads, run
from app.isl.data.video_sources.hf_listing import RemoteRefused
from app.isl.data.video_sources.listings import Item
from app.isl.paths import isl_paths

REPO_ROOT = Path(__file__).resolve().parents[3]


class FakeFetcher:
    """Writes ``body(item)`` after a short sleep and records the highest number of concurrent fetches."""

    def __init__(self, delay=0.01, fail=None, body=None):
        self.delay, self.fail = delay, fail or {}
        self.body = body or (lambda it: it.id.encode())
        self._lock = threading.Lock()
        self.active = self.max_active = self.closed = self.calls = 0

    def fetch(self, item, dest):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if item.id in self.fail:
                raise self.fail[item.id]
            data = self.body(item)
            dest.write_bytes(data)
            import hashlib

            return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        finally:
            with self._lock:
                self.active -= 1

    def close_archives(self, keep=None):
        self.closed += 1


def url_items(n):
    return [Item(f"isl_static/{i}", f"G{i % 5}", {"kind": "url", "url": f"https://m/{i}"}, ".jpg", order=i)
            for i in range(n)]


def extract_ok(mode, src, out, step_fps, max_s):
    assert open(src, "rb").read()
    return {"ok": True, "n_frames": 1, "hands_frac": 1.0}


def tmp_files(paths, spec):
    d = paths.tmp / f"video_{spec.name}"
    return list(d.iterdir()) if d.exists() else []


def rows(paths, spec):
    return read_manifest(paths.manifest(spec.manifest_name), only_ok=False)


def no_fetch_threads_alive():
    return not [t for t in threading.enumerate() if t.name.startswith("fetch-") and t.is_alive()]


def test_choose_fetch_threads():
    url = url_items(3)
    z = [Item("w/a", "A", {"kind": "zip", "archive": "u", "member": "a.mp4"}, ".mp4")]
    h = [Item("h/1", "A", {"kind": "hfrow", "dataset": "d", "split": "train", "row": 1}, ".jpg")]
    assert choose_fetch_threads(url) == 8 and choose_fetch_threads(url, 3) == 3 and choose_fetch_threads(url, 0) == 1
    for other in (z, h, url + z, url + h, []):
        assert choose_fetch_threads(other) == 1 and choose_fetch_threads(other, 8) == 1


@pytest.mark.parametrize("requested", [None, 8])
def test_concurrent_url_fetch_one_row_per_item_and_stats(tmp_path, requested):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    items = url_items(40)
    dup = {"isl_static/11": b"same-bytes", "isl_static/12": b"same-bytes"}
    f = FakeFetcher(fail={"isl_static/5": OSError("HTTP 500")}, body=lambda it: dup.get(it.id, it.id.encode()))
    st = run(spec, items, paths, f, extract=extract_ok, min_free_gb=0, fetch_threads=requested)
    assert st["fetch_threads"] == 8 and f.max_active > 1 and f.calls == 40
    assert st["todo"] == 40 and st["ok"] == 38 and st["fetch_failed"] == 1 and st["duplicates"] == 1
    assert st["failed"] == 0 and st["stopped"] is None
    got = rows(paths, spec)
    assert len(got) == 40 and Counter(r["id"] for r in got) == Counter(it.id for it in items)
    assert {r["id"] for r in got if not r["ok"]} == {"isl_static/5"}
    assert f.closed == 1 and not tmp_files(paths, spec) and no_fetch_threads_alive()
    # resume: only the failed item is fetched again
    f2 = FakeFetcher()
    st2 = run(spec, items, paths, f2, extract=extract_ok, min_free_gb=0, fetch_threads=requested)
    assert st2["todo"] == 1 and st2["ok"] == 1 and f2.calls == 1


@pytest.mark.parametrize("kind", ["zip", "hfrow", "mixed"])
def test_zip_and_hfrow_stay_single_threaded(tmp_path, kind):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("wslp")
    zips = [Item(f"wslp/{i}", "A", {"kind": "zip", "archive": "u", "member": f"{i}.mp4"}, ".mp4", order=i)
            for i in range(12)]
    hf = [Item(f"hf/{i}", "A", {"kind": "hfrow", "dataset": "d", "split": "train", "row": i}, ".jpg", order=i)
          for i in range(12)]
    items = {"zip": zips, "hfrow": hf, "mixed": url_items(6) + zips[:6]}[kind]
    f = FakeFetcher(delay=0.005)
    st = run(spec, items, paths, f, extract=extract_ok, min_free_gb=0, fetch_threads=8)
    assert st["fetch_threads"] == 1 and f.max_active == 1 and st["ok"] == 12 and f.closed == 1
    assert not tmp_files(paths, spec)


def test_time_limit_stops_with_8_threads(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    lock, n = threading.Lock(), [0]

    def clock():
        with lock:
            n[0] += 1
            return 0.0 if n[0] <= 6 else 1e6

    f = FakeFetcher()
    st = run(spec, url_items(50), paths, f, extract=extract_ok, min_free_gb=0, max_minutes=1, clock=clock,
             fetch_threads=8)
    assert st["stopped"] == "time" and 0 < st["ok"] < 50 and f.calls == st["ok"]
    assert len(rows(paths, spec)) == st["ok"] and f.closed == 1
    assert not tmp_files(paths, spec) and no_fetch_threads_alive()


def test_disk_guard_stops_with_8_threads(tmp_path, monkeypatch):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    st = run(spec, url_items(20), paths, FakeFetcher(), extract=extract_ok, min_free_gb=1e9, fetch_threads=8)
    assert st["stopped"] == "disk" and st["ok"] == 0 and not rows(paths, spec)
    lock, n = threading.Lock(), [0]

    def free(path):
        with lock:
            n[0] += 1
            return 100.0 if n[0] <= 5 else 0.0

    monkeypatch.setattr(engine, "free_gb", free)
    f = FakeFetcher()
    st = run(spec, url_items(20), paths, f, extract=extract_ok, min_free_gb=10, fetch_threads=8)
    assert st["stopped"] == "disk" and st["ok"] == 5 and f.calls == 5
    assert not tmp_files(paths, spec) and no_fetch_threads_alive()


def test_remote_refusal_stops_with_8_threads(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    f = FakeFetcher(fail={"isl_static/7": RemoteRefused("HTTP 429 x4", permanent=False)})
    st = run(spec, url_items(200), paths, f, extract=extract_ok, min_free_gb=0, fetch_threads=8)
    assert st["stopped"] == "remote" and not st["remote_permanent"] and "429" in st["remote_error"]
    assert st["fetch_failed"] == 1 and 0 < st["ok"] < 199 and f.calls < 200
    got = rows(paths, spec)
    assert len(got) == st["ok"] + 1 and [r["id"] for r in got if not r["ok"]] == ["isl_static/7"]
    assert f.closed == 1 and not tmp_files(paths, spec) and no_fetch_threads_alive()


@pytest.mark.parametrize("workers", [1, 2])
def test_extract_exception_leaves_no_temp_files(tmp_path, workers, monkeypatch):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    if workers > 1:  # a thread pool stands in for the process pool (same Future interface, no spawn cost)
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(engine, "ProcessPoolExecutor", lambda n, mp_context=None: ThreadPoolExecutor(n))
    calls = [0]

    def extract(mode, src, out, step_fps, max_s):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("extractor crashed")
        return extract_ok(mode, src, out, step_fps, max_s)

    f = FakeFetcher()
    with pytest.raises(RuntimeError, match="extractor crashed"):
        run(spec, url_items(60), paths, f, workers=workers, extract=extract, min_free_gb=0, fetch_threads=8)
    assert not tmp_files(paths, spec) and no_fetch_threads_alive() and f.closed == 1
    assert f.calls < 60  # stopped fetching after the failure


def _load_script():
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sp = importlib.util.spec_from_file_location("ingest_isl_video_fetch_threads_test",
                                                REPO_ROOT / "scripts" / "ingest_isl_video.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------- circuit breaker

class FailSome(FakeFetcher):
    """Every fetch of an item whose number matches ``bad`` fails with ``exc``."""

    def __init__(self, bad, exc=None, **kw):
        super().__init__(**kw)
        self.bad, self.exc = bad, exc or PermissionError("HTTP 403 for https://m/x")

    def fetch(self, item, dest):
        if self.bad(int(item.id.rsplit("/", 1)[1])):
            with self._lock:
                self.calls += 1
            time.sleep(self.delay)
            raise self.exc
        return super().fetch(item, dest)


@pytest.mark.parametrize("threads", [1, 8])
def test_failure_storm_trips_the_breaker(tmp_path, threads):
    """A 403 / 429 storm stops the run after FETCH_FAIL_STREAK new failures in a row instead of writing a failed row
    for every remaining item and exiting 0 ("finished")."""
    m = _load_script()
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    items = url_items(300)
    limit = engine.FETCH_FAIL_STREAK
    f = FailSome(lambda i: i >= 20)
    st = run(spec, items, paths, f, extract=extract_ok, min_free_gb=0, fetch_threads=threads)
    assert st["stopped"] == "remote" and st["remote_permanent"] is False and "in a row" in st["remote_error"]
    # with 8 threads some failures can land before the last success (which resets the count) and a few after the
    # trip, so only "well before the end" is exact
    assert st["ok"] == 20 and st["fetch_failed"] >= limit and f.calls < 150
    assert len(rows(paths, spec)) == st["ok"] + st["fetch_failed"]
    assert not tmp_files(paths, spec) and no_fetch_threads_alive() and f.closed == 1
    assert m.remote_rc(st) == m.EXIT_MORE  # progressed: the queue starts the next chunk
    # next chunk, server still refusing: the ids that already failed are retried but do not count
    st2 = run(spec, items, paths, FailSome(lambda i: i >= 20), extract=extract_ok, min_free_gb=0,
              fetch_threads=threads)
    assert st2["stopped"] == "remote" and st2["ok"] == 0
    assert st["fetch_failed"] + limit <= st2["fetch_failed"] <= st["fetch_failed"] + limit + threads
    assert m.remote_rc(st2) == m.EXIT_ERR  # no progress: the queue counts a failed try and backs off
    # server back: everything left is done
    st3 = run(spec, items, paths, FakeFetcher(delay=0.001), extract=extract_ok, min_free_gb=0, fetch_threads=threads)
    assert st3["stopped"] is None and st3["ok"] == 280


@pytest.mark.parametrize("threads", [1, 8])
def test_breaker_ignores_retried_items_and_resets_on_success(tmp_path, threads):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    items = url_items(150)
    gone = lambda i: i < 60  # noqa: E731 - a block of permanently missing files (more than the limit)
    st = run(spec, items[:60], paths, FailSome(gone), extract=extract_ok, min_free_gb=0, fetch_threads=threads,
             fail_streak=0)  # 0 = breaker off
    assert st["fetch_failed"] == 60 and st["stopped"] is None
    # the 60 known failures still fail but do not count; new failures with a success every 10th item: no stop
    f = FailSome(lambda i: gone(i) or i % 10 != 0, delay=0.002)
    st = run(spec, items, paths, f, extract=extract_ok, min_free_gb=0, fetch_threads=threads)
    assert st["stopped"] is None and f.calls == 150
    assert st["ok"] == 9 and st["fetch_failed"] == 141  # ok: 60, 70, ..., 140


# --------------------------------------------------------------------------------------------- unexpected errors

@pytest.mark.parametrize("threads", [1, 8])
def test_unexpected_fetch_error_stops_with_error(tmp_path, threads, monkeypatch):
    """mkstemp failing (EMFILE / ENOSPC) must not end the run as finished (stopped None, exit 0)."""
    m = _load_script()
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    real, lock, n = engine.temp_path, threading.Lock(), [0]

    def temp_path(*a, **kw):
        with lock:
            n[0] += 1
            if n[0] >= 10:
                raise OSError(errno.EMFILE, "Too many open files")
        return real(*a, **kw)

    monkeypatch.setattr(engine, "temp_path", temp_path)
    f = FakeFetcher()
    st = run(spec, url_items(200), paths, f, extract=extract_ok, min_free_gb=0, fetch_threads=threads)
    assert st["stopped"] == "error" and "Too many open files" in st["fetch_error"]
    assert st["ok"] == f.calls == 9 and st["fetch_failed"] == 0 and len(rows(paths, spec)) == 9
    assert not tmp_files(paths, spec) and no_fetch_threads_alive() and f.closed == 1
    monkeypatch.setattr(engine, "temp_path", real)
    st2 = run(spec, url_items(200), paths, FakeFetcher(delay=0.001), extract=extract_ok, min_free_gb=0,
              fetch_threads=threads)
    assert st2["stopped"] is None and st2["ok"] == 191
    # and the script turns it into exit 2 (the queue retries) - see test_run_locked_* below for the mapping
    assert m.EXIT_ERR == 2


# --------------------------------------------------------------------------------------------- CLI plumbing

@pytest.fixture()
def cli(tmp_path, monkeypatch):
    """The real ``ingest_isl_video`` script with listing, hand model, lock and ``engine.run`` stubbed out."""
    m = _load_script()
    spec = get_source("mendeley_static")
    seen = {"stats": None}
    url = url_items(4)
    zips = [Item(f"wslp/{i}", "A", {"kind": "zip", "archive": "u", "member": f"{i}.mp4"}, ".mp4") for i in range(4)]
    state = {"items": url}

    def fake_run(spec_, items, paths, fetcher, **kw):
        seen.update(kw)
        return dict({"todo": len(items), "ok": 0, "failed": 0, "duplicates": 0, "stopped": None},
                    **(seen["stats"] or {}))

    monkeypatch.setattr(m, "build_items", lambda name, paths, per_class=None: (spec, state["items"], object()))
    monkeypatch.setattr(m, "ensure_hand_model", lambda: None)
    monkeypatch.setattr(m, "source_lock", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(m, "_setup_logging", lambda *a, **kw: None)
    monkeypatch.setattr(m.os, "nice", lambda n: 0)
    monkeypatch.setattr(engine, "run", fake_run)

    def call(*extra, items="url", stats=None):
        state["items"], seen["stats"] = (url if items == "url" else zips), stats
        seen.pop("fetch_threads", None)
        rc = m.main(["--root", str(tmp_path / "isl"), "run", "mendeley_static", "--workers", "1", *extra])
        return rc, seen.get("fetch_threads")

    return m, call


def test_cli_passes_fetch_threads_to_the_engine(cli):
    m, call = cli
    assert call() == (0, 8)  # flag left out = auto
    assert call("--fetch-threads", "auto") == (0, 8)
    assert call("--fetch-threads", "3") == (0, 3)
    assert call("--fetch-threads", "0") == (0, 1)
    assert call("--fetch-threads", "8", items="zip") == (0, 1)
    assert call(items="zip") == (0, 1)


def test_run_locked_without_fetch_threads_attribute(cli, tmp_path):
    """Callers that build their own Namespace (the gated cislr delegation) get auto."""
    m, _ = cli
    seen = {}
    eng = SimpleNamespace(choose_fetch_threads=engine.choose_fetch_threads, dump_items=lambda *a: None,
                          run=lambda *a, **kw: seen.update(kw) or {"todo": 4, "stopped": None})
    args = SimpleNamespace(workers="1", source="mendeley_static", per_class=None, limit=0, max_minutes=0.0,
                           min_free_gb=10.0)
    assert m._run_locked(args, isl_paths(tmp_path / "isl"), get_source("mendeley_static"), eng) == 0
    assert seen["fetch_threads"] == 8


def test_run_exit_codes_for_engine_stops(cli):
    m, call = cli
    assert call(stats={"stopped": "error", "fetch_error": "OSError: EMFILE"})[0] == m.EXIT_ERR == 2
    assert call(stats={"stopped": "remote", "remote_permanent": False, "ok": 5})[0] == m.EXIT_MORE
    assert call(stats={"stopped": "remote", "remote_permanent": False, "ok": 0})[0] == m.EXIT_ERR
    assert call(stats={"stopped": "time"})[0] == m.EXIT_MORE
    assert call(stats={"stopped": None})[0] == 0
