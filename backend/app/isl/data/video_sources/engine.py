"""Generic resumable ingestion: fetch one item at a time into a temp file, extract landmarks in a process pool,
delete the temp file, append a manifest row.

* Resume: ids with an ``ok: true`` row in ``data/isl/manifests/<manifest>.jsonl`` are skipped (failures are retried).
* One fetcher thread prefetches at most ``2 x workers`` items, so temp disk use is a few clips/images.
* Zip members are read through one shared seekable archive object (``RangeFile``) in archive order, so range
  reads are sequential and the 8 MB block cache works.
* Disk guard: before every fetch the free space under the data root must be >= ``min_free_gb``; otherwise the run
  stops cleanly (exit status ``stopped: "disk"``) and the queue script waits.
* ``max_minutes``: stop taking new items after that long (pending items finish), so a queue can re-plan worker
  counts between chunks.
* ``max_rss_gb``: soft memory cap per process (this one and every extraction worker). Above it the caches are
  dropped (``gc``, ``malloc_trim``, the HF row-URL cache); if a process stays above, the run stops taking items
  (``stopped: "memory"``, fetched but unstarted items are deleted and retried next run) and
  ``scripts/isl_queue.sh`` restarts the source in fresh processes. All ingestion jobs share one ~14 GB memory group.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import shutil
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

from app.isl.data.manifest import append_jsonl, read_manifest
from app.isl.data.video_sources.catalog import SourceSpec
from app.isl.data.video_sources.fetch import temp_path, url_to_file, zip_member_to_file
from app.isl.data.video_sources.listings import Item, safe_name
from app.isl.paths import IslPaths

log = logging.getLogger(__name__)

HF_ROWS = "https://datasets-server.huggingface.co/rows"


class Fetcher:
    """Fetch an :class:`Item` to a local temp file. ``archives`` maps an archive key to an open ``ZipFile``
    (or a factory returning one); ``get`` replaces ``requests.get`` in tests."""

    def __init__(self, archives: dict[str, zipfile.ZipFile | Callable[[], zipfile.ZipFile]] | None = None, get=None,
                 sleep=time.sleep):
        self._archives = dict(archives or {})
        self._open: dict[str, zipfile.ZipFile] = {}
        self.get = get
        self.sleep = sleep
        self._rows: dict[tuple[str, str, int], str] = {}  # (dataset, split, row) -> image src (signed, ~1 h)

    def _zip(self, key: str) -> zipfile.ZipFile:
        if key not in self._open:
            z = self._archives[key]
            self._open[key] = z() if callable(z) else z
        return self._open[key]

    def close_archives(self, keep: str | None = None) -> None:
        for k in [k for k in self._open if k != keep]:
            try:
                self._open.pop(k).close()
            except Exception:  # noqa: BLE001
                pass

    def _get(self):
        if self.get is None:
            from app.isl.data.remote import session

            self.get = session().get
        return self.get

    def _hf_row_src(self, dataset: str, split: str, row: int) -> str:
        key = (dataset, split, row)
        if key not in self._rows:
            off = row - row % 100
            for i in range(6):  # the rows API rate-limits (429) and has 503 spells
                r = self._get()(HF_ROWS, params={"dataset": dataset, "config": "default", "split": split,
                                                 "offset": off, "length": 100}, timeout=120)
                if r.status_code == 200:
                    break
                if r.status_code != 429 and r.status_code < 500:
                    raise OSError(f"rows API HTTP {r.status_code}")
                self.sleep(min(120, 5 * 2 ** i))
            else:
                raise OSError(f"rows API HTTP {r.status_code} after retries")
            for rr in r.json().get("rows", []):
                img = (rr.get("row") or {}).get("image") or {}
                if img.get("src"):
                    self._rows[(dataset, split, int(rr["row_idx"]))] = img["src"]
        return self._rows.pop(key)

    def fetch(self, item: Item, dest: Path) -> dict:
        f = item.fetch
        kind = f["kind"]
        if kind == "zip":
            self.close_archives(keep=f["archive"])  # one archive open at a time
            sha, n, crc = zip_member_to_file(self._zip(f["archive"]), f["member"], dest)
            return {"sha256": sha, "bytes": n, "crc32": crc}
        if kind == "url":
            sha, n = url_to_file(f["url"], dest, get=self._get(), sleep=self.sleep)
            exp = f.get("sha256")
            if exp and exp != sha:
                raise OSError(f"sha256 mismatch (expected {exp[:12]}..., got {sha[:12]}...)")
            return {"sha256": sha, "bytes": n}
        if kind == "hfrow":
            try:
                src = self._hf_row_src(f["dataset"], f["split"], int(f["row"]))
                sha, n = url_to_file(src, dest, get=self._get(), sleep=self.sleep)
            except PermissionError:  # signed asset URL expired: refresh the page once
                self._rows.pop((f["dataset"], f["split"], int(f["row"])), None)
                src = self._hf_row_src(f["dataset"], f["split"], int(f["row"]))
                sha, n = url_to_file(src, dest, get=self._get(), sleep=self.sleep)
            return {"sha256": sha, "bytes": n}
        raise ValueError(f"unknown fetch kind {kind}")


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def out_path(spec: SourceSpec, paths: IslPaths, item: Item) -> Path:
    name = item.out_name or (safe_name(item.id.split("/", 1)[1] if "/" in item.id else item.id) + ".npz")
    return paths.landmarks / spec.manifest_name / name


def rel_path(p: Path, paths: IslPaths) -> str:
    """Manifest paths are written as ``data/isl/...`` (what ``IslPaths.resolve`` expects)."""
    try:
        return "data/isl/" + p.relative_to(paths.root).as_posix()
    except ValueError:
        return str(p)


def run(spec: SourceSpec, items: list[Item], paths: IslPaths, fetcher: Fetcher, *, workers: int = 1,
        limit: int = 0, max_minutes: float = 0.0, min_free_gb: float = 10.0, extract: Callable | None = None,
        progress: Callable[[str], None] | None = None, clock=time.time, max_rss_gb: float = 0.0,
        rss: Callable | None = None) -> dict:
    """Process every not-yet-done item. ``extract(mode, src, out, step_fps, max_s) -> dict`` defaults to
    ``extract.work``; with ``workers <= 1`` it runs in this process (tests)."""
    if extract is None:
        from app.isl.data.video_sources.extract import work as extract
    man = paths.manifest(spec.manifest_name)
    prior = read_manifest(man)
    done = {r["id"] for r in prior}
    # byte-identical files (same SHA-256) are extracted once; later copies get a row with ``duplicate_of``
    by_sha = {r["sha256"]: r for r in prior if r.get("sha256") and not r.get("duplicate_of")}
    waiting: dict[str, list[dict]] = {}  # sha -> duplicate rows waiting for the in-flight original
    todo = [it for it in items if it.id not in done]
    todo.sort(key=lambda it: (it.fetch.get("archive", ""), it.order, it.id))
    if limit:
        todo = todo[:limit]
    stats = {"items": len(items), "already_done": len(items) - len([i for i in items if i.id not in done]),
             "todo": len(todo), "ok": 0, "failed": 0, "fetch_failed": 0, "duplicates": 0, "stopped": None,
             "secs": 0.0}
    if not todo:
        return stats
    tmp_dir = paths.tmp / f"video_{spec.name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for stale in tmp_dir.glob("*"):  # leftovers of a killed run (this source only)
        if stale.is_file():
            stale.unlink(missing_ok=True)
    t0 = clock()
    q: queue.Queue = queue.Queue(maxsize=max(2, 2 * workers))
    stop = threading.Event()

    def base_meta(it: Item) -> dict:
        m = {"id": it.id, "source": spec.name, "gloss": it.gloss, "licence": spec.licence, "profile": spec.profile}
        m.update(it.meta)
        return m

    def producer():
        try:
            for it in todo:
                if stop.is_set():
                    break
                if max_minutes and clock() - t0 > max_minutes * 60:
                    stats["stopped"] = "time"
                    break
                if free_gb(paths.root) < min_free_gb:
                    stats["stopped"] = "disk"
                    log.warning("free disk below %.0f GB: stopping %s", min_free_gb, spec.name)
                    break
                dest = temp_path(tmp_dir, "v_", it.suffix or ".bin")
                try:
                    finfo = fetcher.fetch(it, dest)
                except Exception as e:  # noqa: BLE001
                    dest.unlink(missing_ok=True)
                    meta = base_meta(it)
                    meta.update({"ok": False, "error": f"fetch: {type(e).__name__}: {e}"[:300]})
                    q.put(("failed", it, meta, None))
                    continue
                q.put(("ok", it, finfo, dest))
        finally:
            fetcher.close_archives()
            q.put(None)

    th = threading.Thread(target=producer, daemon=True, name=f"fetch-{spec.name}")
    th.start()
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) if workers > 1 else None
    pending: list[tuple[dict, Path, Future | dict]] = []
    from app.isl.data.pose_sources.common import malloc_trim, rss_gb

    rss = rss or rss_gb

    def worst_rss() -> tuple[float, int | None]:
        pids = [None] + (list(getattr(pool, "_processes", None) or {}) if pool is not None else [])
        vals = [(float(rss(pid)), pid) for pid in pids]
        return max(vals, key=lambda v: v[0])

    def memory_exceeded() -> bool:
        if not max_rss_gb or max_rss_gb <= 0:
            return False
        gb, _ = worst_rss()
        if gb <= max_rss_gb:
            return False
        import gc

        fetcher._rows.clear()
        gc.collect()
        malloc_trim()
        gb2, pid = worst_rss()
        log.warning("%s: memory guard: RSS %.2f GB > cap %.2f GB; after dropping caches %.2f GB (%s)", spec.name, gb,
                    max_rss_gb, gb2, "this process" if pid is None else f"worker {pid}")
        return gb2 > max_rss_gb

    def write_duplicate(dup: dict, orig: dict) -> None:
        for k, v in orig.items():
            if k not in dup and k not in ("id",):
                dup[k] = v
        dup["path"], dup["ok"], dup["duplicate_of"] = orig["path"], True, orig["id"]
        append_jsonl(man, dup)
        stats["duplicates"] += 1

    def finish(entry):
        meta, dest, res = entry
        info = res.result() if isinstance(res, Future) else res
        dest.unlink(missing_ok=True)
        meta.update(info)
        append_jsonl(man, meta)
        stats["ok" if info.get("ok") else "failed"] += 1
        sha = meta.get("sha256")
        for dup in waiting.pop(sha, []) if sha else []:
            if info.get("ok"):
                write_duplicate(dup, meta)
            else:  # the original failed: the copy is retried on the next run
                dup.update({"ok": False, "error": "original failed: " + str(info.get("error"))[:200]})
                append_jsonl(man, dup)
        if info.get("ok") and sha:
            by_sha.setdefault(sha, meta)
        n = stats["ok"] + stats["failed"]
        if progress and (n % 50 == 0 or not info.get("ok")):
            el = clock() - t0
            progress(f"{spec.name}: {n}/{len(todo)} this run, {stats['ok']} ok, {stats['failed']} failed, "
                     f"{el / max(1, n):.2f} s/item")

    try:
        while True:
            entry = q.get()
            if entry is None:
                break
            status, it, payload, dest = entry
            if stats["stopped"] == "memory":  # fetched after the memory stop: retried on the next run
                if dest is not None:
                    dest.unlink(missing_ok=True)
                continue
            if status == "failed":
                append_jsonl(man, payload)
                stats["fetch_failed"] += 1
                continue
            out = out_path(spec, paths, it)
            meta = base_meta(it)
            meta.update(payload)
            sha = payload.get("sha256")
            if sha and (sha in by_sha or sha in waiting):
                dest.unlink(missing_ok=True)
                if sha in by_sha:
                    write_duplicate(meta, by_sha[sha])
                else:
                    waiting[sha].append(meta)
                continue
            if sha:
                waiting[sha] = []
            meta["path"] = rel_path(out, paths)
            args = (spec.mode, str(dest), str(out), spec.step_fps, spec.max_s)
            if pool is None:
                finish((meta, dest, extract(*args)))
            else:
                pending.append((meta, dest, pool.submit(extract, *args)))
                while pending and (len(pending) >= 2 * workers or pending[0][2].done()):
                    finish(pending.pop(0))
            if stats["stopped"] != "memory" and memory_exceeded():
                stats["stopped"] = "memory"
                stop.set()
                log.error("%s: stopping for memory (resumable; the queue restarts this source)", spec.name)
        for entry in pending:
            finish(entry)
        pending.clear()
    except BaseException:
        stop.set()
        raise
    finally:
        stop.set()
        for meta, dest, _ in pending:
            dest.unlink(missing_ok=True)
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        # drain the queue so the producer thread can exit, deleting fetched temp files
        while th.is_alive() or not q.empty():
            try:
                e = q.get(timeout=1)
            except queue.Empty:
                continue
            if e and e[0] == "ok" and e[3] is not None:
                e[3].unlink(missing_ok=True)
    stats["secs"] = round(clock() - t0, 1)
    return stats


def summarise(manifest: Path) -> dict:
    """Counts for ``status``: rows, ok, failed, labelled, distinct glosses, quality pass rate."""
    from app.isl.data.manifest import iter_jsonl

    last = {}
    for r in iter_jsonl(manifest):
        if r.get("id"):
            last[r["id"]] = r
    rows = list(last.values())
    ok = [r for r in rows if r.get("ok")]
    good = [r for r in ok if int(r.get("n_frames") or 0) >= 1 and float(r.get("hands_frac") or 0) >= 0.2]
    return {"ids": len(rows), "ok": len(ok), "failed": len(rows) - len(ok),
            "duplicates": sum(1 for r in ok if r.get("duplicate_of")),
            "labelled": sum(1 for r in ok if r.get("gloss")),
            "glosses": len({str(r.get("gloss")).lower() for r in ok if r.get("gloss")}),
            "hands_ok": len(good),
            "mean_secs": round(sum(float(r.get("secs") or 0) for r in ok) / max(1, len(ok)), 2)}


def dump_items(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps([it.__dict__ for it in items], ensure_ascii=False))
    tmp.replace(path)


def load_items(path: Path) -> list[Item] | None:
    if not path.exists():
        return None
    try:
        return [Item(**d) for d in json.loads(path.read_text())]
    except (ValueError, TypeError):
        return None
