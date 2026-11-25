"""CISLR ingestion (Exploration-Lab/CISLR, gated on Hugging Face; docs/ISL_DESIGN.md section 5.7).

The 1.17 GB video zip is never downloaded as a whole: it is opened through :class:`RangeFile` (the zip
central directory is read from the end with range requests), each member video is streamed to a temp file,
converted to a ``RawSequence`` with the app's own extractor (``scripts/ingest_isl.py`` ``extract_full``,
15 fps) and deleted. The job is resumable per member through ``data/isl/manifests/cislr.jsonl``. Peak disk
use: one video per worker.

Licence: AFL-3.0 on the dataset card, but the videos come from public online sources whose rights the
licence may not cover, so CISLR is used in the ``research`` profile only until reviewed.
"""

from __future__ import annotations

import csv
import logging
import re
import shutil
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from app.isl.data.manifest import append_jsonl, read_manifest
from app.isl.paths import IslPaths

log = logging.getLogger(__name__)

CISLR_REPO = "Exploration-Lab/CISLR"
CSV_FILES = ("dataset.csv", "prototype.csv", "test.csv")
ZIP_PATH = "CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

_UID_KEYS = ("uid", "video_id", "videoid", "video", "id", "filename", "file", "file_name", "name", "path")
_WORD_KEYS = ("word", "gloss", "label", "text", "sign", "class", "english")
_CAT_KEYS = ("category", "categories", "topic", "domain", "pos")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150]


def normalise_cislr_row(row: dict) -> dict | None:
    """Map a CSV row with unknown column names onto ``{"uid", "word", "category"}`` (column names are matched
    case-insensitively; the exact format is confirmed once access is granted)."""
    low = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None}
    uid = next((low[k] for k in _UID_KEYS if low.get(k)), None)
    word = next((low[k] for k in _WORD_KEYS if low.get(k)), None)
    if not uid or not word:
        return None
    uid = PurePosixPath(str(uid)).name
    if PurePosixPath(uid).suffix.lower() in VIDEO_EXT:
        uid = PurePosixPath(uid).stem
    cat = next((low[k] for k in _CAT_KEYS if low.get(k)), None)
    return {"uid": uid, "word": str(word), "category": cat}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [r for r in (normalise_cislr_row(x) for x in csv.DictReader(fh)) if r]


def csv_index(paths: IslPaths) -> tuple[dict[str, dict], dict[str, str]]:
    """uid -> row (dataset.csv), uid -> "prototype" | "test" (prototype.csv / test.csv)."""
    base = paths.raw / "cislr"
    rows = {r["uid"]: r for r in read_csv_rows(base / "dataset.csv")}
    split: dict[str, str] = {}
    for name in ("prototype", "test"):
        for r in read_csv_rows(base / f"{name}.csv"):
            split[r["uid"]] = name
            rows.setdefault(r["uid"], r)
    return rows, split


def download_small_files(paths: IslPaths, get=None) -> list[Path]:
    """The three CSVs (a few hundred KB) into ``data/isl/raw/cislr/``."""
    from app.isl.data.remote import hf_url, session

    get = get or session().get
    out = []
    base = paths.raw / "cislr"
    base.mkdir(parents=True, exist_ok=True)
    for f in CSV_FILES:
        dest = base / f
        if not dest.exists() or dest.stat().st_size == 0:
            r = get(hf_url(CISLR_REPO, f), timeout=120)
            if r.status_code != 200:
                raise PermissionError(f"HTTP {r.status_code} for {f}")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
        out.append(dest)
    return out


def default_extract(src: str, out: str) -> dict:
    """Same extractor and settings as INCLUDE ingestion (whole clip, 15 fps)."""
    from app.isl.data.ingest_bridge import ingest_module

    return ingest_module().work_item("full", src, out)


def video_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    ms = [i for i in zf.infolist() if not i.is_dir() and "__MACOSX" not in i.filename
          and PurePosixPath(i.filename).suffix.lower() in VIDEO_EXT]
    return sorted(ms, key=lambda i: i.header_offset)  # archive order = sequential range reads


def ingest_cislr(paths: IslPaths, archive, extract: Callable[[str, str], dict] | None = None, limit: int = 0,
                 workers: int = 1, max_tmp_bytes: int = 2 << 30, progress: Callable[[str], None] | None = None) -> dict:
    """Convert every (not yet done) video member of the CISLR zip to landmarks. ``archive`` is any seekable
    binary file object (``RangeFile`` in production, ``BytesIO`` in tests)."""
    extract = extract or default_extract
    man_path = paths.manifest("cislr")
    done = {r["id"] for r in read_manifest(man_path)}
    rows, split = csv_index(paths)
    out_dir = paths.landmarks / "cislr"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.tmp.mkdir(parents=True, exist_ok=True)
    stats = {"members": 0, "skipped_done": 0, "processed": 0, "ok": 0, "failed": 0}
    zf = zipfile.ZipFile(archive)
    members = video_members(zf)
    stats["members"] = len(members)
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    pending: list[tuple[dict, Path, Future | dict]] = []
    tmp_bytes = 0

    def finish(item):
        nonlocal tmp_bytes
        meta, tmp, fut = item
        info = fut.result() if isinstance(fut, Future) else fut
        try:
            tmp_bytes -= tmp.stat().st_size
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
        meta.update(info)
        append_jsonl(man_path, meta)
        stats["ok" if info.get("ok") else "failed"] += 1
        if progress:
            progress(f"{meta['id']} ok={info.get('ok')}")

    try:
        for info in members:
            uid = PurePosixPath(info.filename).stem
            cid = f"cislr/{uid}"
            if cid in done:
                stats["skipped_done"] += 1
                continue
            if limit and stats["processed"] >= limit:
                break
            row = rows.get(uid, {})
            gloss = row.get("word") or PurePosixPath(info.filename).parent.name
            name = _safe(uid) + ".npz"
            out = out_dir / name
            tmp = paths.tmp / f"cislr_{_safe(uid)}{PurePosixPath(info.filename).suffix.lower()}"
            with zf.open(info) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            tmp_bytes += tmp.stat().st_size
            meta = {"id": cid, "source": "cislr", "gloss": gloss, "uid": uid, "category": row.get("category"),
                    "member": info.filename, "path": f"data/isl/landmarks/cislr/{name}", "split": split.get(uid)}
            stats["processed"] += 1
            if pool is None:
                finish((meta, tmp, extract(str(tmp), str(out))))
            else:
                pending.append((meta, tmp, pool.submit(extract, str(tmp), str(out))))
                while pending and (len(pending) >= 2 * workers or tmp_bytes > max_tmp_bytes or pending[0][2].done()):
                    finish(pending.pop(0))
        for item in pending:
            finish(item)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return stats
