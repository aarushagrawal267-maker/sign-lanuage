"""Ingest ISL video / image sources that need our own MediaPipe extraction (generic, streaming, resumable).

Each item (a zip member read through HTTP range requests, a single video/image URL, or a Hugging Face dataset
row) is fetched to a temp file (hashed with SHA-256 on the way), converted to a RawSequence .npz and deleted, and
a row is appended to data/isl/manifests/<source>.jsonl. Re-running skips ids that already have an ok row.
Sources, licences and profiles: backend/app/isl/data/video_sources/catalog.py and docs/DATASETS_ISL.md 3.3.

Usage (repository root; CPU work runs with nice 10):
  python scripts/ingest_isl_video.py sources                       # catalogue + manifest counts
  python scripts/ingest_isl_video.py probe [SOURCE ...]            # are the files reachable? (CISLR: granted/pending)
  python scripts/ingest_isl_video.py run SOURCE [--workers N|auto] [--limit N] [--max-minutes M] [--per-class N]
  python scripts/ingest_isl_video.py list SOURCE [--per-class N]    # item list only (manifests/<source>_items.json)
  python scripts/ingest_isl_video.py islrtc-full [--paths A/Alphabet.mp4 ...]   # full-length ISLRTC compilations
  python scripts/ingest_isl_video.py status

Exit codes of `run`: 0 finished, 10 stopped after --max-minutes (more to do), 4 stopped for low disk,
3 gated dataset not accessible yet, 5 stopped for memory (--max-rss-gb; resumable, the queue restarts it),
6 another run of the same manifest is active (lock manifests/<manifest>.lock or a process scan), 2 error.
An HF alphabet listing that Hugging Face keeps refusing (429 / 503) exits 10 when it saved new rows (resumes next
chunk) and 2 when it made no progress; raw/<source>/listing_status.json says how far it got.
CISLR has exactly one ingestion path: this `run cislr` (the queue runs it as soon as the 1-byte probe of
dataset.csv answers); scripts/ingest_isl_gated.py cislr delegates here and shares the lock.
--workers auto: 1 while INCLUDE (1 worker) and ISLRTC (3 workers) ingestion run, otherwise the CPUs they leave free
(3 of 4 when neither runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("GLOG_minloglevel", "2")

log = logging.getLogger("ingest_isl_video")
EXIT_MORE, EXIT_DISK, EXIT_GATED, EXIT_ERR, EXIT_MEMORY, EXIT_BUSY = 10, 4, 3, 2, 5, 6


# --------------------------------------------------------------------------------------------- helpers

def _paths(root: str | None):
    from app.isl.paths import isl_paths

    return isl_paths(root or (ROOT / "data" / "isl"))


def _get(url: str, retries: int = 6, **kw):
    """GET with retries and backoff on 429 / 5xx / connection errors (the HF rows API answers 503 at times)."""
    from app.isl.data.remote import session

    kw.setdefault("timeout", 300)
    err = None
    for i in range(retries):
        try:
            r = session().get(url, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                err = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return r
        except Exception as e:  # noqa: BLE001
            if getattr(getattr(e, "response", None), "status_code", 500) < 500:
                raise
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(120, 5 * 2 ** i))
    raise OSError(f"GET {url} failed after {retries} tries: {err}")


def _cached_json(path: Path, url: str, max_age_days: float = 30.0):
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return data


def resolve_redirect(url: str) -> str:
    """Mendeley download URLs redirect (302) to a public S3 object; range requests go there directly."""
    from app.isl.data.remote import session

    r = session().get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
    try:
        if r.status_code not in (200, 206):
            raise OSError(f"HTTP {r.status_code} for {url}")
        return r.url
    finally:
        r.close()


def open_remote_zip(url: str, size: int | None = None, block_mb: int = 8):
    import zipfile

    from app.isl.data.remote import RangeFile, remote_size

    size = size or remote_size(url)
    return zipfile.ZipFile(RangeFile([url], [size], block=block_mb << 20, cache_blocks=8))


def members_of(zf) -> list[tuple[str, int]]:
    from app.isl.data.video_sources.fetch import UnsafeMember, safe_member_name

    out = []
    for i in zf.infolist():
        if i.is_dir() or "__MACOSX" in i.filename:
            continue
        try:
            safe_member_name(i.filename)
        except UnsafeMember as e:
            log.warning("skipping unsafe member: %s", e)
            continue
        out.append((i.filename, i.header_offset))
    return out


def mendeley(paths, dataset: str, version: int = 1) -> tuple[dict, dict[str, str]]:
    from app.isl.data.video_sources.catalog import MENDELEY_API
    from app.isl.data.video_sources.listings import mendeley_folder_paths

    raw = paths.raw / "mendeley"
    meta = _cached_json(raw / f"{dataset}.json", f"{MENDELEY_API}/{dataset}")
    folders = _cached_json(raw / f"{dataset}_folders_v{version}.json", f"{MENDELEY_API}/{dataset}/folders/{version}")
    return meta, mendeley_folder_paths(folders)


def ensure_hand_model() -> None:
    from app.isl.data.video_sources.extract import HAND_MODEL_URL, hand_model_path

    p = hand_model_path()
    if p.exists() and p.stat().st_size > 1_000_000:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), HAND_MODEL_URL], check=True, timeout=600)
    tmp.replace(p)
    log.info("downloaded %s (%d bytes)", p, p.stat().st_size)


def running_video_jobs(table=None) -> dict[str, bool]:
    """Are the long INCLUDE / ISLRTC ingestion jobs (scripts/ingest_isl.py) running? Matches the source argument
    exactly, so ``ingest_isl.py islrtc-recut`` (1 small worker) does not count as the 3-worker ISLRTC job."""
    from app.isl.data.pose_sources.common import find_running

    return {"include": bool(find_running("ingest_isl.py", "include", table=table)),
            "islrtc": bool(find_running("ingest_isl.py", "islrtc", table=table))}


def auto_workers(ncpu: int | None = None, jobs: dict[str, bool] | None = None) -> int:
    ncpu = ncpu or os.cpu_count() or 4
    jobs = running_video_jobs() if jobs is None else jobs
    if jobs.get("include"):
        return 1
    busy = 3 if jobs.get("islrtc") else 0
    return max(1, min(3, ncpu - busy - (0 if busy else 1)))


# --------------------------------------------------------------------------------------------- listings

def build_items(name: str, paths, per_class: int | None = None):
    """Items + Fetcher for one source (metadata is cached under data/isl/raw/)."""
    from app.isl.data.remote import hf_access, hf_tree, hf_url
    from app.isl.data.video_sources import listings as L
    from app.isl.data.video_sources.catalog import get_source
    from app.isl.data.video_sources.engine import Fetcher

    spec = get_source(name)
    ex = spec.extra
    raw = paths.raw / name
    raw.mkdir(parents=True, exist_ok=True)

    if name == "wslp":
        csv_path = raw / "train_WR.csv"
        if not csv_path.exists():
            csv_path.write_bytes(_get(hf_url(ex["repo"], ex["csv"])).content)
        url = hf_url(ex["repo"], ex["zip"])
        zf = open_remote_zip(url)
        items = L.wslp_items(csv_path.read_text(encoding="utf-8"), members_of(zf), url, split="train")
        return spec, items, Fetcher({url: zf})

    if name == "wslp_unlabelled":
        items, archives = [], {}
        for split, zp in ex["zips"].items():
            url = hf_url(ex["repo"], zp)
            zf = open_remote_zip(url)
            archives[url] = zf
            items += L.wslp_items("uid\n", members_of(zf), url, split=split, labelled=False)
        return spec, items, Fetcher(archives)

    if name == "cislr":
        from app.isl.data.cislr import csv_index, download_small_files

        state = hf_access(ex["repo"])
        if state != "granted":
            raise PermissionError(f"CISLR access: {state}")
        download_small_files(paths)
        rows, split = csv_index(paths)
        url = hf_url(ex["repo"], ex["zip"])
        zf = open_remote_zip(url)
        return spec, L.cislr_items(rows, split, members_of(zf), url), Fetcher({url: zf})

    if name == "isl_fingerspelling":
        for f in ("letter_annotations.csv", "letter_timestamps.csv", "fingerspelling_annotations_updated.csv",
                  "signer_split.csv", "split_info.csv", "README.md"):
            if not (raw / f).exists():
                (raw / f).write_bytes(_get(hf_url(ex["repo"], f)).content)
        tree_path = raw / "videos_tree.json"
        if not tree_path.exists():
            tree_path.write_text(json.dumps(hf_tree(ex["repo"], "videos")))
        files = [t["path"] for t in json.loads(tree_path.read_text()) if t.get("type") == "file"]
        items = L.fingerspelling_items((raw / "letter_annotations.csv").read_text(encoding="utf-8"),
                                       (raw / "letter_timestamps.csv").read_text(encoding="utf-8"), files,
                                       lambda p: hf_url(ex["repo"], p))
        return spec, items, Fetcher()

    if name in ("isl_csltr", "isl_csltr_words"):
        meta, _ = mendeley(paths, ex["dataset"], ex["version"])
        f = meta["files"][0]
        url = resolve_redirect(f["content_details"]["download_url"])
        zf = open_remote_zip(url, int(f["size"]))
        members = members_of(zf)
        if name == "isl_csltr":
            gl_member = next((m for m, _ in members if m.endswith("ISL Corpus sign glosses.csv")), None)
            glosses = L.csltr_glosses(zf.read(gl_member).decode("utf-8", "replace")) if gl_member else {}
            items = L.csltr_sentence_items(members, url, glosses, ex["folder"])
        else:
            items = L.csltr_word_items(members, url, ex["folder"])
        return spec, items, Fetcher({url: zf})

    if name == "mendeley_emergency":
        meta, fpaths = mendeley(paths, ex["dataset"], ex["version"])
        items, archives = [], {}
        for a in L.emergency_archives(meta["files"], fpaths, ex["folder"]):
            url = resolve_redirect(a["url"])
            zf = open_remote_zip(url, a["size"])
            items += L.emergency_items(a["word"], url, members_of(zf))
            zf.close()
            archives[url] = (lambda u=url, s=a["size"]: open_remote_zip(u, s))
        return spec, items, Fetcher(archives)

    if name == "mendeley_agri":
        meta, fpaths = mendeley(paths, ex["dataset"], ex["version"])
        return spec, L.agri_items(meta["files"], fpaths, ex["folder"]), Fetcher()

    if name == "mendeley_static":
        meta, fpaths = mendeley(paths, ex["dataset"], ex["version"])
        items = L.static_items(meta["files"], fpaths, per_class=per_class or ex["per_class"])
        return spec, items, Fetcher()

    if name in ("hf_hemg_alphabet", "hf_akrit_alphabet"):
        # polite, resumable rows-API listing (backoff, Retry-After, progress saved after every page, status in
        # raw/<source>/listing_status.json); raises hf_listing.ListingError when HF keeps refusing
        from app.isl.data.video_sources.hf_listing import build_hf_items

        return spec, build_hf_items(spec, raw, per_class=per_class), Fetcher()

    raise ValueError(f"no listing for {name}")


# --------------------------------------------------------------------------------------------- commands

def cmd_sources(args) -> int:
    from app.isl.data.video_sources.catalog import SOURCES
    from app.isl.data.video_sources.engine import summarise

    paths = _paths(args.root)
    for s in SOURCES.values():
        st = summarise(paths.manifest(s.manifest_name))
        print(f"{s.name:20s} {s.profile:15s} {s.licence:30s} {s.mode:6s} ok={st['ok']:6d} failed={st['failed']:4d} "
              f"glosses={st['glosses']:5d}  {s.url}")
    return 0


def cmd_probe(args) -> int:
    from app.isl.data.remote import hf_access, hf_url, session
    from app.isl.data.video_sources.catalog import MENDELEY_API, SOURCES

    names = args.names or list(SOURCES)
    worst = 0
    for n in names:
        s = SOURCES[n]
        ex = s.extra
        try:
            if s.gated:
                state = hf_access(ex["repo"], probe=ex.get("probe"))  # a data file (dataset.csv), not the card
                print(f"{n}: {state}")
                worst = max(worst, 0 if state == "granted" else EXIT_GATED)
                continue
            if "repo" in ex:
                path = ex.get("zip") or (list(ex["zips"].values())[0] if "zips" in ex else "README.md")
                r = session().get(hf_url(ex["repo"], path), headers={"Range": "bytes=0-0"}, timeout=60)
                code = r.status_code
                r.close()
            elif "dataset" in ex and "/" in ex["dataset"]:
                r = session().get("https://datasets-server.huggingface.co/rows",
                                  params={"dataset": ex["dataset"], "config": "default", "split": ex["splits"][0],
                                          "offset": 0, "length": 1}, timeout=60)
                code = r.status_code
            else:
                r = session().get(f"{MENDELEY_API}/{ex['dataset']}", timeout=120)
                code = r.status_code
            print(f"{n}: HTTP {code}")
            worst = max(worst, 0 if code in (200, 206) else EXIT_ERR)
        except Exception as e:  # noqa: BLE001
            print(f"{n}: error {type(e).__name__}: {e}")
            worst = EXIT_ERR
    return worst


def _setup_logging(paths, name: str) -> None:
    (paths.root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(paths.root / "logs" / f"video_{name}.log")],
    )


def source_lock(paths, spec, name: str, table=None):
    """Exclusive lock for one manifest (wslp and wslp_unlabelled share ``wslp.jsonl``; ``cislr`` is shared with
    scripts/ingest_isl_gated.py) plus a process scan for an older run without a lock. Returns the open lock file
    (keep it until the run ends) or None when another run is active."""
    import fcntl

    from app.isl.data.pose_sources.common import find_running

    others = find_running(Path(__file__).name, ("run", name), table=table)
    if others:
        log.error("%s: another run is active (pid %s); exiting", name, others)
        return None
    lock_path = paths.manifests / f"{spec.manifest_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        log.error("%s: %s is locked by another run; exiting", name, lock_path)
        return None
    fh.write(f"{os.getpid()} {name}\n")
    fh.flush()
    return fh


def cmd_run(args) -> int:
    from app.isl.data.video_sources import engine
    from app.isl.data.video_sources.catalog import get_source

    paths = _paths(args.root)
    _setup_logging(paths, args.source)
    spec = get_source(args.source)
    lock = source_lock(paths, spec, args.source)
    if lock is None:
        return EXIT_BUSY
    try:
        return _run_locked(args, paths, spec, engine)
    finally:
        lock.close()


def _run_locked(args, paths, spec, engine) -> int:
    if spec.mode in ("hands", "image"):
        ensure_hand_model()
    workers = auto_workers() if args.workers == "auto" else max(1, int(args.workers))
    try:
        spec, items, fetcher = build_items(args.source, paths, per_class=args.per_class)
    except PermissionError as e:
        log.info("%s: %s", args.source, e)
        return EXIT_GATED
    except OSError as e:
        rc = _listing_rc(e)
        if rc is None:
            raise
        log.error("%s: listing incomplete: %s", args.source, e)
        return rc
    engine.dump_items(paths.manifests / f"{spec.name}_items.json", items)  # what this run selected (audit)
    log.info("%s: %d items listed, %d workers, profile %s, licence %s", spec.name, len(items), workers, spec.profile,
             spec.licence)
    stats = engine.run(spec, items, paths, fetcher, workers=workers, limit=args.limit, max_minutes=args.max_minutes,
                       min_free_gb=args.min_free_gb, progress=log.info,
                       max_rss_gb=getattr(args, "max_rss_gb", 0.0) or 0.0)
    log.info("%s run finished: %s", spec.name, stats)
    if stats["stopped"] == "memory":
        return EXIT_MEMORY
    if stats["stopped"] == "disk":
        return EXIT_DISK
    if stats["stopped"] == "time":
        return EXIT_MORE
    if args.limit and stats["todo"] >= args.limit:
        return EXIT_MORE
    return 0


def _listing_rc(e: OSError) -> int | None:
    """Exit code for an HF listing the server refused (hf_listing.ListingError): 10 (more to do) when this run saved
    new rows - the queue starts the next chunk, which resumes the listing, without counting a failed try - else 2
    (the queue counts it and backs off). None for any other OSError."""
    from app.isl.data.video_sources.hf_listing import ListingError

    if not isinstance(e, ListingError):
        return None
    return EXIT_MORE if e.progressed and not e.permanent else EXIT_ERR


def cmd_list(args) -> int:
    """Write manifests/<source>_items.json (the run's item list) without fetching or extracting any item. Holds
    the source's manifest lock, so it never runs alongside an ingestion run of the same source (exit 6 then)."""
    from app.isl.data.video_sources import engine
    from app.isl.data.video_sources.catalog import get_source

    paths = _paths(args.root)
    _setup_logging(paths, args.source)
    spec = get_source(args.source)
    lock = source_lock(paths, spec, args.source)
    if lock is None:
        return EXIT_BUSY
    try:
        try:
            spec, items, fetcher = build_items(args.source, paths, per_class=args.per_class)
        except PermissionError as e:
            log.info("%s: %s", args.source, e)
            return EXIT_GATED
        except OSError as e:
            rc = _listing_rc(e)
            if rc is None:
                raise
            log.error("%s: listing incomplete: %s", args.source, e)
            return rc
        fetcher.close_archives()
        out = paths.manifests / f"{spec.name}_items.json"
        engine.dump_items(out, items)
        log.info("%s: %d items listed -> %s (nothing fetched)", spec.name, len(items), out)
        return 0
    finally:
        lock.close()


ISLRTC_FULL_PATTERN = r"(alphabet|fingerspell)"


def islrtc_full_targets(index: list[dict], done: set[str]) -> list[str]:
    """ISLRTC entries that are alphabet / fingerspelling / number-range compilations (cut at 10 s by the default
    job): names matching ``alphabet|fingerspell`` and every ``Numbers/`` entry that is a compilation (a range or a
    regional set in brackets)."""
    import re

    from app.isl.vocab.normalize import parse_islrtc_name

    out = []
    for f in index:
        p = f["path"]
        stem = Path(p).stem
        hit = bool(re.search(ISLRTC_FULL_PATTERN, stem, re.I))
        if not hit and p.startswith("Numbers/"):
            try:
                hit = parse_islrtc_name(p).kind == "compilation"
            except Exception:  # noqa: BLE001
                hit = False
        if hit and f"islrtc/{p}" not in done:
            out.append(p)
    return out


def cmd_islrtc_full(args) -> int:
    """Re-extract ISLRTC compilation entries full length (up to --max-s) into landmarks/islrtc_full/ through
    scripts/ingest_isl.py's own backward-compatible code path (--only --full --out-subdir islrtc_full), run
    in this process so the long ``ingest_isl.py islrtc`` job's process name stays unique."""
    import ingest_isl  # scripts/ is on sys.path (this file's folder)

    from app.isl.data.manifest import read_manifest

    paths = _paths(args.root)
    _setup_logging(paths, "islrtc_full")
    index = json.loads(paths.islrtc_index.read_text()) if paths.islrtc_index.exists() else []
    done = {r["id"] for r in read_manifest(paths.manifest("islrtc_full"))}
    targets = list(args.paths) if args.paths else islrtc_full_targets(index, done)
    targets = [t for t in targets if f"islrtc/{t}" not in done]
    log.info("islrtc-full: %d entries to extract: %s", len(targets), targets)
    if not targets:
        return 0
    ns = argparse.Namespace(only=targets, full=True, max_s=args.max_s, out_subdir="islrtc_full", limit=0,
                            workers=max(1, int(auto_workers() if args.workers == "auto" else args.workers)),
                            tmp=str(paths.tmp / "video_islrtc_full"))  # not "islrtc_*": the main job unlinks those
    ingest_isl.run_islrtc(ns)
    return 0


def cmd_status(args) -> int:
    from app.isl.data.video_sources.catalog import SOURCES
    from app.isl.data.video_sources.engine import summarise

    paths = _paths(args.root)
    out = {s.name: summarise(paths.manifest(s.manifest_name)) for s in SOURCES.values()}
    out["islrtc_full"] = summarise(paths.manifest("islrtc_full"))
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="ISL data root (default data/isl)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sources")
    sub.add_parser("status")
    p = sub.add_parser("probe")
    p.add_argument("names", nargs="*")
    p = sub.add_parser("run")
    p.add_argument("source")
    p.add_argument("--workers", default="1", help="number of extraction processes, or 'auto'")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=0.0, help="stop taking new items after this long (exit 10)")
    p.add_argument("--min-free-gb", type=float, default=10.0)
    p.add_argument("--per-class", type=int, default=None, help="image sets: sample size per class")
    p.add_argument("--max-rss-gb", type=float, default=1.5,
                   help="soft memory cap per process (this one and each worker); stop with exit 5 above it; 0 = off")
    p = sub.add_parser("list", help="write manifests/<source>_items.json only (no fetch / extraction)")
    p.add_argument("source")
    p.add_argument("--per-class", type=int, default=None, help="image sets: sample size per class")
    p = sub.add_parser("islrtc-full")
    p.add_argument("--paths", nargs="*", help="ISLRTC paths (default: alphabet / fingerspell / number compilations)")
    p.add_argument("--max-s", type=float, default=180.0)
    p.add_argument("--workers", default="1")
    args = ap.parse_args(argv)
    try:
        os.nice(10)
    except OSError:
        pass
    os.environ.setdefault("SLR_TORCH_THREADS", "2")
    cmds = {"sources": cmd_sources, "probe": cmd_probe, "run": cmd_run, "list": cmd_list,
            "islrtc-full": cmd_islrtc_full, "status": cmd_status}
    try:
        return cmds[args.cmd](args)
    except KeyboardInterrupt:
        raise
    except Exception:  # noqa: BLE001
        logging.getLogger("ingest_isl_video").exception("failed")
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
