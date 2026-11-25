"""Ingest the gated ISL datasets (CISLR, iSign) as soon as Hugging Face access is granted.

Nothing large is ever stored: CISLR's 1.17 GB zip and iSign's 170 GB pose archive (a zip split into four
parts) are read through HTTP range requests (one member at a time), converted to the app's RawSequence
landmark files and the source bytes are discarded. Both jobs are resumable per member through their
manifests (data/isl/manifests/{cislr,isign}.jsonl).

Usage (from the repository root; runs with nice 10):
  python scripts/ingest_isl_gated.py --check [cislr|isign]   # granted/pending; exit 0 all granted, 3 otherwise
  python scripts/ingest_isl_gated.py cislr [--workers 2] [--limit N]   # delegates to ingest_isl_video.py run cislr
  python scripts/ingest_isl_gated.py isign [--subset 0] [--mode zip|stream] [--limit N] [--swap-hands]

CISLR has ONE ingestion path, ``scripts/ingest_isl_video.py run cislr`` (what scripts/isl_queue.sh runs as soon
as access is granted): ``cislr`` here only delegates to it in-process, sharing its lock
(``manifests/cislr.lock``), so the two can never write ``manifests/cislr.jsonl`` at the same time.
iSign: ``manifests/isign.lock`` (plus a process scan) keeps one copy; uids already ingested from GloSL
(``data/isl/derived/glosl_isign_uids.txt``, the same poses re-packaged) are skipped, so with ``--subset 0`` only
the iSign clips GloSL lacks are converted. The queue starts it when the iSign probe answers.

Access check: a 1-byte range request for a data file (the dataset card README.md is public even without
access, so it cannot be used). Hugging Face credentials come from the egress proxy or the HF_TOKEN
environment variable; they are never printed or stored.

Licences: CISLR (AFL-3.0 on the card, pending review -> research profile only), iSign (CC BY-NC-SA 4.0,
research only, never commercial). See docs/DATASETS_ISL.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("GLOG_minloglevel", "2")

log = logging.getLogger("ingest_isl_gated")


def _setup_logging(paths, name: str) -> None:
    paths.root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(paths.root / "logs" / f"{name}.log")],
    )


def check(repos: list[str]) -> int:
    from app.isl.data.remote import hf_access

    res = {r: hf_access(r) for r in repos}
    for r, v in res.items():
        print(f"{r}: {v}")
    return 0 if all(v == "granted" for v in res.values()) else 3


def run_cislr(args, paths) -> int:
    """Delegate to the single CISLR path (``ingest_isl_video.py run cislr``, same ids/paths/manifest plus
    sha256/crc32) in this process; its manifest lock refuses a concurrent queue run (exit 6)."""
    import ingest_isl_video  # scripts/ is on sys.path (this file's folder)

    log.info("cislr: delegating to scripts/ingest_isl_video.py run cislr (the one CISLR ingestion path)")
    argv = ["run", "cislr", "--workers", str(max(1, min(2, args.workers)))]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.root:
        argv = ["--root", str(args.root), *argv]
    return ingest_isl_video.main(argv)


def isign_wanted(sentences: dict[str, str], skip: set[str], subset: list[str] | None) -> set[str]:
    """iSign uids to convert: the subset (or every sentence) minus the uids GloSL already provides."""
    base = set(subset) if subset is not None else set(sentences)
    return base - skip


def read_uid_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def run_isign(args, paths) -> int:
    from app.isl.data.isign import (
        CSV_FILES,
        ISIGN_REPO,
        POSE_PARTS,
        ingest_isign_stream,
        ingest_isign_zip,
        last_offset,
        read_sentences,
        select_subset,
    )
    from app.isl.data.remote import ConcatStream, RangeFile, hf_access, hf_tree, hf_url, session, sniff_format
    from app.isl.paths import PACK_DIR
    from app.isl.vocab.build import read_vocabulary

    state = hf_access(ISIGN_REPO)
    if state != "granted":
        log.info("iSign access: %s (request access on the dataset page, then re-run)", state)
        return 3
    lock = _isign_lock(paths)
    if lock is None:
        return 6
    raw_dir = paths.raw / "isign"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in CSV_FILES:  # ~10 MB in total
        dest = raw_dir / f
        if not dest.exists():
            r = session().get(hf_url(ISIGN_REPO, f), timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
    sentences = read_sentences(paths)
    subset = None
    if args.subset:
        vocab = PACK_DIR / "vocabulary.jsonl"
        words = []
        if vocab.exists():
            for e in read_vocabulary(vocab):
                words += [e.headword] + list(e.synonyms)
        subset = select_subset(sentences, words, n=args.subset)
        log.info("iSign subset: %d of %d sentences contain a vocabulary word", len(subset), len(sentences))
    skip_path = Path(args.skip_uids) if args.skip_uids else paths.root / "derived" / "glosl_isign_uids.txt"
    skip = read_uid_file(skip_path)
    wanted = isign_wanted(sentences, skip, subset)
    log.info("iSign: %d uids to convert (%d already ingested from GloSL, %s, are skipped)", len(wanted),
             len((set(subset) if subset is not None else set(sentences)) & skip), skip_path.name)
    if not wanted:
        return 0
    sizes = {t["path"]: int(t.get("size") or 0) for t in hf_tree(ISIGN_REPO)}
    urls = [hf_url(ISIGN_REPO, p) for p in POSE_PARTS]
    part_sizes = [sizes[p] for p in POSE_PARTS]
    log.info("iSign poses: %d parts, %.1f GB in total (never stored)", len(urls), sum(part_sizes) / 1e9)
    if args.mode == "zip":
        rf = RangeFile(urls, part_sizes, block=args.block_mb << 20, cache_blocks=16)
        fmt = sniff_format(rf.read_at(0, 512))
        if fmt != "zip":
            log.warning("archive starts like %r, not a zip; use --mode stream", fmt)
            return 2
        stats = ingest_isign_zip(paths, rf, wanted=wanted, sentences=sentences, limit=args.limit,
                                 swap_hands=args.swap_hands, progress=lambda m: log.info(m))
        log.info("iSign done: %s; transferred %.1f MB in %d requests", stats, rf.bytes_fetched / 1e6, rf.requests)
    else:
        head = RangeFile(urls[:1], part_sizes[:1]).read_at(0, 512)
        fmt = sniff_format(head)
        start = last_offset(paths) if fmt in ("zip", "tar") else 0
        if start:
            log.info("resuming the stream at byte %d", start)
        stream = ConcatStream(urls, part_sizes, start=start)
        stats = ingest_isign_stream(paths, stream, fmt, wanted=wanted, sentences=sentences, limit=args.limit,
                                    swap_hands=args.swap_hands, start_offset=start, progress=lambda m: log.info(m))
        log.info("iSign stream done: %s", stats)
    return 0


def _isign_lock(paths):
    import fcntl

    from app.isl.data.pose_sources.common import find_running

    if find_running(Path(__file__).name, "isign"):
        log.error("isign: another ingest_isl_gated.py isign is running; exiting")
        return None
    lp = paths.manifests / "isign.lock"
    lp.parent.mkdir(parents=True, exist_ok=True)
    fh = lp.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        log.error("isign: %s is locked by another run; exiting", lp)
        return None
    return fh


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", choices=["cislr", "isign"])
    ap.add_argument("--check", action="store_true", help="only report access (exit 0 granted, 3 pending)")
    ap.add_argument("--workers", type=int, default=1, help="extraction processes (max 2 while other jobs run)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0,
                    help="iSign: only the N sentences with the most vocabulary words (0 = all not in GloSL)")
    ap.add_argument("--skip-uids", default=None,
                    help="iSign: file of uids to skip (default data/isl/derived/glosl_isign_uids.txt)")
    ap.add_argument("--mode", choices=["zip", "stream"], default="zip", help="iSign archive access mode")
    ap.add_argument("--block-mb", type=int, default=8, help="range-request block size")
    ap.add_argument("--swap-hands", action="store_true", help="iSign: swap left/right hands (see docs)")
    ap.add_argument("--root", help="ISL data root (default data/isl)")
    args = ap.parse_args(argv)
    try:
        os.nice(10)
    except OSError:
        pass
    from app.isl.data.cislr import CISLR_REPO
    from app.isl.data.isign import ISIGN_REPO
    from app.isl.paths import isl_paths

    paths = isl_paths(args.root or (ROOT / "data" / "isl"))
    if args.check or not args.source:
        repos = [CISLR_REPO, ISIGN_REPO] if not args.source else [CISLR_REPO if args.source == "cislr" else ISIGN_REPO]
        return check(repos)
    _setup_logging(paths, args.source)
    return run_cislr(args, paths) if args.source == "cislr" else run_isign(args, paths)


if __name__ == "__main__":
    sys.exit(main())
