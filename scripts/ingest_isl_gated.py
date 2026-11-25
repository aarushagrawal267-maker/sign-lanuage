"""Ingest the gated ISL datasets (CISLR, iSign) as soon as Hugging Face access is granted.

Nothing large is ever stored: CISLR's 1.17 GB zip and iSign's 170 GB pose archive (a zip split into four
parts) are read through HTTP range requests (one member at a time), converted to the app's RawSequence
landmark files and the source bytes are discarded. Both jobs are resumable per member through their
manifests (data/isl/manifests/{cislr,isign}.jsonl).

Usage (from the repository root; runs with nice 10):
  python scripts/ingest_isl_gated.py --check                 # granted/pending per repo; exit 0 all granted, 3 otherwise
  python scripts/ingest_isl_gated.py cislr [--workers 2] [--limit N]
  python scripts/ingest_isl_gated.py isign [--subset 10000] [--mode zip|stream] [--limit N] [--swap-hands]

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
    from app.isl.data.cislr import CISLR_REPO, ZIP_PATH, download_small_files, ingest_cislr
    from app.isl.data.remote import RangeFile, hf_access, hf_url, remote_size

    state = hf_access(CISLR_REPO)
    if state != "granted":
        log.info("CISLR access: %s (request access on the dataset page, then re-run)", state)
        return 3
    download_small_files(paths)
    url = hf_url(CISLR_REPO, ZIP_PATH)
    size = remote_size(url)
    log.info("CISLR zip: %.2f GB, reading members with range requests", size / 1e9)
    rf = RangeFile([url], [size], block=args.block_mb << 20, cache_blocks=8)
    stats = ingest_cislr(paths, rf, limit=args.limit, workers=max(1, min(2, args.workers)),
                         progress=lambda m: log.info(m))
    log.info("CISLR done: %s; transferred %.1f MB in %d requests", stats, rf.bytes_fetched / 1e6, rf.requests)
    return 0


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
    raw_dir = paths.raw / "isign"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in CSV_FILES:  # ~10 MB in total
        dest = raw_dir / f
        if not dest.exists():
            r = session().get(hf_url(ISIGN_REPO, f), timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
    sentences = read_sentences(paths)
    wanted = None
    if args.subset:
        vocab = PACK_DIR / "vocabulary.jsonl"
        words = []
        if vocab.exists():
            for e in read_vocabulary(vocab):
                words += [e.headword] + list(e.synonyms)
        wanted = set(select_subset(sentences, words, n=args.subset))
        log.info("iSign subset: %d of %d sentences contain a vocabulary word", len(wanted), len(sentences))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", choices=["cislr", "isign"])
    ap.add_argument("--check", action="store_true", help="only report access (exit 0 granted, 3 pending)")
    ap.add_argument("--workers", type=int, default=1, help="extraction processes (max 2 while other jobs run)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset", type=int, default=10_000, help="iSign: sentences to convert (0 = all)")
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
