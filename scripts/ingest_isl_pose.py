"""Ingest pose-native Indian Sign Language sources into RawSequence .npz files (no video decoding).

Sources (licences and profiles: docs/DATASETS_ISL.md, "Pose-native sources"):
  glosl          GloSL-ISL (pedroodb/glosl-isl, HF): iSign/ISLTranslate sentence poses, CC BY-NC 4.0, research.
  isl500         ISL500 / ISL-DATA (HF): 500 words x 15 signers, MediaPipe hands + MMPose WholeBody, research.
  openhands-raw  OpenHands raw ISL poses (Zenodo 5558897): unlabelled, pre-training, research (budget-capped).
  openhands-fs   OpenHands Indian fingerspelling (Zenodo 6731224, Indian.zip): 36 letters/digits, research.
  status         counts per manifest and disk use.

Every job is resumable: finished items are recorded (ok: true) in data/isl/manifests/<source>.jsonl (same
conventions as scripts/ingest_isl.py: id, source, gloss, path relative to the repo, n_frames, fps, hands_frac,
ok, error, secs) and skipped on the next run. Outputs go to data/isl/landmarks/<source>/. Files are converted in
memory (no archives on disk) and the jobs stop cleanly when free disk space would fall under --min-free-gb.

Storage: landmark arrays are float16 by default (--dtype float32 to keep full precision) and strided to about
--target-fps (default 15, like INCLUDE; 0 keeps every frame). RawSequence.load reads both; the pose adapters
upcast to float32. openhands-fs runs our own MediaPipe extractor on its tiny MP4s (exact parity; pickles are never
loaded).

Usage (from the repo root; long jobs detached, see docs/DATASETS_ISL.md):
  nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py glosl --workers 8
  nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py isl500 --workers 6
  nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py openhands-raw --budget-gb 12
  nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py openhands-fs
  .venv/bin/python scripts/ingest_isl_pose.py status
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("GLOG_minloglevel", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from app.isl.data.manifest import read_manifest  # noqa: E402
from app.isl.data.pose_sources import common, fingerspell, glosl, isl500, openhands  # noqa: E402
from app.isl.data.pose_sources.posefmt import holistic_pose_to_raw  # noqa: E402

DATA = ROOT / "data" / "isl"
HF = "https://huggingface.co"
log = logging.getLogger("ingest_isl_pose")


# --------------------------------------------------------------------------------------------- shared

class Manifest:
    """Append-only JSON-lines manifest (thread-safe); ``done`` = ids with an ok record."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {r["id"] for r in read_manifest(path, only_ok=True)}
        self.lock = threading.Lock()

    def add(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self.lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            if rec.get("ok"):
                self.done.add(rec["id"])


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def http_get(url: str, retries: int = 8, timeout: float = 120.0) -> bytes:
    s = openhands._session()
    err = None
    for i in range(retries):
        try:
            r = s.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                return r.content
            err = f"HTTP {r.status_code}"
            if r.status_code in (401, 403, 404):
                raise PermissionError(f"{err} for {url}")
            if r.status_code == 429:
                time.sleep(min(300, float(r.headers.get("Retry-After") or 30)))
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** i))
    raise OSError(f"GET failed after {retries} tries: {url}: {err}")


def hf_sha(repo: str) -> str:
    return json.loads(http_get(f"{HF}/api/datasets/{repo}"))["sha"]


def hf_tree(repo: str, folder: str, revision: str, recursive: bool = False) -> list[dict]:
    url = f"{HF}/api/datasets/{repo}/tree/{revision}/{urllib.parse.quote(folder)}" + ("?recursive=true" if recursive else "")
    out = []
    s = openhands._session()
    while url:
        for i in range(8):
            r = s.get(url, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(min(60, 2 ** i))
        else:
            raise OSError(f"tree listing failed: {url} ({r.status_code})")
        out += r.json()
        url = None
        for part in r.headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1: part.find(">")]
    return out


def hf_file(repo: str, path: str, revision: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def fetch_small(url: str, dest: Path) -> Path:
    if not dest.exists() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(http_get(url))
        tmp.replace(dest)
    return dest


MEM: common.MemoryGuard | None = None  # set in main() (--max-rss-gb)


def mem_check(force: bool = False) -> None:
    if MEM is not None:
        MEM.check(force=force)


def run_pool(items, worker, workers: int, guard: common.DiskGuard, label: str, log_every: int = 500) -> dict:
    """Run ``worker(item) -> record`` over ``items`` with a bounded in-flight window; stops on DiskFull or when
    the memory guard (``--max-rss-gb``) trips (``stats["stopped"]`` starts with ``memory:`` then). A record
    with ``deferred: true`` was not written and is counted separately."""
    stats = {"processed": 0, "ok": 0, "failed": 0, "deferred": 0, "stopped": None}
    t0 = time.time()
    it = iter(items)
    with ThreadPoolExecutor(workers) as pool:
        running = set()
        exhausted = False
        while True:
            while not exhausted and len(running) < workers * 3:
                try:
                    guard.check()
                    mem_check()
                except common.DiskFull as e:
                    stats["stopped"] = str(e)
                    log.error("%s: stopping: %s", label, e)
                    exhausted = True
                    break
                except common.MemoryCap as e:
                    stats["stopped"] = f"memory: {e}"
                    log.error("%s: stopping for memory (resumable, exit %d): %s", label, common.EXIT_MEMORY, e)
                    exhausted = True
                    break
                try:
                    running.add(pool.submit(worker, next(it)))
                except StopIteration:
                    exhausted = True
            if not running:
                break
            done, running = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                rec = f.result()
                stats["processed"] += 1
                stats["deferred" if rec.get("deferred") else "ok" if rec.get("ok") else "failed"] += 1
                if stats["processed"] % log_every == 0:
                    rate = stats["processed"] / max(1e-6, time.time() - t0)
                    log.info("%s: %d processed (%d ok, %d failed) %.1f/s, free %.1f GB", label, stats["processed"],
                             stats["ok"], stats["failed"], rate, common.free_gb(DATA))
    log.info("%s: finished this run: %s", label, stats)
    return stats


def stop_code(stats: dict) -> int:
    s = str(stats.get("stopped") or "")
    return common.EXIT_MEMORY if s.startswith("memory") else 3 if s else 0


def stride_info(fps: float, target: float) -> int:
    return common.stride_for(fps, target)


# --------------------------------------------------------------------------------------------- GloSL

def run_glosl(args) -> int:
    raw_dir = DATA / "raw" / "glosl"
    out_dir = DATA / "landmarks" / "glosl"
    man = Manifest(DATA / "manifests" / "glosl.jsonl")
    sha_file = raw_dir / "revision.txt"
    if sha_file.exists():
        sha = sha_file.read_text().strip()
    else:
        sha = hf_sha(glosl.REPO)
        raw_dir.mkdir(parents=True, exist_ok=True)
        sha_file.write_text(sha + "\n")
    for f in ("annotations.csv", "card.json", "README.md", "splits/glosl_v1.csv", "diagnostics/joint_train_exclude.csv"):
        fetch_small(hf_file(glosl.REPO, f, sha), raw_dir / f)
    try:
        fetch_small(glosl.ISLTRANSLATE_CSV, raw_dir / "ISLTranslate.csv")
    except Exception as e:  # noqa: BLE001
        log.warning("ISLTranslate.csv not fetched (%s); in_isltranslate will be null", e)
    rows = glosl.read_annotations(raw_dir / "annotations.csv")
    split_v1 = glosl.read_split_file(raw_dir / "splits" / "glosl_v1.csv")
    exclude = glosl.read_id_column(raw_dir / "diagnostics" / "joint_train_exclude.csv")
    islt = glosl.read_id_column(raw_dir / "ISLTranslate.csv", "uid") if (raw_dir / "ISLTranslate.csv").exists() else None
    derived = DATA / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "glosl_isign_uids.txt").write_text("".join(sorted(r["id"] + "\n" for r in rows)))
    log.info("glosl: %d annotation rows (revision %s), %d done", len(rows), sha[:12], len(man.done))
    todo = [r for r in rows if glosl.clip_id(r["id"]) not in man.done]
    if args.limit:
        todo = todo[: args.limit]
    log.info("glosl: %d clips to process", len(todo))
    guard = common.DiskGuard(DATA, min_free_gb=args.min_free_gb, every=100)

    def work(r: dict) -> dict:
        t0 = time.time()
        gid = r["id"]
        bucket = Path(r["pose_file"]).parent.name or "x"
        out = out_dir / bucket / (common.safe_name(gid) + ".npz")
        rec = {"id": glosl.clip_id(gid), "source": "glosl", "gloss": "", "text": r.get("text"),
               "split": split_v1.get(gid) or r.get("split"), "split_annotations": r.get("split"),
               "video_ref": r.get("video_ref"), "isign_uid": gid,
               "in_isltranslate": (gid in islt) if islt is not None else None,
               "joint_train_exclude": gid in exclude, "pose_file": r["pose_file"], "revision": sha,
               "path": rel(out), "licence": glosl.LICENCE}
        try:
            buf = http_get(hf_file(glosl.REPO, r["pose_file"], sha))
            seq, hdr = holistic_pose_to_raw(buf, glosl.EXTRACTOR_VERSION, z_mode="raw")
            del buf
            asp = common.hand_aspect([seq.left, seq.right])
            stride = common.stride_for(hdr["fps"], args.target_fps)
            saved = common.save_compact(seq, out, dtype=args.dtype, stride=stride)
            rec.update(common.seq_stats(saved))
            rec.update({"ok": True, "fps": round(hdr["fps"] / stride, 3), "fps_native": hdr["fps"],
                        "n_frames_native": len(seq), "stride": stride, "dtype": args.dtype, "width": hdr["width"],
                        "height": hdr["height"], "aspect_est": asp[0] if asp else None,
                        "aspect_cv": asp[1] if asp else None, "extractor_version": glosl.EXTRACTOR_VERSION})
        except Exception as e:  # noqa: BLE001
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        rec["secs"] = round(time.time() - t0, 2)
        man.add(rec)
        return rec

    stats = run_pool(todo, work, args.workers, guard, "glosl")
    return stop_code(stats)


# --------------------------------------------------------------------------------------------- ISL500

def isl500_index(sha: str, raw_dir: Path) -> tuple[list[dict], dict]:
    f_h5, f_pose = raw_dir / "index_mediapipe.json", raw_dir / "index_pose.json"
    if not f_h5.exists():
        items = []
        for u in range(1, 16):
            items += [x for x in hf_tree(isl500.REPO, f"Landmarks/MediaPipe/ISL_DATA_USER{u:03d}", sha)
                      if x.get("type") == "file" and x["path"].endswith(".h5")]
        f_h5.write_text(json.dumps([{"path": x["path"], "size": x.get("size")} for x in items]))
    if not f_pose.exists():
        items = []
        for u in range(1, 16):
            items += [x for x in hf_tree(isl500.REPO, f"Landmarks/Pose/User{u:03d}", sha, recursive=True)
                      if x.get("type") == "file" and x["path"].endswith(".npy")]
        f_pose.write_text(json.dumps([{"path": x["path"], "size": x.get("size")} for x in items]))
    h5 = json.loads(f_h5.read_text())
    pose: dict[tuple[str, str], list[str]] = {}
    for x in json.loads(f_pose.read_text()):
        parts = x["path"].split("/")
        user = isl500.user_of(parts[2])
        key = isl500.clip_key(Path(parts[-1]).stem)
        pose.setdefault((user, key), []).append(x["path"])
    for v in pose.values():  # exact-name file first ("Round1/<name>.npy" for plain names)
        v.sort(key=lambda p: (0 if "/Round1/" in p else 1, p))
    return h5, pose


def read_pd_splits(raw_dir: Path) -> dict[tuple[str, str], str]:
    out = {}
    for split, f in (("train", "PersonDependentTrain.csv"), ("test", "PersonDependentTest.csv")):
        p = raw_dir / f
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.endswith(".h5"):
                continue
            parts = line.split("/")
            try:
                user = isl500.user_of(parts[-2])
            except ValueError:
                continue
            out[(user, isl500.clip_key(Path(parts[-1]).stem))] = split
    return out


def run_isl500(args) -> int:
    import h5py

    raw_dir = DATA / "raw" / "isl500"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = DATA / "landmarks" / "isl500"
    man = Manifest(DATA / "manifests" / "isl500.jsonl")
    sha_file = raw_dir / "revision.txt"
    if not sha_file.exists():
        sha_file.write_text(hf_sha(isl500.REPO) + "\n")
    sha = sha_file.read_text().strip()
    for f in ("README.md", "MappingFiles/PersonDependentTrain.csv", "MappingFiles/PersonDependentTest.csv"):
        fetch_small(hf_file(isl500.REPO, f, sha), raw_dir / Path(f).name)
    h5_files, pose_idx = isl500_index(sha, raw_dir)
    pd = read_pd_splits(raw_dir)
    # Per-signer fallback frame size = the most common size estimated from the hands over ALL of that signer's
    # clips (isl500.signer_size), applied only after every clip of the run has been estimated: clips without an
    # estimate are deferred to a second pass, so the result no longer depends on which thread finished first.
    sizes = isl500.SignerSizes()
    prior_ok = read_manifest(man.path, only_ok=True)
    for r in prior_ok:
        if r.get("size_from") == "hands":
            sizes.add(r["user"], (r["width"], r["height"]))
    items, all_items = [], []
    seen = set()
    for x in sorted(h5_files, key=lambda x: x["path"]):
        parts = x["path"].split("/")
        user = isl500.user_of(parts[2])
        key = isl500.clip_key(Path(parts[-1]).stem)
        cid = isl500.clip_id(user, key)
        if cid in seen:
            continue
        seen.add(cid)
        all_items.append((cid, user, key, x["path"], pose_idx.get((user, key), [])))
        if cid in man.done:
            continue
        items.append(all_items[-1])
    if args.limit:
        items = items[: args.limit]
    log.info("isl500: %d h5 clips, %d with MMPose, %d to process (revision %s)", len(seen),
             sum(1 for k in {(isl500.user_of(x['path'].split('/')[2]), isl500.clip_key(Path(x['path']).stem)) for x in h5_files} if k in pose_idx),
             len(items), sha[:12])
    guard = common.DiskGuard(DATA, min_free_gb=args.min_free_gb, every=100)
    def work(item, fallback: bool = False) -> dict:
        cid, user, key, h5_path, npys = item
        t0 = time.time()
        out = out_dir / user / (common.safe_name(key) + ".npz")
        rec = {"id": cid, "source": "isl500", "gloss": isl500.gloss_of(key), "user": user, "key": key,
               "h5_file": h5_path, "pd_split": pd.get((user, key)), "path": rel(out), "revision": sha,
               "licence": isl500.LICENCE}
        try:
            if not npys:
                raise LookupError("no MMPose WholeBody file for this clip (hands-only; skipped)")
            with h5py.File(io.BytesIO(http_get(hf_file(isl500.REPO, h5_path, sha))), "r") as hf:
                hands = hf["intermediate"][()]
                fps = float(hf.attrs.get("target_fps") or hf.attrs.get("native_fps") or 30.0)
            if hands.ndim != 4 or hands.shape[1:] != (2, 21, 3):
                raise ValueError(f"unexpected hands shape {hands.shape}")
            wb = None
            for npy in npys:  # several candidates: keep the one with the same frame count
                cand = np.load(io.BytesIO(http_get(hf_file(isl500.REPO, npy, sha))), allow_pickle=False)
                if cand.ndim != 3 or cand.shape[1:] != (133, 3):
                    continue
                if wb is None or abs(len(cand) - len(hands)) < abs(len(wb) - len(hands)):
                    wb, rec["pose_file"] = cand, npy
                if len(cand) == len(hands):
                    break
            if wb is None:
                raise ValueError("MMPose file has an unexpected shape")
            rec["frames_h5"], rec["frames_wb"] = int(len(hands)), int(len(wb))
            if abs(len(wb) - len(hands)) > 2:
                raise ValueError(f"frame counts differ: h5 {len(hands)} vs WholeBody {len(wb)}")
            size = isl500.estimate_size(hands, wb)
            if size is not None:
                rec["size_from"] = "hands"
                sizes.add(user, size)
            elif not fallback:
                return {"id": cid, "deferred": True}  # second pass, once the signer's sizes are all known
            else:
                size, rec["size_from"] = sizes.fallback(user)
            seq = isl500.isl500_to_raw(hands, wb, size, fps)
            stride = common.stride_for(fps, args.target_fps)
            saved = common.save_compact(seq, out, dtype=args.dtype, stride=stride)
            rec.update(common.seq_stats(saved))
            rec.update(common.active_window(saved))
            rec.update({"ok": True, "fps": round(fps / stride, 3), "fps_native": fps, "stride": stride,
                        "dtype": args.dtype, "width": size[0], "height": size[1],
                        "extractor_version": isl500.EXTRACTOR_VERSION})
        except Exception as e:  # noqa: BLE001
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        rec["secs"] = round(time.time() - t0, 2)
        man.add(rec)
        return rec

    deferred_ids: set[str] = set()
    base_work = work

    def first_pass(item) -> dict:
        rec = base_work(item)
        if rec.get("deferred"):
            deferred_ids.add(rec["id"])
        return rec

    stats = run_pool(items, first_pass, args.workers, guard, "isl500")
    if stats["stopped"]:
        return stop_code(stats)  # deferred clips have no manifest row: the next run retries them
    # fallback pass: deferred clips + earlier fallback rows whose size differs from the deterministic one
    by_id = {it[0]: it for it in all_items}
    redo = [by_id[c] for c in sorted(deferred_ids) if c in by_id]
    for r in prior_ok:
        if r.get("size_from") in ("user", "default") and r["id"] in by_id:
            want, _ = sizes.fallback(r["user"])
            if (r.get("width"), r.get("height")) != tuple(want) and r["id"] not in deferred_ids:
                redo.append(by_id[r["id"]])
    if redo:
        log.info("isl500: %d clips without a hand-based size estimate: second pass with the per-signer size %s",
                 len(redo), {u: sizes.fallback(u)[0] for u in sorted({it[1] for it in redo})})
        stats2 = run_pool(redo, lambda it: base_work(it, fallback=True), args.workers, guard, "isl500-fallback")
        return stop_code(stats2)
    return 0


def run_isl500_links(args) -> int:
    """Write data/isl/derived/isl500_links.jsonl: each ISL500 gloss linked onto the vocabulary."""
    from app.isl.paths import PACK_DIR
    from app.isl.vocab.build import VocabIndex

    vocab = VocabIndex.load(PACK_DIR / "vocabulary.jsonl")
    linker = isl500.make_linker(vocab)
    glosses = {r["gloss"] for r in read_manifest(DATA / "manifests" / "isl500.jsonl", only_ok=False) if r.get("gloss")}
    idx = DATA / "raw" / "isl500" / "index_mediapipe.json"
    if idx.exists():
        glosses |= {isl500.gloss_of(isl500.clip_key(Path(x["path"]).stem)) for x in json.loads(idx.read_text())}
    glosses = sorted(glosses)
    rows, kinds = [], {}
    for g in glosses:
        concept, kind, review, sugg = isl500.link_gloss(g, linker)
        rows.append({"gloss": g, "concept_id": concept, "link": kind, "needs_review": review, "see_also": sugg})
        kinds[kind] = kinds.get(kind, 0) + 1
    out = DATA / "derived" / "isl500_links.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    log.info("isl500 links: %d glosses -> %s (%s)", len(rows), rel(out), kinds)
    print(json.dumps({"glosses": len(rows), "links": kinds}))
    return 0


# --------------------------------------------------------------------------------------------- OpenHands raw

def run_openhands_raw(args) -> int:
    import zipfile

    import h5py

    from app.isl.data.pose_sources import zran
    from app.isl.data.remote import RangeFile

    out_dir = DATA / "landmarks" / "openhands_raw"
    raw_dir = DATA / "raw" / "openhands_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    man = Manifest(DATA / "manifests" / "openhands_raw.jsonl")
    members_done_f = DATA / "manifests" / "openhands_raw_members_done.txt"
    members_done = set(members_done_f.read_text().split("\n")) if members_done_f.exists() else set()
    rec = json.loads(http_get(openhands.RECORD))
    zips = sorted((f for f in rec["files"] if f["key"].endswith(".zip")), key=lambda f: f["size"])
    if args.only:
        zips = [z for z in zips if z["key"] in set(args.only)]
    guard = common.DiskGuard(DATA, min_free_gb=args.min_free_gb, budget_dir=out_dir, budget_gb=args.budget_gb, every=25)
    seen_videos: dict[str, set[str]] = {}
    member_of_video: dict[tuple[str, str], str] = {}
    for r in read_manifest(man.path, only_ok=True):
        member_of_video.setdefault((r["channel"], r["video"]), r["member"])
    n_new = 0
    for z in zips:
        channel = z["key"].removesuffix(".zip")
        url = z["links"]["self"]
        cd = RangeFile([url], [z["size"]], block=1 << 20, cache_blocks=4)
        with zipfile.ZipFile(cd) as zf:
            infos = [i for i in zf.infolist() if i.filename.endswith(".h5")]
        for info in sorted(infos, key=lambda i: i.file_size):
            if not openhands.safe_member(info.filename):
                log.warning("skipping unsafe member name %r", info.filename)
                continue
            mkey = f"{z['key']}:{info.filename}"
            if mkey in members_done:
                continue
            try:
                guard.check(force=True)
                mem_check(force=True)
            except common.DiskFull as e:
                log.error("openhands-raw: stopping: %s", e)
                return 3
            except common.MemoryCap as e:
                log.error("openhands-raw: stopping for memory (resumable, exit %d): %s", common.EXIT_MEMORY, e)
                return common.EXIT_MEMORY
            reader = openhands.PrefetchReader(lambda s, e, u=url: openhands.range_fetch(u, s, e), z["size"],
                                              block=8 << 20, ahead=args.workers)
            try:
                start, csize, usize, method = openhands.zip_member_span(reader, info)
                if method != 8:
                    raise OSError(f"unsupported zip method {method}")
                # access points persisted next to the dataset index: a restart seeks to the point before its
                # next chunk instead of re-inflating the member from byte 0 (NIOS: 11 GB of downloads)
                pts = zran.PointFile(raw_dir / (common.safe_name(mkey) + ".points"), mkey, csize, usize)
                stream = openhands.SeekableInflate(reader, start, csize, usize, points_file=pts)
                if stream.points_loaded:
                    log.info("openhands-raw: %s: %d restore points loaded (up to %.2f GB of %.2f GB inflated)", mkey,
                             stream.points_loaded, stream.persisted_out / 1e9, usize / 1e9)
                elif not stream.native:
                    log.warning("openhands-raw: libz not available: restore points are not persisted")
                idx_file = raw_dir / (common.safe_name(mkey) + ".index.json")
                if idx_file.exists():
                    entries = [tuple(e) for e in json.loads(idx_file.read_text())]
                else:
                    t0 = time.time()
                    log.info("openhands-raw: %s: listing datasets (%.2f GB compressed, %.2f GB raw)", mkey,
                             csize / 1e9, usize / 1e9)
                    hf = h5py.File(stream, "r")
                    try:
                        entries = openhands.list_datasets(hf)
                    finally:
                        hf.close()
                    idx_file.write_text(json.dumps(entries))
                    log.info("openhands-raw: %s: %d datasets listed in %.0f s (%.2f GB fetched)", mkey, len(entries),
                             time.time() - t0, reader.fetched_bytes / 1e9)
                n_new += _openhands_member(args, stream, entries, channel, mkey, info.filename, man, out_dir, guard,
                                           member_of_video)
                with members_done_f.open("a") as fh:
                    fh.write(mkey + "\n")
                members_done.add(mkey)
            except common.DiskFull as e:
                log.error("openhands-raw: stopping: %s", e)
                return 3
            except common.MemoryCap as e:
                log.error("openhands-raw: stopping for memory (resumable, exit %d): %s", common.EXIT_MEMORY, e)
                return common.EXIT_MEMORY
            finally:
                reader.close()
            if args.limit and n_new >= args.limit:
                return 0
    log.info("openhands-raw: all members done (%d new chunks this run)", n_new)
    return 0


def _openhands_member(args, stream, entries, channel, mkey, member, man, out_dir, guard, member_of_video) -> int:
    hdr_cache: dict[int, dict] = {}

    def header(addr: int) -> dict:
        if addr not in hdr_cache:
            hdr_cache[addr] = openhands.parse_v1_header(stream.read_range(addr, 4096))
        return hdr_cache[addr]

    n = 0
    t0 = time.time()
    stride = common.stride_for(openhands.ASSUMED_FPS, args.target_fps)

    def relief() -> None:  # memory guard: page cache, metadata pages (pass 2 reads headers directly), read-ahead
        stream.pages.clear()
        stream.meta.clear()
        stream.meta_bytes = 0
        stream.meta_cap = 0
        stream.r.ahead = max(2, stream.r.ahead // 2)
        hdr_cache.clear()

    if MEM is not None:
        MEM.relief = [relief]
    for name, kp_e, vis_e in openhands.iter_pairs(entries):
        cid = f"openhands_raw/{channel}/{name}"
        video = openhands.video_key(name)
        if cid in man.done or kp_e is None:
            continue
        prev = member_of_video.get((channel, video))
        if prev is not None and prev != member:  # same video already ingested from another file of this channel
            continue
        guard.check()
        mem_check()
        out = out_dir / channel / (common.safe_name(name) + ".npz")
        rec = {"id": cid, "source": "openhands_raw", "gloss": "", "channel": channel, "video": video, "member": member,
               "path": rel(out), "licence": openhands.LICENCE}
        t1 = time.time()
        try:
            kp = openhands.read_contiguous(stream, header(kp_e[0]))
            vis = openhands.read_contiguous(stream, header(vis_e[0])) if vis_e is not None else None
            asp = common.hand_aspect(openhands.hands_nan(kp[:: max(1, len(kp) // 1500)]))
            aspect = openhands.snap_aspect(asp[0] if asp else None) or 16 / 9
            seq = openhands.openhands_to_raw(kp, vis, aspect=aspect)
            del kp, vis
            saved = common.save_compact(seq, out, dtype=args.dtype, stride=stride)
            rec.update(common.seq_stats(saved))
            rec.update({"ok": True, "fps": round(openhands.ASSUMED_FPS / stride, 3), "fps_native": openhands.ASSUMED_FPS,
                        "fps_assumed": True, "n_frames_native": len(seq), "stride": stride, "dtype": args.dtype,
                        "aspect_est": asp[0] if asp else None, "aspect": round(aspect, 4), "width": saved.width,
                        "height": saved.height, "extractor_version": openhands.EXTRACTOR_VERSION})
            member_of_video.setdefault((channel, video), member)
        except openhands.HeaderError as e:
            rec.update({"ok": False, "error": f"HeaderError: {e}"[:300]})
        except (OSError, EOFError, ValueError) as e:
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        rec["secs"] = round(time.time() - t1, 2)
        man.add(rec)
        n += 1
        if n == 1:  # how far the restart had to inflate / download to reach its first new chunk
            log.info("openhands-raw: %s: first new chunk at %.2f GB (inflated): %.2f GB downloaded, %d restores, "
                     "%.0f s", mkey, kp_e[0] / 1e9, stream.r.fetched_bytes / 1e9, stream.restores, time.time() - t0)
        if n % 50 == 0:
            log.info("openhands-raw: %s: %d chunks this run, %.1f MB/s fetched, free %.1f GB", mkey, n,
                     stream.r.fetched_bytes / 1e6 / max(1.0, time.time() - t0), common.free_gb(DATA))
    return n


# --------------------------------------------------------------------------------------------- OpenHands fingerspelling

def run_openhands_fs(args) -> int:
    import zipfile

    from app.isl.data.ingest_bridge import ingest_module

    raw_dir = DATA / "raw" / "openhands_fs"
    out_dir = DATA / "landmarks" / "openhands_fs"
    man = Manifest(DATA / "manifests" / "openhands_fs.jsonl")
    rec = json.loads(http_get(fingerspell.RECORD))
    f = next(x for x in rec["files"] if x["key"] == fingerspell.ZIP_NAME)
    zpath = raw_dir / fingerspell.ZIP_NAME
    if not zpath.exists() or zpath.stat().st_size != f["size"]:
        zpath.unlink(missing_ok=True)
        fetch_small(f["links"]["self"], zpath)
    ing = ingest_module()
    tmp_root = DATA / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if not openhands.safe_member(info.filename):
                log.warning("skipping unsafe member %r", info.filename)
                continue
            meta = fingerspell.parse_member(info.filename)
            if meta is None:
                continue
            cid = fingerspell.clip_id(meta)
            if cid in man.done:
                continue
            out = out_dir / meta["split"] / meta["gloss"] / (common.safe_name(meta["n"]) + ".npz")
            fd, name = tempfile.mkstemp(prefix="ohfs_", suffix=".mp4", dir=tmp_root)
            try:
                with os.fdopen(fd, "wb") as fh, zf.open(info) as src:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        fh.write(b)
                info_d = ing.work_item("full", name, str(out))
            finally:
                Path(name).unlink(missing_ok=True)
            r = {"id": cid, "source": "openhands_fs", "gloss": meta["gloss"], "symbol": meta["symbol"],
                 "kind": meta["kind"], "split": meta["split"], "member": info.filename, "path": rel(out),
                 "licence": fingerspell.LICENCE, "extractor_version": "holistic-v1"}
            r.update(info_d)
            man.add(r)
            n += 1
            if args.limit and n >= args.limit:
                break
    log.info("openhands-fs: %d clips processed this run (total ok %d)", n, len(man.done))
    return 0


# --------------------------------------------------------------------------------------------- status

def run_status(args) -> int:
    out = {"free_gb": round(common.free_gb(DATA), 1)}
    for src in ("glosl", "isl500", "openhands_raw", "openhands_fs"):
        recs = read_manifest(DATA / "manifests" / f"{src}.jsonl", only_ok=False)
        ok = [r for r in recs if r.get("ok")]
        d = DATA / "landmarks" / src
        size = sum(p.stat().st_blocks * 512 for p in d.rglob("*.npz")) if d.exists() else 0
        out[src] = {"records": len(recs), "ok": len(ok), "failed": len(recs) - len(ok),
                    "frames": int(sum(r.get("n_frames") or 0 for r in ok)), "disk_gb": round(size / 1e9, 2)}
    print(json.dumps(out, indent=1))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["glosl", "isl500", "isl500-links", "openhands-raw", "openhands-fs", "status"])
    ap.add_argument("--workers", type=int, default=4, help="download threads (network-bound)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--target-fps", type=float, default=common.DEFAULT_TARGET_FPS, help="0 keeps every frame")
    ap.add_argument("--min-free-gb", type=float, default=common.MIN_FREE_GB)
    ap.add_argument("--budget-gb", type=float, default=12.0, help="openhands-raw: max size of its landmark folder")
    ap.add_argument("--only", nargs="+", help="openhands-raw: zip names to process (e.g. Sign_Library.zip)")
    ap.add_argument("--max-rss-gb", type=float, default=common.DEFAULT_MAX_RSS_GB,
                    help="soft memory cap of this process: drop caches, then stop cleanly (exit 5); 0 = off")
    ap.add_argument("--respawn-on-memory", type=int, default=3,
                    help="after a memory stop, restart this job in a fresh process up to N times")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if args.source not in ("status", "isl500-links"):
        import fcntl

        # a process scan as well as the lock: the first glosl / isl500 jobs were started before locks existed
        others = common.find_running(Path(__file__).name, args.source)
        if others:
            log.error("another %s job is running (pid %s, found by process scan); exiting", args.source,
                      ", ".join(map(str, others)))
            sys.exit(4)

        lock_path = DATA / "manifests" / f"{args.source.replace('-', '_')}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log.error("another %s job is running (lock %s); exiting", args.source, rel(lock_path))
            sys.exit(4)
        lock.write(str(os.getpid()))
        lock.flush()
    global MEM
    if args.source not in ("status", "isl500-links") and args.max_rss_gb > 0:
        MEM = common.MemoryGuard(max_gb=args.max_rss_gb, every=20, log=log)
    fn = {"glosl": run_glosl, "isl500": run_isl500, "isl500-links": run_isl500_links,
          "openhands-raw": run_openhands_raw, "openhands-fs": run_openhands_fs, "status": run_status}[args.source]
    rc = fn(args)
    if rc == common.EXIT_MEMORY:
        n = int(os.environ.get("ISL_MEM_RESPAWNS", "0"))
        if n < args.respawn_on_memory:
            log.warning("%s: stopped for memory (RSS %.2f GB); restarting in a fresh process (%d of %d)",
                        args.source, common.rss_gb(), n + 1, args.respawn_on_memory)
            logging.shutdown()
            time.sleep(30)
            os.environ["ISL_MEM_RESPAWNS"] = str(n + 1)
            os.execv(sys.executable, [sys.executable, *sys.argv])  # the lock fd is close-on-exec
        log.error("%s: stopped for memory %d times; giving up (resumable: start it again)", args.source, n)
    sys.exit(rc)


if __name__ == "__main__":
    main()
