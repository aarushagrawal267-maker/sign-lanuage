"""Download Indian Sign Language video sources and keep only MediaPipe landmarks.

Videos are streamed one at a time (or one INCLUDE zip at a time), converted to
RawSequence .npz files with the same extractor the app uses, and deleted, so the
whole corpus never has to fit on disk. The job is resumable: finished items are
recorded in a manifest (JSON lines) and skipped on the next run.

Sources:
  include  - INCLUDE (AI4Bharat), 263 ISL words, ~4.3k videos, CC BY 4.0
             https://zenodo.org/records/4010759
  islrtc   - ISLRTC Indian Sign Language Dictionary (data.gov.in, GODL-India),
             ~13.6k words/letters/numbers, one video each; HEVC re-encode on
             Hugging Face: silentone0725/Indian_Sign_Language_Data.gov_Rencoded
             Only the opening sign is kept (the video then cuts to an
             "Explanation of ..." card and a signed explanation).

Usage:
  python scripts/ingest_isl.py include --workers 2
  python scripts/ingest_isl.py islrtc --workers 2 [--limit N]
  # re-extract specific long ISLRTC entries in full (no title-card cut) into their own folder/manifest:
  python scripts/ingest_isl.py islrtc --only A/Alphabet.mp4 Numbers/Numbers_(0-1_Crore).mp4 \
      --full --max-s 120 --out-subdir islrtc_full

  # re-extract the clips the old title-card detector stopped (card / truncated) with card-v2, own folder + manifest:
  python scripts/ingest_isl.py islrtc-recut --workers 1 [--follow]

Title-card detector: ``card-v2`` (:class:`CardDetector`) since 25 Nov 2026; the jobs started before that keep the
old rule (left strip std < 6, which also fires on plain grey backgrounds). Records carry ``card_detector``.

Options added after the first run (backward compatible: the default behaviour, ids and manifest format are
unchanged): --only, --max-s, --out-subdir, --full. INCLUDE videos stored one level deeper in an Extra/
folder now get the right gloss/category fields; their ids keep the original formula so a restarted job
does not process them twice (the ISL adapters rebuild canonical ids from the member path).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("GLOG_minloglevel", "2")

DATA = ROOT / "data" / "isl"
INCLUDE_RECORD = "https://zenodo.org/api/records/4010759"
HF_REPO = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

log = logging.getLogger("ingest_isl")

# --------------------------------------------------------------------------- worker side

_EX = None


def _extractor():
    global _EX
    if _EX is None:
        from app.landmarks.extractor import HolisticExtractor

        _EX = HolisticExtractor(max_side=640)
    return _EX


def _frames(path: str, step_fps: float, max_s: float):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 240:
        fps = 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / step_fps)))
    return cap, fps, step, w, h


def extract_full(path: str, out: str, step_fps: float = 15.0, max_s: float = 20.0) -> dict:
    """Whole clip (INCLUDE clips are one sign each)."""
    cap, fps, step, w, h = _frames(path, step_fps, max_s)

    def gen():
        i = 0
        try:
            while True:
                ok, f = cap.read()
                if not ok or i / fps > max_s:
                    break
                if i % step == 0:
                    yield f, i * 1000.0 / fps
                i += 1
        finally:
            cap.release()

    seq = _extractor().extract_frames(gen(), w, h)
    seq.save(Path(out))
    pr = seq.presence()
    return {"n_frames": len(seq), "fps": fps / step, "hands_frac": float((pr["left"] | pr["right"]).mean()) if len(seq) else 0.0}


def extract_opening_sign(path: str, out: str, step_fps: float = 12.0, max_s: float = 10.0) -> dict:
    """ISLRTC dictionary clip: keep frames until the person disappears (title card)."""
    cap, fps, step, w, h = _frames(path, step_fps, max_s)
    stop = {"reason": "max_s"}

    def gen():
        i, seen_hands_gap, k = 0, 0, 0
        try:
            while True:
                ok, f = cap.read()
                if not ok:
                    stop["reason"] = "eof"
                    break
                if i / fps > max_s + INTRO_MAX_S:  # _cut_at_card applies max_s after an opening intro
                    break
                if i % step == 0:
                    yield f, i * 1000.0 / fps
                    k += 1
                i += 1
        finally:
            cap.release()

    ex = _extractor()
    # The extractor consumes a generator and we cannot peek at its results mid-stream, so the
    # title card is detected cheaply on pixels: card frames are near-uniform with no person.
    seq = ex.extract_frames(_cut_at_card(gen(), stop, max_s=max_s), w, h)
    pr = seq.presence()
    pose = pr["pose"]
    # Trim: drop trailing frames without pose, and keep up to the first long pose gap after the start.
    n = len(seq)
    end = n
    run = 0
    started = False
    for t in range(n):
        if pose[t]:
            started = True
            run = 0
        elif started:
            run += 1
            if run >= 4:
                end = t - run + 1
                stop["reason"] = stop.get("reason", "") + "+posegap"
                break
    hands = pr["left"] | pr["right"]
    while end > 0 and not hands[end - 1]:
        end -= 1
    seq = seq.slice(0, max(end, 0)) if end < n else seq
    seq.save(Path(out))
    h2 = seq.presence()
    return {
        "n_frames": len(seq), "fps": fps / step, "stop": stop["reason"], "card_detector": CARD_DETECTOR,
        "intro_ms": round(float(stop.get("intro_ms", 0.0)), 1),
        "hands_frac": float((h2["left"] | h2["right"]).mean()) if len(seq) else 0.0,
        "duration_s": float(len(seq) * step / fps),
    }


CARD_DETECTOR = "card-v2"
CARD_DIFF = 40.0        # a pixel differs from the left-strip colour by more than this (max over B, G, R)
CARD_EMPTY = 0.10       # centre-region share of such pixels below which nobody stands there (signer: 0.25-0.8)
CARD_PLAIN_STD = 20.0   # an empty frame must also have a plain left strip (black, blank, title card)
CARD_SHIFT = 40.0       # left-strip colour moved this far from the opening background: another screen
CARD_SHIFT_STD = 25.0   # ... and that strip is fairly plain
CARD_RUN = 3            # consecutive card frames that end the sign
CARD_MIN_MS = 800.0
CARD_PERSON = 0.25      # centre share above which a frame shows the signer (intro skipping)
INTRO_MAX_S = 15.0      # longest opening intro skipped before the sign (the "Academic Terms" intro: ~6.5 s)


def card_features(frame) -> tuple[float, "object", float]:
    """(grey std of the left strip, its mean B/G/R colour, share of centre pixels that differ from that colour)
    on a 160x90 thumbnail. The left 60 columns are where ISLRTC signers never stand; the centre (columns 50-130,
    rows 8-90) is where they always stand."""
    import cv2
    import numpy as np

    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    left = small[:, :60]
    lstd = float(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).std())
    lmean = left.reshape(-1, 3).mean(axis=0)
    centre = small[8:, 50:130]
    fg = float((np.abs(centre - lmean).max(axis=2) > CARD_DIFF).mean())
    return lstd, lmean, fg


class CardDetector:
    """Title-card / end-of-sign detector for ISLRTC dictionary videos (``card-v2``).

    trim-1/2 used "the left strip is near-uniform" (grey std < 6), which also holds for the plain grey
    background most signers stand in front of: 53 of 71 ``card`` stops were mid-sign (docs/DATASETS_ISL.md 7.3).
    A frame is now card-like only when the picture really changed:

    * **empty**: nobody in the centre (< 10 % of centre pixels differ from the background colour; a signer gives
      25-80 %) and a plain left strip (fade to black, blank or plain title card); or
    * **other screen**: the left strip's colour moved > 40 levels from the opening background (median over the
      frames up to 1 s that show a person) and is fairly plain (the blue "Explanation" card, a coloured wipe).

    ``CARD_RUN`` consecutive card-like frames after ``CARD_MIN_MS`` end the sign. Checked on the misfired labelled
    clips (Driver (Sign 2), Dominance, Emission, Break a Leg, Bribe, Energy Infrastructure: no stop any more) and the
    correct ones (Cardiovascular Medicine: fade to black at 5.6 s; Definition: "Explanation" card at 3.2 s)."""

    def __init__(self):
        self.run = 0
        self.ref_samples: list = []
        self.ref = None

    def is_card(self, frame, t_ms: float) -> bool:
        import numpy as np

        lstd, lmean, fg = card_features(frame)
        if self.ref is None:
            if fg >= 2 * CARD_EMPTY and t_ms <= 1000.0:
                self.ref_samples.append(lmean)
            if t_ms > 1000.0 or len(self.ref_samples) >= 12:
                self.ref = np.median(np.stack(self.ref_samples), axis=0) if self.ref_samples else lmean
        empty = fg < CARD_EMPTY and lstd < CARD_PLAIN_STD
        shifted = (self.ref is not None and lstd < CARD_SHIFT_STD
                   and float(np.abs(lmean - self.ref).max()) > CARD_SHIFT)
        return empty or shifted

    def update(self, frame, t_ms: float) -> bool:
        """True once the sign has ended (``CARD_RUN`` card-like frames in a row after ``CARD_MIN_MS``)."""
        if self.is_card(frame, t_ms):
            self.run += 1
            return self.run >= CARD_RUN and t_ms > CARD_MIN_MS
        self.run = 0
        return False


def _cut_at_card(frames, stop, max_s: float | None = None):
    """Stop the stream at the end-of-sign screen (title card, fade, wipe): see :class:`CardDetector`.

    Opening intro: when the video starts on a card-like frame (an empty or animated title screen, e.g. the
    "Academic Terms in Indian Sign Language" series, about 6.5 s), frames are skipped until the signer appears
    (centre share >= ``CARD_PERSON`` and not card-like), at most ``INTRO_MAX_S``; the kept frames' timestamps then
    start at 0 and ``stop["intro_ms"]`` records the skipped time. Both card rules used to stop such videos at
    0.88 s with no sign at all. ``max_s`` (seconds, optional) caps the kept part after the intro."""
    det = CardDetector()
    t0 = None  # time of the first kept frame
    intro = False
    for i, (f, t) in enumerate(frames):
        if t0 is None:
            if i == 0:
                lstd, _lm, fg = card_features(f)
                intro = fg < CARD_EMPTY and lstd < CARD_PLAIN_STD
            if intro:
                lstd, _lm, fg = card_features(f)
                if fg < CARD_PERSON or (fg < CARD_EMPTY and lstd < CARD_PLAIN_STD):
                    if t > INTRO_MAX_S * 1000.0:
                        stop["reason"] = "card"  # no signer after a long intro: give up as before
                        return
                    continue
            t0 = t
            stop["intro_ms"] = t0
        rel = t - t0
        if max_s is not None and rel > max_s * 1000.0:
            return  # stop["reason"] stays "max_s"
        if det.update(f, rel):
            stop["reason"] = "card"
            return
        yield f, rel


def work_item(kind: str, src: str, out: str, max_s: float | None = None) -> dict:
    t0 = time.time()
    try:
        if kind == "full":
            info = extract_full(src, out) if max_s is None else extract_full(src, out, max_s=max_s)
        else:
            info = extract_opening_sign(src, out) if max_s is None else extract_opening_sign(src, out, max_s=max_s)
        info["ok"] = True
    except Exception as e:  # keep the job going; failures are listed in the manifest
        info = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    info["secs"] = round(time.time() - t0, 1)
    return info


# --------------------------------------------------------------------------- driver side


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok"):
                    self.done.add(r["id"])
        self.lock = threading.Lock()

    def add(self, rec: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec.get("ok"):
            self.done.add(rec["id"])


def _download(url: str, dest: Path, retries: int = 5) -> None:
    for i in range(retries):
        try:
            r = subprocess.run(
                ["curl", "-sSfL", "--retry", "3", "--connect-timeout", "30", "-o", str(dest), url],
                capture_output=True, text=True, timeout=7200,
            )
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                return
            err = r.stderr.strip()
        except subprocess.TimeoutExpired:
            err = "timeout"
        log.warning("download failed (%s) try %d: %s", url, i + 1, err)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"download failed: {url}")


def _gloss_from_name(name: str) -> str:
    stem = Path(name).stem.replace("_", " ").strip()
    return " ".join(stem.split())


def run_include(args) -> None:
    out_dir = DATA / "landmarks" / "include"
    out_dir.mkdir(parents=True, exist_ok=True)
    man = Manifest(DATA / "manifests" / "include.jsonl")
    rec = json.loads(urllib.request.urlopen(INCLUDE_RECORD, timeout=60).read())
    zips = sorted((f for f in rec["files"] if f["key"].endswith(".zip")), key=lambda f: f["key"])
    done_zips = DATA / "manifests" / "include_zips_done.txt"
    finished = set(done_zips.read_text().split()) if done_zips.exists() else set()
    tmp_root = Path(args.tmp)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for z in zips:
            if z["key"] in finished:
                continue
            zdir = Path(tempfile.mkdtemp(prefix="include_", dir=tmp_root))
            try:
                zpath = zdir / z["key"]
                log.info("downloading %s (%.2f GB)", z["key"], z["size"] / 1e9)
                _download(z["links"]["self"], zpath)
                with zipfile.ZipFile(zpath) as zf:
                    members = [m for m in zf.namelist() if Path(m).suffix.lower() in VIDEO_EXT and "__MACOSX" not in m]
                    zf.extractall(zdir / "x", members)
                zpath.unlink()
                futs = []
                for m in members:
                    parts = Path(m).parts
                    # <Category>/<NN. Word>/<file>; the id keeps this original formula (resume compatibility)
                    word_dir = parts[-2] if len(parts) >= 2 else ""
                    category = parts[-3] if len(parts) >= 3 else z["key"].split("_")[0]
                    vid = f"include/{category}/{word_dir}/{Path(m).stem}".replace(" ", "_")
                    # gloss/category ignore an "Extra" folder level (<Category>/<NN. Word>/Extra/<file>)
                    dirs = [p for p in parts[:-1] if p.lower() != "extra"]
                    if len(dirs) >= 2:
                        category, word_dir = dirs[-2], dirs[-1]
                    gloss = word_dir.split(".", 1)[1].strip() if "." in word_dir else word_dir
                    if vid in man.done:
                        continue
                    out = out_dir / (vid.split("/", 1)[1].replace("/", "__") + ".npz")
                    meta = {"id": vid, "source": "include", "gloss": gloss, "category": category,
                            "zip": z["key"], "member": m, "path": str(out.relative_to(ROOT))}
                    futs.append((meta, pool.submit(work_item, "full", str(zdir / "x" / m), str(out))))
                for meta, f in futs:
                    meta.update(f.result())
                    man.add(meta)
                with done_zips.open("a") as fh:
                    fh.write(z["key"] + "\n")
                log.info("finished %s: %d videos (total ok %d)", z["key"], len(futs), len(man.done))
            finally:
                shutil.rmtree(zdir, ignore_errors=True)


def _hf_list(folder: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/{urllib.parse.quote(folder)}"
    items, next_url = [], url
    while next_url:
        req = urllib.request.Request(next_url)
        with urllib.request.urlopen(req, timeout=60) as r:
            items += json.loads(r.read())
            link = r.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1: part.find(">")]
    return items


def run_islrtc(args) -> None:
    sub = getattr(args, "out_subdir", None) or "islrtc"
    out_dir = DATA / "landmarks" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    man = Manifest(DATA / "manifests" / f"{sub}.jsonl")
    kind = "full" if getattr(args, "full", False) else "opening"
    max_s = getattr(args, "max_s", None)
    index_path = DATA / "manifests" / "islrtc_index.json"
    if index_path.exists():
        files = json.loads(index_path.read_text())
    else:
        files = []
        for folder in [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "Numbers"]:
            for it in _hf_list(folder):
                if it.get("type") == "file" and Path(it["path"]).suffix.lower() in VIDEO_EXT:
                    files.append({"path": it["path"], "size": it.get("size", 0)})
            log.info("indexed %s: %d files so far", folder, len(files))
        index_path.write_text(json.dumps(files))
    todo = [f for f in files if f"islrtc/{f['path']}" not in man.done]
    only = getattr(args, "only", None)
    if only:
        want = {o.removeprefix("islrtc/") for o in only}
        todo = [f for f in todo if f["path"] in want]
        missing = want - {f["path"] for f in files}
        if missing:
            log.warning("--only: %d ids not in the ISLRTC index: %s", len(missing), sorted(missing)[:10])
    if args.limit:
        todo = todo[: args.limit]
    log.info("islrtc: %d of %d videos to process", len(todo), len(files))
    tmp_root = Path(args.tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)
    if sub == "islrtc":  # never delete the temp files of a concurrently running default job otherwise
        for stale in tmp_root.glob("islrtc_*"):
            if stale.is_file():  # tmp/islrtc_recut/ is the re-extraction job's folder, not a stale download
                stale.unlink(missing_ok=True)

    # Prefetch downloads in a thread so the CPU workers never wait on the network.
    import queue

    q: queue.Queue = queue.Queue(maxsize=args.workers * 3)

    def downloader():
        for f in todo:
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{urllib.parse.quote(f['path'])}"
            fd, name = tempfile.mkstemp(prefix="islrtc_", suffix=Path(f["path"]).suffix, dir=tmp_root)
            os.close(fd)  # curl writes by path; an open fd would keep deleted files' disk space allocated
            dest = Path(name)
            try:
                _download(url, dest)
                q.put((f, dest))
            except Exception as e:  # noqa: BLE001
                dest.unlink(missing_ok=True)
                man.add({"id": f"islrtc/{f['path']}", "source": "islrtc", "ok": False, "error": str(e)[:300]})
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()
    pending = []
    n = 0
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        while True:
            item = q.get()
            if item is None:
                break
            f, dest = item
            rel = f["path"]
            folder = Path(rel).parts[0]
            out = out_dir / (rel.replace("/", "__").rsplit(".", 1)[0] + ".npz")
            meta = {"id": f"islrtc/{rel}", "source": "islrtc", "gloss": _gloss_from_name(rel),
                    "category": "number" if folder == "Numbers" else "dictionary",
                    "path": str(out.relative_to(ROOT))}
            if kind == "full":
                meta["stop"] = "full"
            meta.update(getattr(args, "extra_meta", None) or {})
            pending.append((meta, dest, pool.submit(work_item, kind, str(dest), str(out), max_s)))
            while len(pending) >= args.workers * 2 or (pending and pending[0][2].done()):
                meta, d, fut = pending.pop(0)
                meta.update(fut.result())
                d.unlink(missing_ok=True)
                man.add(meta)
                n += 1
                if n % 100 == 0:
                    log.info("islrtc: %d processed this run (total ok %d)", n, len(man.done))
        for meta, d, fut in pending:
            meta.update(fut.result())
            d.unlink(missing_ok=True)
            man.add(meta)
    log.info("islrtc done: total ok %d", len(man.done))


RECUT_SUBDIR = "islrtc_recut"


def recut_targets(main_manifest: Path, trim_sidecar: Path | None, done: set[str]) -> list[str]:
    """ISLRTC paths to re-extract with the card-v2 detector: every clip the running job (old detector, left-strip
    std < 6) stopped at a ``card``, plus any clip the trimmer marked ``truncated``, minus those already in the
    recut manifest. The running job's manifest and files are only read."""
    from app.isl.data.manifest import iter_jsonl

    ids: dict[str, None] = {}
    for r in iter_jsonl(main_manifest):
        if r.get("ok") and str(r.get("stop", "")).startswith("card") and r.get("card_detector") != CARD_DETECTOR:
            ids[r["id"]] = None
    if trim_sidecar is not None:
        for r in iter_jsonl(trim_sidecar):
            if r.get("method") == "truncated" and r.get("id"):
                ids[r["id"]] = None
    return [i.removeprefix("islrtc/") for i in ids if i not in done]


def run_islrtc_recut(args) -> None:
    """Re-extract ISLRTC clips that the old title-card detector cut short into ``landmarks/islrtc_recut/`` and
    ``manifests/islrtc_recut.jsonl`` (same record format as islrtc.jsonl plus ``recut_of``/``card_detector``; a
    recut record supersedes the islrtc.jsonl record with the same id). With ``--follow`` it re-scans every
    ``--poll-min`` minutes while ``ingest_isl.py islrtc`` still runs (that job keeps the old detector), and exits
    once the job has ended and nothing is left."""
    from app.isl.data.manifest import read_manifest
    from app.isl.data.pose_sources.common import find_running

    main_man = DATA / "manifests" / "islrtc.jsonl"
    trim = DATA / "derived" / "islrtc_trim.jsonl"
    recut_man = DATA / "manifests" / f"{RECUT_SUBDIR}.jsonl"
    tmp = Path(args.tmp) / "islrtc_recut"  # not "islrtc_*" directly under tmp/: the main job unlinks those
    tmp.mkdir(parents=True, exist_ok=True)
    final = False
    while True:
        done = {r["id"] for r in read_manifest(recut_man)}
        targets = recut_targets(main_man, trim if trim.exists() else None, done)
        if args.limit:
            targets = targets[: args.limit]
        main_running = bool(find_running("ingest_isl.py", "islrtc"))
        log.info("islrtc-recut: %d clips to re-extract (%d done; main islrtc job %s)", len(targets), len(done),
                 "running" if main_running else "not running")
        if targets:
            ns = argparse.Namespace(only=targets, full=False, max_s=args.max_s, out_subdir=RECUT_SUBDIR, limit=0,
                                    workers=max(1, args.workers), tmp=str(tmp),
                                    extra_meta={"recut_of": "islrtc", "recut_reason": "card-v1 false stops"})
            run_islrtc(ns)
        if not args.follow or (not main_running and final):
            break
        if not main_running:
            final = True  # one more scan for clips written just before the job ended, then stop
            continue
        time.sleep(args.poll_min * 60)
    log.info("islrtc-recut: finished (%d records)", len(read_manifest(recut_man)))


def _single_instance(name: str, argv_source: str) -> object:
    """flock on manifests/<name>.lock plus a process scan; exits with status 4 when another copy runs."""
    import fcntl

    from app.isl.data.pose_sources.common import find_running

    others = find_running(Path(__file__).name, argv_source)
    if others:
        log.error("another %s job is running (pid %s); exiting", argv_source, others)
        sys.exit(4)
    lock_path = DATA / "manifests" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.error("another %s job holds %s; exiting", argv_source, lock_path)
        sys.exit(4)
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["include", "islrtc", "islrtc-recut"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tmp", default=str(DATA / "tmp"))
    ap.add_argument("--only", nargs="+", help="ISLRTC: process only these paths (e.g. A/Alphabet.mp4)")
    ap.add_argument("--max-s", type=float, default=None, help="seconds kept per video (default 10 / 20)")
    ap.add_argument("--out-subdir", default=None,
                    help="ISLRTC: landmarks/<name>/ and manifests/<name>.jsonl instead of islrtc")
    ap.add_argument("--full", action="store_true", help="ISLRTC: keep the whole clip (no title-card cut)")
    ap.add_argument("--follow", action="store_true",
                    help="islrtc-recut: keep re-scanning while the islrtc job runs (it still uses the old detector)")
    ap.add_argument("--poll-min", type=float, default=30.0, help="islrtc-recut --follow: minutes between scans")
    args = ap.parse_args()
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.source == "islrtc-recut":
        _lock = _single_instance(RECUT_SUBDIR, "islrtc-recut")  # noqa: F841 (held until exit)
        run_islrtc_recut(args)
        return
    (run_include if args.source == "include" else run_islrtc)(args)


if __name__ == "__main__":
    main()
