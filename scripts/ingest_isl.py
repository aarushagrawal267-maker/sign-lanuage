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
                if i / fps > max_s:
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
    seq = ex.extract_frames(_cut_at_card(gen(), stop), w, h)
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
        "n_frames": len(seq), "fps": fps / step, "stop": stop["reason"],
        "hands_frac": float((h2["left"] | h2["right"]).mean()) if len(seq) else 0.0,
        "duration_s": float(len(seq) * step / fps),
    }


def _cut_at_card(frames, stop):
    """Stop the stream at the 'Explanation of ...' title card.

    The card is a light, almost uniform frame with only centred text: very low pixel variance in
    the left half where the signer normally stands. A run of 3 such frames ends the sign.
    """
    import cv2

    run = 0
    for f, t in frames:
        small = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA)
        left = cv2.cvtColor(small[:, :60], cv2.COLOR_BGR2GRAY)
        if float(left.std()) < 6.0:
            run += 1
            if run >= 3 and t > 800:
                stop["reason"] = "card"
                return
        else:
            run = 0
        yield f, t


def work_item(kind: str, src: str, out: str) -> dict:
    t0 = time.time()
    try:
        info = extract_full(src, out) if kind == "full" else extract_opening_sign(src, out)
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
                    # <Category>/<NN. Word>/<file>
                    word_dir = parts[-2] if len(parts) >= 2 else ""
                    category = parts[-3] if len(parts) >= 3 else z["key"].split("_")[0]
                    gloss = word_dir.split(".", 1)[1].strip() if "." in word_dir else word_dir
                    vid = f"include/{category}/{word_dir}/{Path(m).stem}".replace(" ", "_")
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
    out_dir = DATA / "landmarks" / "islrtc"
    out_dir.mkdir(parents=True, exist_ok=True)
    man = Manifest(DATA / "manifests" / "islrtc.jsonl")
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
    if args.limit:
        todo = todo[: args.limit]
    log.info("islrtc: %d of %d videos to process", len(todo), len(files))
    tmp_root = Path(args.tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)
    for stale in tmp_root.glob("islrtc_*"):
        stale.unlink(missing_ok=True)

    # Prefetch downloads in a thread so the CPU workers never wait on the network.
    import queue

    q: queue.Queue = queue.Queue(maxsize=args.workers * 3)

    def downloader():
        for f in todo:
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{urllib.parse.quote(f['path'])}"
            dest = Path(tempfile.mkstemp(prefix="islrtc_", suffix=Path(f["path"]).suffix, dir=tmp_root)[1])
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
            pending.append((meta, dest, pool.submit(work_item, "opening", str(dest), str(out))))
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["include", "islrtc"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tmp", default=str(DATA / "tmp"))
    args = ap.parse_args()
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (run_include if args.source == "include" else run_islrtc)(args)


if __name__ == "__main__":
    main()
