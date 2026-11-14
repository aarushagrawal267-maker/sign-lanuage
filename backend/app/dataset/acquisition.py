"""WLASL100 video acquisition and validation.

* YouTube links are fetched once per YouTube id with ``yt-dlp`` (several WLASL
  instances can point at different frame ranges of one YouTube video).
* ASLPro links are Flash (.swf) files; they are converted to MP4 with ffmpeg.
* All other hosts are plain HTTP downloads.

Every attempt is logged in ``download_attempts``. Failures are recorded and the
video stays ``failed`` - nothing is substituted from other sources. Videos are
stored under ``data/videos`` (git-ignored) and are never redistributed.
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import requests
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import DownloadAttempt, Video, utcnow
from app.db.session import session_scope

USER_AGENT = "Mozilla/5.0 (research downloader for WLASL100; non-commercial)"
YT_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/v/)([A-Za-z0-9_-]{11})")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v")


def classify_url(url: str) -> str:
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "aslpro" in u:
        return "aslpro"
    return "http"


def youtube_id(url: str) -> str | None:
    m = YT_RE.search(url)
    return m.group(1) if m else None


def raw_path_for(video: Video) -> Path:
    """Where the raw (untrimmed) source file for this instance lives."""
    vdir = get_settings().videos_dir
    kind = classify_url(video.url)
    if kind == "youtube":
        yid = youtube_id(video.url) or video.video_id
        return vdir / "youtube" / f"{yid}.mp4"
    return vdir / "direct" / f"{video.video_id}.mp4"


def find_existing(video: Video) -> Path | None:
    p = raw_path_for(video)
    if p.exists() and p.stat().st_size > 0:
        return p
    # Videos supplied by the WLASL authors on request: data/videos/provided/<video_id>.<ext>
    prov = get_settings().videos_dir / "provided"
    for ext in VIDEO_EXTS:
        q = prov / f"{video.video_id}{ext}"
        if q.exists() and q.stat().st_size > 0:
            return q
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadError(RuntimeError):
    def __init__(self, msg: str, http_status: int | None = None):
        super().__init__(msg)
        self.http_status = http_status


def _http_get(url: str, dest: Path, referer: str | None = None) -> None:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    timeout = get_settings().download_timeout_s
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                raise DownloadError(f"HTTP {r.status_code}", r.status_code)
            ctype = r.headers.get("content-type", "")
            if "text/html" in ctype:
                raise DownloadError(f"server returned an HTML page ({ctype}), not a video", r.status_code)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            tmp.replace(dest)
    except requests.RequestException as e:
        raise DownloadError(f"{type(e).__name__}: {e}") from e


def _ffmpeg_to_mp4(src: Path, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DownloadError("ffmpeg not found on PATH (needed to convert ASLPro .swf files)")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0 or not dest.exists():
        raise DownloadError(f"ffmpeg conversion failed: {res.stderr.strip()[:300]}")


def _yt_dlp(url: str, dest: Path) -> None:
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise DownloadError("yt-dlp is not installed") from e
    opts = {
        "outtmpl": str(dest.with_suffix(".%(ext)s")),
        "format": "bv*[ext=mp4][height<=720]/bv*[height<=720]/b[height<=720]/bv*/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 2,
        "socket_timeout": get_settings().download_timeout_s,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise DownloadError(f"yt-dlp: {str(e)[:300]}") from e
    if dest.exists():
        return
    produced = [p for p in dest.parent.glob(dest.stem + ".*") if p.suffix in VIDEO_EXTS]
    if not produced:
        raise DownloadError("yt-dlp finished but produced no video file")
    if produced[0].suffix != ".mp4":
        produced[0].rename(dest)  # container may differ; OpenCV/ffmpeg read it by content
    else:
        produced[0].replace(dest)


def download_one(video: Video) -> tuple[Path, str]:
    """Download the raw source for ``video``; returns (path, method)."""
    existing = find_existing(video)
    if existing:
        return existing, "existing"
    dest = raw_path_for(video)
    dest.parent.mkdir(parents=True, exist_ok=True)
    kind = classify_url(video.url)
    if kind == "youtube":
        _yt_dlp(video.url, dest)
    elif kind == "aslpro":
        with tempfile.TemporaryDirectory() as td:
            swf = Path(td) / f"{video.video_id}.swf"
            _http_get(video.url, swf, referer="http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi")
            _ffmpeg_to_mp4(swf, dest)
    else:
        _http_get(video.url, dest)
    return dest, kind


@dataclass
class ProbeResult:
    ok: bool
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    error: str | None = None


def probe_video(path: Path, full_decode: bool = True) -> ProbeResult:
    """Open the file with OpenCV and (optionally) decode every frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ProbeResult(ok=False, error="OpenCV could not open the file")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if full_decode:
            n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
        if n <= 0 or w <= 0 or h <= 0:
            return ProbeResult(ok=False, fps=fps, width=w, height=h, frame_count=n, error="no decodable frames")
        if not fps or fps > 240:
            fps = 25.0
        return ProbeResult(ok=True, fps=float(fps), width=w, height=h, frame_count=n, duration_s=n / fps)
    finally:
        cap.release()


def clip_range(video: Video, frame_count: int) -> tuple[int, int]:
    """0-based [start, end) frame range of the annotated sign within the source file.

    WLASL frame_start/frame_end are 1-based and inclusive; frame_end == -1 means
    the sign lasts until the end of the video.
    """
    start = max(0, (video.frame_start or 1) - 1)
    end = frame_count if (video.frame_end is None or video.frame_end < 0) else min(frame_count, video.frame_end)
    return start, max(start, end)


def validate_video(db, video: Video) -> bool:
    path = Path(video.file_path) if video.file_path else find_existing(video)
    if path is None or not path.exists():
        video.status = "failed" if video.status != "pending" else "pending"
        video.error = video.error or "file missing"
        return False
    pr = probe_video(path)
    video.file_path = str(path)
    if not pr.ok:
        video.status = "invalid"
        video.error = pr.error
        return False
    video.measured_fps, video.width, video.height = pr.fps, pr.width, pr.height
    video.frame_count, video.duration_s = pr.frame_count, pr.duration_s
    s, e = clip_range(video, pr.frame_count)
    video.clip_frames = e - s
    if video.clip_frames < 4:
        video.status = "invalid"
        video.error = f"annotated clip has {video.clip_frames} frames (range {video.frame_start}-{video.frame_end} of {pr.frame_count})"
        return False
    video.sha256 = video.sha256 or sha256_file(path)
    video.status = "valid"
    video.error = None
    return True


ProgressFn = Callable[[float, str], None]


def acquire(
    video_ids: list[str] | None = None,
    class_indices: list[int] | None = None,
    retry_failed: bool = False,
    limit: int | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Download + validate WLASL100 videos. Only WLASL100 rows exist in the DB."""
    with session_scope() as db:
        q = select(Video)
        if video_ids:
            q = q.where(Video.video_id.in_(video_ids))
        if class_indices:
            q = q.where(Video.class_index.in_(class_indices))
        rows = list(db.scalars(q.order_by(Video.class_index, Video.video_id)))
        todo = [v.video_id for v in rows if v.status in ("pending",) or (retry_failed and v.status in ("failed", "invalid"))]
    if limit:
        todo = todo[:limit]

    counts = {"requested": len(todo), "valid": 0, "failed": 0, "invalid": 0}
    # YouTube instances sharing one source file must not be downloaded concurrently.
    lock_by_src: dict[str, object] = {}
    import threading

    glock = threading.Lock()

    def work(vid: str) -> str:
        with session_scope() as db:
            v = db.get(Video, vid)
            src = str(raw_path_for(v))
            with glock:
                lk = lock_by_src.setdefault(src, threading.Lock())
            attempt = DownloadAttempt(video_id=vid, method=classify_url(v.url))
            with lk:  # type: ignore[attr-defined]
                try:
                    path, method = download_one(v)
                    attempt.method = method
                    attempt.ok = True
                    v.file_path = str(path)
                    v.acquired_via = method
                    v.status = "downloaded"
                    v.error = None
                except DownloadError as e:
                    attempt.error = str(e)
                    attempt.http_status = e.http_status
                    v.status = "failed"
                    v.error = str(e)
                except Exception as e:  # unexpected - record, keep going
                    attempt.error = f"{type(e).__name__}: {e}"
                    v.status = "failed"
                    v.error = attempt.error
            attempt.finished_at = utcnow()
            db.add(attempt)
            if v.status == "downloaded":
                validate_video(db, v)
                if attempt.method not in ("existing",):
                    time.sleep(random.uniform(0.3, 0.8))  # be polite to hosts
            return v.status

    done = 0
    workers = max(1, get_settings().download_workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, vid): vid for vid in todo}
        for fut in as_completed(futs):
            status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if progress:
                progress(done / max(1, len(todo)), f"{done}/{len(todo)} processed ({counts['valid']} valid, {counts['failed']} failed)")
            if should_stop and should_stop():
                for f in futs:
                    f.cancel()
                counts["stopped"] = True
                break
    return counts


def scan_and_validate(progress: ProgressFn | None = None) -> dict:
    """Validate files already on disk (e.g. videos provided by the WLASL authors)."""
    counts = {"valid": 0, "invalid": 0, "missing": 0}
    with session_scope() as db:
        ids = [v.video_id for v in db.scalars(select(Video))]
    for i, vid in enumerate(ids):
        with session_scope() as db:
            v = db.get(Video, vid)
            p = find_existing(v)
            if p is None:
                counts["missing"] += 1
            else:
                v.file_path = str(p)
                if v.acquired_via is None:
                    v.acquired_via = "provided" if "provided" in p.parts else "existing"
                ok = validate_video(db, v)
                counts["valid" if ok else "invalid"] += 1
        if progress and i % 20 == 0:
            progress((i + 1) / len(ids), f"{i + 1}/{len(ids)} checked")
    return counts
