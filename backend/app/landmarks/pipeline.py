"""Batch landmark extraction over validated WLASL100 videos."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.dataset.acquisition import clip_range
from app.db.models import LandmarkFile, Video
from app.db.session import session_scope
from app.features.spec import EXTRACTOR_VERSION
from app.landmarks.extractor import HolisticExtractor


def landmark_path(video_id: str) -> Path:
    return get_settings().landmarks_dir / f"{video_id}.npz"


def extract_all(redo: bool = False, limit: int | None = None, video_ids: list[str] | None = None,
                progress: Callable[[float, str], None] | None = None,
                should_stop: Callable[[], bool] | None = None) -> dict:
    with session_scope() as db:
        q = select(Video).where(Video.status == "valid")
        if video_ids:
            q = q.where(Video.video_id.in_(video_ids))
        vids = list(db.scalars(q.order_by(Video.class_index, Video.video_id)))
        done = {lf.video_id for lf in db.scalars(select(LandmarkFile).where(LandmarkFile.extractor_version == EXTRACTOR_VERSION))}
        todo = [(v.video_id, v.file_path, v.frame_start, v.frame_end, v.frame_count) for v in vids if redo or v.video_id not in done]
    if limit:
        todo = todo[:limit]
    ex = HolisticExtractor()
    counts = {"requested": len(todo), "ok": 0, "low_quality": 0, "failed": 0}
    for i, (vid, path, fs, fe, fc) in enumerate(todo):
        if should_stop and should_stop():
            counts["stopped"] = True
            break
        status, err, seq = "ok", None, None
        try:
            tmpv = Video(frame_start=fs, frame_end=fe)
            start, end = clip_range(tmpv, fc or 10**9)
            seq, _fps = ex.extract_video(Path(path), start, end)
            if len(seq) == 0:
                raise ValueError("no frames decoded in annotated range")
            rates = seq.detection_rates()
            if rates["pose"] == 0 or rates["any_hand"] < 0.1:
                status = "low_quality"
            seq.save(landmark_path(vid))
        except Exception as e:  # noqa: BLE001 - per-video failure is recorded
            status, err = "failed", f"{type(e).__name__}: {e}"
        with session_scope() as db:
            lf = db.get(LandmarkFile, vid) or LandmarkFile(video_id=vid, path="")
            lf.status, lf.error, lf.extractor_version = status, err, EXTRACTOR_VERSION
            if seq is not None and status != "failed":
                r = seq.detection_rates()
                lf.path = str(landmark_path(vid))
                lf.n_frames = len(seq)
                lf.pose_rate, lf.left_rate, lf.right_rate = r["pose"], r["left"], r["right"]
                lf.any_hand_rate, lf.face_rate = r["any_hand"], r["face"]
            db.merge(lf)
        counts[status] += 1
        if progress:
            progress((i + 1) / max(1, len(todo)), f"{i + 1}/{len(todo)} videos ({counts['ok']} ok, {counts['failed']} failed)")
    return counts
