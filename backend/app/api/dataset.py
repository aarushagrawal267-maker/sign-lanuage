"""Dataset endpoints: annotations, inventory, acquisition, validation, extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.jobs import get_jobs
from app.dataset.acquisition import classify_url
from app.db.models import DownloadAttempt, Gloss, LandmarkFile, MetaKV, Video
from app.db.session import get_db
from app.features.build import sequence_quality
from app.landmarks.raw import RawSequence

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

UNAVAILABLE = ("failed", "invalid")


def _host(url: str) -> str:
    kind = classify_url(url)
    if kind == "youtube":
        return "youtube"
    from urllib.parse import urlparse

    return urlparse(url).netloc.replace("www.", "") or "unknown"


@router.get("/status", summary="Dataset provenance and status counts")
def dataset_status(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count()).select_from(Video)) or 0
    by_status = dict(db.execute(select(Video.status, func.count()).group_by(Video.status)).all())
    lm = dict(db.execute(select(LandmarkFile.status, func.count()).group_by(LandmarkFile.status)).all())
    prov = db.get(MetaKV, "annotations")
    synthetic = db.get(MetaKV, "synthetic")
    return {
        "annotations_loaded": total > 0,
        "total_videos": total,
        "status_counts": by_status,
        "landmark_counts": lm,
        "provenance": prov.value if prov else None,
        "synthetic": synthetic.value if synthetic else None,
    }


@router.get("/inventory", summary="Per-gloss WLASL100 inventory")
def inventory(db: Session = Depends(get_db)):
    glosses = list(db.scalars(select(Gloss).order_by(Gloss.class_index)))
    videos = list(db.scalars(select(Video)))
    lms = {lf.video_id: lf for lf in db.scalars(select(LandmarkFile))}
    rows = []
    by_class: dict[int, list[Video]] = defaultdict(list)
    for v in videos:
        by_class[v.class_index].append(v)
    for g in glosses:
        vs = by_class[g.class_index]
        valid = [v for v in vs if v.status == "valid"]
        durs = [v.clip_frames / v.measured_fps for v in valid if v.clip_frames and v.measured_fps]
        res = Counter(f"{v.width}x{v.height}" for v in valid if v.width)
        fps = Counter(round(v.measured_fps) for v in valid if v.measured_fps)
        rows.append({
            "class_index": g.class_index,
            "gloss": g.gloss,
            "annotated": len(vs),
            "valid": len(valid),
            "unavailable": sum(v.status in UNAVAILABLE for v in vs),
            "pending": sum(v.status in ("pending", "downloaded") for v in vs),
            "landmarks": sum(1 for v in vs if v.video_id in lms and lms[v.video_id].status == "ok"),
            "signers": sorted({v.signer_id for v in vs}),
            "split": dict(Counter(v.split for v in vs)),
            "variations": len({v.variation_id for v in vs}),
            "median_duration_s": float(np.median(durs)) if durs else None,
            "resolutions": dict(res.most_common(3)),
            "fps": dict(fps.most_common(3)),
        })
    return {"glosses": rows}


@router.get("/summary", summary="Aggregate dataset statistics and quality")
def summary(db: Session = Depends(get_db)):
    videos = list(db.scalars(select(Video)))
    lms = list(db.scalars(select(LandmarkFile)))
    hosts: dict[str, Counter] = defaultdict(Counter)
    for v in videos:
        hosts[_host(v.url)][v.status] += 1
    signer = Counter(v.signer_id for v in videos)
    signer_split: dict[int, Counter] = defaultdict(Counter)
    for v in videos:
        signer_split[v.signer_id][v.split] += 1
    valid = [v for v in videos if v.status == "valid" and v.clip_frames and v.measured_fps]
    clip_frames = np.array([v.clip_frames for v in valid]) if valid else np.array([])
    durations = clip_frames / np.array([v.measured_fps for v in valid]) if valid else np.array([])
    seq_rec = None
    if len(clip_frames):
        med = float(np.median(clip_frames))
        seq_rec = {
            "median_frames": med,
            "p10_frames": float(np.percentile(clip_frames, 10)),
            "p90_frames": float(np.percentile(clip_frames, 90)),
            "recommended_seq_len": int(min(64, max(32, 8 * round(med / 8)))),
            "basis": f"{len(clip_frames)} validated clips",
        }
    hist = None
    if len(durations):
        counts, edges = np.histogram(durations, bins=np.arange(0, max(6.5, durations.max() + 0.5), 0.5))
        hist = [{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(c)} for i, c in enumerate(counts)]
    errors = Counter((v.error or "")[:80] for v in videos if v.status in UNAVAILABLE)
    return {
        "videos": len(videos),
        "status_counts": dict(Counter(v.status for v in videos)),
        "official_split": dict(Counter(v.split for v in videos)),
        "hosts": [{"host": h, "total": sum(c.values()), **dict(c)} for h, c in sorted(hosts.items(), key=lambda x: -sum(x[1].values()))],
        "signers": [{"signer_id": s, "videos": n, "split": dict(signer_split[s])} for s, n in signer.most_common()],
        "variation_counts": dict(Counter(v.variation_id for v in videos)),
        "duration_histogram": hist,
        "sequence_length": seq_rec,
        "landmarks": {
            "status_counts": dict(Counter(lf.status for lf in lms)),
            "mean_pose_rate": float(np.mean([lf.pose_rate for lf in lms])) if lms else None,
            "mean_any_hand_rate": float(np.mean([lf.any_hand_rate for lf in lms])) if lms else None,
            "mean_face_rate": float(np.mean([lf.face_rate for lf in lms])) if lms else None,
        },
        "top_errors": [{"error": e, "count": n} for e, n in errors.most_common(10)],
    }


@router.get("/videos", summary="List WLASL100 video instances")
def list_videos(
    db: Session = Depends(get_db),
    class_index: int | None = None,
    status: str | None = None,
    split: str | None = None,
    signer_id: int | None = None,
    offset: int = 0,
    limit: int = Query(50, le=500),
):
    q = select(Video)
    if class_index is not None:
        q = q.where(Video.class_index == class_index)
    if status:
        q = q.where(Video.status == status)
    if split:
        q = q.where(Video.split == split)
    if signer_id is not None:
        q = q.where(Video.signer_id == signer_id)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = list(db.scalars(q.order_by(Video.class_index, Video.video_id).offset(offset).limit(limit)))
    lms = {lf.video_id: lf for lf in db.scalars(select(LandmarkFile).where(LandmarkFile.video_id.in_([v.video_id for v in rows])))}
    return {
        "total": total,
        "items": [
            {
                "video_id": v.video_id, "gloss": v.gloss, "class_index": v.class_index, "signer_id": v.signer_id,
                "split": v.split, "source": v.source, "host": _host(v.url), "url": v.url, "status": v.status,
                "error": v.error, "frame_start": v.frame_start, "frame_end": v.frame_end, "variation_id": v.variation_id,
                "fps": v.measured_fps, "width": v.width, "height": v.height, "duration_s": v.duration_s,
                "clip_frames": v.clip_frames, "has_file": bool(v.file_path and Path(v.file_path).exists()),
                "landmarks": None if v.video_id not in lms else {
                    "status": lms[v.video_id].status, "frames": lms[v.video_id].n_frames,
                    "any_hand_rate": lms[v.video_id].any_hand_rate, "pose_rate": lms[v.video_id].pose_rate,
                    "face_rate": lms[v.video_id].face_rate,
                },
            }
            for v in rows
        ],
    }


@router.get("/videos/{video_id}/attempts", summary="Download attempts for one video")
def video_attempts(video_id: str, db: Session = Depends(get_db)):
    rows = db.scalars(select(DownloadAttempt).where(DownloadAttempt.video_id == video_id).order_by(DownloadAttempt.id.desc()))
    return [{"method": a.method, "ok": a.ok, "http_status": a.http_status, "error": a.error,
             "started_at": a.started_at, "finished_at": a.finished_at} for a in rows]


@router.get("/videos/{video_id}/file", summary="Stream your local copy of a video (never proxied from the web)")
def video_file(video_id: str, db: Session = Depends(get_db)):
    v = db.get(Video, video_id)
    if v is None or not v.file_path or not Path(v.file_path).exists():
        raise HTTPException(404, "no local copy of this video")
    return FileResponse(v.file_path, media_type="video/mp4")


@router.get("/videos/{video_id}/landmarks", summary="Raw extracted landmarks for one video")
def video_landmarks(video_id: str, db: Session = Depends(get_db), max_frames: int = 400):
    lf = db.get(LandmarkFile, video_id)
    if lf is None or not lf.path or not Path(lf.path).is_file():
        raise HTTPException(404, "landmarks not extracted for this video")
    v = db.get(Video, video_id)
    seq = RawSequence.load(Path(lf.path))
    step = max(1, int(np.ceil(len(seq) / max_frames)))
    frames = [seq.frame_to_json(i) for i in range(0, len(seq), step)]
    return {
        "video_id": video_id, "gloss": v.gloss if v else None, "signer_id": v.signer_id if v else None,
        "width": seq.width, "height": seq.height, "frame_step": step, "frames": frames,
        "quality": sequence_quality(seq), "extractor_version": seq.extractor_version,
    }


# ----------------------------------------------------------------- actions
class AcquireRequest(BaseModel):
    class_indices: list[int] | None = Field(None, description="Restrict to these WLASL100 classes")
    video_ids: list[str] | None = None
    limit: int | None = Field(None, ge=1, description="Maximum videos to attempt in this run")
    retry_failed: bool = False


class ExtractRequest(BaseModel):
    limit: int | None = Field(None, ge=1)
    redo: bool = False
    video_ids: list[str] | None = None


@router.post("/annotations", summary="Fetch official WLASL annotations, select WLASL100, build splits")
def load_annotations(fetch: bool = True, force: bool = False):
    from app.dataset.annotations import AnnotationError, load_and_ingest
    from app.db.session import session_scope
    from app.splits.splits import ensure_default_splits

    try:
        with session_scope() as db:
            res = load_and_ingest(db, fetch=fetch, force=force)
            splits = ensure_default_splits(db)
            res["splits"] = [s.name for s in splits]
    except AnnotationError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # network errors
        raise HTTPException(502, f"could not fetch annotations: {e}") from e
    return res


@router.post("/acquire", summary="Download + validate WLASL100 videos (background job)")
def acquire_videos(req: AcquireRequest):
    from app.dataset.acquisition import acquire

    def run(progress, stop):
        return acquire(req.video_ids, req.class_indices, req.retry_failed, req.limit, progress, stop)

    try:
        return {"job_id": get_jobs().submit("acquire", req.model_dump(), run)}
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@router.post("/scan", summary="Validate video files already on disk (incl. author-provided videos)")
def scan_local():
    from app.dataset.acquisition import scan_and_validate

    try:
        return {"job_id": get_jobs().submit("scan", {}, lambda p, s: scan_and_validate(p))}
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@router.post("/extract", summary="Extract MediaPipe Holistic landmarks for valid videos (background job)")
def extract_landmarks(req: ExtractRequest):
    from app.landmarks.extractor import ExtractorUnavailable, HolisticExtractor
    from app.landmarks.pipeline import extract_all

    try:
        HolisticExtractor()
    except ExtractorUnavailable as e:
        raise HTTPException(400, str(e)) from e

    def run(progress, stop):
        return extract_all(req.redo, req.limit, req.video_ids, progress, stop)

    try:
        return {"job_id": get_jobs().submit("extract", req.model_dump(), run)}
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
