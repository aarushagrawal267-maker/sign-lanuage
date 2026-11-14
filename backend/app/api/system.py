"""System, overview, splits, jobs and feature-spec endpoints."""

from __future__ import annotations

import platform
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.jobs import get_jobs
from app.db.models import Experiment, Gloss, Job, LandmarkFile, RecognitionEvent, Split, Video
from app.db.session import get_db
from app.features.spec import FeatureConfig, raw_spec
from app.inference import registry

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", summary="Liveness check")
def health():
    return {"ok": True}


@router.get("/system", summary="Runtime information (versions, device, assets)")
def system():
    import torch

    s = get_settings()
    try:
        import mediapipe

        mp_version = mediapipe.__version__
    except Exception:  # pragma: no cover
        mp_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mediapipe": mp_version,
        "holistic_model_present": s.holistic_model_path.exists(),
        "data_dir": str(s.data_dir),
    }


@router.get("/features/spec", summary="Raw landmark layout and feature layout (shared with the browser)")
def feature_spec():
    p, info = registry.get_predictor()
    fcfg = p.feature_config if p else FeatureConfig()
    return {"raw": raw_spec(), "feature_config": fcfg.to_dict(), "feature_dim": fcfg.dim, "layout": fcfg.layout(),
            "from_active_model": p is not None}


@router.get("/models/active", summary="Active recognition model")
def active_model():
    p, info = registry.get_predictor()
    return info or {"experiment_id": None}


@router.get("/overview", summary="Data for the overview dashboard (all values from the database)")
def overview(db: Session = Depends(get_db)):
    n_gloss = db.scalar(select(func.count()).select_from(Gloss)) or 0
    status = dict(db.execute(select(Video.status, func.count()).group_by(Video.status)).all())
    lm_ok = db.scalar(select(func.count()).select_from(LandmarkFile).where(LandmarkFile.status == "ok")) or 0
    ex_counts = dict(db.execute(select(Experiment.status, func.count()).group_by(Experiment.status)).all())
    active = db.scalar(select(Experiment).where(Experiment.is_active.is_(True)))
    latest = db.scalar(select(Experiment).where(Experiment.status == "completed").order_by(Experiment.id.desc()))
    running = list(db.scalars(select(Job).where(Job.status.in_(("queued", "running")))))
    events = list(db.scalars(select(RecognitionEvent).order_by(RecognitionEvent.id.desc()).limit(12)))
    # training samples by official split with landmarks
    rows = db.execute(select(Video.split, func.count()).join(LandmarkFile, LandmarkFile.video_id == Video.video_id)
                      .where(LandmarkFile.status == "ok").group_by(Video.split)).all()

    def exp_brief(e: Experiment | None):
        if e is None:
            return None
        return {"id": e.id, "name": e.name, "architecture": e.architecture, "split_name": e.split_name,
                "finished_at": e.finished_at, "is_synthetic": e.is_synthetic, "n_params": e.n_params,
                "val": {k: (e.val_metrics or {}).get(k) for k in ("top1", "top5", "macro_f1", "n")} if e.val_metrics else None,
                "test": {k: (e.test_metrics or {}).get(k) for k in ("top1", "top5", "macro_f1", "n")} if e.test_metrics else None,
                "latency": (e.val_metrics or {}).get("latency")}

    ev_counts = Counter(e.gloss for e in events)
    return {
        "vocabulary_size": n_gloss,
        "video_status": status,
        "total_videos": sum(status.values()),
        "landmark_sequences": lm_ok,
        "samples_by_official_split": dict(rows),
        "experiments": ex_counts,
        "active_model": exp_brief(active),
        "latest_experiment": exp_brief(latest),
        "running_jobs": [{"id": j.id, "kind": j.kind, "progress": j.progress, "message": j.message} for j in running],
        "recent_events": [{"gloss": e.gloss, "score": e.score, "source": e.source, "created_at": e.created_at} for e in events],
        "recent_event_counts": dict(ev_counts),
    }


# ------------------------------------------------------------------ splits
@router.get("/splits", tags=["splits"], summary="Dataset partitions with leakage reports")
def list_splits(db: Session = Depends(get_db)):
    return [{"id": s.id, "name": s.name, "kind": s.kind, "signer_independent": s.signer_independent, "seed": s.seed,
             "config": s.config, "leakage_report": s.leakage_report, "created_at": s.created_at}
            for s in db.scalars(select(Split).order_by(Split.id))]


@router.post("/splits/rebuild", tags=["splits"], summary="Rebuild official + signer-grouped splits")
def rebuild_splits(seed: int = 0, db: Session = Depends(get_db)):
    from app.splits.splits import ensure_default_splits

    used = db.scalar(select(func.count()).select_from(Experiment)) or 0
    if used:
        raise HTTPException(409, "experiments reference the current splits; rebuilding would invalidate them")
    return {"splits": [s.name for s in ensure_default_splits(db, seed)]}


@router.post("/splits/{split_id}/check", tags=["splits"], summary="Re-run the automated leakage checks")
def check_split(split_id: int, db: Session = Depends(get_db)):
    from app.splits.splits import refresh_leakage

    s = db.get(Split, split_id)
    if s is None:
        raise HTTPException(404, "split not found")
    return refresh_leakage(db, s)


# -------------------------------------------------------------------- jobs
@router.get("/jobs", tags=["jobs"], summary="Recent background jobs")
def list_jobs(limit: int = 20, db: Session = Depends(get_db)):
    return [{"id": j.id, "kind": j.kind, "status": j.status, "progress": j.progress, "message": j.message,
             "params": j.params, "result": {k: v for k, v in (j.result or {}).items() if k != "traceback"},
             "created_at": j.created_at, "finished_at": j.finished_at}
            for j in db.scalars(select(Job).order_by(Job.id.desc()).limit(limit))]


@router.post("/jobs/{job_id}/cancel", tags=["jobs"], summary="Request cancellation of a job")
def cancel_job(job_id: int):
    if not get_jobs().cancel(job_id):
        raise HTTPException(404, "job is not running")
    return {"cancelling": job_id}
