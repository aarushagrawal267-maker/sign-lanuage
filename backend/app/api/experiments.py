"""Experiments: training jobs, evaluation, activation and history."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.jobs import get_jobs
from app.db.models import Experiment, Job, Split, utcnow
from app.db.session import get_db, session_scope
from app.features.spec import FeatureConfig
from app.inference import registry
from app.training.trainer import TrainConfig

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class FeatureOptions(BaseModel):
    seq_len: int = Field(48, ge=8, le=128)
    use_face: bool = True
    use_velocity: bool = True
    max_gap: int = Field(5, ge=0, le=30)


class TrainRequest(BaseModel):
    name: str | None = Field(None, max_length=120)
    split_name: str = "signer-holdout"
    architecture: str = Field("gru", pattern="^(gru|lstm|tcn)$")
    hidden: int = Field(192, ge=16, le=1024)
    layers: int = Field(2, ge=1, le=6)
    bidirectional: bool = True
    dropout: float = Field(0.3, ge=0.0, le=0.9)
    lr: float = Field(1e-3, gt=0, le=1.0)
    weight_decay: float = Field(1e-2, ge=0, le=1.0)
    batch_size: int = Field(32, ge=2, le=512)
    epochs: int = Field(80, ge=1, le=500)
    patience: int = Field(20, ge=1, le=500)
    label_smoothing: float = Field(0.1, ge=0, le=0.5)
    class_weighting: str = Field("none", pattern="^(none|balanced)$")
    augment: bool = True
    seed: int = 42
    device: str = Field("auto", pattern="^(auto|cpu|cuda)$")
    features: FeatureOptions = FeatureOptions()


def _summary(e: Experiment) -> dict:
    vm, tm = e.val_metrics or {}, e.test_metrics or {}
    last = (e.history or [])[-1] if e.history else None
    return {
        "id": e.id, "name": e.name, "status": e.status, "architecture": e.architecture, "split_name": e.split_name,
        "dataset_version": e.dataset_version, "created_at": e.created_at, "finished_at": e.finished_at,
        "is_active": e.is_active, "is_synthetic": e.is_synthetic, "device": e.device, "n_params": e.n_params,
        "seq_len": (e.feature_config or {}).get("seq_len") or (e.config or {}).get("features", {}).get("seq_len"),
        "epochs_run": len(e.history or []), "epochs_planned": (e.config or {}).get("epochs"),
        "last_epoch": last, "error": e.error, "checkpoint_path": e.checkpoint_path,
        "val": {k: vm.get(k) for k in ("top1", "top5", "macro_f1", "n", "best_epoch", "recommended_threshold")} if vm else None,
        "test": {k: tm.get(k) for k in ("top1", "top5", "macro_f1", "n", "evaluated_at")} if tm else None,
    }


@router.get("", summary="List experiments (most recent first)")
def list_experiments(db: Session = Depends(get_db)):
    return [_summary(e) for e in db.scalars(select(Experiment).order_by(Experiment.id.desc()))]


def _csv(df, filename: str):
    from fastapi.responses import Response

    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/export.csv", summary="Export all experiments with their metrics as CSV")
def export_experiments(db: Session = Depends(get_db)):
    import pandas as pd

    rows = []
    for e in db.scalars(select(Experiment).order_by(Experiment.id)):
        s = _summary(e)
        rows.append({
            "id": e.id, "name": e.name, "status": e.status, "architecture": e.architecture, "split": e.split_name,
            "signer_independent": ((e.data_summary or {}).get("split_info") or {}).get("signer_independent") if e.split_id is None else bool(db.get(Split, e.split_id).signer_independent),
            "dataset_version": e.dataset_version, "seq_len": s["seq_len"], "n_params": e.n_params, "epochs_run": s["epochs_run"],
            "val_top1": (s["val"] or {}).get("top1"), "val_top5": (s["val"] or {}).get("top5"), "val_macro_f1": (s["val"] or {}).get("macro_f1"),
            "test_top1": (s["test"] or {}).get("top1"), "test_top5": (s["test"] or {}).get("top5"), "test_macro_f1": (s["test"] or {}).get("macro_f1"),
            "synthetic": e.is_synthetic, "checkpoint": e.checkpoint_path, "created_at": e.created_at, "config": e.config,
        })
    return _csv(pd.DataFrame(rows), "experiments.csv")


@router.get("/{exp_id}/per-class.csv", summary="Per-class validation (and test, if evaluated) metrics as CSV")
def export_per_class(exp_id: int, db: Session = Depends(get_db)):
    import pandas as pd

    e = db.get(Experiment, exp_id)
    if e is None:
        raise HTTPException(404, "experiment not found")
    frames = []
    for part, m in (("val", e.val_metrics or {}), ("test", e.test_metrics or {})):
        if m.get("per_class"):
            frames.append(pd.DataFrame(m["per_class"]).assign(partition=part))
    if not frames:
        raise HTTPException(404, "no per-class metrics recorded")
    return _csv(pd.concat(frames), f"experiment{exp_id}_per_class.csv")


@router.get("/{exp_id}", summary="Full experiment record incl. history and metrics")
def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    e = db.get(Experiment, exp_id)
    if e is None:
        raise HTTPException(404, "experiment not found")
    split = db.get(Split, e.split_id) if e.split_id else None
    return {
        **_summary(e), "config": e.config, "feature_config": e.feature_config, "data_summary": e.data_summary,
        "history": e.history, "val_metrics": e.val_metrics, "test_metrics": e.test_metrics,
        "split": (e.data_summary or {}).get("split_info") if split is None else {
            "name": split.name, "kind": split.kind, "signer_independent": split.signer_independent,
            "leakage_passed": (split.leakage_report or {}).get("passed"),
            "n_signers": (split.leakage_report or {}).get("n_signers"),
        },
    }


@router.post("", summary="Create an experiment and start training in the background")
def create_experiment(req: TrainRequest, db: Session = Depends(get_db)):
    split = db.scalar(select(Split).where(Split.name == req.split_name))
    if split is None:
        raise HTTPException(400, f"split {req.split_name!r} not found - load the annotations first")
    cfg = TrainConfig(**{k: v for k, v in req.model_dump().items() if k not in ("name", "split_name", "features")},
                      features=req.features.model_dump())
    try:
        cfg.validate()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    synthetic = db.scalar(select(Job).where(Job.kind == "train", Job.status.in_(("queued", "running"))))
    if synthetic is not None:
        raise HTTPException(409, f"training job #{synthetic.id} is already running")
    from app.db.models import MetaKV

    is_syn = db.get(MetaKV, "synthetic") is not None
    name = req.name or f"{req.architecture.upper()} · {split.name} · T={req.features.seq_len}"
    e = Experiment(name=name, architecture=cfg.architecture, config=asdict(cfg),
                   feature_config=FeatureConfig.from_dict(cfg.features).to_dict(),
                   split_id=split.id, split_name=split.name, status="queued", is_synthetic=is_syn)
    db.add(e)
    db.flush()
    exp_id = e.id
    db.commit()

    from app.training.trainer import run_training

    def run(progress, stop):
        return run_training(exp_id, progress, stop)

    def on_error(msg: str):
        with session_scope() as s:
            x = s.get(Experiment, exp_id)
            x.status = "cancelled" if msg == "cancelled" else "failed"
            x.error = None if msg == "cancelled" else msg
            x.finished_at = utcnow()

    try:
        job_id = get_jobs().submit("train", {"experiment_id": exp_id}, run, on_error)
    except RuntimeError as err:
        e.status, e.error = "failed", str(err)
        raise HTTPException(409, str(err)) from err
    return {"experiment_id": exp_id, "job_id": job_id}


@router.post("/{exp_id}/cancel", summary="Stop a running training job")
def cancel(exp_id: int, db: Session = Depends(get_db)):
    jobs = db.scalars(select(Job).where(Job.kind == "train", Job.status.in_(("queued", "running"))))
    for j in jobs:
        if (j.params or {}).get("experiment_id") == exp_id:
            get_jobs().cancel(j.id)
            return {"cancelled": True}
    raise HTTPException(404, "no running job for this experiment")


@router.post("/{exp_id}/activate", summary="Use this model for live and offline recognition")
def activate(exp_id: int):
    try:
        registry.activate(exp_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"active": exp_id}


@router.post("/{exp_id}/evaluate-test", summary="Run held-out TEST evaluation (after model selection)")
def evaluate_test(exp_id: int):
    from app.training.trainer import run_test_evaluation

    try:
        return run_test_evaluation(exp_id)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/{exp_id}", summary="Delete an experiment and its checkpoint")
def delete_experiment(exp_id: int, db: Session = Depends(get_db)):
    e = db.get(Experiment, exp_id)
    if e is None:
        raise HTTPException(404, "experiment not found")
    if e.is_active:
        raise HTTPException(400, "cannot delete the active model")
    if e.status in ("queued", "running"):
        raise HTTPException(400, "cancel the training job first")
    if e.checkpoint_path and Path(e.checkpoint_path).exists():
        Path(e.checkpoint_path).unlink()
    db.delete(e)
    return {"deleted": exp_id}
