"""Holds the active model used for live and offline recognition."""

from __future__ import annotations

import threading
from pathlib import Path

from sqlalchemy import select

from app.core.config import resolve_checkpoint
from app.db.models import Experiment
from app.db.session import session_scope
from app.models.checkpoint import Predictor

_lock = threading.Lock()
_cache: dict = {"id": None, "predictor": None}


def active_experiment_id() -> int | None:
    with session_scope() as db:
        e = db.scalar(select(Experiment).where(Experiment.is_active.is_(True)))
        return e.id if e else None


def get_predictor() -> tuple[Predictor | None, dict | None]:
    """Returns (predictor, info) for the active experiment, loading it on first use."""
    with session_scope() as db:
        e = db.scalar(select(Experiment).where(Experiment.is_active.is_(True)))
        ck = resolve_checkpoint(e.checkpoint_path) if e is not None else None
        if e is None or ck is None:
            return None, None
        info = {"experiment_id": e.id, "name": e.name, "architecture": e.architecture, "split": e.split_name,
                "is_synthetic": e.is_synthetic, "val_top1": (e.val_metrics or {}).get("top1"),
                "checkpoint": e.checkpoint_path}
        eid, path = e.id, str(ck)
    with _lock:
        # Keyed by id *and* checkpoint path so a different database (or a re-imported model)
        # with the same experiment id never reuses a stale predictor.
        if _cache["id"] != (eid, path) or _cache["predictor"] is None:
            _cache["predictor"] = Predictor(Path(path), "cpu")
            _cache["id"] = (eid, path)
        p = _cache["predictor"]
    info["threshold"] = p.recommended_threshold
    info["seq_len"] = p.feature_config.seq_len
    info["feature_config"] = p.feature_config.to_dict()
    info["num_classes"] = len(p.class_names)
    return p, info


def activate(exp_id: int) -> None:
    with session_scope() as db:
        target = db.get(Experiment, exp_id)
        if target is None or target.status != "completed":
            raise ValueError("only completed experiments can be activated")
        for e in db.scalars(select(Experiment).where(Experiment.is_active.is_(True))):
            e.is_active = False
        target.is_active = True
    with _lock:
        _cache["id"], _cache["predictor"] = None, None
