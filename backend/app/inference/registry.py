"""Holds the active model used for live and offline recognition."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from sqlalchemy import select

from app.core.config import resolve_checkpoint
from app.db.models import Experiment
from app.db.session import session_scope
from app.models.checkpoint import Predictor

log = logging.getLogger(__name__)

NO_ACTIVE_MODEL = "No active model. Train and activate a model in the Model Laboratory."
_RESTORE_HINT = ("Activate another model in the Model Laboratory, or restore the pretrained model with "
                 "'python -m app.cli import-model --activate' (from the backend folder).")

_lock = threading.Lock()
_cache: dict = {"id": None, "predictor": None}


def active_experiment_id() -> int | None:
    with session_scope() as db:
        e = db.scalar(select(Experiment).where(Experiment.is_active.is_(True)))
        return e.id if e else None


def load_active() -> tuple[Predictor | None, dict | None, str | None]:
    """Returns (predictor, info, problem) for the active experiment, loading it on first use.

    ``problem`` is a human-readable reason when no predictor is available: no active model,
    a missing checkpoint file, or a checkpoint that cannot be loaded. It never raises for those
    cases, so a broken model file gives the user a clear message instead of a server error.
    """
    with session_scope() as db:
        e = db.scalar(select(Experiment).where(Experiment.is_active.is_(True)))
        if e is None:
            return None, None, NO_ACTIVE_MODEL
        ck = resolve_checkpoint(e.checkpoint_path)
        if ck is None:
            msg = (f"The active model '{e.name}' (experiment {e.id}) has no checkpoint file at "
                   f"{e.checkpoint_path or '(none recorded)'}. {_RESTORE_HINT}")
            log.warning(msg)
            return None, None, msg
        info = {"experiment_id": e.id, "name": e.name, "architecture": e.architecture, "split": e.split_name,
                "is_synthetic": e.is_synthetic, "val_top1": (e.val_metrics or {}).get("top1"),
                "checkpoint": e.checkpoint_path}
        eid, name, path = e.id, e.name, str(ck)
    with _lock:
        # Keyed by id *and* checkpoint path so a different database (or a re-imported model)
        # with the same experiment id never reuses a stale predictor.
        if _cache["id"] != (eid, path) or _cache["predictor"] is None:
            _cache["id"], _cache["predictor"] = None, None
            try:
                _cache["predictor"] = Predictor(Path(path), "cpu")
            except Exception as exc:  # noqa: BLE001 - corrupt / incompatible checkpoint
                msg = (f"The checkpoint of the active model '{name}' (experiment {eid}) could not be loaded "
                       f"({type(exc).__name__}: {str(exc)[:300]}). {_RESTORE_HINT}")
                log.error("%s [%s]", msg, path)
                return None, None, msg
            _cache["id"] = (eid, path)
        p = _cache["predictor"]
    info["threshold"] = p.recommended_threshold
    info["seq_len"] = p.feature_config.seq_len
    info["feature_config"] = p.feature_config.to_dict()
    info["num_classes"] = len(p.class_names)
    return p, info, None


def get_predictor() -> tuple[Predictor | None, dict | None]:
    """Returns (predictor, info) for the active experiment, or (None, None) when unavailable."""
    p, info, _ = load_active()
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
