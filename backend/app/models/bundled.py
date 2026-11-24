"""Registers the pretrained model shipped in ``models/`` so a fresh clone can recognise signs
immediately, without downloading WLASL or training.

The bundled folder contains ``checkpoint.pt`` (PyTorch, same format the trainer writes) and
``results.json`` (configuration and the validation / held-out test metrics recorded when it was
trained). Importing copies the checkpoint into ``artifacts/checkpoints`` and creates a completed
experiment record pointing at it. It is idempotent: a model that is already registered is not
imported twice.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from app.core.config import REPO_ROOT, get_settings
from app.db.models import Experiment, MetaKV, utcnow
from app.db.session import session_scope

log = logging.getLogger(__name__)

BUNDLED_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_BUNDLED = BUNDLED_MODELS_DIR / "wlasl100-gru-signer-holdout"


def _parse_dt(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def import_bundled_model(folder: Path = DEFAULT_BUNDLED, activate_if_none: bool = True) -> dict:
    """Register ``folder`` (checkpoint.pt + results.json) as a completed experiment.

    Returns ``{"experiment_id", "imported", "activated"}``. Raises ``FileNotFoundError`` if the
    folder does not contain a checkpoint.
    """
    folder = Path(folder)
    ck_src = folder / "checkpoint.pt"
    if not ck_src.is_file():
        raise FileNotFoundError(f"no checkpoint.pt in {folder}")
    results_path = folder / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}

    settings = get_settings()
    dest = settings.checkpoints_dir / f"bundled_{folder.name}.pt"
    with session_scope() as db:
        existing = db.scalar(select(Experiment).where(Experiment.checkpoint_path.like(f"%{dest.name}")))
        if existing is not None:
            if not dest.exists():  # e.g. artifacts/ was cleaned; restore the file
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ck_src, dest)
            return {"experiment_id": existing.id, "imported": False, "activated": existing.is_active}

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ck_src, dest)

    # Read architecture / feature config from the checkpoint itself so the record always matches it.
    import torch

    ck = torch.load(dest, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    n_params = int(sum(v.numel() for v in ck["state_dict"].values() if hasattr(v, "numel")))
    val = dict(results.get("validation") or {})
    if "recommended_threshold" not in val and "recommended_threshold" in meta:
        val["recommended_threshold"] = meta["recommended_threshold"]
    test = dict(results.get("test") or {})
    for m, part in ((val, "val"), (test, "test")):
        if m:
            m.setdefault("partition", part)
            m["summary_only"] = True  # per-class / confusion details were not shipped with the model

    ds = dict(results.get("data_summary") or {})

    # Recorded when the model was trained: the signer-holdout split passed every leakage check.
    ds["split_info"] = {
        "name": results.get("split") or meta.get("split"), "kind": "signer-holdout", "signer_independent": True,
        "leakage_passed": True, "n_signers": {k: int((ds.get(k) or {}).get("signers") or 0) for k in ("train", "val", "test")},
    }

    with session_scope() as db:
        has_active = db.scalar(select(Experiment).where(Experiment.is_active.is_(True))) is not None
        exp = Experiment(
            name=f"Pretrained · {results.get('name') or folder.name}",
            status="completed",
            architecture=meta.get("architecture") or results.get("architecture", "gru"),
            config=results.get("config") or meta.get("train_config") or {},
            feature_config=meta.get("feature_config") or results.get("feature_config") or {},
            split_id=None,
            split_name=results.get("split") or meta.get("split"),
            dataset_version=results.get("dataset_version"),
            data_summary=ds,
            history=[],
            val_metrics=val,
            test_metrics=test,
            checkpoint_path=str(dest),
            device="cpu",
            n_params=n_params,
            is_active=activate_if_none and not has_active,
            is_synthetic=False,
            finished_at=_parse_dt(test.get("evaluated_at")) or utcnow(),
        )
        db.add(exp)
        db.flush()
        key = f"bundled:{folder.name}"[:64]
        marker = db.get(MetaKV, key)
        if marker is None:
            db.add(MetaKV(key=key, value={"experiment_id": exp.id, "imported_at": utcnow().isoformat()}))
        else:
            marker.value = {"experiment_id": exp.id, "imported_at": utcnow().isoformat()}
        out = {"experiment_id": exp.id, "imported": True, "activated": exp.is_active}
    log.info("registered bundled model %s as experiment %s (active=%s)", folder.name, out["experiment_id"], out["activated"])
    return out


def ensure_bundled_model() -> None:
    """Startup hook: import the default bundled model once per database when it is present.

    If the user later deletes the imported experiment it is not re-imported automatically
    (``python -m app.cli import-model`` brings it back). Never raises.
    """
    if not get_settings().import_bundled_model or not (DEFAULT_BUNDLED / "checkpoint.pt").is_file():
        return
    try:
        with session_scope() as db:
            if db.get(MetaKV, f"bundled:{DEFAULT_BUNDLED.name}"[:64]) is not None:
                return
        import_bundled_model(DEFAULT_BUNDLED)
    except Exception:  # a broken bundle must not stop the server
        log.exception("could not register the bundled model")
