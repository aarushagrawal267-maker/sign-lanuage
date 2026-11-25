"""Closed-set core classifier for ISL (docs/ISL_DESIGN.md 6.3 a): the existing ``RecurrentClassifier``
trained on INCLUDE clips plus the ISLRTC clips of the same concepts (another signer), with strong augmentation.
The checkpoint uses the ASL format, so ``Predictor`` loads it unchanged (class names = concept ids)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.evaluation.metrics import confidence_curve, pick_threshold
from app.features.spec import FeatureConfig
from app.isl.models.calibration import fit_temperature
from app.isl.training.common import ClipDataset, IslData, items_for, set_threads
from app.models.checkpoint import frame_mask
from app.models.networks import build_model
from app.training.trainer import TrainConfig, classification_val_metrics, fit, set_seed


def train_classifier(data: IslData, out: Path, fcfg: FeatureConfig, cfg: TrainConfig, *, min_clips: int = 2,
                     islrtc_oversample: int = 4, smoke: bool = False, threads: int = 2, progress=None) -> dict:
    set_threads(threads)
    set_seed(cfg.seed)
    t0 = time.time()
    train_inc = data.include_part("train")
    counts = Counter(c.concept_id for c in train_inc)
    classes = sorted(c for c, n in counts.items() if n >= min_clips)
    if len(classes) < 2:
        raise RuntimeError(f"not enough INCLUDE training data: {len(classes)} concepts with >= {min_clips} clips")
    label_of = {c: i for i, c in enumerate(classes)}
    isl_same = [c for c in data.islrtc if c.concept_id in label_of]
    train_items = items_for(train_inc, label_of, fcfg) + items_for(isl_same, label_of, fcfg) * islrtc_oversample
    val_items = items_for(data.include_part("val"), label_of, fcfg, with_alternatives=False)
    if not val_items:
        raise RuntimeError("no INCLUDE validation clips for the trained concepts")
    train_ds = ClipDataset(train_items, fcfg, train=True, seed=cfg.seed)
    val_ds = ClipDataset(val_items, fcfg, train=False)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g,
                              drop_last=len(train_ds) > cfg.batch_size)
    val_loader = DataLoader(val_ds, batch_size=64)
    model = build_model(cfg.architecture, fcfg.dim, len(classes), **cfg.model_kwargs())
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    def loss_fn(m, x, y):
        logits = m(x, frame_mask(x, fcfg))
        return crit(logits, y), logits

    meta = {"architecture": cfg.architecture, "model_kwargs": cfg.model_kwargs(), "feature_config": fcfg.to_dict(),
            "class_names": classes, "language": "isl", "profile": data.profile, "split": data.split.get("name"),
            "split_label": data.split.get("label"), "train_config": asdict(cfg), "smoke": smoke,
            "data": {**data.summary(), "train_items": len(train_items), "val_items": len(val_items),
                     "islrtc_clips_of_classes": len(isl_same)}}
    res = fit(model, train_loader, lambda m: classification_val_metrics(m, val_loader, fcfg, "cpu"), loss_fn, cfg, out,
              meta, progress=progress)
    # calibration on validation: temperature + the classic confidence threshold
    ck = torch.load(out, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    xs = torch.stack([torch.from_numpy(f) for f in val_ds._cache])
    ys = np.array([it.label for it in val_items])
    with torch.no_grad():
        logits = model(xs, frame_mask(xs, fcfg)).numpy()
    T = fit_temperature(logits, ys)
    z = logits / T
    probs = np.exp(z - z.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    curve = confidence_curve(probs, ys)
    ck["meta"].update(temperature=T, recommended_threshold=pick_threshold(curve), val_top1=float((probs.argmax(1) == ys).mean()),
                      val_top5=float(np.mean([y in np.argsort(-p)[:5] for p, y in zip(probs, ys)])),
                      train_seconds=time.time() - t0, history=res.history)
    torch.save(ck, out)
    return {"checkpoint": str(out), "classes": len(classes), "best_epoch": res.best_epoch,
            "val_top1": ck["meta"]["val_top1"], "val_top5": ck["meta"]["val_top5"], "temperature": T,
            "train_items": len(train_items), "val_items": len(val_items), "seconds": round(time.time() - t0, 1)}
