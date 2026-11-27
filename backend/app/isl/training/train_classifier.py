"""Closed-set core classifier for ISL (docs/ISL_DESIGN.md 6.3 a and 15.2): the existing ``RecurrentClassifier``
trained on every labelled word clip of the profile with strong augmentation. The checkpoint uses the ASL format,
so ``Predictor`` loads it unchanged (class names = concept ids).

Classes: INCLUDE concepts with at least ``min_clips`` training clips (as before) plus, with ``data.extra``
(ISL500, WSLP, CISLR, Mendeley ...), every concept that has at least ``min_signers`` different signer groups and
``min_clips`` clips across the training sources (ISLRTC counts as one signer; with ``dictionary_group`` ISLRTC, WSLP
and CISLR together count as one signer, docs/ISL_DESIGN.md 15.32). Batches are class-balanced
(class drawn with probability ~ count ** ``class_power``; inside a class, clips by per-source weight); ISLRTC clips
of the classes join training with random trim ends. Selection on the validation part (macro F1), then
temperature scaling on validation. Runs in the resumable loop (``loop.py``).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.evaluation.metrics import confidence_curve, pick_threshold
from app.features.spec import FeatureConfig
from app.isl.models.calibration import fit_temperature
from app.isl.training.common import (SIGNER_GROUPING_DICTIONARY, SIGNER_GROUPING_SOURCE, IslData, LazyClipDataset,
                                     multi_signer_counts, set_threads)
from app.isl.training.loop import (LoopConfig, atomic_save, balanced_batches, clips_digest, digest, run_loop,
                                   train_config_signature)
from app.isl.training.multisource import SAMPLE_WEIGHTS
from app.models.checkpoint import frame_mask
from app.models.networks import build_model
from app.training.trainer import TrainConfig, set_seed


def classifier_classes(data: IslData, min_clips: int = 2, min_signers: int = 2, use_extra: bool = True,
                       dictionary_group: bool = False) -> list[str]:
    """``dictionary_group``: ISLRTC, WSLP and CISLR clips are one signer group (docs 15.32), so a concept seen only in
    the dictionary and its second views (research: about 3,770 of 4,354 classes) is no class; it is served by
    retrieval. Default False: the grouping before 15.32 (every WSLP / CISLR clip without a signer id is a group)."""
    train_inc = data.include_part("train")
    counts = Counter(c.concept_id for c in train_inc)
    classes = {c for c, n in counts.items() if n >= min_clips}
    if use_extra and data.extra:
        pool = train_inc + data.extra_part("train") + [c for c in data.islrtc if c.concept_id]
        groups = multi_signer_counts(pool, dictionary_group)
        n = Counter(c.concept_id for c in pool)
        classes |= {c for c, g in groups.items() if len(g) >= min_signers and n[c] >= max(min_clips, 2)
                    and any(x.concept_id == c for x in data.extra_part("train"))}
    return sorted(classes)


def train_classifier(data: IslData, out: Path, fcfg: FeatureConfig, cfg: TrainConfig, *, min_clips: int = 2,
                     islrtc_oversample: int = 4, smoke: bool = False, threads: int = 2, progress=None,
                     min_signers: int = 2, use_extra: bool = True, class_power: float = 0.5,
                     source_weights: dict | None = None, steps_per_epoch: int | None = None, resume: bool = False,
                     max_minutes: float = 0.0, state_path: Path | None = None, cache_items: int = 12000,
                     dictionary_group: bool = False) -> dict:
    set_threads(threads)
    set_seed(cfg.seed)
    t0 = time.time()
    classes = classifier_classes(data, min_clips, min_signers, use_extra, dictionary_group)
    if len(classes) < 2:
        raise RuntimeError(f"not enough training data: {len(classes)} concepts with >= {min_clips} clips")
    label_of = {c: i for i, c in enumerate(classes)}
    train_clips = [c for c in data.include_part("train") if c.concept_id in label_of]
    if use_extra:
        train_clips += [c for c in data.extra_part("train") if c.concept_id in label_of]
    isl_same = [c for c in data.islrtc if c.concept_id in label_of]
    sw = {**SAMPLE_WEIGHTS, **(source_weights or {})}
    # ISLRTC: one clip per entry, so it is weighted up (the old x4 oversampling) inside its class
    clips = train_clips + isl_same
    weights = [float(sw.get(c.source, 1.0)) * (islrtc_oversample if c.source == "islrtc" else 1.0) for c in clips]
    labels = [label_of[c.concept_id] for c in clips]
    ds = LazyClipDataset(clips, labels, fcfg, train=True, seed=cfg.seed, cache_items=cache_items)
    val_clips = [c for c in (data.include_part("val") + (data.extra_part("val") if use_extra else []))
                 if c.concept_id in label_of]
    vds = LazyClipDataset(val_clips, [label_of[c.concept_id] for c in val_clips], fcfg, train=False, cache_items=0)
    vx, vy = [], []
    for i in range(len(vds)):
        f = vds.features(i)
        if f is not None:
            vx.append(f)
            vy.append(vds.labels[i])
    if not vx:
        raise RuntimeError("no validation clips for the trained concepts")
    VX = torch.from_numpy(np.stack(vx).astype(np.float32))
    VY = np.array(vy)
    model = build_model(cfg.architecture, fcfg.dim, len(classes), **cfg.model_kwargs())
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    n_steps = steps_per_epoch or max(1, len(clips) // max(1, cfg.batch_size))

    def batches(epoch: int):
        for s, idx in enumerate(balanced_batches(labels, weights, cfg.batch_size, n_steps, cfg.seed, epoch, class_power)):
            yield (epoch, s, idx)

    def loss_fn(m, item):
        epoch, s, idx = item
        b = ds.batch(idx, np.random.default_rng([cfg.seed, epoch, s]))
        if b is None:
            return torch.zeros((), requires_grad=True)
        x, y = b
        return crit(m(x, frame_mask(x, fcfg)), y)

    @torch.no_grad()
    def logits_of(m) -> np.ndarray:
        m.eval()
        return np.concatenate([m(VX[a:a + 256], frame_mask(VX[a:a + 256], fcfg)).numpy() for a in range(0, len(VX), 256)])

    def val_fn(m) -> dict:
        from sklearn.metrics import f1_score

        lg = logits_of(m)
        pred = lg.argmax(1)
        return {"val_acc": float((pred == VY).mean()),
                "val_macro_f1": float(f1_score(VY, pred, labels=sorted(set(VY.tolist())), average="macro",
                                               zero_division=0))}

    src = Counter(c.source for c in clips)
    # the class space, explicit (docs 15.32): a classifier trained with another signer grouping has other classes,
    # and evaluation never pools it with this one (cv folds, E1a); older checkpoints have no field ("source")
    class_space = {"signer_grouping": SIGNER_GROUPING_DICTIONARY if dictionary_group else SIGNER_GROUPING_SOURCE,
                   "min_signers": min_signers, "min_clips": min_clips, "use_extra": use_extra,
                   "n_classes": len(classes), "classes_sha256": digest(classes)}
    meta = {"architecture": cfg.architecture, "model_kwargs": cfg.model_kwargs(), "feature_config": fcfg.to_dict(),
            "class_names": classes, "language": "isl", "profile": data.profile, "split": data.split.get("name"),
            "split_label": data.split.get("label"), "train_config": asdict(cfg), "smoke": smoke,
            "class_space": class_space,
            "data": {**data.summary(), "train_items": len(train_clips), "val_items": len(vx),
                     "islrtc_clips_of_classes": len(isl_same), "train_sources": dict(src),
                     "min_signers": min_signers, "class_power": class_power}}

    def save_best(epoch: int, row: dict) -> None:
        from app.models.checkpoint import save_checkpoint

        save_checkpoint(out, model, {**meta, "best_epoch": epoch})

    lc = LoopConfig(epochs=cfg.epochs, steps_per_epoch=n_steps, lr=cfg.lr, weight_decay=cfg.weight_decay,
                    patience=cfg.patience, max_minutes=max_minutes, seed=cfg.seed,
                    extra={"kind": "classifier", "model": cfg.model_kwargs(), "classes": len(classes),
                           "items": len(clips), "features": fcfg.to_dict(),
                           # every option that changes what is trained (docs/ISL_DESIGN.md 15.19): a finished or
                           # half-done state of other settings or other data is refused, never resumed
                           "train": train_config_signature(cfg), "split": data.split.get("name"),
                           "min_clips": min_clips, "min_signers": min_signers, "use_extra": use_extra,
                           "islrtc_oversample": islrtc_oversample, "class_power": class_power,
                           "source_weights": sw, "class_names": digest(classes),
                           "data": clips_digest(clips, labels, weights), "val": clips_digest(val_clips),
                           # only when on, so a state of the default settings written before 15.32 still resumes
                           **({"dictionary_group": True} if dictionary_group else {})})
    res = run_loop(model, batches, loss_fn, lc, state_path or Path(str(out) + ".state"), val_fn=val_fn,
                   select_key="val_macro_f1", save_best=save_best, fresh=not resume,
                   progress=(lambda m: progress(None, m)) if progress else None)
    # calibration on validation: temperature + the classic confidence threshold (best checkpoint)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    logits = logits_of(model)
    T = fit_temperature(logits, VY)
    z = logits / T
    probs = np.exp(z - z.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    curve = confidence_curve(probs, VY)
    ck["meta"].update(temperature=T, recommended_threshold=pick_threshold(curve),
                      val_top1=float((probs.argmax(1) == VY).mean()),
                      val_top5=float(np.mean([y in np.argsort(-p)[:5] for p, y in zip(probs, VY)])),
                      train_seconds=time.time() - t0, history=res.history, step_seconds=res.step_seconds,
                      finished=res.finished)
    atomic_save(ck, out)
    return {"checkpoint": str(out), "classes": len(classes), "best_epoch": res.best_epoch,
            "val_top1": ck["meta"]["val_top1"], "val_top5": ck["meta"]["val_top5"], "temperature": T,
            "train_items": len(train_clips), "val_items": len(vx), "train_sources": dict(src),
            "finished": res.finished, "step_seconds": res.step_seconds, "class_space": class_space,
            "seconds": round(time.time() - t0, 1)}
