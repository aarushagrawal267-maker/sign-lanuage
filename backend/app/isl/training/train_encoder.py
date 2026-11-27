"""Metric-learning sign encoder training (docs/ISL_DESIGN.md 6.3 b and 15.2).

Loss = SupCon + 0.5 x SubCenterArcFace over every training concept. Positives: INCLUDE clips of the same concept
(other sessions), cross-source pairs (INCLUDE <-> ISLRTC <-> ISL500 / WSLP / CISLR / Mendeley of one linked
concept) and, for single-clip dictionary concepts, two strong augmentations with different trim ends (P x K
sampler with replacement; within a concept, clips are drawn by per-source weight).

Phase 2 additions: ``init_from`` warm-starts the trunk from the self-supervised pre-training checkpoint
(``pretrain.py``); ``data.extra`` (every other labelled source of the profile) joins the training set;
``exclude_concepts`` removes **every** clip of those concepts from the supervised losses, ISLRTC included (the
held-out-concept one-shot protocol E2); ``exclude_sources`` removes whole sources from supervision (research: WSLP, a
second view of the ISLRTC dictionary videos whose "pairs" with ISLRTC teach nothing, docs/ISL_DESIGN.md 15.32);
training runs in the resumable loop (``loop.py``).

Validation (model selection): retrieval recall@1 / @5 of held-out INCLUDE (+ signer-held-out extra) validation
clips against a gallery of ISLRTC references plus a few training clips per concept and source.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.features.spec import FeatureConfig
from app.isl.models.encoder import SignEncoder, load_encoder_weights
from app.isl.models.losses import SubCenterArcFace, SupConLoss
from app.isl.training.common import ClipDataset, IslData, LazyClipDataset, class_space_of, set_threads
from app.isl.training.loop import (LoopConfig, atomic_save, clips_digest, digest, file_digest, pk_batches, run_loop,
                                   train_config_signature)
from app.isl.training.multisource import SAMPLE_WEIGHTS
from app.models.checkpoint import frame_mask
from app.training.trainer import TrainConfig, set_seed

log = logging.getLogger(__name__)

ENCODER_VERSION = "isl-enc-1"


def feature_config_diff(init_fc: dict | None, fcfg: FeatureConfig) -> dict:
    """Fields where a warm-start checkpoint's feature config differs from ``fcfg``: ``{field: [init, current]}``.
    Both sides are compared with their defaults filled in (``to_dict`` omits e.g. a zero ``hand_size_floor``), so an
    old checkpoint without a field counts as having the default. ``init_fc is None`` (not recorded) gives ``{}``."""
    if init_fc is None:
        return {}
    a, b = asdict(FeatureConfig.from_dict(init_fc)), asdict(fcfg)
    return {k: [a[k], b[k]] for k in b if a.get(k) != b[k]}


class _EncoderWithHead(nn.Module):
    def __init__(self, enc: SignEncoder, arc: SubCenterArcFace):
        super().__init__()
        self.enc, self.arc = enc, arc


@torch.no_grad()
def embed_items(enc: SignEncoder, ds, fcfg: FeatureConfig, batch: int = 128) -> np.ndarray:
    enc.eval()
    out = []
    for a in range(0, len(ds), batch):
        x = torch.stack([torch.from_numpy(ds.features(i)) for i in range(a, min(len(ds), a + batch))])
        out.append(enc(x, frame_mask(x, fcfg)).numpy())
    return np.concatenate(out) if out else np.zeros((0, enc.emb_dim), np.float32)


@torch.no_grad()
def embed_array(enc: SignEncoder, feats: np.ndarray, fcfg: FeatureConfig, batch: int = 128) -> np.ndarray:
    enc.eval()
    out = []
    for a in range(0, len(feats), batch):
        x = torch.from_numpy(np.asarray(feats[a:a + batch], np.float32))
        out.append(enc(x, frame_mask(x, fcfg)).numpy())
    return np.concatenate(out) if out else np.zeros((0, enc.emb_dim), np.float32)


def retrieval_metrics(q: np.ndarray, q_lab: list, g: np.ndarray, g_lab: list) -> dict:
    if len(q) == 0 or len(g) == 0:
        return {"val_r1": 0.0, "val_r5": 0.0, "n": 0}
    labs = sorted(set(g_lab))
    pos = {c: i for i, c in enumerate(labs)}
    gi = np.array([pos[c] for c in g_lab])
    s = q @ g.T
    cs = np.full((len(q), len(labs)), -2.0, np.float32)
    for b in range(len(q)):
        np.maximum.at(cs[b], gi, s[b])
    order = np.argsort(-cs, axis=1)
    r1 = r5 = 0
    for b, lab in enumerate(q_lab):
        top = [labs[j] for j in order[b, :5]]
        r1 += int(top[0] == lab)
        r5 += int(lab in top)
    return {"val_r1": r1 / len(q), "val_r5": r5 / len(q), "n": len(q)}


def _features_cache(clips, fcfg) -> tuple[np.ndarray, list]:
    ds = LazyClipDataset(clips, [0] * len(clips), fcfg, train=False, cache_items=0)
    feats, kept = [], []
    for i, c in enumerate(clips):
        f = ds.features(i)
        if f is not None:
            feats.append(f.astype(np.float16))
            kept.append(c)
    return (np.stack(feats) if feats else np.zeros((0, fcfg.seq_len, fcfg.dim), np.float16)), kept


def encoder_training_clips(data: IslData, exclude: set[str], use_extra: bool = True,
                           exclude_sources: tuple[str, ...] | list[str] | set[str] = ()) -> list:
    """Supervised training clips: INCLUDE train + ISLRTC + (extra sources' train part), minus every clip of an
    excluded concept (E2 held-out concepts never reach a supervised loss, ISLRTC included) and every clip of an
    excluded source (``exclude_sources``, e.g. ``("wslp",)``, docs/ISL_DESIGN.md 15.32; default none)."""
    xs = set(exclude_sources or ())
    clips = [c for c in data.include_part("train") if c.concept_id not in exclude]
    clips += [c for c in data.islrtc if c.concept_id and c.concept_id not in exclude]
    if use_extra:
        clips += [c for c in data.extra_part("train") if c.concept_id not in exclude]
    return [c for c in clips if c.source not in xs] if xs else clips


# the core sources: the encoder's validation gallery is built from ISLRTC and its queries from INCLUDE, so leaving
# either out of supervision gives a validation that does not match the training list (docs 15.33)
CORE_SOURCES = ("include", "islrtc")


def check_exclude_sources(names) -> list[str]:
    """The sorted source names of ``exclude_sources``; ``ValueError`` for a name that is no word source of any profile
    (a typo such as ``wlsp`` would otherwise train that source in silently) or for a core source (docs 15.33)."""
    from app.isl.profiles import WORD_SOURCES

    xs = sorted(set(names or ()))
    known = {n for v in WORD_SOURCES.values() for n in v}
    unknown = [n for n in xs if n not in known]
    if unknown:
        raise ValueError(f"unknown source(s) {', '.join(unknown)}; word sources: "
                         f"{', '.join(sorted(known - set(CORE_SOURCES)))}")
    core = [n for n in xs if n in CORE_SOURCES]
    if core:
        raise ValueError(f"{', '.join(core)} cannot be left out of encoder supervision (core source: the validation "
                         "gallery and queries are built from it)")
    return xs


def encoder_excluded_sources(meta: dict | None) -> list[str]:
    """The sources an encoder checkpoint was trained without (``[]`` for one written before 15.32)."""
    return class_space_of(meta)["excluded_sources"]


def train_encoder(data: IslData, out: Path, fcfg: FeatureConfig, cfg: TrainConfig, *, emb_dim: int = 256,
                  p: int = 32, k: int = 2, arc_weight: float = 0.5, exclude_concepts: set[str] | None = None,
                  smoke: bool = False, threads: int = 2, batches_per_epoch: int | None = None, progress=None,
                  init_from: Path | None = None, use_extra: bool = True, source_weights: dict | None = None,
                  resume: bool = False, max_minutes: float = 0.0, state_path: Path | None = None,
                  max_val_queries: int = 1500, max_gallery: int = 8000, per_concept_gallery: int = 2,
                  version: str = ENCODER_VERSION, cache_items: int = 12000,
                  exclude_sources: tuple[str, ...] | list[str] = ()) -> dict:
    set_threads(threads)
    set_seed(cfg.seed)
    t0 = time.time()
    excl = set(exclude_concepts or ())
    xsrc = check_exclude_sources(exclude_sources)
    train_clips = encoder_training_clips(data, excl, use_extra, xsrc)
    concepts = sorted({c.concept_id for c in train_clips})
    label_of = {c: i for i, c in enumerate(concepts)}
    if len(train_clips) < 4 or len(concepts) < 2:
        raise RuntimeError("not enough clips to train the encoder")
    labels = [label_of[c.concept_id] for c in train_clips]
    sw = {**SAMPLE_WEIGHTS, **(source_weights or {})}
    weights = [float(sw.get(c.source, 1.0)) for c in train_clips]
    ds = LazyClipDataset(train_clips, labels, fcfg, train=True, seed=cfg.seed, cache_items=cache_items)
    n_batches = batches_per_epoch or max(1, len(train_clips) // max(1, p * k))
    enc = SignEncoder(fcfg.dim, hidden=cfg.hidden, layers=cfg.layers, emb_dim=emb_dim, dropout=cfg.dropout)
    init = load_encoder_weights(enc, Path(init_from)) if init_from else None
    init_fc = init["meta"].get("feature_config") if init else None
    init_fc_diff = feature_config_diff(init_fc, fcfg)
    if init_fc_diff:  # accepted as a warm start (e.g. a pretrain without the hand-size floor), but made visible
        log.warning("warm start %s was trained with another feature config (field: [init, current]) %s; "
                    "kept as the init, recorded in meta pretraining.init_feature_config (docs/ISL_DESIGN.md 15.15)",
                    init_from, init_fc_diff)
    arc = SubCenterArcFace(emb_dim, len(concepts), k=3)
    both = _EncoderWithHead(enc, arc)
    supcon = SupConLoss(0.1)

    def batches(epoch: int):
        for s, idx in enumerate(pk_batches(labels, p, k, n_batches, cfg.seed, epoch, weights)):
            yield (epoch, s, idx)

    def loss_fn(m, item):
        epoch, s, idx = item
        b = ds.batch(idx, np.random.default_rng([cfg.seed, epoch, s]))
        if b is None:
            return torch.zeros((), requires_grad=True)
        x, y = b
        z = m.enc(x, frame_mask(x, fcfg))
        return supcon(z, y) + arc_weight * m.arc(z, y)

    # validation: queries = held-out INCLUDE / extra clips; gallery = ISLRTC + a few training clips per concept
    rng = random.Random(cfg.seed)
    vq = [c for c in data.labelled_part("val") if c.concept_id not in excl and c.source not in xsrc]
    if len(vq) > max_val_queries:
        vq = rng.sample(vq, max_val_queries)
    gal = [c for c in data.islrtc if c.concept_id in label_of]
    if len(gal) > max_gallery:
        gal = rng.sample(gal, max_gallery)
    per = defaultdict(list)
    for c in train_clips:
        if c.source != "islrtc":
            per[(c.concept_id, c.source)].append(c)
    for v in per.values():
        gal += rng.sample(v, min(per_concept_gallery, len(v)))
    q_feats, vq = _features_cache(vq, fcfg)
    g_feats, gal = _features_cache(gal, fcfg)
    q_lab = [c.concept_id for c in vq]
    g_lab = [c.concept_id for c in gal]

    def val_fn(m) -> dict:
        if not len(q_feats) or not len(g_feats):
            return {"val_r1": 0.0, "val_r5": 0.0}
        return retrieval_metrics(embed_array(m.enc, q_feats, fcfg), q_lab, embed_array(m.enc, g_feats, fcfg), g_lab)

    model_kwargs = {"hidden": cfg.hidden, "layers": cfg.layers, "emb_dim": emb_dim, "dropout": cfg.dropout}
    src_counts = defaultdict(int)
    for c in train_clips:
        src_counts[c.source] += 1
    # the supervised class space, explicit (docs 15.32): encoders trained without a source (or on other concepts)
    # are never mixed silently; E2 reads excluded_sources to rebuild the exact training list of a fold encoder
    class_space = {"excluded_sources": xsrc, "use_extra": use_extra, "n_concepts": len(concepts),
                   "concepts_sha256": digest(concepts)}
    meta = {"kind": "isl-encoder", "encoder_version": version, "model_kwargs": model_kwargs,
            "feature_config": fcfg.to_dict(), "profile": data.profile, "language": "isl", "smoke": smoke,
            "train_config": asdict(cfg), "excluded_concepts": sorted(excl), "class_space": class_space,
            "pretraining": ({"init_from": init["path"], "loaded_tensors": init["loaded"], "skipped": init["skipped"],
                             **(init["meta"].get("pretraining") or {}),
                             "init_feature_config": init_fc, "init_feature_config_diff": init_fc_diff}
                            if init else None),
            "data": {**data.summary(), "train_items": len(train_clips), "concepts": len(concepts),
                     "train_sources": dict(src_counts), "val_queries": len(vq), "gallery": len(gal)}}

    def save_best(epoch: int, row: dict) -> None:
        atomic_save({"state_dict": enc.state_dict(), "meta": {**meta, "best_epoch": epoch}}, out)

    lc = LoopConfig(epochs=cfg.epochs, steps_per_epoch=n_batches, lr=cfg.lr, weight_decay=cfg.weight_decay,
                    patience=cfg.patience, max_minutes=max_minutes, seed=cfg.seed,
                    extra={"kind": "encoder", "model": model_kwargs, "p": p, "k": k, "concepts": len(concepts),
                           "items": len(train_clips), "excluded": len(excl), "init": str(init_from or ""),
                           "features": fcfg.to_dict(),
                           # every option that changes what is trained or which epoch is kept (docs 15.19)
                           "train": train_config_signature(cfg), "split": data.split.get("name"),
                           "version": version, "arc_weight": arc_weight, "use_extra": use_extra,
                           "source_weights": sw, "excluded_concepts": digest(sorted(excl)),
                           "init_sha": file_digest(init_from), "data": clips_digest(train_clips, labels, weights),
                           "val": [max_val_queries, max_gallery, per_concept_gallery, clips_digest(vq),
                                   clips_digest(gal)],
                           # only when set, so a state of the default settings written before 15.32 still resumes
                           **({"exclude_sources": xsrc} if xsrc else {})})
    res = run_loop(both, batches, loss_fn, lc, state_path or Path(str(out) + ".state"), val_fn=val_fn,
                   select_key="val_r1", save_best=save_best, fresh=not resume,
                   progress=(lambda m: progress(None, m)) if progress else None)
    if not out.exists():
        raise RuntimeError("encoder training produced no checkpoint")
    ck = torch.load(out, map_location="cpu", weights_only=False)
    best = next((h for h in res.history if h["epoch"] == res.best_epoch), {})
    ck["meta"].update(val_r1=best.get("val_r1"), val_r5=best.get("val_r5"), history=res.history,
                      train_seconds=time.time() - t0, step_seconds=res.step_seconds, finished=res.finished)
    atomic_save(ck, out)
    return {"checkpoint": str(out), "concepts": len(concepts), "train_items": len(train_clips),
            "train_sources": dict(src_counts), "best_epoch": res.best_epoch, "val_r1": best.get("val_r1"),
            "val_r5": best.get("val_r5"), "val_queries": len(vq), "finished": res.finished,
            "resumed_from": res.resumed_from, "step_seconds": res.step_seconds, "pretrained": bool(init),
            "class_space": class_space,
            "seconds": round(time.time() - t0, 1)}


__all__ = ["ENCODER_VERSION", "ClipDataset", "embed_items", "retrieval_metrics", "train_encoder",
           "encoder_training_clips", "encoder_excluded_sources", "check_exclude_sources"]
