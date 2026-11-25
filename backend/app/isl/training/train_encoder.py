"""Metric-learning sign encoder training (docs/ISL_DESIGN.md 6.3 b).

Loss = SupCon + 0.5 x SubCenterArcFace over every training concept. Positives: INCLUDE clips of the same concept
(other sessions), cross-source pairs (INCLUDE <-> ISLRTC of one linked concept) and, for single-clip dictionary
concepts, two strong augmentations with different trim ends (P x K sampler with replacement).

Validation (model selection): retrieval recall@1 / @5 of INCLUDE validation clips against a gallery of ISLRTC
references plus INCLUDE training-clip centroids.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.features.spec import FeatureConfig
from app.isl.models.encoder import SignEncoder
from app.isl.models.losses import SubCenterArcFace, SupConLoss
from app.isl.training.common import ClipDataset, IslData, Item, PKSampler, items_for, set_threads
from app.models.checkpoint import frame_mask
from app.training.trainer import TrainConfig, fit, set_seed

ENCODER_VERSION = "isl-enc-1"


class _EncoderWithHead(nn.Module):
    def __init__(self, enc: SignEncoder, arc: SubCenterArcFace):
        super().__init__()
        self.enc, self.arc = enc, arc


@torch.no_grad()
def embed_items(enc: SignEncoder, ds: ClipDataset, fcfg: FeatureConfig, batch: int = 128) -> np.ndarray:
    enc.eval()
    out = []
    for a in range(0, len(ds), batch):
        x = torch.stack([torch.from_numpy(ds.features(i)) for i in range(a, min(len(ds), a + batch))])
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


def train_encoder(data: IslData, out: Path, fcfg: FeatureConfig, cfg: TrainConfig, *, emb_dim: int = 256,
                  p: int = 32, k: int = 2, arc_weight: float = 0.5, exclude_concepts: set[str] | None = None,
                  smoke: bool = False, threads: int = 2, batches_per_epoch: int | None = None, progress=None) -> dict:
    set_threads(threads)
    set_seed(cfg.seed)
    t0 = time.time()
    excl = exclude_concepts or set()
    train_inc = [c for c in data.include_part("train") if c.concept_id not in excl]
    concepts = sorted({c.concept_id for c in train_inc} | {c.concept_id for c in data.islrtc})
    label_of = {c: i for i, c in enumerate(concepts)}
    train_items: list[Item] = items_for(train_inc, label_of, fcfg) + items_for(data.islrtc, label_of, fcfg)
    if len(train_items) < 4:
        raise RuntimeError("not enough clips to train the encoder")
    ds = ClipDataset(train_items, fcfg, train=True, seed=cfg.seed)
    sampler = PKSampler([it.label for it in train_items], p=p, k=k, batches=batches_per_epoch, seed=cfg.seed)
    loader = DataLoader(ds, batch_sampler=sampler)
    enc = SignEncoder(fcfg.dim, hidden=cfg.hidden, layers=cfg.layers, emb_dim=emb_dim, dropout=cfg.dropout)
    arc = SubCenterArcFace(emb_dim, len(concepts), k=3)
    both = _EncoderWithHead(enc, arc)
    supcon = SupConLoss(0.1)

    def loss_fn(m, x, y):
        z = m.enc(x, frame_mask(x, fcfg))
        return supcon(z, y) + arc_weight * m.arc(z, y), None

    # validation gallery: ISLRTC references + INCLUDE training centroids; queries: INCLUDE val clips
    val_items = items_for([c for c in data.include_part("val") if c.concept_id not in excl], label_of | {
        c.concept_id: -1 for c in data.include_part("val") if c.concept_id not in label_of}, fcfg, with_alternatives=False)
    q_ds = ClipDataset(val_items, fcfg, train=False)
    g_items = items_for(data.islrtc, label_of, fcfg, with_alternatives=False) + items_for(train_inc, label_of, fcfg,
                                                                                         with_alternatives=False)
    g_ds = ClipDataset(g_items, fcfg, train=False)

    def val_fn(m) -> dict:
        q = embed_items(m.enc, q_ds, fcfg)
        g = embed_items(m.enc, g_ds, fcfg)
        return retrieval_metrics(q, [it.clip.concept_id for it in val_items], g, [it.clip.concept_id for it in g_items])

    model_kwargs = {"hidden": cfg.hidden, "layers": cfg.layers, "emb_dim": emb_dim, "dropout": cfg.dropout}
    meta = {"kind": "isl-encoder", "encoder_version": ENCODER_VERSION, "model_kwargs": model_kwargs,
            "feature_config": fcfg.to_dict(), "profile": data.profile, "language": "isl", "smoke": smoke,
            "train_config": asdict(cfg), "excluded_concepts": sorted(excl), "pretraining": None,
            "data": {**data.summary(), "train_items": len(train_items), "concepts": len(concepts),
                     "val_queries": len(val_items), "gallery": len(g_items)}}
    res = fit(both, loader, val_fn, loss_fn, cfg, out, meta, select_key="val_r1", save_module=enc,
              steps_per_epoch=len(sampler), progress=progress)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    best = next((h for h in res.history if h["epoch"] == res.best_epoch), {})
    ck["meta"].update(val_r1=best.get("val_r1"), val_r5=best.get("val_r5"), history=res.history,
                      train_seconds=time.time() - t0)
    torch.save(ck, out)
    return {"checkpoint": str(out), "concepts": len(concepts), "train_items": len(train_items),
            "best_epoch": res.best_epoch, "val_r1": best.get("val_r1"), "val_r5": best.get("val_r5"),
            "val_queries": len(val_items), "seconds": round(time.time() - t0, 1)}
