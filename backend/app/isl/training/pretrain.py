"""Self-supervised pre-training of the sign encoder on unlabelled ISL pose data (docs/ISL_DESIGN.md 15.2).

Labelled ISL words are scarce (INCLUDE ~4k clips, ISLRTC one clip per word), while unlabelled ISL pose data is
plentiful (GloSL-ISL ~127k sentence clips, OpenHands ~1,100 h, ISL-CSLTR sentences, WSLP unlabelled clips, and
the word clips themselves without their labels). The encoder trunk is pre-trained on all of it with two
objectives that fit a CPU budget:

* **temporal contrastive (NT-Xent)**: two crops of the same sequence that are close in time (overlapping or
  adjacent, 1-4 s each) are positives, every other crop in the batch is a negative; each crop gets the strong
  signer augmentation (body proportions, rotation, scale, speed, hand dropout). For short word clips both views
  are the whole clip with different augmentations (SimCLR-style).
* **masked-frame modelling**: about 25 % of the frames of view A (spans of 3) are replaced by a learned mask
  token after the input projection; a linear head reconstructs their feature vectors from the BiGRU states
  (MSE on masked frames that have a hand).

Loss = NT-Xent(A_masked, B) + ``mfm_weight`` x MFM. The encoder is the same ``SignEncoder`` that fine-tuning
uses, saved as an ordinary encoder checkpoint (``meta.pretraining`` records sources, clips, steps) that
``train_encoder(init_from=...)`` warm-starts from. Labels are never read. Evaluation clips are excluded by the
caller (``multisource.pretrain_clips(exclude_ids=...)``), so test videos are not seen even without labels.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.features.build import FeatureError, featurize, prepare_body_frame
from app.features.spec import FeatureConfig
from app.isl.augment import StrongAugmentConfig, augment_strong
from app.isl.data.sources import Clip
from app.isl.models.encoder import SignEncoder
from app.isl.training.loop import LoopConfig, atomic_save, run_loop
from app.models.checkpoint import frame_mask

log = logging.getLogger(__name__)

PRETRAIN_VERSION = "isl-enc-pre-1"


@dataclass
class PretrainConfig:
    hidden: int = 192
    layers: int = 2
    emb_dim: int = 256
    dropout: float = 0.2
    batch: int = 48
    epochs: int = 20
    steps_per_epoch: int = 400
    lr: float = 1e-3
    temperature: float = 0.1
    mfm_weight: float = 1.0
    mask_ratio: float = 0.25
    mask_span: int = 3
    win_min_s: float = 1.0
    win_max_s: float = 4.0
    max_offset_frac: float = 0.5
    val_frac: float = 0.02
    val_batches: int = 4
    cache_items: int = 2048  # LRU of loaded sequences (each well under 1 MB as float32 crops)
    seed: int = 0


class _Pretrainer(nn.Module):
    def __init__(self, enc: SignEncoder, in_dim: int, hidden: int):
        super().__init__()
        self.enc = enc
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.recon = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, in_dim))


def nt_xent(za: torch.Tensor, zb: torch.Tensor, temperature: float) -> tuple[torch.Tensor, float]:
    """Symmetric InfoNCE over 2B unit vectors; returns the loss and the view-matching top-1 accuracy."""
    z = torch.cat([za, zb], 0)
    s = z @ z.T / temperature
    n = za.shape[0]
    s = s.masked_fill(torch.eye(2 * n, dtype=torch.bool), -1e4)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    loss = F.cross_entropy(s, target)
    acc = float((s.argmax(1) == target).float().mean())
    return loss, acc


def span_mask(valid: np.ndarray, ratio: float, span: int, rng: np.random.Generator) -> np.ndarray:
    """(B, T) bool: spans of ``span`` frames covering about ``ratio`` of each row's valid frames."""
    B, T = valid.shape
    out = np.zeros((B, T), bool)
    for b in range(B):
        idx = np.flatnonzero(valid[b])
        if len(idx) < 4:
            continue
        n_spans = max(1, int(round(ratio * len(idx) / span)))
        for s in rng.choice(idx, size=min(n_spans, len(idx)), replace=False):
            out[b, s:s + span] = True
        out[b] &= valid[b]
    return out


class ViewSampler:
    """Two augmented, temporally close crops of one sequence -> (2, T, D) features."""

    def __init__(self, clips: list[Clip], fcfg: FeatureConfig, cfg: PretrainConfig, aug: StrongAugmentConfig | None = None):
        self.clips, self.fcfg, self.cfg = clips, fcfg, cfg
        self.aug = aug or StrongAugmentConfig()
        self._cache: OrderedDict = OrderedDict()
        self.failures = Counter()

    def _raw(self, i: int):
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        raw = self.clips[i].load()
        self._cache[i] = raw
        if len(self._cache) > self.cfg.cache_items:
            self._cache.popitem(last=False)
        return raw

    def _feat(self, raw, a: int, b: int, rng: np.random.Generator) -> np.ndarray:
        bf = prepare_body_frame(raw.slice(a, b), self.fcfg)
        return featurize(augment_strong(bf, rng, self.aug), self.fcfg)

    def pair(self, i: int, rng: np.random.Generator) -> np.ndarray | None:
        try:
            raw = self._raw(i)
        except (OSError, ValueError, KeyError) as e:
            self.failures[type(e).__name__] += 1
            return None
        n = len(raw)
        if n < 6:
            self.failures["short"] += 1
            return None
        t = np.asarray(raw.t_ms, np.float64)
        fps = (n - 1) * 1000.0 / max(1.0, t[-1] - t[0]) if n > 1 else 12.0
        w = int(round(rng.uniform(self.cfg.win_min_s, self.cfg.win_max_s) * fps))
        try:
            if n <= w + 2:  # a word clip: two augmentations of the whole clip
                return np.stack([self._feat(raw, 0, n, rng), self._feat(raw, 0, n, rng)])
            a = int(rng.integers(0, n - w))
            off = int(rng.integers(-int(self.cfg.max_offset_frac * w), int(self.cfg.max_offset_frac * w) + 1))
            b = min(max(0, a + off), n - w)
            return np.stack([self._feat(raw, a, a + w, rng), self._feat(raw, b, b + w, rng)])
        except FeatureError:
            self.failures["no_body"] += 1
            return None


def pretrain_encoder(clips: list[Clip], out: Path, fcfg: FeatureConfig, cfg: PretrainConfig, *, profile: str,
                     state_path: Path | None = None, max_minutes: float = 0.0, fresh: bool = False,
                     threads: int = 2, smoke: bool = False, progress=None) -> dict:
    torch.set_num_threads(max(1, threads))
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    if len(clips) < 4:
        raise RuntimeError(f"pre-training needs unlabelled clips; found {len(clips)}")
    order = list(range(len(clips)))
    random.Random(cfg.seed).shuffle(order)
    n_val = max(2, int(cfg.val_frac * len(clips))) if len(clips) >= 20 else 2
    val_idx, train_idx = order[:n_val], order[n_val:]
    # sample sources evenly in proportion to sqrt(size) so one huge corpus does not drown the others
    src_of = [clips[i].source for i in train_idx]
    cnt = Counter(src_of)
    wts = np.array([1.0 / np.sqrt(cnt[s]) for s in src_of])
    cum = np.cumsum(wts / wts.sum())
    sampler = ViewSampler(clips, fcfg, cfg)
    enc = SignEncoder(fcfg.dim, hidden=cfg.hidden, layers=cfg.layers, emb_dim=cfg.emb_dim, dropout=cfg.dropout)
    model = _Pretrainer(enc, fcfg.dim, cfg.hidden)
    flags = fcfg.dim - fcfg.flag_dim

    def make_batch(idx: list[int], rng: np.random.Generator):
        pairs = []
        for i in idx:
            p = sampler.pair(i, rng)
            tries = 0
            while p is None and tries < 5:
                p = sampler.pair(int(rng.integers(len(clips))), rng)
                tries += 1
            if p is not None:
                pairs.append(p)
        if len(pairs) < 2:
            return None
        x = torch.from_numpy(np.stack(pairs).astype(np.float32))  # (B, 2, T, D)
        return x[:, 0], x[:, 1]

    def batches(epoch: int):
        rng = random.Random(f"{cfg.seed}:{epoch}")
        for s in range(cfg.steps_per_epoch):
            idx = [train_idx[min(len(train_idx) - 1, int(np.searchsorted(cum, rng.random())))] for _ in range(cfg.batch)]
            yield (epoch, s, idx)

    def forward(m: _Pretrainer, xa: torch.Tensor, xb: torch.Tensor, rng: np.random.Generator):
        ma, mb = frame_mask(xa, fcfg), frame_mask(xb, fcfg)
        hand = ((xa[..., flags] > 0.5) | (xa[..., flags + 1] > 0.5)).numpy()
        rep = torch.from_numpy(span_mask(hand, cfg.mask_ratio, cfg.mask_span, rng))
        za, ha = m.enc.encode(xa, ma, rep, m.mask_token)
        zb, _ = m.enc.encode(xb, mb)
        l_c, acc = nt_xent(za, zb, cfg.temperature)
        if rep.any():
            pred = m.recon(ha[rep])
            l_m = F.mse_loss(pred, xa[rep])
        else:
            l_m = torch.zeros(())
        return l_c, l_m, acc

    def loss_fn(m, item):
        epoch, s, idx = item
        rng = np.random.default_rng([cfg.seed, epoch, s])
        b = make_batch(idx, rng)
        if b is None:
            return torch.zeros((), requires_grad=True)
        l_c, l_m, _ = forward(m, *b, rng)
        return l_c + cfg.mfm_weight * l_m

    val_cache: list = []

    def val_fn(m) -> dict:
        m.eval()
        rng = np.random.default_rng([cfg.seed, 99])
        if not val_cache:
            for v in range(cfg.val_batches):
                idx = [val_idx[(v * cfg.batch + j) % len(val_idx)] for j in range(min(cfg.batch, max(2, len(val_idx))))]
                b = make_batch(idx, rng)
                if b is not None:
                    val_cache.append(b)
        lc, lm, accs = [], [], []
        with torch.no_grad():
            for xa, xb in val_cache:
                c, mm, a = forward(m, xa, xb, np.random.default_rng([cfg.seed, 7]))
                lc.append(float(c))
                lm.append(float(mm))
                accs.append(a)
        if not lc:
            return {"val": 0.0}
        return {"val_contrastive": float(np.mean(lc)), "val_mfm": float(np.mean(lm)), "val_view_acc": float(np.mean(accs)),
                "val": -(float(np.mean(lc)) + cfg.mfm_weight * float(np.mean(lm)))}

    sources = dict(Counter(c.source for c in clips))
    meta = {"kind": "isl-encoder", "encoder_version": PRETRAIN_VERSION, "feature_config": fcfg.to_dict(),
            "model_kwargs": {"hidden": cfg.hidden, "layers": cfg.layers, "emb_dim": cfg.emb_dim, "dropout": cfg.dropout},
            "profile": profile, "language": "isl", "smoke": smoke,
            "pretraining": {"objective": "temporal NT-Xent + masked-frame modelling", "config": asdict(cfg),
                            "sources": sources, "clips": len(clips), "labels_used": False}}

    def save_best(epoch: int, row: dict) -> None:
        atomic_save({"state_dict": enc.state_dict(), "meta": {**meta, "best_epoch": epoch, "val": row}}, out)

    lc = LoopConfig(epochs=cfg.epochs, steps_per_epoch=cfg.steps_per_epoch, lr=cfg.lr, patience=max(3, cfg.epochs),
                    max_minutes=max_minutes, seed=cfg.seed,
                    extra={"pretrain": asdict(cfg), "clips": len(clips), "dim": fcfg.dim})
    res = run_loop(model, batches, loss_fn, lc, state_path or out.with_suffix(".state.pt"), val_fn=val_fn,
                   select_key="val", save_best=save_best, fresh=fresh,
                   progress=(lambda m: progress(m)) if progress else None)
    if out.exists():
        ck = torch.load(out, map_location="cpu", weights_only=False)
        ck["meta"]["history"] = res.history
        ck["meta"]["pretraining"]["steps_done"] = sum(1 for _ in res.history) * cfg.steps_per_epoch
        atomic_save(ck, out)
    return {"checkpoint": str(out), **res.to_json(), "clips": len(clips), "sources": sources,
            "val_clips": len(val_idx), "load_failures": dict(sampler.failures),
            "last": res.history[-1] if res.history else None, "seconds": round(time.time() - t0, 1)}
