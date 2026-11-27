"""Resumable CPU training loop for the long phase-2 runs (pre-training, encoder fine-tuning, classifier).

``trainer.fit`` (shared with ASL) restarts from scratch when a process dies. A multi-hour run on a shared,
busy machine has to survive restarts, so this loop:

* writes a **state file** (model, optimiser, scheduler, epoch, step inside the epoch, history, best score,
  early-stopping counter, torch RNG) atomically every ``save_every_s`` seconds and at the end of every epoch;
* on start, resumes from that state when its config signature matches (otherwise it refuses, so a changed
  command line never silently continues an incompatible run: delete the state file or pass ``fresh=True``);
* draws each epoch's batches from a generator seeded by ``(seed, epoch)``, so after a restart in the middle of
  an epoch the already-trained batches are skipped by index (no data loading) and the run continues;
* can stop cleanly after ``max_minutes`` (state saved, ``finished = False``) so a job can run in parts;
* saves the **best** module separately (``save_best``) whenever the selection metric improves.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class LoopConfig:
    epochs: int = 30
    steps_per_epoch: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_frac: float = 0.05
    patience: int = 10
    grad_clip: float = 1.0
    save_every_s: float = 300.0
    max_minutes: float = 0.0  # 0 = no time budget
    seed: int = 0
    extra: dict = field(default_factory=dict)  # anything else that must match for a resume (model size, data)

    def signature(self) -> str:
        d = asdict(self)
        for k in ("save_every_s", "max_minutes", "patience"):  # may change between parts of one run
            d.pop(k)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class LoopResult:
    finished: bool
    epoch: int
    best_epoch: int
    best_metric: float
    history: list[dict]
    resumed_from: dict | None
    seconds: float
    step_seconds: float | None  # mean wall time per optimisation step in this process

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "history"} | {"epochs_logged": len(self.history)}


def atomic_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


class StateMismatch(RuntimeError):
    pass


def digest(obj) -> str:
    """Short, stable hash of any JSON-able value (for ``LoopConfig.extra``)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def clips_digest(clips, labels=None, weights=None) -> str:
    """Hash of a training list in sampling order: clip id, trim slice, landmark file name, label and weight. Any
    change to the data a run trains on (another clip, re-trim, relink, re-weight) changes it, so a saved state of
    the old data is refused instead of resumed (docs/ISL_DESIGN.md 15.19)."""
    rows = []
    for i, c in enumerate(clips):
        r = [c.clip_id, c.start, c.end, Path(c.path).name]
        if labels is not None:
            r.append(labels[i])
        if weights is not None:
            r.append(round(float(weights[i]), 6))
        rows.append(r)
    return digest(rows)


def file_digest(path) -> str | None:
    """SHA-256 (first 16 hex) of a file, None when there is no file (e.g. a warm-start checkpoint)."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def train_config_signature(cfg) -> dict:
    """Every field of a shared ``TrainConfig`` that changes training (all but early-stopping patience, which may
    change between parts of one run, and the device)."""
    d = asdict(cfg)
    for k in ("patience", "device"):
        d.pop(k, None)
    return d


def run_loop(model: nn.Module, batches: Callable[[int], Iterator], loss_fn: Callable, cfg: LoopConfig,
             state_path: Path, *, val_fn: Callable[[nn.Module], dict] | None = None, select_key: str = "val",
             save_best: Callable[[int, dict], None] | None = None, params=None, fresh: bool = False,
             progress: Callable[[str], None] | None = None) -> LoopResult:
    """``batches(epoch)`` yields ``(x, y)`` (or any batch ``loss_fn`` accepts) for ``cfg.steps_per_epoch`` steps,
    deterministically per epoch (indices from a ``(seed, epoch)`` RNG). ``loss_fn(model, batch) -> loss``.
    ``val_fn(model) -> dict`` with ``select_key``; without it the training loss (negated) selects."""
    t_start = time.time()
    params = list(params) if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(1, cfg.epochs * cfg.steps_per_epoch)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    sig = cfg.signature()
    epoch, step, history, best, best_epoch, bad = 1, 0, [], -math.inf, 0, 0
    resumed = None
    state_path = Path(state_path)
    if state_path.exists() and not fresh:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        if st.get("signature") != sig:
            raise StateMismatch(f"{state_path} belongs to a run with other settings ({st.get('config')}); "
                                "delete it or pass --fresh to start over")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        epoch, step, history = st["epoch"], st["step"], st["history"]
        best, best_epoch, bad = st["best"], st["best_epoch"], st["bad"]
        torch.set_rng_state(st["torch_rng"])
        resumed = {"epoch": epoch, "step": step, "finished": st.get("finished", False)}
        if st.get("finished"):
            return LoopResult(True, epoch, best_epoch, best, history, resumed, 0.0, None)

    def save_state(finished: bool = False) -> None:
        atomic_save({"signature": sig, "config": asdict(cfg), "model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "epoch": epoch, "step": step, "history": history, "best": best,
                     "best_epoch": best_epoch, "bad": bad, "torch_rng": torch.get_rng_state(), "finished": finished,
                     "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}, state_path)

    last_save = time.time()
    deadline = t_start + cfg.max_minutes * 60 if cfg.max_minutes > 0 else None
    step_times: list[float] = []
    finished = False
    while epoch <= cfg.epochs:
        model.train()
        t_ep = time.time()
        losses = []
        it = batches(epoch)
        for i, batch in enumerate(it):
            if i >= cfg.steps_per_epoch:
                break
            if i < step:  # already trained before the restart (the generator only yields indices here)
                continue
            t0 = time.perf_counter()
            loss = loss_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            step = i + 1
            losses.append(float(loss.detach()))
            step_times.append(time.perf_counter() - t0)
            now = time.time()
            if now - last_save >= cfg.save_every_s:
                save_state()
                last_save = now
            if deadline is not None and now >= deadline:
                save_state()
                if progress:
                    progress(f"time budget reached at epoch {epoch} step {step}; state saved")
                return LoopResult(False, epoch, best_epoch, best, history, resumed, time.time() - t_start,
                                  float(np.mean(step_times)) if step_times else None)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t_ep, 1)}
        vm = val_fn(model) if val_fn is not None else {}
        row.update(vm)
        score = float(vm.get(select_key, -(row["train_loss"] or 0.0))) if vm else -(row["train_loss"] or 0.0)
        history.append(row)
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            if save_best is not None:
                save_best(epoch, row)
        else:
            bad += 1
        if progress:
            progress(f"epoch {epoch}/{cfg.epochs} loss {row['train_loss']} {select_key} {score:.4f}")
        epoch, step = epoch + 1, 0
        if bad >= cfg.patience:
            break
    finished = True
    save_state(finished=True)
    return LoopResult(finished, epoch - 1, best_epoch, best, history, resumed, time.time() - t_start,
                      float(np.mean(step_times)) if step_times else None)


# --------------------------------------------------------------------------------------------- samplers

def balanced_batches(labels: list[int], weights: list[float] | None, batch: int, n_batches: int, seed: int,
                     epoch: int, class_power: float = 0.5, *, groups: list[str] | None = None,
                     group_quota: dict[str, float] | None = None) -> Iterator[list[int]]:
    """Class-balanced batches: a class is drawn with probability proportional to ``count ** class_power``
    (0 = uniform over classes, 1 = natural frequency), then an item of it by its per-item weight
    (source weighting).

    With ``group_quota`` (and ``groups``, the quota group of every item; docs/ISL_DESIGN.md 15.38) the draw has two
    stages: first a group with probability ``group_quota[g]`` (renormalised over the groups that have items), then a
    class among that group's items by ``count_in_group ** class_power``, then an item of that class and group by its
    weight. Without it the draws are exactly the old ones."""
    if group_quota is not None:
        if groups is None or len(groups) != len(labels):
            raise ValueError("group_quota needs the quota group of every item")
        yield from _quota_batches(labels, weights, batch, n_batches, seed, epoch, class_power, groups, group_quota)
        return
    rng = random.Random(f"{seed}:{epoch}")
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    cls = sorted(by)
    cw = np.array([len(by[c]) ** class_power for c in cls], np.float64)
    cw /= cw.sum()
    cum = np.cumsum(cw)
    w = np.asarray(weights if weights is not None else np.ones(len(labels)), np.float64)
    for _ in range(n_batches):
        out = []
        for _ in range(batch):
            c = cls[min(len(cls) - 1, int(np.searchsorted(cum, rng.random())))]
            idx = by[c]
            ww = w[idx]
            out.append(idx[rng.choices(range(len(idx)), weights=ww)[0]] if len(idx) > 1 else idx[0])
        yield out


def _quota_batches(labels, weights, batch, n_batches, seed, epoch, class_power, groups, group_quota):
    rng = random.Random(f"{seed}:{epoch}")
    w = np.asarray(weights if weights is not None else np.ones(len(labels)), np.float64)
    by: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, (y, g) in enumerate(zip(labels, groups)):
        by[g][y].append(i)
    gnames = sorted(g for g in by if group_quota.get(g, 0.0) > 0)
    if not gnames:
        raise ValueError(f"source quota {group_quota}: no item in a group with a share > 0")
    gq = np.array([group_quota[g] for g in gnames], np.float64)
    gcum = np.cumsum(gq / gq.sum())
    plan = []  # per group: (classes, class cumsum, per class (items, cumulative item weights))
    for g in gnames:
        cls = sorted(by[g])
        cw = np.array([len(by[g][c]) ** class_power for c in cls], np.float64)
        items = [(by[g][c], list(np.cumsum(w[by[g][c]]))) for c in cls]
        plan.append((cls, np.cumsum(cw / cw.sum()), items))
    for _ in range(n_batches):
        out = []
        for _ in range(batch):
            cls, ccum, items = plan[min(len(gnames) - 1, int(np.searchsorted(gcum, rng.random())))]
            idx, cum = items[min(len(cls) - 1, int(np.searchsorted(ccum, rng.random())))]
            out.append(idx[rng.choices(range(len(idx)), cum_weights=cum)[0]] if len(idx) > 1 else idx[0])
        yield out


def expected_draw_shares(labels: list[int], weights: list[float] | None, keys: list[str], class_power: float = 0.5,
                         groups: list[str] | None = None, group_quota: dict[str, float] | None = None) -> dict:
    """Expected share of draws per ``keys`` value (e.g. the source of every item) under :func:`balanced_batches`
    with the same settings (for the run's meta: how often an INCLUDE clip is seen)."""
    w = np.asarray(weights if weights is not None else np.ones(len(labels)), np.float64)
    out: dict[str, float] = defaultdict(float)
    if group_quota is None:
        parts = {"": (1.0, list(range(len(labels))))}
    else:
        members = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        live = {g: group_quota.get(g, 0.0) for g in members if group_quota.get(g, 0.0) > 0}
        tot = sum(live.values())
        parts = {g: (q / tot, members[g]) for g, q in live.items()}
    for share, idx in parts.values():
        by = defaultdict(list)
        for i in idx:
            by[labels[i]].append(i)
        cw = {c: len(v) ** class_power for c, v in by.items()}
        ctot = sum(cw.values())
        for c, v in by.items():
            wt = w[v].sum()
            for i in v:
                out[keys[i]] += share * cw[c] / ctot * (w[i] / wt if wt > 0 else 1.0 / len(v))
    return {k: round(v, 6) for k, v in sorted(out.items())}


def pk_batches(labels: list[int], p: int, k: int, n_batches: int, seed: int, epoch: int,
               weights: list[float] | None = None, *, groups: list[str] | None = None,
               bridge: "BridgePlan | None" = None) -> Iterator[list[int]]:
    """P labels x K items (with replacement for small labels); half of P from multi-item labels. With
    ``weights`` (per item, e.g. per source) the K items of a label are drawn by weight.

    ``groups`` (the recording group of every item, e.g. ``signer_group_key(c, dictionary_group=True)``): a "multi"
    label then needs at least 2 distinct groups (two ISLRTC rows of one word are one group, the dictionary), not
    just 2 items. ``bridge`` (:func:`bridge_plan`, docs/ISL_DESIGN.md 15.40): first ``bridge.p`` bridge labels per
    batch, each with K = one anchor (ISLRTC) item plus K-1 signer clips, the signer source drawn by
    ``bridge.weights`` over the sources that label has; then multi labels up to half of P (the bridge labels count
    toward that half), then uniform. Without both the draws are exactly the old ones."""
    if groups is not None or bridge is not None:
        yield from _bridge_pk_batches(labels, p, k, n_batches, seed, epoch, weights, groups, bridge)
        return
    rng = random.Random(f"{seed}:{epoch}")
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    multi = sorted(y for y, v in by.items() if len(v) > 1)
    allc = sorted(by)
    p = min(p, len(allc))
    for _ in range(n_batches):
        n_multi = min(len(multi), p // 2)
        chosen = set(rng.sample(multi, n_multi)) if n_multi else set()
        rest = [y for y in allc if y not in chosen]
        chosen |= set(rng.sample(rest, min(len(rest), p - len(chosen))))
        batch = []
        for y in sorted(chosen):
            idx = by[y]
            if weights is not None and len(idx) > k:
                batch += rng.choices(idx, weights=[weights[i] for i in idx], k=k)
            else:
                batch += rng.sample(idx, k) if len(idx) >= k else [rng.choice(idx) for _ in range(k)]
        yield batch


@dataclass
class BridgePlan:
    """The bridge labels of :func:`pk_batches` (docs/ISL_DESIGN.md 15.40): per label, its anchor items (ISLRTC rows)
    and its signer items by signer source (``include`` / ``isl500`` / ``mendeley``...), with the source weights."""
    p: int
    weights: dict[str, float]
    anchors: dict[int, list[int]]
    signers: dict[int, dict[str, list[int]]]

    @property
    def labels(self) -> list[int]:
        return sorted(self.anchors)

    def source_probs(self, y: int) -> dict[str, float]:
        """Probability of each signer source for one signer clip of label ``y`` (weights renormalised over the
        sources ``y`` has)."""
        av = {s: self.weights[s] for s in self.signers[y]}
        tot = sum(av.values())
        return {s: w / tot for s, w in sorted(av.items())}


def bridge_plan(labels: list[int], anchor: list[bool], signer_source: list[str | None], weights: dict[str, float],
                p: int) -> BridgePlan:
    """``anchor[i]``: item i is a dictionary anchor (an ISLRTC row); ``signer_source[i]``: the signer source key of
    item i (a key of ``weights``) or None. A bridge label has at least one anchor and one signer item whose source
    has a weight > 0."""
    anchors: dict[int, list[int]] = defaultdict(list)
    signers: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, y in enumerate(labels):
        if anchor[i]:
            anchors[y].append(i)
        elif signer_source[i] is not None and weights.get(signer_source[i], 0.0) > 0:
            signers[y][signer_source[i]].append(i)
    keep = sorted(set(anchors) & set(signers))
    return BridgePlan(int(p), {s: float(w) for s, w in weights.items() if w > 0},
                      {y: anchors[y] for y in keep}, {y: dict(signers[y]) for y in keep})


def _bridge_pk_batches(labels, p, k, n_batches, seed, epoch, weights, groups, bridge):
    rng = random.Random(f"{seed}:{epoch}")
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    if groups is not None:
        gs = defaultdict(set)
        for i, y in enumerate(labels):
            gs[y].add(groups[i])
        multi = sorted(y for y in by if len(gs[y]) > 1)
    else:
        multi = sorted(y for y, v in by.items() if len(v) > 1)
    allc = sorted(by)
    p = min(p, len(allc))
    blabels = bridge.labels if bridge is not None else []
    bprobs = {y: bridge.source_probs(y) for y in blabels}
    for _ in range(n_batches):
        nb = min(bridge.p, len(blabels), p) if bridge is not None else 0
        bchosen = set(rng.sample(blabels, nb)) if nb else set()
        avail = [y for y in multi if y not in bchosen]
        n_multi = min(len(avail), max(0, p // 2 - nb))
        chosen = bchosen | (set(rng.sample(avail, n_multi)) if n_multi else set())
        rest = [y for y in allc if y not in chosen]
        chosen |= set(rng.sample(rest, min(len(rest), p - len(chosen))))
        batch = []
        for y in sorted(chosen):
            if y in bchosen:
                batch.append(rng.choice(bridge.anchors[y]))
                srcs = list(bprobs[y])
                for _ in range(max(1, k - 1)):
                    s = rng.choices(srcs, weights=[bprobs[y][x] for x in srcs])[0] if len(srcs) > 1 else srcs[0]
                    batch.append(rng.choice(bridge.signers[y][s]))
                continue
            idx = by[y]
            if weights is not None and len(idx) > k:
                batch += rng.choices(idx, weights=[weights[i] for i in idx], k=k)
            else:
                batch += rng.sample(idx, k) if len(idx) >= k else [rng.choice(idx) for _ in range(k)]
        yield batch


def expected_bridge_pairs(bridge: BridgePlan, p: int, k: int, n_labels: int, steps: int, source: str) -> dict:
    """Expected anchor <-> ``source`` pairs per bridge label that has ``source`` over ``steps`` batches (for the
    meta: the plan's target is >= 50 INCLUDE <-> ISLRTC pairs per INCLUDE concept over the run)."""
    labs = bridge.labels
    if not labs:
        return {"labels": 0}
    nb = min(bridge.p, len(labs), p, n_labels)
    per = []
    for y in labs:
        pr = bridge.source_probs(y).get(source)
        if pr:
            per.append(steps * nb / len(labs) * max(1, k - 1) * pr)
    if not per:
        return {"labels": 0}
    per.sort()
    return {"labels": len(per), "min": round(per[0], 1), "median": round(per[len(per) // 2], 1),
            "max": round(per[-1], 1)}
