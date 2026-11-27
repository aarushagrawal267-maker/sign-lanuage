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

Validation (model selection), ``val_protocol``:

* ``mixed`` (default, as before): retrieval recall@1 / @5 of held-out INCLUDE (+ signer-held-out extra) validation
  clips against a gallery of ISLRTC references plus a few training clips per concept and source (``val_r1``).
* ``dictionary`` (QA plan 1.5, docs/ISL_DESIGN.md 15.40): queries = every INCLUDE validation clip, ISL500
  validation stratified at ``dict_val_per_signer`` clips per signer and every Mendeley validation clip; gallery = the
  ISLRTC dictionary rows only (no training clip). Per-source recall@1 / @5 are logged; the epoch is selected on
  ``val_dict_select = 0.5 * r1_include + 0.5 * mean(r1 of the other sources)``. The old ``val_r1`` is still logged.
  Held-out concepts (one-shot folds) are left out of both queries and gallery (critic 11).

Sampler: the P x K sampler of ``loop.pk_batches``; with ``bridge_p`` > 0 (docs 15.40) that many labels per batch are
"bridge" concepts (an ISLRTC row and a signer clip) drawn as one ISLRTC row + one signer clip, the signer source by
``bridge_weights``, and a "multi" label needs two recording groups (the dictionary is one group).
"""

from __future__ import annotations

import logging
import math
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
from app.isl.training.common import signer_split_signature as _signer_split_sig  # docs 15.42
from app.isl.training.common import (DICTIONARY_SOURCES, ClipDataset, IslData, LazyClipDataset, class_space_of,
                                     set_threads, signer_group_key)
from app.isl.training.loop import (LoopConfig, atomic_save, bridge_plan, clips_digest, digest, expected_bridge_pairs,
                                   file_digest, pk_batches, run_loop, train_config_signature)
from app.isl.training.multisource import SAMPLE_WEIGHTS
from app.models.checkpoint import frame_mask
from app.training.trainer import TrainConfig, set_seed

log = logging.getLogger(__name__)

ENCODER_VERSION = "isl-enc-1"

# validation protocols (docs 15.40): "mixed" = the old one (default), "dictionary" = QA plan 1.5
VAL_PROTOCOLS = ("mixed", "dictionary")
# the query sources of the dictionary protocol (a family key: every mendeley_* source is "mendeley")
DICT_VAL_SOURCES = ("include", "isl500", "mendeley")
DICT_VAL_PER_SIGNER = 300  # ISL500 validation clips per signer (stratified sample)
DICT_SELECT_KEY = "val_dict_select"
# bridge sampler (QA plan 1.5): the signer source of a bridge pair; renormalised over the sources a concept has
DEFAULT_BRIDGE_WEIGHTS = {"include": 0.5, "isl500": 0.35, "mendeley": 0.15}
DEFAULT_BRIDGE_P = 12
BRIDGE_ANCHOR = "islrtc"


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


def source_family(source: str) -> str:
    """``mendeley`` for every ``mendeley_*`` source, else the source name (the per-source validation keys)."""
    return "mendeley" if source.startswith("mendeley") else source


def signer_key(source: str, weights: dict[str, float]) -> str | None:
    """The key of ``weights`` a training source draws under as a bridge signer clip: its own name when named, else
    its family (``mendeley``), else None (not a signer source of the bridge)."""
    if source in weights:
        return source
    fam = source_family(source)
    return fam if fam in weights else None


def parse_bridge_weights(spec) -> dict[str, float] | None:
    """``--source-weights``: ``include=0.5,isl500=0.35,mendeley=0.15`` (the default), ``default`` or a dict. Keys are
    signer sources or the ``mendeley`` family; a dictionary source (the anchor side) or an unknown name is refused,
    and so are negative, non-finite or all-zero weights. None / ``none`` -> None (the default weights)."""
    from app.isl.profiles import WORD_SOURCES
    from app.isl.training.source_quota import parse_source_quota

    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "none", "default")):
        return None
    try:
        w = parse_source_quota(spec)
    except ValueError as e:
        raise ValueError(str(e).replace("source quota", "source weights")) from None
    known = ({s for v in WORD_SOURCES.values() for s in v} - set(DICTIONARY_SOURCES)) | {"mendeley"}
    bad = sorted(set(w) - known)
    if bad:
        raise ValueError(f"source weights: {bad} cannot be a bridge signer source (the dictionary is the anchor "
                         f"side); signer sources: {sorted(known)}")
    return w


def dictionary_val_sets(data: IslData, excl: set[str] = frozenset(), xsrc=(), per_signer: int = DICT_VAL_PER_SIGNER,
                        seed: int = 0) -> tuple[list, list, dict]:
    """Queries and gallery of the ``dictionary`` validation protocol (docs 15.40).

    Gallery: every ISLRTC dictionary row of a concept not in ``excl`` (no training clip). Queries: every INCLUDE
    validation clip, ISL500 validation clips stratified at ``per_signer`` per signer, every Mendeley validation clip,
    of concepts not in ``excl`` (held-out one-shot concepts never take part, critic 11), of sources not in ``xsrc``
    and with a dictionary row (a query whose word has no row cannot be retrieved). Returns ``(queries, gallery,
    info)``; ``info`` counts the queries per source before and after the dictionary-row filter."""
    xs = set(xsrc or ())
    excl = set(excl or ())
    gal = [c for c in data.islrtc if c.concept_id and c.concept_id not in excl]
    gconcepts = {c.concept_id for c in gal}
    cand = [c for c in data.include_part("val") if c.source not in xs and c.concept_id not in excl]
    isl = defaultdict(list)
    for c in data.extra_part("val"):
        if c.source in xs or c.concept_id in excl:
            continue
        fam = source_family(c.source)
        if fam == "isl500":
            isl[c.group or c.clip_id].append(c)
        elif fam in DICT_VAL_SOURCES:
            cand.append(c)
    rng = random.Random(f"{seed}:dictionary-val")
    for g in sorted(isl):
        v = sorted(isl[g], key=lambda c: c.clip_id)
        cand += rng.sample(v, per_signer) if 0 < per_signer < len(v) else v
    q = [c for c in cand if c.concept_id in gconcepts]
    by_src = defaultdict(lambda: [0, 0])
    for c in cand:
        by_src[source_family(c.source)][0] += 1
    for c in q:
        by_src[source_family(c.source)][1] += 1
    info = {"gallery": len(gal), "gallery_concepts": len(gconcepts), "per_signer": per_signer,
            "isl500_signers": len(isl), "excluded_concepts": len(excl),
            "queries": {s: {"candidates": a, "with_dictionary_row": b} for s, (a, b) in sorted(by_src.items())}}
    return q, gal, info


def _top5(q: np.ndarray, g: np.ndarray, g_lab: list) -> tuple[list, np.ndarray]:
    """Best gallery score per label (max over a label's rows), then the 5 best labels per query."""
    labs = sorted(set(g_lab))
    pos = {c: i for i, c in enumerate(labs)}
    order = np.argsort(np.array([pos[c] for c in g_lab]), kind="stable")
    gi = np.array([pos[g_lab[j]] for j in order])
    starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
    out = []
    for a in range(0, len(q), 512):
        s = q[a:a + 512] @ g[order].T
        cs = np.maximum.reduceat(s, starts, axis=1)
        out.append(np.argsort(-cs, axis=1, kind="stable")[:, :5])
    return labs, (np.concatenate(out) if out else np.zeros((0, 5), np.int64))


def dictionary_val_metrics(q: np.ndarray, q_clips: list, g: np.ndarray, g_lab: list,
                           hits: dict | None = None) -> dict:
    """Per-source recall@1 / @5 of the dictionary protocol: ``val_dict_r1_<src>``, ``val_dict_r5_<src>``,
    ``val_dict_n_<src>`` for every query source family present, the pooled ``val_dict_r1`` / ``val_dict_r5`` and the
    selection key ``val_dict_select = 0.5 * r1_include + 0.5 * mean(r1 of the other sources)`` (only one side
    present: that side; no query: 0). ``hits``: when a dict, filled with ``query_key(clip) -> 0/1`` (hit@1 of every
    query), for a paired comparison of two encoders on the same queries (encoder-probe ``--bar``, docs 15.41)."""
    out: dict = {}
    if len(q) == 0 or len(g) == 0:
        out.update(val_dict_r1=0.0, val_dict_r5=0.0, val_dict_n=0)
        out[DICT_SELECT_KEY] = 0.0
        return out
    labs, top = _top5(q, g, g_lab)
    hit1 = defaultdict(int)
    hit5 = defaultdict(int)
    n = defaultdict(int)
    for b, c in enumerate(q_clips):
        s = source_family(c.source)
        t = [labs[j] for j in top[b]]
        n[s] += 1
        h = int(t[0] == c.concept_id)
        hit1[s] += h
        hit5[s] += int(c.concept_id in t)
        if hits is not None:
            hits[query_key(c)] = h
    for s in sorted(n):
        out[f"val_dict_r1_{s}"] = hit1[s] / n[s]
        out[f"val_dict_r5_{s}"] = hit5[s] / n[s]
        out[f"val_dict_n_{s}"] = n[s]
    out["val_dict_r1"] = sum(hit1.values()) / len(q_clips)
    out["val_dict_r5"] = sum(hit5.values()) / len(q_clips)
    out["val_dict_n"] = len(q_clips)
    out[DICT_SELECT_KEY] = dictionary_select(out)
    return out


def query_key(c) -> str:
    """The key of a probe query in ``hits``: ``<source>|<clip id>``."""
    return f"{c.source}|{c.clip_id}"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value of ``b`` new-only and ``c`` old-only hits (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def paired_hits(new: dict, old: dict, source: str | None = "include") -> dict:
    """Paired hit@1 comparison of two probes on their common queries (only ``source`` when given): ``new_only`` /
    ``bar_only`` = queries only one encoder gets right, ``delta_r1`` on the common queries and the exact McNemar
    p-value. Clips of one signer session are not independent, so the p-value is optimistic (docs 15.41)."""
    keys = sorted(k for k in set(new) & set(old) if source is None or k.split("|", 1)[0] == source)
    b = sum(1 for k in keys if new[k] and not old[k])
    c = sum(1 for k in keys if old[k] and not new[k])
    n = len(keys)
    return {"source": source, "common": n, "new_only": b, "bar_only": c,
            "new_r1": round(sum(new[k] for k in keys) / n, 4) if n else None,
            "bar_r1": round(sum(old[k] for k in keys) / n, 4) if n else None,
            "delta_r1": round((b - c) / n, 4) if n else None, "p_mcnemar": round(mcnemar_exact(b, c), 4)}


def dictionary_select(m: dict) -> float:
    """``0.5 * r1_include + 0.5 * mean(r1 of the other query sources)`` from a :func:`dictionary_val_metrics` row."""
    inc = m.get("val_dict_r1_include") if m.get("val_dict_n_include") else None
    others = [m[f"val_dict_r1_{s}"] for s in sorted({k[len("val_dict_n_"):] for k in m if k.startswith("val_dict_n_")})
              if s != "include" and m.get(f"val_dict_n_{s}")]
    oth = sum(others) / len(others) if others else None
    if inc is not None and oth is not None:
        return 0.5 * inc + 0.5 * oth
    return float(inc if inc is not None else (oth if oth is not None else 0.0))


class _FeatureBank:
    """Validation features cached once (float16) for several clip lists that share clips (the old and the
    dictionary protocol): every clip is featurised and embedded once per epoch."""

    def __init__(self, lists: dict[str, list], fcfg: FeatureConfig):
        uniq, key_of = [], {}
        for v in lists.values():
            for c in v:
                key = (c.clip_id, c.start, c.end)
                if key not in key_of:
                    key_of[key] = len(uniq)
                    uniq.append(c)
        feats, kept = _features_cache(uniq, fcfg)
        row = {(c.clip_id, c.start, c.end): i for i, c in enumerate(kept)}
        self.feats = feats
        self.rows = {n: np.array([row[(c.clip_id, c.start, c.end)] for c in v if (c.clip_id, c.start, c.end) in row],
                                 np.int64) for n, v in lists.items()}
        self.clips = {n: [c for c in v if (c.clip_id, c.start, c.end) in row] for n, v in lists.items()}

    def embed(self, enc, fcfg) -> np.ndarray:
        return embed_array(enc, self.feats, fcfg)


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
                  exclude_sources: tuple[str, ...] | list[str] = (), bridge_p: int = 0,
                  bridge_weights: dict | None = None, val_protocol: str = "mixed",
                  dict_val_per_signer: int = DICT_VAL_PER_SIGNER) -> dict:
    """``bridge_p`` > 0 turns the bridge sampler on (that many bridge labels per batch, signer sources drawn by
    ``bridge_weights``, default :data:`DEFAULT_BRIDGE_WEIGHTS`); ``val_protocol="dictionary"`` selects the epoch on the
    dictionary protocol (docs 15.40). Defaults: exactly the old sampler and validation."""
    if val_protocol not in VAL_PROTOCOLS:
        raise ValueError(f"val_protocol must be one of {VAL_PROTOCOLS}, not {val_protocol!r}")
    if bridge_p < 0:
        raise ValueError("bridge_p must be >= 0")
    if bridge_p == 0 and parse_bridge_weights(bridge_weights) is not None:  # never ignored silently (docs 15.41)
        raise ValueError("bridge_weights (--source-weights) need bridge_p > 0 (--bridge-p)")
    if dict_val_per_signer < 0:
        raise ValueError("dict_val_per_signer must be >= 0 (0 = every ISL500 validation clip)")
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

    # bridge sampler (docs 15.40): off unless bridge_p > 0, so the default draws are exactly the old ones
    bw = dict(parse_bridge_weights(bridge_weights) or DEFAULT_BRIDGE_WEIGHTS) if bridge_p > 0 else None
    plan, groups, sampler = None, None, {"kind": "pk", "p": p, "k": k}
    if bw is not None:
        groups = [signer_group_key(c, dictionary_group=True) for c in train_clips]
        plan = bridge_plan(labels, [c.source == BRIDGE_ANCHOR for c in train_clips],
                           [signer_key(c.source, bw) for c in train_clips], bw, bridge_p)
        gs = defaultdict(set)
        for y, g in zip(labels, groups):
            gs[y].add(g)
        by_src = defaultdict(int)
        for y in plan.labels:
            for s_ in plan.signers[y]:
                by_src[s_] += 1
        sampler = {"kind": "bridge", "p": p, "k": k, "bridge_p": bridge_p, "weights": bw, "anchor": BRIDGE_ANCHOR,
                   "bridge_labels": len(plan.labels), "bridge_labels_by_source": dict(sorted(by_src.items())),
                   "multi_labels": sum(1 for v in gs.values() if len(v) > 1), "multi_rule": "two recording groups",
                   "expected_pairs_per_label": {
                       s_: expected_bridge_pairs(plan, p, k, len(concepts), cfg.epochs * n_batches, s_)
                       for s_ in sorted(by_src)}}
        if not plan.labels:
            log.warning("bridge sampler: no concept has an %s row and a signer clip of %s; plain P x K batches",
                        BRIDGE_ANCHOR, sorted(bw))
        if progress:
            progress(None, f"bridge sampler: {len(plan.labels)} bridge concepts {sampler['bridge_labels_by_source']}, "
                           f"{bridge_p} of P={p} per batch; expected pairs per label over the run "
                           f"{sampler['expected_pairs_per_label']}")

    def batches(epoch: int):
        for s, idx in enumerate(pk_batches(labels, p, k, n_batches, cfg.seed, epoch, weights, groups=groups,
                                           bridge=plan)):
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
    dict_info = None
    if val_protocol == "mixed":
        q_feats, vq = _features_cache(vq, fcfg)
        g_feats, gal = _features_cache(gal, fcfg)
        q_lab = [c.concept_id for c in vq]
        g_lab = [c.concept_id for c in gal]

        def val_fn(m) -> dict:
            if not len(q_feats) or not len(g_feats):
                return {"val_r1": 0.0, "val_r5": 0.0}
            return retrieval_metrics(embed_array(m.enc, q_feats, fcfg), q_lab, embed_array(m.enc, g_feats, fcfg),
                                     g_lab)
    else:  # dictionary protocol (docs 15.40); the old val_r1 is still computed on its own sets, as a secondary
        dq, dg, dict_info = dictionary_val_sets(data, excl, xsrc, dict_val_per_signer, cfg.seed)
        bank = _FeatureBank({"q": vq, "g": gal, "dq": dq, "dg": dg}, fcfg)
        vq, gal, dq, dg = bank.clips["q"], bank.clips["g"], bank.clips["dq"], bank.clips["dg"]
        q_lab = [c.concept_id for c in vq]
        g_lab = [c.concept_id for c in gal]
        dg_lab = [c.concept_id for c in dg]
        dict_info = {**dict_info, "queries_used": len(dq), "gallery_used": len(dg),
                     # what encoder-probe compares its own sets with (docs 15.41)
                     "query_digest": clips_digest(dq), "gallery_digest": clips_digest(dg)}

        def val_fn(m) -> dict:
            z = bank.embed(m.enc, fcfg)
            r = ({"val_r1": 0.0, "val_r5": 0.0} if not len(vq) or not len(gal) else
                 retrieval_metrics(z[bank.rows["q"]], q_lab, z[bank.rows["g"]], g_lab))
            d = dictionary_val_metrics(z[bank.rows["dq"]], dq, z[bank.rows["dg"]], dg_lab)
            if progress:
                per = " ".join(f"{s}->islrtc r1 {d[f'val_dict_r1_{s}']:.3f} (n={d[f'val_dict_n_{s}']})"
                               for s in DICT_VAL_SOURCES if f"val_dict_n_{s}" in d)
                progress(None, f"val dictionary: {per}; select {d[DICT_SELECT_KEY]:.4f}; old val_r1 "
                               f"{r['val_r1']:.4f}")
            return {**r, **d}

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
                     "train_sources": dict(src_counts), "val_queries": len(vq), "gallery": len(gal)},
            "sampler": sampler, "val_protocol": val_protocol,
            "select_key": DICT_SELECT_KEY if val_protocol == "dictionary" else "val_r1",
            **({"val_dictionary_sets": dict_info} if dict_info is not None else {})}

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
                           **({"exclude_sources": xsrc} if xsrc else {}),
                           # bridge sampler and dictionary validation (docs 15.40): only when on (rule of 15.19)
                           **({"bridge": {"p": bridge_p, "weights": bw, "anchor": BRIDGE_ANCHOR,
                                          "multi": "groups", "labels": digest(plan.labels)}} if plan is not None
                              else {}),
                           **({"val_protocol": val_protocol,
                               "val_dict": [dict_val_per_signer, clips_digest(dq), clips_digest(dg)]}
                              if val_protocol == "dictionary" else {}),
                           # signer split v2 (docs 15.42): only when not v1 (rule of 15.19)
                           **_signer_split_sig(data)})
    select_key = DICT_SELECT_KEY if val_protocol == "dictionary" else "val_r1"
    res = run_loop(both, batches, loss_fn, lc, state_path or Path(str(out) + ".state"), val_fn=val_fn,
                   select_key=select_key, save_best=save_best, fresh=not resume,
                   progress=(lambda m: progress(None, m)) if progress else None)
    if not out.exists():
        raise RuntimeError("encoder training produced no checkpoint")
    ck = torch.load(out, map_location="cpu", weights_only=False)
    best = next((h for h in res.history if h["epoch"] == res.best_epoch), {})
    val_dict = {k: v for k, v in best.items() if k.startswith("val_dict")} or None
    ck["meta"].update(val_r1=best.get("val_r1"), val_r5=best.get("val_r5"), history=res.history,
                      train_seconds=time.time() - t0, step_seconds=res.step_seconds, finished=res.finished,
                      **({"val_dict": val_dict} if val_dict else {}))
    atomic_save(ck, out)
    return {"checkpoint": str(out), "concepts": len(concepts), "train_items": len(train_clips),
            "train_sources": dict(src_counts), "best_epoch": res.best_epoch, "val_r1": best.get("val_r1"),
            "val_r5": best.get("val_r5"), "val_queries": len(vq), "finished": res.finished,
            "resumed_from": res.resumed_from, "step_seconds": res.step_seconds, "pretrained": bool(init),
            "class_space": class_space,
            **({"val_protocol": val_protocol, "select_key": select_key, "val_dict": val_dict,
                "sampler": sampler} if val_protocol != "mixed" or plan is not None else {}),
            "seconds": round(time.time() - t0, 1)}


def probe_encoder(data: IslData, ckpt: Path, *, per_signer: int = DICT_VAL_PER_SIGNER, seed: int | None = None,
                  threads: int = 2, data_seed: int | None = None) -> dict:
    """Measure any encoder checkpoint on the ``dictionary`` validation protocol (docs 15.40, critic 3): the concepts
    it was trained without (``excluded_concepts``, one-shot folds) are left out of queries and gallery, and so are
    the sources it was trained without. ``data`` must be loaded with the encoder's split (the caller checks).

    The result names exactly what was measured (docs 15.41): digests of the queries (all, and INCLUDE alone) and of
    the gallery, ``per_signer``, profile, split, ``data_seed`` (the seed the data was loaded with) and the link
    settings; ``hits`` holds every query's hit@1 for a paired comparison with another probe."""
    from app.isl.models.encoder import encoder_kwargs

    set_threads(threads)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = ck.get("meta") or {}
    if meta.get("kind") != "isl-encoder":
        raise ValueError(f"{ckpt} is not an ISL encoder checkpoint (kind={meta.get('kind')!r})")
    fcfg = FeatureConfig.from_dict(meta["feature_config"])
    enc = SignEncoder(fcfg.dim, **encoder_kwargs(meta))
    enc.load_state_dict(ck["state_dict"])
    excl = set(meta.get("excluded_concepts") or [])
    xsrc = encoder_excluded_sources(meta)
    seed = int((meta.get("train_config") or {}).get("seed", 0)) if seed is None else seed
    t0 = time.time()
    dq, dg, info = dictionary_val_sets(data, excl, xsrc, per_signer, seed)
    bank = _FeatureBank({"dq": dq, "dg": dg}, fcfg)
    z = bank.embed(enc, fcfg)
    hits: dict = {}
    m = dictionary_val_metrics(z[bank.rows["dq"]], bank.clips["dq"], z[bank.rows["dg"]],
                               [c.concept_id for c in bank.clips["dg"]], hits=hits)
    by = {s: {"n": m[f"val_dict_n_{s}"], "r1": round(m[f"val_dict_r1_{s}"], 4), "r5": round(m[f"val_dict_r5_{s}"], 4)}
          for s in sorted({k[len("val_dict_n_"):] for k in m if k.startswith("val_dict_n_")})}
    qd, gd = clips_digest(bank.clips["dq"]), clips_digest(bank.clips["dg"])
    tv = meta.get("val_dictionary_sets") or {}
    # an encoder selected on the dictionary protocol: are these its own validation sets? (None = not recorded)
    same = ((tv["query_digest"] == qd and tv["gallery_digest"] == gd) if tv.get("query_digest")
            and tv.get("per_signer") == per_signer else None)
    return {"kind": "isl-encoder-probe", "protocol": "dictionary", "encoder": str(ckpt),
            "encoder_sha256": file_digest(ckpt), "encoder_version": meta.get("encoder_version"),
            "profile": data.profile, "split": data.split.get("name"), "encoder_split": (meta.get("data") or {}).get("split"),
            "excluded_concepts": len(excl), "excluded_sources": xsrc, "seed": seed, "data_seed": data_seed,
            "per_signer": per_signer, "link": digest(data.link or {}),
            "query_digest": qd, "gallery_digest": gd, "same_sets_as_training": same,
            "include_query_digest": clips_digest([c for c in bank.clips["dq"] if source_family(c.source) == "include"]),
            "by_source": by, "r1": round(m["val_dict_r1"], 4), "r5": round(m["val_dict_r5"], 4),
            "select": round(m[DICT_SELECT_KEY], 4), "include_r1": by.get("include", {}).get("r1"),
            "queries_used": len(bank.clips["dq"]), "gallery_used": len(bank.clips["dg"]), "sets": info,
            "trained_val": {"val_protocol": meta.get("val_protocol", "mixed"), "val_r1": meta.get("val_r1"),
                            "val_dict": meta.get("val_dict")},
            "seconds": round(time.time() - t0, 1), "hits": hits}


__all__ = ["ENCODER_VERSION", "ClipDataset", "embed_items", "retrieval_metrics", "train_encoder",
           "encoder_training_clips", "encoder_excluded_sources", "check_exclude_sources", "dictionary_val_sets",
           "dictionary_val_metrics", "dictionary_select", "parse_bridge_weights", "probe_encoder",
           "DEFAULT_BRIDGE_WEIGHTS", "DEFAULT_BRIDGE_P", "VAL_PROTOCOLS", "paired_hits", "mcnemar_exact", "query_key"]
