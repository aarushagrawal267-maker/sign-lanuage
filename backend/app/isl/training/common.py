"""Shared ISL training utilities: clip selection per profile/split, a BodyFrame cache, datasets and the
P x K batch sampler (docs/ISL_DESIGN.md sections 6.3 and 6.6)."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from app.features.build import BodyFrame, FeatureError, featurize, prepare_body_frame
from app.features.spec import FeatureConfig
from app.isl.augment import StrongAugmentConfig, augment_strong
from app.isl.data.sources import Clip, IncludeAdapter, IslrtcAdapter
from app.isl.paths import PACK_DIR, IslPaths
from app.isl.profiles import get_profile
from app.landmarks.raw import RawSequence

log = logging.getLogger(__name__)

ISL_FEATURES = FeatureConfig(seq_len=48, use_face=True, use_velocity=True, max_gap=5, canonical_hand=True)


def slice_bf(bf: BodyFrame, a: int, b: int) -> BodyFrame:
    sl = slice(a, b)
    return BodyFrame(bf.pose[sl], bf.left[sl], bf.right[sl], bf.face[sl], {k: v[sl] for k, v in bf.present.items()},
                     bf.t[sl], bf.center, bf.scale)


@dataclass
class Item:
    clip: Clip
    label: int
    bf: BodyFrame  # body frame of the clip, possibly longer than the default end (alternative trim ends)
    ends: list[int] = field(default_factory=list)  # candidate end frames relative to bf (training picks one)
    default_end: int = 0


def load_item(clip: Clip, label: int, fcfg: FeatureConfig, with_alternatives: bool = True) -> Item | None:
    """Load a clip into an ``Item``. ISLRTC trims keep the first-sign slice as the default end and its
    alternative ends (later rests / dips) for training-time variety."""
    try:
        if clip.source == "islrtc" and clip.end is not None and with_alternatives:
            full = RawSequence.load(clip.path)
            alts = [int(a) for a in clip.quality.get("trim_alternatives") or [] if clip.end < int(a) <= len(full)]
            stop = max([clip.end, *alts])
            raw = full.slice(clip.start, stop)
            ends = sorted({clip.end - clip.start, *[a - clip.start for a in alts]})
            bf = prepare_body_frame(raw, fcfg)
            return Item(clip, label, bf, ends, clip.end - clip.start)
        raw = clip.load()
        if len(raw) < 4:
            return None
        bf = prepare_body_frame(raw, fcfg)
        return Item(clip, label, bf, [len(bf)], len(bf))
    except (FeatureError, OSError, ValueError, KeyError) as e:
        log.debug("skip %s: %s", clip.clip_id, e)
        return None


class ClipDataset(Dataset):
    """Items -> (features (T, D), label). Training: strong augmentation and a random trim end (ISLRTC);
    evaluation: deterministic features of the default slice, computed once."""

    def __init__(self, items: list[Item], fcfg: FeatureConfig, train: bool, seed: int = 0,
                 aug: StrongAugmentConfig | None = None, end_jitter_p: float = 0.3):
        self.items = items
        self.fcfg = fcfg
        self.train = train
        self.aug = aug or StrongAugmentConfig()
        self.rng = np.random.default_rng(seed)
        self.end_jitter_p = end_jitter_p
        self._cache = None if train else [featurize(slice_bf(it.bf, 0, it.default_end), fcfg) for it in items]

    def __len__(self) -> int:
        return len(self.items)

    def features(self, i: int) -> np.ndarray:
        it = self.items[i]
        if self._cache is not None:
            return self._cache[i]
        end = it.default_end
        if len(it.ends) > 1 and self.rng.random() < self.end_jitter_p:
            end = int(self.rng.choice(it.ends))
        return featurize(augment_strong(slice_bf(it.bf, 0, max(2, end)), self.rng, self.aug), self.fcfg)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.features(i)), self.items[i].label


class PKSampler(Sampler):
    """Batches of P labels x K samples. Labels with fewer than K items are sampled with replacement (a
    single-clip concept yields K different augmentations of its clip: SimCLR-style positives). Half of the
    labels in a batch are drawn from multi-clip labels when available."""

    def __init__(self, labels: list[int], p: int = 32, k: int = 2, batches: int | None = None, seed: int = 0):
        self.by = defaultdict(list)
        for i, y in enumerate(labels):
            self.by[y].append(i)
        self.multi = [y for y, v in self.by.items() if len(v) > 1]
        self.all = list(self.by)
        self.p, self.k = min(p, len(self.all)), k
        self.batches = batches or max(1, len(labels) // max(1, self.p * self.k))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            n_multi = min(len(self.multi), self.p // 2)
            chosen = set(self.rng.sample(self.multi, n_multi)) if n_multi else set()
            rest = [y for y in self.all if y not in chosen]
            chosen |= set(self.rng.sample(rest, min(len(rest), self.p - len(chosen))))
            batch = []
            for y in chosen:
                idx = self.by[y]
                batch += self.rng.sample(idx, self.k) if len(idx) >= self.k else [self.rng.choice(idx) for _ in range(self.k)]
            yield batch


# ------------------------------------------------------------------------------------------- data selection

def load_vocab(path: Path | None = None):
    from app.isl.vocab.build import VocabIndex

    return VocabIndex.load(path or PACK_DIR / "vocabulary.jsonl")


def primary_split_name(paths: IslPaths, fold: int = 0) -> str:
    rp = paths.splits / "splits_report.json"
    base = "include-session-cv5"
    if rp.exists():
        try:
            base = json.loads(rp.read_text()).get("primary_split") or base
        except ValueError:
            pass
    return f"{base}-fold{fold}"


def load_split(paths: IslPaths, name: str) -> dict:
    p = paths.splits / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"split {name} not found in {paths.splits}; run 'python -m app.isl.cli splits' first")
    return json.loads(p.read_text())


@dataclass
class IslData:
    include: list[Clip]
    islrtc: list[Clip]
    assign: dict[str, str]  # INCLUDE clip id -> train | val | test
    split: dict
    profile: str
    unassigned: int = 0

    def include_part(self, part: str) -> list[Clip]:
        return [c for c in self.include if self.assign.get(c.clip_id) == part and c.concept_id]

    def summary(self) -> dict:
        parts = {p: len(self.include_part(p)) for p in ("train", "val", "test")}
        return {"profile": self.profile, "split": self.split.get("name"), "split_label": self.split.get("label"),
                "signer_independent": self.split.get("signer_independent"), "include": parts,
                "include_unassigned": self.unassigned, "islrtc": len(self.islrtc),
                "islrtc_untrimmed": sum(1 for c in self.islrtc if c.end is None and c.quality.get("kind") != "phrase"),
                "include_concepts": len({c.concept_id for c in self.include if c.concept_id})}


def load_isl_data(paths: IslPaths, vocab, profile: str = "commercial-safe", split_name: str | None = None,
                  max_islrtc: int | None = None, seed: int = 0) -> IslData:
    prof = get_profile(profile)
    include, islrtc = [], []
    if "include" in prof.sources:
        include = list(IncludeAdapter(paths, vocab=vocab).clips())
    if "islrtc" in prof.sources:
        islrtc = [c for c in IslrtcAdapter(paths, vocab=vocab).clips() if c.concept_id]
    if max_islrtc and len(islrtc) > max_islrtc:
        rng = random.Random(seed)
        islrtc = rng.sample(islrtc, max_islrtc)
    name = split_name or primary_split_name(paths)
    split = load_split(paths, name)
    assign = split.get("assign", {})
    unassigned = sum(1 for c in include if c.clip_id not in assign)
    return IslData(include, islrtc, assign, split, prof.name, unassigned)


def items_for(clips: list[Clip], label_of: dict[str, int], fcfg: FeatureConfig, with_alternatives: bool = True) -> list[Item]:
    out = []
    for c in clips:
        if c.concept_id not in label_of:
            continue
        it = load_item(c, label_of[c.concept_id], fcfg, with_alternatives)
        if it is not None:
            out.append(it)
    return out


def engine_crops(raw: RawSequence) -> list[RawSequence]:
    """The engine's three test-time crops (whole, without the first / last 10 %)."""
    n = len(raw)
    crops = [raw]
    if n >= 12:
        k = max(1, n // 10)
        crops += [raw.slice(k, n), raw.slice(0, n - k)]
    return crops


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
