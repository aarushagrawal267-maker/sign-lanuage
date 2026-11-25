"""Shared ISL training utilities: clip selection per profile/split, a BodyFrame cache, datasets and the
P x K batch sampler (docs/ISL_DESIGN.md sections 6.3 and 6.6)."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
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
    # phase 2: every other labelled word source of the profile (training multisource.attach_all_sources)
    extra: list[Clip] = field(default_factory=list)
    extra_assign: dict[str, str] = field(default_factory=dict)  # clip id -> train | val | test (signer-held-out)
    linker: object = None
    dedup: object = None
    multisource: dict = field(default_factory=dict)
    islrtc_stats: dict = field(default_factory=dict)  # re-extracted / truncated ISLRTC clips (resolve_islrtc)

    def include_part(self, part: str) -> list[Clip]:
        return [c for c in self.include if self.assign.get(c.clip_id) == part and c.concept_id]

    def extra_part(self, part: str, sources: tuple[str, ...] | None = None) -> list[Clip]:
        return [c for c in self.extra if self.extra_assign.get(c.clip_id) == part and c.concept_id
                and (sources is None or c.source in sources)]

    def labelled_part(self, part: str) -> list[Clip]:
        """INCLUDE clips of a split part plus the other labelled sources' clips of that part."""
        return self.include_part(part) + self.extra_part(part)

    def summary(self) -> dict:
        parts = {p: len(self.include_part(p)) for p in ("train", "val", "test")}
        extra = {}
        for c in self.extra:
            extra.setdefault(c.source, Counter())[self.extra_assign.get(c.clip_id, "train")] += 1
        return {"extra_sources": {k: dict(v) for k, v in extra.items()}, "profile": self.profile, "split": self.split.get("name"), "split_label": self.split.get("label"),
                "signer_independent": self.split.get("signer_independent"), "include": parts,
                "include_unassigned": self.unassigned, "islrtc": len(self.islrtc),
                "islrtc_untrimmed": sum(1 for c in self.islrtc if c.end is None and c.quality.get("kind") != "phrase"),
                "islrtc_reextraction": dict(self.islrtc_stats),
                "islrtc_trim_versions": dict(Counter(str(c.quality.get("trim_version")) for c in self.islrtc
                                                     if c.end is not None)),
                "include_concepts": len({c.concept_id for c in self.include if c.concept_id})}


# ISLRTC rows re-extracted after the first pass (card-v2 title-card detector; ``scripts/ingest_isl.py
# islrtc-recut``): a row here supersedes the islrtc.jsonl row with the same id
ISLRTC_RECUT_MANIFESTS = ("islrtc_recut",)


def islrtc_overrides(paths: IslPaths) -> dict[str, dict]:
    """id -> re-extracted ISLRTC manifest row (ok rows with frames only; later manifests win)."""
    from app.isl.data.manifest import read_manifest

    out: dict[str, dict] = {}
    for name in ISLRTC_RECUT_MANIFESTS:
        f = paths.manifest(name)
        if f.exists():
            for r in read_manifest(f):
                if r.get("n_frames"):
                    out[r["id"]] = r
    return out


def resolve_islrtc(clips: list[Clip], paths: IslPaths) -> tuple[list[Clip], dict]:
    """Prefer re-extracted ISLRTC files and keep cut-short clips out until they are re-extracted.

    * a clip with a re-extracted row uses that file; its trim is kept only when the trim sidecar says it was
      computed on that file (``landmarks``), otherwise the clip is used untrimmed until ``cli trim`` runs;
    * a clip whose trim is ``truncated`` (the old title-card detector stopped it mid-sign) and that has no
      re-extracted file yet is **excluded** from training, the index and evaluation galleries."""
    from dataclasses import replace

    from app.isl.data.manifest import read_manifest
    from app.isl.data.sources import passes_quality

    over = islrtc_overrides(paths)
    trims = {r["id"]: r for r in read_manifest(paths.trim_sidecar, only_ok=False)} if paths.trim_sidecar.exists() else {}
    out, st = [], Counter()
    for c in clips:
        if c.end is not None and c.clip_id in trims:
            c = replace(c, quality={**c.quality, "trim_version": trims[c.clip_id].get("trim_version")})
        o = over.get(c.clip_id)
        if o is not None:
            if not passes_quality(o):
                st["recut_failed_quality"] += 1
                continue
            t = trims.get(c.clip_id) or {}
            if t.get("landmarks") == o["path"]:
                c = replace(c, path=paths.resolve(o["path"]), quality={**c.quality, "recut": True,
                                                                         "stop": o.get("stop")})
                if t.get("method") == "truncated":
                    st["truncated_excluded"] += 1
                    continue
            else:
                q = {k: v for k, v in c.quality.items() if not k.startswith("trim_")}
                c = replace(c, path=paths.resolve(o["path"]), start=0, end=None,
                            quality={**q, "recut": True, "stop": o.get("stop"), "n_frames": o.get("n_frames")})
                st["recut_untrimmed"] += 1
            st["recut_used"] += 1
        elif c.quality.get("trim_method") == "truncated":
            st["truncated_excluded"] += 1
            continue
        out.append(c)
    return out, dict(st)


def load_isl_data(paths: IslPaths, vocab, profile: str = "commercial-safe", split_name: str | None = None,
                  max_islrtc: int | None = None, seed: int = 0) -> IslData:
    prof = get_profile(profile)
    include, islrtc = [], []
    islrtc_stats: dict = {}
    if "include" in prof.sources:
        include = list(IncludeAdapter(paths, vocab=vocab).clips())
    if "islrtc" in prof.sources:
        ad = IslrtcAdapter(paths, vocab=vocab, overrides=islrtc_overrides(paths))
        islrtc, islrtc_stats = resolve_islrtc([c for c in ad.clips() if c.concept_id], paths)
    if max_islrtc and len(islrtc) > max_islrtc:
        # smoke runs: keep the dictionary clips of INCLUDE concepts first (the ones E1 / E2 / E3 evaluate), then fill
        rng = random.Random(seed)
        inc = {c.concept_id for c in include if c.concept_id}
        core = [c for c in islrtc if c.concept_id in inc]
        rest = [c for c in islrtc if c.concept_id not in inc]
        rng.shuffle(core)
        islrtc = core[:max_islrtc] + rng.sample(rest, max(0, min(len(rest), max_islrtc - len(core))))
    name = split_name or primary_split_name(paths)
    split = load_split(paths, name)
    assign = split.get("assign", {})
    unassigned = sum(1 for c in include if c.clip_id not in assign)
    return IslData(include, islrtc, assign, split, prof.name, unassigned, islrtc_stats=islrtc_stats)


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


class LazyClipDataset:
    """Clips loaded on demand with a bounded LRU of body frames, for training sets too large to cache whole
    (ISL500 alone has tens of thousands of clips). Same features as :class:`ClipDataset`: training = strong
    augmentation + a random ISLRTC trim end; evaluation = deterministic features of the default slice."""

    def __init__(self, clips: list[Clip], labels: list[int], fcfg: FeatureConfig, train: bool, seed: int = 0,
                 aug: StrongAugmentConfig | None = None, end_jitter_p: float = 0.3, cache_items: int = 12000):
        from collections import OrderedDict

        self.clips, self.labels, self.fcfg, self.train = clips, labels, fcfg, train
        self.aug = aug or StrongAugmentConfig()
        self.rng = np.random.default_rng(seed)
        self.end_jitter_p = end_jitter_p
        self.cache_items = cache_items
        self._cache: OrderedDict = OrderedDict()
        self.bad: set[int] = set()

    def __len__(self) -> int:
        return len(self.clips)

    def item(self, i: int) -> Item | None:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        if i in self.bad:
            return None
        it = load_item(self.clips[i], self.labels[i], self.fcfg, with_alternatives=self.train)
        if it is None:
            self.bad.add(i)
            return None
        self._cache[i] = it
        if len(self._cache) > self.cache_items:
            self._cache.popitem(last=False)
        return it

    def features(self, i: int, rng: np.random.Generator | None = None) -> np.ndarray | None:
        it = self.item(i)
        if it is None:
            return None
        if not self.train:
            return featurize(slice_bf(it.bf, 0, it.default_end), self.fcfg)
        rng = rng or self.rng
        end = it.default_end
        if len(it.ends) > 1 and rng.random() < self.end_jitter_p:
            end = int(rng.choice(it.ends))
        return featurize(augment_strong(slice_bf(it.bf, 0, max(2, end)), rng, self.aug), self.fcfg)

    def batch(self, idx: list[int], rng: np.random.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
        xs, ys = [], []
        for i in idx:
            f = self.features(i, rng)
            if f is not None:
                xs.append(f)
                ys.append(self.labels[i])
        if len(xs) < 2:
            return None
        return torch.from_numpy(np.stack(xs).astype(np.float32)), torch.tensor(ys, dtype=torch.long)


def multi_signer_counts(clips: list[Clip]) -> dict[str, set[str]]:
    """concept -> distinct signer groups (ISLRTC counts as one group)."""
    out: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        if c.concept_id:
            out[c.concept_id].add(f"{c.source}:{c.group or c.clip_id}")
    return out
