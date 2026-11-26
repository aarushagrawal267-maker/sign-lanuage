"""Fingerspelling letter recogniser from every letter source of a profile (docs/ISL_DESIGN.md 6.4, 15.2 and 15.12).

Sources (``multisource.LETTER_SOURCES``):

* **ISL-Fingerspelling** (research): continuous fingerspelled names with per-letter frame boundaries and signer
  ids; letter examples = the middle half of each letter segment, "noise" = the frames of space segments and
  frames outside any letter (transitions near letter boundaries are skipped: their label is ambiguous);
* **OpenHands Indian fingerspelling** (research): short clips of one letter / digit; examples = middle frames;
* **static alphabet images** (Mendeley: commercial-safe; Hugging Face sets: research): one frame each (the motion
  part of the feature vector is zero, like a still hold).

The model is the existing ``LetterClassifier`` MLP over ``letter_features_seq`` frames (same features the live
speller uses on hold means). The checkpoint format is unchanged, so ``Speller`` / ``LetterPredictor`` load it and
the per-user alphabet calibration (``fingerspell/prototypes.py``) keeps working next to it.

**Content duplicates** (15.12). Before any feature is read, items are grouped by content (same landmark file and
slice, or same SHA-256 of the source file; the engine's ``duplicate_of`` rows reuse the original's landmark file, so
a ``duplicate_of`` chain is one content). A content filed under two different labels (Mendeley static has three such
photo pairs) is dropped **entirely**; a content repeated under one label is kept once (no double weight). Counts and
examples go into ``data.dedup``.

**Groups.** Every example has a *unit*: the held-out unit for evaluation and calibration.

* ISL-Fingerspelling: the signer (``isl_fingerspelling:signer_00``) - real people;
* Mendeley static: the **age group** (``mendeley_static:adults``). The manifest's group is age x sleeves, but the
  same person can appear in both sleeve groups, so sleeves are never split apart (they are the sampling subgroups);
* Hugging Face alphabets, OpenHands fingerspelling and anything else without group ids: the whole source
  (``hf_akrit_alphabet``). Such a unit is never held out: it always stays whole on the training side (akrit's own
  ``test`` rows are frames / re-crops of its training takes, so they are never a test set).

**E5 and calibration share one grouped hold-out**: leave-one-unit-out over every unit of a source that has at least
two grouped units. Each fold trains on everything except the held-out unit; its out-of-fold logits give E5 (letter
top-1 on the held-out letter examples, per unit, per source, pooled; word decoding for held-out signers) and,
pooled over the folds, the temperature. There is no early stopping or epoch selection (fixed epochs), so no other
choice is made on held-out data. A profile without such a source reports ``calibration.status = "not grouped"`` and
keeps temperature 1.0 (a random photo split would only measure near-copies of training photos).

**Sampling.** Every epoch draws up to ``max_per_class`` examples per class, split evenly over the class's subgroups
(signers, Mendeley age x sleeves groups, whole sources; a subgroup with fewer examples gives all it has and the rest
is shared by the others), cycling through a seeded permutation of each subgroup, so over the epochs every example is
used and no group dominates a class. ``training.coverage`` reports the fraction of examples seen at least once.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.features.build import FeatureError
from app.isl.data.sources import Clip
from app.isl.fingerspell.features import LETTER_DIM, letter_features_seq
from app.isl.fingerspell.model import LETTERS, NOISE, LetterClassifier
from app.isl.models.calibration import fit_temperature

log = logging.getLogger(__name__)

DIGIT_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "nine": "9"}

SIGNER_SOURCES = frozenset({"isl_fingerspelling"})  # Clip.group is a real signer id
AGE_GROUP_SOURCES = frozenset({"mendeley_static"})  # Clip.group = "static-<age>-<sleeves>"
_STATIC_GROUP = re.compile(r"static-([a-z]+)-(.+)")
WHOLE = "whole_source"


def symbol_of(text: str | None) -> str | None:
    """'a' / 'A' / 'R' -> 'A'; '7' / 'seven' -> '7'; anything else (Hindi vowels, 'E1', words) -> None."""
    t = (text or "").strip()
    if len(t) == 1 and t.isascii() and t.isalnum():
        return t.upper()
    return DIGIT_WORDS.get(t.lower())


def unit_of(clip: Clip) -> tuple[str, str, str]:
    """``(unit, kind, subgroup)``: the held-out unit, its kind (``signer`` / ``age_group`` / ``whole_source``) and
    the finer sampling subgroup."""
    g = clip.group or ""
    if clip.source in SIGNER_SOURCES and g:
        u = f"{clip.source}:{g}"
        return u, "signer", u
    if clip.source in AGE_GROUP_SOURCES:
        m = _STATIC_GROUP.fullmatch(g)
        if m:
            return f"{clip.source}:{m.group(1)}", "age_group", f"{clip.source}:{m.group(1)}-{m.group(2)}"
    return clip.source, WHOLE, clip.source


@dataclass
class LetterSet:
    X: list[np.ndarray] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    group: list[str] = field(default_factory=list)  # held-out unit (see unit_of)
    source: list[str] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)  # held-out word tests: {clip, start, end, word, group}
    skipped: Counter = field(default_factory=Counter)
    sub: list[str] = field(default_factory=list)  # sampling subgroup
    unit_kind: dict = field(default_factory=dict)  # unit -> signer | age_group | whole_source
    dedup: dict = field(default_factory=dict)

    def add(self, feats: np.ndarray, label: int, group: str, source: str, sub: str | None = None) -> None:
        for f in np.atleast_2d(feats):
            self.X.append(np.asarray(f, np.float32))
            self.y.append(label)
            self.group.append(group)
            self.source.append(source)
            self.sub.append(sub or group)

    def subgroups(self) -> list[str]:
        return self.sub if len(self.sub) == len(self.y) else list(self.group)

    def kind(self, unit: str) -> str:
        if unit in self.unit_kind:
            return self.unit_kind[unit]
        return "signer" if any(w["group"] == unit for w in self.words) else WHOLE

    def summary(self) -> dict:
        units = Counter(self.group)
        return {"examples": len(self.y), "letters": len({LETTERS[i] for i in self.y if i != NOISE}),
                "noise": sum(1 for i in self.y if i == NOISE), "signers": len({w["group"] for w in self.words}),
                "groups": {u: {"kind": self.kind(u), "examples": n} for u, n in sorted(units.items())},
                "per_source": dict(Counter(self.source)), "words": len(self.words), "skipped": dict(self.skipped),
                "dedup": self.dedup}


def _mid(a: int, b: int, frac: float = 0.5) -> tuple[int, int]:
    n = b - a
    k = max(1, int(round(n * frac)))
    s = a + (n - k) // 2
    return s, s + k


# --------------------------------------------------------------------------------------------- content duplicates

def _label_key(clip: Clip) -> str:
    sym = symbol_of(clip.raw_gloss)
    return sym if sym is not None else (clip.raw_gloss or "").strip().lower()


def _content_keys(clip: Clip) -> list[str]:
    """Same landmark file and slice = same content (``duplicate_of`` rows point at the original's file); same
    SHA-256 of the whole source file = same content (copies extracted twice, or the same photo in two sources)."""
    keys = [f"path:{clip.path}#{clip.start}:{clip.end}"]
    sha = (clip.quality or {}).get("sha256")
    if sha and not clip.start and clip.end is None:
        keys.append(f"sha:{sha}")
    return keys


def dedupe_by_content(clips: list[Clip], max_examples: int = 20) -> tuple[list[bool], dict]:
    """Keep mask over ``clips`` and counts. A content under more than one label is dropped in full (every copy);
    a content repeated under one label keeps its first item (manifest order: the original before its copies)."""
    parent = list(range(len(clips)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, c in enumerate(clips):
        for k in _content_keys(c):
            j = first.setdefault(k, i)
            if j != i:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(clips)):
        comps[find(i)].append(i)
    keep = [True] * len(clips)
    st = {"items": len(clips), "cross_label_contents": 0, "cross_label_items": 0, "same_label_copies": 0,
          "cross_label_examples": [], "per_source": {}}
    per_src: dict[str, Counter] = defaultdict(Counter)
    for members in comps.values():
        if len(members) == 1:
            continue
        labels = sorted({_label_key(clips[i]) for i in members})
        if len(labels) > 1:
            st["cross_label_contents"] += 1
            st["cross_label_items"] += len(members)
            for i in members:
                keep[i] = False
                per_src[clips[i].source]["cross_label_items"] += 1
            if len(st["cross_label_examples"]) < max_examples:
                st["cross_label_examples"].append({"ids": [clips[i].clip_id for i in members], "labels": labels})
            log.warning("letter set: dropped %d items with the same content under labels %s: %s", len(members),
                        labels, ", ".join(clips[i].clip_id for i in members))
        else:
            for i in sorted(members)[1:]:
                keep[i] = False
                st["same_label_copies"] += 1
                per_src[clips[i].source]["same_label_copies"] += 1
    st["per_source"] = {k: dict(v) for k, v in sorted(per_src.items())}
    st["kept"] = sum(keep)
    if st["cross_label_items"] or st["same_label_copies"]:
        log.info("letter set dedup: %d contents under two or more labels (%d items dropped), %d same-label copies "
                 "dropped, %d of %d items kept", st["cross_label_contents"], st["cross_label_items"],
                 st["same_label_copies"], st["kept"], st["items"])
    return keep, st


def collect_letter_examples(clips_with_letters: list[tuple[Clip, list[dict]]], letter_clips: list[Clip],
                            image_clips: list[Clip], noise_stride: int = 3, dedupe: bool = True) -> LetterSet:
    ls = LetterSet()
    allc = [c for c, _ in clips_with_letters] + list(letter_clips) + list(image_clips)
    keep, ls.dedup = dedupe_by_content(allc) if dedupe else ([True] * len(allc), {})
    k_fs = keep[:len(clips_with_letters)]
    k_cl = keep[len(clips_with_letters):len(clips_with_letters) + len(letter_clips)]
    k_im = keep[len(clips_with_letters) + len(letter_clips):]

    def unit(clip: Clip) -> tuple[str, str]:
        u, kind, sub = unit_of(clip)
        ls.unit_kind.setdefault(u, kind)
        return u, sub

    for (clip, segs), ok in zip(clips_with_letters, k_fs):
        if not ok:
            continue
        try:
            feats, _ = letter_features_seq(clip.load())
        except (FeatureError, OSError, ValueError):
            ls.skipped["fs_no_body"] += 1
            continue
        T = len(feats)
        g, sg = unit(clip)
        covered = np.zeros(T, bool)
        word, wstart = [], None
        for s in segs:
            a, b = max(0, min(T, s["start"])), max(0, min(T, s["end"]))
            if b <= a:
                continue
            covered[a:b] = True
            sym = symbol_of(s["letter"])
            if s["letter"].strip() == "":
                ls.add(feats[a:b:noise_stride], NOISE, g, clip.source, sg)
                if word:
                    ls.words.append({"clip": clip, "start": wstart, "end": a, "word": "".join(word), "group": g})
                word, wstart = [], None
                continue
            if sym is None:
                ls.skipped["fs_unknown_letter"] += 1
                continue
            m0, m1 = _mid(a, b)
            ls.add(feats[m0:m1], LETTERS.index(sym), g, clip.source, sg)
            if wstart is None:
                wstart = a
            word.append(sym)
        if word:
            ls.words.append({"clip": clip, "start": wstart, "end": T, "word": "".join(word), "group": g})
        out = np.flatnonzero(~covered)[::noise_stride]
        if len(out):
            ls.add(feats[out], NOISE, g, clip.source, sg)
    for clip, ok in zip(letter_clips, k_cl):
        if not ok:
            continue
        sym = symbol_of(clip.raw_gloss)
        if sym is None:
            ls.skipped["clip_unknown_label"] += 1
            continue
        try:
            feats, _ = letter_features_seq(clip.load())
        except (FeatureError, OSError, ValueError):
            ls.skipped["clip_no_body"] += 1
            continue
        m0, m1 = _mid(0, len(feats))
        g, sg = unit(clip)
        ls.add(feats[m0:m1], LETTERS.index(sym), g, clip.source, sg)
    for clip, ok in zip(image_clips, k_im):
        if not ok:
            continue
        sym = symbol_of(clip.raw_gloss)
        if sym is None:
            ls.skipped["image_unknown_label"] += 1
            continue
        try:
            feats, _ = letter_features_seq(clip.load())
        except (FeatureError, OSError, ValueError):
            ls.skipped["image_no_body"] += 1
            continue
        g, sg = unit(clip)
        ls.add(feats[:1], LETTERS.index(sym), g, clip.source, sg)
    return ls


# --------------------------------------------------------------------------------------------- balanced sampling

def _water_fill(sizes: list[int], quota: int, start: int = 0) -> list[int]:
    """Split ``quota`` as evenly as possible over groups of the given sizes (a small group gives all it has); the
    remainder goes to the groups from ``start`` on (rotated every epoch, so no group is always short)."""
    alloc = [0] * len(sizes)
    left = min(quota, sum(sizes))
    order = [(start + i) % len(sizes) for i in range(len(sizes))] if sizes else []
    while left > 0:
        open_ = [i for i in order if alloc[i] < sizes[i]]
        share = max(1, left // len(open_))
        for i in open_:
            t = min(share, sizes[i] - alloc[i], left)
            alloc[i] += t
            left -= t
            if left == 0:
                break
    return alloc


class BalancedEpochs:
    """Per epoch: up to ``per_class`` examples of every class (all of them with ``per_class`` None / 0), split
    evenly over the class's subgroups; each subgroup is read cyclically through a seeded permutation (re-shuffled
    when it wraps), so successive epochs use different examples until all have been used."""

    def __init__(self, y: np.ndarray, sub: np.ndarray, per_class: int | None, rng: np.random.Generator):
        self.rng = rng
        self.classes: list[tuple[int, list[list]]] = []  # (class, cells [indices (permuted), cursor])
        self.quota: dict[int, int] = {}
        self.n_epoch = 0
        for c in sorted({int(v) for v in y}):
            groups = defaultdict(list)
            for i in np.flatnonzero(y == c):
                groups[sub[i]].append(i)
            cells = [[rng.permutation(np.asarray(groups[n])), 0] for n in sorted(groups)]
            n_c = sum(len(cl[0]) for cl in cells)
            self.quota[c] = n_c if not per_class else min(per_class, n_c)
            self.classes.append((c, cells))
        self.seen = np.zeros(len(y), bool)

    def _take(self, cell: list, take: int) -> np.ndarray:
        idx, cur = cell
        if cur + take <= len(idx):
            cell[1] = cur + take
            return idx[cur:cur + take]
        head = idx[cur:]
        cell[0] = idx = self.rng.permutation(idx)
        cell[1] = take - len(head)
        return np.concatenate([head, idx[:cell[1]]])

    def epoch(self) -> np.ndarray:
        out = []
        for c, cells in self.classes:
            alloc = _water_fill([len(cl[0]) for cl in cells], self.quota[c], start=self.n_epoch)
            out += [self._take(cl, k) for cl, k in zip(cells, alloc) if k]
        self.n_epoch += 1
        e = np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)
        self.seen[e] = True
        return e

    @property
    def size(self) -> int:
        return int(sum(self.quota.values()))


def _fit_mlp(X: np.ndarray, y: np.ndarray, epochs: int, seed: int, max_per_class: int | None,
             sub: np.ndarray | None = None, noise_sd: float = 0.03, batch: int = 128,
             stats: dict | None = None) -> LetterClassifier:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    sub = np.asarray(sub) if sub is not None else np.zeros(len(y), dtype=object)
    plan = BalancedEpochs(y, sub, max_per_class, rng)
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    yt = torch.from_numpy(np.asarray(y, np.int64))
    counts = np.zeros(len(LETTERS))
    for c, q in plan.quota.items():
        counts[c] = q
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / max(1, (counts > 0).sum()), 0.0)
    model = LetterClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=0.05)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        ep = plan.epoch()
        perm = torch.from_numpy(ep[rng.permutation(len(ep))])
        for a in range(0, len(perm), batch):
            idx = perm[a:a + batch]
            xb = Xt[idx] + noise_sd * torch.randn_like(Xt[idx])
            loss = crit(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    if stats is not None:
        secs = time.time() - t0
        stats.update({"examples": int(len(y)), "epoch_examples": plan.size, "epochs": epochs,
                      "max_per_class": max_per_class, "seconds": round(secs, 1),
                      "seconds_per_epoch": round(secs / max(1, epochs), 2),
                      "coverage": round(float(plan.seen.mean()) if len(y) else 0.0, 4)})
    return model


@torch.no_grad()
def _logits(model: LetterClassifier, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, len(LETTERS)), np.float32)
    return model(torch.from_numpy(np.asarray(X, np.float32))).numpy()


def _save(model: LetterClassifier, out: Path, meta: dict) -> None:
    from app.isl.training.loop import atomic_save

    atomic_save({"state_dict": model.state_dict(), "meta": meta}, out)


# --------------------------------------------------------------------------------------------- grouped hold-out

def grouped_units(ls: LetterSet) -> tuple[list[str], str | None]:
    """The held-out units of E5 and calibration: every unit of a source with at least two grouped units (signers,
    age groups). Whole-source units never leave the training side. ``([], reason)`` when there is none."""
    by_src: dict[str, set] = defaultdict(set)
    whole: set[str] = set()
    letters_in = Counter(u for u, c in zip(ls.group, ls.y) if c != NOISE)
    for u, s in set(zip(ls.group, ls.source)):
        if ls.kind(u) == WHOLE:
            whole.add(s)
        elif letters_in[u]:
            by_src[s].add(u)
    units = sorted(u for us in by_src.values() if len(us) >= 2 for u in us)
    if units:
        return units, None
    have = ", ".join(f"{s}: {len(us)}" for s, us in sorted(by_src.items())) or "none"
    return [], ("no letter source with at least 2 groups (signers or age groups) - grouped hold-out needs at least 2 "
                f"per source; grouped units per source: {have}"
                + (f"; without group ids (kept whole on the training side): {', '.join(sorted(whole))}" if whole else ""))


def _ece(logits: np.ndarray, y: np.ndarray, T: float, bins: int = 15) -> float:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(float((pred[m] == y[m]).mean()) - float(conf[m].mean()))
    return float(e)


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))


MIN_CAL_EXAMPLES = 200  # below this many held-out examples a fitted temperature is noise: keep 1.0


def calibrate(logits: np.ndarray, y: np.ndarray, units: list[str], how: str, min_examples: int | None = None) -> dict:
    """Temperature on out-of-group logits (``units`` = the held-out units they come from)."""
    min_examples = MIN_CAL_EXAMPLES if min_examples is None else min_examples
    if len(y) == 0:
        return {"status": "not grouped", "temperature": 1.0, "reason": "no held-out examples"}
    if len(y) < min_examples:
        return {"status": "too few held-out examples", "temperature": 1.0, "n": int(len(y)),
                "n_groups": len(set(units)), "held_out_units": sorted(set(units)),
                "reason": f"{len(y)} held-out examples from grouped units (< {min_examples}): temperature kept at 1.0"}
    T = fit_temperature(logits, y)
    return {"status": "grouped", "temperature": T, "how": how, "held_out_units": sorted(set(units)), "n": int(len(y)),
            "n_groups": len(set(units)), "nll_t1": round(_nll(logits, y, 1.0), 4), "nll": round(_nll(logits, y, T), 4),
            "ece_t1": round(_ece(logits, y, 1.0), 4), "ece": round(_ece(logits, y, T), 4)}


def evaluate_grouped(ls: LetterSet, epochs: int, seed: int, max_per_class: int | None, lexicon_words: list[str],
                     units: list[str] | None = None, max_words: int = 300) -> tuple[dict, dict]:
    """E5: leave-one-unit-out over :func:`grouped_units`. Returns ``(e5, oof)`` where ``oof`` holds the
    out-of-fold logits / labels / units of every held-out example (letters and noise) for the temperature."""
    from app.isl.evaluation.report import grouped_bootstrap, interval_fields, interval_reason
    from app.isl.fingerspell.decoder import LexiconTrie, SpellDecoder
    from app.isl.fingerspell.holds import detect_holds

    X = np.stack(ls.X) if ls.X else np.zeros((0, LETTER_DIM), np.float32)
    y, g, src = np.asarray(ls.y), np.asarray(ls.group), np.asarray(ls.source)
    sub = np.asarray(ls.subgroups())
    reason = None
    if units is None:
        units, reason = grouped_units(ls)
    oof = {"logits": [], "y": [], "units": []}
    if not units:
        return {"measurable": False, "reason": reason or "no grouped units", "n": 0, "metrics": {}}, oof
    per_letter = defaultdict(lambda: [0, 0])
    hits, hit_units, hit_src = [], [], []
    word_rows = []
    trie = LexiconTrie([(w, 0.0) for w in sorted(set(lexicon_words) | {w["word"] for w in ls.words})])
    dec = SpellDecoder(trie)
    folds = {}
    for u in units:
        tr, te = g != u, g == u
        if tr.sum() < 10 or not te.any() or len(set(y[tr].tolist()) - {NOISE}) < 2:
            folds[u] = {"skipped": True}
            continue
        fst: dict = {}
        m = _fit_mlp(X[tr], y[tr], epochs, seed, max_per_class, sub[tr], stats=fst)
        lg = _logits(m, X[te])
        oof["logits"].append(lg)
        oof["y"].append(y[te])
        oof["units"] += [u] * int(te.sum())
        let = y[te] != NOISE
        pred = lg[let].copy()
        pred[:, NOISE] = -1e9  # letter frames: which letter, not whether it is a letter
        hit = pred.argmax(1) == y[te][let]
        for yy, h in zip(y[te][let], hit):
            per_letter[LETTERS[yy]][0] += int(h)
            per_letter[LETTERS[yy]][1] += 1
        hits += hit.astype(float).tolist()
        hit_units += [u] * len(hit)
        hit_src += src[te][let].tolist()
        folds[u] = {"kind": ls.kind(u), "n": int(len(hit)), "letter_top1": float(hit.mean()) if len(hit) else None,
                    "train_examples": int(tr.sum()), "seconds": fst.get("seconds")}
        for w in [w for w in ls.words if w["group"] == u][:max_words]:
            try:
                raw = w["clip"].load().slice(w["start"], w["end"])
                feats, _ = letter_features_seq(raw)
                holds = detect_holds(raw, feats=feats)
            except (FeatureError, OSError, ValueError):
                continue
            if not holds:
                word_rows.append({"group": u, "top1": 0.0, "top5": 0.0, "raw": 0.0})
                continue
            lp = torch.log_softmax(torch.from_numpy(_logits(m, np.stack([h.mean_features for h in holds]))), -1).numpy()
            res = dec.decode(list(lp), k=5)
            words = [x[0].upper() for x in res]
            raw_s = "".join(LETTERS[int(np.argmax(r))] for r in lp if int(np.argmax(r)) != NOISE)
            word_rows.append({"group": u, "top1": float(bool(words) and words[0] == w["word"]),
                              "top5": float(w["word"] in words[:5]), "raw": float(raw_s == w["word"])})
    oof = {"logits": np.concatenate(oof["logits"]) if oof["logits"] else np.zeros((0, len(LETTERS)), np.float32),
           "y": np.concatenate(oof["y"]) if oof["y"] else np.zeros(0, np.int64), "units": oof["units"]}
    done = [u for u in units if not folds[u].get("skipped")]
    if not hits:
        return {"measurable": False, "reason": "no held-out group had letter examples and a trainable remainder",
                "n": 0, "metrics": {}, "folds": folds}, oof
    series = {"letter_top1": (hits, hit_units)}
    metrics = {"letter_top1": float(np.mean(hits))}
    hs, hu = np.asarray(hits), np.asarray(hit_units)
    hsrc = np.asarray(hit_src)
    for s in sorted(set(hit_src)):
        m_ = hsrc == s
        metrics[f"letter_top1_{s}"] = float(hs[m_].mean())
        series[f"letter_top1_{s}"] = (hs[m_].tolist(), hu[m_].tolist())
    metrics["letter_top1_group_mean"] = float(np.mean([folds[u]["letter_top1"] for u in done
                                                       if folds[u].get("letter_top1") is not None]))
    out = {"measurable": True, "groups": done, "group_kinds": {u: ls.kind(u) for u in done},
           "signers": [u for u in done if ls.kind(u) == "signer"], "n": len(hits), "metrics": metrics,
           **interval_fields(series), "folds": folds,
           "per_letter": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(per_letter.items())},
           "protocol": "leave-one-group-out: ISL-Fingerspelling signers and Mendeley static age groups are held out "
                       "one at a time; sources without group ids (HF alphabets, OpenHands fingerspelling) always stay "
                       "whole on the training side; letter top-1 on the held-out group's letter examples (noise "
                       "masked); exact-content duplicates removed before the split"}
    if word_rows:
        out["words"] = {"n": len(word_rows),
                        "top1": float(np.mean([r["top1"] for r in word_rows])),
                        "top5": float(np.mean([r["top5"] for r in word_rows])),
                        "raw_letters_exact": float(np.mean([r["raw"] for r in word_rows])),
                        "top1_interval": grouped_bootstrap([r["top1"] for r in word_rows], [r["group"] for r in word_rows]),
                        "top1_interval_reason": interval_reason([r["top1"] for r in word_rows],
                                                                [r["group"] for r in word_rows]),
                        "n_groups": len({r["group"] for r in word_rows}),
                        "label": "held-out signer; word boundaries from the annotation spaces; letters from the live "
                                 "hold detector; lexicon = spelling lexicon + every annotated word (closed vocabulary). "
                                 "Spelled words are only ever suggested in the app, never auto-accepted."}
        out["metrics"].update(word_top1=out["words"]["top1"], word_top5=out["words"]["top5"])
    return out, oof


def calibration_holdout(ls: LetterSet, min_examples: int = 200) -> list[str]:
    """One unit per grouped source (the smallest with at least ``min_examples`` letter examples, else the largest):
    the hold-out used for the temperature when E5 is not run."""
    units, _ = grouped_units(ls)
    n = Counter(u for u, c in zip(ls.group, ls.y) if c != NOISE)
    by_src = defaultdict(list)
    for u in units:
        by_src[u.split(":", 1)[0]].append(u)
    out = []
    for us in by_src.values():
        big = [u for u in us if n[u] >= min_examples]
        out.append(min(big, key=lambda u: (n[u], u)) if big else max(us, key=lambda u: (n[u], u)))
    return sorted(out)


def evaluate_loso(ls: LetterSet, epochs: int, seed: int, max_per_class: int | None, lexicon_words: list[str],
                  max_words: int = 300) -> dict:
    """Backward-compatible name of :func:`evaluate_grouped` (E5 only)."""
    return evaluate_grouped(ls, epochs, seed, max_per_class, lexicon_words, max_words=max_words)[0]


def train_fingerspell(ls: LetterSet, out: Path, *, profile: str, epochs: int = 40, seed: int = 0,
                      max_per_class: int | None = 4000, lexicon_words: list[str] | None = None, evaluate: bool = True,
                      smoke: bool = False) -> dict:
    t0 = time.time()
    if len({c for c in ls.y if c != NOISE}) < 2:
        return {"skipped": True, "reason": "fewer than two letters with examples in this profile's letter sources",
                **ls.summary()}
    X, y = np.stack(ls.X), np.asarray(ls.y)
    g, sub = np.asarray(ls.group), np.asarray(ls.subgroups())
    units, why = grouped_units(ls)
    if evaluate:
        e5, oof = evaluate_grouped(ls, epochs, seed, max_per_class, lexicon_words or [], units=units)
        if not e5.get("measurable") and why:
            e5["reason"] = why
        cal = (calibrate(oof["logits"], oof["y"], oof["units"], "out-of-fold logits of the E5 folds")
               if len(oof["y"]) else {"status": "not grouped", "temperature": 1.0, "reason": e5.get("reason") or why})
    else:
        e5 = {"measurable": False, "reason": "not evaluated"}
        hold_units = calibration_holdout(ls) if units else []
        hold = np.isin(g, hold_units)
        if hold.any() and (~hold).sum() > 10 and len(set(y[~hold].tolist()) - {NOISE}) >= 2:
            m_cal = _fit_mlp(X[~hold], y[~hold], epochs, seed, max_per_class, sub[~hold])
            cal = calibrate(_logits(m_cal, X[hold]), y[hold], g[hold].tolist(), "one held-out group per grouped source")
        else:
            cal = {"status": "not grouped", "temperature": 1.0,
                   "reason": why or "the grouped hold-out leaves no trainable remainder"}
    if cal["status"] != "grouped":
        log.warning("letter temperature not grouped (kept 1.0): %s", cal.get("reason"))
    T = float(cal["temperature"])
    fit_stats: dict = {}
    model = _fit_mlp(X, y, epochs, seed, max_per_class, sub, stats=fit_stats)
    signers = max(1, len({w["group"] for w in ls.words}))  # known signer ids only (other sources: unknown)
    meta = {"kind": "isl-letters", "labels": LETTERS, "in_dim": LETTER_DIM, "hidden": [128, 64], "temperature": T,
            "trained_letters": sorted({LETTERS[i] for i in y if i != NOISE}), "n_signers": signers,
            "source": ",".join(sorted(set(ls.source))), "profile": profile, "data": ls.summary(), "e5": e5,
            "calibration": cal, "training": fit_stats, "smoke": smoke, "n_examples": int(len(y))}
    _save(model, out, meta)
    return {"checkpoint": str(out), "temperature": T, "letters": meta["trained_letters"], "n_signers": signers,
            "data": ls.summary(), "e5": {k: v for k, v in e5.items() if k != "per_letter"}, "calibration": cal,
            "training": fit_stats, "seconds": round(time.time() - t0, 1)}


def letter_data(paths, profile: str, vocab=None, report: dict | None = None) -> LetterSet:
    """Read every letter source a profile allows (missing sources are skipped and reported). ``duplicate_of`` rows
    are read too, so that :func:`dedupe_by_content` sees every label a content was filed under."""
    from app.isl.training.multisource import LETTER_SOURCES, open_adapter

    fs, clips, images = [], [], []
    for name in LETTER_SOURCES[profile]:
        ad = open_adapter(name, paths, vocab, profile)
        if report is not None:
            report[name] = {"available": ad is not None}
        if ad is None:
            continue
        if hasattr(ad, "include_duplicates"):
            ad.include_duplicates = True
        if name == "isl_fingerspelling":
            from app.isl.data.video_sources.adapters import letter_samples

            got = list(letter_samples(ad))
            fs += got
            n = len(got)
        else:
            got = list(ad.clips())
            (images if all(int(c.quality.get("n_frames") or 1) <= 1 for c in got[:50]) else clips).extend(got)
            n = len(got)
        if report is not None:
            report[name]["clips"] = n
    return collect_letter_examples(fs, clips, images)
