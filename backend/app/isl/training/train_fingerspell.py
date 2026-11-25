"""Fingerspelling letter recogniser from every letter source of a profile (docs/ISL_DESIGN.md 6.4 and 15.2).

Sources (``multisource.LETTER_SOURCES``):

* **ISL-Fingerspelling** (research): continuous fingerspelled names with per-letter frame boundaries and signer
  ids; letter examples = the middle half of each letter segment, "noise" = the frames of space segments and
  frames outside any letter (transitions near letter boundaries are skipped: their label is ambiguous);
* **OpenHands Indian fingerspelling** (research): short clips of one letter / digit; examples = middle frames;
* **static alphabet images** (Mendeley: commercial-safe; Hugging Face sets: research): one frame each (the motion
  part of the feature vector is zero, like a still hold).

The model is the existing ``LetterClassifier`` MLP over ``letter_features_seq`` frames (same features the live
speller uses on hold means), trained class-balanced with feature noise, temperature-scaled on held-out signers'
frames. The checkpoint format is unchanged, so ``Speller`` / ``LetterPredictor`` load it and the per-user
alphabet calibration (``fingerspell/prototypes.py``) keeps working next to it.

Evaluation (E5) is **leave-one-signer-out** over the sources with signer ids (ISL-Fingerspelling): per-letter
accuracy on the held-out signer's letter frames, and word-level decoding of the held-out signer's words
(word boundaries taken from the annotation spaces, letters from the live hold detector + model, then the
dictionary-guided ``SpellDecoder``). With fewer than two signers E5 reports "not measurable".
"""

from __future__ import annotations

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

DIGIT_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "nine": "9"}


def symbol_of(text: str | None) -> str | None:
    """'a' / 'A' / 'R' -> 'A'; '7' / 'seven' -> '7'; anything else (Hindi vowels, 'E1', words) -> None."""
    t = (text or "").strip()
    if len(t) == 1 and t.isascii() and t.isalnum():
        return t.upper()
    return DIGIT_WORDS.get(t.lower())


@dataclass
class LetterSet:
    X: list[np.ndarray] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    group: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)  # held-out word tests: {clip, start, end, word, group}
    skipped: Counter = field(default_factory=Counter)

    def add(self, feats: np.ndarray, label: int, group: str, source: str) -> None:
        for f in np.atleast_2d(feats):
            self.X.append(np.asarray(f, np.float32))
            self.y.append(label)
            self.group.append(group)
            self.source.append(source)

    def summary(self) -> dict:
        return {"examples": len(self.y), "letters": len({LETTERS[i] for i in self.y if i != NOISE}),
                "noise": sum(1 for i in self.y if i == NOISE), "signers": len(set(self.group)),
                "per_source": dict(Counter(self.source)), "words": len(self.words), "skipped": dict(self.skipped)}


def _mid(a: int, b: int, frac: float = 0.5) -> tuple[int, int]:
    n = b - a
    k = max(1, int(round(n * frac)))
    s = a + (n - k) // 2
    return s, s + k


def collect_letter_examples(clips_with_letters: list[tuple[Clip, list[dict]]], letter_clips: list[Clip],
                            image_clips: list[Clip], noise_stride: int = 3) -> LetterSet:
    ls = LetterSet()
    for clip, segs in clips_with_letters:
        try:
            feats, _ = letter_features_seq(clip.load())
        except (FeatureError, OSError, ValueError):
            ls.skipped["fs_no_body"] += 1
            continue
        T = len(feats)
        g = clip.group or clip.clip_id
        covered = np.zeros(T, bool)
        word, wstart = [], None
        for s in segs:
            a, b = max(0, min(T, s["start"])), max(0, min(T, s["end"]))
            if b <= a:
                continue
            covered[a:b] = True
            sym = symbol_of(s["letter"])
            if s["letter"].strip() == "":
                ls.add(feats[a:b:noise_stride], NOISE, g, clip.source)
                if word:
                    ls.words.append({"clip": clip, "start": wstart, "end": a, "word": "".join(word), "group": g})
                word, wstart = [], None
                continue
            if sym is None:
                ls.skipped["fs_unknown_letter"] += 1
                continue
            m0, m1 = _mid(a, b)
            ls.add(feats[m0:m1], LETTERS.index(sym), g, clip.source)
            if wstart is None:
                wstart = a
            word.append(sym)
            covered[a:b] = True
        if word:
            ls.words.append({"clip": clip, "start": wstart, "end": T, "word": "".join(word), "group": g})
        out = np.flatnonzero(~covered)[::noise_stride]
        if len(out):
            ls.add(feats[out], NOISE, g, clip.source)
    for clip in letter_clips:
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
        ls.add(feats[m0:m1], LETTERS.index(sym), clip.group or clip.clip_id, clip.source)
    for clip in image_clips:
        sym = symbol_of(clip.raw_gloss)
        if sym is None:
            ls.skipped["image_unknown_label"] += 1
            continue
        try:
            feats, _ = letter_features_seq(clip.load())
        except (FeatureError, OSError, ValueError):
            ls.skipped["image_no_body"] += 1
            continue
        ls.add(feats[:1], LETTERS.index(sym), clip.group or clip.source, clip.source)
    return ls


def _fit_mlp(X: np.ndarray, y: np.ndarray, epochs: int, seed: int, max_per_class: int, noise_sd: float = 0.03,
             batch: int = 128) -> LetterClassifier:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # class balance: cap every class, then sample batches uniformly over the capped set
    by = defaultdict(list)
    for i, c in enumerate(y):
        by[int(c)].append(i)
    keep = np.concatenate([rng.choice(v, min(len(v), max_per_class), replace=False) for v in by.values()])
    Xt = torch.from_numpy(X[keep].astype(np.float32))
    yt = torch.from_numpy(y[keep].astype(np.int64))
    counts = np.bincount(y[keep], minlength=len(LETTERS)).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / max(1, (counts > 0).sum()), 0.0)
    model = LetterClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=0.05)
    for _ in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(Xt)))
        for a in range(0, len(perm), batch):
            idx = perm[a:a + batch]
            xb = Xt[idx] + noise_sd * torch.randn_like(Xt[idx])
            loss = crit(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _logits(model: LetterClassifier, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, len(LETTERS)), np.float32)
    return model(torch.from_numpy(np.asarray(X, np.float32))).numpy()


def _save(model: LetterClassifier, out: Path, meta: dict) -> None:
    from app.isl.training.loop import atomic_save

    atomic_save({"state_dict": model.state_dict(), "meta": meta}, out)


def evaluate_loso(ls: LetterSet, epochs: int, seed: int, max_per_class: int, lexicon_words: list[str],
                  max_words: int = 300) -> dict:
    """E5: leave-one-signer-out over the signers of continuous fingerspelling (sources with letter segments)."""
    from app.isl.fingerspell.decoder import LexiconTrie, SpellDecoder
    from app.isl.fingerspell.holds import detect_holds

    X, y = np.stack(ls.X) if ls.X else np.zeros((0, LETTER_DIM), np.float32), np.asarray(ls.y)
    g = np.asarray(ls.group)
    signers = sorted({w["group"] for w in ls.words})
    if len(signers) < 2:
        return {"measurable": False, "reason": f"{len(signers)} signer(s) with letter-annotated continuous "
                "fingerspelling; leave-one-signer-out needs at least 2", "n": 0, "metrics": {}}
    per_letter = defaultdict(lambda: [0, 0])
    letter_hits, letter_groups = [], []
    word_rows = []
    trie = LexiconTrie([(w, 0.0) for w in sorted(set(lexicon_words) | {w["word"] for w in ls.words})])
    dec = SpellDecoder(trie)
    for s in signers:
        tr, te = g != s, (g == s) & (y != NOISE)
        if tr.sum() < 10 or te.sum() == 0:
            continue
        m = _fit_mlp(X[tr], y[tr], epochs, seed, max_per_class)
        pred = _logits(m, X[te])
        pred[:, NOISE] = -1e9  # letter frames: which letter, not whether it is a letter
        hit = pred.argmax(1) == y[te]
        for yy, h in zip(y[te], hit):
            per_letter[LETTERS[yy]][0] += int(h)
            per_letter[LETTERS[yy]][1] += 1
        letter_hits += hit.astype(float).tolist()
        letter_groups += [s] * len(hit)
        for w in [w for w in ls.words if w["group"] == s][:max_words]:
            try:
                raw = w["clip"].load().slice(w["start"], w["end"])
                feats, _ = letter_features_seq(raw)
                holds = detect_holds(raw, feats=feats)
            except (FeatureError, OSError, ValueError):
                continue
            if not holds:
                word_rows.append({"group": s, "top1": 0.0, "top5": 0.0, "raw": 0.0})
                continue
            lp = torch.log_softmax(torch.from_numpy(_logits(m, np.stack([h.mean_features for h in holds]))), -1).numpy()
            res = dec.decode(list(lp), k=5)
            words = [x[0].upper() for x in res]
            raw_s = "".join(LETTERS[int(np.argmax(r))] for r in lp if int(np.argmax(r)) != NOISE)
            word_rows.append({"group": s, "top1": float(bool(words) and words[0] == w["word"]),
                              "top5": float(w["word"] in words[:5]), "raw": float(raw_s == w["word"])})
    from app.isl.evaluation.report import grouped_bootstrap

    out = {"measurable": True, "signers": signers, "n": len(letter_hits),
           "metrics": {"letter_top1": float(np.mean(letter_hits)) if letter_hits else None},
           "intervals": {"letter_top1": grouped_bootstrap(letter_hits, letter_groups)},
           "per_letter": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(per_letter.items())}}
    if word_rows:
        out["words"] = {"n": len(word_rows),
                        "top1": float(np.mean([r["top1"] for r in word_rows])),
                        "top5": float(np.mean([r["top5"] for r in word_rows])),
                        "raw_letters_exact": float(np.mean([r["raw"] for r in word_rows])),
                        "top1_interval": grouped_bootstrap([r["top1"] for r in word_rows], [r["group"] for r in word_rows]),
                        "label": "held-out signer; word boundaries from the annotation spaces; letters from the live "
                                 "hold detector; lexicon = spelling lexicon + every annotated word (closed vocabulary). "
                                 "Spelled words are only ever suggested in the app, never auto-accepted."}
        out["metrics"].update(word_top1=out["words"]["top1"], word_top5=out["words"]["top5"])
    return out


def train_fingerspell(ls: LetterSet, out: Path, *, profile: str, epochs: int = 40, seed: int = 0,
                      max_per_class: int = 4000, lexicon_words: list[str] | None = None, evaluate: bool = True,
                      smoke: bool = False) -> dict:
    t0 = time.time()
    if len({c for c in ls.y if c != NOISE}) < 2:
        return {"skipped": True, "reason": "fewer than two letters with examples in this profile's letter sources",
                **ls.summary()}
    X, y = np.stack(ls.X), np.asarray(ls.y)
    g = np.asarray(ls.group)
    e5 = evaluate_loso(ls, epochs, seed, max_per_class, lexicon_words or []) if evaluate else {"measurable": False}
    # temperature on one held-out group (largest signer group of the annotated data, else a random 10 %)
    rng = np.random.default_rng(seed)
    hold = np.zeros(len(y), bool)
    if e5.get("signers"):
        hold = g == e5["signers"][0]
    else:
        hold[rng.choice(len(y), max(1, len(y) // 10), replace=False)] = True
    T = 1.0
    if hold.any() and (~hold).sum() > 10:
        m_cal = _fit_mlp(X[~hold], y[~hold], epochs, seed, max_per_class)
        T = fit_temperature(_logits(m_cal, X[hold]), y[hold])
    model = _fit_mlp(X, y, epochs, seed, max_per_class)
    signers = max(1, len({w["group"] for w in ls.words}))  # known signer ids only (other sources: unknown)
    meta = {"kind": "isl-letters", "labels": LETTERS, "in_dim": LETTER_DIM, "hidden": [128, 64], "temperature": T,
            "trained_letters": sorted({LETTERS[i] for i in y if i != NOISE}), "n_signers": signers,
            "source": ",".join(sorted(set(ls.source))), "profile": profile, "data": ls.summary(), "e5": e5,
            "smoke": smoke, "n_examples": int(len(y))}
    _save(model, out, meta)
    return {"checkpoint": str(out), "temperature": T, "letters": meta["trained_letters"], "n_signers": signers,
            "data": ls.summary(), "e5": {k: v for k, v in e5.items() if k != "per_letter"},
            "seconds": round(time.time() - t0, 1)}


def letter_data(paths, profile: str, vocab=None, report: dict | None = None) -> LetterSet:
    """Read every letter source a profile allows (missing sources are skipped and reported)."""
    from app.isl.training.multisource import LETTER_SOURCES, open_adapter

    fs, clips, images = [], [], []
    for name in LETTER_SOURCES[profile]:
        ad = open_adapter(name, paths, vocab, profile)
        if report is not None:
            report[name] = {"available": ad is not None}
        if ad is None:
            continue
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
