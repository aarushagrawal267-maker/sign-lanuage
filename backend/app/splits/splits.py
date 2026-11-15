"""Dataset partitions and leakage checks.

* ``official``: the WLASL split as published. It is NOT signer-independent
  (most val/test signers also appear in train); it is kept for comparability
  with the literature and always labelled as such.
* ``signer-holdout``: every signer is assigned to exactly one partition.
* ``signer-cv5-fold{k}``: 5-fold grouped cross-validation over signers
  (test = fold k, val = fold k+1, train = the rest).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import Split, SplitAssignment, Video

PARTITIONS = ("train", "val", "test")


@dataclass
class VideoMeta:
    video_id: str
    class_index: int
    signer_id: int
    official: str
    sha256: str | None = None
    source_key: str | None = None  # raw source file (YouTube id or video id)


def load_meta(db: Session) -> list[VideoMeta]:
    from app.dataset.acquisition import classify_url, youtube_id

    out = []
    for v in db.scalars(select(Video).order_by(Video.video_id)):
        src = youtube_id(v.url) if classify_url(v.url) == "youtube" else v.video_id
        out.append(VideoMeta(v.video_id, v.class_index, v.signer_id, v.split, v.sha256, src or v.video_id))
    return out


def leakage_report(assign: dict[str, str], meta: list[VideoMeta], n_classes: int = 100) -> dict:
    by_id = {m.video_id: m for m in meta}
    signers: dict[str, set[int]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    classes: dict[str, Counter] = defaultdict(Counter)
    for vid, part in assign.items():
        m = by_id[vid]
        signers[part].add(m.signer_id)
        if m.sha256:
            hashes[part].add(m.sha256)
        if m.source_key:
            sources[part].add(m.source_key)
        classes[part][m.class_index] += 1

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    signer_overlap = {f"{a}-{b}": sorted(signers[a] & signers[b]) for a, b in pairs}
    hash_overlap = {f"{a}-{b}": len(hashes[a] & hashes[b]) for a, b in pairs}
    source_overlap = {f"{a}-{b}": len(sources[a] & sources[b]) for a, b in pairs}
    eval_overlap = {}
    for part in ("val", "test"):
        n = sum(1 for vid, p in assign.items() if p == part)
        seen = sum(1 for vid, p in assign.items() if p == part and by_id[vid].signer_id in signers["train"])
        eval_overlap[part] = {"videos": n, "videos_with_train_signer": seen}
    dup_assign = len(assign) != len(set(assign))
    signer_independent = all(len(v) == 0 for v in signer_overlap.values())
    return {
        "n_videos": {p: sum(1 for x in assign.values() if x == p) for p in PARTITIONS},
        "n_signers": {p: len(signers[p]) for p in PARTITIONS},
        "signer_overlap": {k: {"count": len(v), "signers": v} for k, v in signer_overlap.items()},
        "content_hash_overlap": hash_overlap,
        "source_file_overlap": source_overlap,
        "eval_videos_with_train_signers": eval_overlap,
        "classes_missing": {p: sorted(set(range(n_classes)) - set(classes[p])) for p in PARTITIONS},
        "duplicate_assignments": dup_assign,
        "signer_independent": signer_independent,
        "passed": signer_independent and not dup_assign and all(v == 0 for v in hash_overlap.values())
        and all(v == 0 for v in source_overlap.values()),
    }


def official_assignment(meta: list[VideoMeta]) -> dict[str, str]:
    return {m.video_id: m.official for m in meta}


def signer_groups(meta: list[VideoMeta]) -> dict[int, int]:
    """Union signers that share a raw source file (some YouTube compilations carry
    several signer ids), so the same footage can never land in two partitions."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_signer: dict[str, int] = {}
    for m in meta:
        find(m.signer_id)
        if m.source_key:
            if m.source_key in first_signer:
                a, b = find(first_signer[m.source_key]), find(m.signer_id)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first_signer[m.source_key] = m.signer_id
    return {s: find(s) for s in parent}


def _score(assign_s: dict[int, str], vids_by_signer: dict[int, list[VideoMeta]], target: dict[str, float],
           n_classes: int) -> float:
    counts = Counter()
    cls = defaultdict(set)
    for s, p in assign_s.items():
        counts[p] += len(vids_by_signer[s])
        for m in vids_by_signer[s]:
            cls[p].add(m.class_index)
    total = sum(counts.values())
    frac_err = sum(abs(counts[p] / total - target[p]) for p in PARTITIONS)
    coverage = sum(len(cls[p]) for p in ("val", "test")) / (2 * n_classes)
    return coverage - 2.0 * frac_err


def signer_holdout(meta: list[VideoMeta], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0,
                   trials: int = 3000, n_classes: int = 100) -> dict[str, str]:
    """Assign whole signers to partitions, targeting video fractions and class coverage.

    Signer sizes are very uneven in WLASL, so we randomly search signer orders
    and keep the assignment that best balances fractions and val/test class coverage.
    """
    rng = np.random.default_rng(seed)
    group = signer_groups(meta)
    vids_by_signer: dict[int, list[VideoMeta]] = defaultdict(list)
    for m in meta:
        vids_by_signer[group[m.signer_id]].append(m)
    signer_ids = sorted(vids_by_signer)
    total = len(meta)
    target = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    best, best_score = None, -np.inf
    for _ in range(trials):
        order = rng.permutation(signer_ids)
        a: dict[int, str] = {}
        n_test = n_val = 0
        for s in order:
            k = len(vids_by_signer[s])
            if n_test + k <= test_frac * total * 1.15 and n_test < test_frac * total:
                a[s] = "test"
                n_test += k
            elif n_val + k <= val_frac * total * 1.15 and n_val < val_frac * total:
                a[s] = "val"
                n_val += k
            else:
                a[s] = "train"
        sc = _score(a, vids_by_signer, target, n_classes)
        if sc > best_score:
            best, best_score = a, sc
    assert best is not None
    return {m.video_id: best[group[m.signer_id]] for m in meta}


def signer_kfold(meta: list[VideoMeta], k: int = 5, seed: int = 0) -> list[dict[str, str]]:
    """Grouped k-fold over signers, balancing fold sizes (largest-first greedy)."""
    rng = np.random.default_rng(seed)
    group = signer_groups(meta)
    sizes = Counter(group[m.signer_id] for m in meta)
    signers = sorted(sizes, key=lambda s: (-sizes[s], rng.random()))
    fold_of: dict[int, int] = {}
    load = [0] * k
    for s in signers:
        f = int(np.argmin(load))
        fold_of[s] = f
        load[f] += sizes[s]
    folds = []
    for i in range(k):
        val_fold = (i + 1) % k
        folds.append({
            m.video_id: ("test" if fold_of[group[m.signer_id]] == i
                         else "val" if fold_of[group[m.signer_id]] == val_fold else "train")
            for m in meta
        })
    return folds


def save_split(db: Session, name: str, kind: str, assign: dict[str, str], meta: list[VideoMeta],
               seed: int | None = None, config: dict | None = None) -> Split:
    report = leakage_report(assign, meta)
    old = db.scalar(select(Split).where(Split.name == name))
    if old is not None:
        db.execute(delete(SplitAssignment).where(SplitAssignment.split_id == old.id))
        sp = old
        sp.kind, sp.seed, sp.config = kind, seed, config or {}
    else:
        sp = Split(name=name, kind=kind, seed=seed, config=config or {})
        db.add(sp)
        db.flush()
    sp.signer_independent = bool(report["signer_independent"])
    sp.leakage_report = report
    db.add_all(SplitAssignment(split_id=sp.id, video_id=v, partition=p) for v, p in assign.items())
    db.flush()
    return sp


def ensure_default_splits(db: Session, seed: int = 0) -> list[Split]:
    meta = load_meta(db)
    if not meta:
        return []
    out = [save_split(db, "official", "official", official_assignment(meta), meta)]
    out.append(save_split(db, "signer-holdout", "signer_grouped", signer_holdout(meta, seed=seed), meta, seed,
                          {"val_frac": 0.15, "test_frac": 0.15}))
    for i, a in enumerate(signer_kfold(meta, 5, seed)):
        out.append(save_split(db, f"signer-cv5-fold{i}", "signer_grouped_cv", a, meta, seed, {"k": 5, "fold": i}))
    return out


def get_assignment(db: Session, split_id: int) -> dict[str, str]:
    rows = db.execute(select(SplitAssignment.video_id, SplitAssignment.partition).where(SplitAssignment.split_id == split_id))
    return {v: p for v, p in rows}


def refresh_leakage(db: Session, split: Split) -> dict:
    """Recompute the report (e.g. after downloads add content hashes)."""
    report = leakage_report(get_assignment(db, split.id), load_meta(db))
    split.leakage_report = report
    split.signer_independent = bool(report["signer_independent"])
    return report


def build_available_holdout(db: Session, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15) -> Split:
    """Signer-grouped holdout over the videos that were actually acquired (status=valid).

    The default signer-holdout is computed over all annotated instances; when many
    URLs are dead its partitions become unbalanced. This split re-balances over
    what is available, still keeping every signer group in one partition.
    """
    valid = {v for (v,) in db.execute(select(Video.video_id).where(Video.status == "valid"))}
    meta = [m for m in load_meta(db) if m.video_id in valid]
    if not meta:
        raise ValueError("no valid videos yet")
    assign = signer_holdout(meta, val_frac=val_frac, test_frac=test_frac, seed=seed)
    return save_split(db, "signer-holdout-available", "signer_grouped", assign, meta, seed,
                      {"val_frac": val_frac, "test_frac": test_frac, "basis": f"{len(meta)} valid videos"})
