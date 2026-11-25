"""ISL splits (docs/ISL_DESIGN.md sections 5.4 and 5.5).

* ``include-official``: AI4Bharat's ``include_{train,val,test}.txt``. **Not signer-independent**: consecutive
  takes from one recording run are spread across train and test.
* ``include-session-cv5``: sessions = runs of consecutive camera file numbers for one word (gap <= 3);
  whole sessions per fold.
* ``include-pseudo-signer-cv5`` (primary): sessions clustered into pseudo-signers from scale-free body
  proportions; whole pseudo-signers per fold. Estimated, not verified identities.
* ``oneshot-concept-cv5``: concept folds for the cross-dataset one-shot protocol (section 5.5).

Split files are JSON under ``data/isl/splits/`` (local, derived, never committed)::

    {"name", "kind", "signer_independent", "seed", "created", "source_manifest_sha256", "label",
     "assign": {clip_id: "train"|"val"|"test"}, "leakage": {...}, "unmatched": {...}}
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

from app.isl.data.sources import Clip, include_key_from_parts, include_match_key, include_clip_id, read_split_list
from app.isl.paths import IslPaths

PARTS = ("train", "val", "test")
OFFICIAL_BASE = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths"
OFFICIAL_FILES = ("include_train", "include_val", "include_test", "include50_train", "include50_val", "include50_test")
# SHA-256 of the lists as fetched on 25 Nov 2026 (a changed upstream file is reported, not silently used).
OFFICIAL_SHA256 = {
    "include_train": "1d3f0f21c10f8a6b4bf984aea5354f6841b53cc2b9a11a63fd459c1f527d8ea5",
    "include_val": "75556b02dc9833e5c5a12a4e43294778eaa97f23fd72ba2114d26596ad694ea1",
    "include_test": "48a6219b6b78c68e4831f03171ac377fcf980430bd7ea557106e111fe59d0e65",
    "include50_train": "ba36abcf06c5cfd2f7dc66babe8c0d46d407ebba50700ff2041b590dae2609f9",
    "include50_val": "60ed8603c22a5dbc06570cd3d9d466e561d071bf402dde1f5668e061e1ba34d0",
    "include50_test": "e8bd54ec642e2eea52f82730b4418b82268a4a34aa2df17e8bf66d54f2e1882e",
}
LABELS = {
    "official": "official split, not signer-independent (same recording session in train and test)",
    "session": "session-independent",
    "pseudo_signer": "estimated signer-independent (pseudo-signers from body proportions; not verified identities)",
    "oneshot": "cross-dataset one-shot",
}
_RE_NUM = re.compile(r"(\d+)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    if path.exists():
        h.update(path.read_bytes())
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------------------------- official lists

def fetch_official(paths: IslPaths, force: bool = False, timeout: int = 60) -> dict[str, dict]:
    """Download the AI4Bharat split lists into ``splits/official/`` and check them against known hashes."""
    out = {}
    paths.official_splits.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_FILES:
        dest = paths.official_splits / f"{name}.txt"
        if force or not dest.exists():
            with urllib.request.urlopen(f"{OFFICIAL_BASE}/{name}.txt", timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
        digest = sha256_file(dest)
        out[name] = {"path": str(dest), "lines": len(read_split_list(dest)), "sha256": digest,
                     "matches_known": digest == OFFICIAL_SHA256.get(name)}
    return out


def official_paths(paths: IslPaths, prefix: str = "include") -> dict[str, list[str]]:
    return {p: read_split_list(paths.official_splits / f"{prefix}_{p}.txt") for p in PARTS}


def include_official(clips: Iterable[Clip], lists: dict[str, list[str]]) -> tuple[dict[str, str], dict]:
    """Map official split paths onto INCLUDE clips. Returns (assign, report). Every path is either matched or
    listed (``not_ingested`` = in a list but not (yet) in the manifest; ``not_listed`` = clip in no list)."""
    by_key = {include_match_key(c.clip_id): c.clip_id for c in clips}
    assign: dict[str, str] = {}
    not_ingested: dict[str, list[str]] = {p: [] for p in PARTS}
    unparsable: list[str] = []
    seen_twice: list[str] = []
    for part in PARTS:
        for line in lists.get(part, []):
            key = include_key_from_parts(PurePosixPath(line).parts)
            if key is None:
                unparsable.append(line)
                continue
            mk = include_match_key(include_clip_id(*key))
            cid = by_key.get(mk)
            if cid is None:
                not_ingested[part].append(line)
                continue
            if cid in assign and assign[cid] != part:
                seen_twice.append(line)
            assign[cid] = part
    not_listed = sorted(set(by_key.values()) - set(assign))
    n_list = sum(len(v) for v in lists.values())
    report = {
        "list_paths": n_list, "matched": len(assign),
        "not_ingested": {p: len(v) for p, v in not_ingested.items()},
        "not_ingested_examples": {p: v[:5] for p, v in not_ingested.items()},
        "not_listed": len(not_listed), "not_listed_ids": not_listed[:200],
        "unparsable": unparsable, "in_two_partitions": seen_twice,
    }
    return assign, report


# --------------------------------------------------------------------------------------------- sessions

def file_number(clip_id: str) -> int | None:
    m = _RE_NUM.findall(PurePosixPath(clip_id).name)
    return int(m[-1]) if m else None


def include_sessions(clips: Iterable[Clip], gap: int = 3) -> dict[str, str]:
    """clip_id -> session id. A session = run of consecutive camera file numbers (gap <= ``gap``) of one word."""
    by_word: dict[str, list[tuple[int, str]]] = defaultdict(list)
    out: dict[str, str] = {}
    for c in clips:
        word = c.quality.get("word_dir") or str(PurePosixPath(c.clip_id).parent)
        num = file_number(c.clip_id)
        if num is None:
            out[c.clip_id] = f"{word}#{PurePosixPath(c.clip_id).name}"
        else:
            by_word[word].append((num, c.clip_id))
    for word, items in by_word.items():
        items.sort()
        start, prev = None, None
        for num, cid in items:
            if prev is None or num - prev > gap:
                start = num
            out[cid] = f"{word}#{start}"
            prev = num
    return out


# --------------------------------------------------------------------------------------------- pseudo-signers

def body_descriptor(raw) -> np.ndarray | None:
    """Scale-free body proportions (medians / high percentiles over frames) for pseudo-signer clustering:
    upper arm, forearm and hand length, eye distance, nose-chin, mouth width, cheek width, nose height,
    all divided by shoulder width. Returns None when the body is not resolved."""
    from app.features.spec import FACE_IDX

    T = len(raw)
    if T == 0:
        return None
    asp = raw.aspect
    P = raw.pose[..., :2].astype(np.float64).copy()
    P[..., 0] *= asp
    F = raw.face[..., :2].astype(np.float64).copy()
    F[..., 0] *= asp
    H = [h[..., :2].astype(np.float64).copy() for h in (raw.left, raw.right)]
    for h in H:
        h[..., 0] *= asp
    sw = np.linalg.norm(P[:, 11] - P[:, 12], axis=1)
    ok = np.isfinite(sw) & (sw > 1e-3)
    if ok.sum() < 3:
        return None
    swm = float(np.median(sw[ok]))

    def dist(a, b, pct=90):
        d = np.linalg.norm(a - b, axis=1)
        d = d[np.isfinite(d)]
        return float(np.percentile(d, pct)) / swm if len(d) else np.nan

    fi = {k: i for i, k in enumerate(FACE_IDX)}
    feats = [
        dist(P[:, 11], P[:, 13]), dist(P[:, 12], P[:, 14]),  # upper arms
        dist(P[:, 13], P[:, 15]), dist(P[:, 14], P[:, 16]),  # forearms
        dist(H[0][:, 0], H[0][:, 9]), dist(H[1][:, 0], H[1][:, 9]),  # hand length
        dist(F[:, fi[33]], F[:, fi[263]], 50),  # outer eye corners
        dist(F[:, fi[1]], F[:, fi[152]], 50),  # nose tip - chin
        dist(F[:, fi[61]], F[:, fi[291]], 50),  # mouth width
        dist(F[:, fi[234]], F[:, fi[454]], 50),  # cheek width
        dist(P[:, 0], (P[:, 11] + P[:, 12]) / 2, 50),  # nose height above shoulders
    ]
    v = np.array(feats, dtype=np.float64)
    if np.isnan(v).sum() > 4:
        return None
    return v


def _union_find(n: int):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union


def include_pseudo_signers(clips: list[Clip], sessions: dict[str, str], k_range: tuple[int, int] = (4, 12),
                           must_link_distance: int = 30, seed: int = 0,
                           descriptors: dict[str, np.ndarray] | None = None) -> tuple[dict[str, dict], dict]:
    """Cluster sessions into pseudo-signers. Returns (clip_id -> {session, pseudo_signer, silhouette}, report).

    Must-link prior: sessions are chained into recording runs ("blocks") in camera-number order while the
    gap is <= ``must_link_distance`` and no word repeats; a block is one pass of one signer through the word
    list, and blocks (not sessions) are clustered."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if descriptors is None:
        descriptors = {}
        for c in clips:
            try:
                d = body_descriptor(c.load())
            except Exception:  # noqa: BLE001
                d = None
            if d is not None:
                descriptors[c.clip_id] = d
    sess_clips: dict[str, list[str]] = defaultdict(list)
    for c in clips:
        sess_clips[sessions[c.clip_id]].append(c.clip_id)
    sess_ids = sorted(sess_clips)
    # number range per session
    rng_of = {}
    for s in sess_ids:
        nums = [file_number(c) for c in sess_clips[s] if file_number(c) is not None]
        rng_of[s] = (min(nums), max(nums)) if nums else None
    # Recording runs: sessions in camera-number order are chained while the gap is <= must_link_distance
    # and no word repeats (a signer records each word once per pass through the list; a repeated word
    # means a new pass, possibly by another signer, even when it follows immediately).
    word_of = {s: s.rsplit("#", 1)[0] for s in sess_ids}
    block_of: dict[str, int] = {}
    order = sorted((s for s in sess_ids if rng_of[s]), key=lambda s: rng_of[s][0])
    cur, cur_words, prev_end = -1, set(), None
    for s in order:
        lo, hi = rng_of[s]
        if prev_end is None or lo - prev_end > must_link_distance or word_of[s] in cur_words:
            cur += 1
            cur_words = set()
        block_of[s] = cur
        cur_words.add(word_of[s])
        prev_end = hi
    for s in sess_ids:
        if s not in block_of:
            cur += 1
            block_of[s] = cur
    blocks: dict[int, list[str]] = defaultdict(list)
    for s in sess_ids:
        blocks[block_of[s]].append(s)
    block_ids = sorted(blocks)
    X, keep = [], []
    for b in block_ids:
        ds = [descriptors[c] for s in blocks[b] for c in sess_clips[s] if c in descriptors]
        if ds:
            X.append(np.nanmedian(np.stack(ds), axis=0))
            keep.append(b)
    report: dict = {"sessions": len(sess_ids), "blocks": len(block_ids), "blocks_with_descriptor": len(keep)}
    labels: dict[int, int] = {}
    sil = float("nan")
    if len(keep) >= 3:
        Xa = np.stack(X)
        col_med = np.nanmedian(Xa, axis=0)
        Xa = np.where(np.isnan(Xa), col_med, Xa)
        Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
        best = None
        lo, hi = k_range
        for k in range(max(2, lo), min(hi, len(keep) - 1) + 1):
            cl = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(Xa)
            if len(set(cl.labels_)) < 2:
                continue
            s = float(silhouette_score(Xa, cl.labels_, metric="cosine"))
            report.setdefault("silhouette_by_k", {})[k] = round(s, 4)
            if best is None or s > best[0]:
                best = (s, k, cl.labels_)
        if best is not None:
            sil, k, lab = best
            labels = {b: int(l) for b, l in zip(keep, lab)}
            report["k"] = k
    report["silhouette"] = None if np.isnan(sil) else round(sil, 4)
    report["weak"] = bool(np.isnan(sil) or sil < 0.15)
    out: dict[str, dict] = {}
    next_label = max(labels.values(), default=-1) + 1
    for b in block_ids:
        if b not in labels:  # no descriptor: its own pseudo-signer (never merged with others)
            labels[b] = next_label
            next_label += 1
        for s in blocks[b]:
            for c in sess_clips[s]:
                out[c] = {"session": s, "pseudo_signer": f"ps{labels[b]:02d}", "block": f"blk{b:04d}",
                          "silhouette": report["silhouette"]}
    report["sizes"] = dict(Counter(v["pseudo_signer"] for v in out.values()))
    return out, report


# --------------------------------------------------------------------------------------------- grouped folds

def grouped_kfold(items: dict[str, str], k: int = 5, seed: int = 0,
                  merge: Iterable[Iterable[str]] = ()) -> list[dict[str, str]]:
    """``items``: clip_id -> group. Groups listed together in ``merge`` (e.g. duplicate clusters) are joined.
    Fold i: test = fold i, val = fold i+1, train = the rest (largest-first greedy balancing)."""
    groups = sorted(set(items.values()))
    gi = {g: i for i, g in enumerate(groups)}
    find, union = _union_find(len(groups))
    for cl in merge:
        gs = [gi[items[c]] for c in cl if c in items]
        for a, b in zip(gs, gs[1:]):
            union(a, b)
    root = {c: find(gi[g]) for c, g in items.items()}
    sizes = Counter(root.values())
    rng = np.random.default_rng(seed)
    order = sorted(sizes, key=lambda r: (-sizes[r], rng.random()))
    load = [0] * k
    fold_of = {}
    for r in order:
        f = int(np.argmin(load))
        fold_of[r] = f
        load[f] += sizes[r]
    folds = []
    for i in range(k):
        vf = (i + 1) % k
        folds.append({c: ("test" if fold_of[r] == i else "val" if fold_of[r] == vf else "train")
                      for c, r in root.items()})
    return folds


def leakage_report(assign: dict[str, str], group_of: dict[str, str], class_of: dict[str, str | None],
                   duplicates: Iterable[Iterable[str]] = ()) -> dict:
    groups: dict[str, set] = defaultdict(set)
    classes: dict[str, set] = defaultdict(set)
    for c, p in assign.items():
        groups[p].add(group_of.get(c))
        if class_of.get(c):
            classes[p].add(class_of[c])
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlap = {f"{a}-{b}": len((groups[a] & groups[b]) - {None}) for a, b in pairs}
    dup = 0
    for cl in duplicates:
        parts = {assign[c] for c in cl if c in assign}
        if len(parts) > 1:
            dup += 1
    all_classes = set().union(*classes.values()) if classes else set()
    return {
        "n": {p: sum(1 for v in assign.values() if v == p) for p in PARTS},
        "n_groups": {p: len(groups[p] - {None}) for p in PARTS},
        "group_overlap": overlap,
        "duplicate_overlap": dup,
        "classes_missing": {p: sorted(all_classes - classes[p]) for p in ("val", "test")},
        "passed": all(v == 0 for v in overlap.values()) and dup == 0,
    }


def split_doc(name: str, kind: str, assign: dict[str, str], leakage: dict, seed: int | None,
              manifest: Path, signer_independent: str, extra: dict | None = None) -> dict:
    d = {"name": name, "kind": kind, "signer_independent": signer_independent, "label": LABELS.get(kind, kind),
         "seed": seed, "created": now_iso(), "source_manifest_sha256": sha256_file(manifest),
         "assign": dict(sorted(assign.items())), "leakage": leakage}
    if extra:
        d.update(extra)
    return d


def write_split(paths: IslPaths, doc: dict) -> Path:
    p = paths.splits / f"{doc['name']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    tmp.replace(p)
    return p


# --------------------------------------------------------------------------------------------- one-shot

def oneshot_concepts(entries) -> list[dict]:
    """Concepts usable for the cross-dataset one-shot protocol: an INCLUDE variant linked without review
    (``exact``) AND at least one indexable ISLRTC variant."""
    out = []
    for e in entries:
        inc = [v for v in e.variants if v.source == "include"]
        isl = [v for v in e.variants if v.source == "islrtc" and v.index]
        if not inc or not isl:
            continue
        reviewed_ok = all((v.link == "exact" and not v.needs_review) for v in inc)
        out.append({"concept_id": e.concept_id, "category": e.category,
                    "include_variants": [v.variant_id for v in inc],
                    "islrtc_variants": [v.variant_id for v in isl],
                    "links": sorted({v.link or "" for v in inc}),
                    "usable": reviewed_ok})
    return out


def oneshot_folds(concepts: list[dict], k: int = 5, seed: int = 0) -> dict[str, int]:
    """Usable linked concepts -> fold, stratified by INCLUDE category (round robin within each category
    after a seeded shuffle)."""
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    for c in concepts:
        if c["usable"]:
            by_cat[c.get("category") or "other"].append(c["concept_id"])
    fold: dict[str, int] = {}
    offset = 0
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        for j, cid in enumerate(ids):
            fold[cid] = (offset + j) % k
        offset += len(ids)
    return fold


def oneshot_protocol(fold_of: dict[str, int], clips: list[Clip], k: int) -> dict:
    """Per fold: query clip ids (INCLUDE clips of fold concepts) and the concepts whose INCLUDE/CISLR clips must
    be excluded from encoder training (ISLRTC references stay in the gallery)."""
    folds = []
    for i in range(k):
        held = sorted(c for c, f in fold_of.items() if f == i)
        hs = set(held)
        queries = sorted(c.clip_id for c in clips if c.source == "include" and c.concept_id in hs)
        folds.append({"fold": i, "held_out_concepts": held, "queries": queries,
                      "exclude_from_training": {"include": "all clips of held_out_concepts",
                                                "cislr": "all clips of held_out_concepts",
                                                "islrtc": "no supervised loss on held_out_concepts (kept in gallery)"},
                      "gallery": ["full ISLRTC index (all indexable variants)", "263-concept restricted gallery"]})
    return {"k": k, "folds": folds}
