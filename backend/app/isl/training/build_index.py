"""Nearest-neighbour index over the dictionary (docs/ISL_DESIGN.md 6.3 b).

Rows: every indexable ISLRTC clip as its first-sign trim (``first_sign``) plus, when the trim found a later
boundary, the slice up to the first alternative end (``alt_end``, slightly lower weight) so a wrong first cut is
less harmful; and one centroid per concept of the INCLUDE clips allowed for the index (``centroid``).

Phase 2 (``extra``): every other labelled reference of the profile (ISL500, WSLP, CISLR, Mendeley, ...) with a
per-source row weight (``multisource.INDEX_WEIGHTS``): sources with several clips of a concept give one centroid
per (concept, source) plus up to ``max_group_centroids`` per-signer centroids (different signers' forms stay
separate rows); sources with one clip per concept give one row per clip (``kind = "clip"``).

Two indexes per work folder (docs/ISL_DESIGN.md 15.21): ``work/index`` from the training parts only
(``contains_eval = False``), which fit-fusion and evaluation use, and ``work/index_final`` from every reference
(``--all-include``, ``contains_eval = True``), which only the shipped pack uses. Both record the SHA-256 of the
encoder file they were embedded with, the split whose assignment chose the clips and a digest of those clip ids;
every centroid row lists the clips it was built from (``clips``), so the leak guard sees every row (15.22).

``twin_sources`` (docs 15.32): rows of these sources (research: WSLP, a second view of the ISLRTC videos) are left out
for every concept that has an ISLRTC row, since they repeat that row. The count is recorded in ``row_filters``.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.features.build import featurize
from app.isl.data.sources import Clip
from app.isl.models.encoder import EncoderPredictor
from app.isl.models.index import EmbeddingIndex, clip_ids_sha256, file_sha256
from app.isl.training.common import ClipDataset, Item, items_for, slice_bf


def _embed_feats(enc: EncoderPredictor, feats: list[np.ndarray], batch: int = 128) -> np.ndarray:
    out = []
    for a in range(0, len(feats), batch):
        out.append(enc.embed(np.stack(feats[a:a + batch]))[0])
    return np.concatenate(out) if out else np.zeros((0, enc.dim), np.float32)


def index_clip_ids(include: list[Clip], extra: list[Clip] | None) -> list[str]:
    """The INCLUDE and other-source clip ids an index build is given (labelled ones only: the rest is skipped)."""
    return [c.clip_id for c in include if c.concept_id] + [c.clip_id for c in (extra or []) if c.concept_id]


def index_training_clip_ids(data) -> list[str]:
    """The clip ids the ``index`` step (training parts only) would use with this data and split assignment; the
    leak guard compares their digest with ``train_clip_ids_sha256`` of the index (docs/ISL_DESIGN.md 15.22)."""
    return index_clip_ids(data.include_part("train"), data.extra_part("train"))


WSLP_TWIN_MODES = ("keep", "drop", "auto")


def twin_sources_for(mode: str, encoder_meta: dict | None) -> tuple[str, ...]:
    """The sources whose ISLRTC twins ``build-index --wslp-twins MODE`` leaves out (docs 15.32). ``keep`` (default):
    none, as before. ``drop``: WSLP. ``auto``: WSLP when the encoder was trained without WSLP
    (``class_space.excluded_sources``), so an index of an older encoder is built exactly as before."""
    from app.isl.training.common import class_space_of

    if mode not in WSLP_TWIN_MODES:
        raise ValueError(f"wslp twins mode must be one of {WSLP_TWIN_MODES}, not {mode!r}")
    if mode == "drop" or (mode == "auto" and "wslp" in class_space_of(encoder_meta)["excluded_sources"]):
        return ("wslp",)
    return ()


def drop_dictionary_twins(extra: list[Clip], dictionary_concepts: set[str],
                          twin_sources: tuple[str, ...] | list[str]) -> tuple[list[Clip], dict]:
    """``extra`` without the clips of ``twin_sources`` whose concept has a dictionary (ISLRTC) row, and a summary."""
    ts = set(twin_sources or ())
    if not ts:
        return list(extra), {}
    keep, dropped = [], Counter()
    for c in extra:
        if c.source in ts and c.concept_id in dictionary_concepts:
            dropped[c.source] += 1
        else:
            keep.append(c)
    kept = Counter(c.source for c in keep if c.source in ts)
    return keep, {"sources": sorted(ts), "dropped": int(sum(dropped.values())), "dropped_per_source": dict(dropped),
                  "kept_per_source": dict(kept),
                  "rule": "clips of these sources whose concept has an ISLRTC row are not indexed (docs 15.32)"}


def _extra_rows(enc: EncoderPredictor, clips: list[Clip], weights: dict, max_group_centroids: int,
                batch: int = 128) -> tuple[list[np.ndarray], list[dict]]:
    from app.isl.training.common import LazyClipDataset

    fcfg = enc.feature_config
    ds = LazyClipDataset(clips, [0] * len(clips), fcfg, train=False, cache_items=0)
    zs: list[np.ndarray] = []
    kept: list[Clip] = []
    buf, bufc = [], []
    for i, c in enumerate(clips):
        f = ds.features(i)
        if f is None:
            continue
        buf.append(f)
        bufc.append(c)
        if len(buf) >= batch:
            zs.append(_embed_feats(enc, buf))
            kept += bufc
            buf, bufc = [], []
    if buf:
        zs.append(_embed_feats(enc, buf))
        kept += bufc
    if not kept:
        return [], []
    Z = np.concatenate(zs)
    by = defaultdict(list)
    for c, z in zip(kept, Z):
        by[(c.concept_id, c.source)].append((c, z))
    vecs, rows = [], []

    def unit(v):
        return v / max(1e-8, float(np.linalg.norm(v)))

    for (cid, src), lst in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        w = float(weights.get(src, 0.95))
        if len(lst) == 1:
            c, z = lst[0]
            vecs.append(unit(z))
            rows.append({"variant_id": c.variant_id, "concept_id": cid, "source": src, "kind": "clip", "weight": w,
                         "clip": c.clip_id})
            continue
        vecs.append(unit(np.mean([z for _, z in lst], axis=0)))
        rows.append({"variant_id": lst[0][0].variant_id, "concept_id": cid, "source": src, "kind": "centroid",
                     "weight": w, "n_clips": len(lst), "clips": sorted(c.clip_id for c, _ in lst)})
        groups = defaultdict(list)
        for c, z in lst:
            if c.group:
                groups[c.group].append((c.clip_id, z))
        if len(groups) > 1:
            for g in sorted(groups, key=lambda g: (-len(groups[g]), g))[:max_group_centroids]:
                vecs.append(unit(np.mean([z for _, z in groups[g]], axis=0)))
                rows.append({"variant_id": lst[0][0].variant_id, "concept_id": cid, "source": src,
                             "kind": "signer_centroid", "weight": w, "group": g, "n_clips": len(groups[g]),
                             "clips": sorted(i for i, _ in groups[g])})
    return vecs, rows


def build_index(enc: EncoderPredictor, islrtc: list[Clip], include: list[Clip], out_dir: Path, profile: str,
                alt_weight: float = 0.95, centroid_weight: float = 1.0, extra: list[Clip] | None = None,
                source_weights: dict | None = None, max_group_centroids: int = 3, contains_eval: bool = False,
                split: str = "", twin_sources: tuple[str, ...] | list[str] = ()) -> dict:
    """``contains_eval``: the caller says whether ``include`` / ``extra`` hold any validation or test clip (True for
    ``--all-include``); it is written to the index meta and evaluation refuses such an index. ``split``: the name of
    the split whose assignment chose ``include`` / ``extra`` (recorded with a digest of their clip ids; an index
    without it is refused for fitting and evaluation, docs 15.22). ``twin_sources``: leave out these sources' clips
    of concepts that have an ISLRTC row (docs 15.32); the clip digest still covers every given clip."""
    t0 = time.time()
    fcfg = enc.feature_config
    concepts = {c.concept_id for c in islrtc} | {c.concept_id for c in include}
    label_of = {c: 0 for c in concepts if c}
    rows, feats = [], []
    for it in items_for(islrtc, label_of, fcfg, with_alternatives=True):
        base = {"variant_id": it.clip.variant_id, "concept_id": it.clip.concept_id, "source": "islrtc",
                "clip": it.clip.clip_id}
        feats.append(featurize(slice_bf(it.bf, 0, it.default_end), fcfg))
        rows.append({**base, "kind": "first_sign", "weight": 1.0})
        later = [e for e in it.ends if e > it.default_end]
        if later:
            feats.append(featurize(slice_bf(it.bf, 0, later[0]), fcfg))
            rows.append({**base, "kind": "alt_end", "weight": alt_weight})
    emb = _embed_feats(enc, feats)
    # INCLUDE centroids
    inc_items: list[Item] = items_for(include, label_of, fcfg, with_alternatives=False)
    by = defaultdict(list)
    if inc_items:
        z = _embed_feats(enc, ClipDataset(inc_items, fcfg, train=False)._cache)
        for it, v in zip(inc_items, z):
            by[(it.clip.concept_id, it.clip.variant_id)].append((it.clip.clip_id, v))
    cents, crow = [], []
    for (cid, vid), lst in sorted(by.items(), key=lambda kv: kv[0]):
        m = np.mean([v for _, v in lst], axis=0)
        cents.append(m / max(1e-8, np.linalg.norm(m)))
        crow.append({"variant_id": vid, "concept_id": cid, "source": "include", "kind": "centroid",
                     "weight": centroid_weight, "n_clips": len(lst), "clips": sorted(i for i, _ in lst)})
    if cents:
        emb = np.concatenate([emb, np.stack(cents)]) if len(emb) else np.stack(cents)
        rows += crow
    n_extra = 0
    row_filters: dict = {}
    if extra:
        from app.isl.training.multisource import INDEX_WEIGHTS

        given = [c for c in extra if c.concept_id]
        if twin_sources:
            dict_concepts = {r["concept_id"] for r in rows if r["source"] == "islrtc" and r["concept_id"]}
            given, twins = drop_dictionary_twins(given, dict_concepts, twin_sources)
            row_filters["wslp_twins" if set(twin_sources) == {"wslp"} else "dictionary_twins"] = twins
        vecs, erows = _extra_rows(enc, given, {**INDEX_WEIGHTS, **(source_weights or {})}, max_group_centroids)
        if vecs:
            emb = np.concatenate([emb, np.stack(vecs)]) if len(emb) else np.stack(vecs)
            rows += erows
            n_extra = len(erows)
    if not rows:
        raise RuntimeError("nothing to index")
    enc_path = getattr(enc, "path", None)
    idx = EmbeddingIndex.build(emb, rows, enc.version, fcfg.to_dict(), profile, contains_eval=bool(contains_eval),
                               encoder_sha256=file_sha256(enc_path) if enc_path is not None else "", split=split or "",
                               train_clip_ids_sha256=clip_ids_sha256(index_clip_ids(include, extra)),
                               row_filters=row_filters)
    idx.save(out_dir)
    return {"rows": len(rows), "concepts": len(idx.concepts), "first_sign": sum(r["kind"] == "first_sign" for r in rows),
            "alt_end": sum(r["kind"] == "alt_end" for r in rows), "centroids": len(crow), "extra_rows": n_extra,
            "rows_per_source": dict(Counter(r["source"] for r in rows)),
            "contains_eval": idx.meta.contains_eval, "encoder_sha256": idx.meta.encoder_sha256,
            "split": idx.meta.split, "train_clip_ids_sha256": idx.meta.train_clip_ids_sha256,
            "row_filters": idx.meta.row_filters,
            "seconds": round(time.time() - t0, 1), "dir": str(out_dir)}
