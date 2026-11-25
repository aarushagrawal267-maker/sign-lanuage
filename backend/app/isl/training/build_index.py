"""Nearest-neighbour index over the dictionary (docs/ISL_DESIGN.md 6.3 b).

Rows: every indexable ISLRTC clip as its first-sign trim (``first_sign``) plus, when the trim found a later
boundary, the slice up to the first alternative end (``alt_end``, slightly lower weight) so a wrong first cut is
less harmful; and one centroid per concept of the INCLUDE clips allowed for the index (``centroid``).
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.features.build import featurize
from app.isl.data.sources import Clip
from app.isl.models.encoder import EncoderPredictor
from app.isl.models.index import EmbeddingIndex
from app.isl.training.common import ClipDataset, Item, items_for, slice_bf


def _embed_feats(enc: EncoderPredictor, feats: list[np.ndarray], batch: int = 128) -> np.ndarray:
    out = []
    for a in range(0, len(feats), batch):
        out.append(enc.embed(np.stack(feats[a:a + batch]))[0])
    return np.concatenate(out) if out else np.zeros((0, enc.dim), np.float32)


def build_index(enc: EncoderPredictor, islrtc: list[Clip], include: list[Clip], out_dir: Path, profile: str,
                alt_weight: float = 0.95, centroid_weight: float = 1.0) -> dict:
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
            by[(it.clip.concept_id, it.clip.variant_id)].append(v)
    cents, crow = [], []
    for (cid, vid), vs in sorted(by.items()):
        m = np.mean(vs, axis=0)
        cents.append(m / max(1e-8, np.linalg.norm(m)))
        crow.append({"variant_id": vid, "concept_id": cid, "source": "include", "kind": "centroid",
                     "weight": centroid_weight, "n_clips": len(vs)})
    if cents:
        emb = np.concatenate([emb, np.stack(cents)]) if len(emb) else np.stack(cents)
        rows += crow
    if not rows:
        raise RuntimeError("nothing to index")
    idx = EmbeddingIndex.build(emb, rows, enc.version, fcfg.to_dict(), profile)
    idx.save(out_dir)
    return {"rows": len(rows), "concepts": len(idx.concepts), "first_sign": sum(r["kind"] == "first_sign" for r in rows),
            "alt_end": sum(r["kind"] == "alt_end" for r in rows), "centroids": len(crow),
            "seconds": round(time.time() - t0, 1), "dir": str(out_dir)}
