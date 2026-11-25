"""Brute-force cosine nearest-neighbour index (docs/ISL_DESIGN.md section 6.3 b).

Files in the index folder::

    embeddings.f16.npy   (N, D) float16, L2-normalised
    rows.jsonl           {"row", "variant_id", "concept_id", "source", "kind", "weight"}
    meta.json            {"encoder_version", "dim", "n", "created", "feature_config", "profile", "sha256"}

~25k x 256 cosine scores are one small matrix product (a few ms on CPU), so no ANN library is needed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


class IndexVersionMismatch(ValueError):
    """The index was built with another encoder version than the one loaded."""


@dataclass
class IndexMeta:
    encoder_version: str
    dim: int
    n: int
    created: str = ""
    feature_config: dict = field(default_factory=dict)
    profile: str = ""
    sha256: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


class EmbeddingIndex:
    def __init__(self, emb: np.ndarray, rows: list[dict], meta: IndexMeta):
        emb = np.asarray(emb, np.float32).reshape(-1, meta.dim) if len(rows) else np.zeros((0, meta.dim), np.float32)
        if emb.shape[0] != len(rows):
            raise ValueError(f"index has {emb.shape[0]} embeddings but {len(rows)} rows")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.emb = emb / np.maximum(norms, 1e-8)
        self.rows = rows
        self.meta = meta
        self.weights = np.asarray([float(r.get("weight", 1.0)) for r in rows], np.float32)
        self.concepts: list[str] = sorted({r["concept_id"] for r in rows})
        self.concept_pos = {c: i for i, c in enumerate(self.concepts)}
        self.row_concept = np.asarray([self.concept_pos[r["concept_id"]] for r in rows], np.int64)
        # rows grouped by concept for a vectorised per-concept max (np.maximum.reduceat)
        self._order = np.argsort(self.row_concept, kind="stable")
        rc = self.row_concept[self._order]
        self._starts = np.flatnonzero(np.r_[True, rc[1:] != rc[:-1]]) if len(rc) else np.zeros(0, np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    # ------------------------------------------------------------------ io
    @classmethod
    def build(cls, emb: np.ndarray, rows: list[dict], encoder_version: str, feature_config: dict | None = None,
              profile: str = "") -> EmbeddingIndex:
        rows = [{**r, "row": i} for i, r in enumerate(rows)]
        emb = np.asarray(emb, np.float32)
        meta = IndexMeta(encoder_version, int(emb.shape[1]) if emb.ndim == 2 and emb.size else 0, len(rows),
                         datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), feature_config or {}, profile)
        if meta.dim == 0:
            raise ValueError("cannot build an empty index")
        return cls(emb, rows, meta)

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        e16 = self.emb.astype(np.float16)
        np.save(folder / "embeddings.f16.npy", e16)
        with (folder / "rows.jsonl").open("w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.meta.n = len(self.rows)
        self.meta.sha256 = hashlib.sha256(e16.tobytes()).hexdigest()
        (folder / "meta.json").write_text(json.dumps(self.meta.to_json(), indent=1))
        return folder

    @classmethod
    def load(cls, folder: Path, expect_encoder_version: str | None = None) -> EmbeddingIndex:
        meta = IndexMeta(**json.loads((folder / "meta.json").read_text()))
        if expect_encoder_version is not None and meta.encoder_version != expect_encoder_version:
            raise IndexVersionMismatch(f"index {folder} was built with encoder {meta.encoder_version!r}, "
                                       f"but the loaded encoder is {expect_encoder_version!r}; rebuild the index")
        emb = np.load(folder / "embeddings.f16.npy", allow_pickle=False).astype(np.float32)
        rows = [json.loads(ln) for ln in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        return cls(emb, rows, meta)

    # --------------------------------------------------------------- search
    def scores(self, q: np.ndarray) -> np.ndarray:
        """(B, D) queries -> (B, N) cosine similarities (weights applied as a multiplicative factor)."""
        q = np.atleast_2d(np.asarray(q, np.float32))
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        return (q @ self.emb.T) * self.weights[None, :]

    def search(self, q: np.ndarray, k: int = 50) -> list[list[tuple[int, float]]]:
        s = self.scores(q)
        out = []
        for row in s:
            kk = min(k, len(row))
            idx = np.argpartition(-row, kk - 1)[:kk] if kk else np.zeros(0, int)
            idx = idx[np.argsort(-row[idx])]
            out.append([(int(i), float(row[i])) for i in idx])
        return out

    def concept_matrix(self, q: np.ndarray) -> np.ndarray:
        """(B, n_concepts) max cosine over each concept's rows (concept order = ``self.concepts``)."""
        s = self.scores(q)
        if not len(self.rows):
            return np.zeros((s.shape[0], 0), np.float32)
        return np.maximum.reduceat(s[:, self._order], self._starts, axis=1).astype(np.float32)

    def concept_scores(self, q: np.ndarray, k: int = 50) -> list[dict[str, float]]:
        m = self.concept_matrix(q)
        res = []
        for row in m:
            kk = min(k, len(row))
            idx = np.argsort(-row)[:kk]
            res.append({self.concepts[i]: float(row[i]) for i in idx})
        return res

    def best_row(self, q: np.ndarray | None, concept_id: str, row_scores: np.ndarray | None = None) -> dict | None:
        """The best-matching row of one concept (to report which variant matched). ``row_scores`` (N,) can be
        passed to avoid recomputing the scores for every candidate."""
        if concept_id not in self.concept_pos:
            return None
        s = row_scores if row_scores is not None else self.scores(q).mean(0)
        idx = np.where(self.row_concept == self.concept_pos[concept_id])[0]
        return self.rows[int(idx[np.argmax(s[idx])])]
