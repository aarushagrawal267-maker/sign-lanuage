"""Brute-force cosine nearest-neighbour index (docs/ISL_DESIGN.md section 6.3 b).

Files in the index folder::

    embeddings.f16.npy   (N, D) float16, L2-normalised
    rows.jsonl           {"row", "variant_id", "concept_id", "source", "kind", "weight"}
    meta.json            {"encoder_version", "dim", "n", "created", "feature_config", "profile", "sha256",
                          "contains_eval", "encoder_sha256", "split", "train_clip_ids_sha256", "row_filters"}

``contains_eval`` says whether the index holds references of validation / test clips (``build-index --all-include``,
the shipped ``index_final``); ``encoder_sha256`` is the SHA-256 of the encoder file the rows were embedded with
(``ENCODER_VERSION`` is a constant, so it cannot tell two trainings apart). Index files written before these fields
existed load with ``contains_eval = None`` (unknown) and ``encoder_sha256 = ""`` (docs/ISL_DESIGN.md 15.21).
``split`` and ``train_clip_ids_sha256`` say which split's training parts were indexed and which clips exactly; rows
built from several clips (centroids) list them in ``clips`` (docs/ISL_DESIGN.md 15.22). ``row_filters`` records rows
left out on purpose, e.g. WSLP twins of ISLRTC rows (``{"wslp_twins": {"dropped": n, ...}}``, 15.32; ``{}`` = none).

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


class IndexEncoderMismatch(IndexVersionMismatch):
    """The index was embedded with another encoder FILE (SHA-256) than the one loaded: same version string, other
    weights (e.g. a retrained encoder.pt with a stale index)."""


class LeakageError(RuntimeError):
    """An evaluation / fitting input may contain the clips it is measured on (docs/ISL_DESIGN.md 15.21)."""


class IndexMissingError(LeakageError):
    """A measuring caller (fit-fusion, evaluation) found no training-only ``work/index`` while another index is there
    (``index_final``, or ``index.old`` / ``index.tmp`` of an interrupted save): it would otherwise run without
    retrieval and say nothing (docs/ISL_DESIGN.md 15.22)."""


_SHA_CACHE: dict[tuple, str] = {}


def clip_ids_sha256(ids) -> str:
    """SHA-256 of a set of clip ids (sorted, one per line): which clips an index was built from."""
    return hashlib.sha256("\n".join(sorted({str(i) for i in ids})).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, cached per (path, size, mtime) so repeated loads of one encoder hash it once."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


@dataclass
class IndexMeta:
    encoder_version: str
    dim: int
    n: int
    created: str = ""
    feature_config: dict = field(default_factory=dict)
    profile: str = ""
    sha256: str = ""
    # True: built with validation / test references (--all-include, index_final); False: training parts only;
    # None: unknown (an index file written before this field existed)
    contains_eval: bool | None = None
    encoder_sha256: str = ""  # SHA-256 of the encoder file used ("" = unknown: older index or no file)
    # the split whose assignment chose the indexed clips, and clip_ids_sha256 of the INCLUDE + other-source clips
    # given to the build ("" = unknown: an index written before docs 15.22; refused for fitting / evaluation)
    split: str = ""
    train_clip_ids_sha256: str = ""
    # rows of the given clips left out on purpose (docs 15.32), e.g. {"wslp_twins": {"dropped": 4274, ...}}; the
    # clip digest above still covers every given clip, so the leak guard compares like with like
    row_filters: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, d: dict) -> IndexMeta:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


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
              profile: str = "", contains_eval: bool | None = None, encoder_sha256: str = "", split: str = "",
              train_clip_ids_sha256: str = "", row_filters: dict | None = None) -> EmbeddingIndex:
        rows = [{**r, "row": i} for i, r in enumerate(rows)]
        emb = np.asarray(emb, np.float32)
        meta = IndexMeta(encoder_version, int(emb.shape[1]) if emb.ndim == 2 and emb.size else 0, len(rows),
                         datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), feature_config or {}, profile,
                         contains_eval=contains_eval, encoder_sha256=encoder_sha256, split=split or "",
                         train_clip_ids_sha256=train_clip_ids_sha256, row_filters=dict(row_filters or {}))
        if meta.dim == 0:
            raise ValueError("cannot build an empty index")
        return cls(emb, rows, meta)

    def save(self, folder: Path) -> Path:
        """Write the three files into a sibling temporary folder, then swap it in: a failed or killed build never
        leaves new embeddings next to an old meta.json (whose ``contains_eval`` would then describe other rows)."""
        import shutil

        folder = Path(folder)
        folder.parent.mkdir(parents=True, exist_ok=True)
        tmp = folder.with_name(folder.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        e16 = self.emb.astype(np.float16)
        np.save(tmp / "embeddings.f16.npy", e16)
        with (tmp / "rows.jsonl").open("w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.meta.n = len(self.rows)
        self.meta.sha256 = hashlib.sha256(e16.tobytes()).hexdigest()
        (tmp / "meta.json").write_text(json.dumps(self.meta.to_json(), indent=1))
        if folder.exists():
            old = folder.with_name(folder.name + ".old")
            shutil.rmtree(old, ignore_errors=True)
            folder.replace(old)
            tmp.replace(folder)
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.replace(folder)
        return folder

    @classmethod
    def load(cls, folder: Path, expect_encoder_version: str | None = None,
             encoder_path: Path | None = None) -> EmbeddingIndex:
        """``encoder_path``: the encoder file in use. When the index records the SHA-256 of its encoder, a different
        file raises ``IndexEncoderMismatch`` (an index of an older index format records none: not checked)."""
        meta = IndexMeta.from_json(json.loads((folder / "meta.json").read_text()))
        if expect_encoder_version is not None and meta.encoder_version != expect_encoder_version:
            raise IndexVersionMismatch(f"index {folder} was built with encoder {meta.encoder_version!r}, "
                                       f"but the loaded encoder is {expect_encoder_version!r}; rebuild the index")
        if encoder_path is not None and meta.encoder_sha256:
            have = file_sha256(encoder_path)
            if have != meta.encoder_sha256:
                raise IndexEncoderMismatch(f"index {folder} was embedded with encoder file sha256 "
                                           f"{meta.encoder_sha256[:12]}..., but {encoder_path} has {have[:12]}... "
                                           "(same version string, other weights); rebuild the index")
        emb = np.load(folder / "embeddings.f16.npy", allow_pickle=False).astype(np.float32)
        rows = [json.loads(ln) for ln in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        return cls(emb, rows, meta)

    # ------------------------------------------------------------ leak guard
    def row_clip_ids(self) -> set[str]:
        """Every clip id an index row was built from: ``clip`` (one-clip rows) and ``clips`` (centroids)."""
        out: set[str] = set()
        for r in self.rows:
            if r.get("clip"):
                out.add(r["clip"])
            out.update(r.get("clips") or ())
        return out

    def check_for_evaluation(self, encoder_path: Path | None = None, eval_clip_ids=None, where: str = "",
                             split: str | None = None, train_clip_ids=None) -> dict:
        """Refuse (``LeakageError``) an index that cannot be shown to be clean for fitting or evaluation:

        * ``contains_eval`` is True (``--all-include`` / ``index_final``: it holds validation and test references);
        * ``contains_eval`` is unknown (an index file written before the field existed). Until this check, the
          driver's ``index_final`` step overwrote ``work/index``, so an old ``work/index`` may hold evaluation
          clips: rebuild it with the ``index`` step (no ``--all-include``);
        * ``encoder_path`` is given and the index records another (or no) encoder SHA-256;
        * the index does not record its ``split`` and ``train_clip_ids_sha256`` (written before docs 15.22: its
          centroid rows do not list their clips either);
        * ``split`` is given and differs from the recorded one (e.g. ``fit-fusion --split <other>``);
        * ``train_clip_ids`` is given (the clips the ``index`` step would use now) and their digest differs from the
          recorded one: the split assignment or the data changed since the build, so old centroids may hold clips
          that are validation / test clips now;
        * a row was built from one of ``eval_clip_ids`` (``clip`` of one-clip rows, ``clips`` of centroids).

        Returns a small summary for reports."""
        m = self.meta
        name = where or "the index"
        fix = ("rebuild it from the training parts only: scripts/phase2_train.sh --profile <p> --force "
               "index,fusion,trim_eval,evaluate,index_final,pack (or `python -m app.isl.cli build-index` without "
               "--all-include)")
        if m.contains_eval is True:
            raise LeakageError(f"{name} contains validation / test references (built with --all-include: the "
                               f"shipped index_final); it must not be used to fit or evaluate; {fix}")
        if m.contains_eval is not True and m.contains_eval is not False:
            raise LeakageError(f"{name} does not record whether it contains validation / test references (written "
                               f"before contains_eval existed, possibly by the old index_final step, which "
                               f"overwrote work/index); {fix}")
        if encoder_path is not None:
            if not m.encoder_sha256:
                raise LeakageError(f"{name} does not record the SHA-256 of its encoder file, so it cannot be shown "
                                   f"to match {encoder_path}; {fix}")
            have = file_sha256(encoder_path)
            if have != m.encoder_sha256:
                raise LeakageError(f"{name} was embedded with another encoder file (sha256 {m.encoder_sha256[:12]}... "
                                   f"vs {have[:12]}... of {encoder_path}): stale index; {fix}")
        if not m.split or not m.train_clip_ids_sha256:
            raise LeakageError(f"{name} does not record its split and the clips it was built from (written before "
                               f"docs 15.22; its centroids do not list their clips), so it cannot be shown to hold "
                               f"no validation / test clip; {fix}")
        if split is not None and m.split != split:
            raise LeakageError(f"{name} was built from the training parts of split {m.split!r}, but this run uses "
                               f"split {split!r}: its rows may hold this split's validation / test clips; {fix}")
        if train_clip_ids is not None and clip_ids_sha256(train_clip_ids) != m.train_clip_ids_sha256:
            raise LeakageError(f"{name} was built from other training clips than the current data has (the split "
                               f"assignment or the data changed since the build): old centroids may hold clips that "
                               f"are validation / test clips now; {fix}")
        n_checked = 0
        if eval_clip_ids is not None:
            ids = set(eval_clip_ids)
            hit = sorted(self.row_clip_ids() & ids)
            n_checked = len(ids)
            if hit:
                raise LeakageError(f"{name} has rows built from {len(hit)} evaluation clips (e.g. {hit[:3]}); {fix}")
        return {"contains_eval": m.contains_eval, "encoder_sha256": m.encoder_sha256, "split": m.split,
                "encoder_sha256_checked": encoder_path is not None, "split_checked": split is not None,
                "train_clip_ids_checked": train_clip_ids is not None, "eval_clip_ids_checked": n_checked}

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
