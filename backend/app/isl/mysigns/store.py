"""Personal and regional signs ("My signs", docs/ISL_DESIGN.md section 7).

Storage (local, git-ignored)::

    data/my_signs/<pack_id>/signs.jsonl         one row per sign version (last row per sign_id wins; deletes are tombstones)
    data/my_signs/<pack_id>/takes/<id>/<n>.npz  RawSequence of each take (landmarks only, never video)
    data/my_signs/<pack_id>/emb_<key>.npy       cached embeddings (+ emb_<key>.rows.json), key = embedder_key()

Embeddings depend on the encoder, so takes are always kept as landmarks and re-embedded lazily when the pack's
encoder changes. The cache key is the encoder version **plus** a hash of its weights and feature config: every
ISL encoder is ``isl-enc-1`` and the pack id is always ``isl``, so the version alone would keep a new pack's
users on the old pack's vectors. Rows also carry ``gloss`` and ``takes`` (relative paths) so the data layer's
``MySignsAdapter`` can read them.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
import weakref
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.features.build import FeatureError, sequence_quality
from app.isl.models.index import EmbeddingIndex
from app.landmarks.raw import RawSequence

MAX_TAKE_FRAMES = 300
MIN_TAKE_MS, MAX_TAKE_MS = 300.0, 4000.0
MIN_HAND_RATE = 0.5
TEXT_LIMITS = {"text_en": 80, "text_hi": 80, "region": 80, "note": 300, "author": 80}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
SIGN_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def clean_text(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    v = _CTRL.sub(" ", str(value)).strip()
    v = re.sub(r"\s+", " ", v)[:limit]
    return v or None


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class MySign:
    sign_id: str
    pack: str
    concept_id: str | None
    text_en: str
    text_hi: str | None = None
    region: str | None = None
    note: str | None = None
    n_takes: int = 0
    origin: str = "recorded"  # recorded | imported:<pack title>
    author: str | None = None
    created: str = ""
    enabled: bool = True
    deleted: bool = False
    theta: float | None = None  # per-sign acceptance cosine (lowest leave-one-out self-similarity - margin)
    licence: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> MySign:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class AddResult:
    sign: MySign | None
    take_quality: list[dict]
    consistency: float | None
    conflicts: list[dict]
    self_test: dict
    accepted: bool
    message: str

    def to_json(self) -> dict:
        return {"sign": self.sign.to_json() if self.sign else None, "take_quality": self.take_quality,
                "consistency": self.consistency, "conflicts": self.conflicts, "self_test": self.self_test,
                "accepted": self.accepted, "message": self.message}


def take_quality(raw: RawSequence) -> dict:
    q = sequence_quality(raw)
    dur = float(raw.t_ms[-1] - raw.t_ms[0]) if len(raw) > 1 else 0.0
    q["duration_ms"] = dur
    problems = []
    if len(raw) > MAX_TAKE_FRAMES:
        problems.append(f"too long ({len(raw)} frames, at most {MAX_TAKE_FRAMES})")
    if not q["usable"]:
        problems.append("body or hands not detected well enough")
    if q["any_hand_rate"] < MIN_HAND_RATE:
        problems.append("hands visible in less than half of the recording")
    if not (MIN_TAKE_MS <= dur <= MAX_TAKE_MS):
        problems.append(f"duration {dur / 1000:.1f} s (needs 0.3 to 4 s)")
    q["problems"] = problems
    q["ok"] = not problems
    return q


def loo_theta(z: np.ndarray, theta_margin: float = 0.05, s_min: float = -1.0) -> tuple[list[float], float | None]:
    """Leave-one-out self-similarity of each take (best cosine to the sign's other takes) and the per-sign
    acceptance cosine derived from it (lowest LOO similarity minus a margin, never below ``s_min``)."""
    n = len(z)
    if n < 2:
        return [], None
    S = z @ z.T
    loo = [float(max(S[i, j] for j in range(n) if j != i)) for i in range(n)]
    return loo, max(min(loo) - theta_margin, s_min)


def theta_from_embeddings(z: np.ndarray, *, s_min: float = -1.0, c_min: float = 0.5,
                          theta_margin: float = 0.05) -> float | None:
    """Per-sign acceptance cosine from the takes' embeddings (``loo_theta``), or ``None`` (= never accepted without
    asking) when there are fewer than 2 takes, a vector is not finite, or the takes are not consistent enough
    (mean pairwise cosine below ``c_min``). The one rule for imported signs, re-embedded signs (``user_index``) and
    the E4 protocol (docs/ISL_DESIGN.md 15.27)."""
    z = np.asarray(z, np.float32)
    if len(z) < 2 or not np.all(np.isfinite(z)) or _pairwise_mean(z) < c_min:
        return None
    return loo_theta(z, theta_margin, s_min)[1]


def calibrate_theta(takes: list[RawSequence], embedder, *, s_min: float = -1.0, c_min: float = 0.5,
                    theta_margin: float = 0.05) -> float | None:
    """Per-sign acceptance cosine computed with *this* app's embedder (used for imported signs, whose stored
    value came from another encoder and is untrusted). ``None`` (= never accepted without asking) when the takes
    cannot be embedded, are fewer than 2, or are not consistent enough to be trusted."""
    if embedder is None or len(takes) < 2:
        return None
    try:
        z = embedder.embed_raws(takes)
    except FeatureError:
        return None
    return theta_from_embeddings(z, s_min=s_min, c_min=c_min, theta_margin=theta_margin)


def _pairwise_mean(z: np.ndarray) -> float:
    n = len(z)
    if n < 2:
        return 1.0
    s = z @ z.T
    return float((s.sum() - np.trace(s)) / (n * (n - 1)))


_EMBEDDER_KEYS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def embedder_key(embedder) -> str:
    """Identity of the vectors an embedder produces: ``<version>-<sha256[:12]>`` over its feature config(s) and,
    for a trained encoder, its weights. Computed once per embedder object (loaded encoders are not trained
    further). An embedder without a feature config or a model (test doubles) is keyed by its version."""
    try:
        return _EMBEDDER_KEYS[embedder]
    except (KeyError, TypeError):
        pass
    h = hashlib.sha256()
    inner = getattr(embedder, "embedder", None)
    for obj in (embedder, inner):
        fc = getattr(obj, "feature_config", None)
        if fc is not None and hasattr(fc, "to_dict"):
            h.update(json.dumps(fc.to_dict(), sort_keys=True, default=str).encode())
    for obj in (embedder, inner):
        model = getattr(obj, "model", None)
        if model is not None and hasattr(model, "state_dict"):
            for name, t in model.state_dict().items():
                h.update(name.encode())
                h.update(t.detach().cpu().numpy().tobytes())
            break
    key = f"{embedder.version}-{h.hexdigest()[:12]}"
    try:
        _EMBEDDER_KEYS[embedder] = key
    except TypeError:
        pass
    return key


class MySignsStore:
    def __init__(self, root: Path, pack_id: str):
        self.pack_id = pack_id
        self.dir = Path(root) / pack_id
        self.file = self.dir / "signs.jsonl"
        self._lock = threading.RLock()
        self._cache: dict = {}
        self._rows_cache: tuple[tuple, dict[str, dict]] | None = None

    # ------------------------------------------------------------------ rows
    def _rows(self) -> dict[str, dict]:
        """Last row per sign_id. Parsed once per (mtime, size) of signs.jsonl: the recogniser looks signs up for
        every candidate of every segment, so re-reading the whole file each time would not scale. Callers must
        not mutate the returned rows."""
        with self._lock:
            stamp = self.stamp()
            if self._rows_cache is not None and self._rows_cache[0] == stamp:
                return self._rows_cache[1]
            rows = self._read_rows()
            self._rows_cache = (stamp, rows)
            return rows

    def _read_rows(self) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        if not self.file.exists():
            return rows
        for ln in self.file.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and r.get("sign_id"):
                rows[r["sign_id"]] = r
        return rows

    def _append(self, *signs: MySign) -> None:
        lines = []
        for sign in signs:
            row = sign.to_json()
            row["gloss"] = sign.text_en
            row["takes"] = [f"takes/{sign.sign_id}/{i}.npz" for i in range(sign.n_takes)]
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock, self.file.open("a", encoding="utf-8") as fh:
            fh.write("".join(lines))
            self._rows_cache = None

    def list(self, include_disabled: bool = True) -> list[MySign]:
        with self._lock:
            out = [MySign.from_json(r) for r in self._rows().values()]
        out = [s for s in out if not s.deleted and (include_disabled or s.enabled)]
        return sorted(out, key=lambda s: (s.text_en.lower(), s.created))

    def get(self, sign_id: str) -> MySign | None:
        r = self._rows().get(str(sign_id))
        if r is None:
            return None
        s = MySign.from_json(r)
        return None if s.deleted else s

    def takes(self, sign_id: str) -> list[RawSequence]:
        s = self.get(sign_id)
        if s is None:
            return []
        return [RawSequence.load(self.dir / "takes" / sign_id / f"{i}.npz") for i in range(s.n_takes)
                if (self.dir / "takes" / sign_id / f"{i}.npz").exists()]

    def stamp(self) -> tuple:
        try:
            st = self.file.stat()
            return (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            return (0, 0)

    # --------------------------------------------------------------- writes
    def save_sign(self, sign: MySign, takes: list[RawSequence]) -> MySign:
        """Store a sign and its takes as given (no checks: used by ``add`` and by pack import)."""
        with self._lock:
            d = self.dir / "takes" / sign.sign_id
            d.mkdir(parents=True, exist_ok=True)
            for i, raw in enumerate(takes):
                raw.save(d / f"{i}.npz")
            sign.n_takes = len(takes)
            self._append(sign)
        return sign

    def update(self, sign_id: str, **fields) -> MySign:
        with self._lock:
            s = self.get(sign_id)
            if s is None:
                raise KeyError(sign_id)
            for k, v in fields.items():
                if v is None or k not in ("text_en", "text_hi", "region", "note", "enabled", "concept_id"):
                    continue
                if k in TEXT_LIMITS:
                    v = clean_text(v, TEXT_LIMITS[k])
                    if k == "text_en" and not v:
                        raise ValueError("text_en must not be empty")
                elif k == "concept_id":
                    v = clean_text(v, 160)  # "" unlinks the sign from its dictionary concept
                setattr(s, k, bool(v) if k == "enabled" else v)
            self._append(s)
            return s

    def delete(self, sign_id: str) -> None:
        with self._lock:
            s = self.get(sign_id)
            if s is None:
                raise KeyError(sign_id)
            s.deleted = True
            self._append(s)
            shutil.rmtree(self.dir / "takes" / sign_id, ignore_errors=True)

    def delete_all(self) -> int:
        """Tombstone every sign in one append (``delete`` per sign would re-read the file each time)."""
        with self._lock:
            signs = self.list()
            for s in signs:
                s.deleted = True
            if signs:
                self._append(*signs)
            for s in signs:
                shutil.rmtree(self.dir / "takes" / s.sign_id, ignore_errors=True)
            return len(signs)

    # ------------------------------------------------------------ embedding
    def user_index(self, embedder, include_disabled: bool = False) -> EmbeddingIndex | None:
        """Index over every take of every enabled sign; row ``concept_id`` = ``mysign:<sign_id>``.
        Cached in memory and on disk per (``embedder_key``, signs.jsonl stamp).

        Each sign's acceptance cosine is recomputed from the vectors of THIS embedder (``theta_from_embeddings``,
        read with :meth:`user_theta`): the ``theta`` stored at enrolment is a cosine of the encoder of that day, and
        after a pack update the takes are re-embedded, so the old value is on another scale (docs 15.27)."""
        ekey = embedder_key(embedder)
        key = (ekey, self.stamp(), include_disabled)
        with self._lock:
            if self._cache.get("key") == key:
                return self._cache["index"]
            signs = self.list(include_disabled=include_disabled)
            signs = [s for s in signs if include_disabled or s.enabled]
            rows, _embs = [], []
            sign_keys: dict[str, list[str]] = {}
            disk = self.dir / f"emb_{re.sub(r'[^A-Za-z0-9_.-]', '_', ekey)}.npy"
            disk_rows = disk.with_suffix(".rows.json")
            cached: dict[str, np.ndarray] = {}
            if disk.exists() and disk_rows.exists():
                try:
                    old = json.loads(disk_rows.read_text())
                    arr = np.load(disk, allow_pickle=False)
                    cached = {k: arr[i] for i, k in enumerate(old) if i < len(arr)}
                except (ValueError, OSError):
                    cached = {}
            keys = []
            for s in signs:
                todo = []
                for i in range(s.n_takes):
                    p = self.dir / "takes" / s.sign_id / f"{i}.npz"
                    try:
                        st = p.stat()
                    except FileNotFoundError:
                        continue
                    # keyed by file identity: a sign id re-imported with new takes must not reuse old vectors
                    k = f"{s.sign_id}/{i}@{st.st_mtime_ns}:{st.st_size}"
                    if k not in cached:
                        todo.append((k, RawSequence.load(p)))
                    keys.append(k)
                    sign_keys.setdefault(s.sign_id, []).append(k)
                    rows.append({"variant_id": f"mysign:{s.sign_id}", "concept_id": f"mysign:{s.sign_id}",
                                 "linked_concept": s.concept_id, "source": "mysigns", "kind": "take", "weight": 1.0})
                if todo:
                    try:
                        z = embedder.embed_raws([r for _, r in todo])
                    except FeatureError:
                        z = np.zeros((len(todo), embedder.dim), np.float32)
                    for (k, _), v in zip(todo, z):
                        cached[k] = v
            if not rows:
                idx = None
            else:
                emb = np.stack([cached[k] for k in keys])
                idx = EmbeddingIndex.build(emb, rows, embedder.version, profile="user")
                try:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    np.save(disk, emb.astype(np.float32))
                    disk_rows.write_text(json.dumps(keys))
                except OSError:
                    pass
            # each sign's vectors under THIS embedder; thetas are computed from them on demand, per enrolment
            # setting (user_theta), so a pack's own enrol.c_min is applied, not a hard-coded 0.5 (docs 15.28)
            vecs = {sid: np.stack([cached[k] for k in ks]) for sid, ks in sign_keys.items()}
            self._cache = {"key": key, "index": idx, "vecs": vecs, "thetas": {}, "ekey": ekey}
            return idx

    def user_theta(self, sign_id: str, embedder, *, c_min: float = 0.5,
                   theta_margin: float = 0.05) -> float | None:
        """The sign's acceptance cosine for ``embedder``'s vectors (``theta_from_embeddings`` of the takes that
        :meth:`user_index` embedded); ``None`` when it has fewer than 2 usable takes or they are not consistent under
        this embedder (never auto-accepted). ``c_min`` / ``theta_margin``: the pack's enrolment settings, the ones
        :meth:`add` used (``fusion.enrol``); results are cached per (sign, c_min, theta_margin)."""
        if self.user_index(embedder) is None:
            return None
        with self._lock:
            memo = self._cache.setdefault("thetas", {})
            mk = (sign_id, float(c_min), float(theta_margin))
            if mk not in memo:
                z = (self._cache.get("vecs") or {}).get(sign_id)
                memo[mk] = (None if z is None else
                            theta_from_embeddings(z, c_min=float(c_min), theta_margin=float(theta_margin)))
            return memo[mk]

    # -------------------------------------------------------------- enrolment
    def add(self, text_en: str, takes: list[RawSequence], *, embedder=None, text_hi: str | None = None,
            concept_id: str | None = None, region: str | None = None, note: str | None = None,
            author: str | None = None, dry_run: bool = False, global_search=None, s_min: float = -1.0,
            c_min: float = 0.5, min_takes: int = 3, max_takes: int = 5, theta_margin: float = 0.05,
            describe_concept=None) -> AddResult:
        """Enrolment with the rules of docs/ISL_DESIGN.md 7.2.

        ``global_search(z) -> list[(concept_id, cosine)]`` (best first) searches the pack's dictionary index;
        ``describe_concept(concept_id) -> str`` names a concept for messages."""
        text_en = clean_text(text_en, TEXT_LIMITS["text_en"]) or ""
        tq = [take_quality(t) for t in takes]
        base = dict(take_quality=tq, consistency=None, conflicts=[], self_test={"hits": 0, "n": 0})
        if not text_en:
            return AddResult(None, accepted=False, message="Please type the English word for this sign.", **base)
        if not (min_takes <= len(takes) <= max_takes) and not (dry_run and len(takes) >= 1):
            return AddResult(None, accepted=False,
                             message=f"Please record the sign {min_takes} to {max_takes} times (you recorded {len(takes)}).",
                             **base)
        bad = [i + 1 for i, q in enumerate(tq) if not q["ok"]]
        if bad:
            return AddResult(None, accepted=False, message=(
                f"Recording {', '.join(map(str, bad))} could not be used: "
                + "; ".join(sorted({p for i in bad for p in tq[i - 1]["problems"]}))
                + ". Please record again with your hands and upper body in view."), **base)
        if embedder is None:
            return AddResult(None, accepted=False, message="Sign embedding is not available for this language.", **base)
        try:
            z = embedder.embed_raws(takes)
        except FeatureError as e:
            return AddResult(None, accepted=False, message=f"Could not read the recordings: {e}", **base)
        cons = _pairwise_mean(z)
        base["consistency"] = cons
        # conflicts: global dictionary + other user signs
        own_concept = concept_id
        hits: dict[str, float] = {}
        if global_search is not None:
            for zi in z:
                for cid, sc in global_search(zi[None, :])[:5]:
                    if cid != own_concept:
                        hits[cid] = max(hits.get(cid, -1.0), sc)
        uidx = self.user_index(embedder)
        if uidx is not None:
            cm = uidx.concept_matrix(z).mean(0)
            for j, cid in enumerate(uidx.concepts):
                hits[cid] = max(hits.get(cid, -1.0), float(cm[j]))
        conflicts = []
        for cid, sc in sorted(hits.items(), key=lambda kv: -kv[1]):
            if sc >= cons and len(conflicts) < 5:
                name = None
                if cid.startswith("mysign:"):
                    other = self.get(cid.removeprefix("mysign:"))
                    name = f"{other.text_en} (My signs)" if other else cid
                elif describe_concept is not None:
                    name = describe_concept(cid)
                conflicts.append({"concept_id": cid, "score": sc, "name": name or cid})
        base["conflicts"] = conflicts
        # leave-one-out self test against this sign's other takes, user signs and the dictionary
        n = len(z)
        loo, theta = loo_theta(z, theta_margin, s_min)
        best_rival = max(hits.values()) if hits else -1.0
        hits_n = sum(int(own >= best_rival) for own in loo)
        base["self_test"] = {"hits": hits_n, "n": n if n >= 2 else 0}
        if n >= 2 and cons < c_min:
            return AddResult(None, accepted=False, message=(
                "Your recordings look different from each other. Please sign the same way each time."), **base)
        if dry_run:
            msg = "Looks good." if not conflicts else (
                f"This looks like '{conflicts[0]['name']}'. Save anyway as your own version?")
            return AddResult(None, accepted=len(takes) >= min_takes, message=msg, **base)
        sign = MySign(sign_id=str(uuid.uuid4()), pack=self.pack_id, concept_id=concept_id, text_en=text_en,
                      text_hi=clean_text(text_hi, TEXT_LIMITS["text_hi"]), region=clean_text(region, TEXT_LIMITS["region"]),
                      note=clean_text(note, TEXT_LIMITS["note"]), author=clean_text(author, TEXT_LIMITS["author"]),
                      created=now_iso(), theta=theta)
        self.save_sign(sign, takes)
        msg = f"Saved. Recognised {hits_n} of {n} recordings in the self-test."
        if conflicts:
            msg += f" Note: it looks similar to '{conflicts[0]['name']}'."
        return AddResult(sign, accepted=True, message=msg, **base)

