"""Per-user alphabet calibration (docs/ISL_DESIGN.md 6.4): the user signs each letter once (or more) in
"My signs"; each letter is stored as the mean hand-shape features of the main hold of that recording, and
letters are recognised by distance to these prototypes. This works without any trained letter model and adapts
to the user's own hands, camera and regional letter forms. Stored at ``data/my_signs/<pack>/alphabet.npz``
(landmark-derived numbers only)."""

from __future__ import annotations

import threading
import zipfile
from pathlib import Path

import numpy as np

from app.features.build import FeatureError
from app.isl.fingerspell.features import LETTER_DIM, letter_features_seq
from app.isl.fingerspell.holds import detect_holds
from app.isl.fingerspell.model import LETTERS, NOISE
from app.landmarks.raw import RawSequence

MIN_LETTERS = 5
# ``IslPack.alphabet()`` makes a new store per request, so the lock must be shared per file, not per instance:
# two letters recorded at once would otherwise both read-modify-write the file and one would be lost.
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(str(Path(path).resolve()), threading.Lock())


def hold_features(raw: RawSequence, min_ms: float = 150.0, v_max: float = 0.45) -> tuple[np.ndarray, float] | None:
    """Mean letter features of the longest hold of a recording (the letter); None without hands/body."""
    try:
        feats, t = letter_features_seq(raw)
    except FeatureError:
        return None
    holds = detect_holds(raw, min_ms, v_max, feats=feats)
    if holds:
        h = max(holds, key=lambda x: x.end - x.start)
        return h.mean_features, float(h.t_end - h.t_start)
    pres = feats[:, 120 + 3 + 10] + feats[:, 120 + 3 + 11] > 0  # presence flags
    if not pres.any():
        return None
    mid = np.where(pres)[0]
    return feats[mid[len(mid) // 4: max(len(mid) // 4 + 1, 3 * len(mid) // 4)]].mean(0), 0.0


class PrototypeLetters:
    """Same interface as ``LetterPredictor`` (``labels``, ``log_probs``, ``n_signers``, ``trained_letters``)."""

    def __init__(self, protos: dict[str, list[np.ndarray]]):
        self.labels = list(LETTERS)
        self.protos = {k: np.stack(v) for k, v in protos.items() if v}
        self.trained_letters = sorted(self.protos)
        self.n_signers = 1
        self.temperature = 1.0
        allp = [p for v in self.protos.values() for p in v]
        if len(allp) >= 2:
            P = np.stack(allp)
            d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
            med = float(np.median(d[np.triu_indices(len(P), 1)]))
        else:
            med = 1.0
        self.sigma = max(1e-3, 0.35 * med)
        self.noise_dist = 0.75 * med

    def log_probs(self, feats: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(feats, np.float64))
        logits = np.full((len(X), len(self.labels)), -1e4)
        for k, P in self.protos.items():
            d2 = ((X[:, None, :] - P[None]) ** 2).sum(-1).min(1)
            logits[:, self.labels.index(k)] = -d2 / (2 * self.sigma ** 2)
        logits[:, NOISE] = -(self.noise_dist ** 2) / (2 * self.sigma ** 2)
        m = logits.max(1, keepdims=True)
        return (logits - m - np.log(np.exp(logits - m).sum(1, keepdims=True))).astype(np.float32)


class AlphabetStore:
    def __init__(self, folder: Path):
        self.path = Path(folder) / "alphabet.npz"
        self._lock = _file_lock(self.path)

    def load(self) -> dict[str, list[np.ndarray]]:
        """The calibration, or {} when there is none or the file is unreadable (the language list and the
        recogniser read it on every request, so a damaged file must not turn into server errors)."""
        if not self.path.exists():
            return {}
        try:
            with np.load(self.path, allow_pickle=False) as z:
                labels, feats = z["labels"].tolist(), np.asarray(z["feats"], np.float32)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return {}
        if feats.ndim != 2 or feats.shape[1:] != (LETTER_DIM,):
            return {}
        out: dict[str, list[np.ndarray]] = {}
        for lab, f in zip(labels, feats):
            out.setdefault(str(lab), []).append(f)
        return out

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.load().items())}

    def add(self, letter: str, feats: np.ndarray, keep: int = 3) -> dict[str, int]:
        letter = letter.upper()
        if letter not in LETTERS or letter == "noise":
            raise ValueError("letter must be A-Z or 0-9")
        if feats.shape != (LETTER_DIM,) or not np.all(np.isfinite(feats)):
            raise ValueError("invalid letter features")
        with self._lock:
            d = self.load()
            d.setdefault(letter, []).append(feats.astype(np.float32))
            d[letter] = d[letter][-keep:]
            self._save(d)
        return self.counts()

    def _save(self, d: dict) -> None:
        labels = [k for k, v in d.items() for _ in v]
        feats = [f for v in d.values() for f in v]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"alphabet.{threading.get_ident()}.tmp.npz")
        np.savez(tmp, labels=np.array(labels, dtype="U8"), feats=np.stack(feats).astype(np.float32))
        tmp.replace(self.path)

    def reset(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)

    def model(self) -> PrototypeLetters | None:
        d = self.load()
        return PrototypeLetters(d) if len(d) >= MIN_LETTERS else None
