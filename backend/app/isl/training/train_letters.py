"""Letter-classifier training (docs/ISL_DESIGN.md 6.4).

Input: recordings of the manual alphabet signed in order (A -> Z): holds are detected and aligned to the letters
with a monotonic dynamic programme that may skip spurious holds (it keeps the longest holds in order). Frames
between holds are "noise". The ISLRTC dictionary has no A-Z sequence video (its "Alphabet" entry is the sign
for the word), so real training data has to come from recordings (for example a school in Indore, or users'
alphabet calibration); until 5+ signers exist the module is labelled experimental.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.features.build import FeatureError
from app.isl.fingerspell.features import LETTER_DIM, letter_features_seq
from app.isl.fingerspell.holds import Hold, detect_holds
from app.isl.fingerspell.model import LETTERS, NOISE, LetterClassifier
from app.landmarks.raw import RawSequence


def align_in_order(holds: list[Hold], n_letters: int) -> list[int] | None:
    """Pick ``n_letters`` holds in order maximising the total hold duration (skips spurious short holds).
    Returns hold indices, or None when there are fewer holds than letters."""
    H = len(holds)
    if H < n_letters:
        return None
    dur = np.array([h.end - h.start for h in holds], np.float64)
    NEG = -1e18
    dp = np.full((H + 1, n_letters + 1), NEG)
    dp[:, 0] = 0.0
    for i in range(1, H + 1):
        for j in range(1, min(i, n_letters) + 1):
            dp[i, j] = max(dp[i - 1, j], dp[i - 1, j - 1] + dur[i - 1])
    out, i, j = [], H, n_letters
    while j > 0:
        if dp[i, j] == dp[i - 1, j] and i - 1 >= j:
            i -= 1
        else:
            out.append(i - 1)
            i, j = i - 1, j - 1
    return out[::-1]


def examples_from_sequence(raw: RawSequence, letters: list[str]) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
    """(letter features, labels, noise features) from one in-order recording."""
    try:
        feats, _ = letter_features_seq(raw)
    except FeatureError:
        return [], [], []
    holds = detect_holds(raw, feats=feats)
    pick = align_in_order(holds, len(letters))
    if pick is None:
        return [], [], []
    X, y = [], []
    in_hold = np.zeros(len(feats), bool)
    for li, hi in enumerate(pick):
        h = holds[hi]
        for f in range(h.start, h.end):
            X.append(feats[f])
            y.append(LETTERS.index(letters[li]))
        in_hold[h.start:h.end] = True
    noise = [feats[i] for i in np.where(~in_hold)[0]]
    return X, y, noise


def train_letter_model(X: list[np.ndarray], y: list[int], noise: list[np.ndarray], out: Path, *, epochs: int = 60,
                       seed: int = 0, n_signers: int = 1, source: str = "", smoke: bool = False) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xa = np.asarray(X + noise, np.float32).reshape(-1, LETTER_DIM)
    ya = np.asarray(list(y) + [NOISE] * len(noise), np.int64)
    if len(set(y)) < 2:
        raise RuntimeError("need examples of at least two letters")
    model = LetterClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xt, yt = torch.from_numpy(Xa), torch.from_numpy(ya)
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(len(Xt)))
        for a in range(0, len(perm), 64):
            idx = perm[a:a + 64]
            xb = Xt[idx] + 0.03 * torch.randn_like(Xt[idx])  # feature noise as augmentation
            loss = crit(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((model(Xt).argmax(1) == yt).float().mean())
    meta = {"kind": "isl-letters", "labels": LETTERS, "in_dim": LETTER_DIM, "hidden": [128, 64], "temperature": 1.0,
            "trained_letters": sorted({LETTERS[i] for i in y}), "n_signers": n_signers, "source": source,
            "train_acc": acc, "n_examples": len(Xa), "smoke": smoke}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, out)
    return {"checkpoint": str(out), "train_acc_on_training_frames": acc, "examples": len(Xa),
            "letters": meta["trained_letters"], "seconds": round(time.time() - t0, 1)}
