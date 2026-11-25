"""ISLRTC first-sign refinement, stage 2: a small learned boundary model (docs/ISL_DESIGN.md section 5.3).

Training data is synthetic, built from clips whose sign boundary we trust:

* clean sign A: an INCLUDE clip (one sign each) or an ISLRTC clip trimmed by stage 1 with high confidence
  (``source_end``/``rest``, confidence >= 0.8);
* a 0-300 ms transition;
* an explanation-like tail: 2-6 s of an ISLRTC ``max_s`` clip B **after** its stage-1 end.

Label = the frame where A ends. To avoid a trivial "the person changed" cue, A and B are each converted to
per-frame activity features in their own body frame and joined in feature space (with an interpolated
transition), never as raw landmarks.

Model: 1-layer BiGRU (hidden 64) over the per-frame activity features, predicting a boundary probability
per frame. Decoding takes the first peak above 0.5 after the minimum sign length and accepts it only when it
agrees (within 2 frames) with a stage-1 candidate boundary; the method is then ``learned``. Deviation from the
design: the regular per-frame feature vector is not added to the input (activity features only), which keeps
training to a few CPU minutes; it can be added later if the boundary error stays high.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.isl.data.trim import METHOD_CONFIDENCE, TrimConfig, TrimResult, activity_signal, candidate_boundaries
from app.landmarks.raw import RawSequence

FEATURE_NAMES = ("wy_l", "wy_r", "hand_l", "hand_r", "up_l", "up_r", "rest", "dip", "speed", "shape", "dt")


def frame_features(raw: RawSequence, cfg: TrimConfig = TrimConfig()) -> np.ndarray:
    """(T, 11) float32 per-frame activity features (scaled to roughly unit range)."""
    s = activity_signal(raw, cfg)
    T = len(s["t"])
    wy = [np.nan_to_num(s[f"wrist_y_{k}"], nan=2.0) for k in ("left", "right")]
    dt = np.diff(s["t"], prepend=s["t"][0] - float(s["dt"][0])) / 100.0
    X = np.stack([
        np.clip(wy[0], -2, 2), np.clip(wy[1], -2, 2), s["hand_left"], s["hand_right"],
        s["up_left"], s["up_right"], s["rest"], s["dip"], np.clip(s["speed"] / 3.0, 0, 3),
        np.clip(s["shape_change"] / 3.0, 0, 3), np.clip(dt, 0, 3),
    ], axis=1).astype(np.float32)
    assert X.shape == (T, len(FEATURE_NAMES))
    return X


def make_example(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, fps: float = 12.0,
                 max_transition_ms: float = 300.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Join feature sequences A + transition + B. Returns (X, y, boundary_index)."""
    n_tr = int(round(rng.uniform(0, max_transition_ms) / 1000.0 * fps))
    if n_tr > 0:
        w = np.linspace(0, 1, n_tr + 2)[1:-1, None]
        tr = (1 - w) * a[-1:] + w * b[:1]
    else:
        tr = np.zeros((0, a.shape[1]), np.float32)
    X = np.concatenate([a, tr, b]).astype(np.float32)
    y = np.zeros(len(X), np.float32)
    k = len(a)
    y[max(0, k - 1): k + 1] = 1.0  # the end of A (+1 frame tolerance)
    return X, y, k


@dataclass
class BoundaryTrainResult:
    epochs: int
    train_examples: int
    val_examples: int
    val_within_2: float  # fraction of validation examples whose decoded end is within 2 frames
    val_mean_abs_err: float


def _model(hidden: int = 64):
    import torch.nn as nn

    class BoundaryNet(nn.Module):
        def __init__(self, d_in: int = len(FEATURE_NAMES), h: int = hidden):
            super().__init__()
            self.gru = nn.GRU(d_in, h, num_layers=1, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * h, 1)

        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o).squeeze(-1)

    return BoundaryNet()


def _pad(batch):
    import torch

    L = max(len(x) for x, _ in batch)
    X = torch.zeros(len(batch), L, batch[0][0].shape[1])
    Y = torch.zeros(len(batch), L)
    M = torch.zeros(len(batch), L)
    for i, (x, y) in enumerate(batch):
        X[i, : len(x)] = torch.from_numpy(x)
        Y[i, : len(y)] = torch.from_numpy(y)
        M[i, : len(y)] = 1
    return X, Y, M


def decode(prob: np.ndarray, min_index: int, threshold: float = 0.5) -> int | None:
    """First local peak above ``threshold`` at or after ``min_index``."""
    for i in range(max(0, min_index), len(prob)):
        if prob[i] >= threshold and (i + 1 >= len(prob) or prob[i] >= prob[i + 1]):
            return i + 1  # exclusive end
    return None


def train_boundary(clean: list[np.ndarray], tails: list[np.ndarray], n_examples: int = 2000, epochs: int = 15,
                   seed: int = 0, threads: int = 2, val_frac: float = 0.2, min_sign_frames: int = 5):
    """Train on synthetic joins of ``clean`` (feature arrays of clean signs) and ``tails`` (explanation-like
    feature arrays). Returns (model, BoundaryTrainResult)."""
    import torch

    torch.set_num_threads(max(1, threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if not clean or not tails:
        raise ValueError("need clean signs and tails to train the boundary model")
    # split sources (not examples) so validation joins unseen clips
    ci = rng.permutation(len(clean))
    ti = rng.permutation(len(tails))
    nc_val = max(1, int(len(clean) * val_frac)) if len(clean) > 1 else 0
    nt_val = max(1, int(len(tails) * val_frac)) if len(tails) > 1 else 0
    c_tr, c_va = ci[nc_val:] if nc_val else ci, ci[:nc_val] if nc_val else ci
    t_tr, t_va = ti[nt_val:] if nt_val else ti, ti[:nt_val] if nt_val else ti

    def examples(cs, ts, n):
        out = []
        for _ in range(n):
            a = clean[int(rng.choice(cs))]
            b = tails[int(rng.choice(ts))]
            x, y, k = make_example(a, b, rng)
            out.append((x, y, k))
        return out

    train = examples(c_tr, t_tr, n_examples)
    val = examples(c_va, t_va, max(50, n_examples // 5))
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    pos_w = torch.tensor(10.0)
    bs = 32
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        for s in range(0, len(order), bs):
            X, Y, M = _pad([train[i][:2] for i in order[s: s + bs]])
            logit = model(X)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, Y, pos_weight=pos_w, reduction="none")
            loss = (loss * M).sum() / M.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    errs = []
    with torch.no_grad():
        for x, _y, k in val:
            p = torch.sigmoid(model(torch.from_numpy(x)[None]))[0].numpy()
            e = decode(p, min_sign_frames)
            errs.append(abs((e if e is not None else len(x)) - k))
    errs = np.array(errs, dtype=np.float64)
    res = BoundaryTrainResult(epochs, len(train), len(val), float((errs <= 2).mean()), float(errs.mean()))
    return model, res


def save_model(model, path: Path, meta: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path: Path):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = _model()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck.get("meta", {})


def refine(raw: RawSequence, stage1: TrimResult, model, cfg: TrimConfig = TrimConfig(), tol: int = 2) -> TrimResult:
    """Apply the learned model; override stage 1 only when the learned end agrees with a candidate boundary."""
    import torch

    if stage1.method in ("empty", "full"):
        return stage1
    x = frame_features(raw, cfg)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x)[None]))[0].numpy()
    sig = activity_signal(raw, cfg)
    dt = float(sig["dt"][0])
    min_idx = stage1.start + int(round(cfg.min_sign_ms / dt))
    e = decode(p, min_idx)
    if e is None:
        return stage1
    cands = [c + 1 for c in candidate_boundaries(sig, sig["t"], after=stage1.start)] + [stage1.end] + list(stage1.alternatives)
    near = [c for c in cands if abs(c - e) <= tol]
    if not near:
        return stage1
    new_end = min(near, key=lambda c: abs(c - e))
    if new_end == stage1.end:
        return stage1
    alts = sorted({stage1.end, *stage1.alternatives} - {new_end})[:2]
    t = sig["t"]
    return TrimResult(stage1.start, new_end, stage1.t_start_ms, float(t[new_end - 1]), "learned",
                      METHOD_CONFIDENCE["learned"], alts, stage1.notes + [f"stage1 {stage1.method} end {stage1.end}"])
