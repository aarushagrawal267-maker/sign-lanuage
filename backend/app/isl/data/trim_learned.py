"""Learned end for ISLRTC clips that run straight from the sign into the explanation (trim-2, stage 2).

Stage 1 (``trim.py``) ends a sign at a rest/dip/pause or at the end of a bounded clip; when the signer goes
straight on into the explanation it can only fall back to ``cap`` (sign-length prior). This module trains the
stage-2 BiGRU of ``boundary.py`` (same architecture and per-frame activity features) on **same-signer spliced
clips** instead of cross-signer joins:

* take an ISLRTC ``max_s`` clip whose sign ends in an unambiguous hands-down run (first hands-down run of
  >= 150 ms after the start is >= 300 ms long, 1-8 s after the start);
* cut the hands-down frames out and bridge the gap with 0, 2 or 4 interpolated frames
  (:func:`app.isl.data.trim_synth.splice_out`), so the sign runs straight into the same signer's explanation;
* label = the last frame before the hands went down. The unspliced clip is added too (boundary = rest start).

Only ``cap`` results are changed (``refine_cap``): the learned end becomes the main cut when the model is
confident (peak >= ``threshold``) and the stage-1 prior cut becomes the ``second`` cut. Rest/dip/source_end
ends are kept: on the by-eye labels they are right about 90% of the time and the model did not beat them.

Usage (from ``backend/``)::

    python -m app.isl.data.trim_learned train [--half tune] [--epochs 10]   # -> artifacts/isl/trim_boundary.pt
    python -m app.isl.data.trim_learned eval  [--half report] [--model PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from app.isl.data.trim import (
    TrimConfig,
    TrimResult,
    _runs,
    _second_cut,
    _with_second,
    activity_signal,
    find_first_sign,
)
from app.landmarks.raw import RawSequence

MODEL_VERSION = "trim-learned-1"


def default_model_path() -> Path:
    from app.core.config import get_settings

    return get_settings().artifacts_dir / "isl" / "trim_boundary.pt"


def splice_point(raw: RawSequence, cfg: TrimConfig = TrimConfig()) -> tuple[int, int] | None:
    """(a, b): the unambiguous hands-down run after the first sign, or None."""
    try:
        sig = activity_signal(raw, cfg)
    except Exception:  # noqa: BLE001 - FeatureError and malformed clips
        return None
    t = sig["t"]
    act = sig["up_left"] | sig["up_right"]
    runs = [(a, b) for a, b in _runs(act) if t[b - 1] - t[a] >= 250]
    if not runs:
        return None
    t0 = t[runs[0][0]]
    dips = [(a, b) for a, b in _runs(sig["dip"]) if a > runs[0][0] and t[b - 1] - t[a] >= 150]
    if not dips:
        return None
    a, b = dips[0]
    if t[b - 1] - t[a] >= 300 and 1000 <= t[a] - t0 <= 8000 and b < len(t) - 12:
        return int(a), int(b)
    return None


def spliced_examples(raws: list[RawSequence], bridges: tuple[int, ...] = (0, 2, 4), with_original: bool = True,
                     cfg: TrimConfig = TrimConfig()) -> list[tuple[RawSequence, int]]:
    """(sequence, exclusive true end) pairs built from clips that have a :func:`splice_point`."""
    from app.isl.data.trim_synth import splice_out

    out = []
    for raw in raws:
        sp = splice_point(raw, cfg)
        if sp is None:
            continue
        a, b = sp
        for br in bridges:
            out.append((splice_out(raw, a, b, br), a))
        if with_original:
            out.append((raw, a))
    return out


def train(examples: list[tuple[RawSequence, int]], epochs: int = 10, seed: int = 0, threads: int = 2,
          cfg: TrimConfig = TrimConfig(), log=None):
    """Per-frame boundary model (``boundary._model``) on spliced examples; label = frames end-1 and end."""
    import torch

    from app.isl.data.boundary import _model, _pad, frame_features

    torch.set_num_threads(max(1, threads))
    torch.manual_seed(seed)
    data = []
    for x, e in examples:
        X = frame_features(x, cfg)
        y = np.zeros(len(X), np.float32)
        y[max(0, e - 1): e + 1] = 1.0
        data.append((X, y))
    if not data:
        raise ValueError("no training examples")
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(data))
        total = 0.0
        for s in range(0, len(order), 32):
            X, Y, M = _pad([data[i] for i in order[s: s + 32]])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X), Y, pos_weight=torch.tensor(10.0),
                                                                        reduction="none")
            loss = (loss * M).sum() / M.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if log:
            log(f"epoch {ep + 1}/{epochs} loss {total:.3f}")
    model.eval()
    return model


def save(model, path: Path, meta: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {"version": MODEL_VERSION, **meta}}, path)


def load(path: Path):
    import torch

    from app.isl.data.boundary import _model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = _model()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck.get("meta", {})


def boundary_prob(raw: RawSequence, model, cfg: TrimConfig = TrimConfig()) -> np.ndarray:
    import torch

    from app.isl.data.boundary import frame_features

    X = frame_features(raw, cfg)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X)[None]))[0].numpy()


def refine_cap(raw: RawSequence, res: TrimResult, model, n_words: int = 1, cfg: TrimConfig = TrimConfig(),
               threshold: float = 0.5) -> TrimResult:
    """Learned end for a ``cap`` result (other methods are returned unchanged)."""
    if res.method != "cap" or len(raw) < 4:
        return res
    p = boundary_prob(raw, model, cfg)
    t = raw.t_ms
    t0 = t[res.start]
    max_sign = min(cfg.max_sign_ms + cfg.per_word_max_ms * max(0, n_words - 1), cfg.hard_max_sign_ms)
    lo = int(np.searchsorted(t, t0 + cfg.min_sign_ms))
    hi = int(np.searchsorted(t, t0 + max_sign))
    if hi <= lo:
        return res
    k = lo + int(np.argmax(p[lo:hi]))
    if p[k] < threshold:
        return res
    end = k + 1
    if end == res.end:
        return res
    sig = activity_signal(raw, cfg)
    second = _second_cut(sig, t0, res.start, end, n_words, cfg, [res.end])
    notes = res.notes + [f"{MODEL_VERSION}: end {end} (p={p[k]:.2f}); stage-1 cap end {res.end}"]
    return TrimResult(res.start, end, res.t_start_ms, float(t[end - 1]), "cap", res.confidence,
                      _with_second([a for a in res.alternatives if a != end], second), notes, second)


# ------------------------------------------------------------------------------------------------ CLI

def _records(half: str | None):
    from app.isl.data.trim_bench import SKIP_KINDS, _half, _paths, islrtc_records, parsed

    paths = _paths()
    recs = [r for r in islrtc_records(paths) if r.get("stop") == "max_s" and parsed(r["id"]).kind not in SKIP_KINDS
            and (half is None or _half(r["id"]) == half)]
    return paths, sorted(recs, key=lambda r: r["id"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.isl.data.trim_learned")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--half", choices=["tune", "report"], default=None, help="train on one half only (for eval)")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--out", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--half", choices=["tune", "report"], default="report")
    ev.add_argument("--model", default=None)
    ev.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    threads = int(os.environ.get("SLR_TORCH_THREADS", "2"))
    if args.cmd == "train":
        paths, recs = _records(args.half)
        t0 = time.time()
        raws = [RawSequence.load(paths.resolve(r["path"])) for r in recs]
        ex = spliced_examples(raws)
        print(f"clips {len(recs)}, examples {len(ex)} ({time.time() - t0:.0f}s)", flush=True)
        model = train(ex, epochs=args.epochs, threads=threads, log=lambda m: print(m, flush=True))
        out = Path(args.out) if args.out else default_model_path()
        save(model, out, {"half": args.half, "clips": len(recs), "examples": len(ex), "epochs": args.epochs})
        print(json.dumps({"model": str(out), "secs": round(time.time() - t0)}), flush=True)
        return 0
    from app.isl.data.trim_bench import parsed, score_errors, words_for
    from app.isl.data.trim_synth import splice_out

    model, meta = load(Path(args.model) if args.model else default_model_path())
    if meta.get("half") == args.half or meta.get("half") is None:
        print(f"warning: model trained on half={meta.get('half')!r}; evaluating on {args.half!r} is not held out",
              file=sys.stderr)
    paths, recs = _records(args.half)
    e1, e2, best2 = [], [], []
    for r in recs:
        raw = RawSequence.load(paths.resolve(r["path"]))
        sp = splice_point(raw)
        if sp is None:
            continue
        x = splice_out(raw, sp[0], sp[1], 2)
        truth = float(x.t_ms[sp[0] - 1])
        pn = parsed(r["id"])
        nw = words_for(pn.kind, pn.text)
        s1 = find_first_sign(x, "max_s", n_words=nw)
        s2 = refine_cap(x, s1, model, nw, threshold=args.threshold)
        e1.append(float(x.t_ms[s1.end - 1]) - truth)
        err = float(x.t_ms[s2.end - 1]) - truth
        e2.append(err)
        if s2.second:
            alt = float(x.t_ms[min(len(x), s2.second) - 1]) - truth
            err = alt if abs(alt) < abs(err) else err
        best2.append(err)
    print(json.dumps({"half": args.half, "stage1": score_errors(e1), "learned_cap": score_errors(e2),
                      "learned_cap_best_of_2": score_errors(best2)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
