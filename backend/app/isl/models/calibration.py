"""Fitting the fusion parameters and decision thresholds on validation data (docs/ISL_DESIGN.md 6.3 d).

Everything here is a small grid search over cached validation evidence (``FusionInputs``), so it runs in
seconds on CPU. Targets: accepted words >= 95 % correct for ``balanced`` (98 % ``strict``, 90 % ``lenient``);
"did you mean" only when the true word is in the top 5 for >= 80 % of such cases. When a target cannot be
reached on the available data, the closest achievable value is used and the *measured* numbers are stored in
``FusionParams.achieved`` (the UI shows those, never the target).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from app.isl.models.fusion import POLICIES, FusionInputs, FusionParams, FusionResult, decide, fuse, normalize_accept

TARGETS = {"strict": 0.98, "balanced": 0.95, "lenient": 0.90}  # on the 95 % lower bound of accepted precision
SUGGEST_TOP5_TARGET = 0.80


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """95 % Wilson lower bound of a proportion: a precision target only counts as reached when the validation
    data supports it (5 of 5 correct gives 0.57, not 1.0)."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Temperature T minimising the NLL of softmax(logits / T). ``logits`` may be log-probabilities."""
    if len(y) == 0:
        return 1.0
    grid = grid if grid is not None else np.exp(np.linspace(math.log(0.25), math.log(6.0), 49))
    best_t, best = 1.0, float("inf")
    for t in grid:
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        nll = float(np.mean(lse - z[np.arange(len(y)), y]))
        if nll < best:
            best_t, best = float(t), nll
    return best_t


def true_prob(res: FusionResult, truth: str) -> float:
    try:
        return float(res.probs[res.keys.index(truth)])
    except ValueError:
        return 0.0


def fusion_nll(inputs: list[FusionInputs], truths: list[str], params: FusionParams, floor: float = 1e-6) -> float:
    if not inputs:
        return float("nan")
    return float(np.mean([-math.log(max(floor, true_prob(fuse(i, params), t))) for i, t in zip(inputs, truths)]))


def fit_fusion_weights(inputs: list[FusionInputs], truths: list[str], base: FusionParams,
                       grid: dict | None = None) -> tuple[FusionParams, dict]:
    has_c = any(i.cls_logp is not None for i in inputs)
    has_r = any(i.ret is not None for i in inputs)
    g = grid or {
        "alpha": [0.5, 1.0, 1.5] if has_c else [0.0],
        "beta": [0.5, 1.0, 1.5] if has_r else [0.0],
        "gamma": [0.0, 0.5, 1.0],
        "tau_r": [0.02, 0.05, 0.1] if has_r else [base.tau_r],
    }
    best, best_p, tried = float("inf"), base, 0
    for a, b, c, t in itertools.product(g["alpha"], g["beta"], g["gamma"], g["tau_r"]):
        p = base.with_(alpha=a, beta=b, gamma=c, tau_r=t)
        nll = fusion_nll(inputs, truths, p)
        tried += 1
        if nll < best:
            best, best_p = nll, p
    return best_p, {"nll": best, "tried": tried, "grid": g}


def _decisions(results: list[FusionResult], params: FusionParams, policy: str) -> list[tuple[str, str]]:
    return [decide(r.probs, r.agree, len(r.per_crop_top1), params, policy, r.best_cos) for r in results]


def summarize(results: list[FusionResult], truths: list[str], params: FusionParams, policy: str) -> dict:
    """Risk/coverage numbers of the full decision rule (E3)."""
    n = len(results)
    dec = _decisions(results, params, policy)
    acc = [r.keys[0] == t if r.keys else False for r, t in zip(results, truths)]
    top5 = [t in r.keys[:5] for r, t in zip(results, truths)]
    a = [i for i, (d, _) in enumerate(dec) if d == "accept"]
    s = [i for i, (d, _) in enumerate(dec) if d == "suggest"]
    ab = [i for i, (d, _) in enumerate(dec) if d == "abstain"]
    return {
        "policy": policy, "n": n,
        "top1": float(np.mean(acc)) if n else None, "top5": float(np.mean(top5)) if n else None,
        "coverage": len(a) / n if n else None,
        "accepted_precision": float(np.mean([acc[i] for i in a])) if a else None,
        "silent_error_rate": sum(not acc[i] for i in a) / n if n else None,
        "suggest_rate": len(s) / n if n else None,
        "suggest_top5_hit": float(np.mean([top5[i] for i in s])) if s else None,
        "abstain_rate": len(ab) / n if n else None,
        "n_accepted": len(a),
    }


def fit_thresholds(results: list[FusionResult], truths: list[str], params: FusionParams,
                   targets: dict | None = None, min_accepted: int = 5) -> tuple[FusionParams, dict]:
    # every policy always gets a threshold (a missing one made decide() raise): unknown targets are ignored and
    # missing ones use the default target
    targets = {pol: float((targets or {}).get(pol, TARGETS[pol])) for pol in POLICIES}
    rep: dict = {"n": len(results)}
    if not results:
        return params, {**rep, "note": "no validation data: defaults kept (not calibrated)"}
    # out-of-vocabulary guard: 95 % of in-vocabulary queries pass
    cos = np.array([r.best_cos for r in results])
    s_min = float(np.quantile(cos, 0.05)) if (cos > -1).any() else -1.0
    params = params.with_(s_min=s_min)
    p1 = np.array([r.probs[0] if len(r.probs) else 0.0 for r in results])
    p2 = np.array([r.probs[1] if len(r.probs) > 1 else 0.0 for r in results])
    agree = np.array([r.agree for r in results])
    crops = np.array([len(r.per_crop_top1) for r in results])
    correct = np.array([bool(r.keys) and r.keys[0] == t for r, t in zip(results, truths)])
    in_cov = cos >= s_min
    accept = {}
    for pol, target in targets.items():
        best_t, best_prec, chosen = None, -1.0, None
        for t in np.round(np.arange(0.20, 0.991, 0.01), 2):
            m = (p1 >= t) & (p1 - p2 >= params.margin) & (agree >= np.minimum(params.min_agree, crops)) & in_cov
            if m.sum() < min_accepted:
                continue
            prec = wilson_lower(int(correct[m].sum()), int(m.sum()))  # lower bound, not the point estimate
            if prec >= target and chosen is None:
                chosen = float(t)
            if prec > best_prec or (prec == best_prec and best_t is not None and t > best_t):
                best_t, best_prec = float(t), prec
        accept[pol] = chosen if chosen is not None else (best_t if best_t is not None else params.accept[pol])
        rep[f"{pol}_target_reached"] = chosen is not None
    params = params.with_(accept=normalize_accept(accept))  # all three, strict >= balanced >= lenient
    # "did you mean": smallest top-5 mass with >= 80 % top-5 hits among shown suggestions
    dec = _decisions(results, params.with_(suggest_min_top5_mass=0.0), "balanced")
    idx = [i for i, (d, _) in enumerate(dec) if d == "suggest"]
    mass = np.array([float(np.sum(results[i].probs[:5])) for i in idx])
    hit = np.array([truths[i] in results[i].keys[:5] for i in idx])
    chosen_m, best_m, best_hit = None, params.suggest_min_top5_mass, -1.0
    for m in np.round(np.arange(0.0, 1.0, 0.05), 2):
        sel = mass >= m
        if sel.sum() < max(3, 0.2 * len(idx)):
            continue
        h = float(hit[sel].mean())
        if h >= SUGGEST_TOP5_TARGET and chosen_m is None:
            chosen_m = float(m)
        if h > best_hit:
            best_m, best_hit = float(m), h
    params = params.with_(suggest_min_top5_mass=chosen_m if chosen_m is not None else best_m)
    rep["suggest_target_reached"] = chosen_m is not None
    achieved = {pol: summarize(results, truths, params, pol) for pol in targets}
    params = params.with_(achieved=achieved, fitted=True)
    rep["achieved"] = achieved
    return params, rep
