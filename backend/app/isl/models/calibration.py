"""Fitting the fusion parameters and decision thresholds on validation data (docs/ISL_DESIGN.md 6.3 d).

Everything here is a small grid search over cached validation evidence (``FusionInputs``), so it runs in
seconds on CPU. Targets: accepted words >= 95 % correct for ``balanced`` (98 % ``strict``, 90 % ``lenient``);
"did you mean" only when the true word is in the top 5 for >= 80 % of such cases (fitted per policy). When a
target cannot be reached on the available data, the policies get graded effective targets below the best achievable
bound (``choose_policy_thresholds``, docs 15.17), so strict / balanced / lenient still differ, and the *measured*
numbers - including each policy's achieved lower bound - are stored in ``FusionParams.achieved`` (the UI shows
those, never the target). The classifier alone (sliding mode) gets its own per-policy thresholds the same way
(``fit_classifier_thresholds``). ``ts`` scans are ascending.
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


THRESHOLD_GRID = np.round(np.arange(0.20, 0.991, 0.01), 2)


def choose_policy_thresholds(ts: np.ndarray, n_acc: np.ndarray, n_ok: np.ndarray, targets: dict,
                             min_accepted: int = 5, n_total: int | None = None) -> tuple[dict | None, dict]:
    """Per-policy accept thresholds from a scan: ``n_acc[i]`` accepted, ``n_ok[i]`` of them correct at ``ts[i]``.

    * **target**: some threshold has a 95 % Wilson lower bound (LB) of accepted precision >= the policy's target ->
      the smallest such threshold (most coverage at the promised precision; the rule before docs 15.17).
    * **fallback** (target unreachable on this data): the targets keep their spacing below the best achievable
      bound LB* (ties -> larger threshold). With no target reachable, effective target = LB* - (strictest target -
      policy target), i.e. 0 / 0.03 / 0.08 below LB* for 0.98 / 0.95 / 0.90. When a more lenient target IS
      reachable, the unreachable ones are spread between the strictest reachable target and LB* in the same
      proportions. The policy takes the smallest threshold whose LB reaches its effective target; the strictest
      unreachable one takes the LB* threshold itself. Before 15.17 every unreachable policy took the LB* threshold,
      so all three could be identical (commercial-safe: 0.96 / 0.96 / 0.96) and the policy switch did nothing.

    The thresholds are ordered strict >= balanced >= lenient by construction and differ whenever the scan has
    distinct thresholds at the effective targets. Returns (thresholds or None when no threshold has
    ``min_accepted`` accepts, per-policy report with the bound each policy actually achieves)."""
    ts, n_acc, n_ok = np.asarray(ts, float), np.asarray(n_acc, int), np.asarray(n_ok, int)
    valid = n_acc >= min_accepted
    lbs = np.array([wilson_lower(int(k), int(n)) for k, n in zip(n_ok, n_acc)])
    rep: dict = {"targets": dict(targets), "min_accepted": min_accepted}
    if not valid.any():
        rep["note"] = f"no threshold accepts >= {min_accepted} validation queries: defaults kept"
        rep["policies"] = {pol: {"rule": "default", "target": float(tg), "target_reached": False}
                           for pol, tg in targets.items()}
        return None, rep
    eps = 1e-12
    vi = np.flatnonzero(valid)
    best_lb = float(lbs[vi].max())
    best_i = int(max(i for i in vi if lbs[i] >= best_lb - eps))  # ties -> larger threshold (fewer accepts)
    order = sorted(targets, key=lambda k: -targets[k])  # strictest first
    top = float(targets[order[0]])
    reached = {pol: best_lb >= float(targets[pol]) - eps for pol in order}
    floor = max((float(targets[p]) for p in order if reached[p]), default=None)
    rep.update(best_lower_bound=best_lb, best_threshold=float(ts[best_i]))
    out, pols = {}, {}
    for pol in order:
        tg = float(targets[pol])
        if reached[pol]:
            eff, rule = tg, "target"
        elif floor is None:
            eff, rule = best_lb - (top - tg), "fallback"
        else:
            eff, rule = floor + (best_lb - floor) * (tg - floor) / (top - floor), "fallback"
        if rule == "fallback" and eff >= best_lb - eps:
            i = best_i
        else:
            i = int(next(j for j in vi if lbs[j] >= eff - eps))  # exists: best_i qualifies
        out[pol] = float(ts[i])
        pols[pol] = {"rule": rule, "target": tg, "target_reached": reached[pol], "effective_target": round(eff, 4),
                     "threshold": float(ts[i]), "precision_lower_bound": round(float(lbs[i]), 4),
                     "precision": round(float(n_ok[i] / n_acc[i]), 4), "n_accepted": int(n_acc[i]),
                     **({"coverage": round(float(n_acc[i] / n_total), 4)} if n_total else {})}
    rep["policies"] = pols
    miss = [p for p in order if not reached[p]]
    if miss:
        rep["note"] = (f"target not reachable on validation for {', '.join(miss)}: the best 95 % lower bound of "
                       f"accepted precision is {best_lb:.3f} (threshold {ts[best_i]:.2f}); those policies use "
                       "graded effective targets below it (see precision_lower_bound per policy), not their targets")
    return out, rep


def _scan(p1: np.ndarray, base: np.ndarray, correct: np.ndarray, ts: np.ndarray = THRESHOLD_GRID):
    """Accepted / correct counts of ``(p1 >= t) & base`` for every t."""
    n_acc = np.array([int(((p1 >= t) & base).sum()) for t in ts])
    n_ok = np.array([int(((p1 >= t) & base & correct).sum()) for t in ts])
    return n_acc, n_ok


def _fit_suggest_mass(results: list[FusionResult], truths: list[str], params: FusionParams, policy: str,
                      default: float) -> tuple[float, dict]:
    """Smallest top-5 mass with >= 80 % top-5 hits among the suggestions ``policy`` would show (its own pool: the
    queries it does not accept). Fitting on the balanced pool only gave the lenient policy, whose pool keeps only
    the hard cases, 0.75 top-5 hits on commercial-safe validation (docs 15.17)."""
    dec = _decisions(results, params.with_(suggest_min_top5_mass=0.0, suggest_min_top5_mass_by_policy={}), policy)
    idx = [i for i, (d, _) in enumerate(dec) if d == "suggest"]
    mass = np.array([float(np.sum(results[i].probs[:5])) for i in idx])
    hit = np.array([truths[i] in results[i].keys[:5] for i in idx])
    chosen_m, best_m, best_hit = None, default, -1.0
    for m in np.round(np.arange(0.0, 1.0, 0.05), 2):
        sel = mass >= m
        if sel.sum() < max(3, 0.2 * len(idx)):
            continue
        h = float(hit[sel].mean())
        if h >= SUGGEST_TOP5_TARGET and chosen_m is None:
            chosen_m = float(m)
        if h > best_hit:
            best_m, best_hit = float(m), h
    m = chosen_m if chosen_m is not None else best_m
    sel = mass >= m
    return m, {"mass": m, "target_reached": chosen_m is not None, "pool": len(idx), "shown": int(sel.sum()),
               "top5_hit": round(float(hit[sel].mean()), 4) if sel.any() else None}


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
    base = (p1 - p2 >= params.margin) & (agree >= np.minimum(params.min_agree, crops)) & (cos >= s_min)
    n_acc, n_ok = _scan(p1, base, correct)
    accept, trep = choose_policy_thresholds(THRESHOLD_GRID, n_acc, n_ok, targets, min_accepted, len(results))
    for pol in POLICIES:
        rep[f"{pol}_target_reached"] = bool(trep["policies"][pol]["target_reached"])
    rep["accept_fit"] = trep
    params = params.with_(accept=normalize_accept(accept if accept is not None else params.accept))
    # "did you mean": per policy, on that policy's own suggestion pool; the single value (read by older code and
    # by packs without the per-policy field) is the balanced one, as before
    masses, srep = {}, {}
    for pol in POLICIES:
        masses[pol], srep[pol] = _fit_suggest_mass(results, truths, params, pol, params.suggest_min_top5_mass)
    params = params.with_(suggest_min_top5_mass=masses["balanced"], suggest_min_top5_mass_by_policy=masses)
    rep["suggest_target_reached"] = bool(srep["balanced"]["target_reached"])
    rep["suggest_fit"] = srep
    achieved = {}
    for pol in POLICIES:
        tp = trep["policies"][pol]
        achieved[pol] = {**summarize(results, truths, params, pol), "threshold": params.accept[pol],
                         "target": tp["target"], "target_reached": tp["target_reached"], "rule": tp["rule"],
                         "precision_lower_bound": tp.get("precision_lower_bound"),
                         "suggest_min_top5_mass": masses[pol], "suggest_target_reached": srep[pol]["target_reached"]}
    params = params.with_(achieved=achieved, fitted=True)
    rep["achieved"] = achieved
    return params, rep


def classifier_alone_decisions(per_crop: list[np.ndarray], threshold: float, margin: float,
                               min_agree: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """The engine's classifier-alone rule (sliding mode / classic engine) on one segment per row of ``per_crop``
    ((crops, C) temperature-scaled probabilities): accept when the crop-mean top-1 probability >= ``threshold``,
    beats the runner-up by ``margin`` and enough crops agree. There is no "did you mean" step: every segment that
    is not accepted is "not sure". Returns (mean probabilities (N, C), accepted (N,) bool)."""
    P, p1, base = _classifier_alone_parts(per_crop, margin, min_agree)
    return P, (p1 >= threshold) & base


def _classifier_alone_parts(per_crop: list[np.ndarray], margin: float, min_agree: int = 2):
    P = np.stack([p.mean(0) for p in per_crop])
    srt = -np.sort(-P, axis=1)
    p1, p2 = srt[:, 0], (srt[:, 1] if P.shape[1] > 1 else np.zeros(len(P)))
    top = P.argmax(1)
    agree = np.array([int((p.argmax(1) == t).sum()) for p, t in zip(per_crop, top)])
    need = np.array([min(min_agree, len(p)) for p in per_crop])
    return P, p1, (p1 - p2 >= margin) & (agree >= need)


def fit_classifier_thresholds(per_crop: list[np.ndarray], class_names: list[str], truths: list[str], margin: float,
                              min_agree: int = 2, targets: dict | None = None,
                              min_accepted: int = 5) -> tuple[dict | None, dict]:
    """Per-policy thresholds of the classifier ALONE (sliding mode) on validation: ``per_crop`` are the
    temperature-scaled (crops, C) probabilities the engine sees. Every validation query counts; a query whose word
    is not one of the classifier's classes can only be a wrong accept (in the app nothing stops anyone signing it).
    Same targets and graded fallback as the fused thresholds (``choose_policy_thresholds``)."""
    targets = {pol: float((targets or {}).get(pol, TARGETS[pol])) for pol in POLICIES}
    rep: dict = {"n": len(per_crop), "margin": margin, "min_agree": min_agree}
    if not per_crop:
        return None, {**rep, "note": "no validation data"}
    P, p1, base = _classifier_alone_parts(per_crop, margin, min_agree)
    correct = np.array([class_names[int(i)] == t for i, t in zip(P.argmax(1), truths)])
    rep["n_in_classes"] = int(sum(t in set(class_names) for t in truths))
    n_acc, n_ok = _scan(p1, base, correct)
    accept, trep = choose_policy_thresholds(THRESHOLD_GRID, n_acc, n_ok, targets, min_accepted, len(per_crop))
    return (normalize_accept(accept) if accept is not None else None), {**rep, **trep}
