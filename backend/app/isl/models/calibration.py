"""Fitting the fusion parameters and decision thresholds on validation data (docs/ISL_DESIGN.md 6.3 d).

Everything here is a small grid search over cached validation evidence (``FusionInputs``), so it runs in
seconds on CPU. Targets: accepted words >= 95 % correct for ``balanced`` (98 % ``strict``, 90 % ``lenient``);
"did you mean" only when the true word is in the top 5 for >= 80 % of such cases (fitted per policy, on the worst
major source). Certification (docs 15.25): a policy auto-accepts only when the design-effect-adjusted 95 % Wilson
lower bound of accepted precision of EVERY major source (>= 100 clips, >= 30 accepts, >= 2 signer groups; docs
15.26) is >= its floor
(0.95 / 0.90 / 0.85); otherwise it is confirm-only and its top candidate becomes "did you mean". The *measured*
numbers - each policy's certified worst-source bound and its per-source rows - are stored in
``FusionParams.achieved`` (the UI shows those, never the target). The classifier alone (sliding mode) gets its own
per-policy thresholds the same way (``fit_sliding_thresholds`` on validation clips replayed through the real sliding
mode, docs 15.18; the segment-level ``fit_classifier_thresholds`` is only the fallback and is never certified).
``ts`` scans are ascending.

TODO (docs 15.25): the clips that certify are the validation clips that also selected the models. Out-of-fold
predictions of the cross-validation folds (with their source and signer group) can be added as extra calibration
columns (``certify_scan``) or as a per-source dispersion prior (``phi_prior``) without changing this API.
"""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple

import numpy as np

from app.isl.models.fusion import (POLICIES, USER_PREFIX, FusionInputs, FusionParams, FusionResult, decide, fuse,
                                   normalize_accept)

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


# fit_temperature's default grid, shared with the compact path below (fit-fusion, docs 15.24)
TEMPERATURE_GRID = np.exp(np.linspace(math.log(0.25), math.log(6.0), 49))


def temperature_nll_terms(logits: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """(len(grid), N): per temperature and row, the term ``fit_temperature`` averages (log-sum-exp of logits / T
    minus the true logit / T), with exactly its arithmetic. Rows are independent, so fit-fusion computes them clip
    by clip and keeps 49 numbers per clip instead of its full classifier row (docs 15.24)."""
    grid = grid if grid is not None else TEMPERATURE_GRID
    out = []
    for t in grid:
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        out.append(lse - z[np.arange(len(y)), y])
    return np.stack(out) if out else np.zeros((0, len(y)))


def fit_temperature_terms(terms: np.ndarray, grid: np.ndarray | None = None) -> float:
    """``fit_temperature`` from ``temperature_nll_terms`` (rows = the grid, columns = clips): the same choice."""
    grid = grid if grid is not None else TEMPERATURE_GRID
    terms = np.ascontiguousarray(terms)
    if terms.ndim != 2 or terms.shape[1] == 0:
        return 1.0
    best_t, best = 1.0, float("inf")
    for t, row in zip(grid, terms):
        nll = float(np.mean(row))
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


def weight_grid(has_c: bool, has_r: bool, base: FusionParams) -> dict:
    """The 3 x 3 x 3 x 3 grid of ``fit_fusion_weights`` (fewer points without a classifier or retrieval)."""
    return {
        "alpha": [0.5, 1.0, 1.5] if has_c else [0.0],
        "beta": [0.5, 1.0, 1.5] if has_r else [0.0],
        "gamma": [0.0, 0.5, 1.0],
        "tau_r": [0.02, 0.05, 0.1] if has_r else [base.tau_r],
    }


def fit_fusion_weights(inputs: list[FusionInputs], truths: list[str], base: FusionParams,
                       grid: dict | None = None) -> tuple[FusionParams, dict]:
    has_c = any(i.cls_logp is not None for i in inputs)
    has_r = any(i.ret is not None for i in inputs)
    g = grid or weight_grid(has_c, has_r, base)
    best, best_p, tried, points = float("inf"), base, 0, []
    for a, b, c, t in itertools.product(g["alpha"], g["beta"], g["gamma"], g["tau_r"]):
        p = base.with_(alpha=a, beta=b, gamma=c, tau_r=t)
        nll = fusion_nll(inputs, truths, p)
        tried += 1
        points.append((nll, p))  # in grid order; fit-fusion re-scores the best few on every clip (docs 15.24)
        if nll < best:
            best, best_p = nll, p
    return best_p, {"nll": best, "tried": tried, "grid": g, "points": points}


def is_open_vocabulary(key: str, closed_set) -> bool:
    """True when ``key`` is a word outside the classifier (retrieval only, docs 15.25). ``closed_set`` None = not
    checked (False); a user sign ("mysign:...") is never open vocabulary (its own rule applies)."""
    return closed_set is not None and bool(key) and not key.startswith(USER_PREFIX) and key not in closed_set


def _decisions(results: list[FusionResult], params: FusionParams, policy: str,
               closed_set=None) -> list[tuple[str, str]]:
    return [decide(r.probs, r.agree, len(r.per_crop_top1), params, policy, r.best_cos,
                   open_vocabulary=is_open_vocabulary(r.keys[0] if r.keys else "", closed_set)) for r in results]


def summarize(results: list[FusionResult], truths: list[str], params: FusionParams, policy: str,
              closed_set=None) -> dict:
    """Risk/coverage numbers of the full decision rule (E3). With a pack's (gated) parameters a confirm-only policy
    accepts nothing (docs 15.25); ``closed_set`` (the classifier's classes) also applies the open-vocabulary gate."""
    n = len(results)
    dec = _decisions(results, params, policy, closed_set)
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


# 0.20-0.99 by 0.01, then 0.991-0.999 by 0.001 (docs 15.25): with the 0.01 grid alone strict and balanced often
# ended on the same last point, so strict could not be stricter than balanced
THRESHOLD_GRID = np.unique(np.round(np.r_[np.round(np.arange(0.20, 0.991, 0.01), 2),
                                          np.round(np.arange(0.991, 0.9995, 0.001), 3)], 3))

# ------------------------------------------------------------------ per-source certification (docs 15.25)
# Absolute floors on the CERTIFIED worst-source bound. Below its floor a policy is confirm-only: it never auto-accepts
# and its top candidate becomes "did you mean". These floors are below the 0.98 goal; the app keeps a confirm tap
# even for a certified policy and shows the certified number, not the target.
FLOORS = {"strict": 0.95, "balanced": 0.90, "lenient": 0.85}
CERT_VERSION = "cert-1"
MAJOR_MIN_CLIPS = 100  # a source is "major" (judged) with >= this many calibration clips ...
MAJOR_MIN_ACCEPTS = 30  # ... >= this many accepts at the lowest threshold of the scan ...
MAJOR_MIN_GROUPS = 2  # ... and >= this many signer groups (docs 15.26)
# A source of n clips can never have a bound above wilson_lower(n, n): 0.910 for 39 clips, 0.95 only from 73 clips.
# A smaller source would block strict by construction (Mendeley, 39 clips, did so on research), so it is reported
# (``major: False``, "too small to certify") but neither certifies nor blocks. 100 leaves room for a few errors.
assert wilson_lower(MAJOR_MIN_CLIPS, MAJOR_MIN_CLIPS) >= FLOORS["strict"]
DEFF_MIN_GROUPS = 3  # fewer groups with accepts: the dispersion cannot be estimated -> prior or DEFF_FALLBACK
PHI_MIN_ERRORS = 10  # fewer accepted errors: no phi from that threshold (with 1 error it is ~1 by construction,
#                      with 0 it is undefined); the source's phi comes from the thresholds that have them
DEFF_FALLBACK = 5.0  # ISL500 with 4 signers measured about 5 (Model QA verify-wilson-cluster)
TOO_SMALL_NOTE = (f"too small to certify (fewer than {MAJOR_MIN_CLIPS} clips, {MAJOR_MIN_ACCEPTS} accepts or "
                  f"{MAJOR_MIN_GROUPS} signer groups): reported, not judged")


def is_major_size(n_clips: int, n_groups: int | None, n_accepts: int | None = None) -> bool:
    """The size part of the major-source rule (docs 15.26): enough clips that the source could be certified at the
    strict floor at all, enough signer groups (None = groups unknown, every clip its own group) and, for a scan,
    enough accepts at its lowest threshold."""
    g = n_clips if n_groups is None else n_groups
    return bool(n_clips >= MAJOR_MIN_CLIPS and g >= MAJOR_MIN_GROUPS
                and (n_accepts is None or n_accepts >= MAJOR_MIN_ACCEPTS))


def wilson_lower_deff(k: float, n: float, deff: float = 1.0, z: float = 1.96) -> float:
    """``wilson_lower`` on the effective sample size n / DEFF: clips of one signer are not independent, so a
    design effect DEFF > 1 (errors clustered by signer) widens the interval."""
    d = max(1.0, float(deff))
    return wilson_lower(k / d, n / d, z)


def pearson_dispersion(n_g: np.ndarray, e_g: np.ndarray) -> float | None:
    """Pearson dispersion phi of grouped binomial errors (``e_g`` errors of ``n_g`` accepts per group): about 1 when
    errors are spread like independent coin flips, larger when some signers fail much more often than others.
    None with fewer than ``DEFF_MIN_GROUPS`` groups that have accepts; 1.0 when the error rate is 0 or 1 (nothing
    to compare)."""
    n_g, e_g = np.asarray(n_g, float), np.asarray(e_g, float)
    keep = n_g > 0
    G = int(keep.sum())
    if G < DEFF_MIN_GROUPS:
        return None
    n_g, e_g = n_g[keep], e_g[keep]
    p = float(e_g.sum() / n_g.sum())
    if p <= 0.0 or p >= 1.0:
        return 1.0
    return float(((e_g - n_g * p) ** 2 / (n_g * p * (1 - p))).sum() / (G - 1))


def _labels(n: int, sources, groups):
    src = ["all"] * n if sources is None else [str(x) if x not in (None, "") else "unknown" for x in sources]
    if groups is None:
        return src, None
    keys = [(src[i], str(g)) if g not in (None, "") else (src[i], f"\0clip{i}") for i, g in enumerate(groups)]
    _, gid = np.unique(np.array([f"{a}\0{b}" for a, b in keys], dtype=object), return_inverse=True)
    return src, gid.astype(np.int64)


def scan_by_source(acc: np.ndarray, ok: np.ndarray, sources=None, groups=None, phi_prior: dict | None = None,
                   major=None) -> dict:
    """Per source, per scan row: accepts, correct accepts, the design effect DEFF and the design-effect-adjusted
    Wilson lower bound. ``acc`` / ``ok`` are (thresholds, clips) booleans with ascending thresholds; ``sources`` /
    ``groups`` per clip (None: one source "all" / every clip its own group, DEFF 1 - the clip-level bound used before
    docs 15.25). A source is major when ``is_major_size`` (>= ``MAJOR_MIN_CLIPS`` clips, >= ``MAJOR_MIN_ACCEPTS``
    accepts in row 0, >= ``MAJOR_MIN_GROUPS`` groups), unless ``major`` names the major sources.

    DEFF (docs 15.26) is one number per source: DEFF = max(1, phi_s), where phi_s is the median of the Pearson
    dispersions of the rows with >= ``DEFF_MIN_GROUPS`` groups that have accepts and >= ``PHI_MIN_ERRORS`` accepted
    errors. Accepts and errors only shrink as the threshold grows, so those are the low rows; the clustering they
    measure is applied at the high thresholds too, where the errors run out (estimated row by row, phi fell to 1
    exactly there and certified on that). The median keeps one noisy row from setting it. Without any such row (too
    few groups or errors), ``phi_prior`` (source -> phi, e.g. from cross-validation groups) or the fallback 5.0 -
    never 1."""
    acc, ok = np.atleast_2d(np.asarray(acc, bool)), np.atleast_2d(np.asarray(ok, bool))
    T, N = acc.shape
    src, gid = _labels(N, sources, groups)
    src_a = np.array(src, dtype=object)
    out = {}
    for s in sorted(set(src)):
        m = src_a == s
        a, o = acc[:, m], ok[:, m]
        n_acc, n_ok = a.sum(1).astype(int), o.sum(1).astype(int)
        if gid is None:
            g_inv, G_all = None, None
        else:
            _, g_inv = np.unique(gid[m], return_inverse=True)
            G_all = int(g_inv.max()) + 1 if len(g_inv) else 0
        is_major = (s in set(major)) if major is not None else is_major_size(int(m.sum()), G_all, int(n_acc[0]))
        phi_s, n_rows = None, 0
        if g_inv is None:
            deff, phi_from = np.ones(T), "clips (no groups given)"
        else:
            phis = []
            if G_all >= DEFF_MIN_GROUPS:
                for t in np.flatnonzero(n_acc - n_ok >= PHI_MIN_ERRORS):
                    ng = np.bincount(g_inv, weights=a[t], minlength=G_all)
                    eg = np.bincount(g_inv, weights=a[t] & ~o[t], minlength=G_all)
                    ph = pearson_dispersion(ng, eg)
                    if ph is not None:
                        phis.append(ph)
            n_rows = len(phis)
            if phis:
                phi_s, phi_from = float(np.median(phis)), "validation"
            elif phi_prior and s in phi_prior and phi_prior[s] is not None:
                phi_s, phi_from = float(phi_prior[s]), "prior"
            else:
                phi_s, phi_from = DEFF_FALLBACK, "fallback"
            deff = np.full(T, max(1.0, phi_s))
        lb = np.array([wilson_lower_deff(int(k), int(n_), d) for k, n_, d in zip(n_ok, n_acc, deff)])
        out[s] = {"n": int(m.sum()), "n_groups": G_all, "major": is_major, "n_acc": n_acc, "n_ok": n_ok,
                  "deff": deff, "phi_from": [phi_from] * T, "lb": lb,
                  "phi": None if phi_s is None else round(phi_s, 4), "phi_rows": n_rows,
                  **({} if is_major else {"note": TOO_SMALL_NOTE if major is None else "not a listed major source"})}
    return out


def _source_row(b: dict, i: int) -> dict:
    na, no = int(b["n_acc"][i]), int(b["n_ok"][i])
    return {"precision": round(no / na, 4) if na else None, "lower_bound": round(float(b["lb"][i]), 4),
            "coverage": round(na / b["n"], 4) if b.get("n") else None, "n_accepted": na, "n": b.get("n"),
            "n_groups": b.get("n_groups"), "deff": round(float(b["deff"][i]), 3), "phi_from": b["phi_from"][i],
            "major": bool(b["major"]), **({"note": b["note"]} if b.get("note") else {})}


def choose_policy_thresholds(ts: np.ndarray, n_acc: np.ndarray, n_ok: np.ndarray, targets: dict,
                             min_accepted: int = 5, n_total: int | None = None, *, by_source: dict | None = None,
                             floors: dict | None = None) -> tuple[dict | None, dict]:
    """Per-policy accept thresholds and their certification from a scan (docs 15.25): ``n_acc[i]`` accepted,
    ``n_ok[i]`` of them correct at ``ts[i]`` (pooled), and ``by_source`` (``scan_by_source``) the same per source.

    * The certified bound at a threshold is the WORST major source's design-effect-adjusted 95 % Wilson lower bound
      (LB). A pooled bound hides a weak source: pooled 93 % ISL500 data from 2 signers said strict, INCLUDE was 0.92.
    * Each policy aims at ``min(target, best worst-source LB)`` and takes the smallest threshold whose worst-source
      LB reaches it (most coverage at that bound), so strict >= balanced >= lenient.
    * It is ``certified`` when that bound is >= the policy's floor (``FLOORS``: 0.95 / 0.90 / 0.85), else
      ``confirm_only``: the app never auto-accepts with it, and its top candidate becomes "did you mean". This
      replaces the graded fallback of docs 15.17. With no major source nothing is certified.

    ``by_source`` None = one pooled source with the clip-level bound (callers without sources or groups). Returns
    (thresholds or None when no threshold has ``min_accepted`` accepts, report)."""
    ts, n_acc, n_ok = np.asarray(ts, float), np.asarray(n_acc, int), np.asarray(n_ok, int)
    floors = {pol: float((floors or FLOORS).get(pol, FLOORS.get(pol, 0.0))) for pol in targets}
    pooled = np.array([wilson_lower(int(k), int(n)) for k, n in zip(n_ok, n_acc)])
    if by_source is None:
        by_source = {"all": {"n": n_total, "n_groups": None, "major": True, "n_acc": n_acc, "n_ok": n_ok,
                             "deff": np.ones(len(ts)), "phi_from": ["clips (no groups given)"] * len(ts), "lb": pooled}}
    majors = sorted(s for s, b in by_source.items() if b["major"])
    if majors:
        L = np.stack([by_source[s]["lb"] for s in majors])
        worst, worst_src = L.min(0), [majors[int(j)] for j in L.argmin(0)]
    else:
        worst, worst_src = np.zeros(len(ts)), [None] * len(ts)
    valid = n_acc >= min_accepted
    rep: dict = {"targets": dict(targets), "floors": floors, "min_accepted": min_accepted, "major_sources": majors,
                 "certification": "worst major source, design-effect-adjusted 95 % Wilson lower bound",
                 "sources": {s: {"n": b.get("n"), "n_groups": b.get("n_groups"), "major": bool(b["major"]),
                                 **({"note": b["note"]} if b.get("note") else {})}
                             for s, b in by_source.items()}}
    if not valid.any():
        rep["note"] = f"no threshold accepts >= {min_accepted} validation queries: defaults kept, nothing certified"
        rep["policies"] = {pol: {"rule": "confirm_only", "target": float(tg), "floor": floors[pol],
                                 "target_reached": False, "certified_precision": None,
                                 "reason": "too little validation data"} for pol, tg in targets.items()}
        return None, rep
    eps = 1e-12
    vi = np.flatnonzero(valid)
    rank = worst if majors else pooled
    best = float(rank[vi].max())
    best_worst = float(worst[vi].max())
    rep.update(best_lower_bound=round(best_worst, 4),
               best_threshold=float(ts[next(j for j in vi if worst[j] >= best_worst - eps)]),
               best_pooled_lower_bound=round(float(pooled[vi].max()), 4))
    out, pols = {}, {}
    for pol in sorted(targets, key=lambda k: -targets[k]):  # strictest first
        tg, fl = float(targets[pol]), floors[pol]
        level = min(tg, best)
        i = int(next(j for j in vi if rank[j] >= level - eps))
        certified = bool(majors) and bool(worst[i] >= fl - eps)
        out[pol] = float(ts[i])
        pols[pol] = {"rule": "certified" if certified else "confirm_only", "target": tg, "floor": fl,
                     "target_reached": bool(majors) and bool(worst[i] >= tg - eps),
                     "effective_target": round(level, 4),
                     "threshold": float(ts[i]), "precision_lower_bound": round(float(rank[i]), 4),
                     "certified_precision": round(float(worst[i]), 4) if certified else None,
                     "worst_source": worst_src[i], "pooled_lower_bound": round(float(pooled[i]), 4),
                     "precision": round(float(n_ok[i] / n_acc[i]), 4), "n_accepted": int(n_acc[i]),
                     **({"coverage": round(float(n_acc[i] / n_total), 4)} if n_total else {}),
                     "by_source": {s: _source_row(b, i) for s, b in sorted(by_source.items())}}
        if not certified:
            pols[pol]["reason"] = ("no major source (>= %d clips, >= %d accepts and >= %d signer groups)"
                                   % (MAJOR_MIN_CLIPS, MAJOR_MIN_ACCEPTS, MAJOR_MIN_GROUPS) if not majors else
                                   f"worst-source lower bound {worst[i]:.3f} ({worst_src[i]}) is below the {pol} "
                                   f"floor {fl:.2f}")
    rep["policies"] = pols
    miss = [p for p in pols if pols[p]["rule"] != "certified"]
    if miss:
        rep["note"] = (f"confirm-only (never auto-accept): {', '.join(miss)}; the best worst-source 95 % lower bound "
                       f"of accepted precision is {best_worst:.3f} (pooled {float(pooled[vi].max()):.3f})")
    return out, rep


def certify_scan(ts: np.ndarray, acc: np.ndarray, ok: np.ndarray, targets: dict, min_accepted: int = 5,
                 n_total: int | None = None, *, sources=None, groups=None, phi_prior: dict | None = None,
                 floors: dict | None = None, major_sources=None) -> tuple[dict | None, dict, dict | None]:
    """``choose_policy_thresholds`` from per-clip (thresholds, clips) accept / correct matrices. Calibration clips
    are just columns with a source and a group, so extra calibration data (e.g. out-of-fold predictions of the
    cross-validation folds, docs 15.25 TODO) can be added by appending columns. ``major_sources``: an explicit list
    (e.g. from a profile) instead of the size rule. Returns (thresholds, report, the per-source scan or None)."""
    acc, ok = np.atleast_2d(np.asarray(acc, bool)), np.atleast_2d(np.asarray(ok, bool))
    by = scan_by_source(acc, ok, sources, groups, phi_prior, major=major_sources) \
        if (sources is not None or groups is not None) else None
    accept, rep = choose_policy_thresholds(ts, acc.sum(1), ok.sum(1), targets, min_accepted, n_total, by_source=by,
                                           floors=floors)
    return accept, rep, by


def _scan_matrix(p1: np.ndarray, base: np.ndarray, correct: np.ndarray, ts: np.ndarray = THRESHOLD_GRID):
    """(thresholds, clips) booleans: accepted ``(p1 >= t) & base`` and accepted-and-correct, for every t."""
    acc = (np.asarray(p1)[None, :] >= np.asarray(ts)[:, None]) & np.asarray(base, bool)[None, :]
    return acc, acc & np.asarray(correct, bool)[None, :]


def _scan(p1: np.ndarray, base: np.ndarray, correct: np.ndarray, ts: np.ndarray = THRESHOLD_GRID):
    """Accepted / correct counts of ``(p1 >= t) & base`` for every t."""
    acc, ok = _scan_matrix(p1, base, correct, ts)
    return acc.sum(1).astype(int), ok.sum(1).astype(int)


def _fit_s_min(cos: np.ndarray, sources=None, groups=None, q: float = 0.05,
               major_sources=None) -> tuple[float, dict]:
    """Out-of-vocabulary guard (docs 15.25): the minimum over major sources of each source's 5 % quantile of the best
    cosine, so >= 95 % of every major source's in-vocabulary queries pass. Pooled, a large source set the guard and a
    smaller one lost 20.8 % of its in-vocabulary queries (INCLUDE on research). Major: ``is_major_size`` on the
    queries with a cosine (docs 15.26: a source too small to certify does not set the guard either), or the listed
    ``major_sources``. Without sources, or without a major source, the pooled quantile."""
    have = cos > -1
    if not have.any():
        return -1.0, {"rule": "no retrieval: guard off"}
    pooled = float(np.quantile(cos[have], q))
    rep: dict = {"pooled_q05": round(pooled, 4)}
    if sources is None:
        rep["rule"] = "pooled 5 % quantile (no sources given)"
        return pooled, rep
    src = np.array([str(x) if x not in (None, "") else "unknown" for x in sources], dtype=object)
    grp = (np.array([str(g) if g not in (None, "") else f"\0clip{i}" for i, g in enumerate(groups)], dtype=object)
           if groups is not None else np.array([f"\0clip{i}" for i in range(len(cos))], dtype=object))
    per, qs = {}, {}
    for s in sorted(set(src)):
        m = (src == s) & have
        ng = len(set(grp[m]))
        per[s] = {"n": int(m.sum()), "n_groups": ng,
                  "q05": round(float(np.quantile(cos[m], q)), 4) if m.any() else None}
        if m.any() and (s in set(major_sources) if major_sources is not None else is_major_size(int(m.sum()), ng)):
            qs[s] = float(np.quantile(cos[m], q))
            per[s]["major"] = True
        elif major_sources is None:
            per[s]["note"] = TOO_SMALL_NOTE
    s_min = min(qs.values()) if qs else pooled
    for s, b in per.items():
        m = (src == s) & have
        b["pass_rate"] = round(float((cos[m] >= s_min).mean()), 4) if m.any() else None
    rep.update(rule="minimum over major sources of each source's 5 % quantile" if qs else
               "pooled 5 % quantile (no major source)", by_source=per, s_min=round(s_min, 4))
    return s_min, rep


def _fit_suggest_mass(results: list[FusionResult], truths: list[str], params: FusionParams, policy: str,
                      default: float, sources=None, major=None) -> tuple[float, dict]:
    """Smallest top-5 mass with >= 80 % top-5 hits among the suggestions ``policy`` would show (its own pool: the
    queries its threshold does not accept). Fitting on the balanced pool only gave the lenient policy, whose pool
    keeps only the hard cases, 0.75 top-5 hits on commercial-safe validation (docs 15.17). With ``sources`` and
    ``major`` (the major sources), the hit rate that must reach 0.80 is the WORST major source's (docs 15.25), and
    no mass may cut a major source's share of the pool below 20 %; sources with fewer than 3 suggestions in the pool
    are not judged."""
    ung = params.with_(suggest_min_top5_mass=0.0, suggest_min_top5_mass_by_policy={}, certification=None)
    dec = _decisions(results, ung, policy)
    idx = [i for i, (d, _) in enumerate(dec) if d == "suggest"]
    mass = np.array([float(np.sum(results[i].probs[:5])) for i in idx])
    hit = np.array([truths[i] in results[i].keys[:5] for i in idx], bool)
    majors = sorted(major) if (sources is not None and major) else None
    src = np.array([str(sources[i]) if sources[i] not in (None, "") else "unknown" for i in idx], dtype=object) \
        if majors is not None else None
    judged = [s for s in majors if (src == s).sum() >= 3] if majors is not None else None

    def hit_at(sel):
        if judged is None or not judged:
            if sel.sum() < max(3, 0.2 * len(idx)):
                return None, None
            return float(hit[sel].mean()), None
        hs = {}
        for s_ in judged:
            ms = src == s_
            k = int((sel & ms).sum())
            if k < max(3, 0.2 * int(ms.sum())):
                return None, None
            hs[s_] = float(hit[sel & ms].mean())
        w = min(hs, key=lambda x: (hs[x], x))
        return hs[w], w

    chosen_m, best_m, best_hit = None, default, -1.0
    for m in np.round(np.arange(0.0, 1.0, 0.05), 2):
        h, _ = hit_at(mass >= m)
        if h is None:
            continue
        if h >= SUGGEST_TOP5_TARGET and chosen_m is None:
            chosen_m = float(m)
        if h > best_hit:
            best_m, best_hit = float(m), h
    m = chosen_m if chosen_m is not None else best_m
    sel = mass >= m
    out = {"mass": m, "target_reached": chosen_m is not None, "pool": len(idx), "shown": int(sel.sum()),
           "top5_hit": round(float(hit[sel].mean()), 4) if sel.any() else None}
    if judged:
        by = {}
        for s_ in sorted(set(src)):
            ms = src == s_
            by[s_] = {"pool": int(ms.sum()), "shown": int((sel & ms).sum()),
                      "top5_hit": round(float(hit[sel & ms].mean()), 4) if (sel & ms).any() else None}
        w = min((s_ for s_ in judged if by[s_]["top5_hit"] is not None), key=lambda x: (by[x]["top5_hit"], x),
                default=None)
        out.update(rule="worst major source's top-5 hit rate >= 0.80", worst_source=w,
                   worst_top5_hit=by[w]["top5_hit"] if w else None, by_source=by)
    else:
        out["rule"] = "pooled top-5 hit rate >= 0.80"
    return m, out


def _open_vocabulary_check(results, acc_row: np.ndarray, ok_row: np.ndarray, closed_set, sources, groups,
                           majors: list, phi_prior, policy_certified: bool) -> dict:
    """Words outside the classifier (retrieval only) at one policy's threshold (docs 15.25): certified only when the
    policy is certified and EVERY major source's design-effect-adjusted lower bound on the open-vocabulary accepts is
    >= the strict floor. Dictionary-only accepts were mostly wrong on research (28 of 31, hub words)."""
    open_m = np.array([is_open_vocabulary(r.keys[0] if r.keys else "", closed_set) for r in results], bool)
    a, o = acc_row & open_m, ok_row & open_m
    floor = FLOORS["strict"]
    by = scan_by_source(a[None], o[None], sources, groups, phi_prior, major=majors) \
        if (sources is not None or groups is not None) else scan_by_source(a[None], o[None], major=majors or ["all"])
    rows = {s: _source_row(b, 0) for s, b in by.items()}
    mj = [s for s in by if by[s]["major"]]
    worst = min((float(by[s]["lb"][0]) for s in mj), default=0.0)
    certified = bool(policy_certified and mj and worst >= floor - 1e-12)
    n_a = int(a.sum())
    return {"rule": "certified" if certified else "confirm_only", "floor": floor, "n_accepted": n_a,
            "precision": round(float(o.sum() / n_a), 4) if n_a else None,
            "worst_lower_bound": round(worst, 4) if mj else None,
            "worst_source": min(mj, key=lambda s: (float(by[s]["lb"][0]), s)) if mj else None, "by_source": rows,
            **({} if certified else {"reason": "policy not certified" if not policy_certified else
                                     f"open-vocabulary worst-source lower bound {worst:.3f} < strict floor {floor}"})}


def fit_thresholds(results: list[FusionResult], truths: list[str], params: FusionParams,
                   targets: dict | None = None, min_accepted: int = 5, *, sources=None, groups=None,
                   closed_set=None, phi_prior: dict | None = None,
                   major_sources=None) -> tuple[FusionParams, dict]:
    """Accept thresholds, their certification, the out-of-vocabulary guard and the "did you mean" masses on
    validation (docs 15.25). ``sources`` / ``groups``: per result, its source and signer group (fit-fusion passes
    them; without them one pooled source with the clip-level bound). ``closed_set``: the classifier's classes, for the
    open-vocabulary check (None = not checked: open-vocabulary words stay confirm-only). ``phi_prior``: per-source
    dispersion for sources with too few groups. ``major_sources``: an explicit list instead of the size rule
    (docs 15.26). The returned parameters are a pack's record (gated): uncertified
    policies are confirm-only. ``achieved`` holds what each policy's THRESHOLD does on validation (the top candidate
    the user confirms), with ``rule``, the certified worst-source precision and ``by_source``."""
    # every policy always gets a threshold (a missing one made decide() raise): unknown targets are ignored and
    # missing ones use the default target
    targets = {pol: float((targets or {}).get(pol, TARGETS[pol])) for pol in POLICIES}
    rep: dict = {"n": len(results)}
    if not results:
        return params.with_(certification={}), {**rep, "note": "no validation data: defaults kept (not calibrated, "
                                                              "every policy confirm-only)"}
    n = len(results)
    src_l = list(sources) if sources is not None else None
    grp_l = list(groups) if groups is not None else None
    if (src_l is not None and len(src_l) != n) or (grp_l is not None and len(grp_l) != n):
        raise ValueError("sources / groups must have one entry per validation result")
    # out-of-vocabulary guard: >= 95 % of every major source's in-vocabulary queries pass
    cos = np.array([r.best_cos for r in results])
    s_min, s_rep = _fit_s_min(cos, src_l, grp_l, major_sources=major_sources)
    rep["s_min_fit"] = s_rep
    params = params.with_(s_min=s_min)
    p1 = np.array([r.probs[0] if len(r.probs) else 0.0 for r in results])
    p2 = np.array([r.probs[1] if len(r.probs) > 1 else 0.0 for r in results])
    agree = np.array([r.agree for r in results])
    crops = np.array([len(r.per_crop_top1) for r in results])
    correct = np.array([bool(r.keys) and r.keys[0] == t for r, t in zip(results, truths)])
    base = (p1 - p2 >= params.margin) & (agree >= np.minimum(params.min_agree, crops)) & (cos >= s_min)
    acc_m, ok_m = _scan_matrix(p1, base, correct)
    accept, trep, by_scan = certify_scan(THRESHOLD_GRID, acc_m, ok_m, targets, min_accepted, n, sources=src_l,
                                         groups=grp_l, phi_prior=phi_prior, major_sources=major_sources)
    for pol in POLICIES:
        rep[f"{pol}_target_reached"] = bool(trep["policies"][pol]["target_reached"])
    rep["accept_fit"] = trep
    params = params.with_(accept=normalize_accept(accept if accept is not None else params.accept))
    ungated = params.with_(certification=None)  # the threshold rule itself (what the top candidate does)
    majors = trep.get("major_sources") or []
    # "did you mean": per policy, on that policy's own suggestion pool, on the worst major source; the single value
    # (read by older code and by packs without the per-policy field) is the balanced one, as before
    masses, srep = {}, {}
    for pol in POLICIES:
        masses[pol], srep[pol] = _fit_suggest_mass(results, truths, ungated, pol, params.suggest_min_top5_mass,
                                                   sources=src_l, major=majors if src_l is not None else None)
    params = params.with_(suggest_min_top5_mass=masses["balanced"], suggest_min_top5_mass_by_policy=masses)
    ungated = params.with_(certification=None)
    rep["suggest_target_reached"] = bool(srep["balanced"]["target_reached"])
    rep["suggest_fit"] = srep
    rules = {pol: trep["policies"][pol]["rule"] for pol in POLICIES}
    ov: dict = {"checked": closed_set is not None, "floor": FLOORS["strict"], "policies": {}}
    if closed_set is not None:
        for pol in POLICIES:
            i = int(np.flatnonzero(np.isclose(THRESHOLD_GRID, params.accept[pol]))[0]) \
                if np.isclose(THRESHOLD_GRID, params.accept[pol]).any() else None
            if i is None:  # a default threshold off the grid: nothing to certify
                ov["policies"][pol] = {"rule": "confirm_only", "reason": "threshold not fitted"}
                continue
            # the open-vocabulary subset rarely has enough errors for its own phi: each source's DEFF at this
            # threshold from the full scan is its prior (never 1 for want of errors, docs 15.26)
            ov_prior = ({s_: float(b["deff"][i]) for s_, b in by_scan.items()} if by_scan is not None else phi_prior)
            ov["policies"][pol] = _open_vocabulary_check(results, acc_m[i], ok_m[i], closed_set, src_l, grp_l,
                                                         majors, ov_prior, rules[pol] == "certified")
    else:
        ov["note"] = "no classifier class list given: words outside the classifier stay confirm-only"
    rep["open_vocabulary"] = ov
    achieved = {}
    for pol in POLICIES:
        tp = trep["policies"][pol]
        achieved[pol] = {**summarize(results, truths, ungated, pol), "threshold": params.accept[pol],
                         "target": tp["target"], "floor": tp.get("floor"), "target_reached": tp["target_reached"],
                         "rule": tp["rule"], "auto_accept": tp["rule"] == "certified",
                         "precision_lower_bound": tp.get("precision_lower_bound"),
                         "certified_precision": tp.get("certified_precision"), "worst_source": tp.get("worst_source"),
                         "pooled_lower_bound": tp.get("pooled_lower_bound"),
                         "by_source": tp.get("by_source") or {},
                         **({"reason": tp["reason"]} if tp.get("reason") else {}),
                         "open_vocabulary_rule": (ov["policies"].get(pol) or {}).get("rule", "confirm_only"),
                         "suggest_min_top5_mass": masses[pol], "suggest_target_reached": srep[pol]["target_reached"]}
    cert = {"version": CERT_VERSION,
            "rule": "a policy auto-accepts only when the design-effect-adjusted 95 % Wilson lower bound of accepted "
                    "precision of EVERY major source is >= its floor; otherwise it is confirm-only (docs 15.25)",
            "targets": targets, "floors": {pol: FLOORS[pol] for pol in POLICIES},
            "major_sources": majors,
            "major_rule": ("listed: " + ", ".join(sorted(major_sources))) if major_sources is not None else
                          f">= {MAJOR_MIN_CLIPS} clips, >= {MAJOR_MIN_ACCEPTS} accepts at the lowest threshold and >= "
                          f"{MAJOR_MIN_GROUPS} signer groups; smaller sources are reported, not judged",
            "deff": f"one per source: max(1, the median Pearson dispersion of accepted errors over the source's "
                    f"groups, over the thresholds with >= {DEFF_MIN_GROUPS} groups and >= {PHI_MIN_ERRORS} accepted "
                    f"errors); no such threshold -> the prior or {DEFF_FALLBACK}",
            "groups_known": grp_l is not None, "sources_known": src_l is not None, "calibration_clips": n,
            "policies": rules, "open_vocabulary": ov,
            "selection_note": "certified on the same validation clips that selected the models (docs 15.25 TODO: "
                              "out-of-fold cross-validation calibration)"}
    params = params.with_(achieved=achieved, fitted=True, certification=cert)
    rep["achieved"] = achieved
    rep["certification"] = {k: v for k, v in cert.items() if k != "open_vocabulary"}
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


class ClassifierSummary(NamedTuple):
    """What the classifier-alone segment rule reads from one segment's (crops, C) probabilities
    (``classifier_summary``): crop-mean top-1 and top-2, the crop-mean top class and each crop's top class. Fit-fusion
    keeps this per clip instead of the full rows (docs 15.24)."""
    p1: np.floating
    p2: np.floating
    top: int
    crop_top: np.ndarray


def classifier_summary(per_crop: np.ndarray) -> ClassifierSummary:
    P = per_crop.mean(0)
    srt = -np.sort(-P)
    p2 = srt[1] if len(P) > 1 else np.float64(0.0)
    return ClassifierSummary(srt[0], p2, int(P.argmax()), per_crop.argmax(1).astype(np.int32))


def _summary_parts(rows: list[ClassifierSummary], margin: float, min_agree: int = 2):
    """``_classifier_alone_parts`` from summaries: the same top classes, p1 and accept mask."""
    p1 = np.array([r.p1 for r in rows])
    p2 = np.array([r.p2 for r in rows])
    top = np.array([r.top for r in rows])
    agree = np.array([int((r.crop_top == r.top).sum()) for r in rows])
    need = np.array([min(min_agree, len(r.crop_top)) for r in rows])
    return top, p1, (p1 - p2 >= margin) & (agree >= need)


def fit_classifier_thresholds(per_crop: list, class_names: list[str], truths: list[str], margin: float,
                              min_agree: int = 2, targets: dict | None = None,
                              min_accepted: int = 5, *, sources=None, groups=None,
                              phi_prior: dict | None = None, major_sources=None) -> tuple[dict | None, dict]:
    """Per-policy thresholds of the classifier ALONE (sliding mode) on validation: ``per_crop`` are the
    temperature-scaled (crops, C) probabilities the engine sees, or their ``classifier_summary`` (the same result).
    Every validation query counts; a query whose word is not one of the classifier's classes can only be a wrong
    accept (in the app nothing stops anyone signing it). Same per-source certification as the fused thresholds
    (``choose_policy_thresholds``, docs 15.25); fit-fusion never certifies this segment proxy for the app."""
    targets = {pol: float((targets or {}).get(pol, TARGETS[pol])) for pol in POLICIES}
    rep: dict = {"n": len(per_crop), "margin": margin, "min_agree": min_agree}
    if not per_crop:
        return None, {**rep, "note": "no validation data"}
    if isinstance(per_crop[0], ClassifierSummary):
        top, p1, base = _summary_parts(per_crop, margin, min_agree)
    else:
        P, p1, base = _classifier_alone_parts(per_crop, margin, min_agree)
        top = P.argmax(1)
    correct = np.array([class_names[int(i)] == t for i, t in zip(top, truths)])
    rep["n_in_classes"] = int(sum(t in set(class_names) for t in truths))
    acc, ok = _scan_matrix(p1, base, correct)
    accept, trep, _ = certify_scan(THRESHOLD_GRID, acc, ok, targets, min_accepted, len(per_crop), sources=sources,
                                   groups=groups, phi_prior=phi_prior, major_sources=major_sources)
    return (normalize_accept(accept) if accept is not None else None), {**rep, **trep}


# ------------------------------------------------------------------ classifier alone, real sliding-mode rule
# A sliding-mode window as recorded by ``fit_fusion.sliding_windows``: (t ms, EMA probabilities, streak, sign
# boundary before this window[, the engine's class key]). Recorded once with a never-accepting threshold; the commits
# for any threshold then follow from the engine's rule below, because EMA and streak do not depend on what was
# committed. The key is what the engine compares for repeat suppression (its vocabulary index of the top class);
# without it the classifier index is used.
SlidingWindow = tuple  # (float, np.ndarray, int, bool[, int])


class PreparedWindows(list):
    """One clip's sliding windows already reduced to what the commit rule reads (``prepare_windows``): per window
    (t, top-1, top-1 minus top-2, top class, streak, boundary, key) instead of the full EMA vector. A replayed
    validation clip then keeps about 1 KB instead of ~10 windows x 35 KB of EMA probabilities (docs 15.23). Every
    function here accepts either form and gives the same result."""


def prepare_windows(windows: list[SlidingWindow]) -> PreparedWindows:
    return windows if isinstance(windows, PreparedWindows) else PreparedWindows(_sliding_prep(windows))


def _sliding_prep(windows: list[SlidingWindow]) -> list[tuple[float, float, float, int, int, bool, int]]:
    if isinstance(windows, PreparedWindows):
        return windows
    out = []
    for w in windows:
        t, ema, streak, boundary = w[:4]
        ema = np.asarray(ema, float)
        srt = np.sort(ema)[::-1]
        cls = int(ema.argmax())
        out.append((float(t), float(srt[0]), float(srt[0] - srt[1]) if len(srt) > 1 else float("inf"),
                    cls, int(streak), bool(boundary), int(w[4]) if len(w) > 4 and w[4] is not None else cls))
    return out


def _sliding_run(prep, threshold: float, margin: float, agree_n: int, cooldown_ms: float) -> list[int]:
    last_key, last_t, out = None, -1e18, []
    for t, p1, gap, cls, streak, boundary, key in prep:
        if boundary:
            last_key = None
        if p1 < threshold or gap < margin or streak < agree_n:
            continue
        if key == last_key or t - last_t < cooldown_ms:
            continue
        last_key, last_t = key, t
        out.append(cls)
    return out


def sliding_commits(windows: list[SlidingWindow], threshold: float, margin: float, agree_n: int,
                    cooldown_ms: float) -> list[int]:
    """Classes the app's sliding mode commits for one recorded clip at ``threshold``: exactly
    ``RecognitionEngine._push_sliding`` / ``_decision`` - EMA top-1 >= threshold, top-1 minus top-2 >= margin, the
    same top class for >= ``agree_n`` windows in a row, no repeat of the last commit until a sign boundary (hands
    gone), and ``cooldown_ms`` between commits (docs 15.18)."""
    return _sliding_run(_sliding_prep(windows), threshold, margin, agree_n, cooldown_ms)


def sliding_commit_matrix(windows: list[list[SlidingWindow]], truths: list[int], ts: np.ndarray, margin: float,
                          agree_n: int, cooldown_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per threshold in ``ts`` and recorded clip (``truths[i]`` = class index of clip i's word, -1 when it is not a
    classifier class): (thresholds, clips) booleans "committed at least once", "committed and every commit is the
    clip's word", and the number of commits. Per clip, so the fit can certify per source and signer group (docs
    15.25)."""
    preps = [_sliding_prep(w) for w in windows]
    ts = np.asarray(ts, float)
    acc = np.zeros((len(ts), len(preps)), bool)
    ok = np.zeros((len(ts), len(preps)), bool)
    com = np.zeros((len(ts), len(preps)), np.int32)
    for j, (prep, y) in enumerate(zip(preps, truths)):
        for i, t in enumerate(ts):
            cs = _sliding_run(prep, float(t), margin, agree_n, cooldown_ms)
            com[i, j] = len(cs)
            if cs:
                acc[i, j] = True
                ok[i, j] = bool(y >= 0 and all(c == y for c in cs))
    return acc, ok, com


def sliding_commit_counts(windows: list[list[SlidingWindow]], truths: list[int], ts: np.ndarray, margin: float,
                          agree_n: int, cooldown_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per threshold in ``ts``, over recorded clips (``truths[i]`` = class index of clip i's word, -1 when it is not a
    classifier class): (clips with >= 1 commit, clips with >= 1 commit that are all the clip's word, commits).
    The unit is the clip (one signer performing one sign), not the window: windows of one clip are not independent."""
    acc, ok, com = sliding_commit_matrix(windows, truths, ts, margin, agree_n, cooldown_ms)
    return acc.sum(1).astype(int), ok.sum(1).astype(int), com.sum(1).astype(int)


def fit_sliding_thresholds(windows: list[list[SlidingWindow]], truths: list[int], margin: float, agree_n: int,
                           cooldown_ms: float, targets: dict | None = None,
                           min_accepted: int = 5, *, sources=None, groups=None,
                           phi_prior: dict | None = None, major_sources=None) -> tuple[dict | None, dict]:
    """Per-policy sliding-mode thresholds of the classifier alone from validation clips replayed through the app's
    sliding mode (docs 15.18). A clip counts as accepted when sliding mode commits anything during it, and as correct
    when every commit is the clip's word; a clip whose word is not a classifier class can only be wrong. Same
    per-source certification as the fused thresholds (``choose_policy_thresholds``, docs 15.25): ``sources`` /
    ``groups`` per replayed clip; a policy whose worst major source misses its floor is confirm-only, and the app's
    sliding mode then never auto-commits with it. ``precision_lower_bound`` is a bound on what sliding mode does on
    validation, not on the segment-level proxy of ``fit_classifier_thresholds``."""
    targets = {pol: float((targets or {}).get(pol, TARGETS[pol])) for pol in POLICIES}
    rep: dict = {"n": len(windows), "n_in_classes": int(sum(y >= 0 for y in truths)), "margin": margin,
                 "agree_n": agree_n, "cooldown_ms": cooldown_ms,
                 "unit": "clip: accepted = sliding mode commits at least once; correct = every commit is its word"}
    if not windows:
        return None, {**rep, "note": "no replayed validation clips"}
    acc, ok, com = sliding_commit_matrix(windows, truths, THRESHOLD_GRID, margin, agree_n, cooldown_ms)
    accept, trep, _ = certify_scan(THRESHOLD_GRID, acc, ok, targets, min_accepted, len(windows), sources=sources,
                                   groups=groups, phi_prior=phi_prior, major_sources=major_sources)
    if accept is not None:
        n_com = com.sum(1)
        for pol, b in trep["policies"].items():
            i = int(np.flatnonzero(np.isclose(THRESHOLD_GRID, b["threshold"]))[0])
            b["commits_per_clip"] = round(float(n_com[i] / len(windows)), 4)
    return (normalize_accept(accept) if accept is not None else None), {**rep, **trep}
