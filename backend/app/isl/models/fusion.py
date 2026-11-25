"""Calibrated fusion of classifier, retrieval and "My signs" scores, with the accept / suggest / abstain rule
(docs/ISL_DESIGN.md section 6.3 d and 7.3).

For one segment and one crop, over the union of candidate concepts::

    z_k = alpha * log p_c(k)            (classifier, temperature t_c; concepts outside it get noncore_logp)
        + beta  * log p_r(k)            (p_r = softmax over every index concept and user sign of cos / tau_r,
                                         user signs get + user_bonus in logit units)
        + gamma * prior_k               (1 for core words, 0.5 single words, 0 multiword/phrases)
    p_f = softmax(z)

Crops are fused separately (agreement = crops whose top-1 equals the mean's top-1) and the probabilities averaged.
All parameters are fitted on validation data (``calibration.py``) and stored in the pack's ``fusion.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

POLICIES = ("strict", "balanced", "lenient")
USER_PREFIX = "mysign:"


@dataclass(frozen=True)
class FusionParams:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    tau_r: float = 0.05
    t_c: float = 1.0
    s_min: float = -1.0  # out-of-vocabulary guard on the best cosine; -1 disables
    accept: dict = field(default_factory=lambda: {"strict": 0.85, "balanced": 0.7, "lenient": 0.55})
    margin: float = 0.10
    min_agree: int = 2
    suggest_min_top5_mass: float = 0.25
    user_bonus: float = 1.0  # logit bonus of user signs in the retrieval softmax
    user_margin: float = 0.05  # cosine margin for the direct "My signs" override (7.3)
    user_theta_floor: float = 0.0  # lowest allowed per-sign acceptance cosine
    noncore_logp: float = math.log(1e-3)
    top_n: int = 50
    version: str = "fusion-1"
    profile: str = ""
    fitted: bool = False  # False = defaults (not calibrated): the UI must say so
    achieved: dict = field(default_factory=dict)  # measured on validation: per policy precision / coverage / n
    enrol: dict = field(default_factory=lambda: {"c_min": 0.5, "min_takes": 3, "max_takes": 5})

    def __post_init__(self):
        # Every policy must always have a threshold: a fusion.json written with only some policies (or none) made
        # decide() raise for the missing ones (HTTP 500 in the API). Missing values are filled from the present
        # ones (or the defaults) and kept ordered strict >= balanced >= lenient.
        object.__setattr__(self, "accept", normalize_accept(self.accept))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict | None) -> FusionParams:
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> FusionParams:
        return cls.from_json(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    def with_(self, **kw) -> FusionParams:
        return replace(self, **kw)


DEFAULT_ACCEPT = {"strict": 0.85, "balanced": 0.7, "lenient": 0.55}


def normalize_accept(accept: dict | None) -> dict:
    """All three policies, strict >= balanced >= lenient. Missing policies take the nearest stricter / more
    lenient value that is present, else the default; unknown keys are dropped."""
    a = {k: float(v) for k, v in dict(accept or {}).items() if k in POLICIES and v is not None}
    if not a:
        return dict(DEFAULT_ACCEPT)
    out = {}
    for pol in POLICIES:
        if pol in a:
            out[pol] = a[pol]
        elif pol == "balanced":
            out[pol] = a.get("strict", a.get("lenient"))
        elif pol == "strict":
            out[pol] = a.get("balanced", a.get("lenient"))
        else:
            out[pol] = a.get("balanced", a.get("strict"))
    out["balanced"] = max(out["balanced"], out["lenient"])
    out["strict"] = max(out["strict"], out["balanced"])
    return out


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


@dataclass
class FusionInputs:
    """Per-crop evidence for one segment. Any part may be missing (None / empty)."""

    cls_logp: np.ndarray | None = None  # (crops, C) log-probabilities at temperature 1
    cls_ids: list[str] = field(default_factory=list)
    ret: np.ndarray | None = None  # (crops, G) max cosine per index concept
    ret_ids: list[str] = field(default_factory=list)
    user: np.ndarray | None = None  # (crops, U) max cosine per user sign
    user_ids: list[str] = field(default_factory=list)  # "mysign:<sign_id>"
    prior: dict[str, float] = field(default_factory=dict)

    @property
    def crops(self) -> int:
        for a in (self.cls_logp, self.ret, self.user):
            if a is not None:
                return int(a.shape[0])
        return 0


@dataclass
class FusionResult:
    keys: list[str]  # candidate keys (concept ids or "mysign:<id>"), sorted by mean probability
    probs: np.ndarray  # (K,) mean fused probability
    per_crop_top1: list[str]
    agree: int
    best_cos: float  # best mean cosine over index concepts and user signs (-1 when no retrieval)
    best_global_cos: float
    best_user: tuple[str, float] | None  # (key, mean cosine)
    components: dict[str, set[str]]  # which components ranked a key in their top-10


def fuse(inp: FusionInputs, p: FusionParams) -> FusionResult:
    crops = inp.crops
    if crops == 0:
        return FusionResult([], np.zeros(0), [], 0, -1.0, -1.0, None, {})
    has_c = inp.cls_logp is not None and len(inp.cls_ids) > 0
    has_r = inp.ret is not None and len(inp.ret_ids) > 0
    has_u = inp.user is not None and len(inp.user_ids) > 0
    comps: dict[str, set[str]] = {"classifier": set(), "retrieval": set(), "my_signs": set()}
    cand: dict[str, None] = {}
    if has_c:
        clp = _log_softmax(inp.cls_logp / max(p.t_c, 1e-3))
        mean_c = clp.mean(0)
        order = np.argsort(-mean_c)
        for i in order[: p.top_n]:
            cand[inp.cls_ids[i]] = None
        comps["classifier"] = {inp.cls_ids[i] for i in order[:10]}
        c_pos = {c: i for i, c in enumerate(inp.cls_ids)}
    if has_r:
        mean_r = inp.ret.mean(0)
        order = np.argsort(-mean_r)
        for i in order[: p.top_n]:
            cand[inp.ret_ids[i]] = None
        comps["retrieval"] = {inp.ret_ids[i] for i in order[:10]}
        r_pos = {c: i for i, c in enumerate(inp.ret_ids)}
        floor_idx = order[min(p.top_n, len(order)) - 1]
    if has_u:
        mean_u = inp.user.mean(0)
        for k in inp.user_ids:
            cand[k] = None
        comps["my_signs"] = {inp.user_ids[i] for i in np.argsort(-mean_u)[:10]}
    keys = list(cand)
    K = len(keys)
    z = np.zeros((crops, K), np.float64)
    if has_c:
        col = np.array([c_pos.get(k, -1) for k in keys])
        cl = np.where(col[None, :] >= 0, clp[:, np.maximum(col, 0)], p.noncore_logp)
        z += p.alpha * cl
    if has_r or has_u:
        logits_all = []
        if has_r:
            logits_all.append(inp.ret / p.tau_r)
        if has_u:
            logits_all.append(inp.user / p.tau_r + p.user_bonus)
        L = np.concatenate(logits_all, axis=1)
        logZ = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
        u_pos = {k: i for i, k in enumerate(inp.user_ids)} if has_u else {}
        cols = []
        for k in keys:
            if k in u_pos:
                cols.append(inp.user[:, u_pos[k]] / p.tau_r + p.user_bonus)
            elif has_r and k in r_pos:
                cols.append(inp.ret[:, r_pos[k]] / p.tau_r)
            elif has_r:  # concept without a reference in the index: neutral floor (the 50th best score)
                cols.append(inp.ret[:, floor_idx] / p.tau_r)
            else:
                cols.append(np.full(crops, -1.0 / p.tau_r))
        z += p.beta * (np.stack(cols, axis=1) - logZ[:, None])
    if p.gamma and inp.prior:
        z += p.gamma * np.array([inp.prior.get(k, 0.0) for k in keys])[None, :]
    pf = np.exp(_log_softmax(z, axis=1))
    mean = pf.mean(0)
    order = np.argsort(-mean)
    keys = [keys[i] for i in order]
    pf, mean = pf[:, order], mean[order]
    per_crop = [keys[int(i)] for i in pf.argmax(1)]
    agree = sum(k == keys[0] for k in per_crop)
    best_g = float(inp.ret.mean(0).max()) if has_r else -1.0
    best_u = None
    if has_u:
        j = int(np.argmax(mean_u))
        best_u = (inp.user_ids[j], float(mean_u[j]))
    best = max(best_g, best_u[1] if best_u else -1.0)
    return FusionResult(keys, mean, per_crop, agree, best, best_g, best_u, comps)


def decide(p_sorted: np.ndarray, agree: int, crops: int, params: FusionParams, policy: str = "balanced",
           best_cos: float | None = None) -> tuple[str, str]:
    """accept   if p1 >= accept[policy] and p1 - p2 >= margin and enough crops agree
       suggest  elif sum(top5) >= suggest_min_top5_mass
       abstain  otherwise ("Not sure - please sign again, or spell it").
    With ``best_cos`` below ``s_min`` the result is always ``abstain`` / ``out_of_vocabulary``."""
    if policy not in params.accept:
        raise ValueError(f"unknown policy {policy!r}; choose one of {sorted(params.accept)}")
    if len(p_sorted) == 0:
        return "abstain", "no_candidates"
    if best_cos is not None and params.s_min > -1.0 and best_cos < params.s_min:
        return "abstain", "out_of_vocabulary"
    p1 = float(p_sorted[0])
    p2 = float(p_sorted[1]) if len(p_sorted) > 1 else 0.0
    need = min(params.min_agree, crops)
    if p1 >= params.accept[policy] and p1 - p2 >= params.margin and agree >= need:
        return "accept", "ok"
    if p1 < params.accept[policy]:
        reason = "low_confidence"
    elif p1 - p2 < params.margin:
        reason = "ambiguous"
    else:
        reason = "unstable"
    if float(np.sum(p_sorted[:5])) >= params.suggest_min_top5_mass:
        return "suggest", reason
    return "abstain", reason
