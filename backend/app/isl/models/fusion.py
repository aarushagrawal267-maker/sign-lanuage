"""Calibrated fusion of classifier, retrieval and "My signs" scores, with the accept / suggest / abstain rule
(docs/ISL_DESIGN.md section 6.3 d and 7.3).

For one segment and one crop, over the union of candidate concepts::

    z_k = alpha * log p_c(k)            (classifier, temperature t_c; concepts outside it get noncore_logp;
                                         user signs too, or - with user_cls_mode "neutral" - the crop's best
                                         classifier log-probability, so the classifier cannot vote against them)
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
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

POLICIES = ("strict", "balanced", "lenient")
USER_CLS_MODES = ("penalty", "neutral")
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
    suggest_min_top5_mass: float = 0.25  # every policy, unless suggest_min_top5_mass_by_policy has that policy
    # per-policy "did you mean" mass (docs 15.17): each policy's suggestion pool differs once the accept thresholds
    # differ. Empty in packs fitted before 15.17: the single value above is used for every policy, as before.
    suggest_min_top5_mass_by_policy: dict = field(default_factory=dict)
    # classifier ALONE (sliding mode, E1), docs 15.17: per-policy accept thresholds on the classifier's
    # temperature-scaled probabilities, fitted by fit-fusion on validation. Valid only for the classifier recorded in
    # ``classifier_fit`` (its fingerprint and temperature); empty in older packs -> the checkpoint's single
    # recommended_threshold, as before.
    classifier_accept: dict = field(default_factory=dict)
    classifier_fit: dict = field(default_factory=dict)
    user_bonus: float = 1.0  # logit bonus of user signs in the retrieval softmax
    user_margin: float = 0.05  # cosine margin for the direct "My signs" override (7.3)
    user_theta_floor: float = 0.0  # lowest allowed per-sign acceptance cosine
    noncore_logp: float = math.log(1e-3)
    # classifier term of a user ("My signs") key: "penalty" = noncore_logp like any non-core concept (default);
    # "neutral" = the crop's best classifier log-probability (neither for nor against). Chosen by fit-fusion on
    # validation few-shot (E4-style) data; kept "penalty" when that data is insufficient.
    user_cls_mode: str = "penalty"
    top_n: int = 50
    version: str = "fusion-1"
    profile: str = ""
    fitted: bool = False  # False = defaults (not calibrated): the UI must say so
    achieved: dict = field(default_factory=dict)  # measured on validation: per policy precision / coverage / n
    enrol: dict = field(default_factory=lambda: {"c_min": 0.5, "min_takes": 3, "max_takes": 5})
    # how fit-fusion chose alpha / beta / gamma / tau_r (docs 15.23): the weight grid runs on a seeded,
    # source-stratified subsample of the validation clips ("subsample", "seed", "n_full", "nll_subsample",
    # "nll_full"). Empty in packs fitted before 15.23 (grid on every validation clip).
    weight_fit: dict = field(default_factory=dict)
    # certification of the accept rule (docs 15.25). None = parameters built in code (unit tests, toy tools): no
    # gate. A dict = a pack's record: a policy auto-accepts only when ``achieved[policy]["rule"]`` is "certified"
    # (with its per-source numbers in ``achieved[policy]["by_source"]``); every other policy is confirm-only (the
    # top candidate becomes "did you mean"). A fusion.json written before 15.25 has no such record and loads as {}:
    # UNCERTIFIED, every policy confirm-only.
    certification: dict | None = None
    # what the parameters were fitted on (docs 15.33): the classifier's class space, the encoder's left-out sources and
    # the kinds of index rows left out (``common.fusion_fit_space``). {} = not recorded (fitted before 15.33): no check.
    # evaluate --all flags a fusion fitted on another class space; build-pack refuses it.
    fitted_on: dict = field(default_factory=dict)

    def __post_init__(self):
        # Every policy must always have a threshold: a fusion.json written with only some policies (or none) made
        # decide() raise for the missing ones (HTTP 500 in the API). Missing values are filled from the present
        # ones (or the defaults) and kept ordered strict >= balanced >= lenient.
        object.__setattr__(self, "accept", normalize_accept(self.accept))
        # optional per-policy values: unknown keys and non-numbers dropped; a missing policy falls back to the single
        # value (suggest) or to the checkpoint threshold (classifier alone: all three or none)
        sm = {}
        for k, v in dict(self.suggest_min_top5_mass_by_policy or {}).items():
            try:
                if k in POLICIES and v is not None and math.isfinite(float(v)):
                    sm[k] = float(v)
            except (TypeError, ValueError):
                continue
        object.__setattr__(self, "suggest_min_top5_mass_by_policy", sm)
        ca = {k: v for k, v in dict(self.classifier_accept or {}).items() if k in POLICIES and v is not None}
        object.__setattr__(self, "classifier_accept", normalize_accept(ca) if ca else {})
        object.__setattr__(self, "classifier_fit", dict(self.classifier_fit or {}))
        if self.user_cls_mode not in USER_CLS_MODES:
            raise ValueError(f"user_cls_mode must be one of {USER_CLS_MODES}, not {self.user_cls_mode!r}")
        if self.certification is not None:
            object.__setattr__(self, "certification", dict(self.certification))
        object.__setattr__(self, "fitted_on", dict(self.fitted_on or {}))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict | None) -> FusionParams:
        d = dict(d or {})
        # a file without a certification record (fitted before docs 15.25) is uncertified: every policy confirm-only
        cert = d.get("certification")
        d["certification"] = dict(cert) if isinstance(cert, dict) else {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> FusionParams:
        return cls.from_json(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    def with_(self, **kw) -> FusionParams:
        return replace(self, **kw)

    def suggest_mass(self, policy: str) -> float:
        """Smallest top-5 mass for a "did you mean" under ``policy`` (the single value in packs before 15.17)."""
        return float(self.suggest_min_top5_mass_by_policy.get(policy, self.suggest_min_top5_mass))

    # ------------------------------------------------------------ certification (docs 15.25)
    @property
    def gated(self) -> bool:
        """True when this is a pack's record (loaded or fitted): uncertified policies are then confirm-only."""
        return self.certification is not None

    def policy_rule(self, policy: str) -> str:
        """"certified" (may auto-accept), "confirm_only" (never auto-accepts) or "ungated" (in-code parameters)."""
        if self.certification is None:
            return "ungated"
        a = (self.achieved or {}).get(policy) or {}
        if self.certification.get("version") and a.get("rule") == "certified" and isinstance(a.get("by_source"), dict):
            return "certified"
        return "confirm_only"

    def confirm_only(self, policy: str) -> bool:
        return self.policy_rule(policy) == "confirm_only"

    def open_vocabulary_confirm_only(self, policy: str) -> bool:
        """Words outside the classifier (retrieval only) auto-accept only when the policy is certified AND the
        open-vocabulary check certified them at the strict floor (docs 15.25)."""
        if self.certification is None:
            return False
        if self.confirm_only(policy):
            return True
        ov = ((self.certification.get("open_vocabulary") or {}).get("policies") or {}).get(policy) or {}
        return ov.get("rule") != "certified"

    def certification_info(self, policy: str) -> dict:
        """What the app shows next to a word (docs 15.25): the rule, the CERTIFIED worst-source precision (the 95 %
        lower bound on the worst major source, never the target), that source, the floor and the target.
        ``confirm_tap`` is always True: even a certified policy is below the 0.98 goal, so the app asks the user to
        confirm before a word is spoken or sent."""
        rule = self.policy_rule(policy)
        a = (self.achieved or {}).get(policy) or {}
        ov = ((((self.certification or {}).get("open_vocabulary") or {}).get("policies") or {}).get(policy) or {})
        return {"rule": rule, "certified_precision": a.get("certified_precision") if rule == "certified" else None,
                "worst_source": a.get("worst_source"), "floor": a.get("floor"), "target": a.get("target"),
                "target_reached": bool(a.get("target_reached")) if rule != "ungated" else None,
                "open_vocabulary_rule": ("ungated" if rule == "ungated" else
                                         "confirm_only" if self.open_vocabulary_confirm_only(policy) else "certified"),
                "fitted": self.fitted, "confirm_tap": True,
                **({"reason": "fusion.json has no certification record (fitted before docs 15.25 or not fitted): "
                              "every word needs a confirm tap"} if rule != "ungated" and not self.certification else {})}


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


class ConceptPrior(Mapping):
    """The fusion prior of every vocabulary concept (``Vocabulary.prior``: 1 core, 0.5 single word, 0 other), read
    only and shared by reference: one per recogniser, not a dict per segment (docs 15.23; a per-segment dict of
    about 4,400 classifier keys was 104 KB per validation clip in fit-fusion). User signs ("mysign:...") and unknown
    keys are missing, so ``get(k, 0.0)`` gives 0 for them, as before. Values are computed once per concept."""

    def __init__(self, vocab):
        self._vocab = vocab
        self._cache: dict[str, float] = {}

    def __getitem__(self, key: str) -> float:
        v = self._cache.get(key)
        if v is None:
            if not isinstance(key, str) or key.startswith(USER_PREFIX) or self._vocab.get(key) is None:
                raise KeyError(key)
            v = self._cache[key] = float(self._vocab.prior(key))
        return v

    def __iter__(self):
        return iter(self._vocab.pos)

    def __len__(self) -> int:
        return len(self._vocab)


@dataclass
class FusionInputs:
    """Per-crop evidence for one segment. Any part may be missing (None / empty).

    **Compact retrieval** (``compact_inputs``, fit-fusion only, docs 15.23): ``ret`` then holds only the columns
    ``fuse`` can read - the ``ret_top`` best index concepts by crop-mean cosine, in that order, plus the columns of the
    classifier classes in ``ret_cover`` that are in the index - and ``ret_lse[tau]`` is the per-crop log-sum-exp of
    ``ret / tau`` over ALL ``ret_size`` index concepts (the softmax normaliser of the retrieval term), so ``fuse`` gives
    the same result as on the full matrix without keeping it.

    **Compact classifier** (``compact_inputs(..., t_cs=...)``, fit-fusion only, docs 15.24): ``cls_logp`` then holds
    only the ``cls_top`` best classes by crop-mean log-probability, best first, plus the classes of kept retrieval
    columns (``ret_cls`` points into these columns), and ``cls_lse[j]`` = (max, log-sum-exp after the max) of
    ``cls_logp / t`` over ALL ``cls_size`` classes, per crop, for each temperature ``t = cls_t[j]``: the two
    numbers ``_log_softmax`` computes, so ``fuse`` gives bit for bit the old classifier log-probabilities."""

    cls_logp: np.ndarray | None = None  # (crops, C) log-probabilities at temperature 1
    cls_ids: list[str] = field(default_factory=list)
    ret: np.ndarray | None = None  # (crops, G) max cosine per index concept
    ret_ids: list[str] = field(default_factory=list)
    user: np.ndarray | None = None  # (crops, U) max cosine per user sign
    user_ids: list[str] = field(default_factory=list)  # "mysign:<sign_id>"
    prior: Mapping[str, float] = field(default_factory=dict)  # a shared ConceptPrior in recogniser evidence
    # compact retrieval only (None / 0 otherwise), see the class docstring
    ret_lse: dict | None = None  # {tau: (crops,) float32 log-sum-exp of ret / tau over every index concept}
    ret_top: int = 0  # leading columns of ``ret``: the best index concepts by crop-mean cosine, best first
    ret_size: int = 0  # index concepts in the full matrix
    ret_cls: np.ndarray | None = None  # (columns,) classifier class index of each kept column, -1 = not a class
    ret_cover: np.ndarray | None = None  # classifier class indices whose index column is kept when they have one
    # compact classifier only (None / 0 otherwise), see the class docstring
    cls_lse: np.ndarray | None = None  # (len(cls_t), 2, crops): max and log-sum-exp of cls_logp / t over every class
    cls_t: tuple = ()  # the temperatures of cls_lse (one shared tuple)
    cls_top: int = 0  # leading columns of ``cls_logp``: the best classes by crop-mean log-probability, best first
    cls_size: int = 0  # classes in the full classifier output

    @property
    def crops(self) -> int:
        for a in (self.cls_logp, self.ret, self.user):
            if a is not None:
                return int(a.shape[0])
        return 0

    @property
    def compact(self) -> bool:
        return self.ret_lse is not None

    @property
    def compact_cls(self) -> bool:
        return self.cls_lse is not None


def _ret_logz(ret: np.ndarray, tau: float) -> np.ndarray:
    """Per-crop log-sum-exp of ``ret / tau``: the exact expression ``fuse`` evaluates on the full matrix."""
    L = ret / tau
    return np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)


def compact_inputs(inp: FusionInputs, taus, top_n: int = 50, cover: int | None = None,
                   ret_pos: dict | None = None, t_cs=None) -> FusionInputs:
    """``inp`` with the retrieval matrix reduced to what ``fuse`` reads for ``top_n`` <= the kept top and ``tau_r``
    in ``taus`` (docs 15.23): the ``top_n`` (at least 10) best index concepts by crop-mean cosine, the columns of
    the ``cover`` (default 2 x top_n) best classifier classes by crop-mean log-probability, and the per-crop
    retrieval normaliser for every tau. For research validation that is about 150 of 12,440 columns, so a clip
    keeps about 2 KB of retrieval evidence instead of 149 KB. ``ret_pos``: concept -> column of ``inp.ret_ids``
    (pass the index's map to avoid rebuilding it per clip). Inputs without retrieval, or already compact, are
    returned unchanged. ``fuse`` raises for a ``top_n`` or ``tau_r`` the compact inputs cannot answer.

    ``t_cs`` (temperatures ``t_c``) also compacts the classifier (docs 15.24): the same ``cover`` best classes plus
    the classes of the kept retrieval columns, and the per-crop softmax normaliser for every ``t_c`` - about 2 KB
    instead of 3 x C float32 (52 KB for research's 4,354 classes)."""
    out = inp
    if inp.ret is not None and len(inp.ret_ids) and not inp.compact:
        out = _compact_retrieval(inp, taus, top_n, cover, ret_pos)
    if t_cs is not None and out.cls_logp is not None and len(out.cls_ids) and not out.compact_cls:
        out = _compact_classifier(out, t_cs, top_n, cover)
    return out


def _cls_normaliser(x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(max, log-sum-exp after the max) of ``x / t`` per row, exactly as ``_log_softmax`` computes them in ``fuse``."""
    xt = x / max(t, 1e-3)
    m = np.max(xt, axis=-1, keepdims=True)
    return m, np.log(np.sum(np.exp(xt - m), axis=-1, keepdims=True))


def _compact_classifier(inp: FusionInputs, t_cs, top_n: int, cover: int | None) -> FusionInputs:
    x = inp.cls_logp
    C = x.shape[1]
    m = int(cover) if cover is not None else 2 * int(top_n)
    order = np.argsort(-x.mean(0))  # the ranking of fuse (and of ret_cover): T > 0 does not change it
    top = [int(i) for i in order[:m]]
    have = set(top)
    extra = sorted({int(c) for c in (inp.ret_cls if inp.ret_cls is not None else ()) if c >= 0} - have)
    kept = top + extra
    new_pos = {c: j for j, c in enumerate(kept)}
    ts = t_cs if isinstance(t_cs, tuple) else tuple(float(t) for t in t_cs)  # pass a tuple to share it
    lse = np.stack([np.stack([a[:, 0] for a in _cls_normaliser(x, t)]) for t in ts])
    ret_cls = ret_cover = None
    if inp.ret_cls is not None:
        ret_cls = np.array([new_pos[int(c)] if c >= 0 else -1 for c in inp.ret_cls], np.int32)
    if inp.ret_cover is not None:
        ret_cover = np.array(sorted(new_pos[int(c)] for c in inp.ret_cover if int(c) in new_pos), np.int32)
    return replace(inp, cls_logp=np.ascontiguousarray(x[:, kept]), cls_ids=[inp.cls_ids[c] for c in kept],
                   cls_lse=lse, cls_t=ts, cls_top=len(top), cls_size=C, ret_cls=ret_cls, ret_cover=ret_cover)


def _compact_retrieval(inp: FusionInputs, taus, top_n: int, cover: int | None, ret_pos: dict | None) -> FusionInputs:
    G = inp.ret.shape[1]
    order = np.argsort(-inp.ret.mean(0))  # the order fuse uses
    n_top = min(G, max(int(top_n), 10))
    cols = [int(i) for i in order[:n_top]]
    have = set(cols)
    ret_cover = None
    has_c = inp.cls_logp is not None and len(inp.cls_ids) > 0
    if has_c:
        pos = ret_pos if ret_pos is not None else {c: i for i, c in enumerate(inp.ret_ids)}
        m = int(cover) if cover is not None else 2 * int(top_n)
        ret_cover = np.sort(np.argsort(-inp.cls_logp.mean(0))[:m]).astype(np.int32)
        for ci in ret_cover:
            j = pos.get(inp.cls_ids[int(ci)])
            if j is not None and j not in have:
                cols.append(int(j))
                have.add(int(j))
    ids = [inp.ret_ids[j] for j in cols]
    ret_cls = None
    if has_c:
        cpos = {c: i for i, c in enumerate(inp.cls_ids)}
        ret_cls = np.array([cpos.get(k, -1) for k in ids], np.int32)
    lse = {float(t): _ret_logz(inp.ret, float(t)) for t in taus}
    return replace(inp, ret=np.ascontiguousarray(inp.ret[:, cols]), ret_ids=ids, ret_lse=lse, ret_top=n_top,
                   ret_size=G, ret_cls=ret_cls, ret_cover=ret_cover)


def _lse_lookup(lse: dict, tau: float):
    for t, v in lse.items():
        if abs(t - tau) <= 1e-12 * max(1.0, abs(tau)):
            return v
    return None


def _t_index(ts: tuple, t: float) -> int | None:
    for j, v in enumerate(ts):
        if abs(v - t) <= 1e-12 * max(1.0, abs(t)):
            return j
    return None


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
    compact = has_r and inp.compact
    comps: dict[str, set[str]] = {"classifier": set(), "retrieval": set(), "my_signs": set()}
    cand: dict[str, None] = {}
    if has_c:
        if inp.compact_cls:
            # the kept columns of _log_softmax(cls_logp / t_c) with the stored normaliser: bit for bit the old values
            # (docs 15.24); the row maximum of the full matrix is 0 - lse
            if has_r and not compact:
                raise ValueError("compact classifier inputs need compact retrieval inputs (docs 15.24)")
            j = _t_index(inp.cls_t, p.t_c)
            if j is None:
                raise ValueError(f"compact fusion inputs have no classifier normaliser for t_c={p.t_c} "
                                 f"(stored: {len(inp.cls_t)} temperatures; docs 15.24)")
            if min(p.top_n, inp.cls_size) > inp.cls_top:
                raise ValueError(f"compact fusion inputs keep the best {inp.cls_top} classifier classes; top_n "
                                 f"{p.top_n} needs more (docs 15.24)")
            cm, cs = inp.cls_lse[j, 0][:, None], inp.cls_lse[j, 1][:, None]
            clp = inp.cls_logp / max(p.t_c, 1e-3) - cm - cs
            clp_max = -cs
        else:
            clp = _log_softmax(inp.cls_logp / max(p.t_c, 1e-3))
            clp_max = None
        mean_c = clp.mean(0)
        order = np.argsort(-mean_c)
        if inp.compact_cls and (order[: p.top_n] >= inp.cls_top).any():
            raise ValueError("compact fusion inputs do not cover this classifier candidate set (docs 15.24)")
        for i in order[: p.top_n]:
            cand[inp.cls_ids[i]] = None
        comps["classifier"] = {inp.cls_ids[i] for i in order[:10]}
        if compact:
            # classifier positions of the candidates only (the full map costs more than the rest of fuse)
            c_pos = {inp.cls_ids[i]: int(i) for i in order[: p.top_n]}
            if inp.ret_cls is not None:
                c_pos.update((k, int(c)) for k, c in zip(inp.ret_ids, inp.ret_cls) if c >= 0)
            if inp.ret_cover is not None and not np.isin(order[: p.top_n], inp.ret_cover).all():
                raise ValueError("compact fusion inputs do not cover this classifier candidate set (docs 15.23)")
        else:
            c_pos = {c: i for i, c in enumerate(inp.cls_ids)}
    if has_r:
        mean_r = inp.ret.mean(0)
        if compact:
            if min(p.top_n, inp.ret_size) > inp.ret_top:
                raise ValueError(f"compact fusion inputs keep the best {inp.ret_top} retrieval columns; top_n "
                                 f"{p.top_n} needs more (docs 15.23)")
            order = np.arange(inp.ret_top)  # the kept top columns are stored in fuse's order
            floor_idx = order[min(p.top_n, inp.ret_size) - 1]
        else:
            order = np.argsort(-mean_r)
            floor_idx = order[min(p.top_n, len(order)) - 1]
        for i in order[: p.top_n]:
            cand[inp.ret_ids[i]] = None
        comps["retrieval"] = {inp.ret_ids[i] for i in order[:10]}
        r_pos = {c: i for i, c in enumerate(inp.ret_ids)}
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
        fill = np.full((crops, K), p.noncore_logp)
        if has_u and p.user_cls_mode == "neutral":
            is_user = np.array([k in set(inp.user_ids) for k in keys])
            fill[:, is_user] = clp.max(1, keepdims=True) if clp_max is None else clp_max
        cl = np.where(col[None, :] >= 0, clp[:, np.maximum(col, 0)], fill)
        z += p.alpha * cl
    if has_r or has_u:
        logits_all = []
        if has_r:
            logits_all.append(inp.ret / p.tau_r)
        if has_u:
            logits_all.append(inp.user / p.tau_r + p.user_bonus)
        if compact:
            # the normaliser over every index concept, stored per tau; user signs join it by log-add-exp. logZ is a
            # per-crop constant of every key, so it cancels in the softmax below up to float rounding (docs 15.23)
            lz = _lse_lookup(inp.ret_lse, p.tau_r)
            if lz is None:
                raise ValueError(f"compact fusion inputs have no retrieval normaliser for tau_r={p.tau_r} "
                                 f"(stored: {sorted(inp.ret_lse)}; docs 15.23)")
            if has_u:
                lu = _ret_logz(logits_all[1], 1.0)
                m = np.maximum(lz, lu)
                logZ = (m + np.log(np.exp(lz - m) + np.exp(lu - m))).astype(lz.dtype)
            else:
                logZ = lz
        else:
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
           best_cos: float | None = None, open_vocabulary: bool = False) -> tuple[str, str]:
    """accept   if p1 >= accept[policy] and p1 - p2 >= margin and enough crops agree
       suggest  elif sum(top5) >= suggest_min_top5_mass (per policy when fitted so, ``FusionParams.suggest_mass``)
       abstain  otherwise ("Not sure - please sign again, or spell it").
    With ``best_cos`` below ``s_min`` the result is always ``abstain`` / ``out_of_vocabulary``.

    Certification (docs 15.25): for a confirm-only policy (``FusionParams.confirm_only``) a query that passes the
    accept rule is never accepted: its top candidate becomes "did you mean" (``suggest`` / ``confirm_only``).
    ``open_vocabulary``: the top candidate is a word outside the classifier (retrieval only); it is accepted only
    when the open-vocabulary check certified the policy, else ``suggest`` / ``open_vocabulary``."""
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
        if params.confirm_only(policy):
            return "suggest", "confirm_only"
        if open_vocabulary and params.open_vocabulary_confirm_only(policy):
            return "suggest", "open_vocabulary"
        return "accept", "ok"
    if p1 < params.accept[policy]:
        reason = "low_confidence"
    elif p1 - p2 < params.margin:
        reason = "ambiguous"
    else:
        reason = "unstable"
    if float(np.sum(p_sorted[:5])) >= params.suggest_mass(policy):
        return "suggest", reason
    return "abstain", reason
