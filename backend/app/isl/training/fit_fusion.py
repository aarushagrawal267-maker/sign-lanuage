"""Fit the fusion weights, temperature, thresholds and out-of-vocabulary guard on validation clips."""

from __future__ import annotations

import time

import numpy as np

from app.features.build import FeatureError, build_features
from app.isl.data.sources import Clip
from app.isl.models.calibration import fit_fusion_weights, fit_temperature, fit_thresholds
from app.isl.models.fusion import FusionInputs, FusionParams, fuse
from app.isl.training.common import engine_crops


def collect_evidence(rec, clips: list[Clip], use_my_signs: bool = False) -> tuple[list[FusionInputs], list[str], list[Clip]]:
    """Per clip: the three engine crops through every component of ``rec`` (an ``IslRecognizer``)."""
    inputs, truths, used = [], [], []
    for c in clips:
        try:
            raw = c.load()
            feats = np.stack([build_features(x, rec.feature_config) for x in engine_crops(raw)])
        except (FeatureError, OSError, ValueError):
            continue
        inp, _ = rec.evidence(feats, use_my_signs=use_my_signs)
        inputs.append(inp)
        truths.append(c.concept_id)
        used.append(c)
    return inputs, truths, used


def fit_fusion(rec, val_clips: list[Clip], profile: str, version: str = "fusion-1") -> tuple[FusionParams, dict]:
    t0 = time.time()
    inputs, truths, used = collect_evidence(rec, val_clips)
    base = FusionParams(profile=profile, version=version)
    rep: dict = {"val_queries": len(inputs)}
    if not inputs:
        return base, {**rep, "note": "no validation clips: default (uncalibrated) fusion parameters"}
    if inputs[0].cls_logp is not None:
        ids = inputs[0].cls_ids
        pos = {c: i for i, c in enumerate(ids)}
        sel = [(i, pos[t]) for i, t in enumerate(truths) if t in pos]
        if sel:
            lg = np.stack([inputs[i].cls_logp[0] for i, _ in sel])
            base = base.with_(t_c=fit_temperature(lg, np.array([y for _, y in sel])))
            rep["temperature"] = base.t_c
    params, wrep = fit_fusion_weights(inputs, truths, base)
    rep["weights"] = {k: getattr(params, k) for k in ("alpha", "beta", "gamma", "tau_r", "t_c")} | {"nll": wrep["nll"]}
    results = [fuse(i, params) for i in inputs]
    params, trep = fit_thresholds(results, truths, params)
    rep["thresholds"] = {"accept": params.accept, "s_min": params.s_min,
                         "suggest_min_top5_mass": params.suggest_min_top5_mass,
                         **{k: v for k, v in trep.items() if k.endswith("reached")}}
    rep["achieved"] = params.achieved
    params, rep["my_signs"] = choose_user_mode(rec, val_clips, params)
    rep["seconds"] = round(time.time() - t0, 1)
    return params, rep


MIN_E4_TESTS = 30  # enrolled test takes needed before the user-sign classifier mode is chosen from data
MIN_E4_GROUPS = 2
MAX_FA_INCREASE = 0.01  # "neutral" may not raise the user-sign false-accept rate by more than this


def choose_user_mode(rec, val_clips: list[Clip], params: FusionParams, ks=(3, 1)) -> tuple[FusionParams, dict]:
    """Pick the classifier term of "My signs" keys (``user_cls_mode``) on validation few-shot data: the val part's
    signer groups enrol ``k`` takes of half of their words (the E4 protocol, run on validation, never on test).
    ``neutral`` is chosen only when it raises top-1 on the enrolled words and the false-accept rate of user signs
    on the other words rises by at most ``MAX_FA_INCREASE``; with too little data the default ``penalty`` stays."""
    from app.isl.evaluation.runners import e4_my_signs

    cache: dict = {}
    res = {m: e4_my_signs(rec, val_clips, ks=ks, params=params.with_(user_cls_mode=m), cache=cache)
           for m in ("penalty", "neutral")}
    rep: dict = {"modes": {}}
    k_used = None
    for k in ks:
        a = (res["penalty"].get("by_k") or {}).get(f"k{k}")
        if a and a["n"] >= MIN_E4_TESTS and a["n_groups"] >= MIN_E4_GROUPS:
            k_used = k
            break
    if k_used is None:
        best = max(((res["penalty"].get("by_k") or {}).get(f"k{k}") or {"n": 0, "n_groups": 0} for k in ks),
                   key=lambda r: r["n"])
        rep.update(chosen=params.user_cls_mode, k=None,
                   reason=f"insufficient validation few-shot data ({best['n']} enrolled test takes, {best['n_groups']} "
                          f"groups; need {MIN_E4_TESTS} and {MIN_E4_GROUPS}): kept {params.user_cls_mode!r}")
        return params, rep
    for m in ("penalty", "neutral"):
        r = res[m]["by_k"][f"k{k_used}"]
        rep["modes"][m] = {x: r.get(x) for x in ("n", "n_groups", "top1_with_enrolment", "gain",
                                                  "user_sign_false_accept_rate", "n_other_words")}
    pen, neu = rep["modes"]["penalty"], rep["modes"]["neutral"]
    gain = neu["top1_with_enrolment"] - pen["top1_with_enrolment"]
    fa_up = (neu["user_sign_false_accept_rate"] or 0.0) - (pen["user_sign_false_accept_rate"] or 0.0)
    choose = "neutral" if gain > 0 and fa_up <= MAX_FA_INCREASE else "penalty"
    rep.update(chosen=choose, k=k_used, top1_gain_neutral=round(gain, 4), false_accept_increase_neutral=round(fa_up, 4),
               reason=("neutral raises enrolled-word top-1 without more user-sign false accepts" if choose == "neutral"
                       else "neutral did not raise enrolled-word top-1 or raised user-sign false accepts"))
    return params.with_(user_cls_mode=choose), rep
