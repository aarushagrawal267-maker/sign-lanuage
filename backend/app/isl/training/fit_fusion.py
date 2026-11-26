"""Fit the fusion weights, temperature, thresholds and out-of-vocabulary guard on validation clips."""

from __future__ import annotations

import time

import numpy as np

from app.features.build import FeatureError, build_features
from app.isl.data.sources import Clip
from app.isl.models.calibration import (fit_classifier_thresholds, fit_fusion_weights, fit_temperature,
                                        fit_thresholds)
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
                         "suggest_min_top5_mass_by_policy": params.suggest_min_top5_mass_by_policy,
                         **{k: v for k, v in trep.items() if k.endswith("reached")},
                         "per_policy": trep.get("accept_fit", {}).get("policies"),
                         "best_lower_bound": trep.get("accept_fit", {}).get("best_lower_bound"),
                         "note": trep.get("accept_fit", {}).get("note")}
    rep["achieved"] = params.achieved
    params, rep["classifier_alone"] = fit_classifier_alone(rec, inputs, truths, params)
    params, rep["my_signs"] = choose_user_mode(rec, val_clips, params)
    rep["seconds"] = round(time.time() - t0, 1)
    return params, rep


def fit_classifier_alone(rec, inputs: list[FusionInputs], truths: list[str],
                         params: FusionParams) -> tuple[FusionParams, dict]:
    """Per-policy thresholds of the classifier ALONE (the app's sliding mode, E1) on the validation evidence fit-fusion
    already has, with the engine's rule: crop-mean probability at the checkpoint's temperature, margin
    ``EngineConfig.min_margin``, >= 2 crops agree (docs 15.17). Stored in ``classifier_accept`` with the
    classifier's fingerprint; without a classifier or validation data nothing is stored and the engine keeps the
    checkpoint's single ``recommended_threshold``."""
    from app.inference.engine import EngineConfig
    from app.isl.recognizer import calibrated_probs, classifier_calibration, classifier_fingerprint

    clf = getattr(rec, "classifier", None)
    if clf is None or not inputs or inputs[0].cls_logp is None:
        return params.with_(classifier_accept={}, classifier_fit={}), {"note": "no classifier: nothing fitted"}
    temp, ck_thr, _ = classifier_calibration(clf)
    names = list(inputs[0].cls_ids)
    per_crop = [calibrated_probs(np.exp(np.atleast_2d(i.cls_logp)), temp) for i in inputs]
    margin = EngineConfig().min_margin
    accept, rep = fit_classifier_thresholds(per_crop, names, truths, margin, min_agree=params.min_agree)
    rep.update(temperature=temp, checkpoint_threshold=ck_thr)
    if accept is None:
        return params.with_(classifier_accept={}, classifier_fit={}), rep
    fit = {"fingerprint": classifier_fingerprint(clf), "temperature": temp, "margin": margin,
           "min_agree": params.min_agree, "n": rep["n"], "n_in_classes": rep.get("n_in_classes"),
           "best_lower_bound": rep.get("best_lower_bound"), "policies": rep.get("policies"),
           "checkpoint_threshold": ck_thr, "note": rep.get("note")}
    return params.with_(classifier_accept=accept, classifier_fit=fit), rep


MIN_E4_TESTS = 30  # enrolled test takes needed before the user-sign classifier mode is chosen from data
MIN_E4_GROUPS = 2
MAX_FA_INCREASE = 0.01  # "neutral" may not raise the user-sign false-accept rate by more than this
# "neutral" must beat "penalty" by more than noise: at least this top-1 gain AND at least MIN_EXTRA_CORRECT more
# correct enrolled takes (with 30 takes one lucky take is a 0.033 gain)
MIN_TOP1_GAIN = 0.02
MIN_EXTRA_CORRECT = 2


def choose_user_mode(rec, val_clips: list[Clip], params: FusionParams, ks=(3, 1)) -> tuple[FusionParams, dict]:
    """Pick the classifier term of "My signs" keys (``user_cls_mode``) on validation few-shot data: the val part's
    signer groups enrol ``k`` takes of half of their words (the E4 protocol, run on validation, never on test).
    ``neutral`` is chosen only when it raises top-1 on the enrolled words by more than noise (``MIN_TOP1_GAIN`` and
    ``MIN_EXTRA_CORRECT`` takes) and the false-accept rate of user signs on the other words rises by at most
    ``MAX_FA_INCREASE``; with too little data the default ``penalty`` stays."""
    from collections import Counter, defaultdict

    from app.isl.evaluation.runners import e4_my_signs

    rep: dict = {"modes": {}}
    # cheap pre-check (no features): signer groups with >= 2 words of >= k+1 takes each
    takes: dict = defaultdict(Counter)
    for c in val_clips:
        if c.concept_id and c.group:
            takes[c.group][c.concept_id] += 1
    k_min = min(ks)
    eligible = [g for g, cnt in takes.items() if sum(1 for n in cnt.values() if n >= k_min + 1) >= 2]
    if len(eligible) < MIN_E4_GROUPS:
        rep.update(chosen=params.user_cls_mode, k=None,
                   reason=f"insufficient validation few-shot data ({len(eligible)} signer groups with repeated takes; "
                          f"need {MIN_E4_GROUPS}): kept {params.user_cls_mode!r}")
        return params, rep
    cache: dict = {}
    res = {"penalty": e4_my_signs(rec, val_clips, ks=ks, params=params.with_(user_cls_mode="penalty"), cache=cache)}
    if any((res["penalty"].get("by_k") or {}).get(f"k{k}", {}).get("n", 0) >= MIN_E4_TESTS for k in ks):
        res["neutral"] = e4_my_signs(rec, val_clips, ks=ks, params=params.with_(user_cls_mode="neutral"), cache=cache)
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
    extra_correct = gain * (pen["n"] or 0)
    clear_gain = gain >= MIN_TOP1_GAIN and extra_correct >= MIN_EXTRA_CORRECT - 1e-9
    choose = "neutral" if clear_gain and fa_up <= MAX_FA_INCREASE else "penalty"
    rep.update(chosen=choose, k=k_used, top1_gain_neutral=round(gain, 4), false_accept_increase_neutral=round(fa_up, 4),
               extra_correct_neutral=round(extra_correct, 1),
               reason=("neutral raises enrolled-word top-1 without more user-sign false accepts" if choose == "neutral"
                       else (f"neutral's top-1 gain ({gain:+.3f}, {extra_correct:+.1f} takes) is within noise "
                             f"(need >= {MIN_TOP1_GAIN} and >= {MIN_EXTRA_CORRECT} takes)" if not clear_gain
                             else "neutral raised user-sign false accepts")))
    return params.with_(user_cls_mode=choose), rep
