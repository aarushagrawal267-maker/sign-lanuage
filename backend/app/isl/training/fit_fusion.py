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
    rep["seconds"] = round(time.time() - t0, 1)
    return params, rep
