"""Fit the fusion weights, temperature, thresholds and out-of-vocabulary guard on validation clips."""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

from app.features.build import FeatureError, build_features
from app.isl.data.sources import Clip
from app.isl.models.calibration import (fit_classifier_thresholds, fit_fusion_weights, fit_temperature,
                                        fit_thresholds, prepare_windows, true_prob, weight_grid)
from app.isl.models.fusion import FusionInputs, FusionParams, compact_inputs, fuse
from app.isl.training.common import engine_crops

# The fusion-weight grid (81 points x every clip, 54 % of research fit-fusion's 8,144 s) runs on a seeded,
# source-stratified subsample of this many validation clips; the chosen point is then scored on all of them
# (docs 15.23). Validation sets of at most this size use every clip, exactly as before.
WEIGHT_FIT_CLIPS = 3000
WEIGHT_FIT_SEED = 0


def collect_evidence(rec, clips: list[Clip], use_my_signs: bool = False, compact: dict | None = None,
                     e4_cache: dict | None = None) -> tuple[list[FusionInputs], list[str], list[Clip]]:
    """Per clip: the three engine crops through every component of ``rec`` (an ``IslRecognizer``).

    ``compact`` = {"taus": [...], "top_n": n} keeps only the retrieval evidence ``fuse`` reads at those ``tau_r``
    values (``fusion.compact_inputs``, docs 15.23; fit-fusion only). ``e4_cache`` (a dict) is filled with
    clip_id -> (inputs, embedding), the entries ``runners.e4_my_signs`` would compute again, for clips with a word
    and a signer group."""
    inputs, truths, used = [], [], []
    ret_pos = getattr(getattr(rec, "index", None), "concept_pos", None) if compact is not None else None
    emb = None
    for c in clips:
        try:
            raw = c.load()
            feats = np.stack([build_features(x, rec.feature_config) for x in engine_crops(raw)])
        except (FeatureError, OSError, ValueError):
            continue
        inp, info = rec.evidence(feats, use_my_signs=use_my_signs)
        if compact is not None:
            pos = ret_pos if (ret_pos is not None and inp.ret_ids is getattr(rec.index, "concepts", None)) else None
            inp = compact_inputs(inp, compact["taus"], compact.get("top_n", 50), ret_pos=pos)
        if e4_cache is not None and c.concept_id and c.group:
            z = (info or {}).get("z")
            if z is None or getattr(rec, "encoder", None) is None:
                if emb is None:
                    from app.isl.models.encoder import FeatureStatsEmbedder

                    emb = FeatureStatsEmbedder(rec.feature_config)
                z = emb.embed(feats)[0]
            e4_cache[c.clip_id] = (inp, z)
        inputs.append(inp)
        truths.append(c.concept_id)
        used.append(c)
    return inputs, truths, used


def weight_subsample(sources: list, n: int = WEIGHT_FIT_CLIPS, seed: int = WEIGHT_FIT_SEED) -> list[int] | None:
    """Indices of a seeded, source-stratified subsample of about ``n`` clips (``sources[i]`` = clip i's source):
    every source keeps its share, largest remainders first, and at least one clip. None when there are at most
    ``n`` clips (then the grid uses every clip, as before docs 15.23)."""
    N = len(sources)
    if n <= 0 or N <= n:
        return None
    by: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sources):
        by[str(s or "")].append(i)
    keys = sorted(by)
    quota = {k: n * len(by[k]) / N for k in keys}
    take = {k: max(1, math.floor(quota[k])) for k in keys}
    rest = n - sum(take.values())
    for k in sorted(keys, key=lambda k: (-(quota[k] - math.floor(quota[k])), k))[:max(0, rest)]:
        take[k] += 1
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for k in keys:
        idx = by[k]
        m = min(take[k], len(idx))
        out += [idx[int(j)] for j in rng.choice(len(idx), m, replace=False)]
    return sorted(out)


def check_index_for_fitting(rec, clips: list[Clip], data=None) -> dict | None:
    """Leak guard (docs 15.21, 15.22): the retrieval index ``rec`` uses must be the training-only one of this very
    encoder file and hold no row built from these clips; ``LeakageError`` otherwise. With ``data`` (``IslData``), the
    index must also record this data's split and exactly the training clips the ``index`` step would use now. None
    when ``rec`` has no index. A recogniser with an index but no encoder file path is refused: the index cannot be
    shown to match the encoder."""
    from app.isl.models.index import LeakageError

    idx = getattr(rec, "index", None)
    if idx is None:
        return None
    enc_path = getattr(getattr(rec, "encoder", None), "path", None)
    if enc_path is None:
        raise LeakageError("the recogniser has a retrieval index but no encoder file path, so the index cannot be "
                           "shown to be embedded with this encoder (docs 15.21)")
    kw: dict = {}
    if data is not None:
        from app.isl.training.build_index import index_training_clip_ids

        kw = {"split": str(data.split.get("name") or ""), "train_clip_ids": index_training_clip_ids(data)}
    return idx.check_for_evaluation(enc_path, eval_clip_ids=[c.clip_id for c in clips], **kw)


def fit_fusion(rec, val_clips: list[Clip], profile: str, version: str = "fusion-1",
               data=None, weight_clips: int = WEIGHT_FIT_CLIPS, weight_seed: int = WEIGHT_FIT_SEED
               ) -> tuple[FusionParams, dict]:
    """``data``: the run's ``IslData``, for the full index leak guard (split + training clips, docs 15.22).

    Cost (docs 15.23): the evidence is computed once - compact (only the retrieval columns ``fuse`` reads), with a
    shared prior - and reused by the "My signs" mode choice; the weight grid runs on a seeded, source-stratified
    subsample of ``weight_clips`` clips (``weight_seed``) and the chosen point is scored on every clip; the sliding
    replay keeps only what the commit rule reads."""
    t0 = time.time()
    guard = check_index_for_fitting(rec, val_clips, data)  # before any work: an eval-including index is refused
    base = FusionParams(profile=profile, version=version)
    taus = sorted({*weight_grid(True, True, base)["tau_r"], float(base.tau_r)})
    e4_cache: dict = {}
    inputs, truths, used = collect_evidence(rec, val_clips, compact={"taus": taus, "top_n": base.top_n},
                                            e4_cache=e4_cache)
    rep: dict = {"val_queries": len(inputs)}
    if guard is not None:
        rep["index_guard"] = guard
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
    sub = weight_subsample([c.source for c in used], weight_clips, weight_seed)
    if sub is None:
        params, wrep = fit_fusion_weights(inputs, truths, base)
    else:
        params, wrep = fit_fusion_weights([inputs[i] for i in sub], [truths[i] for i in sub], base)
    results = [fuse(i, params) for i in inputs]
    nll_full = float(np.mean([-math.log(max(1e-6, true_prob(r, t))) for r, t in zip(results, truths)]))
    wfit = {"subsample": len(sub) if sub is not None else None, "seed": weight_seed if sub is not None else None,
            "max_clips": weight_clips, "stratified_by": "source", "n_full": len(inputs), "grid_points": wrep["tried"],
            "nll_subsample": round(float(wrep["nll"]), 6) if sub is not None else None,
            "nll_full": round(nll_full, 6)}
    if sub is not None:
        cnt: dict = defaultdict(int)
        for i in sub:
            cnt[str(used[i].source or "")] += 1
        wfit["subsample_by_source"] = dict(sorted(cnt.items()))
    params = params.with_(weight_fit=wfit)
    rep["weights"] = {k: getattr(params, k) for k in ("alpha", "beta", "gamma", "tau_r", "t_c")} | {
        "nll": nll_full, "fit": wfit}
    params, trep = fit_thresholds(results, truths, params)
    del results
    rep["thresholds"] = {"accept": params.accept, "s_min": params.s_min,
                         "suggest_min_top5_mass": params.suggest_min_top5_mass,
                         "suggest_min_top5_mass_by_policy": params.suggest_min_top5_mass_by_policy,
                         **{k: v for k, v in trep.items() if k.endswith("reached")},
                         "per_policy": trep.get("accept_fit", {}).get("policies"),
                         "best_lower_bound": trep.get("accept_fit", {}).get("best_lower_bound"),
                         "note": trep.get("accept_fit", {}).get("note")}
    rep["achieved"] = params.achieved
    params, rep["classifier_alone"] = fit_classifier_alone(rec, inputs, truths, params, clips=used)
    params, rep["my_signs"] = choose_user_mode(rec, val_clips, params, cache=e4_cache)
    rep["seconds"] = round(time.time() - t0, 1)
    return params, rep


def sliding_windows(rec, raw, cfg=None) -> list[tuple]:
    """Replay one recorded clip through the app's sliding mode (``RecognitionEngine``, ``cfg`` or the defaults) with a
    threshold nothing reaches, and record every classified window: (t ms, EMA probabilities, streak, sign boundary
    before it, the engine's class key of the top class). EMA, streak and boundaries do not depend on commits, so
    ``calibration.sliding_commits`` gives the commits for any threshold from this one recording (docs 15.18)."""
    from dataclasses import replace

    from app.inference.engine import EngineConfig, RecognitionEngine

    eng = RecognitionEngine(None, recognizer=rec, config=replace(cfg or EngineConfig(), mode="sliding",
                                                                     threshold=2.0))
    eng.set_size(raw.width, raw.height)
    out, boundary = [], False
    for i in range(len(raw.t_ms)):
        had = eng.ema is not None
        msg = eng.push(raw.frame_to_json(i))
        if "provisional" in msg:
            out.append((float(msg["t"]), np.array(eng.ema, float), int(eng.streak), boundary, eng.streak_class))
            boundary = False
        elif had and eng.ema is None:  # hands gone: the engine allows a repeat of the last commit after this
            boundary = True
    return out


def replay_sliding(rec, clips: list[Clip], cfg=None) -> tuple[list, list[int], int]:
    """``sliding_windows`` for every clip: (windows per replayed clip, classifier class index of its word or -1,
    clips that could not be replayed)."""
    pos = {c: i for i, c in enumerate(rec.classifier.class_names)}
    windows, truths, failed = [], [], 0
    for c in clips:
        try:
            w = sliding_windows(rec, c.load(), cfg)
        except (FeatureError, OSError, ValueError):
            failed += 1
            continue
        windows.append(prepare_windows(w))  # what the commit rule reads, not ~10 EMA vectors (docs 15.23)
        truths.append(pos.get(c.concept_id, -1))
    return windows, truths, failed


def fit_classifier_alone(rec, inputs: list[FusionInputs], truths: list[str], params: FusionParams,
                         clips: list[Clip] | None = None) -> tuple[FusionParams, dict]:
    """Per-policy thresholds of the classifier ALONE (the app's sliding mode, E1), stored in ``classifier_accept``
    with the classifier's fingerprint.

    With ``clips`` (fit-fusion passes its validation clips) they are fitted on what sliding mode actually does:
    every clip replayed through ``RecognitionEngine`` in sliding mode at the default ``EngineConfig`` (single-crop
    windows, EMA, a streak of ``agree_n`` windows, repeat suppression, cooldown), one clip = one unit (docs 15.18).
    Without clips (or when none can be replayed) the segment-level proxy of docs 15.17 is used (crop-mean
    probability of three crops, >= 2 crops agree) and ``classifier_fit.rule`` says so: on commercial-safe it
    overstated sliding-mode precision by 4-11 points. When the replay has too few commits, nothing is stored and the
    engine keeps the checkpoint's single ``recommended_threshold``."""
    from app.inference.engine import EngineConfig
    from app.isl.models.calibration import fit_sliding_thresholds
    from app.isl.recognizer import calibrated_probs, classifier_calibration, classifier_fingerprint

    clf = getattr(rec, "classifier", None)
    if clf is None or not inputs or inputs[0].cls_logp is None:
        return params.with_(classifier_accept={}, classifier_fit={}), {"note": "no classifier: nothing fitted"}
    temp, ck_thr, _ = classifier_calibration(clf)
    ecfg = EngineConfig()
    margin = ecfg.min_margin
    windows, failed = [], 0
    if clips:
        windows, wtruths, failed = replay_sliding(rec, clips, ecfg)
    if windows:
        accept, rep = fit_sliding_thresholds(windows, wtruths, margin, ecfg.agree_n, ecfg.cooldown_ms)
        rule = "sliding replay"
        extra = {"engine": {k: getattr(ecfg, k) for k in ("window_ms", "interval_ms", "min_frames", "min_hand_ratio",
                                                          "ema_alpha", "agree_n", "cooldown_ms", "end_absent_ms",
                                                          "min_margin")},
                 "replay_failed": failed, "unit": rep.get("unit")}
    else:
        names = list(inputs[0].cls_ids)
        per_crop = [calibrated_probs(np.exp(np.atleast_2d(i.cls_logp)), temp) for i in inputs]
        accept, rep = fit_classifier_thresholds(per_crop, names, truths, margin, min_agree=params.min_agree)
        rule = "segment proxy"
        extra = {"min_agree": params.min_agree, "replay_failed": failed,
                 "proxy_note": "three-crop segment proxy, not the sliding-mode rule: its lower bounds overstate "
                               "sliding mode (docs 15.18)"}
    rep.update(temperature=temp, checkpoint_threshold=ck_thr, rule=rule, replay_failed=failed)
    if accept is None:
        return params.with_(classifier_accept={}, classifier_fit={}), rep
    fit = {"fingerprint": classifier_fingerprint(clf), "temperature": temp, "margin": margin, "rule": rule,
           "n": rep["n"], "n_in_classes": rep.get("n_in_classes"), "best_lower_bound": rep.get("best_lower_bound"),
           "policies": rep.get("policies"), "checkpoint_threshold": ck_thr, "note": rep.get("note"), **extra}
    return params.with_(classifier_accept=accept, classifier_fit=fit), rep


MIN_E4_TESTS = 30  # enrolled test takes needed before the user-sign classifier mode is chosen from data
MIN_E4_GROUPS = 2
MAX_FA_INCREASE = 0.01  # "neutral" may not raise the user-sign false-accept rate by more than this
# "neutral" must beat "penalty" by more than noise: at least this top-1 gain AND at least MIN_EXTRA_CORRECT more
# correct enrolled takes (with 30 takes one lucky take is a 0.033 gain)
MIN_TOP1_GAIN = 0.02
MIN_EXTRA_CORRECT = 2


def choose_user_mode(rec, val_clips: list[Clip], params: FusionParams, ks=(3, 1),
                     cache: dict | None = None) -> tuple[FusionParams, dict]:
    """Pick the classifier term of "My signs" keys (``user_cls_mode``) on validation few-shot data: the val part's
    signer groups enrol ``k`` takes of half of their words (the E4 protocol, run on validation, never on test).
    ``neutral`` is chosen only when it raises top-1 on the enrolled words by more than noise (``MIN_TOP1_GAIN`` and
    ``MIN_EXTRA_CORRECT`` takes) and the false-accept rate of user signs on the other words rises by at most
    ``MAX_FA_INCREASE``; with too little data the default ``penalty`` stays. ``cache``: per-clip evidence already
    computed (fit-fusion passes its own, ``collect_evidence(e4_cache=...)``, so nothing is computed twice)."""
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
    cache = {} if cache is None else cache
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
