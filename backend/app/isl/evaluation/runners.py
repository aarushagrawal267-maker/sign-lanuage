"""Phase-2 evaluation runners and ``evaluate --all`` (docs/ISL_DESIGN.md 9 and 15.2).

Every accuracy number is reported with the **not-sure rate** (abstain), the **wrong-accept rate** (accepted
and wrong / all queries), the suggest rate, n, the number of signer groups and a 95 % interval from a bootstrap
over **signer groups** (``report.grouped_bootstrap``), never over clips.

Protocols (each one says in its label what it is and is not):

E1a  INCLUDE official split (not signer-independent) - only with a classifier trained on that split
E1b  INCLUDE primary split (session- or estimated signer-independent), + any cross-validation fold models
E1c  other labelled sources' signer-held-out test signers (real signer ids, e.g. ISL500)
E2   open-vocabulary one-shot: per fold, an encoder trained with the fold's concepts (and every concept sharing
     a headword / synonym with them) removed from **all** supervised training; INCLUDE clips of those concepts
     queried against the ISLRTC dictionary; decision thresholds fitted on seen-class validation queries only
E3   fused system (classifier + retrieval + fusion) with accept / suggest / abstain per policy
E4   My signs few-shot: enrol k takes of a test signer's word, test on their other takes; false accepts of user
     signs on words they did not enrol
E5   fingerspelling, leave-one-signer-out (computed when the letter model is trained; read from its checkpoint)
E6   latency per segment
E7   ASL regression: the bundled ASL model's results file is unchanged
E8   trim quality (``cli trim-eval`` report + the trim method mix)

Leakage guards: E1 refuses a classifier trained on another split; E2 refuses an encoder whose excluded
concepts do not cover the (synonym-expanded) held-out set; queries in a duplicate group of any training clip
(same video hash, same landmark file, landmark near-duplicate) are dropped and counted.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.features.build import FeatureError, build_features
from app.isl.evaluation.report import git_commit, grouped_bootstrap
from app.isl.models.calibration import fit_thresholds, summarize
from app.isl.models.fusion import POLICIES, USER_PREFIX, FusionInputs, FusionParams, decide, fuse
from app.isl.training.common import engine_crops

DECISION_KEYS = ("top1", "top5", "coverage", "accepted_precision", "not_sure_rate", "wrong_accept_rate",
                 "suggest_rate", "suggest_top5_hit")


class LeakageError(RuntimeError):
    """An evaluation would use data the model was trained on."""


# --------------------------------------------------------------------------------------------- helpers

def decision_block(results, truths: list[str], params: FusionParams, groups: list) -> dict:
    """Per policy: accuracy with coverage, not-sure (abstain) and wrong-accept rates; grouped intervals."""
    out = {}
    hit1 = [float(bool(r.keys) and r.keys[0] == t) for r, t in zip(results, truths)]
    hit5 = [float(t in r.keys[:5]) for r, t in zip(results, truths)]
    for pol in POLICIES:
        s = summarize(results, truths, params, pol)
        dec = [decide(r.probs, r.agree, len(r.per_crop_top1), params, pol, r.best_cos) for r in results]
        wrong_acc = [float(d == "accept" and not h) for (d, _), h in zip(dec, hit1)]
        not_sure = [float(d == "abstain") for d, _ in dec]
        out[pol] = {**s, "not_sure_rate": s.get("abstain_rate"), "wrong_accept_rate": s.get("silent_error_rate"),
                    "intervals": {"wrong_accept_rate": grouped_bootstrap(wrong_acc, groups),
                                  "not_sure_rate": grouped_bootstrap(not_sure, groups)}}
    return {"policies": out, "intervals": {"top1": grouped_bootstrap(hit1, groups), "top5": grouped_bootstrap(hit5, groups)}}


def _metrics_from(block: dict, extra: dict | None = None) -> dict:
    b = block["policies"]["balanced"]
    return {k: b.get(k) for k in DECISION_KEYS} | (extra or {})


def _features(clip, fcfg):
    return np.stack([build_features(x, fcfg) for x in engine_crops(clip.load())])


def _groups(clips) -> list[str]:
    return [c.group or c.clip_id for c in clips]


def drop_trained_duplicates(queries, train_clips, dedup) -> tuple[list, int]:
    if dedup is None:
        return list(queries), 0
    bad = {dedup.group(c.clip_id) for c in train_clips}
    keep = [c for c in queries if dedup.group(c.clip_id) not in bad]
    return keep, len(queries) - len(keep)


# --------------------------------------------------------------------------------------------- E1 / E3

def check_classifier_split(meta: dict, split_name: str) -> None:
    if meta.get("split") and meta.get("split") != split_name:
        raise LeakageError(f"classifier was trained on split {meta.get('split')!r}, evaluation asked for "
                           f"{split_name!r}: its test clips may be training clips of that model")


def e1_closed_set(rec, clips, label: str, split_name: str, kind: str) -> dict:
    """Classifier alone on ``clips``; decision = classifier probability vs the fused balanced threshold."""
    from sklearn.metrics import f1_score

    if rec.classifier is None:
        return {"label": label, "run": False, "reason": "no classifier"}
    check_classifier_split(rec.classifier.meta if hasattr(rec.classifier, "meta") else {}, split_name)
    ids = list(rec.classifier.class_names)
    pos = {c: i for i, c in enumerate(ids)}
    sel = [c for c in clips if c.concept_id in pos]
    P, used = [], []
    for c in sel:
        try:
            probs, _ = rec.classifier.predict(_features(c, rec.feature_config))
        except (FeatureError, OSError, ValueError):
            continue
        P.append(probs.mean(0))
        used.append(c)
    if not used:
        return {"label": label, "run": False, "reason": "no test clips of the classifier's classes", "n": 0}
    P = np.stack(P)
    y = np.array([pos[c.concept_id] for c in used])
    pred = P.argmax(1)
    hit = (pred == y).astype(float)
    top5 = np.array([y[j] in np.argsort(-P[j])[:5] for j in range(len(y))], float)
    groups = _groups(used)
    thr = float(rec.fusion.accept.get("balanced", 0.7))
    p1 = P.max(1)
    acc = p1 >= thr
    per_cat = defaultdict(list)
    for c, h in zip(used, hit):
        per_cat[c.category or "other"].append(h)
    return {"label": label, "kind": kind, "run": True, "n": len(used), "n_classes": len(set(y.tolist())),
            "n_groups": len(set(groups)),
            "metrics": {"top1": float(hit.mean()), "top5": float(top5.mean()),
                        "macro_f1": float(f1_score(y, pred, labels=sorted(set(y.tolist())), average="macro",
                                                   zero_division=0)),
                        "coverage": float(acc.mean()), "not_sure_rate": float(1 - acc.mean()),
                        "wrong_accept_rate": float((acc & (hit == 0)).mean())},
            "decision_note": f"classifier alone: accept when its probability >= the fused balanced threshold {thr:.2f}; "
                             "the app uses the fused decision (E3)",
            "intervals": {"top1": grouped_bootstrap(hit.tolist(), groups),
                          "top5": grouped_bootstrap(top5.tolist(), groups)},
            "interval_note": None if len(set(groups)) >= 2 else
            "one signer group in this test part: a signer-level interval needs the cross-validation folds (cv)",
            "_hits": hit.tolist(), "_groups": groups,
            "per_category_top1": {k: round(float(np.mean(v)), 4) for k, v in sorted(per_cat.items())}}


def e3_fused(rec, clips, label: str) -> dict:
    from app.isl.training.fit_fusion import collect_evidence

    inputs, truths, used = collect_evidence(rec, clips)
    if not used:
        return {"label": label, "run": False, "n": 0, "reason": "no usable clips"}
    results = [fuse(i, rec.fusion) for i in inputs]
    groups = _groups(used)
    block = decision_block(results, truths, rec.fusion, groups)
    p1 = np.array([r.probs[0] if len(r.probs) else 0.0 for r in results])
    ok = np.array([bool(r.keys) and r.keys[0] == t for r, t in zip(results, truths)])
    order = np.argsort(-p1)
    curve = []
    for frac in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        k = max(1, int(round(frac * len(order))))
        curve.append({"coverage": frac, "risk": float(1 - ok[order[:k]].mean())})
    by_src = defaultdict(list)
    for c, h in zip(used, ok):
        by_src[c.source].append(float(h))
    return {"label": label, "run": True, "n": len(used), "n_groups": len(set(groups)),
            "n_classes": len(set(truths)), "fusion_fitted": rec.fusion.fitted,
            "metrics": _metrics_from(block), **block, "risk_coverage": curve,
            "aurc": float(np.mean([c["risk"] for c in curve])),
            "top1_by_source": {k: {"n": len(v), "top1": float(np.mean(v))} for k, v in sorted(by_src.items())}}


# --------------------------------------------------------------------------------------------- E2

def heldout_set(linker, concepts) -> set[str]:
    return linker.expand(concepts) if linker is not None else set(concepts)


def check_encoder_exclusion(meta: dict, held: set[str]) -> None:
    excl = set(meta.get("excluded_concepts") or [])
    missing = held - excl
    if missing:
        raise LeakageError(f"encoder {meta.get('encoder_version')} was trained with {len(missing)} held-out "
                           f"concepts in its supervised data (e.g. {sorted(missing)[:5]})")


def _retrieval_inputs(enc, feats_list, g_emb, g_concepts, top_n: int = 50) -> list[FusionInputs]:
    labs = sorted(set(g_concepts))
    pos = {c: i for i, c in enumerate(labs)}
    gi = np.array([pos[c] for c in g_concepts])
    order = np.argsort(gi, kind="stable")
    starts = np.flatnonzero(np.r_[True, gi[order][1:] != gi[order][:-1]])
    out = []
    for f in feats_list:
        z, _ = enc.embed(f)
        s = z @ g_emb.T
        m = np.maximum.reduceat(s[:, order], starts, axis=1)
        out.append(FusionInputs(ret=m.astype(np.float32), ret_ids=labs))
    return out


def e2_oneshot_fold(enc, fold: dict, data, linker, fcfg, base_params: FusionParams, restricted: set[str],
                    train_clips_of_encoder: list) -> dict:
    """One fold of the held-out-concept protocol."""
    held = heldout_set(linker, fold["held_out_concepts"])
    check_encoder_exclusion(enc.meta, held)
    # gallery: every ISLRTC reference (one clip per dictionary entry); embedded with this fold's encoder
    gal = [c for c in data.islrtc if c.concept_id]
    g_feats, g_ids = [], []
    for c in gal:
        try:
            g_feats.append(build_features(c.load(), fcfg))
            g_ids.append(c.concept_id)
        except (FeatureError, OSError, ValueError):
            continue
    if not g_feats:
        return {"fold": fold["fold"], "run": False, "reason": "empty gallery"}
    G = np.concatenate([enc.embed(np.stack(g_feats[a:a + 128]))[0] for a in range(0, len(g_feats), 128)])
    gal_concepts = set(g_ids)
    # queries: INCLUDE clips of the fold's concepts from the val / test parts only - those are also excluded from
    # self-supervised pre-training, so a query was never seen in any form (train-part clips were, unlabelled)
    inc_eval = data.include_part("val") + data.include_part("test")
    queries = [c for c in inc_eval if c.concept_id in set(fold["held_out_concepts"]) and c.concept_id in gal_concepts]
    queries, dropped = drop_trained_duplicates(queries, train_clips_of_encoder, getattr(data, "dedup", None))
    calib = [c for c in data.include_part("val") if c.concept_id not in held and c.concept_id in gal_concepts]
    seen = [c for c in data.include_part("test") if c.concept_id not in held and c.concept_id in gal_concepts]

    def run(clips, ids=None):
        feats, used = [], []
        for c in clips:
            try:
                feats.append(_features(c, fcfg))
                used.append(c)
            except (FeatureError, OSError, ValueError):
                continue
        if ids is not None:
            keep = [i for i, g in enumerate(g_ids) if g in ids]
            inputs = _retrieval_inputs(enc, feats, G[keep], [g_ids[i] for i in keep])
        else:
            inputs = _retrieval_inputs(enc, feats, G, g_ids)
        return inputs, [c.concept_id for c in used], used

    ci, ct, _ = run(calib)
    params = base_params.with_(alpha=0.0, gamma=0.0, beta=1.0)
    if ci:
        params, _ = fit_thresholds([fuse(i, params) for i in ci], ct, params)
    out = {"fold": fold["fold"], "run": True, "held_out_concepts": len(fold["held_out_concepts"]),
           "held_out_expanded": len(held), "gallery_rows": len(g_ids), "gallery_concepts": len(gal_concepts),
           "dropped_duplicate_queries": dropped, "calibration_queries": len(ci)}
    for name, ids in (("full", None), ("restricted", restricted & gal_concepts)):
        qi, qt, qu = run(queries, ids)
        if not qi:
            out[name] = {"n": 0}
            continue
        res = [fuse(i, params) for i in qi]
        groups = _groups(qu)
        hits = {k: [float(t in r.keys[:k]) for r, t in zip(res, qt)] for k in (1, 5, 10)}
        block = decision_block(res, qt, params, groups)
        out[name] = {"n": len(qi), "n_groups": len(set(groups)), "n_classes": len(set(qt)),
                     "gallery_concepts": len(ids) if ids is not None else len(gal_concepts),
                     "metrics": _metrics_from(block, {f"top{k}": float(np.mean(v)) for k, v in hits.items()}),
                     "intervals": {f"top{k}": grouped_bootstrap(v, groups) for k, v in hits.items()},
                     "_hits": hits, "_groups": groups, "_block": block}
    si, st, su = run(seen)
    if si:
        res = [fuse(i, params) for i in si]
        out["seen_class"] = {"n": len(si), "top1": float(np.mean([bool(r.keys) and r.keys[0] == t for r, t in zip(res, st)])),
                             "top5": float(np.mean([t in r.keys[:5] for r, t in zip(res, st)])),
                             "label": "INCLUDE test clips of concepts the fold encoder DID train on (same gallery)"}
    return out


def e2_pool(folds: list[dict]) -> dict:
    out = {}
    for name in ("full", "restricted"):
        parts = [f[name] for f in folds if f.get("run") and f.get(name, {}).get("n")]
        if not parts:
            continue
        hits = {k: sum((p["_hits"][k] for p in parts), []) for k in (1, 5, 10)}
        groups = sum((p["_groups"] for p in parts), [])
        pol = {}
        for p_ in POLICIES:
            vals = [p["_block"]["policies"][p_] for p in parts]
            n = sum(v["n"] for v in vals)
            pol[p_] = {k: (sum((v.get(k) or 0) * v["n"] for v in vals) / n if n else None)
                       for k in ("coverage", "not_sure_rate", "wrong_accept_rate", "suggest_rate")}
        out[name] = {"n": len(groups), "n_groups": len(set(groups)),
                     "metrics": {f"top{k}": float(np.mean(v)) for k, v in hits.items()} | pol["balanced"],
                     "intervals": {f"top{k}": grouped_bootstrap(v, groups) for k, v in hits.items()},
                     "policies": pol}
    for f in folds:
        for name in ("full", "restricted"):
            for k in ("_hits", "_groups", "_block"):
                f.get(name, {}).pop(k, None)
    return out


# --------------------------------------------------------------------------------------------- E4

def e4_my_signs(rec, clips, ks=(1, 3, 5), seed: int = 0, max_groups: int = 40) -> dict:
    """Few-shot enrolment on held-out signers' own takes (no store or disk involved)."""
    from app.isl.models.encoder import FeatureStatsEmbedder

    emb = rec.encoder if rec.encoder is not None else FeatureStatsEmbedder(rec.feature_config)
    by = defaultdict(lambda: defaultdict(list))
    for c in clips:
        if c.concept_id and c.group:
            by[c.group][c.concept_id].append(c)
    rng = np.random.default_rng(seed)
    groups = sorted(by)[:max_groups]
    cache: dict[str, tuple] = {}

    def ev(c):
        if c.clip_id not in cache:
            f = _features(c, rec.feature_config)
            inp, _ = rec.evidence(f, use_my_signs=False)
            cache[c.clip_id] = (inp, emb.embed(f)[0])
        return cache[c.clip_id]

    out = {}
    for k in ks:
        rows = {"base": [], "with": [], "fa": [], "g_hit": [], "g_fa": []}
        for g in groups:
            words = {w: v for w, v in by[g].items() if len(v) >= k + 1}
            if len(words) < 2:
                continue
            names = sorted(words)
            rng.shuffle(names)
            enrolled = set(names[: max(1, len(names) // 2)])
            signs = {}
            try:
                for w in enrolled:
                    takes = list(words[w])
                    rng.shuffle(takes)
                    E = np.stack([ev(t)[1].mean(0) for t in takes[:k]])
                    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
                    if k >= 2:
                        sims = E @ E.T
                        loo = [max(np.delete(sims[i], i)) for i in range(k)]
                        theta = max(float(min(loo)) - 0.05, rec.fusion.user_theta_floor, rec.fusion.s_min)
                    else:
                        theta = 1.0
                    signs[w] = (E, theta, takes[k:])
                tests = [(t, w, True) for w in enrolled for t in signs[w][2]]
                tests += [(t, w, False) for w in names if w not in enrolled for t in words[w]]
            except (FeatureError, OSError, ValueError):
                continue
            uid = [f"{USER_PREFIX}{w}" for w in sorted(signs)]
            for t, w, is_enrolled in tests:
                try:
                    inp, z = ev(t)
                except (FeatureError, OSError, ValueError):
                    continue
                zz = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
                U = np.stack([(zz @ signs[s][0].T).max(1) for s in sorted(signs)], axis=1)
                inp2 = FusionInputs(inp.cls_logp, inp.cls_ids, inp.ret, inp.ret_ids, U.astype(np.float32), uid, inp.prior)
                base = fuse(inp, rec.fusion)
                r = fuse(inp2, rec.fusion)
                dec, _ = decide(r.probs, r.agree, len(r.per_crop_top1), rec.fusion, "balanced", r.best_cos)
                top = r.keys[0] if r.keys else ""
                if r.best_user is not None:  # direct "My signs" match (recogniser rule 7.3)
                    uk, uc = r.best_user
                    if uc >= signs[uk.removeprefix(USER_PREFIX)][1] and uc >= r.best_global_cos + rec.fusion.user_margin:
                        top, dec = uk, "accept"
                pred = top.removeprefix(USER_PREFIX)
                if is_enrolled:
                    rows["base"].append(float(bool(base.keys) and base.keys[0] == w))
                    rows["with"].append(float(pred == w))
                    rows["g_hit"].append(g)
                else:
                    rows["fa"].append(float(top.startswith(USER_PREFIX) and dec == "accept"))
                    rows["g_fa"].append(g)
        if rows["with"]:
            out[f"k{k}"] = {"n": len(rows["with"]), "n_groups": len(set(rows["g_hit"])),
                            "top1_without_enrolment": float(np.mean(rows["base"])),
                            "top1_with_enrolment": float(np.mean(rows["with"])),
                            "gain": float(np.mean(rows["with"]) - np.mean(rows["base"])),
                            "interval_with": grouped_bootstrap(rows["with"], rows["g_hit"]),
                            "n_other_words": len(rows["fa"]),
                            "user_sign_false_accept_rate": float(np.mean(rows["fa"])) if rows["fa"] else None,
                            "interval_false_accept": grouped_bootstrap(rows["fa"], rows["g_fa"])}
    label = ("My signs few-shot: a held-out signer enrols k takes of half of their words; tested on their other "
             "takes (gain over no enrolment) and on the words they did not enrol (false accepts of user signs). "
             "Groups are INCLUDE sessions / estimated signers, not verified identities.")
    if not out:
        return {"label": label, "run": False, "reason": "no test signer has enough repeated takes", "n": 0}
    best = out.get("k3") or next(iter(out.values()))
    return {"label": label, "run": True, "n": best["n"], "n_groups": best["n_groups"],
            "metrics": {"top1_with_enrolment_k3": best["top1_with_enrolment"], "gain_k3": best["gain"],
                        "user_sign_false_accept_rate_k3": best["user_sign_false_accept_rate"]},
            "by_k": out}


# --------------------------------------------------------------------------------------------- E5 / E7 / E8

def e5_from_letters(path: Path) -> dict:
    if not path.exists():
        return {"label": "fingerspelling", "run": False, "reason": "no letter model trained for this profile"}
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
    e5 = meta.get("e5") or {}
    lab = ("Fingerspelling, leave-one-signer-out over the letter-annotated continuous fingerspelling signers "
           f"({len(e5.get('signers') or [])}); letter accuracy on letter frames, words via hold detection + "
           "dictionary decoding. Experimental while fewer than 5 signers exist.")
    if not e5.get("measurable"):
        return {"label": lab, "run": False, "reason": e5.get("reason", "not measured"), "data": meta.get("data")}
    return {"label": lab, "run": True, "n": e5.get("n"), "n_groups": len(e5.get("signers") or []),
            "metrics": e5.get("metrics"), "intervals": e5.get("intervals"), "per_letter": e5.get("per_letter"),
            "words": e5.get("words"), "data": meta.get("data"),
            "note": "spelled words are always 'suggest' in the app (never auto-accepted): wrong-accept rate 0 by design"}


def e7_asl_regression(repo_root: Path) -> dict:
    p = repo_root / "models" / "wlasl100-gru-signer-holdout" / "results.json"
    if not p.exists():
        return {"label": "ASL regression", "run": False, "reason": f"{p} missing"}
    try:
        diff = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--",
                               "models/wlasl100-gru-signer-holdout"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        diff = None
    return {"label": "ASL regression: bundled WLASL100 model files unchanged (the full backend test suite checks "
                     "behaviour)", "run": True, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "unchanged_in_git": (diff == "") if diff is not None else None,
            "metrics": {"passed": 1.0 if diff == "" else 0.0}}


def e8_trim(paths, artifacts_dir: Path) -> dict:
    from app.isl.data.manifest import read_manifest

    rep = artifacts_dir / "reports" / "trim_eval.json"
    trims = read_manifest(paths.trim_sidecar, only_ok=False) if paths.trim_sidecar.exists() else []
    mix = Counter(r.get("method") for r in trims)
    out = {"label": "ISLRTC first-sign trim quality (synthetic end-error check, INCLUDE active-frame check) "
                    "from 'cli trim-eval', plus the method mix of the trim sidecar", "run": bool(trims) or rep.exists(),
           "trims": len(trims), "method_mix": dict(mix),
           "low_confidence_frac": (sum(1 for r in trims if float(r.get("confidence") or 0) < 0.5) / len(trims))
           if trims else None}
    if trims:
        out["metrics"] = {"trims": len(trims), "low_confidence_frac": out["low_confidence_frac"],
                          "cap_frac": mix.get("cap", 0) / len(trims)}
    if rep.exists():
        try:
            out["trim_eval"] = json.loads(rep.read_text())
        except ValueError:
            pass
    return out


# --------------------------------------------------------------------------------------------- driver

def evaluate_all(work: Path, data, fcfg, *, profile: str, smoke: bool, vocab_path: Path | None = None,
                 paths=None, out_dirs: list[Path] | None = None, e4_ks=(1, 3, 5), latency_n: int = 20,
                 progress=None) -> dict:
    """Run every protocol whose components exist in ``work`` and write ``eval/report.json`` + ``report.md``."""
    from app.core.config import REPO_ROOT, get_settings
    from app.isl.evaluation.protocols import e6_latency
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.pipeline import load_recognizer
    from app.isl.training.train_encoder import encoder_training_clips

    log = progress or (lambda m: None)
    t0 = time.time()
    split_name = data.split.get("name")
    rec = load_recognizer(work, fcfg, vocab_path, profile=profile)
    prot: dict = {}
    si = data.split.get("signer_independent")
    lab = data.split.get("label") or split_name
    test_inc = data.include_part("test")
    # E1b / E1c / E3 on the primary split's test part
    log("E1b")
    prot["E1b"] = e1_closed_set(rec, test_inc, f"E1b core words, closed set, INCLUDE test part of {split_name} ({lab}; "
                                f"signer-independent: {si})", split_name, "primary")
    cv = sorted((work / "cv").glob("fold*/classifier.pt")) if (work / "cv").exists() else []
    if cv:
        from app.isl.training.common import load_split
        from app.models.checkpoint import Predictor

        folds = []
        for p in cv:
            pr = Predictor(p)
            sname = pr.meta.get("split")
            if not sname or sname == split_name:  # the main model is already E1b
                continue
            try:
                sp = load_split(paths, sname) if paths is not None else None
            except FileNotFoundError:
                continue
            if sp is None:
                continue
            fold_test = [c for c in data.include if sp["assign"].get(c.clip_id) == "test" and c.concept_id]
            r2 = load_recognizer(work, fcfg, vocab_path, profile=profile)
            r2.classifier = pr
            folds.append(e1_closed_set(r2, fold_test, sname, sname, "cv"))
        runs = [f for f in [prot["E1b"], *folds] if f.get("run")]
        tops = [f["metrics"]["top1"] for f in runs]
        if len(runs) > 1:
            hits = sum((f["_hits"] for f in runs), [])
            grp = sum(([f"{i}:{g}" for g in f["_groups"]] for i, f in enumerate(runs)), [])
            prot["E1b"]["cv"] = {"folds": len(runs), "top1_mean": float(np.mean(tops)), "top1_sd": float(np.std(tops)),
                                 "pooled_n": len(hits), "pooled_top1": float(np.mean(hits)),
                                 "pooled_top1_interval": grouped_bootstrap(hits, grp),
                                 "pooled_groups": len(set(grp)),
                                 "label": "every fold's held-out signers, each fold's model trained without them",
                                 "per_fold": [{"split": f["label"], "n": f.get("n"), "top1": f["metrics"]["top1"]}
                                              for f in runs]}
            prot["E1b"]["intervals"]["top1_cv_pooled"] = prot["E1b"]["cv"]["pooled_top1_interval"]
    off = work / "e1a" / "classifier.pt"
    if off.exists() and paths is not None:
        from app.isl.training.common import load_split
        from app.models.checkpoint import Predictor

        log("E1a")
        try:
            sp = load_split(paths, "include-official")
            r2 = load_recognizer(work, fcfg, vocab_path, profile=profile)
            r2.classifier = Predictor(off)
            prot["E1a"] = e1_closed_set(r2, [c for c in data.include if sp["assign"].get(c.clip_id) == "test"
                                             and c.concept_id],
                                        "E1a INCLUDE official split: NOT signer-independent (same recording sessions "
                                        "in train and test); shown only next to E1b", "include-official", "official")
        except FileNotFoundError:
            prot["E1a"] = {"run": False, "reason": "include-official split missing"}
    else:
        prot["E1a"] = {"run": False, "reason": "no classifier trained on the official split (work/e1a/classifier.pt)"}
    extra_test = data.extra_part("test")
    if extra_test:
        log("E1c")
        prot["E1c"] = e1_closed_set(rec, extra_test, "E1c other sources' held-out signers (real signer ids: "
                                    + ", ".join(sorted({c.source for c in extra_test})) + "); signer-independent",
                                    split_name, "extra_signers")
    else:
        prot["E1c"] = {"run": False, "reason": "no other labelled source has a signer-held-out test part yet (needs a "
                       "source with real signer ids and >= 6 signers, e.g. ISL500)"}
    log("E3")
    prot["E3"] = e3_fused(rec, test_inc + extra_test, "E3 fused system (classifier + dictionary retrieval + fusion "
                          "fitted on validation) with accept / suggest / abstain; INCLUDE test part + other sources' "
                          "held-out signers")
    # E2 held-out-concept one-shot
    one = (paths.splits / "oneshot_concepts.json") if paths is not None else None
    fold_models = sorted((work / "oneshot").glob("fold*/encoder.pt")) if (work / "oneshot").exists() else []
    if one is not None and one.exists() and fold_models:
        log("E2")
        doc = json.loads(one.read_text())
        restricted = {c["concept_id"] for c in doc.get("concepts", []) if c.get("usable")}
        folds = []
        for p in fold_models:
            k = int(p.parent.name.removeprefix("fold"))
            fd = next((f for f in doc["protocol"]["folds"] if f["fold"] == k), None)
            if fd is None:
                continue
            enc = EncoderPredictor(p)
            held = heldout_set(getattr(data, "linker", None), fd["held_out_concepts"])
            try:
                folds.append(e2_oneshot_fold(enc, fd, data, getattr(data, "linker", None), fcfg, rec.fusion,
                                             restricted, encoder_training_clips(data, held)))
            except LeakageError as e:
                folds.append({"fold": k, "run": False, "leakage": str(e)})
        pooled = e2_pool(folds)
        full = pooled.get("full", {})
        prot["E2"] = {"label": "E2 open vocabulary, one-shot: concepts held out of ALL supervised training (synonyms "
                               "too); other signers' INCLUDE clips (val/test parts, never seen even unlabelled) vs one "
                               "ISLRTC dictionary clip per word; thresholds "
                               "fitted on seen-class validation queries only. Some misses are regional variants, not "
                               "model errors. Unlabelled pre-training may have seen dictionary clips (no labels).",
                      "run": bool(full), "n": full.get("n", 0), "n_groups": full.get("n_groups"),
                      "metrics": full.get("metrics", {}), "intervals": full.get("intervals", {}),
                      "restricted_gallery": pooled.get("restricted"), "folds": folds}
    else:
        prot["E2"] = {"run": False, "reason": "no held-out-concept encoders (work/oneshot/fold*/encoder.pt) or no "
                      "oneshot_concepts.json; the seen-class retrieval number is in E2_seen"}
    from app.isl.evaluation.protocols import e2_retrieval

    e2s = e2_retrieval(rec, test_inc, "E2_seen: INCLUDE test clips vs ISLRTC rows only; SEEN classes (the encoder "
                       "trained on these concepts' other clips) - not the open-vocabulary number")
    if e2s:
        prot["E2_seen"] = e2s
    log("E4")
    prot["E4"] = e4_my_signs(rec, test_inc + extra_test, ks=e4_ks)
    prot["E5"] = e5_from_letters(work / "letters.pt")
    log("E6")
    prot["E6"] = e6_latency(rec, test_inc + extra_test, n=latency_n)
    prot["E7"] = e7_asl_regression(REPO_ROOT)
    if paths is not None:
        prot["E8"] = e8_trim(paths, get_settings().artifacts_dir)
    for v in prot.values():
        for k in ("_hits", "_groups"):
            v.pop(k, None)
    leak = {"split": split_name, "classifier_split_checked": True, "dedup": getattr(data, "multisource", {}).get("dedup"),
            "links": getattr(data, "multisource", {}).get("links"),
            "pretraining_excludes_eval_clips": True}
    doc = {"schema": 2, "language": "isl", "profile": profile, "smoke": smoke,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": git_commit(),
           "label": ("SMOKE RUN on partial data with tiny settings: proves the pipeline runs end to end. These numbers "
                     "are NOT the accuracy of the app." if smoke else
                     "Measured on held-out data (test parts used once, after all selection on validation). Live "
                     "webcam accuracy is expected to be lower."),
           "data": data.summary(), "leakage_guards": leak, "protocols": prot, "seconds": round(time.time() - t0, 1)}
    md = markdown_report(doc)
    for d in [work / "eval", *(out_dirs or [])]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=float))
        (d / "report.md").write_text(md)
    return doc


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def markdown_report(doc: dict) -> str:
    lines = [f"# ISL evaluation report ({doc['profile']})", "", f"**{doc['label']}**", "",
             f"Created {doc['created']} at commit {doc.get('git_commit')}. Intervals: 95 % bootstrap over signer "
             "groups. Not sure = abstain rate; wrong accept = accepted and wrong / all queries.", "",
             "| Protocol | n | groups | top-1 | 95% CI | top-5 | not sure | wrong accept | label |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pid, p in doc["protocols"].items():
        if p.get("run") is False or (not p.get("metrics") and p.get("run") is None and not p.get("n")):
            lines.append(f"| {pid} | - | - | - | - | - | - | - | not run: {p.get('reason') or p.get('leakage') or ''} |")
            continue
        m = p.get("metrics") or {}
        if "top1" not in m:
            lines.append(f"| {pid} | {p.get('n', '-')} | {p.get('n_groups', '-')} | - | - | - | - | - | "
                         f"{p.get('label', '')} (details below) |")
            continue
        iv = p.get("intervals") or {}
        ci, tag = iv.get("top1"), ""
        if not ci and iv.get("top1_cv_pooled"):
            ci, tag = iv["top1_cv_pooled"], " (cv pooled)"
        lines.append(f"| {pid} | {p.get('n')} | {p.get('n_groups', '-')} | {_fmt(m.get('top1'))} | "
                     f"{'-' if not ci else f'{ci[0]:.3f}-{ci[1]:.3f}{tag}'} | {_fmt(m.get('top5'))} | "
                     f"{_fmt(m.get('not_sure_rate'))} | {_fmt(m.get('wrong_accept_rate'))} | {p.get('label', '')} |")
    cvb = doc["protocols"].get("E1b", {}).get("cv")
    if cvb:
        ci = cvb.get("pooled_top1_interval")
        lines += ["", f"E1b cross-validation: {cvb['folds']} folds, top-1 mean {cvb['top1_mean']:.3f} (sd "
                  f"{cvb['top1_sd']:.3f}); pooled over {cvb['pooled_groups']} held-out signer groups: "
                  f"{cvb['pooled_top1']:.3f}" + (f" (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else "")]
    e4 = doc["protocols"].get("E4", {})
    if e4.get("run"):
        lines += ["", "## E4 My signs", "", "| k | n | without | with | gain | user-sign false accepts |", "|---|---|---|---|---|---|"]
        for k, v in e4["by_k"].items():
            lines.append(f"| {k} | {v['n']} | {_fmt(v['top1_without_enrolment'])} | {_fmt(v['top1_with_enrolment'])} | "
                         f"{_fmt(v['gain'])} | {_fmt(v['user_sign_false_accept_rate'])} |")
    for k in ("E5", "E6", "E7", "E8"):
        p = doc["protocols"].get(k)
        if p:
            lines += ["", f"## {k}", "", "```", json.dumps({x: y for x, y in p.items() if x not in ("trim_eval",)},
                                                           indent=1, default=str)[:1500], "```"]
    return "\n".join(lines) + "\n"
