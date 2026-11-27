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
E4   My signs few-shot: enrol k (3, 5) takes of a test signer's word, test on their other takes with the app's own
     decision rule; false accepts of user signs on words they did not enrol. Headline: INCLUDE test signers only
     (comparable across profiles); ``all_sources`` adds the other sources' held-out signers (docs 15.27)
E5   fingerspelling, leave-one-group-out over signers and Mendeley age groups (computed with the letter model)
E6   latency per segment
E7   ASL regression: the bundled ASL model's results file is unchanged
E8   trim quality (``cli trim-eval`` report of the run's trimmer, version-checked, + the trim method mix)

Leakage guards: E1 refuses a classifier trained on another split; E2 refuses an encoder whose excluded
concepts do not cover the (synonym-expanded) held-out set; queries in a duplicate group of any training clip
(same video hash, same landmark file, landmark near-duplicate) are dropped and counted; the retrieval index must be
the training-only ``work/index`` of this very encoder file (never ``index_final``; docs 15.21).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.features.build import FeatureError, build_features
from app.isl.evaluation.report import (PARTIAL_DATA_LABEL, format_interval, git_commit, grouped_bootstrap,
                                       interval_fields, interval_reason)
from app.isl.models.calibration import (_decisions, classifier_alone_decisions, fit_thresholds, is_major_size,
                                        summarize)
from app.isl.models.fusion import POLICIES, USER_PREFIX, FusionInputs, FusionParams, decide, fuse
from app.isl.models.index import LeakageError  # noqa: F401  (one class for every leak guard; re-exported here)
from app.isl.training.common import engine_crops

E1_POLICY = "balanced"  # E1's summary columns: the sliding mode at the app's default policy (all in "policies")
DECISION_KEYS = ("top1", "top5", "coverage", "accepted_precision", "not_sure_rate", "wrong_accept_rate",
                 "suggest_rate", "suggest_top5_hit")


# --------------------------------------------------------------------------------------------- helpers

def _rate(mask, sel=None):
    m = np.asarray(mask, bool) if sel is None else np.asarray(mask, bool)[sel]
    return float(m.mean()) if len(m) else None


def source_rows(dec: list[str], udec: list[str], hit1: list[float], sources: list, groups: list) -> tuple[dict, str | None]:
    """Per source (docs 15.25): what the app does (``dec``: accept / suggest / abstain with the pack's gates) and what
    the threshold rule alone does (``udec``, the top candidate a confirm tap shows). The worst source is the one with
    the lowest threshold-rule precision among sources big enough to be certified (``is_major_size``: >= 100 queries
    and >= 2 groups, docs 15.26; None when no source qualifies). Smaller sources get rows, not the headline."""
    src = np.array([str(x) if x not in (None, "") else "unknown" for x in sources], dtype=object)
    grp = np.array([str(g) for g in groups], dtype=object)
    h = np.asarray(hit1, float) > 0.5
    a = np.array([d == "accept" for d in dec], bool)
    u = np.array([d == "accept" for d in udec], bool)
    rows, cands = {}, []
    for s_ in sorted(set(src)):
        m = src == s_
        n, ng = int(m.sum()), len(set(grp[m]))
        rows[s_] = {"n": n, "n_groups": ng, "top1": _rate(h, m), "coverage": _rate(a, m),
                    "accepted_precision": _rate(h[m & a]) if (m & a).any() else None,
                    "wrong_accept_rate": _rate(a & ~h, m),
                    "not_sure_rate": _rate(np.array([d == "abstain" for d in dec], bool), m),
                    "suggest_rate": _rate(np.array([d == "suggest" for d in dec], bool), m),
                    "threshold_coverage": _rate(u, m),
                    "threshold_precision": _rate(h[m & u]) if (m & u).any() else None,
                    "threshold_wrong_accept_rate": _rate(u & ~h, m), "threshold_n_accepted": int((m & u).sum())}
        if is_major_size(n, ng):
            cands.append(s_)
        else:
            rows[s_]["note"] = "too small to certify: not the worst-source headline"
    worst = min(cands, key=lambda x: (rows[x]["threshold_precision"] if rows[x]["threshold_precision"] is not None
                                      else 2.0, x), default=None)
    return rows, worst


def decision_block(results, truths: list[str], params: FusionParams, groups: list, sources: list | None = None,
                   closed_set=None) -> dict:
    """Per policy: accuracy with coverage, not-sure (abstain) and wrong-accept rates; grouped intervals. With a
    pack's (gated) parameters a confirm-only policy accepts nothing (docs 15.25): ``rule`` says which, and
    ``threshold_rule`` gives what its threshold alone does (the top candidate the user is asked to confirm).
    ``sources``: per-source rows (``by_source``) and the worst source per policy. ``closed_set``: the classifier's
    classes, for the open-vocabulary gate."""
    out = {}
    hit1 = [float(bool(r.keys) and r.keys[0] == t) for r, t in zip(results, truths)]
    hit5 = [float(t in r.keys[:5]) for r, t in zip(results, truths)]
    ungated = params.with_(certification=None)
    for pol in POLICIES:
        s = summarize(results, truths, params, pol, closed_set)
        dec = [d for d, _ in _decisions(results, params, pol, closed_set)]
        udec = [d for d, _ in _decisions(results, ungated, pol)] if params.gated else dec
        wrong_acc = [float(d == "accept" and not h) for d, h in zip(dec, hit1)]
        not_sure = [float(d == "abstain") for d in dec]
        ua = np.array([d == "accept" for d in udec], bool)
        hh = np.asarray(hit1, float) > 0.5
        out[pol] = {**s, "not_sure_rate": s.get("abstain_rate"), "wrong_accept_rate": s.get("silent_error_rate"),
                    "rule": params.policy_rule(pol),
                    "threshold_rule": {"coverage": _rate(ua), "n_accepted": int(ua.sum()),
                                       "accepted_precision": _rate(hh[ua]) if ua.any() else None,
                                       "wrong_accept_rate": _rate(ua & ~hh)},
                    **interval_fields({"wrong_accept_rate": (wrong_acc, groups), "not_sure_rate": (not_sure, groups)})}
        if sources is not None:
            out[pol]["by_source"], out[pol]["worst_source"] = source_rows(dec, udec, hit1, sources, groups)
    return {"policies": out, **interval_fields({"top1": (hit1, groups), "top5": (hit5, groups)})}


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

def check_classifier_split(meta: dict | None, split_name: str) -> None:
    """Refuse a classifier trained on another split - or one whose checkpoint does not say which split it was
    trained on (then nobody can tell whether the test clips were its training clips)."""
    trained = (meta or {}).get("split")
    if not trained:
        raise LeakageError(f"classifier checkpoint has no 'split' in its meta: cannot show that the test clips of "
                           f"{split_name!r} were not its training clips (retrain it with this code)")
    if trained != split_name:
        raise LeakageError(f"classifier was trained on split {trained!r}, evaluation asked for "
                           f"{split_name!r}: its test clips may be training clips of that model")


def classifier_refusal(classifier, split_name: str) -> str | None:
    """Why a recogniser's classifier may not be evaluated on ``split_name`` (None = allowed, or no classifier)."""
    if classifier is None:
        return None
    try:
        check_classifier_split(getattr(classifier, "meta", None), split_name)
    except LeakageError as e:
        return str(e)
    return None


def refused_entry(label: str, why: str, kind: str | None = None) -> dict:
    """Protocol entry of a run refused by a leakage guard (reported instead of a number)."""
    return {"label": label, **({"kind": kind} if kind else {}), "run": False, "n": 0, "leakage": why,
            "reason": f"refused: {why}"}


def _bits(x) -> str:
    """0/1 per clip as one compact string (report.json stays small and readable)."""
    return "".join("1" if v else "0" for v in np.asarray(x, bool))


def _parse_utc(stamp) -> float | None:
    """A report's ``created`` (``%Y-%m-%dT%H:%M:%SZ``, UTC) as epoch seconds, or None."""
    import calendar

    try:
        return float(calendar.timegm(time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def load_e1_reference(path: Path) -> dict:
    """The E1b of another evaluation report (``--compare-to``, docs 15.27): its split, evaluated clip ids, per-clip
    top-1 / top-5 hits and the classifier's class ids. Reports written before 15.27 have none of these: the class
    ids then come from the classifier checkpoint next to the report (``<report dir>/../classifier.pt``, the layout of
    a work folder and of a pack) and the clip-level rows are not possible (``reason`` says so)."""
    path = Path(path)
    doc = json.loads(path.read_text())
    e1 = (doc.get("protocols") or {}).get("E1b") or {}
    ref = {"path": str(path), "profile": doc.get("profile"), "split": e1.get("split")
           or (doc.get("leakage_guards") or {}).get("split"), "n": e1.get("n"),
           "top1": (e1.get("metrics") or {}).get("top1"), "top5": (e1.get("metrics") or {}).get("top5"),
           "clip_ids": e1.get("clip_ids"), "hit1": (e1.get("per_clip") or {}).get("hit1"),
           "hit5": (e1.get("per_clip") or {}).get("hit5"), "class_ids": e1.get("class_ids"), "notes": []}
    if not ref["split"]:
        ref["notes"].append("the reference report records no split: it could not be verified that it was evaluated "
                            "on the same split")
    if not ref["class_ids"]:
        ck = path.parent.parent / "classifier.pt"
        if ck.exists():
            import torch

            meta = (torch.load(ck, map_location="cpu", weights_only=False) or {}).get("meta") or {}
            ck_split = meta.get("split")
            if ref["split"] and ck_split and ck_split != ref["split"]:
                # not the checkpoint that produced the report: its classes would mask the wrong rows (docs 15.28)
                ref["notes"].append(f"class ids NOT read from {ck}: it was trained on split {ck_split!r}, the report "
                                    f"was evaluated on {ref['split']!r} (the pack was rebuilt after the report)")
            else:
                ref["class_ids"] = list(meta.get("class_names") or []) or None
                ref["notes"].append(f"class ids read from {ck} (the report has none: written before docs 15.27)")
                created = _parse_utc(doc.get("created"))
                if created is None:
                    ref["notes"].append(f"WARNING: the report has no creation time, so {ck} could not be checked "
                                        "against it (it may be a newer checkpoint with other classes)")
                elif ck.stat().st_mtime > created + 60:
                    ref["notes"].append(f"WARNING: {ck} is newer than the report ({doc.get('created')}): it may not "
                                        "be the checkpoint that produced it, so the masked rows may use the wrong "
                                        "class set")
                if not ck_split:
                    ref["notes"].append(f"{ck} records no split: its class set could not be checked")
        else:
            ref["notes"].append("the report has no class ids and there is no classifier.pt next to it")
    if not ref["clip_ids"]:
        ref["notes"].append("the report has no clip ids (written before docs 15.27): re-run evaluate on the reference "
                            "for the common-clip rows")
    return ref


def e1_like_for_like(P: np.ndarray, y: np.ndarray, ids: list[str], used, hit: np.ndarray, top5: np.ndarray,
                     split_name: str, ref: dict, word_hits: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """E1b against a reference report (docs 15.27): top-1 / top-5 of THIS classifier restricted to the reference's
    class set (scores of other classes masked, only clips whose word is a reference class) and to the clip ids both
    reports evaluated, with the reference's own numbers on the same clips. No alias scoring: strict labels only.
    ``word_hits``: (top-1, top-5) per clip on word labels (variant concepts counted as their word, docs 15.37); the
    common-clip row is then also given on word labels (``common_clips_word_labels``)."""
    out = {"reference": {k: ref.get(k) for k in ("path", "profile", "split", "n", "top1", "top5")},
           "notes": list(ref.get("notes") or []), "rule": "strict labels; masked = the reference's classes only"}
    if ref.get("split") and ref["split"] != split_name:
        out["run"] = False
        out["reason"] = f"the reference was evaluated on split {ref['split']!r}, this run on {split_name!r}"
        return out
    out["run"] = True
    clip_ids = [c.clip_id for c in used]

    def rates(sel, hit_, top5_):
        sel = np.asarray(sel, bool)
        if not sel.any():
            return {"n": 0, "top1": None, "top5": None}
        return {"n": int(sel.sum()), "top1": float(hit_[sel].mean()), "top5": float(top5_[sel].mean())}

    rc = set(ref.get("class_ids") or [])
    m_hit = m_top5 = in_ref = None
    if rc:
        keep = np.array([c in rc for c in ids], bool)
        in_ref = np.array([ids[t] in rc for t in y], bool)
        Pm = np.where(keep[None, :], P, -np.inf)
        order = np.argsort(-Pm, axis=1)
        m_hit = (order[:, 0] == y).astype(float)
        m_top5 = np.array([y[j] in order[j, :5] for j in range(len(y))], float)
        out["reference_classes"] = {**rates(in_ref, m_hit, m_top5), "n_reference_classes": len(rc),
                                    "n_shared_classes": int(keep.sum())}
    else:
        out["reference_classes"] = {"run": False, "reason": "no reference class ids"}
    rid = ref.get("clip_ids") or []
    if rid:
        pos = {c: i for i, c in enumerate(rid)}
        common = np.array([c in pos for c in clip_ids], bool)
        row = rates(common, hit, top5)
        h1, h5 = ref.get("hit1") or "", ref.get("hit5") or ""
        ref_idx = [pos[c] for c, k in zip(clip_ids, common) if k]
        if len(h1) == len(rid):
            r1 = np.array([h1[i] == "1" for i in ref_idx], bool)
            own = hit[common] > 0.5
            row["reference_top1"] = float(r1.mean()) if len(r1) else None
            row["discordant_top1"] = {"only_this": int((own & ~r1).sum()), "only_reference": int((~own & r1).sum())}
        if len(h5) == len(rid):
            r5 = np.array([h5[i] == "1" for i in ref_idx], bool)
            row["reference_top5"] = float(r5.mean()) if len(r5) else None
        out["common_clips"] = row
        if word_hits is not None:
            w1, w5 = (np.asarray(x, float) for x in word_hits)
            roww = {**rates(common, w1, w5), "rule": WORD_RULE}
            if len(h1) == len(rid):
                own_w = w1[common] > 0.5
                r1 = np.array([h1[i] == "1" for i in ref_idx], bool)
                roww["reference_top1"] = float(r1.mean()) if len(r1) else None
                roww["discordant_top1"] = {"only_this": int((own_w & ~r1).sum()),
                                           "only_reference": int((~own_w & r1).sum())}
            if len(h5) == len(rid):
                r5 = np.array([h5[i] == "1" for i in ref_idx], bool)
                roww["reference_top5"] = float(r5.mean()) if len(r5) else None
            out["common_clips_word_labels"] = roww
        if m_hit is not None:
            sel = common & in_ref
            row2 = rates(sel, m_hit, m_top5)
            if len(h1) == len(rid):
                r1 = np.array([h1[pos[c]] == "1" for c, k in zip(clip_ids, sel) if k], bool)
                row2["reference_top1"] = float(r1.mean()) if len(r1) else None
            if len(h5) == len(rid):
                r5 = np.array([h5[pos[c]] == "1" for c, k in zip(clip_ids, sel) if k], bool)
                row2["reference_top5"] = float(r5.mean()) if len(r5) else None
            out["reference_classes_common_clips"] = row2
    else:
        out["common_clips"] = {"run": False, "reason": "the reference report has no clip ids"}
    return out


ALIAS_RULE = ("alias labels: a prediction of another member of the clip's alias group counts as correct (classifier "
              "probabilities summed over the group first). Reported next to the strict numbers; every gate, "
              "headline and certification uses STRICT labels (docs 15.34, critic 12)")


WORD_RULE = ("word labels: a prediction of a variant concept (isl.<word>~isl500, docs 15.36) counts as its word; "
             "twins and every other alias stay strict. The variant is the same word and the same output, so these "
             "are the like-for-like numbers for the gates of a variant run (docs 15.37)")


def variant_word_map(link: dict | None) -> dict[str, str]:
    """variant concept -> its word, from the link step's frozen ``variants`` (docs 15.36); empty without variants."""
    return {str(v): str(w) for m in ((link or {}).get("variants") or {}).values() for w, v in (m or {}).items()}


def group_hits(P: np.ndarray, y: np.ndarray, ids: list[str], alias_of: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per clip top-1 / top-5 hits after summing ``P`` over the groups of ``alias_of`` and ranking groups."""
    from app.isl.vocab.aliases import alias_root, sum_by_alias

    roots, Pr = sum_by_alias(P, ids, alias_of)
    rpos = {r: i for i, r in enumerate(roots)}
    ry = np.array([rpos[alias_root(ids[t], alias_of)] for t in y], int)
    order = np.argsort(-np.asarray(Pr), axis=1, kind="stable")
    return order[:, 0] == ry, (order[:, :5] == ry[:, None]).any(1)


def e1_alias_rates(P: np.ndarray, y: np.ndarray, ids: list[str], alias_of: dict, hit: np.ndarray,
                   top5: np.ndarray, rule: str = ALIAS_RULE) -> dict:
    """E1 top-1 / top-5 on alias labels (docs 15.34): the crop-mean probabilities ``P`` (n, classes) summed over each
    alias group, then ranked by group. Strict numbers of the same clips next to them. With ``alias_of`` = the
    variant pairs only (:func:`variant_word_map`) and ``rule`` = ``WORD_RULE``: word labels (docs 15.37)."""
    from app.isl.vocab.aliases import alias_root, sum_by_alias

    roots, _ = sum_by_alias(P, ids, alias_of)
    h1, h5 = group_hits(P, y, ids, alias_of)
    strict1 = np.asarray(hit) > 0.5
    # per group: how many clips of that group change (docs 15.35), so a reader sees where the alias gain comes from
    yroot = [alias_root(ids[t], alias_of) for t in y]
    by_group = {}
    for g in sorted({alias_root(k, alias_of) for k in alias_of}):
        m = np.array([r == g for r in yroot], bool)
        if m.any():
            by_group[g] = {"n": int(m.sum()), "top1_only_alias": int((h1 & ~strict1 & m).sum()),
                           "top1_only_strict": int((~h1 & strict1 & m).sum())}
    return {"rule": rule, "n": int(len(y)), "top1": float(h1.mean()), "top5": float(h5.mean()),
            "top1_strict": float(strict1.mean()), "top5_strict": float(np.mean(top5)),
            "top1_only_alias": int((h1 & ~strict1).sum()), "top1_only_strict": int((~h1 & strict1).sum()),
            "groups_among_classes": int(len(ids) - len(roots)), "by_group": by_group}


def e3_alias_rates(results, truths: list[str], params: FusionParams, alias_of: dict, closed_set=None,
                   rule: str = ALIAS_RULE) -> dict:
    """E3 on alias labels (docs 15.34): the shown word counts as correct when it is in the clip's alias group. The
    decisions are the app's (``params``; with ``params.aliases`` the groups were already summed by ``fuse``)."""
    from app.isl.vocab.aliases import alias_root

    def root(k):
        return alias_root(k, alias_of)

    rt = [root(t) for t in truths]
    h1 = np.array([bool(r.keys) and root(r.keys[0]) == t for r, t in zip(results, rt)], bool)
    s1 = np.array([bool(r.keys) and r.keys[0] == t for r, t in zip(results, truths)], bool)
    h5 = np.array([t in {root(k) for k in r.keys[:5]} for r, t in zip(results, rt)], bool)
    s5 = np.array([t in r.keys[:5] for r, t in zip(results, truths)], bool)
    n = len(results)
    out = {"rule": rule, "n": n, "top1": _rate(h1), "top5": _rate(h5), "top1_strict": _rate(s1),
           "top5_strict": _rate(s5),
           "fusion_sums_groups": bool(getattr(params, "aliases", None)), "policies": {}}
    for pol in POLICIES:
        acc = np.array([d == "accept" for d, _ in _decisions(results, params, pol, closed_set)], bool)
        out["policies"][pol] = {"n_accepted": int(acc.sum()),
                                "accepted_precision": _rate(h1[acc]) if acc.any() else None,
                                "accepted_precision_strict": _rate(s1[acc]) if acc.any() else None,
                                "wrong_accept_rate": _rate(acc & ~h1), "wrong_accept_rate_strict": _rate(acc & ~s1)}
    return out


def e1_closed_set(rec, clips, label: str, split_name: str, kind: str, reference: dict | None = None,
                  alias_of: dict | None = None, variant_of: dict | None = None) -> dict:
    """Classifier alone on ``clips``. Decision columns: the classifier-alone rule of the app's sliding mode, with the
    classifier's own validation temperature and threshold, NOT the fused thresholds of ``fusion.json`` (another
    probability scale: the commercial-safe report of 2026-11-26 compared raw classifier probabilities with the fused
    0.96 and showed "not sure" 0.95-1.0; docs/ISL_DESIGN.md 15.16). The threshold is the sliding mode's for the
    app's default policy (``E1_POLICY``, balanced): fitted per policy by fit-fusion when the pack has it for this
    classifier, else the checkpoint's single threshold (``classifier_policy_thresholds``, docs 15.17). Every policy's
    numbers are in ``policies``. ``alias_of``: also the alias-label top-1 / top-5 in ``alias_labels`` (docs 15.34;
    the strict numbers stay the metrics). ``variant_of``: variant concept -> word (docs 15.37); also the word-label
    numbers in ``word_labels`` (and on the common clips of ``like_for_like``)."""
    from sklearn.metrics import f1_score

    from app.inference.engine import EngineConfig
    from app.isl.recognizer import calibrated_probs, classifier_policy_thresholds

    if rec.classifier is None:
        return {"label": label, "run": False, "reason": "no classifier"}
    why = classifier_refusal(rec.classifier, split_name)
    if why:
        return refused_entry(label, why, kind)
    ids = list(rec.classifier.class_names)
    pos = {c: i for i, c in enumerate(ids)}
    sel = [c for c in clips if c.concept_id in pos]
    cal = classifier_policy_thresholds(rec.classifier, getattr(rec, "fusion", None))
    temp, thr, thr_fitted = cal["temperature"], cal["thresholds"][E1_POLICY], cal["threshold_in_checkpoint"]
    per_crop, used = [], []
    for c in sel:
        try:
            probs, _ = rec.classifier.predict(_features(c, rec.feature_config))
        except (FeatureError, OSError, ValueError):
            continue
        per_crop.append(calibrated_probs(np.atleast_2d(probs), temp))
        used.append(c)
    if not used:
        return {"label": label, "run": False, "reason": "no test clips of the classifier's classes", "n": 0}
    margin = EngineConfig().min_margin
    P, acc = classifier_alone_decisions(per_crop, thr, margin)
    y = np.array([pos[c.concept_id] for c in used])
    pred = P.argmax(1)
    hit = (pred == y).astype(float)
    policies = {}
    for pol in POLICIES:
        _, a = classifier_alone_decisions(per_crop, cal["thresholds"][pol], margin)
        policies[pol] = {"threshold": cal["thresholds"][pol], "coverage": float(a.mean()),
                         "not_sure_rate": float(1 - a.mean()), "wrong_accept_rate": float((a & (hit == 0)).mean()),
                         "accepted_precision": float(hit[a].mean()) if a.any() else None, "n_accepted": int(a.sum())}
    sliding = _e1_sliding_replay(rec, used, cal["thresholds"], kind)
    for pol in POLICIES if sliding else ():
        policies[pol]["sliding_replay"] = sliding[pol]
    srule = getattr(rec, "sliding_rule", None)
    for pol in POLICIES:  # may sliding mode auto-commit with this policy (docs 15.25)
        policies[pol]["rule"] = srule(pol) if callable(srule) else None
    src_note = ("fitted per policy on validation (fusion.json)" if cal["per_policy"] else
                "the checkpoint's validation threshold, the same for every policy"
                + ("" if thr_fitted else " (checkpoint has none: Predictor default)"))
    top5 = np.array([y[j] in np.argsort(-P[j])[:5] for j in range(len(y))], float)
    groups = _groups(used)
    import hashlib as _h

    # which clips and classes these numbers are on (docs 15.27): reports of two packs can then be compared on the
    # same clips (``--compare-to``); the class list only for the main classifier (the reference of such a comparison)
    record = {"split": split_name, "clip_ids": [c.clip_id for c in used],
              "class_ids_sha256": _h.sha256("\n".join(ids).encode()).hexdigest()[:16]}
    if kind in ("primary", "extra_signers"):
        record["per_clip"] = {"hit1": _bits(hit > 0.5), "hit5": _bits(top5 > 0.5)}
    if kind == "primary":
        record["class_ids"] = ids
    word_hits = group_hits(P, y, ids, variant_of) if variant_of else None
    if reference is not None:
        record["like_for_like"] = e1_like_for_like(P, y, ids, used, hit, top5, split_name, reference,
                                                   word_hits=word_hits)
    if alias_of:
        record["alias_labels"] = e1_alias_rates(P, y, ids, alias_of, hit, top5)
    if variant_of:
        record["word_labels"] = e1_alias_rates(P, y, ids, variant_of, hit, top5, rule=WORD_RULE)
        record["word_labels"]["top1_by_source"] = _top1_by_source(used, word_hits[0], hit > 0.5)
    per_cat = defaultdict(list)
    for c, h in zip(used, hit):
        per_cat[c.category or "other"].append(h)
    return {"label": label, "kind": kind, "run": True, "n": len(used), "n_classes": len(set(y.tolist())),
            "n_groups": len(set(groups)),
            "metrics": {"top1": float(hit.mean()), "top5": float(top5.mean()),
                        "macro_f1": float(f1_score(y, pred, labels=sorted(set(y.tolist())), average="macro",
                                                   zero_division=0)),
                        "coverage": float(acc.mean()), "not_sure_rate": float(1 - acc.mean()),
                        "wrong_accept_rate": float((acc & (hit == 0)).mean()),
                        "accepted_precision": float(hit[acc].mean()) if acc.any() else None},
            "decision": {"rule": "classifier alone (sliding-mode rule): accept when the crop-mean probability >= "
                                 "threshold, top-1 minus top-2 >= margin and >= 2 crops agree; no suggestion step, so "
                                 "not sure = 1 - accepted",
                         "policy": E1_POLICY, "temperature": temp, "threshold": thr,
                         "threshold_in_checkpoint": thr_fitted, "checkpoint_threshold": cal["checkpoint_threshold"],
                         "thresholds": cal["thresholds"], "per_policy": cal["per_policy"], "margin": margin,
                         "source": cal["source"], "fallback_reason": cal.get("reason")},
            "policies": policies,
            "sliding_replay_note": None if not sliding else
            "policies.<policy>.sliding_replay: the same clips replayed through the app's real sliding mode "
            "(single-crop windows, EMA, streak, cooldown; docs 15.18); accepted = it committed during the clip, "
            "wrong accept = at least one commit was another word. The columns above are the three-crop segment proxy.",
            "decision_note": f"classifier alone, {E1_POLICY} policy: accept when its probability at temperature "
                             f"{temp:.2f} >= {thr:.2f} ({src_note}); not sure = not accepted (no 'did you mean' "
                             "step); the app's segment mode uses the fused decision (E3)",
            **interval_fields({"top1": (hit.tolist(), groups), "top5": (top5.tolist(), groups)}),
            "interval_note": None if interval_reason(hit.tolist(), groups) is None else
            f"{interval_reason(hit.tolist(), groups)} in this test part: a signer-level interval needs more held-out "
            "signers (the cross-validation folds pool them)",
            "_hits": hit.tolist(), "_groups": groups,
            "per_category_top1": {k: round(float(np.mean(v)), 4) for k, v in sorted(per_cat.items())}, **record}


def _e1_sliding_replay(rec, clips, thresholds: dict, kind: str) -> dict | None:
    """E1b / E1c: the test clips replayed through the app's real sliding mode at each policy's threshold (docs 15.18).
    The decision columns of E1 are a segment-level proxy (three crops of the whole clip, >= 2 agree); sliding mode
    classifies single-crop windows, smooths them (EMA) and commits after a streak, which on commercial-safe gave
    4-11 points lower precision than the proxy. Only for the main classifier (not the CV folds or E1a)."""
    if kind not in ("primary", "extra_signers") or not hasattr(rec, "class_probs") \
            or not all(hasattr(c, "load") for c in clips):
        return None
    from app.inference.engine import EngineConfig
    from app.isl.models.calibration import sliding_commit_matrix
    from app.isl.training.fit_fusion import replay_sliding

    ecfg = EngineConfig()
    rclips: list = []
    windows, truths, failed = replay_sliding(rec, clips, ecfg, used=rclips)
    if not windows:
        return None
    n, out = len(windows), {}
    srcs = [getattr(c, "source", None) for c in rclips]
    grps = _groups(rclips)
    for pol in POLICIES:
        A, O, K = sliding_commit_matrix(windows, truths, [thresholds[pol]], ecfg.min_margin, ecfg.agree_n,
                                        ecfg.cooldown_ms)
        a, o, k = int(A[0].sum()), int(O[0].sum()), int(K[0].sum())
        out[pol] = {"threshold": thresholds[pol], "n": n, "replay_failed": failed, "coverage": a / n,
                    "not_sure_rate": 1 - a / n, "wrong_accept_rate": (a - o) / n,
                    "accepted_precision": o / a if a else None, "n_accepted": a, "commits_per_clip": k / n}
        # per source: the threshold rule (what sliding mode would commit when the policy is certified)
        dec = ["accept" if x else "abstain" for x in A[0]]
        rows, worst = source_rows(dec, dec, [float(x) for x in O[0]], srcs, grps)
        for r in rows.values():
            r.pop("top1", None)  # "correct" here is "every commit is the clip's word", only meaningful when accepted
        out[pol]["by_source"], out[pol]["worst_source"] = rows, worst
    return out


def _top1_by_source(used, word_hit, strict_hit) -> dict:
    """Per source: n, word-label top-1 and strict top-1 (gate (d) reads the ISL500 row, docs 15.37)."""
    by: dict[str, list] = defaultdict(list)
    for c, w, h in zip(used, word_hit, strict_hit):
        by[getattr(c, "source", None) or "?"].append((float(w), float(h)))
    return {k: {"n": len(v), "top1": float(np.mean([a for a, _ in v])), "top1_strict": float(np.mean([b for _, b in v]))}
            for k, v in sorted(by.items())}


def e3_fused(rec, clips, label: str, alias_of: dict | None = None, variant_of: dict | None = None) -> dict:
    from app.isl.training.fit_fusion import collect_evidence

    inputs, truths, used = collect_evidence(rec, clips)
    if not used:
        return {"label": label, "run": False, "n": 0, "reason": "no usable clips"}
    results = [fuse(i, rec.fusion) for i in inputs]
    groups = _groups(used)
    clf = getattr(rec, "classifier", None)
    # the open-vocabulary gate needs the classifier's classes (a pack without a classifier: every word is open)
    closed = set(clf.class_names) if clf is not None else (set() if hasattr(rec, "classifier") else None)
    if getattr(rec.fusion, "aliases", None):  # the app's rule: a group with a class is a classifier word (15.35)
        from app.isl.vocab.aliases import closed_with_aliases

        closed = closed_with_aliases(closed, rec.fusion.aliases)
    block = decision_block(results, truths, rec.fusion, groups, sources=[getattr(c, "source", None) for c in used],
                           closed_set=closed)
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
    word = None
    if variant_of:  # word labels (docs 15.37): a variant counts as its word, per source too (gate (d))
        word = e3_alias_rates(results, truths, rec.fusion, variant_of, closed, rule=WORD_RULE)
        wh = [bool(r.keys) and variant_of.get(r.keys[0], r.keys[0]) == variant_of.get(t, t)
              for r, t in zip(results, truths)]
        word["top1_by_source"] = _top1_by_source(used, wh, ok)
    ach = rec.fusion.achieved or {}
    for pol, b in block["policies"].items():  # what each policy uses, and what validation said it achieves
        a = ach.get(pol) or {}
        b.update(threshold=rec.fusion.accept[pol], suggest_min_top5_mass=rec.fusion.suggest_mass(pol),
                 val_target=a.get("target"), val_target_reached=a.get("target_reached"),
                 val_precision_lower_bound=a.get("precision_lower_bound"), val_rule=a.get("rule"),
                 val_certified_precision=a.get("certified_precision"), val_worst_source=a.get("worst_source"),
                 val_floor=a.get("floor"), val_by_source=a.get("by_source"),
                 certification=rec.fusion.certification_info(pol))
    return {"label": label, "run": True, "n": len(used), "n_groups": len(set(groups)),
            "n_classes": len(set(truths)), "fusion_fitted": rec.fusion.fitted, "metrics_policy": "balanced",
            "metrics": _metrics_from(block), **block, "risk_coverage": curve,
            "aurc": float(np.mean([c["risk"] for c in curve])),
            "top1_by_source": {k: {"n": len(v), "top1": float(np.mean(v))} for k, v in sorted(by_src.items())},
            **({"alias_labels": e3_alias_rates(results, truths, rec.fusion, alias_of, closed)} if alias_of else {}),
            **({"word_labels": word} if word is not None else {})}


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

    ci, ct, cu = run(calib)
    # retrieval only; the fused system's certification does not carry over: uncertified until fitted here
    params = base_params.with_(alpha=0.0, gamma=0.0, beta=1.0, achieved={}, certification={})
    if ci:
        # per-source certification on seen-class validation queries; every word here is open vocabulary
        # (retrieval only), so the strict-floor open-vocabulary check applies (docs 15.25)
        params, _ = fit_thresholds([fuse(i, params) for i in ci], ct, params,
                                   sources=[getattr(c, "source", None) for c in cu],
                                   groups=[getattr(c, "group", None) for c in cu], closed_set=set())
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
        block = decision_block(res, qt, params, groups, sources=[getattr(c, "source", None) for c in qu],
                               closed_set=set())
        if name == "full":
            # held-out-concept false accepts (docs 15.25): the same queries against the gallery WITHOUT their
            # concepts (and synonyms), so every accept is wrong; pairs with the per-source out-of-vocabulary guard
            keep = [j for j, cid in enumerate(qi[0].ret_ids) if cid not in held]
            if keep:
                ids_k = [qi[0].ret_ids[j] for j in keep]
                oov = [fuse(FusionInputs(ret=x.ret[:, keep], ret_ids=ids_k), params) for x in qi]
                out["oov_false_accept"] = oov_false_accepts(oov, params)
        out[name] = {"n": len(qi), "n_groups": len(set(groups)), "n_classes": len(set(qt)),
                     "gallery_concepts": len(ids) if ids is not None else len(gal_concepts),
                     "metrics": _metrics_from(block, {f"top{k}": float(np.mean(v)) for k, v in hits.items()}),
                     **interval_fields({f"top{k}": (v, groups) for k, v in hits.items()}),
                     "_hits": hits, "_groups": groups, "_block": block}
    si, st, su = run(seen)
    if si:
        res = [fuse(i, params) for i in si]
        out["seen_class"] = {"n": len(si), "top1": float(np.mean([bool(r.keys) and r.keys[0] == t for r, t in zip(res, st)])),
                             "top5": float(np.mean([t in r.keys[:5] for r, t in zip(res, st)])),
                             "label": "INCLUDE test clips of concepts the fold encoder DID train on (same gallery)"}
    return out


def oov_false_accepts(results, params: FusionParams) -> dict:
    """Per policy, on queries whose word is NOT in the gallery: false-accept rate with the pack's gates (what the
    app auto-accepts), with the threshold rule alone, and the out-of-vocabulary abstain rate (docs 15.25)."""
    out = {"n": len(results)}
    ungated = params.with_(certification=None)
    for pol in POLICIES:
        dec = _decisions(results, params, pol, set())
        udec = _decisions(results, ungated, pol)
        n = max(1, len(results))
        out[pol] = {"false_accept_rate": sum(d == "accept" for d, _ in dec) / n,
                    "threshold_false_accept_rate": sum(d == "accept" for d, _ in udec) / n,
                    "out_of_vocabulary_rate": sum(r == "out_of_vocabulary" for _, r in dec) / n,
                    "suggest_rate": sum(d == "suggest" for d, _ in dec) / n}
    return out


def e2_pool(folds: list[dict]) -> dict:
    out = {}
    oov = [f["oov_false_accept"] for f in folds if f.get("run") and (f.get("oov_false_accept") or {}).get("n")]
    if oov:
        n = sum(o["n"] for o in oov)
        out["oov_false_accept"] = {"n": n, **{pol: {k: sum(o[pol][k] * o["n"] for o in oov) / n
                                                    for k in oov[0][pol]} for pol in POLICIES},
                                   "label": "held-out-concept queries against a gallery without their concepts: "
                                            "every accept is a false accept"}
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
                     **interval_fields({f"top{k}": (v, groups) for k, v in hits.items()}),
                     "policies": pol}
    for f in folds:
        for name in ("full", "restricted"):
            for k in ("_hits", "_groups", "_block"):
                f.get(name, {}).pop(k, None)
    return out


# --------------------------------------------------------------------------------------------- E4

E4_KS = (3, 5)  # enrolled takes: the app enrols 3 to 5 (MySignsStore.add min_takes / max_takes, docs 15.27)
E4_POLICY = "balanced"  # the app's default policy


def e4_my_signs(rec, clips, ks=E4_KS, seed: int = 0, max_groups: int = 40, params: FusionParams | None = None,
                cache: dict | None = None, policy: str = E4_POLICY, trace: list | None = None,
                keep_hits: bool = False) -> dict:
    """Few-shot enrolment on held-out signers' own takes (no store or disk involved). ``params`` overrides the
    recogniser's fusion parameters (fit-fusion compares the user-sign classifier modes with it); ``cache`` shares
    the per-clip evidence between such calls.

    The decision is the app's (docs 15.27): enrolment vectors are whole-take embeddings as ``MySignsStore`` makes
    them, theta is ``theta_from_embeddings`` of those vectors (the store's rule), and every test take goes through
    ``recognizer.user_sign_decision``, the function ``IslRecognizer.recognize`` uses (crop agreement on the
    override, a confirm-only policy turns it into a suggestion, a user sign on top without the override is only a
    suggestion). Two false-accept rates on the words a signer did NOT enrol: ``user_sign_false_accept_rate`` = what
    the app accepts by itself (0 under a confirm-only policy), ``user_sign_false_accept_rate_threshold_rule`` = the
    same rules without the certification gate (the user sign the app would put first and ask to confirm; with an
    ungated ``params`` both are equal). ``trace``: a list that gets one dict per test take (tests).
    ``keep_hits``: keep the per-take hit lists (``_with``, ``_fa``, ``_fa_threshold``) and the enrolled takes'
    sources (``_with_src``) for paired comparisons.

    Enrolment uses the pack's own consistency rule (``fusion.enrol['c_min']``, as the app's enrolment). A word whose
    k >= 2 takes fail it is refused, as ``MySignsStore.add`` refuses it ("Your recordings look different"): it is not
    a user sign, so all its takes are tested as a word the signer did not enrol, and ``enrolment_refused`` counts it
    (docs 15.28). A signer with no accepted enrolment is skipped."""
    from app.isl.models.encoder import FeatureStatsEmbedder
    from app.isl.mysigns.store import theta_from_embeddings
    from app.isl.recognizer import user_sign_decision

    fp = params or rec.fusion
    ungated = fp.with_(certification=None) if fp.gated else None
    c_min = float((getattr(fp, "enrol", None) or {}).get("c_min", 0.5))  # the app's enrolment rule (docs 15.28)
    is_ov = getattr(rec, "is_open_vocabulary", None)
    has_global = bool(getattr(rec, "has_global_model", True))

    emb = rec.encoder if rec.encoder is not None else FeatureStatsEmbedder(rec.feature_config)
    by = defaultdict(lambda: defaultdict(list))
    for c in clips:
        if c.concept_id and c.group:
            by[c.group][c.concept_id].append(c)
    rng = np.random.default_rng(seed)
    groups = sorted(by)[:max_groups]
    cache = {} if cache is None else cache

    def ev(c):
        if c.clip_id not in cache:
            f = _features(c, rec.feature_config)
            inp, _ = rec.evidence(f, use_my_signs=False)
            cache[c.clip_id] = (inp, emb.embed(f)[0])
        return cache[c.clip_id]

    out = {}
    enrolment = {}
    for k in ks:
        rows = {"base": [], "with": [], "fa": [], "fa_thr": [], "g_hit": [], "g_fa": [], "acc_ok": [], "src": []}
        n_enrol = n_refused = 0
        for g in groups:
            words = {w: v for w, v in by[g].items() if len(v) >= k + 1}
            if len(words) < 2:
                continue
            names = sorted(words)
            rng.shuffle(names)
            enrolled = set(names[: max(1, len(names) // 2)])
            signs = {}
            refused = 0
            refused_takes = {}
            try:
                for w in sorted(enrolled):  # a set: its order (string hashing) changed per process
                    takes = list(words[w])
                    rng.shuffle(takes)
                    # the whole take (crop 0), as MySignsStore.user_index embeds a recorded take
                    E = np.stack([ev(t)[1][0] for t in takes[:k]]).astype(np.float32)
                    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
                    # the store's rule (None: < 2 takes or inconsistent), with the pack's enrolment c_min
                    th = theta_from_embeddings(E, c_min=c_min)
                    if th is None and k >= 2:  # MySignsStore.add refuses this enrolment: not a user sign
                        refused += 1
                        refused_takes[w] = [t.clip_id for t in takes[:k]]
                        continue
                    theta = max(th if th is not None else 1.0, fp.user_theta_floor, fp.s_min)
                    signs[w] = (E, theta, takes[k:], [t.clip_id for t in takes[:k]])
                tests = [(t, w, True) for w in sorted(signs) for t in signs[w][2]]
                tests += [(t, w, False) for w in names if w not in signs for t in words[w]]
            except (FeatureError, OSError, ValueError):
                continue
            n_enrol += len(signs) + refused
            n_refused += refused
            if not signs:  # every enrolment of this signer refused: no user sign to measure
                continue
            order = sorted(signs)
            uid = [f"{USER_PREFIX}{w}" for w in order]

            def uth(key, _s=signs):
                return _s[key.removeprefix(USER_PREFIX)][1]

            for t, w, is_enrolled in tests:
                try:
                    inp, z = ev(t)
                except (FeatureError, OSError, ValueError):
                    continue
                zz = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
                U = np.stack([(zz @ signs[s][0].T).max(1) for s in order], axis=1)
                inp2 = dataclasses.replace(inp, user=U.astype(np.float32), user_ids=uid)  # keeps compact fields
                base = fuse(inp, fp)
                r = fuse(inp2, fp)
                n_crops = len(r.per_crop_top1)
                dec, why, keys, _ = user_sign_decision(r, inp2, n_crops, fp, policy, uth, open_vocabulary=is_ov,
                                                       has_global_model=has_global)
                dec_thr = dec if ungated is None else user_sign_decision(
                    r, inp2, n_crops, ungated, policy, uth, open_vocabulary=is_ov, has_global_model=has_global)[0]
                top = keys[0] if keys else ""
                pred = top.removeprefix(USER_PREFIX)
                if trace is not None:
                    trace.append({"k": k, "group": g, "clip_id": t.clip_id, "word": w, "enrolled": is_enrolled,
                                  "enrolment": {x: signs[x][3] for x in order}, "refused": refused_takes,
                                  "decision": dec, "reason": why,
                                  "top": top, "decision_threshold_rule": dec_thr})
                if is_enrolled:
                    rows["base"].append(float(bool(base.keys) and base.keys[0] == w))
                    rows["with"].append(float(pred == w))
                    rows["acc_ok"].append(float(pred == w and dec == "accept"))
                    rows["g_hit"].append(g)
                    rows["src"].append(getattr(t, "source", None) or "")
                else:
                    rows["fa"].append(float(top.startswith(USER_PREFIX) and dec == "accept"))
                    rows["fa_thr"].append(float(top.startswith(USER_PREFIX) and dec_thr == "accept"))
                    rows["g_fa"].append(g)
        enrolment[f"k{k}"] = {"enrolments": n_enrol, "refused": n_refused}
        if rows["with"]:
            out[f"k{k}"] = {"n": len(rows["with"]), "n_groups": len(set(rows["g_hit"])),
                            "top1_without_enrolment": float(np.mean(rows["base"])),
                            "top1_with_enrolment": float(np.mean(rows["with"])),
                            "gain": float(np.mean(rows["with"]) - np.mean(rows["base"])),
                            "accepted_correct_rate": float(np.mean(rows["acc_ok"])),
                            "interval_with": grouped_bootstrap(rows["with"], rows["g_hit"]),
                            "interval_with_reason": interval_reason(rows["with"], rows["g_hit"]),
                            "n_other_words": len(rows["fa"]), "n_other_groups": len(set(rows["g_fa"])),
                            "user_sign_false_accept_rate": float(np.mean(rows["fa"])) if rows["fa"] else None,
                            "user_sign_false_accepts": int(sum(rows["fa"])),
                            "user_sign_false_accept_rate_threshold_rule":
                                float(np.mean(rows["fa_thr"])) if rows["fa_thr"] else None,
                            "user_sign_false_accepts_threshold_rule": int(sum(rows["fa_thr"])),
                            "interval_false_accept": grouped_bootstrap(rows["fa"], rows["g_fa"]),
                            "interval_false_accept_threshold_rule": grouped_bootstrap(rows["fa_thr"], rows["g_fa"]),
                            "interval_false_accept_reason": interval_reason(rows["fa"], rows["g_fa"]),
                            "enrolments": n_enrol, "enrolment_refused": n_refused, "enrol_c_min": c_min}
            if keep_hits:
                out[f"k{k}"].update(_with=rows["with"], _fa=rows["fa"], _fa_threshold=rows["fa_thr"],
                                    _with_src=rows["src"])
    label = ("My signs few-shot: a held-out signer enrols k takes (3 to 5, as the app) of half of their words; "
             "tested on their other takes (gain over no enrolment) and on the words they did not enrol (false "
             "accepts of user signs). The app's own decision rule (docs 15.27). Groups are INCLUDE sessions / "
             "estimated signers, or other sources' signer ids, not verified identities.")
    if not out:
        n_ref = sum(v["refused"] for v in enrolment.values())
        why = (f"every enrolment was refused by the store's consistency rule (c_min {c_min}; {n_ref} refused)"
               if n_ref and n_ref == sum(v["enrolments"] for v in enrolment.values()) else
               f"no test signer has enough repeated takes (needs >= 2 words with >= {min(ks) + 1} takes each)")
        return {"label": label, "run": False, "reason": why, "n": 0, "ks": list(ks), "enrolment": enrolment,
                "enrol_c_min": c_min}
    best = out.get("k3") or next(iter(out.values()))
    return {"label": label, "run": True, "n": best["n"], "n_groups": best["n_groups"], "ks": list(ks),
            "policy": policy, "rule": fp.policy_rule(policy), "user_cls_mode": fp.user_cls_mode,
            "enrolment": enrolment, "enrol_c_min": c_min,
            "metrics": {"top1_with_enrolment_k3": best["top1_with_enrolment"], "gain_k3": best["gain"],
                        "user_sign_false_accept_rate_k3": best["user_sign_false_accept_rate"],
                        "user_sign_false_accept_rate_threshold_rule_k3":
                            best["user_sign_false_accept_rate_threshold_rule"]},
            "by_k": out}


def e4_populations(rec, include_test, extra_test, ks=E4_KS) -> dict:
    """E4 on two populations (docs 15.27). The headline is INCLUDE test signers only: the same population for every
    profile (commercial-safe has only INCLUDE), so packs compare like for like. ``all_sources`` adds the other
    sources' held-out signers (ISL500 has many repeated takes, which a pooled number would be dominated by). The
    evidence cache is shared, so the second run computes nothing new."""
    cache: dict = {}
    inc = e4_my_signs(rec, include_test, ks=ks, cache=cache)
    inc["population"] = "INCLUDE test signers only (comparable across profiles)"
    if not extra_test:
        inc["all_sources"] = {"run": False, "reason": "no other source has held-out test signers"}
        return inc
    allp = e4_my_signs(rec, list(include_test) + list(extra_test), ks=ks, cache=cache)
    allp["population"] = "INCLUDE test signers + other sources' held-out signers"
    inc["all_sources"] = allp
    return inc


def e4_like_for_like(e4: dict, ref_path: Path, k: str = "k3") -> dict:
    """Critic 4 (docs 15.28): this pack's user-sign false-accept rate (threshold rule, INCLUDE test signers only,
    k = 3) against the reference report's. ``pass`` = this <= reference. A reference written before docs 15.27 has
    no threshold-rule rate: its ``user_sign_false_accept_rate`` is used and a note says so. Recorded, not (yet) a
    gate that blocks a pack."""
    ref_path = Path(ref_path)
    out = {"reference": {"path": str(ref_path)}, "k": k, "metric": "user_sign_false_accept_rate_threshold_rule",
           "rule": "this pack's user-sign false-accept rate (threshold rule) <= the reference's, INCLUDE test "
                   "signers only", "notes": []}
    try:
        doc = json.loads(ref_path.read_text())
    except (OSError, ValueError) as e:
        return {**out, "run": False, "reason": f"cannot read the reference report: {e}"}
    r4 = (doc.get("protocols") or {}).get("E4") or {}
    out["reference"].update(profile=doc.get("profile"), population=r4.get("population"))
    rk = (r4.get("by_k") or {}).get(k) if r4.get("run") else None
    if not rk:
        return {**out, "run": False, "reason": f"the reference report has no E4 {k} result"
                + (f" ({r4.get('reason')})" if r4.get("reason") else "")}
    ref_v = rk.get("user_sign_false_accept_rate_threshold_rule")
    if ref_v is None:
        ref_v = rk.get("user_sign_false_accept_rate")
        out["notes"].append("the reference report has no threshold-rule rate (written before docs 15.27): compared "
                            "with its user_sign_false_accept_rate")
    if not r4.get("population"):
        out["notes"].append("the reference report does not record its E4 population (written before docs 15.27); "
                            "a commercial-safe report is INCLUDE only, other profiles may have pooled other sources")
    own = (e4.get("by_k") or {}).get(k) if e4.get("run") else None
    if not own:
        return {**out, "run": False, "reason": f"this report has no INCLUDE-only E4 {k} result"
                + (f" ({e4.get('reason')})" if e4.get("reason") else "")}
    this_v = own.get("user_sign_false_accept_rate_threshold_rule")
    if ref_v is None or this_v is None:
        return {**out, "run": False, "reason": "a false-accept rate is missing (no other-word takes)"}
    return {**out, "run": True, "reference_rate": float(ref_v), "this_rate": float(this_v),
            "reference_n_other_words": rk.get("n_other_words"), "this_n_other_words": own.get("n_other_words"),
            "pass": bool(this_v <= ref_v)}


# --------------------------------------------------------------------------------------------- pre-training guard

def pretraining_guard(work: Path, data) -> dict:
    """Computed, not asserted: the pre-trained encoder's actual clip list (``<work>/pretrain_clips.txt``, written by
    ``pretrain``) against every evaluation clip and its duplicate group, plus the Sign Library content matches
    (``pretrain_dedup.json``). ``excludes_eval_clips`` is True / False, or None when it cannot be checked."""
    from app.isl.training.multisource import eval_clip_ids, read_pretrain_list
    from app.isl.training.pose_dedup import DEDUP_FILE, load_dedup

    ck = work / "pretrained_encoder.pt"
    if not ck.exists():
        return {"excludes_eval_clips": None, "reason": "no self-supervised pre-training in this build"}
    ids = read_pretrain_list(work)
    if ids is None:
        return {"excludes_eval_clips": None,
                "reason": "pre-training clip list (pretrain_clips.txt) missing: re-run the pretrain step to check"}
    ev = eval_clip_ids(data)
    overlap = sorted(ids & ev)
    out = {"excludes_eval_clips": not overlap, "pretrain_clips": len(ids), "eval_clips_and_duplicates": len(ev),
           "overlap": len(overlap), "overlap_examples": overlap[:10]}
    dd = load_dedup(work / DEDUP_FILE)
    if dd is not None:
        chunks = {m.get("chunk") for m in dd.get("eval_matches") or []}
        seen = sorted(chunks & ids)
        out["sign_library"] = {"reliable": dd.get("reliable"), "drop_channels": dd.get("drop_channels"),
                               "eval_content_matches": dd.get("n_eval_matches", 0), "matched_chunks_in_pretraining":
                               len(seen)}
        if seen:
            out["excludes_eval_clips"] = False
            out["overlap_examples"] += seen[:10]
    gal = {c.clip_id for c in getattr(data, "islrtc", [])}
    out["islrtc_clips_seen_unlabelled"] = len(gal & ids)
    out["islrtc_clips"] = len(gal)
    return out


# --------------------------------------------------------------------------------------------- E5 / E7 / E8

def e5_from_letters(path: Path) -> dict:
    if not path.exists():
        return {"label": "fingerspelling", "run": False, "reason": "no letter model trained for this profile"}
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
    e5 = meta.get("e5") or {}
    groups = e5.get("groups") or e5.get("signers") or []
    lab = ("Fingerspelling, leave-one-group-out over the grouped letter sources (ISL-Fingerspelling signers, "
           f"Mendeley static age groups; {len(groups)} held-out groups; sources without group ids stay on the "
           "training side); letter accuracy on letter frames / photos, words via hold detection + dictionary "
           "decoding. Experimental while fewer than 5 signers exist.")
    cal = meta.get("calibration") or {"status": "not recorded (letter model trained before 15.12)",
                                      "temperature": meta.get("temperature")}
    if not e5.get("measurable"):
        return {"label": lab, "run": False, "reason": e5.get("reason", "not measured"), "data": meta.get("data"),
                "calibration": cal}
    return {"label": lab, "run": True, "n": e5.get("n"), "n_groups": len(groups),
            "metrics": e5.get("metrics"), "intervals": e5.get("intervals"),
            "interval_reasons": e5.get("interval_reasons") or {}, "interval_groups": e5.get("interval_groups") or {},
            "groups": groups, "group_kinds": e5.get("group_kinds"), "folds": e5.get("folds"),
            "per_letter": e5.get("per_letter"), "calibration": cal, "training": meta.get("training"),
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


def e8_trim(paths, artifacts_dir: Path, work: Path | None = None) -> dict:
    """Trim quality of the trims this run used. The synthetic / INCLUDE numbers come from the trim-eval report the
    driver's ``trim_eval`` step writes into the work folder (``<work>/trim_eval.json``; without ``work`` the old
    ``artifacts/reports/trim_eval.json``) and are shown only when that report evaluated the same ``trim_version``
    as the run's trim sidecar, every trim record carries the SHA-256 of the learned model trim-eval evaluated
    (``trim_model_sha256``; none recorded, mixed or another hash -> mismatch) and that model file is unchanged;
    otherwise ``trim_eval_check`` says ``mismatch`` or ``missing`` with the reason, and the label says so."""
    from app.isl.data.manifest import read_manifest
    from app.isl.data.trim_eval import check_report, trim_model_key

    rep = (Path(work) / "trim_eval.json") if work is not None else artifacts_dir / "reports" / "trim_eval.json"
    trims = read_manifest(paths.trim_sidecar, only_ok=False) if paths.trim_sidecar.exists() else []
    mix = Counter(r.get("method") for r in trims)
    versions = Counter(r.get("trim_version") for r in trims)
    hashes = Counter(trim_model_key(r) for r in trims)
    chk = check_report(rep, dict(versions), dict(hashes))
    doc = chk.pop("doc", None)
    label = ("ISLRTC first-sign trim quality (synthetic end-error check, INCLUDE active-frame check) from "
             "'cli trim-eval' of the run's trimmer, plus the method mix of the trim sidecar")
    if chk["status"] != "ok":
        label += (f". Synthetic trim check NOT SHOWN ({chk['status'].upper()}): {chk['reason']}")
    else:
        label += (f" (trim-eval of {chk['trim_version']}, written {chk.get('written_at') or chk.get('file_mtime_utc')})")
    out = {"label": label, "run": bool(trims) or doc is not None,
           "trims": len(trims), "method_mix": dict(mix),
           "trim_version": next(iter(versions)) if len(versions) == 1 else None, "trim_versions": dict(versions),
           "trim_model_hashes": dict(hashes),
           "low_confidence_frac": (sum(1 for r in trims if float(r.get("confidence") or 0) < 0.5) / len(trims))
           if trims else None,
           "trim_eval_check": chk}
    if not out["run"]:
        out["reason"] = "no trim sidecar and " + (chk["reason"] or "no trim-eval report")
    if trims:
        out["metrics"] = {"trims": len(trims), "low_confidence_frac": out["low_confidence_frac"],
                          "cap_frac": mix.get("cap", 0) / len(trims)}
    if doc is not None:
        out["trim_eval"] = doc
    return out


# --------------------------------------------------------------------------------------------- driver

def alias_summary(data, fusion) -> dict:
    """Which alias map the report scores with (docs 15.34): the link step's (``data.alias_of``), else the one
    fusion.json was fitted with. ``fusion_sums_groups``: the app's fused decision sums the groups. A fusion.json
    fitted with another map than the link step's is flagged (``mismatch``: run fit-fusion again), and so is a strict
    fusion.json in a folder the link step linked with aliases (``stale``, docs 15.35)."""
    from app.isl.vocab.aliases import alias_groups, aliases_sha256

    link = dict(getattr(data, "alias_of", None) or {})
    fus = dict(getattr(fusion, "aliases", None) or {})
    al = link or fus
    if not al:
        return {"used": False, "note": "no alias groups (the link step did not turn them on): strict labels only"}
    out = {"used": True, "alias_of": al, "groups": alias_groups(al), "sha256": aliases_sha256(al),
           "source": "link step" if link else "fusion.json", "fusion_sums_groups": bool(fus),
           "mismatch": bool(link and fus and link != fus), "stale": bool(link and not fus),
           "link_alias_version": (getattr(data, "link", None) or {}).get("alias_version"), "rule": ALIAS_RULE}
    vl = variant_links_summary(getattr(data, "link", None))
    if vl:
        out["variant_links"] = vl
    return out


def variant_links_summary(link: dict | None) -> dict | None:
    """The variant-audit file the link step froze (docs 15.36-37): file, sha256, relinks, the evidence scope
    (``parts_used``) and a warning when validation clips influenced the training labels. None without variants."""
    link = link or {}
    if not link.get("variants"):
        return None
    m = dict(link.get("variant_links") or {})
    parts = m.get("parts_used") or {}
    val = sorted(k for k, v in parts.items() if "val" in (v if isinstance(v, list) else [v]))
    out = {k: m.get(k) for k in ("file", "sha256", "audit_version", "n_relinks", "parts_used", "top_k",
                                 "dictionary_rule_usable", "dictionary_false_alarm_rate", "accepted_noisy")
           if k in m}
    out["variant_concepts"] = sorted(variant_word_map(link))
    out["relinked_clips"] = link.get("variant_relinked")
    sc = m.get("split_check") or {}
    if sc:
        out["split_check"] = {k: sc.get(k) for k in ("ok", "override", "moved_groups") if k in sc}
    warn = []
    if val:
        warn.append(f"validation clips of {', '.join(val)} were used as evidence for the training labels (the "
                    "audit's reverse check): validation numbers of these words are not fully held out")
    if m.get("accepted_noisy"):
        warn.append("the audit's dictionary rule was marked not usable and accepted anyway (--accept-noisy)")
    if sc.get("override"):
        warn.append("the audit's ISL500 train signers were not all train at link time (--allow-split-change)")
    if warn:
        out["warnings"] = warn
    return out


def evaluate_all(work: Path, data, fcfg, *, profile: str, smoke: bool, vocab_path: Path | None = None,
                 paths=None, out_dirs: list[Path] | None = None, e4_ks=E4_KS, latency_n: int = 20,
                 progress=None, partial_data: bool | None = None, compare_to: Path | None = None) -> dict:
    """Run every protocol whose components exist in ``work`` and write ``eval/report.json`` + ``report.md``.
    ``compare_to``: another evaluation report.json (e.g. the commercial-safe pack's); E1b then also reports top-1 /
    top-5 on that report's class set and on the clips both evaluated (``E1b.like_for_like``, docs 15.27), and E4
    compares the INCLUDE-only user-sign false-accept rate with the reference's (``E4.like_for_like``, docs 15.28)."""
    from app.core.config import REPO_ROOT, get_settings
    from app.isl.evaluation.protocols import e6_latency
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.pipeline import load_recognizer
    from app.isl.training.common import (class_space_of, classifier_grouping_mismatch, fusion_fit_space,
                                         fusion_space_mismatch)
    from app.isl.training.train_encoder import encoder_excluded_sources, encoder_training_clips

    from app.isl.training.readiness import partial_data_flag, readiness_summary

    log = progress or (lambda m: None)
    t0 = time.time()
    partial = partial_data_flag(work) if partial_data is None else bool(partial_data)
    pre_guard = pretraining_guard(work, data)
    split_name = data.split.get("name")
    rec = load_recognizer(work, fcfg, vocab_path, profile=profile)  # refuses an eval-including / stale index
    prot: dict = {}
    si = data.split.get("signer_independent")
    lab = data.split.get("label") or split_name
    test_inc = data.include_part("test")
    from app.isl.training.fit_fusion import check_index_for_fitting

    # no index row built from a clip that is measured (test) or that fitted the thresholds (val), and the index was
    # built from exactly this split's current training clips (docs 15.21, 15.22)
    index_guard = check_index_for_fitting(rec, test_inc + data.extra_part("test") + data.labelled_part("val"), data)
    # E1b / E1c / E3 on the primary split's test part
    log("E1b")
    reference = load_e1_reference(Path(compare_to)) if compare_to else None
    aliases = alias_summary(data, rec.fusion)
    alias_of = aliases.get("alias_of") or None
    variant_of = variant_word_map(getattr(data, "link", None)) or None  # word labels (docs 15.37)
    prot["E1b"] = e1_closed_set(rec, test_inc, f"E1b core words, closed set, INCLUDE test part of {split_name} ({lab}; "
                                f"signer-independent: {si})", split_name, "primary", reference=reference,
                                alias_of=alias_of, variant_of=variant_of)
    cv = sorted((work / "cv").glob("fold*/classifier.pt")) if (work / "cv").exists() else []
    # class spaces (docs 15.32): a classifier chosen with another signer grouping (e.g. WSLP as its own signers)
    # or an encoder trained with other sources has another class space; they are never pooled silently
    cls_meta, enc_meta = getattr(rec.classifier, "meta", None), getattr(rec.encoder, "meta", None)
    class_space = {"classifier": class_space_of(cls_meta) if cls_meta is not None else None,
                   "encoder": class_space_of(enc_meta) if enc_meta is not None else None,
                   "cv_folds_not_pooled": [], "oneshot_folds_other_sources": []}
    # a fusion.json fitted on another classifier / encoder class space (docs 15.33; one fitted before records none)
    fitted_on = getattr(rec.fusion, "fitted_on", None) or {}
    if fitted_on:
        idx = getattr(rec, "index", None)
        class_space["fusion_fitted_on"] = fitted_on
        class_space["fusion_mismatch"] = fusion_space_mismatch(
            fitted_on, fusion_fit_space(cls_meta, enc_meta, idx.meta.row_filters if idx is not None else None))
    if cv:
        from app.isl.training.common import load_split
        from app.models.checkpoint import Predictor

        folds = []
        for p in cv:
            pr = Predictor(p)
            sname = pr.meta.get("split")
            if not sname or sname == split_name:  # the main model is already E1b
                continue
            mis = classifier_grouping_mismatch(cls_meta, pr.meta) if cls_meta is not None else None
            if mis:
                class_space["cv_folds_not_pooled"].append({"fold": p.parent.name, **mis})
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
                                 "pooled_top1_interval_reason": interval_reason(hits, grp),
                                 "pooled_groups": len(set(grp)),
                                 "label": "every fold's held-out signers, each fold's model trained without them",
                                 "per_fold": [{"split": f["label"], "n": f.get("n"), "top1": f["metrics"]["top1"]}
                                              for f in runs]}
            prot["E1b"].setdefault("intervals", {})["top1_cv_pooled"] = prot["E1b"]["cv"]["pooled_top1_interval"]
            prot["E1b"].setdefault("interval_groups", {})["top1_cv_pooled"] = len(set(grp))
            if prot["E1b"]["cv"]["pooled_top1_interval_reason"]:
                prot["E1b"].setdefault("interval_reasons", {})["top1_cv_pooled"] = \
                    prot["E1b"]["cv"]["pooled_top1_interval_reason"]
        # folds left out for another class space are named next to the CV number itself (docs 15.33), and a CV
        # with every fold left out says why instead of vanishing
        if class_space["cv_folds_not_pooled"]:
            if "cv" in prot["E1b"]:
                prot["E1b"]["cv"]["not_pooled"] = list(class_space["cv_folds_not_pooled"])
            else:
                prot["E1b"]["cv"] = {"run": False, "reason": "cv folds trained with another signer grouping",
                                     "not_pooled": list(class_space["cv_folds_not_pooled"])}
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
                                    split_name, "extra_signers", alias_of=alias_of, variant_of=variant_of)
    else:
        prot["E1c"] = {"run": False, "reason": "no other labelled source has a signer-held-out test part yet (needs a "
                       "source with real signer ids and >= 6 signers, e.g. ISL500)"}
    log("E3")
    # E3 / E4 use the same classifier as E1b: refused with it (a classifier trained on another split, or one whose
    # split is unknown, may have seen these test clips)
    cls_why = classifier_refusal(rec.classifier, split_name)
    e3_label = ("E3 fused system (classifier + dictionary retrieval + fusion fitted on validation) with accept / "
                "suggest / abstain; INCLUDE test part + other sources' held-out signers")
    prot["E3"] = refused_entry(e3_label, cls_why) if cls_why else e3_fused(rec, test_inc + extra_test, e3_label,
                                                                            alias_of=alias_of, variant_of=variant_of)
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
            # the fold encoder's own training list: with the sources it was trained without (docs 15.32)
            xsrc = encoder_excluded_sources(enc.meta)
            if class_space["encoder"] is not None and xsrc != class_space["encoder"]["excluded_sources"]:
                class_space["oneshot_folds_other_sources"].append(
                    {"fold": k, "excluded_sources": xsrc,
                     "main_excluded_sources": class_space["encoder"]["excluded_sources"]})
            try:
                folds.append(e2_oneshot_fold(enc, fd, data, getattr(data, "linker", None), fcfg, rec.fusion,
                                             restricted, encoder_training_clips(data, held, exclude_sources=xsrc)))
            except LeakageError as e:
                folds.append({"fold": k, "run": False, "leakage": str(e)})
        pooled = e2_pool(folds)
        full = pooled.get("full", {})
        seen_gal = pre_guard.get("islrtc_clips_seen_unlabelled")
        prot["E2"] = {"label": "E2 open vocabulary, one-shot: concepts held out of ALL supervised training (synonyms "
                               "too); other signers' INCLUDE clips (val/test parts, never seen even unlabelled) vs one "
                               "ISLRTC dictionary clip per word; thresholds "
                               "fitted on seen-class validation queries only. Some misses are regional variants, not "
                               "model errors. NOTE: self-supervised pre-training saw ISLRTC clips WITHOUT labels, "
                               "including the dictionary reference clips E2 retrieves against"
                               + (f" ({seen_gal} of {pre_guard.get('islrtc_clips')} ISLRTC clips were in the "
                                  "pre-training list)" if seen_gal is not None else "") + ".",
                      "pretraining_saw_gallery_unlabelled": (seen_gal > 0) if seen_gal is not None else None,
                      "run": bool(full), "n": full.get("n", 0), "n_groups": full.get("n_groups"),
                      **({} if full else {"reason": "no queries: no held-out concept has both INCLUDE val/test clips "
                                          "and a dictionary clip in the gallery (smoke runs sample the ISLRTC clips)"
                                          if any(f.get("run") for f in folds) else
                                          "; ".join(str(f.get("leakage") or f.get("reason") or "fold not run")
                                                    for f in folds)}),
                      "metrics": full.get("metrics", {}), "intervals": full.get("intervals", {}),
                      "interval_reasons": full.get("interval_reasons", {}),
                      "interval_groups": full.get("interval_groups", {}),
                      "restricted_gallery": pooled.get("restricted"),
                      "oov_false_accept": pooled.get("oov_false_accept"), "folds": folds}
    else:
        prot["E2"] = {"run": False, "reason": "no held-out-concept encoders (work/oneshot/fold*/encoder.pt) or no "
                      "oneshot_concepts.json; the seen-class retrieval number is in E2_seen"}
    from app.isl.evaluation.protocols import e2_retrieval

    e2s = e2_retrieval(rec, test_inc, "E2_seen: INCLUDE test clips vs ISLRTC rows only; SEEN classes (the encoder "
                       "trained on these concepts' other clips) - not the open-vocabulary number")
    if e2s:
        prot["E2_seen"] = e2s
    log("E4")
    prot["E4"] = (refused_entry("My signs few-shot (uses the word classifier)", cls_why) if cls_why else
                  e4_populations(rec, test_inc, extra_test, ks=e4_ks))
    if compare_to and not cls_why:  # critic 4: user-sign false accepts no worse than the reference pack's
        prot["E4"]["like_for_like"] = e4_like_for_like(prot["E4"], Path(compare_to))
    prot["E5"] = e5_from_letters(work / "letters.pt")
    log("E6")
    prot["E6"] = e6_latency(rec, test_inc + extra_test, n=latency_n)
    prot["E7"] = e7_asl_regression(REPO_ROOT)
    if paths is not None:
        prot["E8"] = e8_trim(paths, get_settings().artifacts_dir, work)
    for v in prot.values():
        for k in ("_hits", "_groups"):
            v.pop(k, None)
    leak = {"split": split_name, "classifier_split_checked": True,
            "classifier_refused": cls_why, "dedup": getattr(data, "multisource", {}).get("dedup"),
            "links": getattr(data, "multisource", {}).get("links"),
            "pretraining_excludes_eval_clips": pre_guard.get("excludes_eval_clips"), "pretraining_check": pre_guard,
            "index": ({"dir": "index", "training_parts_only": (index_guard["contains_eval"] is False
                                                               and index_guard["split_checked"]
                                                               and index_guard["train_clip_ids_checked"]),
                       **index_guard} if index_guard is not None
                      else {"dir": None, "note": "no retrieval index"}),
            "class_space": class_space}
    if smoke:
        label = ("SMOKE RUN on partial data with tiny settings: proves the pipeline runs end to end. These numbers "
                 "are NOT the accuracy of the app.")
    elif partial:
        label = PARTIAL_DATA_LABEL
    else:
        label = ("Measured on held-out data (test parts used once, after all selection on validation). Live "
                 "webcam accuracy is expected to be lower.")
    data_doc = data.summary()
    guard = readiness_summary(work)  # what the data guard saw at the run's start (non-blocking warnings included)
    if guard is not None:
        data_doc = {**data_doc, "readiness": guard}
    doc = {"schema": 2, "language": "isl", "profile": profile, "smoke": smoke, "partial_data": bool(partial or smoke),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": git_commit(),
           "label": label, "data": data_doc, "leakage_guards": leak, "protocols": prot,
           "aliases": {k: v for k, v in aliases.items() if k != "alias_of"},
           "seconds": round(time.time() - t0, 1)}
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


def _ci_cell(p: dict, key: str = "top1") -> str:
    iv, why, ng = p.get("intervals") or {}, p.get("interval_reasons") or {}, p.get("interval_groups") or {}
    ci, tag, k = iv.get(key), "", key
    if not ci and iv.get(f"{key}_cv_pooled"):
        k, tag = f"{key}_cv_pooled", "cv pooled, "
        ci = iv[k]
    if ci:
        n = ng.get(k)
        return f"{ci[0]:.3f}-{ci[1]:.3f} ({tag}{n if n is not None else '?'} groups)"
    if why.get(f"{key}_cv_pooled"):
        return f"none: {why[f'{key}_cv_pooled']} (cv pooled)"
    return f"none: {why[key]}" if why.get(key) else "-"


def _e5_lines(p: dict) -> list[str]:
    """E5 bullets: every letter accuracy with its interval cell (or the reason there is none) and the calibration."""
    out = []
    m = p.get("metrics") or {}
    for key in sorted(k for k in m if k.startswith("letter_top1")):
        if key in (p.get("intervals") or {}):
            out.append(f"* {key}: {_fmt(m[key])} (95% CI {_ci_cell(p, key)}; groups "
                       f"{(p.get('interval_groups') or {}).get(key, '-')})")
        else:
            out.append(f"* {key}: {_fmt(m[key])} (unweighted mean over {p.get('n_groups', '-')} held-out groups; "
                       "no interval)")
    w = p.get("words") or {}
    if w:
        out.append(f"* word top-1: {_fmt(w.get('top1'))} (n {w.get('n')}, 95% CI "
                   + format_interval(w.get("top1_interval"), w.get("n_groups"), w.get("top1_interval_reason")) + ")")
    c = p.get("calibration") or {}
    if c:
        out.append(f"* calibration: **{c.get('status')}**, temperature {_fmt(c.get('temperature'))}"
                   + (f", fitted on {c.get('n')} held-out examples of {c.get('n_groups')} groups "
                      f"({', '.join(c.get('held_out_units') or [])}; {c.get('how')}); NLL {_fmt(c.get('nll_t1'))} -> "
                      f"{_fmt(c.get('nll'))}, ECE {_fmt(c.get('ece_t1'))} -> {_fmt(c.get('ece'))}"
                      if c.get("status") == "grouped" else f" ({c.get('reason', '')})"))
    return out + [""] if out else out


def _policy_lines(doc: dict) -> list[str]:
    """Per-policy decision numbers (docs 15.17): the main table shows E1 and E3 at the balanced policy only."""
    rows = []
    for pid in ("E1b", "E1c", "E3"):
        p = doc["protocols"].get(pid) or {}
        pols = p.get("policies") if p.get("run") else None
        if not isinstance(pols, dict):
            continue
        how = (p.get("decision") or {}).get("source") if pid.startswith("E1") else "fusion.json accept (fused)"
        for pol in ("strict", "balanced", "lenient"):
            b = pols.get(pol) or {}
            val = ""
            if b.get("val_target") is not None:
                val = (f"target {b['val_target']:.2f} {'reached' if b.get('val_target_reached') else 'NOT reached'}"
                       f", val lower bound {_fmt(b.get('val_precision_lower_bound'))}")
            if b.get("rule") or b.get("val_rule"):
                val = (val + "; " if val else "") + f"rule {b.get('rule') or b.get('val_rule')}"
            rows.append(f"| {pid} | {pol} | {_fmt(b.get('threshold'))} | {_fmt(b.get('coverage'))} | "
                        f"{_fmt(b.get('accepted_precision'))} | {_fmt(b.get('not_sure_rate'))} | "
                        f"{_fmt(b.get('wrong_accept_rate'))} | {_fmt(b.get('suggest_rate'))} | {how}"
                        + (f"; {val}" if val else "") + " |")
            sr = b.get("sliding_replay") if pid.startswith("E1") else None
            if isinstance(sr, dict):  # what sliding mode itself does at this threshold (docs 15.18)
                rows.append(f"| {pid} sliding replay | {pol} | {_fmt(sr.get('threshold'))} | "
                            f"{_fmt(sr.get('coverage'))} | {_fmt(sr.get('accepted_precision'))} | "
                            f"{_fmt(sr.get('not_sure_rate'))} | {_fmt(sr.get('wrong_accept_rate'))} | - | "
                            f"{how}; real sliding mode, per clip |")
    if not rows:
        return []
    return ["", "## Decision policies (strict / balanced / lenient)", "",
            "The table above shows the balanced policy (the app's default). E1 = the classifier alone (sliding mode), "
            "E3 = the fused decision (segment mode). A policy whose worst-source validation bound is below its "
            "floor is confirm-only: it accepts nothing by itself (docs 15.25). "
            "E1 rows are the three-crop segment proxy; 'sliding replay' rows are the same clips through the real "
            "sliding mode (docs 15.18).", "",
            "| Protocol | policy | threshold | accepted | accepted precision | not sure | wrong accept | suggest | "
            "thresholds from |", "|---|---|---|---|---|---|---|---|---|", *rows]


def _headline_lines(doc: dict) -> list[str]:
    """The headline (docs 15.25): per policy, the INCLUDE row and the worst-source row of the fused system (E3) and
    of sliding mode (E1b INCLUDE, E1c the other sources), with the validation certification. "auto" = what the app
    accepts without asking; "threshold" = the top candidate at the policy's threshold (what a confirm tap shows)."""
    rows = []
    for pid in ("E3", "E1b", "E1c"):
        p = doc["protocols"].get(pid) or {}
        pols = p.get("policies") if p.get("run") else None
        if not isinstance(pols, dict):
            continue
        for pol in ("strict", "balanced", "lenient"):
            b = pols.get(pol) or {}
            if pid == "E3":
                by, worst = b.get("by_source") or {}, b.get("worst_source")
                rule = b.get("val_rule") or b.get("rule")
                cert = b.get("val_certified_precision")
                kind = "fused (segment mode)"
            else:
                sr = b.get("sliding_replay") or {}
                by, worst = sr.get("by_source") or {}, sr.get("worst_source")
                rule, cert, kind = b.get("rule"), None, "sliding mode"
            if not by:
                continue
            show = [("INCLUDE", "include")] if "include" in by else []
            if worst and worst != "include":
                show.append(("worst source", worst))
            elif worst == "include":
                show = [("INCLUDE = worst source", "include")]
            for tag, src in show:
                r = by.get(src) or {}
                rows.append(f"| {pid} {kind} | {pol} | {rule or '-'} | {_fmt(cert)} | {tag}: {src} | {r.get('n', '-')} | "
                            f"{r.get('n_groups', '-')} | {_fmt(r.get('coverage'))} | {_fmt(r.get('accepted_precision'))} "
                            f"| {_fmt(r.get('wrong_accept_rate'))} | {_fmt(r.get('threshold_coverage'))} | "
                            f"{_fmt(r.get('threshold_precision'))} |")
    if not rows:
        return []
    return ["## Headline: INCLUDE and the worst source", "",
            "Rule = the validation certification (docs 15.25): 'certified' policies may accept without asking; "
            "'confirm_only' policies never do (their top candidate is a 'did you mean'). Certified = the 95 % lower "
            "bound of accepted precision on the WORST major source, adjusted for signer clustering, at least 0.95 / "
            "0.90 / 0.85 (strict / balanced / lenient). The app shows that certified number, never the 0.98 target, "
            "and keeps a confirm tap. 'auto' columns = what the app accepts by itself; 'threshold' columns = the top "
            "candidate at the policy's threshold (what the user is asked to confirm).", "",
            "| Headline | policy | rule (val) | certified worst-source LB (val) | row | n | groups | auto accepted | "
            "auto precision | auto wrong accept | threshold accepted | threshold precision |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, ""]


def _like_for_like_lines(lf: dict | None) -> list[str]:
    if not lf:
        return []
    r = lf.get("reference") or {}
    head = [f"E1b like for like against {r.get('profile') or '-'} ({r.get('path')}; its E1b top-1 {_fmt(r.get('top1'))}, "
            f"n {r.get('n', '-')}), strict labels:"]
    if not lf.get("run"):
        return ["", *head, f"* not compared: {lf.get('reason')}"]
    out = ["", *head]
    for key, name in (("reference_classes", "masked to the reference's classes"),
                      ("common_clips", "the clips both reports evaluated"),
                      ("reference_classes_common_clips", "masked + common clips (like for like)")):
        v = lf.get(key) or {}
        if v.get("run") is False or not v.get("n"):
            out.append(f"* {name}: {v.get('reason') or 'no clips'}")
            continue
        out.append(f"* {name}: n {v['n']}, top-1 {_fmt(v.get('top1'))}, top-5 {_fmt(v.get('top5'))}"
                   + (f"; reference top-1 {_fmt(v.get('reference_top1'))}, top-5 {_fmt(v.get('reference_top5'))}"
                      if "reference_top1" in v else "")
                   + (f"; top-1 only this {v['discordant_top1']['only_this']}, only reference "
                      f"{v['discordant_top1']['only_reference']}" if v.get("discordant_top1") else ""))
    out += [f"* note: {n}" for n in lf.get("notes") or []]
    return out


def _alias_lines(doc: dict) -> list[str]:
    """Alias-label numbers next to the strict ones (docs 15.34). Never a headline or a gate (critic 12)."""
    al = doc.get("aliases") or {}
    if not al.get("used"):
        return []
    out = ["", f"Alias labels ({len(al.get('groups') or {})} groups from the {al.get('source')}; fused decision sums "
               f"the groups: {'yes' if al.get('fusion_sums_groups') else 'no'}). NOT a gate: every number above and "
               "every gate uses strict labels."]
    if al.get("mismatch"):
        out.append("* **fusion.json was fitted with another alias map than the link step's: run fit-fusion again**")
    if al.get("stale"):
        out.append("* **the link step turned aliases on but fusion.json has none (fitted before the link step, or "
                   "strict): run fit-fusion again**")
    for pid in ("E1b", "E1c", "E3"):
        a = (doc["protocols"].get(pid) or {}).get("alias_labels")
        if not a:
            continue
        row = (f"* {pid}: n {a.get('n')}, top-1 strict {_fmt(a.get('top1_strict'))} / alias {_fmt(a.get('top1'))}, "
               f"top-5 strict {_fmt(a.get('top5_strict'))} / alias {_fmt(a.get('top5'))}")
        bg = {g: v.get("top1_only_alias") for g, v in (a.get("by_group") or {}).items() if v.get("top1_only_alias")}
        if bg:
            row += "; clips right only on alias labels: " + ", ".join(f"{g} {v}" for g, v in bg.items())
        b = (a.get("policies") or {}).get("balanced")
        if b:
            row += (f"; balanced accepted precision strict {_fmt(b.get('accepted_precision_strict'))} / alias "
                    f"{_fmt(b.get('accepted_precision'))}")
        out.append(row)
    out += _word_label_lines(doc)
    return out


def _word_label_lines(doc: dict) -> list[str]:
    """Variant links (docs 15.36-37): which audit relinked the training labels, and the word-label numbers (a variant
    counted as its word) that the gates of a variant run use."""
    vl = (doc.get("aliases") or {}).get("variant_links")
    if not vl:
        return []
    fa = vl.get("dictionary_false_alarm_rate")
    out = ["", f"Variant links (docs 15.36-37): {vl.get('n_relinks')} words relinked "
               f"({vl.get('relinked_clips')} ISL500 clips) by {vl.get('file')} (sha256 {str(vl.get('sha256'))[:16]}), "
               f"evidence {vl.get('parts_used')}, dictionary false-alarm rate on agreeing words "
               f"{'-' if fa is None else f'{fa:.3f}'}. Word labels count a variant as its word (twins stay strict); "
               "the gates of a variant run read the word-label numbers."]
    out += [f"* **warning: {w}**" for w in vl.get("warnings") or []]
    for pid in ("E1b", "E1c", "E3"):
        a = (doc["protocols"].get(pid) or {}).get("word_labels")
        if not a:
            continue
        out.append(f"* {pid} word labels: n {a.get('n')}, top-1 strict {_fmt(a.get('top1_strict'))} / word "
                   f"{_fmt(a.get('top1'))}, top-5 strict {_fmt(a.get('top5_strict'))} / word {_fmt(a.get('top5'))}")
    cw = ((doc["protocols"].get("E1b") or {}).get("like_for_like") or {}).get("common_clips_word_labels")
    if cw and cw.get("n"):
        out.append(f"* E1b common clips, word labels: n {cw['n']}, top-1 {_fmt(cw.get('top1'))}, top-5 "
                   f"{_fmt(cw.get('top5'))}; reference top-1 {_fmt(cw.get('reference_top1'))}")
    return out


def _e4_lines(e4: dict) -> list[str]:
    """E4 with the user-sign FALSE-ACCEPT rate first (a precision number: a user sign accepted for a word the signer
    did not enrol), per population; INCLUDE-only is the row to compare packs on (docs 15.27)."""
    pops = [("INCLUDE only", e4)] + ([("all sources", e4["all_sources"])] if isinstance(e4.get("all_sources"), dict)
                                     and e4["all_sources"].get("run") else [])
    if not any(p.get("run") for _, p in pops):
        return _e4_like_for_like_lines(e4.get("like_for_like"))
    lines = ["", f"## E4 My signs (user-sign classifier term: {e4.get('user_cls_mode', 'penalty')}, policy "
             f"{e4.get('policy', 'balanced')}: {e4.get('rule', '-')})", "",
             "False accept = a user sign accepted on a word the signer did not enrol (lower is better). 'auto' = what "
             "the app accepts by itself; 'threshold rule' = the same rule without the certification gate (the user "
             "sign it would put first and ask to confirm). INCLUDE only is the row to compare packs on.", "",
             "| population | k | n | groups | FALSE ACCEPT auto | FALSE ACCEPT threshold rule | other-word takes | "
             "without | with | 95% CI (with) | gain |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, p in pops:
        if not p.get("run"):
            lines.append(f"| {name} | - | - | - | - | - | - | - | - | - | not run: {p.get('reason', '')} |")
            continue
        for k, v in p["by_k"].items():
            lines.append(f"| {name} | {k} | {v['n']} | {v.get('n_groups', '-')} | "
                         f"**{_fmt(v.get('user_sign_false_accept_rate'))}** | "
                         f"**{_fmt(v.get('user_sign_false_accept_rate_threshold_rule'))}** | "
                         f"{v.get('n_other_words', '-')} | {_fmt(v['top1_without_enrolment'])} | "
                         f"{_fmt(v['top1_with_enrolment'])} | "
                         f"{format_interval(v.get('interval_with'), v.get('n_groups'), v.get('interval_with_reason'))} | "
                         f"{_fmt(v['gain'])} |")
    refused = [(name, k, v) for name, p in pops if p.get("run") for k, v in p["by_k"].items()
               if v.get("enrolment_refused")]
    if refused:
        lines += ["", "Enrolments refused by the store's consistency rule (c_min "
                  f"{refused[0][2].get('enrol_c_min')}; those words were tested as not enrolled): "
                  + "; ".join(f"{name} {k}: {v['enrolment_refused']} of {v.get('enrolments')}"
                              for name, k, v in refused)]
    return lines + _e4_like_for_like_lines(e4.get("like_for_like"))


def _e4_like_for_like_lines(lf: dict | None) -> list[str]:
    if not lf:
        return []
    r = lf.get("reference") or {}
    head = (f"E4 like for like against {r.get('profile') or '-'} ({r.get('path')}), user-sign false-accept rate "
            f"(threshold rule, INCLUDE only, {lf.get('k')}):")
    if not lf.get("run"):
        return ["", head, f"* not compared: {lf.get('reason')}"]
    return ["", head,
            f"* **{'PASS' if lf.get('pass') else 'FAIL'}**: this pack {_fmt(lf.get('this_rate'))} (other-word takes "
            f"{lf.get('this_n_other_words', '-')}) vs reference {_fmt(lf.get('reference_rate'))} (other-word takes "
            f"{lf.get('reference_n_other_words', '-')}); rule: {lf.get('rule')}",
            *[f"* note: {n}" for n in lf.get("notes") or []]]


def markdown_report(doc: dict) -> str:
    partial = bool(doc.get("partial_data")) and not doc.get("smoke")
    title = f"# ISL evaluation report ({doc['profile']})" + (" - PARTIAL DATA" if partial else "")
    lines = [title, "", f"**{doc['label']}**", "",
             f"Created {doc['created']} at commit {doc.get('git_commit')}. Intervals: 95 % bootstrap over signer "
             "groups, shown only with at least 5 groups (the group count is printed next to every interval). "
             "Accepted = shown as the answer without asking; wrong accept = accepted and wrong / all queries. Not "
             "sure: E1a/E1b/E1c = 1 - accepted (the classifier alone, as in the app's sliding mode, at the "
             "classifier's own validation temperature and threshold; it has no 'did you mean' step); E2/E3 = abstain "
             "rate only (the rest, 1 - accepted - not sure, are 'did you mean' suggestions).", "",
             *_headline_lines(doc),
             "| Protocol | n | groups | top-1 | 95% CI | top-5 | accepted | not sure | wrong accept | label |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for pid, p in doc["protocols"].items():
        if p.get("run") is False or (not p.get("metrics") and p.get("run") is None and not p.get("n")):
            lines.append(f"| {pid} | - | - | - | - | - | - | - | - | "
                         f"not run: {p.get('reason') or p.get('leakage') or ''} |")
            continue
        m = p.get("metrics") or {}
        if "top1" not in m:
            lines.append(f"| {pid} | {p.get('n', '-')} | {p.get('n_groups', '-')} | - | - | - | - | - | - | "
                         f"{p.get('label', '')} (details below) |")
            continue
        lines.append(f"| {pid} | {p.get('n')} | {p.get('n_groups', '-')} | {_fmt(m.get('top1'))} | {_ci_cell(p)} | "
                     f"{_fmt(m.get('top5'))} | {_fmt(m.get('coverage'))} | {_fmt(m.get('not_sure_rate'))} | "
                     f"{_fmt(m.get('wrong_accept_rate'))} | "
                     f"{p.get('label', '')} |")
    cvb = doc["protocols"].get("E1b", {}).get("cv")
    if cvb and cvb.get("run") is False:
        lines += ["", f"E1b cross-validation NOT run: {cvb.get('reason', '')} ("
                  + "; ".join(f"{f['fold']}: {f['signer_grouping']}" for f in cvb.get("not_pooled") or []) + ")"]
    elif cvb:
        ci = cvb.get("pooled_top1_interval")
        lines += ["", f"E1b cross-validation: {cvb['folds']} folds, top-1 mean {cvb['top1_mean']:.3f} (sd "
                  f"{cvb['top1_sd']:.3f}); pooled over {cvb['pooled_groups']} held-out signer groups: "
                  f"{cvb['pooled_top1']:.3f} (95% CI " + format_interval(ci, cvb['pooled_groups'],
                                                                       cvb.get('pooled_top1_interval_reason')) + ")"]
        if cvb.get("not_pooled"):
            lines.append("NOT pooled (other signer grouping): " + ", ".join(
                f"{f['fold']} ({f['signer_grouping']})" for f in cvb["not_pooled"]))
    lines += _like_for_like_lines((doc["protocols"].get("E1b") or {}).get("like_for_like"))
    lines += _alias_lines(doc)
    oov = (doc["protocols"].get("E2") or {}).get("oov_false_accept")
    if oov:
        lines += ["", f"E2 held-out concepts against a gallery WITHOUT them (n {oov.get('n')}; every accept is wrong): "
                  + "; ".join(f"{pol} false accept {_fmt((oov.get(pol) or {}).get('false_accept_rate'))} (threshold "
                              f"rule {_fmt((oov.get(pol) or {}).get('threshold_false_accept_rate'))}), out of "
                              f"vocabulary {_fmt((oov.get(pol) or {}).get('out_of_vocabulary_rate'))}"
                              for pol in ("strict", "balanced", "lenient"))]
    lines += _policy_lines(doc)
    lines += _e4_lines(doc.get("protocols", {}).get("E4") or {})
    guard = (doc.get("data") or {}).get("readiness")
    if guard:
        lines += ["", "## Data readiness", "",
                  f"* data guard at the run's start ({guard.get('checked') or '-'}): "
                  + ("ready" if guard.get("ready") else "NOT ready (run started with --allow-partial)")]
        lines += [f"* not ready: {r}" for r in (guard.get("reasons") or [])[:20]]
        lines += [f"* warning: {w}" for w in guard.get("warnings") or []]
        if guard.get("error"):
            lines.append(f"* {guard['error']}")
    lg = doc.get("leakage_guards") or {}
    pc = lg.get("pretraining_check") or {}
    lines += ["", "## Leakage guards", "",
              f"* classifier split checked: {lg.get('classifier_split_checked')} (a classifier without a recorded split "
              "is refused, and with it E3 and E4, which use the same classifier)"
              + (f"; REFUSED: {lg['classifier_refused']}" if lg.get("classifier_refused") else ""),
              f"* pre-training excludes evaluation clips: {_fmt(lg.get('pretraining_excludes_eval_clips'))} "
              + (f"(computed: {pc.get('overlap')} of {pc.get('pretrain_clips')} pre-training clips are evaluation "
                 f"clips or their duplicates)" if pc.get("pretrain_clips") is not None else f"({pc.get('reason', '')})")]
    if pc.get("sign_library"):
        lines.append(f"* OpenHands Sign Library: {pc['sign_library']}")
    cs = lg.get("class_space") or {}
    if cs.get("classifier") or cs.get("encoder"):  # docs 15.32
        lines.append(f"* class space: classifier signer grouping "
                     f"{(cs.get('classifier') or {}).get('signer_grouping', '-')}, encoder trained without "
                     f"{', '.join((cs.get('encoder') or {}).get('excluded_sources') or []) or 'no source'}")
    for f in cs.get("cv_folds_not_pooled") or []:
        lines.append(f"* cv {f['fold']} NOT pooled: its classes were chosen with signer grouping "
                     f"{f['signer_grouping']}, the main classifier's with {f['main_signer_grouping']}")
    for f in cs.get("oneshot_folds_other_sources") or []:
        lines.append(f"* E2 fold {f['fold']} encoder was trained without "
                     f"{', '.join(f['excluded_sources']) or 'no source'}, the main encoder without "
                     f"{', '.join(f['main_excluded_sources']) or 'no source'}")
    if cs.get("fusion_mismatch"):
        lines.append("* **fusion.json was fitted on another class space (" + ", ".join(cs["fusion_mismatch"])
                     + "): run fit-fusion again; the fused numbers above use stale parameters**")
    if lg.get("pretraining_excludes_eval_clips") is False:
        lines.append("* **LEAKAGE: pre-training saw evaluation clips; the numbers above are not clean.** Examples: "
                     + ", ".join(pc.get("overlap_examples") or []))
    for k in ("E5", "E6", "E7", "E8"):
        p = doc["protocols"].get(k)
        if p:
            lines += ["", f"## {k}", ""]
            if k == "E8" and p.get("trim_eval_check"):
                c = p["trim_eval_check"]
                lines.append(f"* trim-eval: **{c.get('status')}** - file {c.get('file')} (written "
                             f"{c.get('written_at') or c.get('file_mtime_utc') or '-'}), evaluated "
                             f"{c.get('trim_version') or '-'} (model "
                             f"{((c.get('learned_model') or {}).get('sha256') or 'none')[:12]}); run's trims "
                             f"{c.get('run_trim_versions')}, model hashes "
                             f"{ {k[:12]: v for k, v in (c.get('run_model_hashes') or {}).items()} }"
                             + (f"; {c['reason']}" if c.get("reason") else ""))
                syn = (p.get("trim_eval") or {}).get("synthetic") or {}
                if syn:
                    lines.append(f"* synthetic end error: within 300 ms {_fmt(syn.get('within_300ms'))}, median "
                                 f"{syn.get('median_abs_err_ms')} ms, late {_fmt(syn.get('late_frac'))}, early "
                                 f"{_fmt(syn.get('early_frac'))} (n {syn.get('n')}, learned ends "
                                 f"{syn.get('learned_ends')})")
                lines.append("")
            if k == "E5":
                lines += _e5_lines(p)
            lines += ["```", json.dumps({x: y for x, y in p.items() if x not in ("trim_eval", "per_letter", "data")},
                                        indent=1, default=str)[:1500], "```"]
    return "\n".join(lines) + "\n"
