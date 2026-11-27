"""Out-of-fold calibration clips for INCLUDE (docs/ISL_DESIGN.md 15.42, QA critic 8).

The problem. fit-fusion certified INCLUDE on the primary split's validation part (856 clips, 4 pseudo-signer
groups). Those are the clips that also selected the classifier (early stopping, temperature, quota). So the
certification reused the selection data, and 856 clips could not reach a 0.95 lower bound anyway.

The fix. The cross-validation classifiers ``<work>/cv/foldK/classifier.pt`` (K = 1..4) were each trained and selected
without the test part of their own fold. Those four test parts are, together, exactly the primary split's train and
validation INCLUDE clips (about 3,400 clips over every session); fold 0's test part (the E1b test) is never among them.
Each clip is scored by ITS fold's classifier; the rest of the fused system (encoder, fusion weights, temperature,
aliases) is the primary one, and the retrieval gallery leaves out the clip's own fold (see below).

Assumptions, stated:
* **Exchangeable fold classifiers.** At run time the app uses the fold-0 classifier. A fold-K classifier is trained
  with the same code and settings on another 3/5 of the INCLUDE sessions (``check_fold_classifier`` refuses one of
  other settings), so its accuracy on unseen sessions is taken to be that of the fold-0 classifier. The class lists
  differ a little (a word with too few signers in one fold's train part is not a class there).
* **The encoder.** There is one encoder, trained on the primary train part. For the folds whose test part lies in
  that train part (folds 2-4) the encoder has seen the clips: retrieval is then optimistic. Two defences:
  1. *Leave-fold-out gallery.* Fold K's clips are scored against the training-only index with every row built from a
     fold-K test clip (or a duplicate of one) removed; an INCLUDE centroid is recomputed from its other clips. So no
     clip ever retrieves itself.
  2. *An in-sample screen.* Retrieval top-1 on the in-sample folds minus the out-of-sample fold (fold 1 = the
     primary validation part, which the encoder never trained on), minus the same difference for the fold
     classifiers (all out of sample: a control for fold difficulty). This difference-in-differences is the
     encoder's in-sample advantage. Above ``max_encoder_gain`` (default 0.03) the in-sample folds are NOT used for
     certification (only the out-of-sample folds are). Passing the screen is not proof of no bias (its 95 % interval
     is reported); a per-fold encoder would remove the assumption.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from app.features.build import FeatureError, build_features
from app.isl.data.sources import Clip

OOF_VERSION = "oof-1"
DEFAULT_MAX_ENCODER_GAIN = 0.03
CENTROID_MIN_COS = 0.995  # a recomputed full INCLUDE centroid must match the stored row (float16) this closely
OOF_SOURCE = "include"
ASSUMPTION = ("fold classifiers are exchangeable: a cv/foldK classifier (same code and settings, another 3/5 of the "
              "INCLUDE sessions) is taken to be as accurate on unseen sessions as the fold-0 classifier the app uses; "
              "the encoder, fusion weights and temperature are the primary ones (docs 15.42)")


class OofError(ValueError):
    """Out-of-fold calibration cannot be set up (missing classifier, other settings, unknown split)."""


@dataclass
class OofPart:
    fold: int
    split: str
    classifier_path: Path
    clips: list[Clip]
    # share of the part's clips in the primary split's TRAIN part (the encoder and the index were built from them)
    encoder_in_sample: float = 0.0

    @property
    def in_sample(self) -> bool:
        return self.encoder_in_sample > 0


@dataclass
class OofCalibration:
    """What ``fit_fusion(..., oof=...)`` needs: the parts, the run's data, optional clip -> session clusters for the
    in-sample screen's bootstrap, the screen's threshold and whether in-sample folds are kept anyway."""
    parts: list[OofPart]
    data: object
    sessions: dict[str, str] = field(default_factory=dict)
    max_encoder_gain: float = DEFAULT_MAX_ENCODER_GAIN
    keep_in_sample: bool = False
    n_boot: int = 1000
    seed: int = 0
    progress: object = None  # callable(str) for log lines, or None


# --------------------------------------------------------------------------------------------- setup

def split_base(name: str) -> tuple[str, int]:
    """``include-session-cv5-fold0`` -> (``include-session-cv5``, 0). OofError for a split without folds."""
    m = re.fullmatch(r"(.+)-fold(\d+)", str(name or ""))
    if not m:
        raise OofError(f"split {name!r} is not a fold of a cross-validation split: out-of-fold calibration needs one")
    return m.group(1), int(m.group(2))


def oof_parts(paths, data, work: Path, folds=None) -> list[OofPart]:
    """The INCLUDE test clips of every other fold of the primary split (``folds``: None = every fold file but the
    primary one), each with its ``<work>/cv/foldK/classifier.pt``. OofError for the primary fold, a missing split
    file or classifier, or no clip at all."""
    from app.isl.training.common import load_split

    base, primary = split_base(data.split.get("name"))
    if folds is None:
        folds = sorted(int(m.group(1)) for f in Path(paths.splits).glob(f"{base}-fold*.json")
                       if (m := re.fullmatch(rf"{re.escape(base)}-fold(\d+)\.json", f.name)))
        folds = [k for k in folds if k != primary]
    by_id = {c.clip_id: c for c in data.include if c.concept_id}
    parts = []
    for k in folds:
        k = int(k)
        if k == primary:
            raise OofError(f"fold {k} is the primary split ({data.split.get('name')}): its test part is the E1b test "
                           "and is never used to calibrate")
        name = f"{base}-fold{k}"
        try:
            assign = load_split(paths, name)["assign"]
        except FileNotFoundError as e:
            raise OofError(str(e)) from None
        ck = Path(work) / "cv" / f"fold{k}" / "classifier.pt"
        if not ck.exists():
            raise OofError(f"no cross-validation classifier {ck} (run the driver's cv step first)")
        clips = [by_id[i] for i, p in sorted(assign.items()) if p == "test" and i in by_id]
        ins = sum(1 for c in clips if data.assign.get(c.clip_id) == "train")
        parts.append(OofPart(k, name, ck, clips, ins / len(clips) if clips else 0.0))
    if not any(p.clips for p in parts):
        raise OofError("no out-of-fold INCLUDE clip found (no other fold, or no labelled clip in their test parts)")
    return parts


def check_oof_clips(parts: list[OofPart], data) -> dict:
    """Leak check: no out-of-fold clip (nor a duplicate of one) is in the primary split's TEST part (INCLUDE test or
    another source's test signers), and the parts do not overlap. ``LeakageError`` / OofError otherwise."""
    from app.isl.models.index import LeakageError

    test = {c.clip_id for c in data.include if data.assign.get(c.clip_id) == "test"}
    test |= {c.clip_id for c in getattr(data, "extra", []) if data.extra_assign.get(c.clip_id) == "test"}
    seen: dict[str, int] = {}
    for p in parts:
        for c in p.clips:
            if c.clip_id in seen:
                raise OofError(f"clip {c.clip_id} is in the test part of folds {seen[c.clip_id]} and {p.fold}")
            seen[c.clip_id] = p.fold
    hit = sorted(set(seen) & test)
    dedup = getattr(data, "dedup", None)
    dup_hit: list[str] = []
    if dedup is not None and getattr(dedup, "parent", None):
        tg = {dedup.group(i) for i in test}
        dup_hit = sorted(i for i in seen if dedup.group(i) in tg and i not in test)
    if hit or dup_hit:
        ex = (hit or dup_hit)[:3]
        raise LeakageError(f"{len(hit)} out-of-fold calibration clips are in the primary split's test part and "
                           f"{len(dup_hit)} duplicate a test clip (e.g. {ex}): the E1b test is never used to "
                           "calibrate (docs 15.42)")
    return {"oof_clips": len(seen), "primary_test_clips": len(test), "in_primary_test": 0, "duplicates_of_test": 0,
            "dedup_checked": bool(dedup is not None and getattr(dedup, "parent", None))}


# training settings a fold classifier must share with the primary one (exchangeability, docs 15.42)
_SAME_KEYS = ("profile", "architecture", "model_kwargs", "feature_config")


def _settings(meta: dict) -> dict:
    from app.isl.training.common import class_space_of

    tc = {k: v for k, v in (meta.get("train_config") or {}).items() if k not in ("device",)}
    d = meta.get("data") or {}
    q = meta.get("source_quota")
    return {**{k: meta.get(k) for k in _SAME_KEYS}, "train_config": tc,
            "class_space": {k: v for k, v in class_space_of(meta).items() if k in ("signer_grouping",
                                                                                   "excluded_sources")},
            "source_quota": (q or {}).get("shares") if isinstance(q, dict) else q,
            "select_on": meta.get("select_on") or "macro_f1",
            "min_signers": d.get("min_signers"), "class_power": d.get("class_power"),
            "signer_split": d.get("signer_split") or "v1",
            "extra_sources": sorted((d.get("extra_sources") or {}).keys())}


def check_fold_classifier(main_meta: dict, fold_meta: dict, split_name: str) -> list[str]:
    """What makes a fold classifier NOT exchangeable with the primary one (empty list = fine): another split than
    its fold, or other training settings (architecture, sizes, features, train config, class grouping, source
    quota, selection, signer split, sources)."""
    out = []
    if fold_meta.get("split") != split_name:
        out.append(f"split {fold_meta.get('split')!r} != {split_name!r}")
    a, b = _settings(main_meta), _settings(fold_meta)
    out += [f"{k}: {b[k]!r} != primary {a[k]!r}" for k in a if a[k] != b[k]]
    return out


def load_sessions(paths) -> dict[str, str]:
    """INCLUDE clip id -> recording session (``splits/include_groups.json``; {} when missing): the bootstrap clusters
    of the in-sample screen (sessions never cross folds)."""
    f = Path(paths.splits) / "include_groups.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text())
    except ValueError:
        return {}
    return {k: str(v.get("session")) for k, v in (d.get("clips") or {}).items() if isinstance(v, dict)
            and v.get("session")}


# --------------------------------------------------------------------------------------------- gallery

def include_embeddings(enc, clips: list[Clip], chunk: int = 256) -> dict[str, np.ndarray]:
    """clip id -> the embedding ``build_index`` averages into an INCLUDE centroid (default slice, one crop), computed
    in chunks so only ``chunk`` clips' landmarks are in memory at a time."""
    from app.isl.training.build_index import _embed_feats
    from app.isl.training.common import ClipDataset, items_for

    fcfg = enc.feature_config
    out: dict[str, np.ndarray] = {}
    lab = [c for c in clips if c.concept_id]
    for a in range(0, len(lab), chunk):
        part = lab[a:a + chunk]
        items = items_for(part, {c.concept_id: 0 for c in part}, fcfg, with_alternatives=False)
        if not items:
            continue
        z = _embed_feats(enc, ClipDataset(items, fcfg, train=False)._cache)
        out.update({it.clip.clip_id: v for it, v in zip(items, z)})
    return out


def _row_ids(r: dict) -> list[str]:
    return ([r["clip"]] if r.get("clip") else []) + list(r.get("clips") or [])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(1e-8, float(np.linalg.norm(v)))


def check_centroids(index, emb_of: dict[str, np.ndarray]) -> dict:
    """The INCLUDE centroids rebuilt from ``emb_of`` must match the stored rows (cosine >= CENTROID_MIN_COS): proof
    that the leave-fold-out centroids are computed the way the index step computed them. OofError otherwise."""
    cos = []
    for i, r in enumerate(index.rows):
        if r.get("source") == "include" and r.get("kind") == "centroid":
            ids = list(r.get("clips") or [])
            if not ids or any(x not in emb_of for x in ids):
                continue
            cos.append(float(_unit(np.mean([emb_of[x] for x in ids], axis=0)) @ index.emb[i]))
    if not cos:
        return {"checked": 0}
    if min(cos) < CENTROID_MIN_COS:
        raise OofError(f"the index's INCLUDE centroids cannot be rebuilt from the clips (min cosine {min(cos):.4f} < "
                       f"{CENTROID_MIN_COS}): rebuild the index with this encoder")
    return {"checked": len(cos), "min_cos": round(min(cos), 6)}


def fold_gallery(index, banned: set[str], emb_of: dict[str, np.ndarray]):
    """The index without anything built from a ``banned`` clip: an INCLUDE centroid is recomputed from its other clips
    (dropped when none is left), any other row holding a banned clip is dropped. Returns (index, summary)."""
    from app.isl.models.index import EmbeddingIndex

    rows, vecs = [], []
    st: Counter = Counter()
    for i, r in enumerate(index.rows):
        ids = _row_ids(r)
        if not any(x in banned for x in ids):
            rows.append(r)
            vecs.append(index.emb[i])
            continue
        if r.get("source") == "include" and r.get("kind") == "centroid":
            keep = [x for x in r.get("clips") or [] if x not in banned]
            if not keep:
                st["include_centroids_dropped"] += 1
                continue
            miss = [x for x in keep if x not in emb_of]
            if miss:
                raise OofError(f"no embedding for INCLUDE clips {miss[:3]} of the centroid of {r.get('concept_id')}")
            rows.append({**r, "n_clips": len(keep), "clips": sorted(keep), "oof_recomputed": True})
            vecs.append(_unit(np.mean([emb_of[x] for x in keep], axis=0)))
            st["include_centroids_recomputed"] += 1
        else:
            st[f"dropped_{r.get('source')}_{r.get('kind')}"] += 1
    if not rows:
        raise OofError("the leave-fold-out gallery is empty")
    g = EmbeddingIndex(np.stack(vecs).astype(np.float32), rows, replace(index.meta, n=len(rows)))
    left = g.row_clip_ids() & banned
    if left:  # never: the loop above removed them (the check the plan asks for, on the gallery itself)
        raise OofError(f"the leave-fold-out gallery still holds {len(left)} clips of the fold")
    return g, {"rows": len(rows), "rows_before": len(index.rows), **dict(sorted(st.items()))}


def banned_ids(part: OofPart, index, data) -> set[str]:
    """The fold's test clips plus every index-row clip in the duplicate group of one (``data.dedup``)."""
    ids = {c.clip_id for c in part.clips}
    dedup = getattr(data, "dedup", None)
    if dedup is not None and getattr(dedup, "parent", None):
        groups = {dedup.group(i) for i in ids}
        ids |= {x for x in index.row_clip_ids() if dedup.group(x) in groups}
    return ids


# --------------------------------------------------------------------------------------------- evidence

def _top(ids, mat) -> str | None:
    if mat is None or ids is None or not len(ids):
        return None
    return ids[int(np.argmax(np.asarray(mat).reshape(-1, len(ids)).mean(0)))]


def oof_evidence(rec, oof: OofCalibration, params, progress=None) -> dict:
    """Score every part's clips with its fold classifier, the primary encoder and the leave-fold-out gallery, and
    fuse them with ``params``. Returns rows (one dict per clip: fold, clip, source, group, truth, result, the
    classifier-, retrieval- and fused top-1 hits, in_sample, cluster) and per-part summaries."""
    from app.isl.models.fusion import compact_inputs, fuse
    from app.isl.training.common import engine_crops
    from app.models.checkpoint import Predictor

    data = oof.data
    main_meta = getattr(getattr(rec, "classifier", None), "meta", None) or {}
    if getattr(rec, "classifier", None) is None:
        raise OofError("out-of-fold calibration needs the primary classifier (classifier.pt)")
    idx = getattr(rec, "index", None)
    emb_of: dict = {}
    cent: dict = {"checked": 0}
    if idx is not None and getattr(rec, "encoder", None) is not None:
        need = sorted({x for r in idx.rows if r.get("source") == "include" for x in _row_ids(r)})
        by_id = {c.clip_id: c for c in data.include}
        emb_of = include_embeddings(rec.encoder, [by_id[x] for x in need if x in by_id])
        cent = check_centroids(idx, emb_of)
    taus = [float(params.tau_r)]
    t_cs = (float(params.t_c),)
    rows: list[dict] = []
    parts_rep = []
    for p in oof.parts:
        t0 = time.time()
        clf = Predictor(p.classifier_path)
        bad = check_fold_classifier(main_meta, clf.meta, p.split)
        if bad:
            raise OofError(f"{p.classifier_path} is not exchangeable with the primary classifier: {'; '.join(bad)}")
        r = copy.copy(rec)
        r.classifier = clf
        r._cls_cal = None
        r._class_set = None
        grep: dict = {}
        if idx is not None:
            r.index, grep = fold_gallery(idx, banned_ids(p, idx, data), emb_of)
        pos = r.index.concept_pos if getattr(r, "index", None) is not None else None
        n0 = len(rows)
        for c in p.clips:
            try:
                raw = c.load()
                feats = np.stack([build_features(x, rec.feature_config) for x in engine_crops(raw)])
            except (FeatureError, OSError, ValueError):
                continue
            inp, _ = r.evidence(feats, use_my_signs=False)
            c_top = _top(inp.cls_ids, inp.cls_logp)
            r_top = _top(inp.ret_ids, inp.ret)
            inp = compact_inputs(inp, taus, params.top_n, ret_pos=pos if inp.ret_ids is getattr(r.index, "concepts",
                                                                                                  None) else None,
                                 t_cs=t_cs)
            res = fuse(inp, params)
            del inp
            rows.append({"fold": p.fold, "clip": c.clip_id, "source": OOF_SOURCE, "group": c.group,
                         "truth": c.concept_id, "result": res, "in_sample": data.assign.get(c.clip_id) == "train",
                         "c_hit": c_top == c.concept_id, "r_hit": r_top == c.concept_id,
                         "f_hit": bool(res.keys) and res.keys[0] == c.concept_id,
                         "cluster": oof.sessions.get(c.clip_id) or c.clip_id})
        got = rows[n0:]
        cls_names = list(clf.class_names)
        parts_rep.append({
            "fold": p.fold, "split": p.split, "classifier": str(p.classifier_path),
            "classifier_sha256": hashlib.sha256(Path(p.classifier_path).read_bytes()).hexdigest(),
            "classes": len(cls_names), "classes_shared_with_primary": len(set(cls_names) &
                                                                          set(rec.classifier.class_names)),
            "clips": len(p.clips), "scored": len(got), "encoder_in_sample": round(p.encoder_in_sample, 4),
            "groups": len({x["group"] for x in got if x["group"]}),
            "classifier_top1": _rate(got, "c_hit"), "retrieval_top1": _rate(got, "r_hit"),
            "fused_top1": _rate(got, "f_hit"), "gallery": grep, "seconds": round(time.time() - t0, 1)})
        if progress:
            progress(f"oof fold{p.fold}: {len(got)} clips, classifier top-1 {parts_rep[-1]['classifier_top1']}, "
                     f"retrieval {parts_rep[-1]['retrieval_top1']}, fused {parts_rep[-1]['fused_top1']}")
        del clf, r
    return {"rows": rows, "parts": parts_rep, "centroid_check": cent}


def _rate(rows, key) -> float | None:
    return round(float(np.mean([bool(x[key]) for x in rows])), 4) if rows else None


# --------------------------------------------------------------------------------------------- in-sample screen

def _did(rows_in, rows_out, key) -> float:
    m = lambda rr, k: float(np.mean([bool(x[k]) for x in rr]))  # noqa: E731
    return (m(rows_in, key) - m(rows_out, key)) - (m(rows_in, "c_hit") - m(rows_out, "c_hit"))


def encoder_in_sample_check(rows: list[dict], max_gain: float = DEFAULT_MAX_ENCODER_GAIN, n_boot: int = 1000,
                            seed: int = 0) -> dict:
    """Difference-in-differences of top-1 between the folds whose clips the encoder trained on (in) and those it did
    not (out): (retrieval in - out) - (fold classifier in - out), and the same for the fused top-1. The classifier
    difference controls for fold difficulty (every fold classifier is out of sample). 95 % interval from a cluster
    bootstrap (recording sessions, resampled within each arm). ``passed`` = retrieval DiD <= ``max_gain``.
    Not testable (no out-of-sample or no in-sample clip): ``passed`` None."""
    rin = [x for x in rows if x["in_sample"]]
    rout = [x for x in rows if not x["in_sample"]]
    rep = {"n_in": len(rin), "n_out": len(rout), "max_gain": max_gain,
           "statistic": "(retrieval top-1 in - out) - (fold classifier top-1 in - out); in = folds whose clips the "
                        "encoder trained on"}
    if not rin or not rout:
        return {**rep, "testable": False, "passed": None,
                "reason": "needs clips the encoder trained on AND clips it did not (fold 1 = primary validation)"}
    for arm, rr in (("in", rin), ("out", rout)):
        rep[arm] = {k: _rate(rr, k) for k in ("c_hit", "r_hit", "f_hit")} | {
            "clusters": len({x["cluster"] for x in rr})}
    did_r, did_f = _did(rin, rout, "r_hit"), _did(rin, rout, "f_hit")
    rng = np.random.default_rng(seed)

    def clusters(rr):
        by = defaultdict(list)
        for x in rr:
            by[x["cluster"]].append(x)
        return list(by.values())

    cin, cout = clusters(rin), clusters(rout)
    br, bf = [], []
    for _ in range(max(0, int(n_boot))):
        a = [x for j in rng.integers(0, len(cin), len(cin)) for x in cin[j]]
        b = [x for j in rng.integers(0, len(cout), len(cout)) for x in cout[j]]
        br.append(_did(a, b, "r_hit"))
        bf.append(_did(a, b, "f_hit"))
    ci = (lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
          if v else None)
    passed = bool(did_r <= max_gain)
    return {**rep, "testable": True, "did_retrieval": round(did_r, 4), "did_retrieval_ci95": ci(br),
            "did_fused": round(did_f, 4), "did_fused_ci95": ci(bf), "n_boot": int(n_boot), "passed": passed,
            "note": "a screen, not a proof: passing means the in-sample advantage was not seen above the margin; the "
                    "interval shows how large it could still be"}


def select_rows(ev: dict, oof: OofCalibration) -> tuple[list[dict], dict]:
    """The rows that certify: every out-of-sample fold; the in-sample folds only when the screen passed (or
    ``keep_in_sample``). Returns (rows, report)."""
    rows = ev["rows"]
    chk = encoder_in_sample_check(rows, oof.max_encoder_gain, oof.n_boot, oof.seed)
    use_in = bool(oof.keep_in_sample or chk.get("passed"))
    used = [x for x in rows if (not x["in_sample"]) or use_in]
    folds_used = sorted({x["fold"] for x in used})
    dropped = sorted({x["fold"] for x in rows if x["in_sample"]} - set(folds_used))
    why = None
    if dropped:
        why = ("encoder in-sample screen failed" if chk.get("testable") else
               "encoder in-sample screen not testable") + " (--oof-keep-in-sample uses them anyway)"
    return used, {"version": OOF_VERSION, "assumption": ASSUMPTION, "in_sample_check": chk,
                  "keep_in_sample": bool(oof.keep_in_sample), "folds_used": folds_used, "folds_dropped": dropped,
                  **({"dropped_reason": why} if why else {}),
                  "clips_used": len(used), "groups_used": len({x["group"] for x in used if x["group"]}),
                  "parts": ev["parts"], "centroid_check": ev.get("centroid_check")}
