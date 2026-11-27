"""Out-of-fold calibration clips for INCLUDE (docs/ISL_DESIGN.md 15.42, QA critic 8).

The problem. fit-fusion certified INCLUDE on the primary split's validation part (856 clips, 4 pseudo-signer
groups). Those are the clips that also selected the classifier (early stopping, temperature, quota). So the
certification reused the selection data, and 856 clips could not reach a 0.95 lower bound anyway.

The fix. The cross-validation classifiers ``<work>/cv/foldK/classifier.pt`` (K = 1..4) were each trained and selected
without the test part of their own fold. Those four test parts are, together, exactly the primary split's train and
validation INCLUDE clips (about 3,400 clips over every session); fold 0's test part (the E1b test) is never among them.
Each clip is scored by ITS fold's classifier; the rest of the fused system (encoder, fusion weights, temperature,
aliases) is the primary one, and the retrieval gallery leaves out the clip's own fold (see below).

The E1b test is NOT invisible to the fold classifiers (review of 15.42, docs 15.43). ``grouped_kfold`` gives fold K
validation = fold K+1, so fold 4's classifier was early-stopped and temperature-fitted on fold 0's test part: it
SELECTED on the E1b test. ``oof_parts`` skips such a fold by default (``allow_test_selected`` uses it, recorded).
Folds 1-3 TRAIN on E1b test clips (fold 0 is in their train part). That is training data, not selection: their
probabilities on their own test parts set the INCLUDE thresholds, and E1b precision is then measured at those
thresholds, so E1b is not fully clean of the calibration. Each part records how many of its fold's train and
validation clips (and duplicates of them) are E1b test clips.

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
  2. *An in-sample screen.* Retrieval top-1 on an in-sample fold j minus the out-of-sample fold (fold 1 = the
     primary validation part, which the encoder never trained on) scored against THE SAME leave-fold-j-out gallery,
     minus the same difference for the fold classifiers (all out of sample: a control for fold difficulty). Pooled
     over j (weighted by fold size), this difference-in-differences is the encoder's in-sample advantage. The same
     gallery on both arms matters: a leave-fold-out gallery has fewer clips per INCLUDE centroid, which lowers
     retrieval for reasons that are not memorisation (docs 15.43). Above ``max_encoder_gain`` (default 0.03) the in-sample folds are NOT used for
     certification (only the out-of-sample folds are). Passing the screen is not proof of no bias (its 95 % interval
     is reported); a per-fold encoder would remove the assumption.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import Counter
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
    # clips of this fold's TRAIN / VALIDATION part that are primary (E1b) test clips or duplicates of one (docs 15.43)
    trained_on_primary_test: int = 0
    selected_on_primary_test: int = 0

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
    # provenance (docs 15.43): the folds asked for (None = every fold), whether folds that selected on the E1b test
    # were allowed, and the folds ``oof_parts`` skipped (with their overlap counts)
    requested_folds: list | None = None
    allow_test_selected: bool = False
    skipped: list = field(default_factory=list)


# --------------------------------------------------------------------------------------------- setup

def split_base(name: str) -> tuple[str, int]:
    """``include-session-cv5-fold0`` -> (``include-session-cv5``, 0). OofError for a split without folds."""
    m = re.fullmatch(r"(.+)-fold(\d+)", str(name or ""))
    if not m:
        raise OofError(f"split {name!r} is not a fold of a cross-validation split: out-of-fold calibration needs one")
    return m.group(1), int(m.group(2))


def primary_test_ids(data) -> tuple[set[str], object]:
    """The primary split's test clip ids (the E1b test) and a predicate "is this clip one of them or a duplicate of
    one" (``data.dedup``)."""
    test = {i for i, p in (getattr(data, "assign", None) or {}).items() if p == "test"}
    dedup = getattr(data, "dedup", None)
    if dedup is not None and getattr(dedup, "parent", None):
        tg = {dedup.group(i) for i in test}
        return test, (lambda i: i in test or dedup.group(i) in tg)
    return test, (lambda i: i in test)


def oof_parts(paths, data, work: Path, folds=None, allow_test_selected: bool = False,
              skipped: list | None = None) -> list[OofPart]:
    """The INCLUDE test clips of every other fold of the primary split (``folds``: None = every fold file but the
    primary one), each with its ``<work>/cv/foldK/classifier.pt``. OofError for the primary fold, a missing split
    file or classifier, or no clip at all.

    A fold whose VALIDATION part holds primary test clips (or duplicates of them) selected its classifier on the
    E1b test (``grouped_kfold``: fold K validation = fold K+1, so fold 4 of 5 validates on fold 0). It is skipped
    unless ``allow_test_selected``; ``skipped`` (a list) gets one dict per skipped fold. Every part records how many
    of its train / validation clips are primary test clips (docs 15.43)."""
    from app.isl.training.common import load_split

    base, primary = split_base(data.split.get("name"))
    if folds is None:
        folds = sorted(int(m.group(1)) for f in Path(paths.splits).glob(f"{base}-fold*.json")
                       if (m := re.fullmatch(rf"{re.escape(base)}-fold(\d+)\.json", f.name)))
        folds = [k for k in folds if k != primary]
    skipped = [] if skipped is None else skipped
    _, in_test = primary_test_ids(data)
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
        tr = sum(1 for i, p in assign.items() if p == "train" and in_test(i))
        va = sum(1 for i, p in assign.items() if p == "val" and in_test(i))
        if va and not allow_test_selected:
            skipped.append({"fold": k, "split": name, "selected_on_primary_test_clips": va,
                            "trained_on_primary_test_clips": tr,
                            "reason": f"its validation part holds {va} primary (E1b) test clips: its classifier was "
                                      "early-stopped and temperature-fitted on the E1b test (--oof-allow-test-selected "
                                      "uses it anyway)"})
            continue
        ck = Path(work) / "cv" / f"fold{k}" / "classifier.pt"
        if not ck.exists():
            raise OofError(f"no cross-validation classifier {ck} (run the driver's cv step first)")
        clips = [by_id[i] for i, p in sorted(assign.items()) if p == "test" and i in by_id]
        ins = sum(1 for c in clips if data.assign.get(c.clip_id) == "train")
        parts.append(OofPart(k, name, ck, clips, ins / len(clips) if clips else 0.0, tr, va))
    if not any(p.clips for p in parts):
        why = f" ({len(skipped)} folds skipped: they selected on the E1b test)" if skipped else ""
        raise OofError("no out-of-fold INCLUDE clip found (no other fold, or no labelled clip in their test parts)"
                       + why)
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
    return g, {"rows": len(rows), "rows_before": len(index.rows), **dict(sorted(st.items())),
               "include_support": centroid_support(rows)}


def centroid_support(rows: list[dict]) -> dict:
    """INCLUDE centroids of a gallery and their mean number of clips (docs 15.43: the gallery support per arm)."""
    n = [len(r.get("clips") or []) or int(r.get("n_clips") or 0) for r in rows
         if r.get("source") == "include" and r.get("kind") == "centroid"]
    return {"centroids": len(n), "mean_clips": round(float(np.mean(n)), 3) if n else None}


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
    classifier-, retrieval- and fused top-1 hits, in_sample, cluster) and per-part summaries.

    The out-of-sample parts (no clip the encoder trained on) are scored first and their query embeddings and
    classifier outputs kept; every in-sample fold j's gallery then also scores them (``r_hit_g[j]``,
    ``f_hit_g[j]``), so the in-sample screen compares both arms on the same gallery (docs 15.43). Rows whose truth
    is a primary classifier class the fold classifier does not have are dropped (``dropped_not_a_fold_class``): such
    a word would be retrieval-only for the fold system but is judged as a classifier word."""
    from app.isl.models.fusion import FusionInputs, compact_inputs, fuse
    from app.isl.training.common import engine_crops
    from app.models.checkpoint import Predictor

    data = oof.data
    main_meta = getattr(getattr(rec, "classifier", None), "meta", None) or {}
    if getattr(rec, "classifier", None) is None:
        raise OofError("out-of-fold calibration needs the primary classifier (classifier.pt)")
    primary_classes = set(rec.classifier.class_names)
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
    cache: list[tuple[dict, np.ndarray | None, np.ndarray | None, list]] = []  # out-of-sample rows: (row, z, logp, ids)

    def fused(inp, r_index):
        pos = r_index.concept_pos if r_index is not None else None
        inp = compact_inputs(inp, taus, params.top_n, ret_pos=pos if r_index is not None and inp.ret_ids is getattr(
            r_index, "concepts", None) else None, t_cs=t_cs)
        return fuse(inp, params)

    # out-of-sample parts first: their cached evidence is re-scored on every in-sample fold's gallery
    for p in sorted(oof.parts, key=lambda q: (q.in_sample, q.fold)):
        t0 = time.time()
        clf = Predictor(p.classifier_path)
        bad = check_fold_classifier(main_meta, clf.meta, p.split)
        if bad:
            raise OofError(f"{p.classifier_path} is not exchangeable with the primary classifier: {'; '.join(bad)}")
        fold_classes = set(clf.class_names)
        missing = primary_classes - fold_classes
        r = copy.copy(rec)
        r.classifier = clf
        r._cls_cal = None
        r._class_set = None
        grep: dict = {}
        if idx is not None:
            r.index, grep = fold_gallery(idx, banned_ids(p, idx, data), emb_of)
        g = getattr(r, "index", None)
        n0 = len(rows)
        n_drop = 0
        for c in p.clips:
            if c.concept_id in missing:
                n_drop += 1
                continue
            try:
                raw = c.load()
                feats = np.stack([build_features(x, rec.feature_config) for x in engine_crops(raw)])
            except (FeatureError, OSError, ValueError):
                continue
            inp, info = r.evidence(feats, use_my_signs=False)
            c_top = _top(inp.cls_ids, inp.cls_logp)
            r_top = _top(inp.ret_ids, inp.ret)
            row = {"fold": p.fold, "clip": c.clip_id, "source": OOF_SOURCE, "group": c.group,
                   "truth": c.concept_id, "in_sample": data.assign.get(c.clip_id) == "train",
                   "c_hit": c_top == c.concept_id, "r_hit": r_top == c.concept_id,
                   "cluster": oof.sessions.get(c.clip_id) or c.clip_id}
            if not p.in_sample:
                cache.append((row, info.get("z"), inp.cls_logp, inp.cls_ids))
            res = fused(inp, g)
            del inp
            row["result"] = res
            row["f_hit"] = bool(res.keys) and res.keys[0] == c.concept_id
            rows.append(row)
        got = rows[n0:]
        gm = None
        if p.in_sample and g is not None and cache:  # the out-of-sample clips on THIS fold's gallery (docs 15.43)
            for row, z, logp, ids in cache:
                if z is None:
                    continue
                inp = FusionInputs(cls_logp=logp, cls_ids=ids, ret=g.concept_matrix(z), ret_ids=g.concepts,
                                   prior=rec.prior)
                row.setdefault("r_hit_g", {})[p.fold] = _top(inp.ret_ids, inp.ret) == row["truth"]
                res = fused(inp, g)
                row.setdefault("f_hit_g", {})[p.fold] = bool(res.keys) and res.keys[0] == row["truth"]
                del inp, res
            gm = {"retrieval_top1": _rate([x for x, *_ in cache if p.fold in x.get("r_hit_g", {})],
                                          None, lambda x: x["r_hit_g"][p.fold]),
                  "n": sum(1 for x, *_ in cache if p.fold in x.get("r_hit_g", {}))}
        cls_names = list(clf.class_names)
        parts_rep.append({
            "fold": p.fold, "split": p.split, "classifier": str(p.classifier_path),
            "classifier_sha256": hashlib.sha256(Path(p.classifier_path).read_bytes()).hexdigest(),
            "classes": len(cls_names), "classes_shared_with_primary": len(set(cls_names) & primary_classes),
            "primary_classes_missing": len(missing), "dropped_not_a_fold_class": n_drop,
            "trained_on_primary_test_clips": p.trained_on_primary_test,
            "selected_on_primary_test_clips": p.selected_on_primary_test,
            "clips": len(p.clips), "scored": len(got), "encoder_in_sample": round(p.encoder_in_sample, 4),
            "groups": len({x["group"] for x in got if x["group"]}),
            "classifier_top1": _rate(got, "c_hit"), "retrieval_top1": _rate(got, "r_hit"),
            "fused_top1": _rate(got, "f_hit"), "gallery": grep,
            **({"out_of_sample_on_this_gallery": gm} if gm else {}), "seconds": round(time.time() - t0, 1)})
        if progress:
            progress(f"oof fold{p.fold}: {len(got)} clips, classifier top-1 {parts_rep[-1]['classifier_top1']}, "
                     f"retrieval {parts_rep[-1]['retrieval_top1']}, fused {parts_rep[-1]['fused_top1']}"
                     + (f"; out-of-sample clips on this gallery: retrieval {gm['retrieval_top1']}" if gm else ""))
        del clf, r
    del cache
    parts_rep.sort(key=lambda x: x["fold"])
    rows.sort(key=lambda x: x["fold"])  # stable: clip order within a fold is kept
    return {"rows": rows, "parts": parts_rep, "centroid_check": cent}


def _rate(rows, key, get=None) -> float | None:
    get = get or (lambda x: x[key])
    return round(float(np.mean([bool(get(x)) for x in rows])), 4) if rows else None


# --------------------------------------------------------------------------------------------- in-sample screen

def _arm_arrays(rin: list[dict], rout: list[dict], folds: list[int]) -> dict:
    """0/1 arrays of the screen. In arm: fold index, classifier / retrieval / fused hits. Out arm: classifier hits and,
    per in-sample fold j, the retrieval / fused hits on fold j's gallery (``r_hit_g[j]``; a row without one, e.g. no
    index, uses its own ``r_hit``: then the arms share no gallery, reported as ``gallery_matched`` False)."""
    fi = {f: i for i, f in enumerate(folds)}
    cl_in = {c: i for i, c in enumerate(sorted({x["cluster"] for x in rin}))}
    cl_out = {c: i for i, c in enumerate(sorted({x["cluster"] for x in rout}))}
    b = lambda v: np.asarray(v, np.float64)  # noqa: E731
    return {
        "J": len(folds), "f": np.asarray([fi[x["fold"]] for x in rin], np.int64),
        "c_in": b([bool(x["c_hit"]) for x in rin]), "r_in": b([bool(x["r_hit"]) for x in rin]),
        "fh_in": b([bool(x["f_hit"]) for x in rin]),
        "c_out": b([bool(x["c_hit"]) for x in rout]),
        "r_out": b([[bool((x.get("r_hit_g") or {}).get(f, x["r_hit"])) for f in folds] for x in rout]),
        "fh_out": b([[bool((x.get("f_hit_g") or {}).get(f, x["f_hit"])) for f in folds] for x in rout]),
        "k_in": np.asarray([cl_in[x["cluster"]] for x in rin], np.int64), "n_k_in": len(cl_in),
        "k_out": np.asarray([cl_out[x["cluster"]] for x in rout], np.int64), "n_k_out": len(cl_out)}


def _did_w(A: dict, w_in: np.ndarray, w_out: np.ndarray, which: str) -> tuple[float, np.ndarray]:
    """Pooled and per-fold DiD with row weights (1 = the sample; bootstrap counts otherwise). DiD_j =
    (hit(fold j | G_-j) - hit(out | G_-j)) - (cls(fold j) - cls(out)); pooled = mean of DiD_j weighted by fold j's
    in-arm weight."""
    J = A["J"]
    hin, hout = (A["r_in"], A["r_out"]) if which == "r" else (A["fh_in"], A["fh_out"])
    n_j = np.bincount(A["f"], weights=w_in, minlength=J)
    W = max(float(w_out.sum()), 1e-12)
    h_out = (w_out @ hout) / W
    c_out = float(w_out @ A["c_out"]) / W
    with np.errstate(invalid="ignore", divide="ignore"):
        h_in = np.bincount(A["f"], weights=w_in * hin, minlength=J) / n_j
        c_in = np.bincount(A["f"], weights=w_in * A["c_in"], minlength=J) / n_j
    d = (h_in - h_out) - (c_in - c_out)
    ok = n_j > 0
    return float(np.sum(d[ok] * n_j[ok]) / np.sum(n_j[ok])), d


def _did_unmatched(rin, rout, key) -> float:
    """The statistic before docs 15.43: in-sample rows on their own gallery against out-of-sample rows on theirs."""
    m = lambda rr, k: float(np.mean([bool(x[k]) for x in rr]))  # noqa: E731
    return (m(rin, key) - m(rout, key)) - (m(rin, "c_hit") - m(rout, "c_hit"))


def encoder_in_sample_check(rows: list[dict], max_gain: float = DEFAULT_MAX_ENCODER_GAIN, n_boot: int = 1000,
                            seed: int = 0) -> dict:
    """Difference-in-differences of top-1 between the folds whose clips the encoder trained on (in) and those it did
    not (out), on the SAME gallery (docs 15.43): for each in-sample fold j, (retrieval on fold j - retrieval on the
    out-of-sample clips, both against fold j's leave-fold-out gallery) - (fold classifier on fold j - on the out
    clips), pooled over j weighted by fold size; the same for the fused top-1. The classifier difference controls
    for fold difficulty (every fold classifier is out of sample); the shared gallery controls for its smaller
    support (fewer clips per INCLUDE centroid). 95 % interval from a cluster bootstrap (recording sessions,
    resampled within each arm). ``passed`` = pooled retrieval DiD <= ``max_gain``. Not testable (no out-of-sample or
    no in-sample clip): ``passed`` None. ``did_retrieval_unmatched`` is the old statistic (each arm on its own
    gallery), reported for comparison only."""
    rin = [x for x in rows if x["in_sample"]]
    rout = [x for x in rows if not x["in_sample"]]
    rep = {"n_in": len(rin), "n_out": len(rout), "max_gain": max_gain,
           "statistic": "pooled over in-sample folds j: (retrieval top-1 fold j - out, both on fold j's leave-fold-out "
                        "gallery) - (fold classifier top-1 fold j - out); in = folds whose clips the encoder trained on"}
    if not rin or not rout:
        return {**rep, "testable": False, "passed": None,
                "reason": "needs clips the encoder trained on AND clips it did not (fold 1 = primary validation)"}
    folds = sorted({x["fold"] for x in rin})
    matched = {f: all(f in (x.get("r_hit_g") or {}) for x in rout) for f in folds}
    for arm, rr in (("in", rin), ("out", rout)):
        rep[arm] = {k: _rate(rr, k) for k in ("c_hit", "r_hit", "f_hit")} | {
            "clusters": len({x["cluster"] for x in rr})}
    A = _arm_arrays(rin, rout, folds)
    one_in, one_out = np.ones(len(rin)), np.ones(len(rout))
    did_r, per_r = _did_w(A, one_in, one_out, "r")
    did_f, per_f = _did_w(A, one_in, one_out, "f")
    rng = np.random.default_rng(seed)
    br, bf = [], []
    for _ in range(max(0, int(n_boot))):
        w_in = np.bincount(rng.integers(0, A["n_k_in"], A["n_k_in"]), minlength=A["n_k_in"])[A["k_in"]].astype(float)
        w_out = np.bincount(rng.integers(0, A["n_k_out"], A["n_k_out"]),
                            minlength=A["n_k_out"])[A["k_out"]].astype(float)
        if w_out.sum() == 0 or w_in.sum() == 0:
            continue
        br.append(_did_w(A, w_in, w_out, "r")[0])
        bf.append(_did_w(A, w_in, w_out, "f")[0])
    ci = (lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
          if v else None)
    per_fold = {}
    for j, f in enumerate(folds):
        nj = int(np.sum(A["f"] == j))
        per_fold[str(f)] = {"n_in": nj, "gallery_matched": matched[f],
                            "out_retrieval_top1_on_this_gallery": round(float(A["r_out"][:, j].mean()), 4),
                            "did_retrieval": round(float(per_r[j]), 4), "did_fused": round(float(per_f[j]), 4)}
    passed = bool(did_r <= max_gain)
    return {**rep, "testable": True, "did_retrieval": round(did_r, 4), "did_retrieval_ci95": ci(br),
            "did_fused": round(did_f, 4), "did_fused_ci95": ci(bf), "n_boot": int(n_boot), "passed": passed,
            "gallery_matched": all(matched.values()), "per_fold": per_fold,
            "did_retrieval_unmatched": round(_did_unmatched(rin, rout, "r_hit"), 4),
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
    support = {}
    for p in ev.get("parts") or []:
        gs = (p.get("gallery") or {})
        if gs:
            support[str(p["fold"])] = {"arm": "in" if p.get("encoder_in_sample") else "out",
                                       **(gs.get("include_support") or {}),
                                       "centroids_dropped": gs.get("include_centroids_dropped", 0)}
    if support:
        chk = {**chk, "gallery_support": support}
    return used, {"version": OOF_VERSION, "assumption": ASSUMPTION, "in_sample_check": chk,
                  "keep_in_sample": bool(oof.keep_in_sample), "max_encoder_gain": oof.max_encoder_gain,
                  "requested_folds": oof.requested_folds, "allow_test_selected": bool(oof.allow_test_selected),
                  "folds_skipped": list(oof.skipped), "folds_used": folds_used, "folds_dropped": dropped,
                  **({"dropped_reason": why} if why else {}),
                  "clips_used": len(used), "groups_used": len({x["group"] for x in used if x["group"]}),
                  "parts": ev["parts"], "centroid_check": ev.get("centroid_check")}
