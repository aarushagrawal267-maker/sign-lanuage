"""ISL evaluation protocols (docs/ISL_DESIGN.md 9.1) on cached evidence.

E1  closed-set core classifier on the INCLUDE test fold (top-1 / top-5 / macro F1)
E2  cross-dataset retrieval: INCLUDE test clips against **ISLRTC rows only** (top-1 / 5 / 10); labelled
    "seen-class" unless the encoder was trained with the concepts held out (``oneshot`` folds)
E3  fused system with accept / suggest / abstain: coverage, accepted precision, silent error rate, ...
E6  latency of ``recognize`` on three engine crops (p50 / p95)
"""

from __future__ import annotations

import time

import numpy as np

from app.isl.data.sources import Clip
from app.isl.evaluation.report import grouped_bootstrap
from app.isl.models.calibration import summarize
from app.isl.models.fusion import FusionInputs, fuse
from app.isl.training.common import engine_crops


def e1_classifier(inputs: list[FusionInputs], truths: list[str], clips: list[Clip], label: str) -> dict | None:
    if not inputs or inputs[0].cls_logp is None:
        return None
    ids = inputs[0].cls_ids
    pos = {c: i for i, c in enumerate(ids)}
    sel = [i for i, t in enumerate(truths) if t in pos]
    if not sel:
        return {"label": label, "n": 0, "metrics": {}}
    from sklearn.metrics import f1_score

    P = np.stack([inputs[i].cls_logp.mean(0) for i in sel])
    y = np.array([pos[truths[i]] for i in sel])
    pred = P.argmax(1)
    top5 = np.array([y[j] in np.argsort(-P[j])[:5] for j in range(len(y))])
    groups = [clips[i].group for i in sel]
    hit = (pred == y).astype(float)
    return {"label": label, "n": len(sel), "n_classes": len(set(y.tolist())), "n_groups": len(set(groups)),
            "metrics": {"top1": float(hit.mean()), "top5": float(top5.mean()),
                        "macro_f1": float(f1_score(y, pred, labels=sorted(set(y.tolist())), average="macro", zero_division=0))},
            "intervals": {"top1": grouped_bootstrap(hit.tolist(), groups), "top5": grouped_bootstrap(top5.astype(float).tolist(), groups)}}


def e2_retrieval(rec, clips: list[Clip], label: str) -> dict | None:
    """INCLUDE clips (another signer group) queried against the ISLRTC dictionary rows only."""
    from app.features.build import FeatureError, build_features

    if rec.index is None or rec.encoder is None:
        return None
    idx = rec.index
    mask = np.array([r.get("source") == "islrtc" for r in idx.rows])
    if not mask.any():
        return None
    isl_concepts = sorted({r["concept_id"] for r, m in zip(idx.rows, mask) if m})
    cpos = {c: i for i, c in enumerate(isl_concepts)}
    rc = np.array([cpos.get(r["concept_id"], -1) for r in idx.rows])
    hits: dict[int, list[float]] = {1: [], 5: [], 10: []}
    groups = []
    for c in clips:
        if c.concept_id not in cpos:
            continue
        try:
            f = build_features(c.load(), rec.feature_config)
        except (FeatureError, OSError, ValueError):
            continue
        z, _ = rec.encoder.embed(f)
        s = idx.scores(z)[0]
        cs = np.full(len(isl_concepts), -2.0)
        np.maximum.at(cs, rc[mask], s[mask])
        ranked = [isl_concepts[j] for j in np.argsort(-cs)[:10]]
        for k in hits:
            hits[k].append(float(c.concept_id in ranked[:k]))
        groups.append(c.group)
    return {"label": label, "n": len(groups), "gallery_concepts": len(isl_concepts),
            "metrics": {f"top{k}": (float(np.mean(v)) if v else None) for k, v in hits.items()},
            "intervals": {"top1": grouped_bootstrap(hits[1], groups)}}


def e3_decisions(inputs: list[FusionInputs], truths: list[str], clips: list[Clip], params, label: str) -> dict:
    results = [fuse(i, params) for i in inputs]
    out = {"label": label, "n": len(results), "fusion_fitted": params.fitted, "policies": {}}
    for pol in ("strict", "balanced", "lenient"):
        out["policies"][pol] = summarize(results, truths, params, pol)
    b = out["policies"]["balanced"]
    out["metrics"] = {k: b[k] for k in ("top1", "top5", "coverage", "accepted_precision", "silent_error_rate",
                                       "suggest_rate", "suggest_top5_hit", "abstain_rate")}
    # risk-coverage curve over the fused top-1 probability
    p1 = np.array([r.probs[0] if len(r.probs) else 0.0 for r in results])
    ok = np.array([bool(r.keys) and r.keys[0] == t for r, t in zip(results, truths)])
    order = np.argsort(-p1)
    curve = []
    for frac in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        k = max(1, int(round(frac * len(order)))) if len(order) else 0
        curve.append({"coverage": frac, "risk": float(1 - ok[order[:k]].mean()) if k else None})
    out["risk_coverage"] = curve
    out["aurc"] = float(np.mean([c["risk"] for c in curve if c["risk"] is not None])) if curve and len(order) else None
    return out


def e6_latency(rec, clips: list[Clip], n: int = 20) -> dict:
    times, stages = [], {}
    for c in clips[:n]:
        try:
            crops = engine_crops(c.load())
        except (OSError, ValueError):
            continue
        t0 = time.perf_counter()
        r = rec.recognize(crops, top_k=5)
        times.append((time.perf_counter() - t0) * 1000)
        for k, v in r.detail.items():
            if k.endswith("_ms"):
                stages.setdefault(k, []).append(v)
        stages.setdefault("feature_ms", []).append(r.feature_ms)
    if not times:
        return {"label": "CPU latency per segment (3 crops)", "n": 0, "metrics": {}}
    import os

    try:
        load = os.getloadavg()[0]
    except OSError:
        load = None
    return {"label": "CPU latency per segment (3 crops)" + (f", measured at load average {load:.1f} on "
                                                            f"{os.cpu_count()} CPUs" if load is not None else ""),
            "load_average": load,
            "n": len(times), "metrics": {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))},
            "stages_p50_ms": {k: float(np.percentile(v, 50)) for k, v in stages.items()}}
