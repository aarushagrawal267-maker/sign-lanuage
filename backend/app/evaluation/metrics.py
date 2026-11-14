"""Classification metrics. Every class is reported, including weak ones."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return float("nan")
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def confidence_curve(probs: np.ndarray, y: np.ndarray, steps: int = 21) -> list[dict]:
    """Accuracy vs. coverage when rejecting predictions below a confidence threshold."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    out = []
    for th in np.linspace(0, 1, steps):
        keep = conf >= th
        n = int(keep.sum())
        out.append({
            "threshold": round(float(th), 3),
            "coverage": float(n / len(y)) if len(y) else 0.0,
            "accuracy": float((pred[keep] == y[keep]).mean()) if n else None,
        })
    return out


def compute_metrics(probs: np.ndarray, y: np.ndarray, class_names: list[str],
                    video_ids: list[str] | None = None, signer_ids: list[int] | None = None) -> dict:
    n_classes = len(class_names)
    pred = probs.argmax(axis=1) if len(y) else np.zeros(0, int)
    labels = list(range(n_classes))
    present = sorted(set(int(c) for c in y))
    if len(y):
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        cm = confusion_matrix(y, pred, labels=labels)
        macro_f1_present = float(f1_score(y, pred, labels=present, average="macro", zero_division=0))
    else:
        p = r = f = np.zeros(n_classes)
        s = np.zeros(n_classes, int)
        cm = np.zeros((n_classes, n_classes), int)
        macro_f1_present = float("nan")

    per_class = [
        {"class_index": i, "gloss": class_names[i], "precision": float(p[i]), "recall": float(r[i]),
         "f1": float(f[i]), "support": int(s[i])}
        for i in labels
    ]
    # Most frequent off-diagonal confusions
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:15]
    confusions = [
        {"true": class_names[i // n_classes], "pred": class_names[i % n_classes], "count": int(off.flat[i])}
        for i in flat if off.flat[i] > 0
    ]
    per_signer = {}
    if signer_ids is not None and len(y):
        for sid in sorted(set(signer_ids)):
            idx = np.array([j for j, s_ in enumerate(signer_ids) if s_ == sid])
            per_signer[str(sid)] = {"n": int(len(idx)), "top1": float((pred[idx] == y[idx]).mean())}
    predictions = None
    if video_ids is not None:
        predictions = [
            {"video_id": video_ids[j], "true": int(y[j]), "pred": int(pred[j]), "confidence": float(probs[j].max())}
            for j in range(len(y))
        ]
    return {
        "n": int(len(y)),
        "top1": float((pred == y).mean()) if len(y) else float("nan"),
        "top5": topk_accuracy(probs, y, 5),
        # macro F1 over classes that occur in this partition (classes with no
        # samples cannot be scored); classes_evaluated says how many that is.
        "macro_f1": macro_f1_present,
        "classes_evaluated": len(present),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": confusions,
        "per_signer": per_signer,
        "confidence_curve": confidence_curve(probs, y) if len(y) else [],
        "predictions": predictions,
    }


def pick_threshold(curve: list[dict], target_accuracy: float = 0.85, min_threshold: float = 0.35) -> float:
    """Lowest confidence threshold whose accepted predictions reach target accuracy on validation."""
    for row in curve:
        if row["accuracy"] is not None and row["coverage"] > 0.02 and row["accuracy"] >= target_accuracy:
            return max(min_threshold, row["threshold"])
    return 0.6
