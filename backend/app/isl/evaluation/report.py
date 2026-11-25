"""``results.json`` writer with intervals and honest labels (docs/ISL_DESIGN.md 6.5 and 9.2)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.core.config import REPO_ROOT


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


MIN_INTERVAL_GROUPS = 5  # below this many signer groups a bootstrap "95 % interval" looks precise and is not


def n_groups(groups: list[str | None]) -> int:
    return len({g or "?" for g in groups})


def interval_reason(values: list[float], groups: list[str | None], min_groups: int = MIN_INTERVAL_GROUPS) -> str | None:
    """Why no interval is reported (None = an interval can be computed)."""
    if not values:
        return "no values"
    k = n_groups(groups)
    if k < min_groups:
        return f"too few signer groups ({k})"
    return None


def grouped_bootstrap(values: list[float], groups: list[str | None], n: int = 1000, seed: int = 0,
                      min_groups: int = MIN_INTERVAL_GROUPS) -> list[float] | None:
    """95 % interval of the mean, resampling **groups** (signers / sessions), not clips. ``None`` with fewer
    than ``min_groups`` groups (:func:`interval_reason` says why)."""
    if interval_reason(values, groups, min_groups) is not None:
        return None
    rng = np.random.default_rng(seed)
    by: dict = {}
    for v, g in zip(values, groups):
        by.setdefault(g or "?", []).append(v)
    keys = list(by)
    sums = np.array([np.sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        means.append(sums[idx].sum() / max(1, cnts[idx].sum()))
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def interval_fields(series: dict[str, tuple[list[float], list[str | None]]]) -> dict:
    """``{"intervals": {k: [lo, hi] | None}, "interval_reasons": {k: reason}, "interval_groups": {k: n}}`` for
    several metrics at once (every interval travels with its group count, and every missing one with a reason)."""
    iv, why, ng = {}, {}, {}
    for k, (vals, grp) in series.items():
        iv[k] = grouped_bootstrap(vals, grp)
        ng[k] = n_groups(grp) if vals else 0
        r = interval_reason(vals, grp)
        if r is not None:
            why[k] = r
    return {"intervals": iv, "interval_reasons": why, "interval_groups": ng}


def format_interval(ci: list[float] | None, n_grp: int | None = None, reason: str | None = None) -> str:
    """Markdown cell: ``0.612-0.701 (7 groups)`` or ``none: too few signer groups (2)``."""
    if ci:
        return f"{ci[0]:.3f}-{ci[1]:.3f}" + (f" ({n_grp} groups)" if n_grp is not None else "")
    return f"none: {reason}" if reason else "-"


PARTIAL_DATA_LABEL = ("PARTIAL DATA: trained and evaluated while ingestion was still running or a source manifest was "
                      "incomplete (started with --allow-partial). These numbers are NOT the accuracy of the app.")


def write_results(path: Path, protocols: dict, *, profile: str, smoke: bool, data: dict, extra: dict | None = None,
                  partial_data: bool = False) -> dict:
    if smoke:
        label = ("SMOKE RUN on partial data (few minutes of CPU training): proves the pipeline works end to end. "
                 "These numbers are NOT the accuracy of the app.")
    elif partial_data:
        label = PARTIAL_DATA_LABEL
    else:
        label = "Measured on held-out data; live webcam accuracy is expected to be lower."
    doc = {
        "schema": 1, "language": "isl", "profile": profile, "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(), "smoke": smoke, "partial_data": bool(partial_data or smoke), "label": label,
        "data": data, "protocols": protocols, **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    return doc


def markdown_table(doc: dict) -> str:
    rows = []
    if doc.get("partial_data") and not doc.get("smoke"):
        rows += ["**PARTIAL DATA** - " + PARTIAL_DATA_LABEL, ""]
    rows += ["| Protocol | Label | Metric | Value | 95% interval | n |", "|---|---|---|---|---|---|"]
    for pid, p in doc.get("protocols", {}).items():
        for m, v in (p.get("metrics") or {}).items():
            ci = (p.get("intervals") or {}).get(m)
            why = (p.get("interval_reasons") or {}).get(m)
            ng = (p.get("interval_groups") or {}).get(m)
            cell = format_interval(ci, ng, why) if (ci or why) else ""
            rows.append(f"| {pid} | {p.get('label', '')} | {m} | {v if v is None else round(v, 4)} | {cell} | {p.get('n')} |")
    return "\n".join(rows)
