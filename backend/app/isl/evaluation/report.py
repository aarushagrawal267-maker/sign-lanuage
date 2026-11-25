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


def grouped_bootstrap(values: list[float], groups: list[str | None], n: int = 1000, seed: int = 0) -> list[float] | None:
    """95 % interval of the mean, resampling **groups** (signers / sessions), not clips."""
    if not values:
        return None
    rng = np.random.default_rng(seed)
    by: dict = {}
    for v, g in zip(values, groups):
        by.setdefault(g or "?", []).append(v)
    keys = list(by)
    if len(keys) < 2:
        return None
    sums = np.array([np.sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    means = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        means.append(sums[idx].sum() / max(1, cnts[idx].sum()))
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def write_results(path: Path, protocols: dict, *, profile: str, smoke: bool, data: dict, extra: dict | None = None) -> dict:
    doc = {
        "schema": 1, "language": "isl", "profile": profile, "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(), "smoke": smoke,
        "label": ("SMOKE RUN on partial data (few minutes of CPU training): proves the pipeline works end to end. "
                  "These numbers are NOT the accuracy of the app." if smoke else
                  "Measured on held-out data; live webcam accuracy is expected to be lower."),
        "data": data, "protocols": protocols, **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    return doc


def markdown_table(doc: dict) -> str:
    rows = ["| Protocol | Label | Metric | Value | 95% interval | n |", "|---|---|---|---|---|---|"]
    for pid, p in doc.get("protocols", {}).items():
        for m, v in (p.get("metrics") or {}).items():
            ci = (p.get("intervals") or {}).get(m)
            rows.append(f"| {pid} | {p.get('label', '')} | {m} | {v if v is None else round(v, 4)} | "
                        f"{'' if ci is None else f'{ci[0]}-{ci[1]}'} | {p.get('n')} |")
    return "\n".join(rows)
