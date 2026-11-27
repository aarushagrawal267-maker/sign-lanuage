"""Source quotas for the classifier's two-stage sampler (QA plan 1.4, docs/ISL_DESIGN.md 15.38).

A quota says which share of the training draws each *quota group* gets. A group is a source name (``include``,
``isl500``, ``islrtc``...) or one of two families: ``dictionary`` (ISLRTC and its second views WSLP and CISLR,
:data:`~app.isl.training.common.DICTIONARY_SOURCES`) and ``mendeley`` (every ``mendeley_*`` source). A key that names
a source exactly wins over its family. Groups that have no training clip are dropped and the rest renormalised, so
the same quota works for every split and data cap. A training source whose group has no quota is refused (a new
source must never silently get no draws).
"""

from __future__ import annotations

import math

from app.isl.training.common import DICTIONARY_SOURCES

# research default (QA plan 1.4): INCLUDE first (the app's users sign like INCLUDE), ISL500 for signer variety, the
# dictionary and Mendeley small
DEFAULT_SOURCE_QUOTA = {"include": 0.45, "isl500": 0.35, "dictionary": 0.12, "mendeley": 0.08}
FAMILIES = ("dictionary", "mendeley")
# the sources whose validation clips decide model selection in ``worst_source`` mode (when present)
MAJOR_SOURCES = ("include", "isl500")
OFF = ("", "none", "off", "0")


def _known_keys() -> set[str]:
    from app.isl.profiles import WORD_SOURCES

    return {s for v in WORD_SOURCES.values() for s in v} | set(FAMILIES)


def parse_source_quota(spec: str | dict | None) -> dict[str, float] | None:
    """``"include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08"`` -> dict; ``default`` -> the research default;
    empty / ``none`` -> None (no quota: the old class-balanced sampler). Values must be finite and >= 0, at least
    one > 0. Unknown names are refused (a typo would otherwise drop a source)."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        q = {str(k): float(v) for k, v in spec.items()}
    else:
        s = spec.strip().lower()
        if s in OFF:
            return None
        if s == "default":
            return dict(DEFAULT_SOURCE_QUOTA)
        q = {}
        for part in s.split(","):
            k, sep, v = part.partition("=")
            k = k.strip()
            if not sep or not k:
                raise ValueError(f"source quota: expected name=share, got {part!r}")
            if k in q:
                raise ValueError(f"source quota: {k!r} given twice")
            try:
                q[k] = float(v)
            except ValueError:
                raise ValueError(f"source quota: {part!r} has no number") from None
    for k, v in q.items():
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"source quota: {k}={v} must be a finite share >= 0")
    if not q or sum(q.values()) <= 0:
        raise ValueError("source quota: at least one share must be > 0")
    unknown = sorted(set(q) - _known_keys())
    if unknown:
        raise ValueError(f"source quota: unknown source or group {unknown}; known: {sorted(_known_keys())}")
    return q


def quota_group(source: str, quota: dict[str, float]) -> str | None:
    """The quota group of a source: its own name when the quota names it, else its family, else None."""
    if source in quota:
        return source
    if source in DICTIONARY_SOURCES and "dictionary" in quota:
        return "dictionary"
    if source.startswith("mendeley") and "mendeley" in quota:
        return "mendeley"
    return None


def resolve_source_quota(quota: dict[str, float], sources) -> tuple[dict[str, str], dict[str, float]]:
    """``sources``: the training sources present. Returns ``(group_of, shares)``: source -> quota group, and the
    shares renormalised over the groups present (groups with no clip are dropped). Refuses a present source
    without a group and a quota whose present groups all have share 0."""
    present = sorted(set(sources))
    group_of = {s: quota_group(s, quota) for s in present}
    missing = [s for s, g in group_of.items() if g is None]
    if missing:
        raise ValueError(f"source quota {quota} has no share for training sources {missing} (name them, or their "
                         f"family: {', '.join(FAMILIES)})")
    groups = sorted(set(group_of.values()))
    total = sum(quota[g] for g in groups)
    if total <= 0:
        raise ValueError(f"source quota {quota}: every source present ({present}) has share 0")
    return group_of, {g: quota[g] / total for g in groups}
