"""Alias groups: two concept ids that are the same output word and that signers use for each other (QA plan 1.2,
docs 15.34, 15.35).

The vocabulary keeps some twins apart because their source names differ: "female" and ISLRTC's "Woman / Female",
"big" and "large", "mouse" and "Rat / Mouse". A model that picks the other twin is not wrong for the user, but strict
scoring counts it wrong, and the fused probability of the word is split over two keys, so neither reaches the accept
threshold.

``alias_of`` maps an alias to its root (``{"isl.woman-female": "isl.female"}``). The root is an INCLUDE classifier
class, because INCLUDE labels E1b. The alias can be an INCLUDE class too: ``isl.woman-female`` holds INCLUDE's own
"Woman" class (20 clips) as well as ISLRTC's "Woman / Female" row, so that group joins two INCLUDE classes, "woman"
and "female", and the word shown can be either ("woman / female" or "female"). Every pair needs evidence (the comment
next to it): no pair is added on its name alone. Rules:

* aliases never change training: classes, index rows and labels stay as they are;
* the fused decision may sum the probabilities of one group before it applies the thresholds
  (``FusionParams.aliases``, set by fit-fusion only when the link step turned aliases on); a group is a classifier
  word when any member is (:func:`closed_with_aliases`), whichever member happens to lead;
* evaluation reports strict-label numbers (the gates and headlines) and alias-label numbers next to them
  (critic 12: aliases must never "pass" a gate by redefining correct).

"you" and "you (plural)" stay separate: a real difference in sense (``NEVER_ALIAS``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping

ALIAS_VERSION = "isl-alias-2"  # 2: race pair removed (docs 15.35)

# alias -> root. Twins found by the QA review (plan 1.2); both ids must exist in the vocabulary to be used.
# Evidence (docs 15.35): cosine ranks in the 27 Nov 2026 indexes (research / commercial-safe encoders, train clips only:
# INCLUDE / ISL500 centroids against the ISLRTC dictionary rows, rank among the 13,461 rows).
TWIN_ALIASES: dict[str, str] = {
    # INCLUDE "Woman" -> ISLRTC Woman/Female rank 62 / 2 (Female row 28 / 24); ISL500 "Female" -> Woman/Female
    # rank 1 (Female row 14); ISLRTC Woman/Female vs Female cosine 0.66 / 0.60
    "isl.woman-female": "isl.female",
    # INCLUDE "big / large" -> ISLRTC Large rank 2 / 14, Big 6 / 1; ISL500 "Large" -> Large 2, Big 5;
    # ISLRTC Big vs Large cosine 0.78 / 0.63
    "isl.large": "isl.big",
    # INCLUDE "Mouse" (animals) -> ISLRTC Rat/Mouse rank 35 / 5; ISL500 "Mouse" -> ISLRTC Mouse rank 1. Two forms
    # of the one word "mouse" (ISLRTC Mouse vs Rat/Mouse cosine 0.06 / 0.50), both used for the animal
    "isl.rat-mouse": "isl.mouse",
    # NOT aliased: isl.race~ethnicity (INCLUDE "Race (ethnicity)") matches none of ISLRTC's three "race" rows: Race
    # rank 4813 / 781, Race (Competition) 766 / 1510, Race (Social Science) 10981 / 2469 (review of 1bec36a)
}
# pairs that look like twins but are different senses: never put in one group
NEVER_ALIAS: tuple[frozenset, ...] = (frozenset({"isl.you", "isl.you~plural"}),)

# sources whose glosses number repeated words ("Adult(1)"); only these are stripped (docs 15.35)
STRIP_NUMBERING_SOURCES: frozenset[str] = frozenset({"wslp"})

_SIGN_NO = re.compile(r"^(.*?\S)\s*\(\s*\d+\s*\)\s*$")


def strip_sign_number(text: str | None) -> str:
    """WSLP numbers repeated words: "Adult(1)" -> "Adult", "Bank (2)" -> "Bank". Only a trailing "(digits)" is
    removed; "(1)" alone and "(academic)" are kept."""
    if not text:
        return text or ""
    m = _SIGN_NO.match(text)
    return m.group(1) if m else text


def resolve_aliases(alias_of: Mapping[str, str] | None, known: Iterable[str] | None = None) -> dict[str, str]:
    """``alias_of`` cleaned: chains followed to their final root (a -> b -> c gives a -> c and b -> c), self maps
    dropped, and with ``known`` only pairs whose alias and root are both known concepts. Raises ``ValueError`` on a
    cycle or on a pair in ``NEVER_ALIAS``."""
    raw = {str(a): str(r) for a, r in dict(alias_of or {}).items() if a and r and a != r}
    out: dict[str, str] = {}
    for a in raw:
        seen = [a]
        r = raw[a]
        while r in raw:
            if r in seen:
                raise ValueError(f"alias cycle: {' -> '.join(seen + [r])}")
            seen.append(r)
            r = raw[r]
        out[a] = r
    if known is not None:
        k = set(known)
        out = {a: r for a, r in out.items() if a in k and r in k}
    groups = alias_groups(out)
    for pair in NEVER_ALIAS:
        for members in groups.values():
            if pair <= set(members):
                raise ValueError(f"{sorted(pair)} are different senses and must not be aliases")
    return dict(sorted(out.items()))


def alias_root(key: str, alias_of: Mapping[str, str] | None) -> str:
    return (alias_of or {}).get(key, key)


def alias_groups(alias_of: Mapping[str, str] | None) -> dict[str, list[str]]:
    """root -> sorted members (the root first)."""
    g: dict[str, set[str]] = {}
    for a, r in dict(alias_of or {}).items():
        g.setdefault(r, {r}).add(a)
    return {r: [r] + sorted(m - {r}) for r, m in sorted(g.items())}


def aliases_sha256(alias_of: Mapping[str, str] | None) -> str | None:
    if not alias_of:
        return None
    return hashlib.sha256(json.dumps(dict(sorted(dict(alias_of).items()))).encode()).hexdigest()[:16]


def merge_columns(ids: list[str], alias_of: Mapping[str, str] | None):
    """Columns of one alias group summed: returns ``(roots, index)`` where ``roots`` is the list of group roots in
    first-seen order and ``index[j]`` the group of column j. Use with ``np.add.at`` (see :func:`sum_by_alias`)."""
    pos: dict[str, int] = {}
    idx = []
    for k in ids:
        r = alias_root(k, alias_of)
        if r not in pos:
            pos[r] = len(pos)
        idx.append(pos[r])
    return list(pos), idx


def sum_by_alias(P, ids: list[str], alias_of: Mapping[str, str] | None):
    """``P`` (..., len(ids)) probabilities summed over each alias group: ``(roots, P_root)`` with P_root of shape
    (..., len(roots)). Without aliases it returns ``(ids, P)`` unchanged."""
    import numpy as np

    if not alias_of:
        return list(ids), P
    roots, idx = merge_columns(ids, alias_of)
    if len(roots) == len(ids):
        return roots, P
    P = np.asarray(P)
    out = np.zeros(P.shape[:-1] + (len(roots),), dtype=np.float64)
    np.add.at(np.moveaxis(out, -1, 0), np.asarray(idx), np.moveaxis(P, -1, 0))  # views: writes into out
    return roots, out


def closed_with_aliases(closed_set, alias_of: Mapping[str, str] | None):
    """The classifier's classes with every member of an alias group that has a class in it (docs 15.35): the
    open-vocabulary gate then treats the whole group as a classifier word, whichever member the fused decision
    shows (the summed mass holds the classifier's probability either way). ``None`` stays ``None`` (not checked)."""
    if closed_set is None or not alias_of:
        return closed_set
    out = set(closed_set)
    for members in alias_groups(alias_of).values():
        if out & set(members):
            out |= set(members)
    return out
