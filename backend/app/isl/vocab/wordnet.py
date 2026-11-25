"""Optional WordNet access (NLTK) for synonym-list detection, part of speech and coarse categories.

WordNet is optional: without it the normaliser falls back to plain rules (multi-token names become
``multiword``; only explicit ``x or y`` lists become synonym lists) and every affected entry is flagged
``no_wordnet``. Data lookup order: ``$NLTK_DATA``, ``<isl data>/nltk_data`` (``python -m app.isl.cli wordnet``
downloads it there), then NLTK's default locations.

Licence: WordNet 3.0 licence (permissive, commercial use allowed), see docs/DATASETS_ISL.md.
"""

from __future__ import annotations

import functools
import itertools
import os
from pathlib import Path

WORDNET_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"

# Function words never form a synonym-list piece ("Current and Ampere", "Round in sports").
STOPWORDS = frozenset(
    "a an the and or of in on at to for from by with without into onto over under up down out off as is are "
    "be was were am do does did not no yes it its this that these those he she we they you i my your our their "
    "his her him them me us how what when where which who why vs per via".split()
)

_disabled = False


def disable(flag: bool = True) -> None:
    """Tests use this to force the no-WordNet fallback."""
    global _disabled
    _disabled = flag
    get_wordnet.cache_clear()


@functools.lru_cache(maxsize=1)
def get_wordnet():
    """The NLTK WordNet reader, or None when NLTK or its data is unavailable."""
    if _disabled:
        return None
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except Exception:  # noqa: BLE001
        return None
    candidates = []
    if os.environ.get("NLTK_DATA"):
        candidates.append(os.environ["NLTK_DATA"])
    try:
        from app.isl.paths import isl_paths

        candidates.append(str(isl_paths().nltk_data))
    except Exception:  # noqa: BLE001
        pass
    from app.core.config import REPO_ROOT

    candidates.append(str(REPO_ROOT / "data" / "isl" / "nltk_data"))
    for c in reversed(candidates):
        if c not in nltk.data.path and Path(c).exists():
            nltk.data.path.insert(0, c)
    try:
        wn.ensure_loaded()
        wn.synsets("dog")
    except Exception:  # noqa: BLE001
        return None
    return wn


def download_wordnet(dest: Path) -> Path:
    """Fetch the WordNet zip straight from the nltk_data repository (NLTK's own downloader refuses to run
    through an egress proxy). Returns the zip path."""
    import urllib.request

    target = dest / "corpora" / "wordnet.zip"
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.part")
    with urllib.request.urlopen(WORDNET_URL, timeout=120) as r, tmp.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(target)
    get_wordnet.cache_clear()
    return target


@functools.lru_cache(maxsize=200_000)
def synsets(term: str) -> tuple:
    wn = get_wordnet()
    if wn is None:
        return ()
    try:
        return tuple(wn.synsets(term.replace(" ", "_").lower()))
    except Exception:  # noqa: BLE001
        return ()


def is_lemma(term: str) -> bool:
    return bool(synsets(term))


def _related_synsets(a, b) -> bool:
    if a == b:
        return True
    if a.pos() in "as" and b.pos() in "as":
        sa = set(a.similar_tos()) | {a}
        sb = set(b.similar_tos()) | {b}
        if sa & sb:
            return True
    if a.pos() == b.pos() and a.pos() in "nv":
        try:
            d = a.shortest_path_distance(b)
        except Exception:  # noqa: BLE001
            d = None
        if d is not None and d <= 2:
            return True
    return False


@functools.lru_cache(maxsize=50_000)
def related(a: str, b: str, top_senses: int = 4) -> bool:
    """True when two terms are WordNet synonyms or near-synonyms: they share a synset, are linked by
    ``similar_tos`` (adjectives) or are at most 2 hypernym steps apart (nouns/verbs), considering each
    term's most frequent senses only (rare senses create false links such as "can" = "toilet")."""
    A, B = synsets(a)[:top_senses], synsets(b)[:top_senses]
    if not A or not B:
        return False
    if set(A) & set(B):
        return True
    # share a synset through any sense (e.g. able/can)
    if set(synsets(a)) & set(synsets(b)):
        return True
    return any(_related_synsets(x, y) for x, y in itertools.product(A, B))


def partitions(tokens: list[str], max_piece: int = 3):
    """All ways to cut ``tokens`` into contiguous pieces of at most ``max_piece`` tokens, fewest pieces first."""
    n = len(tokens)
    out = []

    def rec(i: int, acc: list[str]):
        if i == n:
            out.append(list(acc))
            return
        for k in range(1, min(max_piece, n - i) + 1):
            acc.append(" ".join(tokens[i : i + k]))
            rec(i + k, acc)
            acc.pop()

    if n <= 10:
        rec(0, [])
    return sorted(out, key=len)


def synonym_pieces(tokens: list[str]) -> list[str] | None:
    """Split a multi-token name into >= 2 WordNet lemmas that form a connected synonym graph, else None."""
    if get_wordnet() is None or len(tokens) < 2:
        return None
    low = [t.lower() for t in tokens]
    if any(t in STOPWORDS for t in low):
        return None
    if is_lemma(" ".join(low)):
        return None  # the whole name is one lemma ("high five", "run out")
    for parts in partitions(low):
        if len(parts) < 2 or not all(is_lemma(p) for p in parts):
            continue
        # connected graph under `related`
        seen = {0}
        frontier = [0]
        while frontier:
            i = frontier.pop()
            for j in range(len(parts)):
                if j not in seen and related(parts[i], parts[j]):
                    seen.add(j)
                    frontier.append(j)
        if len(seen) == len(parts):
            return parts
    return None


def pos_and_lexname(term: str) -> tuple[str | None, str | None]:
    ss = synsets(term)
    if not ss:
        return None, None
    s = ss[0]
    pos = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}.get(s.pos())
    return pos, s.lexname()


def is_proper_noun(term: str) -> bool:
    """WordNet lists the term only with capitalised lemma names (e.g. "Diwali", "Gujarat")."""
    ss = synsets(term)
    if not ss:
        return False
    key = term.replace(" ", "_").lower()
    names = [lem.name() for s in ss for lem in s.lemmas() if lem.name().lower() == key]
    return bool(names) and all(n[:1].isupper() for n in names)
