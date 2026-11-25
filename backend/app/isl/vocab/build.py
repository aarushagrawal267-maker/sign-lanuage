"""Unified ISL vocabulary builder (docs/ISL_DESIGN.md section 6.1).

Inputs (all optional except the ISLRTC names):
* ISLRTC dictionary file names (``data/isl/manifests/islrtc_index.json``, all 13,665 files; falls back to the
  ids in ``islrtc.jsonl`` when the index is missing),
* INCLUDE word folders with clip counts (official split lists and/or ``include.jsonl``),
* CISLR words (``data/isl/raw/cislr/dataset.csv`` once access is granted),
* the trim sidecar (adds ``trim`` to ISLRTC variants), ``overrides.yaml``.

Output: ``models/packs/isl/vocabulary.jsonl`` (sorted by ``concept_id``; row number = class index) and a
build report. The file only depends on names and split lists, so it is reproducible without landmark data.
"""

from __future__ import annotations

import json
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from app.isl.vocab import wordnet as wnx
from app.isl.vocab.normalize import (
    FULL_CLIP_KINDS,
    NON_INDEX_KINDS,
    ParsedName,
    apply_overrides,
    load_overrides,
    match_key,
    parse_cislr_word,
    parse_include_word,
    parse_islrtc_name,
    slugify,
)
from app.isl.vocab.schema import Review, SourceRef, Variant, VocabEntry

ISLRTC_HF_REPO = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
LICENCES = {
    "islrtc": "GODL-India",
    "include": "CC-BY-4.0",
    "cislr": "AFL-3.0 (pending review)",
    "isign": "CC-BY-NC-SA-4.0",
    "mysigns": "per pack",
}
INCLUDE_URL = "https://zenodo.org/records/4010759"
CISLR_URL = "https://huggingface.co/datasets/Exploration-Lab/CISLR"

LEXNAME_CATEGORY = {
    "noun.animal": "animals", "noun.food": "food", "noun.body": "body", "noun.person": "people",
    "noun.location": "places", "noun.time": "time", "noun.artifact": "objects", "noun.plant": "plants",
    "noun.feeling": "feelings", "noun.communication": "communication", "noun.act": "actions",
    "noun.event": "events", "noun.cognition": "ideas", "noun.attribute": "qualities", "noun.state": "states",
    "noun.quantity": "quantities", "noun.group": "groups", "noun.possession": "money",
    "noun.substance": "substances", "noun.object": "nature", "noun.phenomenon": "nature",
    "noun.process": "processes", "noun.relation": "relations", "noun.shape": "shapes", "noun.motive": "ideas",
}


def islrtc_clip_id(rel_path: str) -> str:
    return f"islrtc/{rel_path}"


def islrtc_variant_id(rel_path: str) -> str:
    p = PurePosixPath(rel_path)
    return f"islrtc:{p.parent}/{p.stem}" if str(p.parent) != "." else f"islrtc:{p.stem}"


def include_variant_id(word_dir: str) -> str:
    return "include:" + word_dir.replace(" ", "_")


def _category_for(pn: ParsedName, kind: str) -> str | None:
    if kind == "number":
        return "numbers"
    if kind == "letter":
        return "letters"
    if pn.domain:
        return pn.domain
    head = (pn.synonyms[0] if pn.synonyms else pn.headword) or ""
    pos, lex = wnx.pos_and_lexname(head) if " " not in head.strip() else (None, None)
    if lex:
        if lex.startswith("verb."):
            return "actions"
        if lex.startswith("adj.") or lex.startswith("adv."):
            return "descriptions"
        return LEXNAME_CATEGORY.get(lex, "other")
    return "other"


def _fingerspell(headword: str) -> str | None:
    w = headword.strip()
    if w.isascii() and w.isalpha() and len(w) <= 15:
        return w.upper()
    return None


def _display(pn: ParsedName) -> str:
    base = " / ".join(pn.synonyms) if pn.kind == "synonym_list" and pn.synonyms else pn.headword
    qual = pn.sense or pn.domain
    return f"{base} ({qual})" if qual else base


@dataclass
class BuildInputs:
    islrtc_paths: list[str]  # "A/Abdicate.mp4", ...
    include_words: dict[str, int] = field(default_factory=dict)  # "Category/NN. Word" -> clip count
    cislr_rows: list[dict] = field(default_factory=list)  # {"uid", "word", ...}
    trims: dict[str, dict] = field(default_factory=dict)  # islrtc clip id -> sidecar record
    overrides: list[dict] = field(default_factory=list)


@dataclass
class BuildResult:
    entries: list[VocabEntry]
    report: dict


def _primary_rank(pn: ParsedName) -> tuple:
    return (pn.kind in NON_INDEX_KINDS, pn.region is not None, pn.sign_no, len(pn.original))


def build_vocabulary(inp: BuildInputs) -> BuildResult:
    parsed: list[tuple[str, ParsedName]] = []
    for rel in sorted(set(inp.islrtc_paths)):
        pn = parse_islrtc_name(rel)
        apply_overrides(pn, islrtc_clip_id(rel), inp.overrides)
        parsed.append((rel, pn))

    by_concept: dict[str, list[tuple[str, ParsedName]]] = defaultdict(list)
    for rel, pn in parsed:
        by_concept[pn.concept_id].append((rel, pn))

    entries: dict[str, VocabEntry] = {}
    for cid, items in by_concept.items():
        items.sort(key=lambda x: _primary_rank(x[1]))
        prim = items[0][1]
        syn: list[str] = []
        for _, pn in items:
            for s in pn.synonyms:
                if s not in syn:
                    syn.append(s)
        variants, sources = [], []
        for rel, pn in items:
            clip = islrtc_clip_id(rel)
            tr = inp.trims.get(clip)
            variants.append(Variant(
                variant_id=islrtc_variant_id(rel), source="islrtc", sign_no=pn.sign_no, region=pn.region,
                kind=pn.kind, index=pn.kind not in NON_INDEX_KINDS, full_clip=pn.kind in FULL_CLIP_KINDS,
                clips=[clip], trim=({"method": tr.get("method"), "confidence": tr.get("confidence")} if tr else None),
                flags=sorted(set(pn.flags)),
            ))
            sources.append(SourceRef(
                dataset="islrtc", id=clip, licence=LICENCES["islrtc"],
                url=f"https://huggingface.co/datasets/{ISLRTC_HF_REPO}/resolve/main/{urllib.parse.quote(rel)}",
            ))
        notes = sorted({f for _, pn in items for f in pn.flags if f in ("needs_review", "no_wordnet")})
        entries[cid] = VocabEntry(
            concept_id=cid, headword=prim.headword, display={"en": _display(prim)}, synonyms=syn,
            sense=prim.sense, domain=prim.domain, kind=prim.kind, category=_category_for(prim, prim.kind),
            pos=(wnx.pos_and_lexname(prim.headword)[0] if " " not in prim.headword else None),
            fingerspell=_fingerspell(prim.headword), number_value=prim.number_value,
            variants=variants, sources=sources, in_classifier=False,
            indexable=any(v.index for v in variants),
            review=Review(status="needs_review" if "needs_review" in notes else "auto", notes=notes),
        )

    linker = _Linker(entries)
    report: dict = {"islrtc_files": len(parsed), "islrtc_concepts": len(entries),
                    "kinds": dict(Counter(pn.kind for _, pn in parsed)),
                    "include": {"words": 0, "links": Counter(), "needs_review": [], "new": []},
                    "cislr": {"words": 0, "links": Counter()}}

    # ---- INCLUDE
    for word_dir, count in sorted(inp.include_words.items()):
        pn = parse_include_word(PurePosixPath(word_dir).name)
        apply_overrides(pn, f"include/{word_dir}", inp.overrides)
        category = PurePosixPath(word_dir).parts[0].replace("_", " ").lower() if "/" in word_dir else None
        cid, link, review, see_also = linker.link(pn)
        report["include"]["words"] += 1
        report["include"]["links"][link] += 1
        var = Variant(
            variant_id=include_variant_id(word_dir), source="include", region="Chennai (INCLUDE signers)",
            kind=pn.kind, clips_count=count, link=link, needs_review=review, flags=sorted(set(pn.flags)),
        )
        src = SourceRef(dataset="include", word_dir=word_dir, licence=LICENCES["include"], url=INCLUDE_URL)
        if cid is None:
            cid = pn.concept_id
            report["include"]["new"].append(word_dir)
            if cid not in entries:
                entries[cid] = VocabEntry(
                    concept_id=cid, headword=pn.headword, display={"en": _display(pn)}, synonyms=pn.synonyms,
                    sense=pn.sense, domain=None, kind=pn.kind, category=category,
                    pos=(wnx.pos_and_lexname(pn.headword)[0] if " " not in pn.headword else None),
                    fingerspell=_fingerspell(pn.headword), variants=[], sources=[],
                )
                linker.add(entries[cid], pn)
        e = entries[cid]
        for c in see_also:
            if c not in e.see_also and c != cid:
                e.see_also.append(c)
        if review:
            report["include"]["needs_review"].append(f"{word_dir} -> {cid} ({link}; see also {see_also})")
            if e.review.status == "auto":
                e.review.status = "needs_review"
            e.review.notes.append(f"include link {link}: {word_dir}" + (f"; see also {', '.join(see_also)}" if see_also else ""))
        e.variants.append(var)
        e.sources.append(src)
        e.in_classifier = True
        e.indexable = True
        if category and (e.category in (None, "other") or not any(v.source == "include" for v in e.variants[:-1])):
            e.category = category
        for s in pn.synonyms:
            if s not in e.synonyms:
                e.synonyms.append(s)

    # ---- CISLR (format confirmed once access is granted; uid + word columns)
    for row in inp.cislr_rows:
        word = str(row.get("word") or row.get("gloss") or "").strip()
        uid = str(row.get("uid") or row.get("id") or "").strip()
        if not word or not uid:
            continue
        pn = parse_cislr_word(word)
        apply_overrides(pn, f"cislr/{word}", inp.overrides)
        cid, link, review, see_also = linker.link(pn)
        report["cislr"]["words"] += 1
        report["cislr"]["links"][link] += 1
        if cid is None:
            cid = pn.concept_id
            if cid not in entries:
                entries[cid] = VocabEntry(
                    concept_id=cid, headword=pn.headword, display={"en": _display(pn)}, synonyms=pn.synonyms,
                    sense=None, domain=pn.domain, kind=pn.kind, category=_category_for(pn, pn.kind),
                    fingerspell=_fingerspell(pn.headword),
                )
                linker.add(entries[cid], pn)
        e = entries[cid]
        for c in see_also:
            if c not in e.see_also and c != cid:
                e.see_also.append(c)
        if review and e.review.status == "auto":
            e.review.status = "needs_review"
        e.variants.append(Variant(variant_id=f"cislr:{uid}", source="cislr", kind=pn.kind, clips=[f"cislr/{uid}"],
                                  link=link, needs_review=review))
        e.sources.append(SourceRef(dataset="cislr", id=f"cislr/{uid}", licence=LICENCES["cislr"], url=CISLR_URL))
        e.indexable = e.indexable or pn.kind not in NON_INDEX_KINDS

    out = [entries[k] for k in sorted(entries)]
    report["concepts"] = len(out)
    report["in_classifier"] = sum(e.in_classifier for e in out)
    report["indexable"] = sum(e.indexable for e in out)
    report["with_multiple_variants"] = sum(sum(v.index for v in e.variants) > 1 for e in out)
    report["regional_variants"] = sorted({v.region for e in out for v in e.variants if v.region and v.source == "islrtc"})
    report["needs_review"] = sum(e.review.status == "needs_review" for e in out)
    report["wordnet"] = wnx.get_wordnet() is not None
    report["include"]["links"] = dict(report["include"]["links"])
    report["cislr"]["links"] = dict(report["cislr"]["links"])
    return BuildResult(out, report)


def _spelling_variant(a: str, b: str) -> bool:
    """British/American spelling pairs that WordNet lists in the same synset (neighbour/neighbor,
    colour/color, grey/gray): edit distance <= 2 and the same first letter."""
    if a[:1] != b[:1] or abs(len(a) - len(b)) > 2:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 2


class _Linker:
    """Links INCLUDE/CISLR words to ISLRTC concepts.

    Merges (the word's clips become a variant of the ISLRTC concept):
      * ``exact``: same name (and the same sense/domain when the word has one);
      * ``qualified``: the word has no sense and the dictionary has exactly one domain-qualified entry with
        that name (INCLUDE "tight" -> ISLRTC "Tight (Clothes)"), ``needs_review``;
      * ``synonym``: the word is a member of an ISLRTC synonym-list name ("cheap" -> "Cheap Inexpensive"),
        ``needs_review``.
    Suggestions only (the word becomes its own concept with ``see_also``, ``needs_review``): WordNet
    synonyms of the most common sense, several domain-qualified candidates, a different sense ("you (plural)"
    vs "you") and token membership in a short multiword name ("male" in "Male Masculine"). Automatic merges
    through WordNet were tried and rejected: they produced links such as bird -> skirt and I -> one.
    """

    def __init__(self, entries: dict[str, VocabEntry]):
        self.plain: dict[str, str] = {}  # match_key(text) -> concept (no domain)
        self.with_qual: dict[tuple[str, str], str] = {}  # (match_key(text), qualifier slug) -> concept
        self.qualified: dict[str, set[str]] = defaultdict(set)  # match_key(text) -> domain-qualified concepts
        self.member: dict[str, set[str]] = defaultdict(set)  # synonym-list member key -> concepts
        self.lemma: dict[str, set[str]] = defaultdict(set)  # single-word headword -> concepts (for WordNet)
        self.token: dict[str, set[str]] = defaultdict(set)  # token of a short multiword name -> concepts
        for e in entries.values():
            self._index(e)

    def _index(self, e: VocabEntry) -> None:
        base, _, qual = e.concept_id.removeprefix("isl.").partition("~")
        key = match_key(base)
        if qual:
            self.with_qual.setdefault((key, qual), e.concept_id)
            self.qualified[key].add(e.concept_id)
        else:
            self.plain.setdefault(key, e.concept_id)
        if e.kind in ("synonym_list", "number"):
            for s in e.synonyms:
                self.member[match_key(s)].add(e.concept_id)
        if not qual and " " not in e.headword.strip() and e.kind in ("word", "number"):
            self.lemma[e.headword.lower()].add(e.concept_id)
        if e.kind == "multiword" and not qual:
            toks = e.headword.lower().split()
            if 2 <= len(toks) <= 4 and not any(t in wnx.STOPWORDS for t in toks):
                for t in toks:
                    self.token[match_key(t)].add(e.concept_id)

    def add(self, e: VocabEntry, pn: ParsedName) -> None:
        self._index(e)

    def link(self, pn: ParsedName) -> tuple[str | None, str, bool, list[str]]:
        """(concept to merge into or None, link kind, needs_review, see_also suggestions)."""
        qual = slugify(pn.sense or pn.domain) if (pn.sense or pn.domain) else None
        words = pn.synonyms if pn.synonyms else [pn.headword]
        keys = [match_key(w) for w in words]
        sugg: list[str] = []
        for k in keys:
            if qual:
                if (k, qual) in self.with_qual:
                    return self.with_qual[(k, qual)], "exact", False, []
                # different sense: suggest, never merge ("you (plural)" is not "you")
                sugg += [c for c in [self.plain.get(k)] if c] + sorted(self.qualified.get(k, ()))
            elif k in self.plain:
                return self.plain[k], "exact", False, []
        if not qual:
            for k in keys:
                q = sorted(self.qualified.get(k, ()))
                if len(q) == 1:
                    return q[0], "qualified", True, []
                sugg += q
            for k in keys:
                cands = sorted(self.member.get(k, ()))
                if cands:
                    return cands[0], "synonym", True, cands[1:]
        if wnx.get_wordnet() is not None:
            for w in words:
                if " " in w.strip():
                    continue
                ss = wnx.synsets(w)[:1]  # most common sense only
                for lem in (lm.name().replace("_", " ").lower() for s in ss for lm in s.lemmas()):
                    if lem != w.lower() and lem in self.lemma:
                        if not qual and _spelling_variant(w.lower(), lem):  # neighbour / neighbor
                            return sorted(self.lemma[lem])[0], "spelling", False, []
                        sugg += sorted(self.lemma[lem])
        for k in keys:
            sugg += sorted(self.token.get(k, ()))[:5]
        sugg = list(dict.fromkeys(sugg))[:8]
        return None, "new", bool(sugg), sugg


# --------------------------------------------------------------------------------------------- I/O helpers

def load_islrtc_paths(paths) -> list[str]:
    """All ISLRTC dictionary paths: the full HF index when present, else the ids seen in the manifest."""
    from app.isl.data.manifest import iter_jsonl

    if paths.islrtc_index.exists():
        try:
            return [f["path"] for f in json.loads(paths.islrtc_index.read_text())]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return sorted({r["id"].removeprefix("islrtc/") for r in iter_jsonl(paths.manifest("islrtc")) if r.get("id")})


def load_include_words(paths) -> dict[str, int]:
    """INCLUDE word folders -> clip count, from the official split lists (complete) and the manifest."""
    from app.isl.data.sources import include_key_from_parts, read_split_list

    counts: Counter = Counter()
    for name in ("include_train", "include_val", "include_test"):
        f = paths.official_splits / f"{name}.txt"
        for line in read_split_list(f):
            key = include_key_from_parts(PurePosixPath(line).parts)
            if key:
                counts[key[0]] += 1
    man: Counter = Counter()
    from app.isl.data.manifest import read_manifest

    for r in read_manifest(paths.manifest("include")):
        key = include_key_from_parts(PurePosixPath(r.get("member") or "").parts)
        if key:
            man[key[0]] += 1
    # word dirs are compared on the underscore form; keep the split-list spelling when both exist
    by_norm = {k.replace(" ", "_").lower(): k for k in counts}
    for k, n in man.items():
        kk = by_norm.get(k.replace(" ", "_").lower())
        if kk is None:
            counts[k] = n
        else:
            counts[kk] = max(counts[kk], n)
    return dict(counts)


def load_cislr_rows(paths) -> list[dict]:
    import csv

    f = paths.raw / "cislr" / "dataset.csv"
    if not f.exists():
        return []
    with f.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    from app.isl.data.cislr import normalise_cislr_row

    return [r for r in (normalise_cislr_row(x) for x in rows) if r]


def load_trims(paths) -> dict[str, dict]:
    from app.isl.data.manifest import read_manifest

    return {r["id"]: r for r in read_manifest(paths.trim_sidecar, only_ok=False)}


def write_vocabulary(entries: list[VocabEntry], path: Path) -> None:
    from app.isl.data.manifest import write_jsonl

    write_jsonl(path, [e.to_json() for e in entries])


def read_vocabulary(path: Path) -> list[VocabEntry]:
    from app.isl.data.manifest import iter_jsonl

    return [VocabEntry.model_validate(r) for r in iter_jsonl(path)]


class VocabIndex:
    """Clip/variant -> concept lookup over a built vocabulary (used by the source adapters)."""

    def __init__(self, entries: list[VocabEntry]):
        self.entries = entries
        self.class_index = {e.concept_id: i for i, e in enumerate(entries)}
        self.by_clip: dict[str, tuple[str, str]] = {}
        self.by_variant: dict[str, str] = {}
        for e in entries:
            for v in e.variants:
                self.by_variant[v.variant_id] = e.concept_id
                for c in v.clips:
                    self.by_clip[c] = (e.concept_id, v.variant_id)

    @classmethod
    def load(cls, path: Path) -> VocabIndex:
        return cls(read_vocabulary(path))

    def concept_of_clip(self, clip_id: str) -> tuple[str | None, str | None]:
        return self.by_clip.get(clip_id, (None, None))

    def concept_of_variant(self, variant_id: str) -> str | None:
        return self.by_variant.get(variant_id)
