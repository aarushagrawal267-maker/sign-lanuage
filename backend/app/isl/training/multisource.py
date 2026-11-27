"""Every usable ISL source for phase-2 training, per data profile (docs/ISL_DESIGN.md 15.2).

Three roles per source:

* **labelled words** (classifier, encoder fine-tuning, retrieval index): INCLUDE, ISLRTC, ISL500, WSLP, CISLR,
  Mendeley emergency / agriculture words;
* **pre-training only** (self-supervised, labels never used): sentence / continuous pose data such as GloSL-ISL,
  OpenHands raw, iSign, ISL-CSLTR sentences, WSLP unlabelled clips - plus the labelled word clips *without labels*;
* **letters** (fingerspelling): ISL-Fingerspelling (per-letter boundaries), OpenHands Indian fingerspelling,
  static alphabet images.

Licence rule: ``commercial-safe`` only takes CC-BY / CC0 / GODL-type sources whose adapter says
``commercial = True``; everything NC / ND / research-only / undocumented goes to ``research`` only. The rule is
enforced twice (the allow-list below **and** the adapter's own ``commercial`` flag): :func:`check_licence`
raises :class:`LicenceError` instead of silently mixing data. My signs are never a training source.

Leakage guards used by training and evaluation:

* **vocabulary level**: :class:`GlossLinker` maps every source's gloss onto one concept id (exact headword, then
  unique synonym), so "Range" from WSLP and "range" from ISL500 are one class and one held-out concept;
  :meth:`GlossLinker.expand` widens a held-out concept set to every concept sharing a headword / synonym key.
* **video / file level**: :class:`DedupIndex` unions clips with the same source-video SHA-256 (``quality.sha256``),
  the same landmark-file SHA-256, or a cluster in ``derived/duplicates.jsonl`` (landmark near-duplicates found by
  ``cli dedup``). Training clips that share a duplicate group with any evaluation clip are dropped.

Adapters for the newer sources live in ``app.isl.data.video_sources`` and ``app.isl.data.pose_sources`` (owned by the
data agents); this module only codes against the ``SourceAdapter`` protocol, so a missing package or manifest just
means "source not available yet". Tests register synthetic adapters with :func:`register_adapter`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.isl.data.sources import Clip
from app.isl.paths import IslPaths
# The source lists live in app.isl.profiles (single source of truth, also behind DataProfile); re-exported here.
from app.isl.profiles import (
    COMMERCIAL,
    LETTER_SOURCES,
    PRETRAIN_SOURCES,
    RESEARCH,
    WORD_SOURCES,
    get_profile,
)

log = logging.getLogger(__name__)

# sources whose ``group`` is a real signer id (signer-independent held-out groups for evaluation)
SIGNER_SOURCES = ("isl500", "mendeley_emergency")
# per-source weight of retrieval index rows (dictionary references first; noisy web clips slightly lower)
INDEX_WEIGHTS = {"islrtc": 1.0, "include": 1.0, "isl500": 1.0, "mendeley_emergency": 0.97, "mendeley_agri": 0.97,
                 "cislr": 0.95, "wslp": 0.95}
# per-source sampling weight in training (relative probability of drawing a clip of that source)
SAMPLE_WEIGHTS = {"include": 1.0, "islrtc": 1.0, "isl500": 1.0, "wslp": 0.8, "cislr": 0.8,
                  "mendeley_emergency": 0.5, "mendeley_agri": 0.5}

_EXTRA: dict[str, Callable] = {}


class LicenceError(ValueError):
    """A source that is not allowed in the requested data profile."""


def register_adapter(name: str, factory: Callable | None) -> None:
    """Register (or with ``None`` remove) an adapter factory ``factory(paths, vocab) -> adapter`` (tests)."""
    if factory is None:
        _EXTRA.pop(name, None)
    else:
        _EXTRA[name] = factory


def _factories() -> dict[str, Callable]:
    from app.isl.data import sources as base

    f: dict[str, Callable] = {k: (lambda p, v, _c=c: _c(paths=p, vocab=v)) for k, c in base.all_adapters().items()}
    wslp = base.all_adapters().get("wslp")
    if wslp is not None:  # the unlabelled WSLP val / test rows share the labelled manifest: a filtered view
        f["wslp_unlabelled"] = lambda p, v: _Unlabelled(wslp(paths=p, vocab=v, require_label=False), "wslp")
    f.update(_EXTRA)
    return f


class _Unlabelled:
    """View of an adapter that keeps only rows without a label (WSLP validation / test clips)."""

    def __init__(self, inner, name: str):
        self.inner, self.name = inner, f"{name}_unlabelled"
        self.licence, self.commercial = inner.licence, inner.commercial

    def available(self) -> bool:
        return self.inner.available()

    def clips(self, *, only_ok: bool = True):
        for c in self.inner.clips(only_ok=only_ok):
            if not c.raw_gloss and c.quality.get("labelled") is not True:
                yield c


def check_licence(profile: str, name: str, adapter) -> None:
    prof = get_profile(profile)
    allowed = set(WORD_SOURCES[prof.name]) | set(PRETRAIN_SOURCES[prof.name]) | set(LETTER_SOURCES[prof.name])
    if name not in allowed:
        raise LicenceError(f"source {name!r} is not allowed in profile {prof.name!r}")
    if prof.commercial and not getattr(adapter, "commercial", False):
        raise LicenceError(f"source {name!r} (licence {getattr(adapter, 'licence', '?')!r}) is not commercial-safe")


def open_adapter(name: str, paths: IslPaths, vocab=None, profile: str | None = None):
    """The adapter for ``name`` or None when its code or manifest is missing. With ``profile`` the licence
    rule is checked (raises :class:`LicenceError`)."""
    fac = _factories().get(name)
    if fac is None:
        return None
    try:
        ad = fac(paths, vocab)
    except Exception as e:  # noqa: BLE001 - a broken optional adapter must not stop training on the others
        log.warning("adapter %s unavailable: %s", name, e)
        return None
    if profile is not None:
        check_licence(profile, name, ad)
    try:
        return ad if ad.available() else None
    except OSError:
        return None


def read_clips(names: Iterable[str], paths: IslPaths, vocab=None, profile: str | None = None,
               report: dict | None = None) -> list[Clip]:
    out: list[Clip] = []
    for n in names:
        ad = open_adapter(n, paths, vocab, profile)
        if ad is None:
            if report is not None:
                report[n] = {"available": False}
            continue
        try:
            cl = list(ad.clips())
        except (OSError, ValueError, KeyError) as e:
            log.warning("source %s: %s", n, e)
            cl = []
        if report is not None:
            report[n] = {"available": True, "clips": len(cl), "licence": getattr(ad, "licence", ""),
                         "commercial": bool(getattr(ad, "commercial", False))}
        out += cl
    return out


# --------------------------------------------------------------------------------------------- linking

def _key(text: str | None) -> str:
    from app.isl.vocab.normalize import match_key

    return match_key(text)


class GlossLinker:
    """Source gloss -> concept id over the unified vocabulary (vocabulary-level dedupe across sources).

    Order: the adapter's own ``concept_id`` (vocabulary built with that source) -> exact headword key
    (preferring the plain concept over domain/sense-qualified ones) -> a *unique* synonym key -> a new
    source-independent concept ``isl.<slug>`` (the same gloss from two sources gives the same new id).
    New concepts are written by :meth:`extra_concepts` so a pack can show them.

    Opt-in (docs 15.34, link step ``--link-aliases``): ``strip_numbering`` removes WSLP's trailing "(N)" before
    linking (only for the sources in ``aliases.STRIP_NUMBERING_SOURCES``: WSLP), so "Adult(1)" links to
    ``isl.adult`` instead of a new ``isl.adult-1``; ``aliases`` (alias -> root, e.g.
    ``app.isl.vocab.aliases.TWIN_ALIASES``) gives :attr:`alias_of`, the alias groups of known concepts.
    Both off by default: links exactly as before.

    ``variants`` (docs 15.36, ``cli variant-audit``): source -> {concept: variant concept}, e.g.
    ``{"isl500": {"isl.hot": "isl.hot~isl500"}}``. Every clip of that source linked to the concept gets the variant
    concept instead (a new concept that shows the same word, ``alias_of`` the concept). Off by default."""

    LINKABLE_KINDS = {"word", "multiword", "synonym_list", "number", "phrase", "letter", None}

    def __init__(self, entries: Iterable = (), *, strip_numbering: bool = False, aliases: dict | None = None,
                 variants: dict | None = None):
        self.strip_numbering = bool(strip_numbering)
        self.variants: dict[str, dict[str, str]] = {str(s): {str(w): str(v) for w, v in (m or {}).items()}
                                                    for s, m in (variants or {}).items() if m}
        # variant concepts are aliases of their concept (docs 15.36): the word shown stays the same
        self.alias_table = dict(aliases or {}) | {v: w for m in self.variants.values() for w, v in m.items()}
        self._display: dict[str, dict] = {}  # concept -> headword / display / kind / category (variant rows only)
        self.by_head: dict[str, list[str]] = defaultdict(list)
        self.by_syn: dict[str, set[str]] = defaultdict(set)
        self.keys_of: dict[str, set[str]] = defaultdict(set)
        self.known: set[str] = set()
        for e in entries:
            cid = e.concept_id if hasattr(e, "concept_id") else e["concept_id"]
            get = (lambda k, _e=e: getattr(_e, k, None)) if hasattr(e, "concept_id") else e.get
            self.known.add(cid)
            if self.variants:
                self._display[cid] = {"headword": get("headword"), "display": get("display"), "kind": get("kind"),
                                      "category": get("category")}
            if get("kind") not in self.LINKABLE_KINDS:
                continue
            hk = _key(get("headword"))
            if hk:
                self.by_head[hk].append(cid)
                self.keys_of[cid].add(hk)
            for s in get("synonyms") or []:
                sk = _key(s)
                if sk:
                    self.by_syn[sk].add(cid)
                    self.keys_of[cid].add(sk)
        self.new: dict[str, dict] = {}
        self.counts: Counter = Counter()

    @classmethod
    def from_vocab(cls, vocab, **kw) -> GlossLinker:
        if vocab is None:
            return cls([], **kw)
        return cls(getattr(vocab, "entries", None) or getattr(vocab, "rows", None) or [], **kw)

    @property
    def alias_of(self) -> dict[str, str]:
        """alias -> root over the concepts this linker knows (vocabulary and new ones); {} when aliases are off."""
        if not self.alias_table:
            return {}
        from app.isl.vocab.aliases import resolve_aliases

        return resolve_aliases(self.alias_table, self.known | set(self.new))

    def link(self, gloss: str, source: str = "") -> tuple[str | None, str]:
        from app.isl.vocab.aliases import STRIP_NUMBERING_SOURCES, strip_sign_number
        from app.isl.vocab.normalize import slugify

        if self.strip_numbering and gloss and source in STRIP_NUMBERING_SOURCES:  # WSLP only (docs 15.35)
            g = strip_sign_number(gloss)
            if g != gloss:
                self.counts["numbering_stripped"] += 1
                self.counts[f"numbering_stripped:{source}"] += 1
                gloss = g
        k = _key(gloss)
        if not k:
            return None, "empty"
        heads = self.by_head.get(k) or []
        if heads:
            plain = [c for c in heads if "~" not in c]
            cid = sorted(plain or heads)[0]
            self.counts["exact"] += 1
            return cid, "exact"
        syn = self.by_syn.get(k) or set()
        if len(syn) == 1:
            self.counts["synonym"] += 1
            return next(iter(syn)), "synonym"
        cid = f"isl.{slugify(gloss)}"
        if cid in self.known:  # slug equals an existing id with a different key (punctuation)
            self.counts["exact"] += 1
            return cid, "exact"
        row = self.new.setdefault(cid, {"concept_id": cid, "headword": " ".join(gloss.split()).lower(),
                                        "display": {"en": " ".join(gloss.split())}, "kind": "word",
                                        "category": "other", "variants": [], "in_classifier": False,
                                        "indexable": True, "review": {"status": "auto", "notes": ["phase-2 link: new"]},
                                        "sources": []})
        if source and source not in row["sources"]:
            row["sources"].append(source)
        self.keys_of[cid].add(k)
        self.counts["new" if len(syn) == 0 else "ambiguous_synonym_new"] += 1
        return cid, "new"

    def variant_of(self, cid: str, source: str) -> str:
        """The variant concept of ``cid`` for clips of ``source`` (docs 15.36), else ``cid``. A variant id that is
        already a vocabulary concept is never used (counted as ``variant_skipped_known``)."""
        v = (self.variants.get(source) or {}).get(cid)
        if not v:
            return cid
        if v in self.known:
            self.counts["variant_skipped_known"] += 1
            return cid
        if v not in self.new:
            base = self.new.get(cid) or self._display.get(cid) or {}
            head = base.get("headword") or cid.split(".", 1)[-1]
            self.new[v] = {"concept_id": v, "headword": head, "display": dict(base.get("display") or {"en": head}),
                           "kind": base.get("kind") or "word", "category": base.get("category") or "other",
                           "variants": [], "in_classifier": False, "indexable": True, "alias_of": cid,
                           "review": {"status": "auto", "notes": [f"phase-2 variant form of {cid} in {source} "
                                                                  "(variant-audit, docs 15.36)"]},
                           "sources": [source]}
        self.counts["variant_relinked"] += 1
        self.counts[f"variant_relinked:{source}"] += 1
        return v

    def link_clips(self, clips: list[Clip]) -> list[Clip]:
        out = []
        for c in clips:
            if c.concept_id:
                self.counts["adapter"] += 1
                v = self.variant_of(c.concept_id, c.source) if self.variants else c.concept_id
                out.append(c if v == c.concept_id else replace(c, concept_id=v,
                                                               quality={**c.quality, "variant_of": c.concept_id}))
                continue
            cid, how = self.link(c.raw_gloss, c.source)
            if cid is None:
                continue
            q = {**c.quality, "link": how}
            v = self.variant_of(cid, c.source) if self.variants else cid
            if v != cid:
                q["variant_of"] = cid
            out.append(replace(c, concept_id=v, variant_id=c.variant_id or f"{c.source}:{_key(c.raw_gloss)}",
                               quality=q))
        return out

    def expand(self, concepts: Iterable[str]) -> set[str]:
        """Every concept that shares a headword / synonym key with one of ``concepts`` (leakage guard)."""
        keys = set()
        base = set(concepts)
        for c in base:
            keys |= self.keys_of.get(c, set())
        out = set(base)
        for k in keys:
            out |= set(self.by_head.get(k, [])) | self.by_syn.get(k, set())
        for cid, ks in self.keys_of.items():
            if ks & keys:
                out.add(cid)
        al = self.alias_of
        if al:  # the whole alias group of every concept found (docs 15.34)
            roots = {al.get(c, c) for c in out}
            out |= roots | {a for a, r in al.items() if r in roots}
        return out

    def extra_concepts(self) -> list[dict]:
        return [self.new[k] for k in sorted(self.new)]


# --------------------------------------------------------------------------------------------- dedupe

def file_sha256(path: Path, cache: dict | None = None) -> str | None:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if cache is not None and key in cache:
        return cache[key]
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    d = h.hexdigest()
    if cache is not None:
        cache[key] = d
    return d


class DedupIndex:
    """Union of clips that are the same recording (video hash, landmark-file hash, near-duplicate cluster)."""

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.reasons: Counter = Counter()

    def _find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str, reason: str = "") -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)
            self.reasons[reason] += 1

    def group(self, clip_id: str) -> str:
        return self._find(clip_id)

    @classmethod
    def build(cls, clips: list[Clip], duplicates_file: Path | None = None, hash_files: bool = True) -> DedupIndex:
        d = cls()
        first: dict[str, str] = {}
        cache: dict = {}
        for c in clips:
            d._find(c.clip_id)
            keys = []
            sha = c.quality.get("sha256") if isinstance(c.quality, dict) else None
            if sha:
                keys.append(("video", sha))
            if hash_files:
                fs = file_sha256(c.path, cache)
                if fs:
                    keys.append(("file", fs))
            for kind, k in keys:
                kk = f"{kind}:{k}"
                if kk in first:
                    d.union(first[kk], c.clip_id, f"same_{kind}")
                else:
                    first[kk] = c.clip_id
        if duplicates_file is not None and Path(duplicates_file).exists():
            from app.isl.data.manifest import iter_jsonl

            for row in iter_jsonl(Path(duplicates_file)):
                cl = row.get("clips") or []
                for x in cl[1:]:
                    d.union(cl[0], x, "landmark_near_duplicate")
        return d


def drop_leaking(train: list[Clip], held: Iterable[Clip], dedup: DedupIndex) -> tuple[list[Clip], int]:
    """Training clips minus those in the duplicate group of any held-out (val/test/query) clip."""
    bad = {dedup.group(c.clip_id) for c in held}
    keep = [c for c in train if dedup.group(c.clip_id) not in bad]
    return keep, len(train) - len(keep)


# --------------------------------------------------------------------------------------------- splits

def signer_split(clips: list[Clip], val_frac: float = 0.1, test_frac: float = 0.15, min_groups: int = 6,
                 seed: int = 0) -> dict[str, str]:
    """clip id -> train | val | test with **whole signers** held out (sources with real signer ids). Sources
    with fewer than ``min_groups`` signers stay in training."""
    by_src: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        if c.source in SIGNER_SOURCES and c.group:
            by_src[c.source].add(c.group)
    part_of_group: dict[str, str] = {}
    for src, groups in by_src.items():
        g = sorted(groups)
        if len(g) < min_groups:
            continue
        random.Random(f"{seed}:{src}").shuffle(g)
        n_test = max(1, round(test_frac * len(g)))
        n_val = max(1, round(val_frac * len(g)))
        for i, x in enumerate(g):
            part_of_group[x] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return {c.clip_id: part_of_group.get(c.group or "", "train") for c in clips}


@dataclass
class MultiSourceReport:
    sources: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)
    dedup: dict = field(default_factory=dict)
    parts: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"sources": self.sources, "links": self.links, "dedup": self.dedup, "parts": self.parts}


def attach_all_sources(data, paths: IslPaths, vocab, profile: str, *, seed: int = 0,
                       duplicates_file: Path | None = None, hash_files: bool = True,
                       max_per_source: int | None = None, link: dict | None = None) -> MultiSourceReport:
    """Add every other labelled word source of ``profile`` to an ``IslData`` (``data.extra`` and
    ``data.extra_assign``), link their glosses, and drop training clips that duplicate evaluation clips.
    INCLUDE keeps its own split; ISLRTC stays a training / gallery source. ``link``: the work folder's link
    settings (:func:`read_link_settings`; default: link as before)."""
    rep = MultiSourceReport()
    names = [n for n in WORD_SOURCES[get_profile(profile).name] if n not in ("include", "islrtc")]
    clips = read_clips(names, paths, vocab, profile, rep.sources)
    if max_per_source:
        by = defaultdict(list)
        for c in clips:
            by[c.source].append(c)
        clips = [c for v in by.values() for c in random.Random(seed).sample(v, min(len(v), max_per_source))]
    linker = GlossLinker.from_vocab(vocab, **linker_options(link))
    clips = linker.link_clips(clips)
    # clips of the core INCLUDE / ISLRTC sources that the vocabulary did not link get linked by gloss too
    data.linker = linker
    assign = signer_split(clips, seed=seed)
    dedup = DedupIndex.build(data.include + data.islrtc + clips,
                             duplicates_file if duplicates_file is not None else paths.derived / "duplicates.jsonl",
                             hash_files=hash_files)
    held = [c for c in data.include if data.assign.get(c.clip_id) in ("val", "test")]
    held += [c for c in clips if assign.get(c.clip_id) in ("val", "test")]
    train_extra = [c for c in clips if assign.get(c.clip_id) == "train"]
    keep, dropped_extra = drop_leaking(train_extra, held, dedup)
    keep_isl, dropped_isl = drop_leaking(data.islrtc, held, dedup)
    data.islrtc = keep_isl
    # duplicates *inside* the evaluation side across sources: keep one copy (first by source preference)
    seen_groups: set[str] = set()
    eval_extra = []
    inc_g = {dedup.group(x.clip_id) for x in held if x.source == "include"}
    for c in [c for c in clips if assign.get(c.clip_id) in ("val", "test")]:
        g = dedup.group(c.clip_id)
        if g in seen_groups or g in inc_g:
            continue
        seen_groups.add(g)
        eval_extra.append(c)
    data.extra = keep + eval_extra
    data.extra_assign = {c.clip_id: assign[c.clip_id] for c in data.extra}
    data.dedup = dedup
    rep.links = dict(linker.counts) | {"new_concepts": len(linker.new)} | (
        {"aliases": len(linker.alias_of)} if linker.alias_table else {}) | (
        {"variant_concepts": sum(1 for m in linker.variants.values() for v in m.values() if v in linker.new)}
        if linker.variants else {})
    rep.dedup = {"unions": dict(dedup.reasons), "dropped_train_extra": dropped_extra,
                 "dropped_train_islrtc": dropped_isl,
                 "dropped_eval_duplicates": sum(1 for c in clips if assign.get(c.clip_id) in ("val", "test"))
                 - len(eval_extra)}
    rep.parts = {src: dict(Counter(data.extra_assign[c.clip_id] for c in data.extra if c.source == src))
                 for src in sorted({c.source for c in data.extra})}
    data.multisource = rep.to_json()
    return rep


def pretrain_clips(paths: IslPaths, vocab, profile: str, exclude_ids: set[str] | None = None,
                   report: dict | None = None, with_word_sources: bool = True) -> list[Clip]:
    """Every clip a profile allows for self-supervised pre-training (labels are never used). Evaluation clips
    (``exclude_ids``) are left out so test videos are never seen, not even unlabelled."""
    prof = get_profile(profile).name
    names = list(PRETRAIN_SOURCES[prof]) + (list(WORD_SOURCES[prof]) if with_word_sources else [])
    clips = read_clips(dict.fromkeys(names), paths, vocab, profile, report)
    ex = exclude_ids or set()
    return [c for c in clips if c.clip_id not in ex]


def eval_clip_ids(data) -> set[str]:
    """Ids of every evaluation clip (INCLUDE val/test of the split, other sources' val/test signers) plus every clip
    in the same duplicate group (same video / landmark file / near-duplicate cluster): copies that
    ``attach_all_sources`` dropped from training or from the evaluation side are re-read by ``pretrain_clips``, so
    excluding by id alone would let a test video reach pre-training."""
    ids = {c.clip_id for c in data.include if data.assign.get(c.clip_id) in ("val", "test")} | {
        c.clip_id for c in getattr(data, "extra", []) if data.extra_assign.get(c.clip_id) in ("val", "test")}
    dedup = getattr(data, "dedup", None)
    if dedup is not None and getattr(dedup, "parent", None):
        groups = {dedup.group(i) for i in ids}
        ids |= {x for x in list(dedup.parent) if dedup.group(x) in groups}
    return ids


def eval_clips(data) -> list[Clip]:
    """The evaluation clips themselves (for content-based checks such as the Sign Library dedupe)."""
    return [c for c in data.include if data.assign.get(c.clip_id) in ("val", "test")] + [
        c for c in getattr(data, "extra", []) if data.extra_assign.get(c.clip_id) in ("val", "test")]


PRETRAIN_LIST = "pretrain_clips.txt"


def write_pretrain_list(work: Path, clips: list[Clip], resume: bool = False) -> dict:
    """``<work>/pretrain_clips.txt``: every clip id the pre-training may draw (union with the previous list when
    a run is resumed, so a leakage check never misses a clip an earlier part saw). Returns name, count, sha256."""
    f = work / PRETRAIN_LIST
    ids = {c.clip_id for c in clips}
    if resume and f.exists():
        ids |= {ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()}
    text = "".join(f"{i}\n" for i in sorted(ids))
    f.write_text(text, encoding="utf-8")
    return {"file": PRETRAIN_LIST, "clips": len(ids), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def read_pretrain_list(work: Path) -> set[str] | None:
    f = work / PRETRAIN_LIST
    if not f.exists():
        return None
    return {ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()}


# --------------------------------------------------------------------------------------------- link settings

LINK_SETTINGS = "link_settings.json"


def link_settings(link_aliases: bool = False, variants: dict | None = None, variant_meta: dict | None = None) -> dict:
    """What the link step decides for a whole work folder (docs 15.34). ``link_aliases``: strip WSLP's "(N)"
    numbering and use the twin alias table. Off: exactly the old linking. ``variants`` / ``variant_meta``: the
    relinks of a variant-links file and its file name and SHA-256 (``variant_audit.load_variant_links``, docs 15.36);
    frozen into the settings, so later steps never re-read the audit file. No variants: no new key (as before)."""
    from app.isl.vocab.aliases import ALIAS_VERSION

    on = bool(link_aliases)
    d = {"version": 1, "strip_sign_numbers": on, "aliases": on, "alias_version": ALIAS_VERSION if on else None}
    if variants:
        d["variants"] = {s: dict(sorted(m.items())) for s, m in sorted(variants.items()) if m}
        d["variant_links"] = dict(variant_meta or {})
    return d


def linker_options(settings: dict | None) -> dict:
    """``GlossLinker`` keyword arguments of link settings (``{}`` or None: the old linking). A settings FILE carries
    the map the link step froze (``alias_of``): later steps use that map, never the table in the current code, so
    every step of one work folder groups the same way after the table changes (docs 15.35). Only the link step itself
    (settings from :func:`link_settings`, no ``alias_of`` yet) reads ``TWIN_ALIASES``."""
    from app.isl.vocab.aliases import TWIN_ALIASES

    s = settings or {}
    if "alias_of" in s and (s.get("aliases") or s.get("variants")):
        al = dict(s.get("alias_of") or {})  # frozen by the link step (twins and / or variant pairs)
    elif s.get("aliases"):
        al = TWIN_ALIASES
    else:
        al = None
    out = {"strip_numbering": bool(s.get("strip_sign_numbers")), "aliases": al}
    if s.get("variants"):  # docs 15.36: only when the link step froze variant relinks
        out["variants"] = dict(s["variants"])
    return out


def read_link_settings(work: Path) -> dict:
    """``<work>/link_settings.json`` written by the link step; ``{}`` when there is none (a work folder linked before
    docs 15.34: every later step links as before)."""
    f = Path(work) / LINK_SETTINGS
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return d if isinstance(d, dict) else {}


def write_link_settings(work: Path, settings: dict, linker: GlossLinker | None = None) -> dict:
    """Write the link step's settings, with the resolved alias map of ``linker``, so every later step of the work
    folder (training, index, fit-fusion, evaluate) links and scores the same way."""
    from app.isl.vocab.aliases import aliases_sha256

    al = linker.alias_of if (linker is not None and (settings.get("aliases") or settings.get("variants"))) else {}
    d = {**settings, "alias_of": al, "alias_sha256": aliases_sha256(al),
         "numbering_stripped": int(linker.counts.get("numbering_stripped", 0)) if linker is not None else 0}
    if settings.get("variants"):  # docs 15.36: how many clips the frozen relinks moved
        d["variant_relinked"] = int(linker.counts.get("variant_relinked", 0)) if linker is not None else 0
    f = Path(work) / LINK_SETTINGS
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(f)
    return d


def write_extra_concepts(linker: GlossLinker, path: Path) -> int:
    rows = linker.extra_concepts()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
