"""Name normalisation for ISL vocabulary sources (docs/ISL_DESIGN.md section 6.1, rules 1-11).

Turns raw source names (ISLRTC dictionary file names, INCLUDE word folders, CISLR words) into a
:class:`ParsedName`: headword, synonyms, sense/domain, region, variant number, kind and flags. Parsing is
pure and deterministic given the WordNet data (optional, see ``wordnet.py``) and ``overrides.yaml``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

from app.isl.vocab import wordnet as wnx

KINDS = ("word", "multiword", "synonym_list", "phrase", "number", "letter", "explanation", "example", "compilation")
# Kinds that are not one recognisable sign of the concept: kept in the vocabulary, never indexed.
NON_INDEX_KINDS = frozenset({"explanation", "example", "compilation"})
# Kinds indexed with the full source clip (no first-sign trim).
FULL_CLIP_KINDS = frozenset({"phrase"})


@dataclass
class ParsedName:
    headword: str  # "pencil" (lower case unless acronym / proper noun)
    synonyms: list[str] = field(default_factory=list)  # ["abhor", "despise", ...] (first = headword); [] if none
    sense: str | None = None  # "number" in "second (number)"
    domain: str | None = None  # "academic" | "math" | ...
    region: str | None = None  # "Indore, Madhya Pradesh"
    sign_no: int = 1
    kind: str = "word"
    number_value: int | None = None
    flags: list[str] = field(default_factory=list)
    text: str = ""  # cleaned full text the concept id is built from ("abhor despise detest hate loathe")
    qualifiers: list[str] = field(default_factory=list)  # extra parenthesised notes beyond the first domain
    original: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def concept_key(self) -> tuple[str, str | None]:
        return slugify(self.text or self.headword), slugify(self.sense or self.domain) if (self.sense or self.domain) else None

    @property
    def concept_id(self) -> str:
        base, qual = self.concept_key
        return f"isl.{base}" + (f"~{qual}" if qual else "")


# --------------------------------------------------------------------------------------------- basics

def clean_text(s: str) -> str:
    """Rule 1: ``_`` -> space, Unicode NFC, collapse whitespace, trim."""
    s = unicodedata.normalize("NFC", s).replace("_", " ")
    return " ".join(s.split()).strip()


def slugify(text: str | None) -> str:
    """"AC to DC Converter" -> "ac-to-dc-converter" (ASCII-folded, lower case, runs of other chars -> '-')."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "x"


def match_key(text: str | None) -> str:
    """Loose key for cross-source matching: lower case, letters and digits only ("T-Shirt" == "t shirt")."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", t)


# --------------------------------------------------------------------------------------------- gazetteer

STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat", "haryana",
    "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
    "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal", "delhi", "new delhi",
    "jammu and kashmir", "ladakh", "puducherry", "chandigarh",
]
CITIES = {
    "indore": "madhya pradesh", "bhopal": "madhya pradesh", "jabalpur": "madhya pradesh", "gwalior": "madhya pradesh",
    "ujjain": "madhya pradesh", "mumbai": "maharashtra", "pune": "maharashtra", "nagpur": "maharashtra",
    "hyderabad": "telangana", "chennai": "tamil nadu", "coimbatore": "tamil nadu", "kolkata": "west bengal",
    "bengaluru": "karnataka", "bangalore": "karnataka", "mysuru": "karnataka", "thiruvananthapuram": "kerala",
    "kochi": "kerala", "munger": "bihar", "patna": "bihar", "ahmedabad": "gujarat", "vadodara": "gujarat",
    "surat": "gujarat", "lucknow": "uttar pradesh", "varanasi": "uttar pradesh", "jaipur": "rajasthan",
    "bhubaneswar": "odisha", "guwahati": "assam", "ranchi": "jharkhand", "raipur": "chhattisgarh",
    "dehradun": "uttarakhand", "shimla": "himachal pradesh",
}
_PLACES = sorted(set(STATES) | set(CITIES), key=lambda p: -len(p.split()))
_SMALL = {"and", "of"}


def _title(place: str) -> str:
    return " ".join(w if w in _SMALL else w.capitalize() for w in place.split())


def parse_region(text: str) -> str | None:
    """"Indore Madhya Pradesh" -> "Indore, Madhya Pradesh"; "Indore" -> "Indore, Madhya Pradesh";
    "Andhra Pradesh Telangana" -> "Andhra Pradesh, Telangana"; None unless every token is a known place."""
    toks = clean_text(text).lower().replace(",", " ").split()
    if not toks:
        return None
    found: list[str] = []
    i = 0
    while i < len(toks):
        for p in _PLACES:
            n = len(p.split())
            if toks[i : i + n] == p.split():
                found.append(p)
                i += n
                break
        else:
            return None
    found = ["odisha" if p == "orissa" else p for p in found]
    out: list[str] = []
    for p in found:
        if p not in out:
            out.append(p)
        if p in CITIES and CITIES[p] not in found:
            out.append(CITIES[p])
    return ", ".join(_title(p) for p in out)


# --------------------------------------------------------------------------------------------- markers

DOMAIN_ALIASES = {
    "maths": "math", "mathematics": "math", "admin": "administrator", "financial term": "finance",
    "financial": "finance", "social science": "social science", "sci": "science", "geo": "geography",
}
NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand lakh "
    "lakhs crore crores million billion and".split()
)

_RE_VARIANT = re.compile(r"^(?:sign)?\s*(\d+)?$", re.I)
_RE_EXPLAIN = re.compile(r"^(?:sign\s+and\s+)?expla(?:i)?n(?:a)?tion(?:\s*\d+)?$", re.I)
_RE_EXAMPLE = re.compile(r"^examples?(?:\s*\d+)?$", re.I)
_RE_PAREN = re.compile(r"\(([^()]*)\)")
_RE_ENGLISH = re.compile(r"\s*-\s*english\s*$", re.I)
_RE_ACRONYM = re.compile(r"^[A-Z][A-Z0-9&.\-]{1,9}$")


def _is_acronym_token(tok: str) -> bool:
    return bool(_RE_ACRONYM.match(tok)) and any(c.isalpha() for c in tok)


def _initials(words: list[str]) -> str:
    return "".join(w[0] for w in words if w and w.lower() not in wnx.STOPWORDS).upper()


def _display_case(text: str) -> str:
    """Rule 10: acronyms stay upper case; proper nouns (gazetteer / WordNet capitalised lemma) keep title case;
    everything else is lower case."""
    toks = text.split()
    if not toks:
        return text
    whole_region = parse_region(text) is not None
    out = []
    for t in toks:
        if _is_acronym_token(t):
            out.append(t)
        elif whole_region or t.lower() in CITIES or t.lower() in {s for st in STATES for s in [st]}:
            out.append(t[:1].upper() + t[1:].lower())
        elif t[:1].isupper() and wnx.is_proper_noun(t):
            out.append(t[:1].upper() + t[1:])
        else:
            out.append(t.lower())
    s = " ".join(out)
    if len(toks) > 1 and wnx.get_wordnet() is not None:
        # a person / place name: the whole name is a capitalised lemma, or some token is a proper noun and
        # the others are unknown to WordNet ("Subhas Chandra Bose")
        proper = [wnx.is_proper_noun(t) for t in toks]
        unknown = [not wnx.is_lemma(t) for t in toks]
        if wnx.is_proper_noun(text) or (any(proper) and all(p or u for p, u in zip(proper, unknown))):
            s = " ".join(w if _is_acronym_token(w) else w[:1].upper() + w[1:] for w in s.split())
    return s


def _classify_tokens(text: str, pn: ParsedName, explicit_or: bool) -> None:
    """Rules 8-9: word / multiword / synonym_list / phrase."""
    toks = text.split()
    if explicit_or and re.search(r"\s(?:or|/)\s", f" {text} ", re.I):
        pieces = [p.strip() for p in re.split(r"\s+(?:or|/)\s+", text, flags=re.I) if p.strip()]
        if len(pieces) >= 2:
            pn.kind = "synonym_list"
            pn.synonyms = [_display_case(p) for p in pieces]
            pn.headword = pn.synonyms[0]
            return
    if len(toks) <= 1:
        pn.kind = "word"
        pn.headword = _display_case(text)
        return
    pieces = wnx.synonym_pieces(toks)
    if pieces:
        pn.kind = "synonym_list"
        pn.synonyms = [_display_case(p) for p in pieces]
        pn.headword = pn.synonyms[0]
        return
    if wnx.get_wordnet() is None:
        pn.flags.append("no_wordnet")
    pn.kind = "phrase" if len(toks) >= 5 else "multiword"
    pn.headword = _display_case(text)


def _handle_paren(content: str, pn: ParsedName, base_words: list[str], acronyms: list[str]) -> None:
    c = clean_text(content)
    low = c.lower()
    if not c:
        return
    if _RE_EXPLAIN.match(c):
        pn.kind = "explanation"
        if low.startswith("sign"):
            pn.flags.append("sign_and_explanation")
        if "explaination" in low or "explantion" in low:
            pn.flags.append("typo_fixed")
        return
    if re.fullmatch(r"expla\w*", low):  # other misspellings ("Explantion")
        pn.kind = "explanation"
        pn.flags.append("typo_fixed")
        return
    if _RE_EXAMPLE.match(c):
        pn.kind = "example"
        return
    m = _RE_VARIANT.match(c)
    if m and (m.group(1) or low.startswith("sign")):
        pn.sign_no = int(m.group(1)) if m.group(1) else 1
        if not low.startswith("sign"):
            pn.flags.append("bare_number_marker")
        return
    if low == "s":
        pn.flags.append("plural_marker")
        return
    region = parse_region(c)
    if region:
        pn.region = region
        return
    acr = c.replace(".", "").replace("-", "")
    if _is_acronym_token(c) and acr in (_initials(base_words), "".join(w[0] for w in base_words if w).upper()):
        acronyms.append(c)
        pn.flags.append("acronym")
        return
    dom = DOMAIN_ALIASES.get(low, low)
    if pn.domain is None:
        pn.domain = dom
    else:
        pn.qualifiers.append(dom)


# --------------------------------------------------------------------------------------------- parsers

def parse_islrtc_name(rel_path: str) -> ParsedName:
    """Parse an ISLRTC dictionary path such as ``P/Pencil_(Sign_3).mp4`` or ``Numbers/12_Twelve.mp4``."""
    pp = PurePosixPath(rel_path)
    folder = pp.parts[0] if len(pp.parts) > 1 else ""
    stem = pp.name
    if "." in stem and stem.rsplit(".", 1)[1].lower() in {"mp4", "mov", "avi", "mkv", "webm", "m4v"}:
        stem = stem.rsplit(".", 1)[0]
    pn = ParsedName(headword="", original=rel_path)
    text = clean_text(stem)
    if _RE_ENGLISH.search(text):  # rule 3
        pn.flags.append("english_marker")
        text = _RE_ENGLISH.sub("", text).strip()
    # rule 2/4/5/6: parenthesised groups
    groups = _RE_PAREN.findall(text)
    base = clean_text(_RE_PAREN.sub(" ", text))
    base = _RE_ENGLISH.sub("", base).strip()
    base_words = base.split()
    extra_syn: list[str] = []
    for g in groups:
        _handle_paren(g, pn, base_words, extra_syn)
    if not base:  # name was only parentheses
        base = clean_text(" ".join(groups)) or stem
        pn.flags.append("needs_review")
    # rule 7: numbers folder
    if folder.lower() == "numbers":
        m = re.match(r"^(\d+)\s+(.+)$", base)
        if m and all(w.lower() in NUMBER_WORDS for w in m.group(2).split()):
            pn.kind = "number"
            pn.number_value = int(m.group(1))
            words = m.group(2).lower()
            pn.text = words
            pn.headword = words
            pn.synonyms = [words, m.group(1)]
            return pn
        if re.match(r"^\d+\s*-\s*\d+$", base) or base.lower().startswith("numbers"):
            pn.kind = "compilation"
    if pn.kind in ("explanation", "example", "compilation"):
        kind = pn.kind
        _classify_tokens(base, pn, explicit_or=False)
        pn.kind = kind
    else:
        _classify_tokens(base, pn, explicit_or=False)
        if len(base) == 1 and base.isalpha():
            pn.kind = "letter"
            pn.headword = base.upper()
    pn.text = base.lower()
    for a in extra_syn:
        if pn.synonyms:
            pn.synonyms.append(a)
        else:
            pn.synonyms = [pn.headword, a]
    return pn


def parse_include_word(word_dir: str) -> ParsedName:
    """INCLUDE word folder: ``63. Winter`` -> winter; ``Store or Shop`` -> synonym list; ``Second (Number)`` ->
    headword "second", sense "number"; ``Ex. Monsoon`` -> monsoon."""
    pn = ParsedName(headword="", original=word_dir)
    text = clean_text(word_dir)
    text = re.sub(r"^\s*(?:\d+|ex)\s*\.\s*", "", text, flags=re.I)
    groups = _RE_PAREN.findall(text)
    base = clean_text(_RE_PAREN.sub(" ", text))
    for g in groups:
        s = clean_text(g).lower()
        if s:
            if pn.sense is None:
                pn.sense = DOMAIN_ALIASES.get(s, s)
            else:
                pn.qualifiers.append(s)
    _classify_tokens(base, pn, explicit_or=True)
    if pn.kind == "synonym_list":
        pn.text = pn.synonyms[0].lower()
    else:
        pn.text = base.lower()
    return pn


def parse_cislr_word(word: str) -> ParsedName:
    """CISLR ``dataset.csv`` word column (format to be confirmed once access is granted): same rules as an
    ISLRTC name without folder, ``x or y`` treated as a synonym list."""
    pn = parse_islrtc_name(f"_/{word}")
    pn.original = word
    return pn


def split_synonyms(tokens: list[str]) -> list[str] | None:
    """Rule 8: partition tokens into WordNet synonyms, or None (also None without WordNet)."""
    return wnx.synonym_pieces(tokens)


def apply_overrides(pn: ParsedName, source_path: str, overrides: list[dict]) -> ParsedName:
    """Rule 11: ``overrides.yaml`` entries ``{match: <regex on the source path>, set: {field: value}}`` win."""
    for ov in overrides:
        pat = ov.get("match")
        if pat and re.search(pat, source_path):
            for k, v in (ov.get("set") or {}).items():
                if k == "flags_add":
                    pn.flags.extend(v)
                elif hasattr(pn, k):
                    setattr(pn, k, v)
            pn.flags.append("override")
    return pn


def load_overrides(path=None) -> list[dict]:
    from pathlib import Path

    import yaml

    p = Path(path) if path else Path(__file__).with_name("overrides.yaml")
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if isinstance(data, dict):
        data = data.get("overrides", [])
    return [d for d in data if isinstance(d, dict)]
