"""Build the tracked Hindi lexicon (docs/ISL_DESIGN.md section 8.1): ``python -m app.isl.cli hindi``.

Sources, in order of precedence for each concept:

1. rows already in the tracked file whose ``review`` is ``reviewed`` or ``disputed`` (a person decided; never
   overwritten by a rebuild);
2. the hand-curated core list ``data/hindi_core.tsv`` (about 900 everyday words; ``source: manual``,
   ``review: curated``) - matched by concept id, by plain headword, or through a synonym list;
3. numbers from a rule table (``source: rule``);
4. IndicTrans2 machine translation (``source: model``, ``review: machine``), read from a resumable cache that
   ``--translate`` fills offline on CPU;
5. nothing: the concept has no row and the app shows "Hindi not available yet / हिंदी उपलब्ध नहीं".

``reviewed`` is true only for ``review == "reviewed"`` (a Hindi speaker confirmed the entry in the app or a
maintainer merged such a confirmation). Curated and rule rows are careful but unreviewed; model rows are drafts.

The input given to the model carries the context needed to pick the right sense: verbs are framed as
"to <verb>" (the output "<x>ने के लिए" is turned into the infinitive "<x>ना"), nouns as "a <noun>" (a leading
"एक" is dropped), a domain or sense as "<word> (<domain>)" (the echoed parenthesis is dropped), and every member
of a synonym list is translated separately and the most frequent result wins (the others become ``hi_alt``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.isl.text.numbers_hi import hindi_number

CORE_TSV = Path(__file__).with_name("data") / "hindi_core.tsv"
MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
KEEP_REVIEWS = ("reviewed", "disputed")
POS_MAP = {"noun": "n", "verb": "v", "adjective": "adj", "adverb": "adv"}


@dataclass
class CoreEntry:
    key: str
    hindi: str
    alt: list[str] = field(default_factory=list)
    pos: str | None = None
    group: str | None = None


def load_core(path: Path = CORE_TSV) -> dict[str, CoreEntry]:
    out: dict[str, CoreEntry] = {}
    header = True
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        cols = ln.split("\t")
        if header:
            header = False
            if cols[0] == "key":
                continue
        cols += [""] * (5 - len(cols))
        key, hi, alt, pos, group = (c.strip() for c in cols[:5])
        if not key or not hi:
            continue
        key = key if key.startswith("isl.") else key.lower()
        out[key] = CoreEntry(key, unicodedata.normalize("NFC", hi),
                             [unicodedata.normalize("NFC", a.strip()) for a in alt.split("|") if a.strip()],
                             pos or None, group or None)
    return out


# ------------------------------------------------------------------------------------- model input/output

def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def frame(text: str, pos: str | None, context: str | None = None) -> str:
    """English text given to the model for one term."""
    t = text.strip()
    single = " " not in t
    if pos == "verb":
        t = f"to {t}"
    elif pos == "noun" and single and t.islower():
        t = f"{_article(t)} {t}"
    if context:
        t = f"{t} ({context})"
    return t


def mt_inputs(row: dict) -> list[str]:
    """Model inputs for one vocabulary row (several for a synonym list)."""
    pos = row.get("pos")
    context = row.get("domain") or row.get("sense")
    kind = row.get("kind", "word")
    if kind == "synonym_list" and row.get("synonyms"):
        return [frame(s, pos, context) for s in row["synonyms"][:5]]
    en = (row.get("display") or {}).get("en") or row.get("headword") or ""
    if context and row.get("headword"):
        en = row["headword"]  # the display text already carries "(context)"
    if kind == "phrase":
        return [en[:1].upper() + en[1:]]
    return [frame(en, pos if kind == "word" else None, context)]


_CUT = re.compile(r"\s*(?:।|\.\.|\|)")
_PAREN = re.compile(r"\s*[\(\[（].*$")
_TO_INF = re.compile(r"^(.*\S)ने के लिए$")
_ACRONYM = re.compile(r"[\u0900-\u097F]+(?:\.\s*[\u0900-\u097F]+)+\.?")
_COPULA = re.compile(r"\s+(?:है|हैं|हूँ|हो)$")


def clean_output(inp: str, out: str) -> str:
    """Post-process one model output (see module docstring). Returns "" when nothing usable is left."""
    t = unicodedata.normalize("NFC", out or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = _CUT.split(t, maxsplit=1)[0].strip()
    if inp.rstrip().endswith(")"):
        t = _PAREN.sub("", t).strip()
    t = t.rstrip(".!,;: ").strip()
    if re.fullmatch(r"[A-Z0-9]{2,8}", inp.split(" (")[0]) and _ACRONYM.fullmatch(t):
        t = re.sub(r"[.\s]", "", t)  # acronyms: "ए. टी. एम." -> "एटीएम"
    if " - " not in inp:
        t = re.sub(r"\s+-\s+", "-", t)  # "टी - शर्ट" -> "टी-शर्ट"
    if inp.startswith("to "):
        m = _TO_INF.match(t)
        if m:
            t = m.group(1) + "ना"
    if re.match(r"^an? \S", inp):
        t = re.sub(r"^(?:एक|ए|एन)\s+", "", t)  # the article, translated ("एक") or transliterated ("ए")
    t = re.sub(r"(\S+)(?: \1){2,}", r"\1", t)  # "खुश खुश खुश खुश" -> "खुश" (keeps one reduplication)
    if " " not in inp.split(" (")[0].removeprefix("to ").removeprefix("a ").removeprefix("an "):
        t = _COPULA.sub("", t).strip()
    if not t or not re.search(r"[ऀ-ॿ0-9]", t):
        return ""
    return t


def choose(outputs: list[str]) -> tuple[str, list[str]]:
    """Most frequent cleaned output (ties: the first one); the other distinct outputs become alternatives."""
    vals = [o for o in outputs if o]
    if not vals:
        return "", []
    counts = Counter(vals)
    best = max(vals, key=lambda v: (counts[v], -vals.index(v)))
    alts = []
    for v in vals:
        if v != best and v not in alts:
            alts.append(v)
    return best, alts[:3]


# ------------------------------------------------------------------------------------------------ cache

def read_cache(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("model") == MODEL_ID and "input" in r:
            out[r["input"]] = r.get("out", "")
    return out


def translate_missing(inputs: list[str], cache_path: Path, translator, batch: int = 24, limit: int | None = None,
                      log=print) -> int:
    """Translate the inputs that are not cached yet (shortest first, so batches pad little). Appends to the
    cache after every batch, so an interrupted run resumes where it stopped. Returns how many were translated."""
    cache = read_cache(cache_path)
    todo = sorted({i for i in inputs if i and i not in cache}, key=lambda s: (len(s), s))
    if limit is not None:
        todo = todo[:limit]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with cache_path.open("a", encoding="utf-8") as fh:
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            outs = translator.translate(chunk)
            for i, o in zip(chunk, outs, strict=True):
                fh.write(json.dumps({"input": i, "model": MODEL_ID, "out": o}, ensure_ascii=False) + "\n")
            fh.flush()
            done += len(chunk)
            if (s // batch) % 20 == 0:
                log(f"translated {done}/{len(todo)}")
    return done


# ------------------------------------------------------------------------------------------------ build

def _row(concept_id: str, english: str, hindi: str, alt: list[str], source: str, review: str,
         model: str | None = None) -> dict:
    alt = [a for a in dict.fromkeys(alt) if a and a != hindi][:4]
    return {"concept_id": concept_id, "english": english, "hindi": hindi, "hi_alt": alt, "source": source,
            "reviewed": review == "reviewed", "review": review, "model": model}


def _core_for(row: dict, core: dict[str, CoreEntry]) -> CoreEntry | None:
    cid = row["concept_id"]
    if cid in core:
        return core[cid]
    if row.get("domain") or row.get("sense") or "~" in cid:
        return None
    hw = (row.get("headword") or "").lower()
    if hw in core:
        return core[hw]
    syn = [s.lower() for s in (row.get("synonyms") or [])]
    if row.get("kind") == "synonym_list":
        for s in syn:
            if s in core:
                return core[s]
    return None


def read_existing(path: Path | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path is None or not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("concept_id"):
            out[r["concept_id"]] = r
    return out


def build_rows(vocab_rows: list[dict], core: dict[str, CoreEntry], cache: dict[str, str],
               existing: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
    """Lexicon rows (sorted by concept id) and a summary of where each row came from."""
    existing = existing or {}
    out: list[dict] = []
    stats: Counter = Counter()
    for r in vocab_rows:
        cid = r["concept_id"]
        en = (r.get("display") or {}).get("en") or r.get("headword") or cid
        old = existing.get(cid)
        if old and old.get("review") in KEEP_REVIEWS and (old.get("hindi") or old.get("hi")):
            keep = dict(old)
            keep.setdefault("hindi", keep.get("hi"))
            keep.pop("hi", None)
            keep["english"] = en
            keep["reviewed"] = keep.get("review") == "reviewed"
            out.append(keep)
            stats["kept_" + old["review"]] += 1
            continue
        c = _core_for(r, core)
        if c is not None:
            out.append(_row(cid, en, c.hindi, c.alt, "manual", "curated"))
            stats["manual"] += 1
            continue
        if r.get("kind") == "number" and isinstance(r.get("number_value"), int):
            out.append(_row(cid, en, hindi_number(r["number_value"]), [str(r["number_value"])], "rule", "curated"))
            stats["rule"] += 1
            continue
        ins = mt_inputs(r)
        if all(i in cache for i in ins):
            best, alts = choose([clean_output(i, cache[i]) for i in ins])
            if best:
                out.append(_row(cid, en, best, alts, "model", "machine", MODEL_ID))
                stats["model"] += 1
                continue
            stats["model_empty"] += 1
            continue
        stats["missing"] += 1
    out.sort(key=lambda x: x["concept_id"])
    stats["rows"] = len(out)
    stats["concepts"] = len(vocab_rows)
    return out, dict(stats)


def asl_vocab_rows(class_names: list[str]) -> list[dict]:
    """WLASL100 glosses as vocabulary-like rows (concept id = gloss, as the ASL pack uses)."""
    return [{"concept_id": g, "headword": g, "display": {"en": g}, "kind": "word"} for g in class_names]


def build_asl_rows(class_names: list[str], core: dict[str, CoreEntry], cache: dict[str, str],
                   existing: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
    rows = asl_vocab_rows(class_names)
    # the ASL concept ids are bare glosses, so match the core list by headword
    core_by_gloss = {k: v for k, v in core.items() if not k.startswith("isl.")}
    return build_rows(rows, core_by_gloss, cache, existing)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def merge_local(base: Path, local: Path) -> int:
    """Fold local corrections (``data/isl/lexicon_hi.local.jsonl``) into the tracked file. Returns how many rows
    changed. Used by a maintainer to contribute reviewed entries back."""
    rows = read_existing(base)
    loc = read_existing(local)
    n = 0
    for cid, r in loc.items():
        if cid not in rows or not (r.get("hi") or r.get("hindi")):
            continue
        cur = rows[cid]
        cur["hindi"] = r.get("hindi") or r.get("hi")
        cur["hi_alt"] = r.get("hi_alt") or cur.get("hi_alt") or []
        cur["review"] = r.get("review", "reviewed")
        cur["reviewed"] = cur["review"] == "reviewed"
        cur["source"] = "manual"
        if r.get("reviewer"):
            cur["reviewer"] = r["reviewer"]
        n += 1
    write_rows(base, [rows[k] for k in sorted(rows)])
    return n
