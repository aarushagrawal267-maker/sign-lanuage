"""Rule-based sentence builder: a sequence of recognised signs (ISL glosses) -> an **approximate** English and
Hindi sentence (docs/ISL_DESIGN.md section 8.2).

ISL word order is typically time - topic - comment, verb-final (subject-object-verb), with question words and
negation at the end. Hindi is also verb-final, so the Hindi output keeps the signed order; English needs
reordering. The builder is deterministic, small and **always labelled approximate**: it does not know tense,
gender, number or idioms, and it never invents words that were not signed (apart from the copula "is/है",
"do/does" and verb endings, each listed in ``rules``).

English rules: (1) a final question word moves to the front, the sentence ends with "?" and gets "do/does" (with
a verb) or the copula (without one); (2) a final negation becomes "do not" before the verb, or "not" after the
copula; (3) time words go first; (4) with exactly one verb at the end after two or more other words, the verb
moves after the first noun phrase (SOV -> SVO); (5) pronoun case ("ME" -> "I" as subject, "I" -> "me" elsewhere);
(6) third-person "-s" on the verb; (7) capital letter and full stop.
Hindi rules: (1) keep the signed order; (2) a final question word moves before the verb, or before the copula,
which is added when there is no verb ("YOUR NAME WHAT" -> "आपका नाम क्या है?"); (3) a final negation moves before
the verb; (4) a verb in the infinitive (-ना) gets the present-tense ending for its subject (masculine forms
assumed); (5) "?" for questions, "।" otherwise. If no rule applies, the words are joined with spaces.
"""

from __future__ import annotations

import functools
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA = Path(__file__).with_name("data")
APPROX_LABEL = {"en": "Approximate sentence (rule-based, may be wrong)",
                "hi": "अनुमानित वाक्य (नियम-आधारित, गलत हो सकता है)"}


@dataclass
class Token:
    concept_id: str
    en: str
    hi: str | None = None
    pos: str | None = None  # n | v | adj | adv | pron | det | wh | neg | time | num | conj | prep | intj | phrase
    role: str | None = None  # "wh" | "neg" | "time" | "pronoun" | "possessive" | None


@dataclass
class SentenceOut:
    text: str
    approximate: bool = True  # always True
    rules: list[str] = field(default_factory=list)
    missing_hi: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


@functools.lru_cache(maxsize=1)
def function_words() -> dict:
    import yaml

    return yaml.safe_load((DATA / "function_words.yaml").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def core_pos() -> dict[str, str]:
    from app.isl.text.hindi_build import load_core

    return {k: v.pos for k, v in load_core().items() if v.pos}


_POS_MAP = {"noun": "n", "verb": "v", "adjective": "adj", "adverb": "adv"}


def _key(en: str) -> str:
    return re.sub(r"\s+", " ", (en or "").strip().lower())


def classify(en: str, pos: str | None = None) -> tuple[str | None, str | None]:
    """(pos, role) of an English headword from the function-word table, then the core list, then ``pos``."""
    fw, k = function_words(), _key(en)
    if k in fw["question_words"]:
        return "wh", "wh"
    if k in fw["negation"]:
        return "neg", "neg"
    if k in fw["pronouns"]:
        return "pron", "pronoun"
    if k in fw["possessives"]:
        return "det", "possessive"
    if k in fw["time_words"]:
        return "time", "time"
    p = core_pos().get(k) or _POS_MAP.get(pos or "", pos)
    return p, None


def make_token(concept_id: str, en: str, hi: str | None = None, pos: str | None = None) -> Token:
    p, role = classify(en, pos)
    return Token(concept_id, en, hi, p, role)


def tokens_from_concepts(concept_ids: list[str], vocab, lexicon=None) -> list[Token]:
    """Tokens for recognised concepts. Synonym lists use their headword ("abhor despise ..." -> "abhor")."""
    out = []
    for cid in concept_ids:
        r = vocab.get(cid) if vocab is not None else None
        if r is None:
            en, pos = cid, None
        else:
            en = r.get("headword") if r.get("kind") == "synonym_list" else (
                (r.get("display") or {}).get("en") or r.get("headword") or cid)
            if r.get("domain") or r.get("sense"):
                en = r.get("headword") or en
            pos = r.get("pos")
        hi = lexicon.text(cid)[0] if lexicon is not None else None
        out.append(make_token(cid, en, hi, pos))
    return out


# ----------------------------------------------------------------------------------------------- helpers

def _split_trailing(tokens: list[Token]) -> tuple[list[Token], Token | None, Token | None, list[str]]:
    """Remove a final question word and a final negation (in either order). Returns (body, wh, neg, rules)."""
    body, wh, neg, rules = list(tokens), None, None, []
    for _ in range(2):
        if body and body[-1].role == "wh" and wh is None and len(body) > 1:
            wh = body.pop()
        elif body and body[-1].role == "neg" and neg is None and len(body) > 1:
            neg = body.pop()
    return body, wh, neg, rules


def _subject_span(body: list[Token]) -> int:
    """Length of the first noun phrase: a possessive/determiner plus the next word, or one word."""
    if not body:
        return 0
    if body[0].role == "possessive" and len(body) > 1:
        return 2
    return 1


def _agree(subject: list[Token]) -> str:
    if len(subject) == 1 and subject[0].role == "pronoun":
        return function_words()["pronouns"][_key(subject[0].en)]["agree"]
    return "3sg"


def _third_person(verb: str) -> str:
    words = verb.split(" ")
    w = words[0]
    if w == "have":
        w = "has"
    elif w == "be":
        w = "is"
    elif re.search(r"(s|sh|ch|x|z|o)$", w):
        w += "es"
    elif re.search(r"[^aeiou]y$", w):
        w = w[:-1] + "ies"
    else:
        w += "s"
    return " ".join([w, *words[1:]])


def _reorder_subject(rest: list[Token], wh: Token | None, rules: list[str]) -> list[Token]:
    """Topic-first signing ("WATER I WANT"): put a later pronoun subject first. "YOU NAME WHAT" (pronoun + noun,
    no verb, a question): the pronoun is read as a possessive ("your name")."""
    rest = list(rest)
    if len(rest) >= 2 and rest[0].role not in ("pronoun", "possessive"):
        j = next((i for i, t in enumerate(rest) if t.role == "pronoun"), None)
        if j is not None and j > 0:
            rest.insert(0, rest.pop(j))
            rules.append("topic first: moved the pronoun subject to the front")
    if (wh is not None and len(rest) == 2 and rest[0].role == "pronoun" and rest[1].pos == "n"
            and _key(rest[0].en) in _POSSESSIVE_OF):
        p = rest[0]
        rest[0] = Token(p.concept_id, _POSSESSIVE_OF[_key(p.en)], None, "det", "possessive")
        rules.append(f'read "{p.en.upper()} {rest[1].en.upper()}" as a possessive')
    return rest


_POSSESSIVE_OF = {"you": "your", "i": "my", "me": "my", "he": "his", "she": "her", "we": "our", "they": "their"}


def _greeting_like(tokens: list[Token]) -> bool:
    return any(t.pos in ("intj", "phrase") for t in tokens)


# ----------------------------------------------------------------------------------------------- English

def _en_word(t: Token, subject: bool) -> str:
    fw, k = function_words(), _key(t.en)
    if t.role == "pronoun":
        p = fw["pronouns"][k]
        return p["en"] if subject else p["en_obj"]
    if t.role == "possessive":
        return fw["possessives"][k]["en"]
    if t.role == "time":
        return fw["time_words"][k]["en"]
    if t.role in ("wh", "neg"):
        return (fw["question_words"] if t.role == "wh" else fw["negation"])[k]["en"]
    return t.en


def build_english(tokens: list[Token]) -> SentenceOut:
    if not tokens:
        return SentenceOut("", True, [], [])
    rules: list[str] = []
    body, wh, neg, _ = _split_trailing(tokens)
    if wh is not None:
        rules.append("moved the question word to the front")
    times = [t for t in body if t.role == "time"]
    rest = [t for t in body if t.role != "time"]
    if times and body[: len(times)] != times:
        rules.append("moved time words to the front")
    if _greeting_like(tokens) or (len(rest) <= 1 and wh is None and neg is None):
        words = [_en_word(t, i == 0) for i, t in enumerate(times + rest)]
        if wh is not None:
            words.insert(0, _en_word(wh, False))
        if neg is not None:
            words.append(_en_word(neg, False))
        return _finish_en(words, wh is not None, rules)
    rest = _reorder_subject(rest, wh, rules)
    verbs = [i for i, t in enumerate(rest) if t.pos == "v"]
    n_subj = _subject_span(rest)
    if len(verbs) == 1 and verbs[0] == len(rest) - 1 and len(rest) - 1 >= 2 and n_subj < len(rest) - 1:
        v = rest.pop()
        rest.insert(n_subj, v)
        rules.append("moved the verb after the subject (SOV to SVO)")
    subject = rest[:n_subj]
    agree = _agree(subject)
    subj_words = [_en_word(t, True) for t in subject]
    if subject and subject[0].role == "pronoun" and _key(subject[0].en) == "me":
        rules.append("pronoun case: ME as subject -> I")
    others = rest[n_subj:]
    has_verb = any(t.pos == "v" for t in others)
    copula = function_words()["copula"]["en"][agree]
    if has_verb:
        vi = next(i for i, t in enumerate(others) if t.pos == "v")
        pre = [_en_word(t, False) for t in others[:vi]]
        verb = others[vi].en
        post = [_en_word(t, False) for t in others[vi + 1:]]
        if (post and _key(verb) in function_words().get("motion_verbs", []) and others[vi + 1].pos == "n"
                and _key(others[vi + 1].en) != "home"):
            post.insert(0, "to")
            rules.append('added "to" after a motion verb')
        aux = "does" if agree == "3sg" else "do"
        if wh is not None:
            words = [_en_word(wh, False), aux, *subj_words, *pre]
            words += (["not"] if neg is not None else []) + [verb, *post]
            rules.append(f'added "{aux}" for the question')
        elif neg is not None:
            words = [*subj_words, *pre, aux, "not", verb, *post]
            rules.append(f'negation: "{aux} not" before the verb')
        else:
            words = [*subj_words, *pre, _third_person(verb) if agree == "3sg" else verb, *post]
            if agree == "3sg":
                rules.append("third-person -s on the verb")
    else:
        rest_words = [_en_word(t, False) for t in others]
        if wh is not None:
            words = [_en_word(wh, False), copula, *subj_words, *(["not"] if neg is not None else []), *rest_words]
        else:
            words = [*subj_words, copula, *(["not"] if neg is not None else []), *rest_words]
            if neg is not None:
                rules.append('negation: "not" after the verb "be"')
        rules.append(f'added the verb "be" ("{copula}")')
    words = [_en_word(t, False) for t in times] + words
    return _finish_en(words, wh is not None, rules)


def _finish_en(words: list[str], question: bool, rules: list[str]) -> SentenceOut:
    words = [w for w in words if w]
    text = " ".join(words).strip()
    if text:
        text = text[0].upper() + text[1:]
        text = re.sub(r"\bi\b", "I", text)
        text += "?" if question else "."
    return SentenceOut(text, True, rules, [])


# ------------------------------------------------------------------------------------------------- Hindi

def _hi_word(t: Token, subject: bool, missing: list[str]) -> str:
    fw, k = function_words(), _key(t.en)
    if t.role == "pronoun":
        p = fw["pronouns"][k]
        return p["hi"] if subject else p["hi_obj"]
    if t.role == "possessive" and t.hi is None:
        return fw["possessives"][k]["hi"]
    if t.role in ("wh", "neg", "time") and t.hi is None:
        table = {"wh": "question_words", "neg": "negation", "time": "time_words"}[t.role]
        return fw[table][k]["hi"]
    if t.hi:
        return t.hi
    missing.append(t.concept_id)
    return f"[{t.en}]"


def build_hindi(tokens: list[Token]) -> SentenceOut:
    if not tokens:
        return SentenceOut("", True, [], [])
    rules: list[str] = []
    missing: list[str] = []
    body, wh, neg, _ = _split_trailing(tokens)
    times = [t for t in body if t.role == "time"]
    rest = [t for t in body if t.role != "time"]
    question = wh is not None
    if _greeting_like(tokens) or (len(rest) <= 1 and wh is None and neg is None):
        words = [_hi_word(t, i == 0, missing) for i, t in enumerate(times + rest)]
        if neg is not None:
            words.append(_hi_word(neg, False, missing))
        if wh is not None:
            words.append(_hi_word(wh, False, missing))
        return _finish_hi(words, question, rules, missing)
    rest = _reorder_subject(rest, wh, rules)
    n_subj = _subject_span(rest)
    subject = rest[:n_subj]
    agree = _agree(subject)
    fw = function_words()
    female = bool(subject) and _key(subject[-1].en) in fw.get("female_subjects", [])
    words_t = [_hi_word(t, i == 0, missing) for i, t in enumerate(times)]
    words = [_hi_word(t, True, missing) for t in subject]
    others = rest[n_subj:]
    last_verb = others[-1] if others and others[-1].pos == "v" else None
    mid = [_hi_word(t, False, missing) for t in (others[:-1] if last_verb is not None else others)]
    words = words_t + words + mid
    wh_w = _hi_word(wh, False, missing) if wh is not None else None
    neg_w = _hi_word(neg, False, missing) if neg is not None else None
    if last_verb is not None:
        vhi = _hi_word(last_verb, False, missing)
        if wh_w:
            words.append(wh_w)
            rules.append("moved the question word before the verb")
        if neg_w:
            words.append(neg_w)
            rules.append("moved the negation before the verb")
        if last_verb.hi and vhi.endswith("ना") and len(vhi) > 2:
            ending, aux = fw["hi_present"][agree]
            if female:
                ending = "ती"
            verb = vhi[:-2] + ending
            words += [verb] if neg_w else [verb, aux]
            rules.append("verb in the present tense (feminine form, from the subject)" if female
                         else "verb in the present tense (masculine form assumed)")
        else:
            words.append(vhi)
    else:
        if wh_w:
            words.append(wh_w)
        if neg_w:
            words.append(neg_w)
            rules.append("moved the negation before the verb")
        words.append(fw["copula"]["hi"][agree])
        rules.append(f'added the verb "होना" ("{fw["copula"]["hi"][agree]}")')
    return _finish_hi(words, question, rules, missing)


def _finish_hi(words: list[str], question: bool, rules: list[str], missing: list[str]) -> SentenceOut:
    text = " ".join(w for w in words if w).strip()
    if text:
        text += "?" if question else "।"
    return SentenceOut(text, True, rules, list(dict.fromkeys(missing)))


def build_sentence(tokens: list[Token], lang: str) -> SentenceOut:
    if lang == "en":
        return build_english(tokens)
    if lang == "hi":
        return build_hindi(tokens)
    raise ValueError("lang must be 'en' or 'hi'")


def build_both(tokens: list[Token]) -> dict:
    """API payload: both sentences, the words used and the approximate label."""
    en, hi = build_english(tokens), build_hindi(tokens)
    return {"approximate": True, "label": APPROX_LABEL,
            "en": en.to_json(), "hi": hi.to_json(),
            "words": [{"concept_id": t.concept_id, "en": t.en, "hi": t.hi, "pos": t.pos, "role": t.role}
                      for t in tokens]}
