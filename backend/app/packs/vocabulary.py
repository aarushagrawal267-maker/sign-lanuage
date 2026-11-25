"""Runtime vocabulary of a language pack: fast lookup and paged search (docs/ISL_DESIGN.md section 4.4).

ISL reads ``vocabulary.jsonl`` (one concept per line, row number = ``class_index``) as plain dicts, which is
several times faster than validating 12k pydantic models at start-up. ASL builds the same structure from the
model's class names.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

ORIGIN = {"islrtc": "ISLRTC dictionary", "include": "INCLUDE", "cislr": "CISLR", "mysigns": "My signs"}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()


def _is_devanagari(s: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in s)


class Vocabulary:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.pos = {r["concept_id"]: i for i, r in enumerate(rows)}
        self._search = [" ".join([_fold(r.get("headword", "")), *(_fold(s) for s in r.get("synonyms", []) or []),
                                  _fold((r.get("display") or {}).get("en", "")), _fold(r.get("concept_id", ""))])
                        for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        rows = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
        return cls(rows)

    @classmethod
    def from_class_names(cls, names: list[str]) -> Vocabulary:
        return cls([{"concept_id": n, "headword": n, "display": {"en": n}, "kind": "word", "in_classifier": True,
                     "indexable": False, "variants": []} for n in names])

    def get(self, concept_id: str) -> dict | None:
        i = self.pos.get(concept_id)
        return None if i is None else self.rows[i]

    def index_of(self, concept_id: str) -> int:
        return self.pos.get(concept_id, -1)

    def english(self, concept_id: str) -> str:
        r = self.get(concept_id)
        if r is None:
            return concept_id
        return (r.get("display") or {}).get("en") or r.get("headword") or concept_id

    def variant(self, concept_id: str, variant_id: str | None) -> dict | None:
        r = self.get(concept_id)
        if r is None or variant_id is None:
            return None
        return next((v for v in r.get("variants", []) if v.get("variant_id") == variant_id), None)

    def prior(self, concept_id: str) -> float:
        """Log-frequency proxy used by fusion: core (multi-signer) words 1, single words 0.5, others 0."""
        r = self.get(concept_id)
        if r is None:
            return 0.0
        if r.get("in_classifier"):
            return 1.0
        return 0.5 if r.get("kind", "word") in ("word", "number") else 0.0

    def search(self, q: str | None = None, *, category: str | None = None, source: str | None = None,
               kind: str | None = None, in_classifier: bool | None = None, has_hi=None, review: str | None = None,
               offset: int = 0, limit: int = 50, hi_lookup=None) -> tuple[int, list[tuple[int, dict]]]:
        qf = _fold(q or "")
        # A query in Devanagari searches the Hindi lexicon (main word and alternatives) instead of the English fields.
        hindi_q = bool(qf) and _is_devanagari(qf) and hi_lookup is not None
        exact, prefix, other = [], [], []
        for i, r in enumerate(self.rows):
            if category and r.get("category") != category:
                continue
            if kind and r.get("kind", "word") != kind:
                continue
            if in_classifier is not None and bool(r.get("in_classifier")) != in_classifier:
                continue
            if source and not any(v.get("source") == source for v in r.get("variants", [])):
                continue
            if review and (r.get("review") or {}).get("status", "auto") != review:
                continue
            if has_hi is not None and hi_lookup is not None and (hi_lookup(r["concept_id"]) is not None) != has_hi:
                continue
            if hindi_q:
                hi = hi_lookup(r["concept_id"])
                if not hi:
                    continue
                main = _fold(hi.get("hi") or "")
                alts = [_fold(a) for a in hi.get("hi_alt") or []]
                if main == qf or qf in alts:
                    exact.append(i)
                elif main.startswith(qf) or any(a.startswith(qf) for a in alts):
                    prefix.append(i)
                elif qf in main or any(qf in a for a in alts):
                    other.append(i)
            elif qf:
                hw = _fold(r.get("headword", ""))
                if hw == qf:
                    exact.append(i)
                elif hw.startswith(qf):
                    prefix.append(i)
                elif qf in self._search[i]:
                    other.append(i)
            else:
                other.append(i)
        hits = exact + prefix + other
        page = hits[max(0, offset): max(0, offset) + max(0, limit)]
        return len(hits), [(i, self.rows[i]) for i in page]
