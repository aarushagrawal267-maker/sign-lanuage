"""Hindi lexicon lookup (docs/ISL_DESIGN.md section 8.1).

The tracked ``lexicon_hi.jsonl`` (one row per concept) is generated offline by ``python -m app.isl.cli hindi``
(``app/isl/text/hindi_build.py``). Row fields::

    {"concept_id": "isl.pencil", "english": "pencil", "hindi": "पेंसिल", "hi_alt": [],
     "source": "manual" | "rule" | "model" | "local", "reviewed": false,
     "review": "reviewed" | "curated" | "machine" | "disputed" | "missing", "model": null | "<model id>"}

``reviewed`` is true only when a Hindi speaker confirmed the entry. ``curated`` rows come from the hand-made
core list or the number rules (careful but not yet confirmed by a native speaker); ``machine`` rows are
IndicTrans2 drafts. Local corrections live in ``data/isl/lexicon_hi.local.jsonl`` and override the base rows per
``concept_id``. Nothing is ever translated at runtime: a concept without a row has no Hindi (the UI shows
"Hindi not available yet / हिंदी उपलब्ध नहीं"). Older rows that use ``hi`` instead of ``hindi`` are accepted.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REVIEW_VALUES = ("reviewed", "curated", "machine", "disputed", "missing")
HI_MISSING = "हिंदी उपलब्ध नहीं"
EN_MISSING = "Hindi not available yet"
REVIEW_LABELS = {
    "reviewed": {"en": "checked by a Hindi speaker", "hi": "हिंदी भाषी द्वारा जाँचा गया"},
    "curated": {"en": "from the checked word list, not yet confirmed by a teacher",
                "hi": "शब्द-सूची से, शिक्षक द्वारा पुष्टि बाकी"},
    "machine": {"en": "machine translation, not yet checked", "hi": "मशीनी अनुवाद, अभी जाँचा नहीं गया"},
    "disputed": {"en": "a reviewer flagged this translation", "hi": "समीक्षक ने इस अनुवाद पर आपत्ति की"},
}


def normalise_row(r: dict) -> dict:
    """Accept both ``hindi`` (current) and ``hi`` (older rows, local corrections); fill ``reviewed``."""
    hi = r.get("hindi") or r.get("hi")
    r["hindi"] = r["hi"] = hi
    r.setdefault("hi_alt", [])
    r["reviewed"] = r.get("review") == "reviewed"
    return r


def _read(path: Path | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path is None or not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("concept_id"):
            out[r["concept_id"]] = normalise_row(r)
    return out


class HindiLexicon:
    def __init__(self, base: Path | None, local: Path | None = None):
        self.base_path, self.local_path = base, local
        self._lock = threading.Lock()
        self.base = _read(base)
        self.local = _read(local)

    def __len__(self) -> int:
        return len(set(self.base) | set(self.local))

    def get(self, concept_id: str) -> dict | None:
        r = self.local.get(concept_id) or self.base.get(concept_id)
        if r is None or not r.get("hi") or r.get("review") == "missing":
            return None
        return r

    def text(self, concept_id: str) -> tuple[str | None, str | None]:
        """(hi, review) for display; (None, None) when there is no usable Hindi."""
        r = self.get(concept_id)
        return (r["hi"], r.get("review")) if r else (None, None)

    def stats(self) -> dict:
        """Counts by source and review state (local corrections included)."""
        merged = {**self.base, **self.local}
        rows = [r for r in merged.values() if r.get("hi") and r.get("review") != "missing"]
        return {"entries": len(rows), "by_source": dict(Counter(r.get("source", "?") for r in rows)),
                "by_review": dict(Counter(r.get("review", "?") for r in rows)),
                "reviewed": sum(1 for r in rows if r.get("review") == "reviewed"), "local": len(self.local)}

    def local_rows(self) -> list[dict]:
        return [self.local[k] for k in sorted(self.local)]

    def set_local(self, concept_id: str, hi: str, hi_alt: list[str] | None = None, review: str = "reviewed",
                  reviewer: str | None = None, english: str | None = None) -> dict:
        if review not in REVIEW_VALUES:
            raise ValueError(f"review must be one of {REVIEW_VALUES}")
        if self.local_path is None:
            raise ValueError("no local lexicon file configured")
        hi = hi.strip()
        row = {"concept_id": concept_id, "english": english, "hindi": hi, "hi": hi,
               "hi_alt": [a.strip() for a in (hi_alt or []) if a.strip()], "source": "local",
               "reviewed": review == "reviewed", "review": review, "reviewer": reviewer,
               "updated": datetime.now(UTC).strftime("%Y-%m-%d")}
        with self._lock:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            with self.local_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.local[concept_id] = row
        return row
