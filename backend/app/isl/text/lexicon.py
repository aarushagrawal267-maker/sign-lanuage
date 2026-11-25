"""Hindi lexicon lookup (docs/ISL_DESIGN.md section 8.1).

The tracked ``lexicon_hi.jsonl`` is generated and reviewed offline (a later task); local corrections live in
``data/isl/lexicon_hi.local.jsonl`` and override the base rows per ``concept_id``. Nothing is ever translated
at runtime: a concept without a row has no Hindi (the UI shows "Hindi not available yet").
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

REVIEW_VALUES = ("reviewed", "machine", "disputed", "missing")


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
            out[r["concept_id"]] = r
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

    def set_local(self, concept_id: str, hi: str, hi_alt: list[str] | None = None, review: str = "reviewed",
                  reviewer: str | None = None) -> dict:
        if review not in REVIEW_VALUES:
            raise ValueError(f"review must be one of {REVIEW_VALUES}")
        if self.local_path is None:
            raise ValueError("no local lexicon file configured")
        row = {"concept_id": concept_id, "hi": hi.strip(), "hi_alt": [a.strip() for a in (hi_alt or []) if a.strip()],
               "source": "local", "review": review, "reviewer": reviewer,
               "updated": datetime.now(UTC).strftime("%Y-%m-%d")}
        with self._lock:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            with self.local_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.local[concept_id] = row
        return row
