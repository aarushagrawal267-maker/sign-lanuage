"""Vocabulary file schema (``models/packs/isl/vocabulary.jsonl``, one concept per line; row number = class index).

See docs/ISL_DESIGN.md section 6.1. Hindi is deliberately not stored here (it is joined from the reviewed
lexicon, section 8).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: str  # islrtc | include | cislr | mysigns
    id: str | None = None  # clip id (ISLRTC) or None
    word_dir: str | None = None  # INCLUDE "<Category>/<NN. Word>"
    licence: str
    url: str | None = None


class Variant(BaseModel):
    model_config = ConfigDict(extra="allow")

    variant_id: str  # "islrtc:P/Pencil_(Sign_2)", "include:Home/33._Pencil", "cislr:<uid>", "mysign:<uuid>"
    source: str
    sign_no: int = 1
    region: str | None = None
    kind: str = "word"  # parse kind of this source name (explanation/example/compilation are not indexed)
    index: bool = True  # used as a recognition reference (embedding index / classifier)
    full_clip: bool = False  # phrase: index the whole source clip, no first-sign trim
    clips: list[str] = Field(default_factory=list)  # clip ids (ISLRTC, CISLR)
    clips_count: int | None = None  # INCLUDE: number of videos of this word folder
    link: str | None = None  # how a non-ISLRTC variant was attached: exact | synonym | token | wordnet | new
    needs_review: bool = False
    trim: dict | None = None  # {"method", "confidence"} from the trim sidecar, when available
    flags: list[str] = Field(default_factory=list)


class Review(BaseModel):
    status: str = "auto"  # auto | needs_review | reviewed
    notes: list[str] = Field(default_factory=list)


class VocabEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    concept_id: str
    headword: str
    display: dict[str, str]
    synonyms: list[str] = Field(default_factory=list)
    sense: str | None = None
    domain: str | None = None
    kind: str = "word"
    category: str | None = None
    pos: str | None = None
    fingerspell: str | None = None
    number_value: int | None = None
    variants: list[Variant] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    see_also: list[str] = Field(default_factory=list)  # related concepts suggested for review (never merged)
    in_classifier: bool = False  # has multi-signer training data (INCLUDE) -> core closed-set class
    indexable: bool = True  # at least one variant can serve as a recognition reference
    review: Review = Field(default_factory=Review)

    def to_json(self) -> dict:
        """Compact JSON: None values and defaults are omitted (``model_validate`` restores them), except
        the identifying fields."""
        d = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        for k in ("concept_id", "headword", "display", "kind", "in_classifier", "indexable"):
            d[k] = getattr(self, k) if k != "display" else self.display
        return d
