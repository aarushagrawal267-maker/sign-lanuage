"""Hindi search in the runtime vocabulary and the Hindi-lexicon fallback of local ISL pack builds (used by the
Vocabulary page and the live results). No network, no model."""

from __future__ import annotations

import json
from pathlib import Path

from app.isl.text.lexicon import HindiLexicon
from app.packs.isl import IslPack
from app.packs.schema import PackManifest
from app.packs.vocabulary import Vocabulary

REPO = Path(__file__).resolve().parents[3]

ROWS = [
    {"concept_id": "isl.water", "headword": "water", "display": {"en": "water"}, "kind": "word"},
    {"concept_id": "isl.water-tank", "headword": "water tank", "display": {"en": "water tank"}, "kind": "word"},
    {"concept_id": "isl.hate", "headword": "hate", "display": {"en": "hate"}, "kind": "word"},
    {"concept_id": "isl.pen", "headword": "pen", "display": {"en": "pen"}, "kind": "word"},
]
LEX = [
    {"concept_id": "isl.water", "hindi": "पानी", "hi_alt": ["जल"], "source": "manual", "review": "curated"},
    {"concept_id": "isl.water-tank", "hindi": "पानी की टंकी", "source": "model", "review": "machine"},
    {"concept_id": "isl.hate", "hindi": "नफ़रत करना", "hi_alt": ["घृणा करना"], "source": "model", "review": "machine"},
]


def _lex(tmp_path: Path) -> HindiLexicon:
    p = tmp_path / "lexicon_hi.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in LEX) + "\n", encoding="utf-8")
    return HindiLexicon(p, tmp_path / "local.jsonl")


def _ids(res) -> list[str]:
    return [r["concept_id"] for _, r in res[1]]


def test_devanagari_query_searches_hindi_main_word_and_alternatives(tmp_path):
    vocab, lex = Vocabulary(ROWS), _lex(tmp_path)
    look = lex.get
    # exact main word ranks before a longer phrase that starts with it
    assert _ids(vocab.search("पानी", hi_lookup=look)) == ["isl.water", "isl.water-tank"]
    # an alternative counts as an exact match
    assert _ids(vocab.search("जल", hi_lookup=look)) == ["isl.water"]
    assert _ids(vocab.search("घृणा", hi_lookup=look)) == ["isl.hate"]
    # nukta-insensitive (नफ़रत and नफरत are both common spellings)
    assert _ids(vocab.search("नफरत", hi_lookup=look)) == ["isl.hate"]
    # substring inside a phrase
    assert _ids(vocab.search("टंकी", hi_lookup=look)) == ["isl.water-tank"]
    # a concept without Hindi never matches a Hindi query; English search is unchanged
    assert _ids(vocab.search("कलम", hi_lookup=look)) == []
    assert _ids(vocab.search("pen", hi_lookup=look)) == ["isl.pen"]
    # without a lexicon a Hindi query falls back to the English fields (and finds nothing here)
    assert vocab.search("पानी")[0] == 0


def _manifest(pack_id: str = "isl") -> PackManifest:
    return PackManifest.model_validate({
        "schema": 1, "id": pack_id, "kind": "isl-hybrid", "sign_language": "ISL", "standard": "ISLRTC",
        "display_name": {"en": "ISL"}, "output_languages": ["en", "hi"], "capabilities": ["words"],
        "profile": "commercial-safe", "vocabulary": "vocabulary.jsonl", "lexicons": {"hi": "lexicon_hi.jsonl"}})


def test_local_build_without_lexicon_falls_back_to_tracked_copy(tmp_path):
    tracked = REPO / "models" / "packs" / "isl" / "lexicon_hi.jsonl"
    pack = IslPack(_manifest("isl"), tmp_path)
    p = pack.component_paths()["lexicon_hi"]
    if tracked.exists():
        assert p == tracked.resolve()
        assert pack.lexicon().get("isl.water") is not None
    # a lexicon inside the pack folder always wins
    own = tmp_path / "lexicon_hi.jsonl"
    own.write_text(json.dumps(LEX[0], ensure_ascii=False) + "\n", encoding="utf-8")
    assert pack.component_paths()["lexicon_hi"] == own.resolve()
    # a pack id with no tracked folder keeps its own (missing) path
    other = IslPack(_manifest("isl-zz-test"), tmp_path / "x")
    assert other.component_paths()["lexicon_hi"] == (tmp_path / "x" / "lexicon_hi.jsonl").resolve()
