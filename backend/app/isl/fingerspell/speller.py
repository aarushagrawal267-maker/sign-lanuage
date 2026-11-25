"""Spelling mode: holds -> letter posteriors -> lexicon decoding (docs/ISL_DESIGN.md section 6.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.features.build import FeatureError
from app.isl.fingerspell.decoder import LexiconTrie, SpellDecoder, normalize_word, to_probs
from app.isl.fingerspell.features import letter_features_seq
from app.isl.fingerspell.holds import detect_holds
from app.isl.fingerspell.model import NOISE, LetterPredictor
from app.landmarks.raw import RawSequence


@dataclass
class SpellResult:
    letters: list[dict] = field(default_factory=list)  # [{"letter", "prob", "start", "end"}] per hold (noise dropped)
    buffer: str = ""  # best raw letter string
    suggestions: list[dict] = field(default_factory=list)  # [{"word", "concept_id", "score", "in_lexicon"}]
    holds: int = 0

    def to_json(self) -> dict:
        return {"letters": self.letters, "buffer": self.buffer, "suggestions": self.suggestions, "holds": self.holds}


class Speller:
    def __init__(self, letters: LetterPredictor, decoder: SpellDecoder, concept_of_word: dict[str, str] | None = None,
                 min_hold_ms: float = 180.0, v_max: float = 0.35):
        self.model = letters
        self.decoder = decoder
        self.concept_of_word = {normalize_word(k): v for k, v in (concept_of_word or {}).items()}
        self.min_hold_ms, self.v_max = min_hold_ms, v_max
        self.experimental = letters.n_signers < 5

    @classmethod
    def load(cls, checkpoint: Path, lexicon: Path, concept_of_word: dict[str, str] | None = None) -> Speller:
        trie = LexiconTrie.from_lines(lexicon.read_text(encoding="utf-8").splitlines()) if lexicon.exists() else LexiconTrie([])
        return cls(LetterPredictor(checkpoint), SpellDecoder(trie), concept_of_word)

    def hold_logprobs(self, raw: RawSequence) -> tuple[list, np.ndarray]:
        try:
            feats, _ = letter_features_seq(raw)
        except FeatureError:
            return [], np.zeros((0, len(self.model.labels)))
        holds = detect_holds(raw, self.min_hold_ms, self.v_max, feats=feats)
        if not holds:
            return [], np.zeros((0, len(self.model.labels)))
        return holds, self.model.log_probs(np.stack([h.mean_features for h in holds]))

    def spell(self, raw: RawSequence, k: int = 5, decode: bool = True) -> SpellResult:
        holds, lp = self.hold_logprobs(raw)
        res = SpellResult(holds=len(holds))
        keep = []
        for h, row in zip(holds, lp):
            j = int(np.argmax(row))
            if j == NOISE:
                continue
            keep.append(row)
            res.letters.append({"letter": self.model.labels[j], "prob": float(np.exp(row[j])), "start": h.start,
                                "end": h.end})
        res.buffer = "".join(x["letter"] for x in res.letters)
        if decode and len(lp):
            dec = self.decoder.decode(list(lp), k=k)
            probs = to_probs([s for _, s, _ in dec])
            res.suggestions = [{"word": w, "concept_id": self.concept_of_word.get(normalize_word(w)), "score": p,
                                "in_lexicon": inlex} for (w, _, inlex), p in zip(dec, probs)]
        return res
