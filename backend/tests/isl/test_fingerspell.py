"""Fingerspelling: lexicon decoding, hold detection, alignment, prototypes, letter model training."""

from __future__ import annotations

import numpy as np

from app.isl.fingerspell.decoder import LexiconTrie, SpellDecoder, normalize_word
from app.isl.fingerspell.features import LETTER_DIM, letter_features_seq
from app.isl.fingerspell.holds import detect_holds
from app.isl.fingerspell.lexicon import build_spelling_lexicon, place_names
from app.isl.fingerspell.model import LETTERS, NOISE, LetterPredictor
from app.isl.fingerspell.prototypes import AlphabetStore, PrototypeLetters, hold_features
from app.isl.training.train_letters import align_in_order, examples_from_sequence, train_letter_model
from tests.isl.helpers import make_trim_sequence


def _lp(letter_probs: dict[str, float], floor: float = 1e-4) -> np.ndarray:
    p = np.full(len(LETTERS), floor)
    for k, v in letter_probs.items():
        p[LETTERS.index(k)] = v
    p /= p.sum()
    return np.log(p)


def _word(w: str, conf: float = 0.9) -> list[np.ndarray]:
    return [_lp({ch: conf}) for ch in w]


def test_trie_and_normalisation():
    t = LexiconTrie([("Indore", 0.0), ("Bhopal", -1.0), ("in", -2.0), ("Madhya Pradesh", -0.5)])
    assert len(t) == 4 and "INDORE" in t and "madhyapradesh" in t and "IND" not in t
    assert normalize_word("Madhya-Pradesh 2") == "MADHYAPRADESH"


def test_decoder_exact_insertion_deletion_and_raw():
    t = LexiconTrie([("Indore", 0.0), ("Indoor", 0.0), ("Bhopal", 0.0)])
    d = SpellDecoder(t)
    assert d.decode(_word("INDORE"), k=3)[0][0] == "Indore"
    # a spurious extra hold (looks like noise) is skipped
    obs = _word("INDO") + [_lp({"noise": 0.9})] + _word("RE")
    assert d.decode(obs, k=3)[0][0] == "Indore"
    # a missed letter (no 'H' hold) is recovered by a deletion
    assert d.decode(_word("BOPAL"), k=3)[0][0] == "Bhopal"
    # an unknown name is still returned as the raw spelled string
    res = d.decode(_word("RAVI"), k=5)
    assert any(w == "RAVI" and not inlex for w, _, inlex in res)


def test_spelling_lexicon_has_vocab_and_places():
    rows = build_spelling_lexicon([{"concept_id": "isl.pencil", "headword": "pencil", "kind": "word"},
                                   {"concept_id": "isl.x", "headword": "A penny saved is a penny earned", "kind": "phrase"}])
    words = dict(rows)
    assert "pencil" in words and "Indore" in words and "Bhopal" in words
    assert not any("penny" in w for w in words)
    assert "Madhya Pradesh" in place_names()


def test_letter_features_and_holds():
    raw = make_trim_sequence([("rest", 4), ("hold", 8), ("sign", 8), ("hold", 8), ("rest", 4)], fps=12.0)
    f, t = letter_features_seq(raw)
    assert f.shape == (len(raw), LETTER_DIM) and np.isfinite(f).all()
    holds = detect_holds(raw, min_ms=300, v_max=0.35, feats=f)
    assert len(holds) == 2 and holds[0].start >= 3 and holds[1].start > holds[0].end
    assert holds[0].mean_features.shape == (LETTER_DIM,)


def test_align_in_order_keeps_longest_holds():
    from app.isl.fingerspell.holds import Hold

    holds = [Hold(0, 5), Hold(6, 7), Hold(8, 14), Hold(15, 20)]
    assert align_in_order(holds, 3) == [0, 2, 3]
    assert align_in_order(holds[:2], 3) is None


def test_prototypes_recognise_calibrated_letters(tmp_path):
    rng = np.random.default_rng(0)
    centres = {ch: rng.normal(0, 1, LETTER_DIM) for ch in "ABCDEF"}
    store = AlphabetStore(tmp_path)
    for ch, c in centres.items():
        store.add(ch, (c + rng.normal(0, 0.05, LETTER_DIM)).astype(np.float32))
    assert store.counts() == {ch: 1 for ch in "ABCDEF"}
    model = store.model()
    assert isinstance(model, PrototypeLetters)
    lp = model.log_probs(np.stack([centres["C"], centres["F"], rng.normal(0, 5, LETTER_DIM)]))
    assert LETTERS[int(lp[0].argmax())] == "C" and LETTERS[int(lp[1].argmax())] == "F"
    assert int(lp[2].argmax()) == NOISE  # far from every prototype -> noise
    store.reset()
    assert store.model() is None


def test_hold_features_from_a_recording():
    raw = make_trim_sequence([("rest", 3), ("hold", 10), ("rest", 3)], fps=12.0)
    f, ms = hold_features(raw)
    assert f.shape == (LETTER_DIM,) and ms > 0


def test_train_letter_model_from_in_order_recording(tmp_path):
    segs = [("rest", 3)]
    for _ in range(3):
        segs += [("hold", 6), ("sign", 5)]
    raw = make_trim_sequence(segs + [("rest", 3)], fps=12.0)
    X, y, noise = examples_from_sequence(raw, ["A", "B", "C"])
    assert len(X) > 0 and set(y) <= {0, 1, 2}
    if len(set(y)) < 2:  # identical synthetic holds may align fewer letters; fabricate a second class
        y = [0] * (len(y) // 2) + [1] * (len(y) - len(y) // 2)
    res = train_letter_model(X, y, noise, tmp_path / "letters.pt", epochs=3, smoke=True)
    lp = LetterPredictor(tmp_path / "letters.pt")
    assert lp.log_probs(np.stack(X[:2])).shape == (2, len(LETTERS)) and res["examples"] > 0
