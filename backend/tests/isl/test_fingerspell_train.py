"""Fingerspelling letter model from all letter sources + leave-one-signer-out E5 (synthetic data, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from app.dataset.synthetic import make_sequence
from app.isl.data.sources import Clip
from app.isl.fingerspell.model import LETTERS, NOISE
from app.isl.training import multisource as ms
from app.isl.training.train_fingerspell import collect_letter_examples, letter_data, symbol_of, train_fingerspell
from app.landmarks.raw import RawSequence

LET = {"A": 0, "B": 3, "C": 6}


def _concat(parts: list[RawSequence]) -> RawSequence:
    t, off = [], 0.0
    for p in parts:
        t.append(p.t_ms - p.t_ms[0] + off)
        off = t[-1][-1] + 1000.0 / 12
    return RawSequence(*(np.concatenate([getattr(p, k) for p in parts]) for k in ("pose", "left", "right", "face")),
                       np.concatenate(t), parts[0].width, parts[0].height)


def _spelled(word: str, signer: int, path) -> tuple[Clip, list[dict]]:
    parts, segs, i = [], [], 0
    for ch in word:
        n = 10
        if ch == " ":
            r = make_sequence(9, signer=signer, n_frames=n, fps=12.0)
        else:
            r = make_sequence(LET[ch], signer=signer, n_frames=n, fps=12.0, rng=np.random.default_rng(LET[ch]))
        parts.append(r)
        segs.append({"letter": ch.lower(), "start": i, "end": i + n})
        i += n
    raw = _concat(parts)
    raw.save(path)
    return Clip(f"islfs/{signer}/{word}", "isl_fingerspelling", word, None, None, "fingerspelling", path,
                group=f"signer_{signer}"), segs


@pytest.fixture(scope="module")
def letters(tmp_path_factory):
    d = tmp_path_factory.mktemp("fs")
    fs = [_spelled(w, s, d / f"{s}_{w.replace(' ', '_')}.npz") for s in (0, 1) for w in ("AB CA", "BC", "CAB")]
    clips = []
    for k, ch in enumerate("ABC"):
        r = make_sequence(LET[ch], signer=7, n_frames=12, fps=12.0, rng=np.random.default_rng(LET[ch]))
        p = d / f"clip_{ch}.npz"
        r.save(p)
        clips.append(Clip(f"openhands_fs/train/{ch.lower()}/{k}", "openhands_fs", ch, None, None, "letter", p,
                          group=f"openhands_fs:{k}"))
    imgs = []
    for k, g in enumerate(["b", "seven", "अं"]):
        r = make_sequence(3, signer=8, n_frames=1, fps=12.0)
        p = d / f"img{k}.npz"
        r.save(p)
        imgs.append(Clip(f"isl_static/{k}", "mendeley_static", g, None, None, None, p, group="static", quality={"n_frames": 1}))
    return collect_letter_examples(fs, clips, imgs), imgs


def test_symbol_of():
    assert symbol_of("a") == "A" and symbol_of("7") == "7" and symbol_of("Seven") == "7"
    assert symbol_of("अं") is None and symbol_of("E1") is None and symbol_of("") is None


def test_collect_examples_letters_noise_words(letters):
    ls, _ = letters
    s = ls.summary()
    assert s["per_source"].keys() == {"isl_fingerspelling", "openhands_fs", "mendeley_static"}
    assert {LETTERS[i] for i in ls.y if i != NOISE} == {"A", "B", "C", "7"}
    assert NOISE in ls.y  # space segments are noise
    assert s["skipped"] == {"image_unknown_label": 1}
    words = sorted({w["word"] for w in ls.words})
    assert words == ["AB", "BC", "CA", "CAB"]
    ab = next(w for w in ls.words if w["word"] == "AB")
    assert (ab["start"], ab["end"]) == (0, 20)


def test_train_letters_loso_and_load_in_speller(letters, tmp_path):
    from app.isl.fingerspell.decoder import LexiconTrie, SpellDecoder
    from app.isl.fingerspell.model import LetterPredictor
    from app.isl.fingerspell.speller import Speller

    ls, _ = letters
    out = tmp_path / "letters.pt"
    r = train_fingerspell(ls, out, profile="research", epochs=8, lexicon_words=["cab"], smoke=True)
    assert r["n_signers"] == 2 and set(r["letters"]) == {"A", "B", "C", "7"}
    e5 = r["e5"]
    assert e5["measurable"] and e5["signers"] == ["signer_0", "signer_1"] and e5["n"] > 0
    assert 0.0 <= e5["metrics"]["letter_top1"] <= 1.0 and "words" in e5
    lp = LetterPredictor(out)
    assert lp.n_signers == 2 and lp.meta["e5"]["measurable"]
    sp = Speller(lp, SpellDecoder(LexiconTrie([("cab", 0.0)])))
    assert sp.experimental  # fewer than 5 signers
    sp.spell(ls.words[0]["clip"].load())  # runs end to end


def test_one_signer_is_reported_as_not_measurable(letters, tmp_path):
    ls, _ = letters
    keep = [i for i, g in enumerate(ls.group) if g != "signer_1"]
    from app.isl.training.train_fingerspell import LetterSet

    one = LetterSet([ls.X[i] for i in keep], [ls.y[i] for i in keep], [ls.group[i] for i in keep],
                    [ls.source[i] for i in keep], [w for w in ls.words if w["group"] != "signer_1"])
    r = train_fingerspell(one, tmp_path / "l.pt", profile="research", epochs=2)
    assert r["e5"]["measurable"] is False and "needs at least 2" in r["e5"]["reason"]


class _Ad:
    def __init__(self, name, commercial, clips):
        self.name, self.commercial, self.licence, self._c = name, commercial, "CC-BY-4.0", clips

    def available(self):
        return True

    def clips(self, *, only_ok=True):
        yield from self._c


def test_letter_data_commercial_profile_uses_only_commercial_sources(letters, tmp_path):
    from app.isl.paths import isl_paths

    _, imgs = letters
    ms.register_adapter("mendeley_static", lambda p, v: _Ad("mendeley_static", True, imgs))
    ms.register_adapter("openhands_fs", lambda p, v: _Ad("openhands_fs", False, imgs))
    try:
        rep = {}
        ls = letter_data(isl_paths(tmp_path), "commercial-safe", report=rep)
        assert set(rep) == {"mendeley_static"} and ls.summary()["per_source"] == {"mendeley_static": 2}
        r = train_fingerspell(ls, tmp_path / "x.pt", profile="commercial-safe", epochs=2)
        assert r["e5"]["measurable"] is False
    finally:
        ms.register_adapter("mendeley_static", None)
        ms.register_adapter("openhands_fs", None)
