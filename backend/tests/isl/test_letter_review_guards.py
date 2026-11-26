"""Letter model review guards (docs/ISL_DESIGN.md 15.13): no example twice in one epoch; a letter model of another
profile (or trained on letter sources outside the profile) is refused by build-pack (the app's pack loader:
``test_pipeline_and_api.test_pack_loader_refuses_a_letter_model_of_another_profile``)."""

from __future__ import annotations

import numpy as np
import pytest

from app.isl.fingerspell.model import LETTERS, LetterClassifier
from app.isl.training.pipeline import check_letters_profile
from app.isl.training.train_fingerspell import BalancedEpochs, _save


def test_balanced_epochs_never_repeat_an_example_within_an_epoch():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 6000)
    sub = np.array(["a", "b", "c"], dtype=object)[rng.integers(0, 3, 6000)]
    plan = BalancedEpochs(y, sub, 700, np.random.default_rng(0))  # cap < class size: subgroups wrap every few epochs
    for _ in range(30):
        e = plan.epoch()
        assert len(e) == plan.size and len(set(e.tolist())) == len(e)
    assert plan.seen.all()


def _letters(path, profile, source, per_source=None):
    _save(LetterClassifier(), path, {"kind": "isl-letters", "labels": LETTERS, "temperature": 1.0, "profile": profile,
                                     "source": source, "data": {"per_source": per_source or {}}, "n_signers": 1})


def test_check_letters_profile(tmp_path):
    p = tmp_path / "letters.pt"
    _letters(p, "commercial-safe", "mendeley_static", {"mendeley_static": 5})
    check_letters_profile(p, "commercial-safe")
    with pytest.raises(ValueError, match="trained for profile"):
        check_letters_profile(p, "research")
    _letters(p, "research", "hf_hemg_alphabet,mendeley_static")
    with pytest.raises(ValueError, match="trained for profile"):
        check_letters_profile(p, "commercial-safe")
    check_letters_profile(p, "research")
    # profile field forged or missing, but the recorded sources give it away
    _letters(p, "commercial-safe", "mendeley_static", {"mendeley_static": 5, "isl_fingerspelling": 9})
    with pytest.raises(ValueError, match="outside profile"):
        check_letters_profile(p, "commercial-safe")
    _letters(p, None, "clip1.npz,clip2.npz")  # legacy train-letters checkpoint (own recordings): accepted
    check_letters_profile(p, "commercial-safe")
