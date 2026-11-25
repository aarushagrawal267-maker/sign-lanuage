"""Data profiles: which sources a model build may use (docs/ISL_DESIGN.md sections 10.1 and 15.2).

This module is the **single source of truth** for the source lists. ``app.isl.training.multisource`` re-exports
``WORD_SOURCES`` / ``PRETRAIN_SOURCES`` / ``LETTER_SOURCES`` from here, and :class:`DataProfile` is derived from the
same tuples, so the phase-1 commands (``profiles``, ``load_profile_clips``) and the phase-2 training can no longer
drift apart.

``commercial-safe`` excludes non-commercial data (iSign, CISLR: its README says CC BY-NC 4.0, WSLP, GloSL, ISL500, ...).
It still uses the ISLRTC dictionary, whose publisher forbids resale and profiteering (ISLRTC FAQ Q.23): fine for a
free, non-profit app with acknowledgement, but a paid release needs ISLRTC's permission (docs/DATASETS_ISL.md). The
licence rule is enforced twice: these allow-lists **and** each adapter's own ``commercial`` flag
(``multisource.check_licence``). The profile name is recorded in every derived artifact so components built from
different profiles are never mixed.

Sources deliberately in **no** profile (``NOT_USED``, with the reason):

* ``wlasl`` - WLASL is **American** Sign Language (the bundled ASL model's data, C-UDA licence), not an ISL source;
  phase 2 does not warm-start ISL models from ASL weights, so it was removed from the ISL profiles.
* ``isl_csltr_words`` - the ISL-CSLTR word-level items are word-labelled (valid word labels, CC BY 4.0) but each one
  is a **single still frame** of a sign that is mostly a movement. The word classifier / encoder read sequences
  (>= 8 frames, velocity features) and the pre-training objectives need time, so a one-frame "clip" would only
  teach that a frozen hand shape is the word; the letter model takes static hand shapes but these are words, not
  letters. Kept ingested (adapter registered) for a later hand-shape study, used by no model.
"""

from __future__ import annotations

from dataclasses import dataclass

COMMERCIAL, RESEARCH = "commercial-safe", "research"

# labelled word sources (the order is the order of preference when the same video appears twice)
WORD_SOURCES: dict[str, tuple[str, ...]] = {
    COMMERCIAL: ("include", "islrtc", "mendeley_emergency", "mendeley_agri"),
    RESEARCH: ("include", "islrtc", "isl500", "wslp", "cislr", "mendeley_emergency", "mendeley_agri"),
}
# unlabelled / sentence sources used only by self-supervised pre-training (the word sources are added there
# without their labels)
PRETRAIN_SOURCES: dict[str, tuple[str, ...]] = {
    COMMERCIAL: ("isl_csltr",),
    RESEARCH: ("isl_csltr", "glosl", "openhands_raw", "isign", "wslp_unlabelled"),
}
# letter / digit sources for the fingerspelling model
LETTER_SOURCES: dict[str, tuple[str, ...]] = {
    COMMERCIAL: ("mendeley_static",),
    RESEARCH: ("mendeley_static", "isl_fingerspelling", "openhands_fs", "hf_hemg_alphabet", "hf_akrit_alphabet"),
}
NOT_USED: dict[str, str] = {
    "wlasl": "American Sign Language (ASL) data, not ISL; no ASL warm start in phase 2",
    "isl_csltr_words": "one still frame per word: word models need sequences, the letter model needs letters",
}


@dataclass(frozen=True)
class DataProfile:
    name: str
    sources: tuple[str, ...]  # labelled word sources: training, index and calibration
    pretraining: tuple[str, ...]  # extra sources used only to initialise weights (labels never read)
    commercial: bool
    letters: tuple[str, ...] = ()  # fingerspelling letter / digit sources

    def all_sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.sources, *self.pretraining, *self.letters)))

    def to_dict(self) -> dict:
        return {"name": self.name, "sources": list(self.sources), "pretraining": list(self.pretraining),
                "letters": list(self.letters), "commercial": self.commercial, "not_used": dict(NOT_USED)}


def _profile(name: str) -> DataProfile:
    return DataProfile(name, WORD_SOURCES[name], PRETRAIN_SOURCES[name], name == COMMERCIAL, LETTER_SOURCES[name])


PROFILES: dict[str, DataProfile] = {n: _profile(n) for n in (COMMERCIAL, RESEARCH)}

DEFAULT_PROFILE = COMMERCIAL


def get_profile(name: str | DataProfile | None = None) -> DataProfile:
    if isinstance(name, DataProfile):
        return name
    name = name or DEFAULT_PROFILE
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown data profile {name!r}; choose one of {sorted(PROFILES)}") from None
