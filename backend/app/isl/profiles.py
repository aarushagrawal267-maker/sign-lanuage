"""Data profiles: which sources a model build may use (docs/ISL_DESIGN.md section 10.1).

``commercial-safe`` excludes non-commercial data (iSign, WLASL, CISLR: its README says CC BY-NC 4.0). It still
uses the ISLRTC dictionary, whose publisher forbids resale and profiteering (ISLRTC FAQ Q.23): fine for a free,
non-profit app with acknowledgement, but a paid release needs ISLRTC's permission (docs/DATASETS_ISL.md). The
profile name is recorded in every derived artifact so components built from different
profiles are never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataProfile:
    name: str
    sources: tuple[str, ...]  # adapters allowed for training, index and calibration
    pretraining: tuple[str, ...]  # extra sources used only to initialise weights
    commercial: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "sources": list(self.sources), "pretraining": list(self.pretraining),
                "commercial": self.commercial}


PROFILES: dict[str, DataProfile] = {
    "commercial-safe": DataProfile("commercial-safe", ("include", "islrtc"), (), True),
    "research": DataProfile("research", ("include", "islrtc", "cislr", "isign"), ("wlasl", "isign"), False),
}

DEFAULT_PROFILE = "commercial-safe"


def get_profile(name: str | DataProfile | None = None) -> DataProfile:
    if isinstance(name, DataProfile):
        return name
    name = name or DEFAULT_PROFILE
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown data profile {name!r}; choose one of {sorted(PROFILES)}") from None
