"""OpenHands Indian fingerspelling set (Zenodo 6731224, ``Indian.zip``, 28 MB): 36 glosses (letters a-z,
digits zero-nine), 172 short MP4 clips (123 train / 49 test) plus pickled poses.

The pickles are **not** loaded (unpickling untrusted data can run code). The clips are tiny, so the ingest runs
our own ``HolisticExtractor`` on the MP4s instead (``scripts/ingest_isl.py`` ``extract_full``, 15 fps), which
gives exact feature parity. Zenodo labels the record CC BY 4.0 but does not say where the clips come from, so
it stays in the **research profile** until the origin is known. Use: fingerspelling smoke tests (E5).
Signer ids are unknown, so ``group`` is the clip itself (``openhands_fs:<split>/<gloss>/<n>``).
"""

from __future__ import annotations

from collections.abc import Iterator

from app.isl.data.pose_sources.common import PoseClip
from app.isl.data.sources import MIN_FRAMES, _Base

RECORD = "https://zenodo.org/api/records/6731224"
URL = "https://zenodo.org/records/6731224"
ZIP_NAME = "Indian.zip"
LICENCE = "CC-BY-4.0 (Zenodo label; clip origin undocumented)"
DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
          "eight": "8", "nine": "9"}


def parse_member(name: str) -> dict | None:
    """``Indian/videos/train/r/1.mp4`` -> {split, gloss, symbol, kind, n}; None for anything else."""
    parts = name.replace("\\", "/").split("/")
    if len(parts) != 5 or parts[1] != "videos" or not parts[4].lower().endswith(".mp4"):
        return None
    split, gloss, file = parts[2], parts[3].lower(), parts[4]
    if split not in ("train", "test"):
        return None
    if gloss in DIGITS:
        kind, symbol = "digit", DIGITS[gloss]
    elif len(gloss) == 1 and gloss.isalpha():
        kind, symbol = "letter", gloss.upper()
    else:
        return None
    return {"split": split, "gloss": gloss, "symbol": symbol, "kind": kind, "n": file.rsplit(".", 1)[0]}


def clip_id(meta: dict) -> str:
    return f"openhands_fs/{meta['split']}/{meta['gloss']}/{meta['n']}"


class OpenHandsFingerspellAdapter(_Base):
    """Letter / digit clips. ``raw_gloss`` is the symbol (``A``, ``7``); ``quality["split"]`` the shipped split."""

    name = "openhands_fs"
    licence = LICENCE
    commercial = False
    manifest_name = "openhands_fs"
    profiles = ("research",)
    pretraining_only = False

    def clips(self, *, only_ok: bool = True) -> Iterator[PoseClip]:
        for rec in self.records(only_ok=only_ok):
            if only_ok and int(rec.get("n_frames") or 0) < MIN_FRAMES // 2:
                continue
            yield PoseClip(
                clip_id=rec["id"], source="openhands_fs", raw_gloss=rec.get("symbol") or rec.get("gloss", ""),
                concept_id=None, variant_id=f"openhands_fs:{rec.get('symbol')}", category=rec.get("kind"),
                path=self.paths.resolve(rec["path"]), group=f"openhands_fs:{rec['id'].removeprefix('openhands_fs/')}",
                fps=rec.get("fps"), licence=self.licence, profiles=self.profiles,
                quality={"n_frames": rec.get("n_frames"), "hands_frac": rec.get("hands_frac"),
                         "split": rec.get("split"), "kind": rec.get("kind")},
            )
