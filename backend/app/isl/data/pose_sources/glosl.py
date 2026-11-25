"""GloSL-ISL (``pedroodb/glosl-isl`` on Hugging Face): iSign / ISLTranslate MediaPipe Holistic poses, one
``.pose`` file per sentence clip, with English text. CC BY-NC 4.0 -> **research profile, pre-training only**.

Card facts (checked 25 Nov 2026): 126,846 clips, 269 h, splits train/dev/test 101,481 / 12,683 / 12,682
(``splits/glosl_v1.csv``: source-video-disjoint, topic-stratified; ``annotations.csv`` carries the same split),
pose-format v0.2, 543 points (POSE 33, FACE 468, LEFT_HAND 21, RIGHT_HAND 21), x/y in pixels of the header
width x height, z raw, missing = (0,0,0) with confidence 0, native fps per file (25 or 29 for almost all).
Most headers say 300 x 300; the hand-bone aspect estimate (``common.hand_aspect``) gives 0.85-1.07 on those,
so they are square signer crops, not placeholders (a few files carry real crop sizes such as 620 x 712).

Ids: the GloSL ``id`` **is the iSign / ISLTranslate ``uid``** (``<video>-<n>``). 30,648 of the 31,221 public
ISLTranslate uids (``ISLTranslate.csv`` on GitHub) are GloSL ids; iSign itself is gated so the remaining ids
cannot be checked yet. The manifest stores ``isign_uid`` and ``in_isltranslate`` per clip and the ingest writes
``derived/glosl_isign_uids.txt``; a future iSign ingest must skip ``isign/<uid>`` for those uids
(:func:`isign_duplicates`).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.isl.data.pose_sources.common import PoseClip
from app.isl.data.sources import MIN_FRAMES, MIN_HANDS_FRAC, _Base

REPO = "pedroodb/glosl-isl"
LICENCE = "CC-BY-NC-4.0"
URL = "https://huggingface.co/datasets/pedroodb/glosl-isl"
ISLTRANSLATE_CSV = "https://raw.githubusercontent.com/Exploration-Lab/ISLTranslate/main/data/ISLTranslate.csv"
EXTRACTOR_VERSION = "glosl-pose-format"
CITATION = ("iSign (Joshi et al., Findings of ACL 2024); ISLTranslate (Joshi et al., Findings of ACL 2023); "
            "CISLR (EMNLP 2022); PoseStitch-SLT (EMNLP 2025); WSLP 2025 overview; GloSL (pedroodb)")


def clip_id(gid: str) -> str:
    return f"glosl/{gid}"


def read_annotations(path: Path) -> list[dict]:
    """Rows of ``annotations.csv`` (text may contain newlines, so the csv module is required)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("id") and r.get("pose_file")]


def read_split_file(path: Path) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return {r["id"]: r["split"] for r in csv.DictReader(fh) if r.get("id")}


def read_id_column(path: Path, column: str = "id") -> set[str]:
    if not Path(path).exists():
        return set()
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return {(r.get(column) or "").strip() for r in csv.DictReader(fh)} - {""}


def isign_duplicates(isign_uids: Iterable[str], glosl_uids: Iterable[str]) -> set[str]:
    """iSign uids already ingested through GloSL (same pose data): skip or drop them on the iSign side."""
    return set(isign_uids) & set(glosl_uids)


class GloslAdapter(_Base):
    """Sentence clips (``raw_gloss`` = English text, ``group`` = source video) for encoder pre-training,
    sentence-level data and negatives. ``quality["split"]`` is the GloSL v1 split."""

    name = "glosl"
    licence = LICENCE
    commercial = False
    manifest_name = "glosl"
    profiles = ("research",)
    pretraining_only = True

    def clips(self, *, only_ok: bool = True) -> Iterator[PoseClip]:
        for rec in self.records(only_ok=only_ok):
            if only_ok and (int(rec.get("n_frames") or 0) < MIN_FRAMES
                            or float(rec.get("hands_frac") or 0.0) < MIN_HANDS_FRAC):
                continue
            yield PoseClip(
                clip_id=rec["id"], source="glosl", raw_gloss=rec.get("text") or "", concept_id=None, variant_id=None,
                category=None, path=self.paths.resolve(rec["path"]), group=f"glosl-video:{rec.get('video_ref')}",
                fps=rec.get("fps"), licence=self.licence, profiles=self.profiles,
                quality={k: rec.get(k) for k in ("n_frames", "hands_frac", "split", "text", "isign_uid",
                                                 "in_isltranslate", "aspect_est", "joint_train_exclude",
                                                 "extractor_version", "dtype", "stride")},
            )

    def uids(self) -> set[str]:
        return {r.get("isign_uid") or r["id"].removeprefix("glosl/") for r in self.records(only_ok=True)}
