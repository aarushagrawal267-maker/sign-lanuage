"""ISL source adapters: one :class:`Clip` per usable landmark sequence (docs/ISL_DESIGN.md section 5.2).

Adapters are read-only over the ingestion manifests and tolerate a manifest that is still growing (see
``manifest.py``). Every clip is a ``RawSequence`` ``.npz`` plus an optional frame slice (ISLRTC first-sign
trims live in a sidecar, never in the ``.npz``).

INCLUDE id mapping: the manifest ``id`` is ``include/{category}/{word_dir}/{stem}`` with spaces -> ``_``.
The ingestion job derived category/word from the last path parts, which is wrong for the 27 videos stored
one level deeper in an ``Extra/`` folder (it produced ``include/5._Beautiful/Extra/MVI_9573`` with gloss
"Extra"). Adapters therefore rebuild the canonical id from the zip ``member`` path with ``Extra/`` removed:
``include/Adjectives/5._Beautiful/MVI_9573``; the manifest id is kept in ``quality["manifest_id"]``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Protocol

from app.isl.data.manifest import read_manifest
from app.isl.paths import IslPaths, isl_paths
from app.isl.profiles import DataProfile, get_profile
from app.landmarks.raw import RawSequence

MIN_FRAMES = 8
MIN_HANDS_FRAC = 0.2
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class Clip:
    clip_id: str  # "include/Seasons/63._Winter/MVI_4997", "islrtc/W/Winter.mp4", "cislr/<uid>", "mysign/<id>/2"
    source: str  # include | islrtc | cislr | isign | mysigns
    raw_gloss: str  # as in the source
    concept_id: str | None  # after vocabulary linking (None = unlinked)
    variant_id: str | None
    category: str | None
    path: Path  # RawSequence .npz
    start: int = 0  # frame slice applied on load (ISLRTC trims)
    end: int | None = None
    group: str | None = None  # signer / pseudo-signer / session id used for grouped splits
    fps: float | None = None
    licence: str = ""
    quality: dict = field(default_factory=dict)  # hands_frac, n_frames, trim confidence...

    def load(self) -> RawSequence:
        raw = RawSequence.load(self.path)
        if self.start or self.end is not None:
            raw = raw.slice(self.start, self.end if self.end is not None else len(raw))
        return raw

    def with_slice(self, start: int, end: int | None) -> Clip:
        return replace(self, start=start, end=end)


class SourceAdapter(Protocol):
    name: str
    licence: str
    commercial: bool

    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]: ...

    def available(self) -> bool: ...


def passes_quality(rec: dict) -> bool:
    return int(rec.get("n_frames") or 0) >= MIN_FRAMES and float(rec.get("hands_frac") or 0.0) >= MIN_HANDS_FRAC


# --------------------------------------------------------------------------------------------- INCLUDE ids

def include_key_from_parts(parts: tuple[str, ...] | list[str]) -> tuple[str, str] | None:
    """``("Adjectives", "5. Beautiful", "Extra", "MVI_9573.MOV")`` -> ``("Adjectives/5. Beautiful", "MVI_9573")``.

    Any ``Extra`` folder level (any case) is dropped; the last three remaining parts are
    category / word folder / file."""
    parts = [p for p in parts if p and p not in ("/", ".")]
    if not parts:
        return None
    file = parts[-1]
    dirs = [p for p in parts[:-1] if p.lower() != "extra"]
    if len(dirs) < 2:
        return None
    stem = PurePosixPath(file).stem if PurePosixPath(file).suffix.lower() in VIDEO_SUFFIXES else file
    return f"{dirs[-2]}/{dirs[-1]}", stem


def include_clip_id(word_dir: str, stem: str) -> str:
    return f"include/{word_dir}/{stem}".replace(" ", "_")


def include_match_key(clip_id: str) -> str:
    """Key used to match official split paths to manifest clips (case-insensitive, spaces == '_')."""
    return clip_id.replace(" ", "_").lower()


def read_split_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------------------------- adapters

class _Base:
    name = ""
    licence = ""
    commercial = True
    manifest_name = ""

    def __init__(self, paths: IslPaths | None = None, vocab=None):
        self.paths = paths or isl_paths()
        self.vocab = vocab

    @property
    def manifest_path(self) -> Path:
        return self.paths.manifest(self.manifest_name)

    def available(self) -> bool:
        return self.manifest_path.exists() and self.manifest_path.stat().st_size > 0

    def records(self, only_ok: bool = True) -> list[dict]:
        return read_manifest(self.manifest_path, only_ok=only_ok)


class IncludeAdapter(_Base):
    """INCLUDE (AI4Bharat): ~16 clips per word, 263 words, several signers. Reads ``include.jsonl`` and,
    when present, ``splits/include_groups.json`` (clip -> session / pseudo-signer)."""

    name = "include"
    licence = "CC-BY-4.0"
    commercial = True
    manifest_name = "include"

    def __init__(self, paths: IslPaths | None = None, vocab=None, groups: dict | None = None,
                 group_key: str = "pseudo_signer"):
        super().__init__(paths, vocab)
        if groups is None and self.paths.include_groups.exists():
            import json

            try:
                groups = json.loads(self.paths.include_groups.read_text()).get("clips", {})
            except (ValueError, AttributeError):
                groups = None
        self.groups = groups or {}
        self.group_key = group_key

    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]:
        from app.isl.vocab.build import include_variant_id
        from app.isl.vocab.normalize import parse_include_word

        seen: set[str] = set()
        for rec in self.records(only_ok=only_ok):
            if only_ok and not passes_quality(rec):
                continue
            key = include_key_from_parts(PurePosixPath(rec.get("member") or "").parts)
            if key is None:  # very old rows without member: fall back to the id
                key = include_key_from_parts(PurePosixPath(rec["id"].removeprefix("include/")).parts + ("x",))
                if key is None:
                    continue
                key = (key[0], PurePosixPath(rec["id"]).name)
            word_dir, stem = key
            cid = include_clip_id(word_dir, stem)
            if cid in seen:  # same video under two manifest ids (e.g. a legacy Extra/ id and a new one)
                continue
            seen.add(cid)
            variant = include_variant_id(word_dir)
            concept = self.vocab.concept_of_variant(variant) if self.vocab is not None else None
            g = self.groups.get(cid) or {}
            yield Clip(
                clip_id=cid, source="include", raw_gloss=parse_include_word(PurePosixPath(word_dir).name).headword,
                concept_id=concept, variant_id=variant, category=word_dir.split("/")[0], path=self.paths.resolve(rec["path"]),
                group=g.get(self.group_key) or g.get("session"), fps=rec.get("fps"), licence=self.licence,
                quality={"n_frames": rec.get("n_frames"), "hands_frac": rec.get("hands_frac"),
                         "manifest_id": rec["id"], "word_dir": word_dir},
            )


class IslrtcAdapter(_Base):
    """ISLRTC dictionary: one clip per entry, one signer. Joins the trim sidecar; skips ``n_frames == 0`` and
    entries whose name marks them as explanation / example / compilation."""

    name = "islrtc"
    licence = "GODL-India"
    commercial = True
    manifest_name = "islrtc"
    SIGNER_GROUP = "islrtc-signer-1"

    def __init__(self, paths: IslPaths | None = None, vocab=None, use_trims: bool = True,
                 include_non_index: bool = False):
        super().__init__(paths, vocab)
        self.use_trims = use_trims
        self.include_non_index = include_non_index

    def trims(self) -> dict[str, dict]:
        if not self.use_trims:
            return {}
        return {r["id"]: r for r in read_manifest(self.paths.trim_sidecar, only_ok=False)}

    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]:
        from app.isl.vocab.normalize import FULL_CLIP_KINDS, NON_INDEX_KINDS, parse_islrtc_name

        trims = self.trims()
        for rec in self.records(only_ok=only_ok):
            if not rec.get("n_frames"):
                continue
            if only_ok and not passes_quality(rec):
                continue
            rel = rec["id"].removeprefix("islrtc/")
            pn = parse_islrtc_name(rel)
            if pn.kind in NON_INDEX_KINDS and not self.include_non_index:
                continue
            concept, variant = (self.vocab.concept_of_clip(rec["id"]) if self.vocab is not None else (None, None))
            tr = trims.get(rec["id"])
            start, end = 0, None
            q = {"n_frames": rec.get("n_frames"), "hands_frac": rec.get("hands_frac"), "stop": rec.get("stop"),
                 "kind": pn.kind}
            if tr and pn.kind not in FULL_CLIP_KINDS and tr.get("method") != "empty":
                start, end = int(tr["start"]), int(tr["end"])
                q.update({"trim_method": tr.get("method"), "trim_confidence": tr.get("confidence"),
                          "trim_alternatives": tr.get("alternatives", [])})
            yield Clip(
                clip_id=rec["id"], source="islrtc", raw_gloss=rec.get("gloss") or pn.headword, concept_id=concept,
                variant_id=variant, category=rec.get("category"), path=self.paths.resolve(rec["path"]),
                start=start, end=end, group=self.SIGNER_GROUP, fps=rec.get("fps"), licence=self.licence, quality=q,
            )


class CislrAdapter(_Base):
    """CISLR (Exploration-Lab): ~4.7k words, one or a few clips each, collected from online ISL videos.
    Licence under review -> research profile only (docs/DATASETS_ISL.md)."""

    name = "cislr"
    licence = "AFL-3.0 (pending review)"
    commercial = False
    manifest_name = "cislr"

    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]:
        for rec in self.records(only_ok=only_ok):
            if only_ok and not passes_quality(rec):
                continue
            concept, variant = (self.vocab.concept_of_clip(rec["id"]) if self.vocab is not None else (None, None))
            yield Clip(
                clip_id=rec["id"], source="cislr", raw_gloss=rec.get("gloss", ""), concept_id=concept,
                variant_id=variant or f"cislr:{rec.get('uid', '')}", category=rec.get("category"),
                path=self.paths.resolve(rec["path"]), group=rec.get("group"), fps=rec.get("fps"), licence=self.licence,
                quality={"n_frames": rec.get("n_frames"), "hands_frac": rec.get("hands_frac"),
                         "split": rec.get("split")},
            )


class ISignAdapter(_Base):
    """iSign poses (sentence-level, CC BY-NC-SA 4.0): research profile only, used for pre-training."""

    name = "isign"
    licence = "CC-BY-NC-SA-4.0"
    commercial = False
    manifest_name = "isign"

    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]:
        for rec in self.records(only_ok=only_ok):
            if only_ok and int(rec.get("n_frames") or 0) < MIN_FRAMES:
                continue
            yield Clip(
                clip_id=rec["id"], source="isign", raw_gloss=rec.get("text", ""), concept_id=None, variant_id=None,
                category=None, path=self.paths.resolve(rec["path"]), group=rec.get("group"), fps=rec.get("fps"),
                licence=self.licence, quality={"n_frames": rec.get("n_frames"), "hands_frac": rec.get("hands_frac"),
                                               "extractor_version": rec.get("extractor_version")},
            )


class MySignsAdapter:
    """User signs (``data/my_signs/<pack>/signs.jsonl``). Never used in global training without an explicit
    opt-in; the store format is owned by ``app/isl/mysigns`` (rows: sign_id, concept_id, gloss, takes[path])."""

    name = "mysigns"
    licence = "per pack"
    commercial = True

    def __init__(self, root: Path | None = None):
        if root is None:
            from app.core.config import get_settings

            root = get_settings().data_dir / "my_signs"
        self.root = Path(root)

    def available(self) -> bool:
        return any(self.root.glob("*/signs.jsonl")) if self.root.exists() else False

    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]:
        from app.isl.data.manifest import iter_jsonl

        if not self.root.exists():
            return
        for f in sorted(self.root.glob("*/signs.jsonl")):
            for rec in iter_jsonl(f):
                if only_ok and rec.get("enabled") is False:
                    continue
                for i, take in enumerate(rec.get("takes") or []):
                    p = take.get("path") if isinstance(take, dict) else take
                    if not p:
                        continue
                    sid = rec.get("sign_id", "")
                    yield Clip(
                        clip_id=f"mysign/{sid}/{i}", source="mysigns", raw_gloss=rec.get("gloss", ""),
                        concept_id=rec.get("concept_id"), variant_id=f"mysign:{sid}", category=None,
                        path=(f.parent / p) if not Path(p).is_absolute() else Path(p),
                        group=f"mysigns:{f.parent.name}", licence=rec.get("licence", self.licence),
                    )


ADAPTERS = {"include": IncludeAdapter, "islrtc": IslrtcAdapter, "cislr": CislrAdapter, "isign": ISignAdapter}
_OPTIONAL_REGISTERED = False


def register_optional_adapters(adapters: dict | None = None) -> dict:
    """Add the video-source adapters (``video_sources.adapters.VIDEO_ADAPTERS``: WSLP, ISL-CSLTR, Mendeley, letters,
    ...) and the pose-source adapters (``pose_sources.register``: GloSL, ISL500, OpenHands) to ``adapters``
    (default: :data:`ADAPTERS`) without overwriting a name that is already there. Done lazily (first use) because
    those packages import this module. Which profile may use which source is decided in ``app.isl.profiles``
    (the single source of truth), not here."""
    global _OPTIONAL_REGISTERED
    reg = ADAPTERS if adapters is None else adapters
    try:
        from app.isl.data.video_sources.adapters import VIDEO_ADAPTERS

        for k, v in VIDEO_ADAPTERS.items():
            reg.setdefault(k, v)
    except ImportError:  # package not present in this checkout
        pass
    try:
        from app.isl.data.pose_sources import register

        register(reg)
    except ImportError:
        pass
    if adapters is None:
        _OPTIONAL_REGISTERED = True
    return reg


def all_adapters() -> dict:
    """:data:`ADAPTERS` with the optional video / pose adapters registered."""
    if not _OPTIONAL_REGISTERED:
        register_optional_adapters()
    return ADAPTERS


def get_adapter(name: str, paths: IslPaths | None = None, vocab=None):
    try:
        return all_adapters()[name](paths=paths, vocab=vocab)
    except KeyError:
        raise ValueError(f"no adapter for source {name!r}") from None


def load_profile_clips(profile: DataProfile | str, paths: IslPaths | None = None, vocab=None,
                       pretraining: bool = False) -> list[Clip]:
    """Every usable clip of the labelled word sources a profile allows (``pretraining`` adds the pre-training-only
    sources). The lists come from ``app.isl.profiles``; a source without a registered adapter (e.g. the
    ``wslp_unlabelled`` view, built by ``training.multisource``) is skipped here."""
    prof = get_profile(profile)
    names = list(prof.sources) + ([s for s in prof.pretraining if s not in prof.sources] if pretraining else [])
    reg = all_adapters()
    out: list[Clip] = []
    for n in names:
        if n not in reg:
            continue
        ad = get_adapter(n, paths, vocab)
        if ad.available():
            out.extend(ad.clips())
    return out
