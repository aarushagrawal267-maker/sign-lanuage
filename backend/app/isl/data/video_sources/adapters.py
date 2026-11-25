"""Read-only :class:`~app.isl.data.sources.Clip` adapters over the video-source manifests.

They follow the same contract as ``app.isl.data.sources`` (tolerant of a growing manifest, quality filter,
``Clip.licence``) and are listed in :data:`VIDEO_ADAPTERS`; registering them in ``sources.ADAPTERS`` and the data
profiles is left to the model owner (docs/DATASETS_ISL.md 3.3 says which profile each one belongs to).

Concept linking: ``vocab.concept_of_clip(id)`` when the vocabulary knows the clip, otherwise ``concept_id`` stays
``None`` and ``raw_gloss`` carries the label (WSLP words, letters, sentences) for the vocabulary builder.
"""

from __future__ import annotations

from collections.abc import Iterator

from app.isl.data.sources import MIN_FRAMES, MIN_HANDS_FRAC, Clip, _Base
from app.isl.data.video_sources.catalog import SOURCES, SourceSpec


class VideoSourceAdapter(_Base):
    spec: SourceSpec
    min_frames = MIN_FRAMES
    require_label = True

    def __init__(self, paths=None, vocab=None, require_label: bool | None = None, include_duplicates: bool = False):
        super().__init__(paths, vocab)
        if require_label is not None:
            self.require_label = require_label
        self.include_duplicates = include_duplicates  # rows with duplicate_of = byte-identical copy of another row

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.spec.manifest_name

    @property
    def licence(self) -> str:  # type: ignore[override]
        return self.spec.licence

    @property
    def commercial(self) -> bool:  # type: ignore[override]
        return self.spec.commercial

    @property
    def manifest_name(self) -> str:  # type: ignore[override]
        return self.spec.manifest_name

    def passes(self, rec: dict) -> bool:
        return int(rec.get("n_frames") or 0) >= self.min_frames and float(rec.get("hands_frac") or 0.0) >= MIN_HANDS_FRAC

    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]:
        for rec in self.records(only_ok=only_ok):
            if only_ok and not self.passes(rec):
                continue
            if self.require_label and not rec.get("gloss"):
                continue
            if rec.get("duplicate_of") and not self.include_duplicates:
                continue
            concept, variant = (self.vocab.concept_of_clip(rec["id"]) if self.vocab is not None
                                and hasattr(self.vocab, "concept_of_clip") else (None, None))
            q = {k: rec.get(k) for k in ("n_frames", "hands_frac", "split", "sha256", "extractor_version",
                                         "hands_filled", "letters", "glosses", "variant", "standard_split",
                                         "signer_split", "labelled", "src_fps") if k in rec}
            yield Clip(
                clip_id=rec["id"], source=self.name, raw_gloss=rec.get("gloss") or "", concept_id=concept,
                variant_id=variant, category=rec.get("category"), path=self.paths.resolve(rec["path"]),
                group=rec.get("group"), fps=rec.get("fps"), licence=self.licence, quality=q,
            )


def _make(spec: SourceSpec, min_frames: int = MIN_FRAMES, require_label: bool = True) -> type[VideoSourceAdapter]:
    return type(f"{spec.name.title().replace('_', '')}Adapter", (VideoSourceAdapter,),
                {"spec": spec, "min_frames": min_frames, "require_label": require_label})


WslpAdapter = _make(SOURCES["wslp"])  # labelled rows only (the unlabelled val/test rows share the manifest)
FingerspellingAdapter = _make(SOURCES["isl_fingerspelling"])
CsltrAdapter = _make(SOURCES["isl_csltr"])
CsltrWordFramesAdapter = _make(SOURCES["isl_csltr_words"], min_frames=1)
EmergencyAdapter = _make(SOURCES["mendeley_emergency"])
AgriAdapter = _make(SOURCES["mendeley_agri"])
StaticAlphabetAdapter = _make(SOURCES["mendeley_static"], min_frames=1)
HemgAlphabetAdapter = _make(SOURCES["hf_hemg_alphabet"], min_frames=1)
AkritAlphabetAdapter = _make(SOURCES["hf_akrit_alphabet"], min_frames=1)

VIDEO_ADAPTERS: dict[str, type[VideoSourceAdapter]] = {
    "wslp": WslpAdapter,
    "isl_fingerspelling": FingerspellingAdapter,
    "isl_csltr": CsltrAdapter,
    "isl_csltr_words": CsltrWordFramesAdapter,
    "mendeley_emergency": EmergencyAdapter,
    "mendeley_agri": AgriAdapter,
    "mendeley_static": StaticAlphabetAdapter,
    "hf_hemg_alphabet": HemgAlphabetAdapter,
    "hf_akrit_alphabet": AkritAlphabetAdapter,
}
# CISLR keeps its existing adapter (app.isl.data.sources.CislrAdapter): same manifest, same ids.


def letter_samples(adapter: VideoSourceAdapter, only_ok: bool = True) -> Iterator[tuple[Clip, list[dict]]]:
    """Fingerspelling clips with their letter boundaries mapped onto our frame indices:
    ``[{"letter", "start", "end"}]`` where start/end index the RawSequence (end exclusive)."""
    import numpy as np

    for clip in adapter.clips(only_ok=only_ok):
        letters = clip.quality.get("letters") or []
        if not letters:
            continue
        raw = clip.load()
        if not len(raw):
            continue
        fps = float(clip.quality.get("src_fps") or 30.0)
        frame_of = np.rint(raw.t_ms * fps / 1000.0).astype(int)  # our sample index -> source frame
        out = []
        for x in letters:
            s = int(np.searchsorted(frame_of, int(x["start_frame"]), side="left"))
            e = int(np.searchsorted(frame_of, int(x["end_frame"]), side="left"))
            out.append({"letter": x["letter"], "start": s, "end": max(e, s + 1)})
        yield clip, out
