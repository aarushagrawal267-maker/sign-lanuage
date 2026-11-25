"""Catalogue of the ISL video / image sources that need our own MediaPipe extraction.

Every source has a licence, a data profile and the extraction mode used for it. Facts were checked on
25 Nov 2026 (sizes from the Hugging Face tree API, Mendeley public API and the zip central directories);
see docs/DATASETS_ISL.md section 3.3 for the measured numbers.

Extraction modes (``backend/app/isl/data/video_sources/extract.py``):

* ``full``  - the INCLUDE path: ``scripts/ingest_isl.py extract_full`` (HolisticExtractor, max side 640) over the
  whole clip, sampled at ``step_fps`` (15 fps like INCLUDE; 30 = every frame for fingerspelling letter timing).
* ``hands`` - ``full`` plus MediaPipe HandLandmarker (VIDEO mode) for frames where Holistic found no hand. Used where
  the signer's body/face is mostly out of frame (Holistic finds hands only through the pose), e.g. the Mendeley
  emergency set (signer behind a board, hands only).
* ``image`` - one still image -> a one-frame ``RawSequence``: Holistic in IMAGE mode (pose, face) + HandLandmarker
  (IMAGE mode, 2 hands) for the hands. Holistic alone found a hand in only 2 of 15 sample alphabet photos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COMMERCIAL_SAFE = "commercial-safe"
RESEARCH = "research"


@dataclass(frozen=True)
class SourceSpec:
    name: str  # CLI name, manifest name (data/isl/manifests/<manifest>.jsonl)
    title: str
    licence: str
    profile: str  # commercial-safe | research (research = never in a commercial build)
    url: str
    citation: str
    kind: str  # video | image
    mode: str  # full | hands | image
    step_fps: float = 15.0
    max_s: float = 20.0
    manifest: str = ""  # default = name
    id_prefix: str = ""  # default = name
    gated: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def manifest_name(self) -> str:
        return self.manifest or self.name

    @property
    def prefix(self) -> str:
        return self.id_prefix or self.name

    @property
    def commercial(self) -> bool:
        return self.profile == COMMERCIAL_SAFE


WSLP_REPO = "Exploration-Lab/WSLP"
FS_REPO = "kirandevraj/ISL-Fingerspelling"
CISLR_REPO = "Exploration-Lab/CISLR"
MENDELEY_API = "https://data.mendeley.com/public-api/datasets"

SOURCES: dict[str, SourceSpec] = {s.name: s for s in [
    SourceSpec(
        "wslp", "WSLP 2025 shared task, word recognition (train split, labelled)", "CC-BY-NC-ND-4.0", RESEARCH,
        f"https://huggingface.co/datasets/{WSLP_REPO}",
        "WSLP 2025 shared tasks (Workshop on Sign Language Processing @ AACL-IJCNLP 2025), Exploration-Lab, IIT Kanpur",
        "video", "full",
        extra={"repo": WSLP_REPO, "zip": "Shared_task_WR/Train-Dataset/train.zip",
               "csv": "Shared_task_WR/Train-Dataset/train_WR.csv", "split": "train"},
        notes="4,398 clips / 4,361 words; CISLR-like single-word clips (our CISLR substitute). ND: landmarks are "
              "derivatives for internal research only, never redistributed."),
    SourceSpec(
        "wslp_unlabelled", "WSLP 2025 word recognition, validation + test clips (labels not published)",
        "CC-BY-NC-ND-4.0", RESEARCH, f"https://huggingface.co/datasets/{WSLP_REPO}",
        "WSLP 2025 shared tasks, Exploration-Lab", "video", "full", manifest="wslp", id_prefix="wslp",
        extra={"repo": WSLP_REPO, "zips": {"val": "Shared_task_WR/Validation-Dataset/val.zip",
                                           "test": "Shared_task_WR/Test-Dataset/test.zip"}},
        notes="635 clips without labels (val 109, test 526): self-supervised / pre-training only, or labelled later "
              "if the organisers publish the answers."),
    SourceSpec(
        "cislr", "CISLR v1.5-a (gated)", "CC-BY-NC-4.0 (README; AFL-3.0 tag)", RESEARCH,
        f"https://huggingface.co/datasets/{CISLR_REPO}",
        "A. Joshi et al., CISLR: Corpus for Indian Sign Language Recognition, EMNLP 2022", "video", "full",
        gated=True, extra={"repo": CISLR_REPO, "zip": "CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip"},
        notes="Same ids, paths and manifest as scripts/ingest_isl_gated.py cislr (cislr/<uid>), plus sha256."),
    SourceSpec(
        "isl_fingerspelling", "ISL-Fingerspelling (continuous, per-letter frame boundaries)", "CC-BY-NC-4.0", RESEARCH,
        f"https://huggingface.co/datasets/{FS_REPO}",
        "Continuous Fingerspelling Dataset for Indian Sign Language, WSLP @ AACL-IJCNLP 2025 (letter annotations: "
        "ICPR 2026)", "video", "full", step_fps=30.0, max_s=60.0, id_prefix="islfs",
        extra={"repo": FS_REPO},
        notes="1,308 segments, 14,685 letters, 3 signers; every frame kept so letter boundaries (source frame "
              "indices) map 1:1 onto t_ms."),
    SourceSpec(
        "isl_csltr", "ISL-CSLTR sentence videos (Mendeley kcmpdxky7p v1)", "CC-BY-4.0", COMMERCIAL_SAFE,
        "https://data.mendeley.com/datasets/kcmpdxky7p/1",
        "R. Elakkiya, B. Natarajan, ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language Translation "
        "and Recognition, Mendeley Data V1, doi:10.17632/kcmpdxky7p.1", "video", "full", max_s=60.0,
        id_prefix="csltr", extra={"dataset": "kcmpdxky7p", "version": 1, "folder": "Videos_Sentence_Level/"},
        notes="687 videos of 100 sentences (7 signers) with sentence-level gloss sequences; continuous signing."),
    SourceSpec(
        "isl_csltr_words", "ISL-CSLTR word-level frames (Mendeley kcmpdxky7p v1)", "CC-BY-4.0", COMMERCIAL_SAFE,
        "https://data.mendeley.com/datasets/kcmpdxky7p/1",
        "R. Elakkiya, B. Natarajan, ISL-CSLTR, Mendeley Data V1, doi:10.17632/kcmpdxky7p.1", "image", "image",
        id_prefix="csltr_word", extra={"dataset": "kcmpdxky7p", "version": 1, "folder": "Frames_Word_Level/"},
        notes="1,036 still frames of words: one frame of a mostly dynamic sign; hand-shape hints only."),
    SourceSpec(
        "mendeley_emergency", "ISL emergency words (Mendeley 2vfdm42337 v1, raw videos)", "CC-BY-4.0",
        COMMERCIAL_SAFE, "https://data.mendeley.com/datasets/2vfdm42337/1",
        "V. Adithya, R. Rajesh, A Video Dataset of the Hand Gestures of Indian Sign Language Words used in Emergency "
        "Situations, Mendeley Data V1, doi:10.17632/2vfdm42337.1", "video", "hands",
        id_prefix="emergency", extra={"dataset": "2vfdm42337", "version": 1, "folder": "Raw_Data"},
        notes="8 words x 26 people x 2 = 416 clips; signer behind a black board, only hands/forearms visible "
              "(Holistic alone: 0-5% frames with a hand), so hands come from HandLandmarker."),
    SourceSpec(
        "mendeley_agri", "ISL agricultural words (Mendeley 6v53kfxf46 v1, original videos)", "CC-BY-4.0",
        COMMERCIAL_SAFE, "https://data.mendeley.com/datasets/6v53kfxf46/1",
        "V. Adithya, R. Rajesh, A video dataset on agricultural words in Indian sign language, Mendeley Data V1, "
        "doi:10.17632/6v53kfxf46.1", "video", "hands",
        id_prefix="agri", extra={"dataset": "6v53kfxf46", "version": 1, "folder": "Original"},
        notes="13 words, 3 deaf farmers outdoors; only the 'Original' folders (78 videos): crop1/2/3 are crops of "
              "the same recordings (duplicates for us)."),
    SourceSpec(
        "mendeley_static", "Static ISL alphabet, Hindi vowels, numerals (Mendeley 7tsw22y96w v1)", "CC-BY-4.0",
        COMMERCIAL_SAFE, "https://data.mendeley.com/datasets/7tsw22y96w/1",
        "A. Singh, S. K. Singh, A. Mittal, B. B. Gupta, Static gestures of Indian Sign Language (ISL) for English "
        "Alphabet, Hindi Vowels and Numerals, Mendeley Data V1, doi:10.17632/7tsw22y96w.1", "image", "image",
        id_prefix="isl_static", extra={"dataset": "7tsw22y96w", "version": 1, "per_class": 300},
        notes="102,470 photos, 46 folders (A-Z with E1/E2, 8 Hindi vowels, 0-9 with 9a/9b), kids/teens/adults x "
              "full/half sleeves; a balanced sample (default 300 per folder, spread over the 6 groups)."),
    SourceSpec(
        "hf_hemg_alphabet", "Hemg/Indian_sign_language_dataset (alphabet + digits images)", "unknown", RESEARCH,
        "https://huggingface.co/datasets/Hemg/Indian_sign_language_dataset", "Hugging Face user Hemg (no card)",
        "image", "image", id_prefix="hf_hemg",
        extra={"dataset": "Hemg/Indian_sign_language_dataset", "splits": ["train"], "per_class": 300},
        notes="42,745 128x128 hand crops, 35 classes, no licence or provenance -> research only."),
    SourceSpec(
        "hf_akrit_alphabet", "akritRihal/Indian_Sign_Language_dataset (alphabet + digits images)", "unknown",
        RESEARCH, "https://huggingface.co/datasets/akritRihal/Indian_Sign_Language_dataset",
        "Hugging Face user akritRihal (no card)", "image", "image", id_prefix="hf_akrit",
        extra={"dataset": "akritRihal/Indian_Sign_Language_dataset", "splits": ["train", "test"], "per_class": 300},
        notes="10,752 images, 33 classes (no h, j, v), no licence or provenance -> research only."),
]}


def get_source(name: str) -> SourceSpec:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown video source {name!r}; choose one of {sorted(SOURCES)}") from None


def profile_sources(profile: str) -> list[str]:
    """Video-source manifests allowed in a data profile (``commercial-safe`` excludes every research source)."""
    names = [s.manifest_name for s in SOURCES.values() if profile == RESEARCH or s.profile == COMMERCIAL_SAFE]
    return list(dict.fromkeys(names))
