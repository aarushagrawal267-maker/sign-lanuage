# Indian Sign Language (ISL) support: architecture and design

Status: **Phase 1 implemented, models smoke-trained only** (25 Nov 2026): no trained ISL model is committed and no ISL
accuracy has been measured; phase 2 (full training and evaluation) is listed in section 15. Branch `claude/indian-sign-language`.
Audience: the agents that build it (Data Engineer, then AI Engineer for models, AI Engineer for Hindi and
sentences, then Frontend Developer) and the project owner.

---

## 0. Summary in plain words (for the owner)

* The app will have **two sign languages**: the existing **ASL** (American, 100 words, works as today)
  and a new **ISL** (Indian, following the ISLRTC dictionary that MP schools teach). You pick the language at the top of the app.
* For ISL, when someone signs a word, the app shows the word in **English and Hindi**.
* It will know **about 12,000 ISL dictionary entries** (from the government ISLRTC dictionary), and it will be
  **much better at about 260 common everyday words** for which we have many example videos (INCLUDE dataset).
* **It cannot be right 100% of the time.** No system can. So the app is built to **never quietly show a wrong word**:
  * If it is sure, it shows the word.
  * If it is not sure, it shows **"Did you mean…?" with the 5 most likely words** and the person taps the right one.
  * If it has no good guess, it says **"Not sure, please sign again or spell it"**.
* **"My signs":** a teacher or student can record their own (for example Indore) version of a sign 3 to 5 times and
  give it an English and Hindi word. The app then recognises it, and a school can **share its signs as a file** with other schools.
* **Spelling mode:** words that are not in the dictionary (names, places like "Indore") can be fingerspelled with the ISL two-hand alphabet.
* Words joined into a sentence are shown as an **approximate** English/Hindi sentence (clearly labelled), and can be read aloud.
* We report accuracy **honestly**: tested on people the model never saw, with the "not sure" rate shown next to the accuracy.

---

## 1. Goals, non-goals and constraints

### 1.1 Goals

| # | Goal | Measured by |
|---|---|---|
| G1 | Recognise isolated ISL words (ISLRTC standard) and show English + Hindi | E1, E2 (section 9) |
| G2 | Maximise coverage: every ISLRTC dictionary entry (about 12k concepts) is reachable, core words are strong | vocabulary coverage, open-vocab top-5 (E2) |
| G3 | Never silently show a wrong word: accept / suggest top-5 / abstain | silent-error rate, risk-coverage curve (E3) |
| G4 | Personal and regional signs (Indore/MP), shareable as packs | few-shot gain (E4), round-trip tests |
| G5 | ASL keeps working, unchanged, as a second language | existing test suite + ASL regression check (E7) |
| G6 | A **commercial-safe** model configuration next to a **research/best** one | licence table (section 10) |

### 1.2 Non-goals (this phase)

* Continuous ISL sentence translation (ISL grammar, classifiers, role shift). The sentence builder only joins recognised words and is labelled approximate.
* ~~Hindi user interface (menus in Hindi).~~ Moved into scope by the owner's frontend task (25 Nov 2026): a simple English/Hindi
  interface toggle covers the navigation, Live, Vocabulary, My signs and status texts (section 13). The research pages
  (Model Laboratory, Dataset, Evaluation, Studio, Analysis) stay English.
* GPU training, full-scale training before ingestion finishes (phase 2).

### 1.3 Constraints

* Machine: 4 CPUs, no GPU, 15 GB RAM, about 45 GB free disk. Ingestion jobs use about 3.5 CPUs until they finish.
  All heavy work: `nice -n 10`, `torch.set_num_threads(1..2)` while ingestion runs, up to 4 after.
* Privacy (and India's DPDP Act 2023): webcam video never leaves the browser (unchanged); only landmarks reach the local backend.
  "My signs" stores landmarks only, locally; export is an explicit user action with a consent text.
* One feature implementation for training, offline analysis and live inference (`backend/app/features/build.py`) must stay the single source of truth.
* Licences: each source tagged; the commercial-safe profile excludes non-commercial data (WLASL, iSign) and anything pending review (CISLR, see 10.2).

---

## 2. What exists today (reuse map)

| Piece | Path | Reused how |
|---|---|---|
| `RawSequence` (.npz: pose 33x4, hands 21x3, face 20x3, t_ms, NaN = missing) | `backend/app/landmarks/raw.py` | Canonical clip format for **every** ISL source and for "My signs" takes. INCLUDE/ISLRTC ingestion already writes it. |
| Holistic extractor | `backend/app/landmarks/extractor.py` | Used by `scripts/ingest_isl.py` and by CISLR ingestion. |
| `FeatureConfig`, raw/feature layout | `backend/app/features/spec.py` | Unchanged layout. Two **opt-in** fields added (`canonical_hand`, default off; `hand_size_floor`, default 0 = off, 0.1 for ISL, see 15.14) so ASL dims and behaviour stay identical. |
| `to_body_frame`, `augment`, `featurize`, `build_features` | `backend/app/features/build.py` | Used as is. Strong augmentation for ISL composes `augment` with extra ops in a new module. |
| `RecurrentClassifier`, `InputProjection`, `MaskedAttentionPool` | `backend/app/models/networks.py` | ISL closed-set classifier = the same class. Encoder trunk reuses projection + BiGRU + attention pool. |
| Checkpoint format + `Predictor` | `backend/app/models/checkpoint.py` | ISL classifier checkpoints use the same `{"state_dict","meta"}` format, so `Predictor` loads them. |
| Training loop | `backend/app/training/trainer.py` | The epoch loop is extracted into a DB-free `fit()` (section 6.6); `run_training` keeps calling it, so ASL training is unchanged. |
| Metrics (`topk_accuracy`, `confidence_curve`, `compute_metrics`, `pick_threshold`) | `backend/app/evaluation/metrics.py` | Reused; ISL adds risk-coverage, open-vocab and grouped bootstrap intervals. |
| Engine (segment boundaries, 3-crop TTA, threshold/margin/agreement, cooldown) | `backend/app/inference/engine.py` | Kept. It gains an optional `Recognizer` (section 4.3); with none, it behaves exactly as today. |
| Active-model registry | `backend/app/inference/registry.py` | Wrapped by the ASL language pack; untouched. |
| WebSocket `/api/ws/recognize`, `/api/analyze/*`, `/api/vocabulary` | `backend/app/api/recognition.py` | Gain an optional `language`; with none, the ASL behaviour is byte-for-byte the same. |
| Synthetic sequences | `backend/app/dataset/synthetic.py` | Basis of ISL unit tests (trimming, fusion, My signs). |
| Live hook and types | `frontend/src/recognition/useLiveRecognition.ts`, `live.ts`, `api/client.ts` | Extended with language, candidates, bilingual text. |
| Ingestion | `scripts/ingest_isl.py` (running) | Read-only from the design's point of view; only backward-compatible additions (section 5.7). |

---

## 3. Architecture overview

### 3.1 Key decisions (short ADRs)

**ADR-ISL-1: Language packs as a file-based registry next to the experiment DB.**
Context: ASL models live as `Experiment` rows (one active model). ISL needs several parts (classifier, encoder, index, letter model, lexicons, fusion parameters) that must be versioned together.
Decision: a *language pack* is a folder with `pack.json` plus artifacts. ASL is wrapped as a pack whose recogniser is the active experiment (`kind: "experiment-registry"`).
Consequences: ASL code paths untouched; ISL artifacts are versioned and swappable as one unit; the Model Laboratory does not yet train ISL (CLI-driven in phase 2). Reversible: a pack can later be registered as an experiment.

**ADR-ISL-2: Hybrid recognition = closed-set classifier for core words + embedding retrieval for the long tail.**
Context: INCLUDE gives about 16 clips per word for 263 words (multi-signer); ISLRTC gives **one** clip per entry for about 12k concepts (one signer per entry; several different signers across the dictionary, see 5.8).
Decision: (a) a classifier trained on the core words, (b) a metric-learning encoder plus a nearest-neighbour index over every dictionary entry and every user sign, (c) a calibrated fusion with an abstain rule.
Consequences: one-shot coverage of the whole dictionary; extra calibration work; two models to keep in sync (the pack versions them together).
Alternative rejected: a 12k-way classifier (one example per class cannot be learned as a softmax class; no way to add user signs without retraining).

**ADR-ISL-3: Landmarks only, `RawSequence` everywhere.**
All sources (INCLUDE, ISLRTC, CISLR, iSign poses, user takes) are converted to `RawSequence`. iSign ships poses in another format, so an adapter converts them (its extractor version is recorded and checked for distribution shift).

**ADR-ISL-4: Abstain by default.** The decision has three outcomes: `accept`, `suggest` (top-5), `abstain`. Thresholds are picked on validation data for a target precision of accepted words (95%), stricter than ASL's 85%.

**ADR-ISL-5: Hindi through a reviewed lexicon file, not live machine translation.** Every concept has a Hindi entry with source and review status, generated offline and reviewed by people. The UI shows the review status. No Hindi is ever invented at runtime.

**ADR-ISL-6: Data profiles.** `commercial-safe` and `research` are explicit, recorded in every checkpoint and every results file, and shown in the UI.

### 3.2 Component diagram (C4 container level)

```
Browser (React)                                     Local backend (FastAPI)
┌──────────────────────────────────────┐            ┌────────────────────────────────────────────────────────────────┐
│ Language switcher (asl-wlasl100|isl) │  landmarks │ /api/ws/recognize?language=isl                                 │
│ webcam → MediaPipe Holistic (WASM)   │ ─────────► │   RecognitionEngine (segments, crops, cooldown)                │
│ Live: word EN + HI, did-you-mean x5, │ ◄───────── │      └─ Recognizer (per pack)                                  │
│   "not sure", spell mode, sentence,  │  states    │          ASL: ClassifierRecognizer(Predictor)  (unchanged)     │
│   speak (speechSynthesis hi-IN/en-IN)│            │          ISL: IslRecognizer                                     │
│ My Signs: record 3-5 takes, export / │  REST      │               ├ core classifier (BiGRU, ~255 concepts)          │
│   import .signpack                   │ ─────────► │               ├ SignEncoder → cosine index (≈13.6k + CISLR)     │
│ Vocabulary: 12k entries, search,     │            │               ├ MySigns index (searched first)                  │
│   Hindi review, skeleton playback    │            │               ├ Fingerspelling (holds → letters → lexicon)       │
└──────────────────────────────────────┘            │               └ Fusion + calibration + abstain rule            │
                                                    │ /api/languages/{id}/vocabulary | my-signs | sentence | lexicon │
                                                    │ PackRegistry: models/packs/* (tracked, small) and              │
                                                    │               artifacts/packs/* (local builds, override)       │
                                                    │ data/isl/ (landmarks, manifests, splits, trims)  data/my_signs/│
                                                    └────────────────────────────────────────────────────────────────┘
Offline (CLI, phase 2):  python -m app.isl.cli  vocab | trim | splits | dedup | train-classifier | train-encoder |
                         build-index | train-letters | fit-fusion | hindi | evaluate | build-pack
```

### 3.3 New code layout

```
backend/app/
  packs/
    __init__.py
    schema.py          PackManifest (pydantic) for pack.json
    registry.py        PackInfo, LanguagePack protocol, PackRegistry, get_registry()
    asl.py             AslExperimentPack (wraps inference/registry.py)
    isl.py             IslPack (file-based; builds IslRecognizer lazily)
  inference/
    recognizer.py      Candidate, Recognition, Recognizer protocol, ClassifierRecognizer
  api/
    languages.py       new router: /api/languages/...
  db/
    migrate.py         ensure_columns(): adds recognition_events.language / detail if missing
  isl/
    __init__.py
    cli.py             python -m app.isl.cli <command>
    profiles.py        DataProfile, PROFILES (commercial-safe, research)
    vocab/
      schema.py        VocabEntry, Variant, SourceRef (pydantic)
      normalize.py     parse_islrtc_name(), parse_include_word(), slugify(), split_synonyms()
      build.py         build_vocabulary() → models/packs/isl/vocabulary.jsonl
      overrides.yaml   manual fixes (tracked)
    data/
      sources.py       Clip, SourceAdapter, IncludeAdapter, IslrtcAdapter, CislrAdapter, ISignAdapter, MySignsAdapter
      trim.py          activity_signal(), candidate_boundaries(), find_first_sign(), TrimResult
      boundary.py      small boundary model trained on synthetic concatenations (refinement stage 2)
      splits.py        include_official(), include_sessions(), include_pseudo_signers(), oneshot_folds()
      dedup.py         near-duplicate detection across sources
      remote.py        hf_access(), RangeFile (seekable HTTP over split parts), resumable member iteration
      cislr.py         CISLR ingestion (zip over HTTP ranges)
      isign.py         iSign poses ingestion (archive streaming, pose-format → RawSequence)
    augment.py         StrongAugmentConfig, augment_strong()
    models/
      encoder.py       SignEncoder
      losses.py        SupConLoss, SubCenterArcFace
      index.py         EmbeddingIndex (numpy cosine top-k), IndexMeta
      fusion.py        FusionParams, fuse(), decide()
      calibration.py   fit_temperature(), fit_fusion_weights()
    fingerspell/
      features.py      letter_frame_features()
      holds.py         detect_holds()
      model.py         LetterClassifier
      decoder.py       LexiconTrie, SpellDecoder
    mysigns/
      store.py         MySign, MySignsStore
      packfile.py      export_pack(), import_pack(), validation
    text/
      lexicon.py       HindiLexicon (base + local overrides)
      sentence.py      build_sentence()
      function_words.yaml
    recognizer.py      IslRecognizer (implements inference.recognizer.Recognizer)
    training/
      common.py        ClipDataset, PKSampler, cache of BodyFrames
      train_classifier.py, train_encoder.py, train_letters.py, build_index.py, fit_fusion.py
    evaluation/
      protocols.py     E1..E7 runners
      report.py        results.json writer with intervals and labels
backend/tests/isl/     test_*.py (synthetic data, no real data needed)
models/packs/
  asl-wlasl100/pack.json            (tracked; points at the existing experiment registry)
  isl/pack.json, vocabulary.jsonl, lexicon_hi.jsonl, ... (tracked when each file < 50 MB; weights only for the commercial-safe build)
scripts/ingest_isl.py               (existing, running)
scripts/ingest_isl_gated.py         (new: CISLR and iSign, resumable streaming)
frontend/src/
  state/language.tsx                LanguageContext + useLanguage()
  components/CandidateList.tsx, SentenceBar.tsx, LanguageSwitcher.tsx, BilingualWord.tsx
  pages/MySignsPage.tsx             (/my-signs)
  lib/speech.ts                     speechSynthesis wrapper
```

---

## 4. Sign language packs (task 1)

### 4.1 Pack folder and `pack.json`

Search order (later roots override earlier ones with the same `id`): `models/packs/` (tracked, shipped) then
`artifacts/packs/` (local builds; git-ignored). Settings: `SLR_PACKS_DIRS` (extra roots, `os.pathsep`-separated),
`SLR_DEFAULT_LANGUAGE=asl-wlasl100`, `SLR_ISL_PROFILE=commercial-safe`.

`models/packs/asl-wlasl100/pack.json`:

```json
{
  "schema": 1,
  "id": "asl-wlasl100",
  "kind": "experiment-registry",
  "sign_language": "ASL",
  "standard": "WLASL100",
  "display_name": {"en": "American Sign Language (WLASL100)", "hi": "अमेरिकी सांकेतिक भाषा"},
  "output_languages": ["en"],
  "capabilities": ["words"],
  "profile": "research",
  "licences": [{"source": "wlasl", "licence": "C-UDA-1.0", "commercial": false}]
}
```

`models/packs/isl/pack.json` (paths relative to the pack folder):

```json
{
  "schema": 1,
  "id": "isl",
  "kind": "isl-hybrid",
  "version": "2026.12.0",
  "sign_language": "ISL",
  "standard": "ISLRTC",
  "display_name": {"en": "Indian Sign Language (ISL)", "hi": "भारतीय सांकेतिक भाषा"},
  "output_languages": ["en", "hi"],
  "capabilities": ["words", "retrieval", "spelling", "my_signs", "sentence"],
  "profile": "commercial-safe",
  "feature_config": {"seq_len": 48, "use_face": true, "use_velocity": true, "max_gap": 5, "canonical_hand": true,
                     "hand_size_floor": 0.1},
  "vocabulary": "vocabulary.jsonl",
  "lexicons": {"hi": "lexicon_hi.jsonl"},
  "components": {
    "classifier":    {"checkpoint": "classifier.pt", "label_space": "concept_id"},
    "encoder":       {"checkpoint": "encoder.pt", "version": "isl-enc-1", "dim": 256},
    "index":         {"dir": "index", "encoder_version": "isl-enc-1"},
    "fingerspelling":{"checkpoint": "letters.pt", "lexicon": "spelling_lexicon.txt"},
    "fusion":        "fusion.json"
  },
  "results": "results.json",
  "licences": [
    {"source": "include", "licence": "CC-BY-4.0", "commercial": true},
    {"source": "islrtc", "licence": "GODL-India", "commercial": true}
  ]
}
```

Missing components degrade gracefully: no classifier → retrieval only; no index → classifier only; neither →
`status: "missing_models"` with a human-readable message (same style as `registry.load_active`'s `problem`).

### 4.2 Registry API (`backend/app/packs/registry.py`)

```python
@dataclass(frozen=True)
class PackInfo:
    id: str
    kind: str                       # "experiment-registry" | "isl-hybrid"
    sign_language: str              # "ASL" | "ISL"
    standard: str
    display_name: dict[str, str]    # {"en": ..., "hi": ...}
    output_languages: list[str]
    capabilities: frozenset[str]
    profile: str                    # "commercial-safe" | "research"
    version: str | None
    licences: list[dict]
    status: str                     # "ready" | "missing_models" | "error"
    message: str | None

    def to_json(self) -> dict: ...

class LanguagePack(Protocol):
    info: PackInfo
    def recognizer(self) -> tuple["Recognizer | None", dict | None, str | None]:
        """(recognizer, model_info, problem) - mirrors inference.registry.load_active()."""
    def vocabulary(self) -> "Vocabulary": ...

class UnknownLanguage(KeyError): ...

class PackRegistry:
    def __init__(self, roots: list[Path]): ...
    def list(self) -> list[PackInfo]: ...
    def get(self, pack_id: str | None) -> LanguagePack:   # None → default id; raises UnknownLanguage
    def default_id(self) -> str: ...
    def reload(self) -> None: ...                         # after build-pack / import

@lru_cache
def get_registry() -> PackRegistry: ...
```

* `AslExperimentPack.recognizer()` calls `inference.registry.load_active()` and wraps the `Predictor` in
  `ClassifierRecognizer`; its `vocabulary()` reuses the logic of today's `GET /api/vocabulary`.
* `IslPack.recognizer()` builds `IslRecognizer` once (thread lock + cache keyed by pack version and file mtimes,
  like the existing `_cache`), about 20 MB of memory.

### 4.3 Recogniser abstraction (`backend/app/inference/recognizer.py`)

```python
@dataclass
class Candidate:
    concept_id: str                  # ASL: the gloss itself ("book"); ISL: "isl.pencil"
    class_index: int                 # ASL: model class index; ISL: row in vocabulary.jsonl (stable per pack version)
    gloss: str                       # English headword (kept for the ASL-compatible TopK shape)
    prob: float                      # calibrated probability after fusion
    text: dict[str, str | None]      # {"en": "pencil", "hi": "पेंसिल"}  (hi None when no reviewed or generated entry)
    hi_review: str | None            # "reviewed" | "machine" | "disputed" | None
    source: str                      # "classifier" | "retrieval" | "classifier+retrieval" | "my_signs" | "spelling"
    variant_id: str | None = None
    region: str | None = None
    origin: str | None = None        # "ISLRTC dictionary" | "INCLUDE" | "My signs: <pack title>"

@dataclass
class Recognition:
    candidates: list[Candidate]      # sorted, length <= top_k
    decision: str                    # "accept" | "suggest" | "abstain"
    reason: str                      # "ok" | "low_confidence" | "ambiguous" | "unstable" | "out_of_vocabulary" | ...
    confidence: float                # calibrated p(top-1)
    agree: int                       # crops agreeing with top-1
    crops: int
    feature_ms: float
    inference_ms: float

class Recognizer(Protocol):
    language: str
    feature_config: FeatureConfig
    recommended_threshold: float
    def recognize(self, crops: list[RawSequence], top_k: int = 5) -> Recognition: ...
    def class_probs(self, crops: list[RawSequence]) -> np.ndarray | None:
        """Closed-set probabilities (sliding mode / EMA). None if the pack has no classifier."""

class ClassifierRecognizer:          # ASL and any Predictor-only pack
    def __init__(self, predictor: Predictor, language: str = "asl-wlasl100"): ...
```

Engine change (`backend/app/inference/engine.py`), backward compatible:

```python
class RecognitionEngine:
    def __init__(self, predictor: Predictor | None, width=640, height=480,
                 config: EngineConfig | None = None, recognizer: Recognizer | None = None): ...
```

* `recognizer is None` → the current code path, untouched (ASL; all existing tests keep passing).
* `recognizer` set → `_finalize` calls `recognizer.recognize(crops)`; the engine still applies cooldown and the
  `min_frames`/hand-ratio gates; `segment_result` gains `decision`, `candidates` (alias of `top`), `segment_id`.
* Sliding mode with a recogniser uses `class_probs()` (closed-set EMA, core words only); full fusion runs in segment mode.
  The UI says so.
* New `EngineConfig` fields (defaults keep ASL behaviour): `policy: str = "balanced"` (`strict|balanced|lenient`,
  maps to the pack's fitted thresholds), `use_my_signs: bool = True`, `spell_end_ms: float = 600.0`, and
  `mode` accepts `"spell"` (section 6.4).

### 4.4 REST API changes

New router `backend/app/api/languages.py` (all ISL routes live under the language prefix; old routes stay as ASL aliases):

| Method and path | Purpose | Response (abridged) |
|---|---|---|
| `GET /api/languages` | list packs | `{"default": "asl-wlasl100", "languages": [PackInfo...]}` |
| `GET /api/languages/{lang}` | one pack + model info + results summary | `{"pack": PackInfo, "model": {...}, "results": {...}}` |
| `GET /api/languages/{lang}/vocabulary?q=&category=&source=&kind=&has_hi=&in_classifier=&review=&offset=0&limit=50` | paged, searchable vocabulary (12k rows for ISL) | `{"total", "items": [VocabRow]}` |
| `GET /api/languages/{lang}/vocabulary/{concept_id}` | full entry, variants, sources, per-class metrics, optional skeleton preview frames (from local landmarks) | `VocabEntry + {"preview": {"width","height","frames":[...]}}` |
| `PATCH /api/languages/{lang}/lexicon/{concept_id}` | local Hindi correction / review (`{"hi", "hi_alt", "review"}`); disabled when `public_demo` | updated lexicon row |
| `POST /api/languages/{lang}/analyze/landmarks` | classify a recorded landmark clip | `Recognition` JSON |
| `POST /api/languages/{lang}/sentence` | `{"tokens": [concept_id...], "lang": ["en","hi"]}` | `{"en": SentenceOut, "hi": SentenceOut}` |
| `GET/POST /api/languages/{lang}/my-signs`, `PATCH/DELETE .../my-signs/{sign_id}`, `GET .../my-signs/export`, `POST .../my-signs/import` | section 7 | section 7 |
| `GET /api/languages/{lang}/results` | evaluation results with protocol labels | `results.json` |

Existing routes:

* `GET /api/vocabulary` → unchanged (ASL). It is the same as `/api/languages/asl-wlasl100/vocabulary` with the legacy shape.
* `POST /api/analyze/video` gains form field `language` (default `asl-wlasl100`).
* `POST /api/analyze/landmarks` gains optional body field `language`.
* `GET /api/models/active` unchanged (ASL experiment).
* `GET /api/recognition/events?language=` filter added.

DB: `db/migrate.py: ensure_columns(engine)` runs after `create_all` in `get_engine()` and adds, only if missing
(`PRAGMA table_info`), `recognition_events.language VARCHAR(32) DEFAULT 'asl-wlasl100'` and
`recognition_events.detail JSON` (stores `concept_id`, `text`, `source`, `decision`, `chosen_by_user`). No other table changes.

### 4.5 WebSocket protocol changes (`/api/ws/recognize`)

Selecting the language, two equivalent ways (both optional; default = ASL):

* query string: `ws://host/api/ws/recognize?language=isl`
* message: `{"type": "config", "language": "isl", "width": 1280, "height": 720, "engine": {...}}`.
  A language change rebuilds the engine and the server answers with a **new `hello`** (the client already handles `hello`).

`hello` gains fields (old fields unchanged):

```json
{"type": "hello", "model": {...}, "engine": {...}, "message": null,
 "language": {"id": "isl", "display_name": {"en": "Indian Sign Language (ISL)", "hi": "भारतीय सांकेतिक भाषा"},
              "output_languages": ["en", "hi"], "capabilities": ["words","retrieval","spelling","my_signs","sentence"],
              "profile": "commercial-safe", "status": "ready"},
 "languages": [{"id": "asl-wlasl100", "...": "..."}, {"id": "isl", "...": "..."}]}
```

`state` message: everything stays; ISL adds fields. `TopK` items become `Candidate` objects (a superset of
`{class_index, gloss, prob}`, so the current frontend still renders them).

```json
{"type": "state", "t": 123456.0, "status": "uncertain", "hands": false, "pose": true, "threshold": 0.71,
 "segment_result": {
   "segment_id": 17, "accepted": false, "decision": "suggest", "reason": "ambiguous",
   "confidence": 0.46, "frames": 41, "duration_ms": 1650, "agreement": "2/3",
   "top": [
     {"concept_id": "isl.pencil", "class_index": 8123, "gloss": "pencil", "prob": 0.46,
      "text": {"en": "pencil", "hi": "पेंसिल"}, "hi_review": "reviewed",
      "source": "classifier+retrieval", "variant_id": "islrtc:P/Pencil_(Sign_3)", "region": null, "origin": "ISLRTC dictionary"},
     {"concept_id": "isl.pen", "...": "..."}
   ],
   "feature_ms": 6.1, "inference_ms": 31.0},
 "commit": null}
```

`commit` (only when `decision == "accept"`): `{"gloss", "class_index", "score", "t"}` as today **plus**
`concept_id`, `text`, `hi_review`, `source`, `segment_id`, `chosen_by_user: false`.

New client → server messages:

| Message | Server reply | Notes |
|---|---|---|
| `{"type": "choose", "segment_id": 17, "concept_id": "isl.pen"}` | `{"type": "choice_ack", "commit": {..., "chosen_by_user": true}}` | user picked from "Did you mean"; recorded as an event with `chosen_by_user` (feedback for later evaluation; never used for training without consent) |
| `{"type": "reject", "segment_id": 17}` | `{"type": "reject_ack"}` | "this was wrong" on an accepted word; removes the last commit from the server-side sentence context and logs feedback |
| `{"type": "config", "engine": {"mode": "spell"}}` | `config_ack` | spelling mode (section 6.4) |

Spell-mode `state` adds `"spell": {"letters": [{"letter": "I", "prob": 0.91}, ...], "buffer": "INDO",
"suggestions": [{"word": "INDORE", "concept_id": "isl.indore", "score": 0.72}, ...]}`, and a finished word arrives as a
normal `segment_result` with candidates whose `source` is `"spelling"`.

Server-side validation keeps today's rules (finite numbers, shapes). `language` must be a known pack id; otherwise
`{"type": "error", "message": "unknown language 'xyz'; available: asl-wlasl100, isl"}` and the connection stays on its current language.

---

## 5. Data (task 3)

### 5.1 What is on disk (facts checked on 25 Nov 2026)

* `data/isl/manifests/islrtc_index.json`: 13,665 files (26 letter folders + `Numbers/` with 43 files), about 78 GB of video in total (not kept).
* `data/isl/manifests/islrtc.jsonl`: growing. In the first 59 rows: `stop=max_s` 31, `eof` 17, `max_s+posegap` 7, `card` 4;
  some rows have `n_frames: 0` (the title card came first, e.g. "Ab (Symbol of Line) (Math)"). These must be filtered out.
* The manifest `fps` and `duration_s` come from the container FPS and are approximate (some clips report 14 s although the cap is 10 s).
  **The npz `t_ms` is authoritative**; adapters must never use the manifest duration for timing.
* In ISLRTC clips, hands are visible almost all the time after the start; the signer seldom drops both hands
  between the sign and the explanation. Trimming therefore cannot rely on "hands absent" alone (section 5.3).
* `data/isl/manifests/include.jsonl`: not written yet (the first INCLUDE zip was downloading).
* INCLUDE official split lists exist in the AI4Bharat code repository: `train_test_paths/include_{train,val,test}.txt`
  (3,127 / 348 / 817 paths such as `Seasons/63. Winter/MVI_4997.MOV`; train + val = the 3,475 training videos and test = the 817 test videos named in the zenodo README; 27 paths carry an extra `Extra/` folder level).
  Fetch from `https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_{train,val,test}.txt`
  (and `include50_*`) and store under `data/isl/splits/official/`.
* **The official INCLUDE split is not signer-independent.** File numbers show that consecutive takes from the same recording
  run are spread across train and test (for example "Winter": MVI_4997 train, MVI_4998 test; MVI_5430 val, MVI_5431 train, MVI_5432 test).
  We report it only labelled as such (section 9).
* CISLR and iSign: **`README.md` returns 200 even without access** (the dataset card is public) while every data file returns
  403 ("not in the authorized list"). The access check must probe a data file (section 5.7).
  Gated files: CISLR `dataset.csv` (240 KB), `prototype.csv`, `test.csv`, `CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip` (1.17 GB),
  `I3D_features.pkl`; iSign `iSign_v1.1.csv` (9.8 MB), `word-presence-dataset_v1.1.csv`, `word-description-dataset_v1.1.csv`,
  `iSign-poses_v1.1_part_aa..ad` (3 x 48.3 GB + 25.3 GB), `iSign-videos_v1.1_part_aa..ab` (32.2 + 25.7 GB).
  Licences on the dataset cards: CISLR `afl-3.0`, iSign `cc-by-nc-sa-4.0`.

### 5.2 Source adapters (`backend/app/isl/data/sources.py`)

```python
@dataclass(frozen=True)
class Clip:
    clip_id: str                 # "include/Seasons/63._Winter/MVI_4997", "islrtc/W/Winter.mp4", "cislr/<uid>", "mysign/<sign_id>/2"
    source: str                  # include | islrtc | cislr | isign | mysigns
    raw_gloss: str               # as in the source
    concept_id: str | None       # after vocabulary linking (None = unlinked)
    variant_id: str | None
    category: str | None
    path: Path                   # RawSequence .npz
    start: int = 0               # frame slice applied on load (ISLRTC trims)
    end: int | None = None
    group: str | None = None     # signer / pseudo-signer / session id used for grouped splits
    fps: float | None = None
    licence: str = ""
    quality: dict = field(default_factory=dict)   # hands_frac, n_frames, trim confidence...

    def load(self) -> RawSequence: ...          # RawSequence.load(path).slice(start, end)

class SourceAdapter(Protocol):
    name: str
    licence: str
    commercial: bool
    def clips(self, *, only_ok: bool = True) -> Iterator[Clip]: ...
    def available(self) -> bool: ...

class IncludeAdapter:    # reads data/isl/manifests/include.jsonl (+ splits/official/*.txt, splits/include_groups.json)
class IslrtcAdapter:     # reads islrtc.jsonl + data/isl/derived/islrtc_trim.jsonl; skips n_frames==0, kind in {explanation}
class CislrAdapter:      # reads data/isl/manifests/cislr.jsonl
class ISignAdapter:      # reads data/isl/manifests/isign.jsonl (research profile only)
class MySignsAdapter:    # reads data/my_signs/<pack>/signs.jsonl (never used in global training without an explicit opt-in)

def load_profile_clips(profile: DataProfile) -> list[Clip]: ...
```

Rules:

* Adapters are read-only over the manifests and tolerate a manifest that is still growing (read line by line, ignore a
  partial last line, keep the last record per `id`, use only `ok: true`).
* INCLUDE id mapping: the manifest `id` is `include/{category}/{word_dir}/{stem}` with spaces → `_`. The official split paths
  map by their last three path parts with `Extra/` removed (`Category/NN. Word/MVI_xxxx`), compared after the same
  space → `_` replacement. Unmatched paths are listed in the split report, never silently dropped.
* Quality filter (`Clip.quality`): `n_frames >= 8`, `hands_frac >= 0.2`, at least one frame with pose (else `FeatureError`).

### 5.3 ISLRTC first-sign refinement (`backend/app/isl/data/trim.py`)

Problem: each dictionary video is "sign, then (often) an explanation". The ingestion job cut at the title card when it
could; many clips (`stop=max_s`) still contain the explanation. We keep the `.npz` untouched and write a **sidecar** of
frame ranges, so the trimming can improve without re-running MediaPipe.

Stage 1, heuristic (no training needed):

```python
@dataclass(frozen=True)
class TrimConfig:
    rest_below: float = 1.1        # wrist y (body frame, shoulder-width units, y down) below this = hand at rest
    v_move: float = 0.6            # wrist speed (units/s) above which a hand is "moving"
    v_hold: float = 0.25           # speed below which both hands are "holding"
    min_active_ms: float = 250.0   # first active run must last at least this long
    min_sign_ms: float = 400.0
    pause_ms: float = 450.0        # a hold this long after min_sign_ms ends the sign
    rest_ms: float = 200.0         # both hands down/absent this long ends the sign
    max_sign_ms: float = 4000.0
    tail_ms: float = 150.0
    version: str = "trim-1"

@dataclass
class TrimResult:
    start: int; end: int                 # frame slice [start, end)
    t_start_ms: float; t_end_ms: float
    method: str                          # "rest" | "pause" | "source_end" | "cap" | "empty"
    confidence: float                    # rest 0.9, pause 0.7, source_end (stop=card|eof) 0.8, cap 0.3
    alternatives: list[int]              # up to 2 other plausible end frames (velocity minima)
    notes: list[str]

def activity_signal(raw: RawSequence) -> dict[str, np.ndarray]:
    """Per frame: up_left/up_right (present and wrist above rest line), speed (max of the two wrists, smoothed over
    ~150 ms, using t_ms), shape_change (mean fingertip displacement in the wrist frame), present."""

def candidate_boundaries(sig: dict[str, np.ndarray], t_ms: np.ndarray) -> list[int]:
    """Local minima of (speed + shape_change) after the first active run - plausible sign ends."""

def find_first_sign(raw: RawSequence, stop: str | None = None, cfg: TrimConfig = TrimConfig()) -> TrimResult: ...
```

Algorithm:

1. `to_body_frame(raw)` (gap filling included); active(t) = any hand up **or** speed > `v_move`.
2. Start = first run of active frames lasting `min_active_ms`, minus 2 frames of pre-roll.
3. End = earliest of, after `start + min_sign_ms`:
   * **rest**: both hands absent or below `rest_below` for `rest_ms` → end at the start of the rest + `tail_ms`;
   * **pause**: speed and shape change below `v_hold` for `pause_ms` (the signer pauses before explaining) → end at pause start + `tail_ms`;
   * **source_end**: the clip ends (`stop` is `card` or `eof`) → keep everything up to the last hand frame;
   * **cap**: `max_sign_ms` reached → end at the deepest velocity minimum in the last 40% of the window.
4. `alternatives` = the next two candidate boundaries, used for training augmentation and for indexing (5.3, stage 3).

Stage 2, learned refinement (`backend/app/isl/data/boundary.py`), run once stage 1 exists:

* Training data is built **synthetically** from what we already know is clean: ISLRTC clips with `stop in {card, eof}` and
  method `rest` with confidence ≥ 0.9, and INCLUDE clips (one sign each). Each example = clean sign A, then a 0–300 ms
  transition, then the first 2–6 s of a `max_s` clip B with its first sign removed (an explanation-like tail). Label = the frame where A ends.
* Model: 1-layer BiGRU (hidden 64) over per-frame activity features + the regular feature vector at 12 fps,
  predicting a boundary probability per frame; decode as the first peak above 0.5 after `min_sign_ms`, constrained to
  stage-1 candidate boundaries. About 10 minutes of CPU training.
* Output overrides stage 1 only when it agrees with a candidate boundary; `method = "learned"`.

Stage 3, robustness instead of perfection:

* Training samples the end frame from `{end} ∪ alternatives` (weights 0.6/0.2/0.2): the encoder learns to tolerate trim errors.
* The index stores **two** embeddings per ISLRTC entry (best end and best alternative) as two rows of the same variant; scoring takes the max.

Sidecar file `data/isl/derived/islrtc_trim.jsonl` (one JSON object per clip):

```json
{"id": "islrtc/A/Abdicate.mp4", "start": 10, "end": 41, "t_start_ms": 833.3, "t_end_ms": 3416.7,
 "method": "pause", "confidence": 0.7, "alternatives": [52, 63], "trim_version": "trim-1"}
```

Checks (in `tests/isl/test_trim.py` and a report): on synthetic "sign + rest + explanation" sequences the end is within ±2
frames; on INCLUDE clips the trimmer keeps ≥ 90% of the active frames; on ISLRTC `card/eof` clips its end is within
300 ms of the true end for ≥ 80% of clips; a review sheet `artifacts/reports/trim_review/*.png` (activity signal,
wrist height and chosen cut for 60 random clips, 20 per method) is generated for a visual check.

### 5.4 Splits (`backend/app/isl/data/splits.py`)

Split files are JSON under `data/isl/splits/` (local, derived, not committed):

```json
{"name": "include-pseudo-signer-cv5-fold0", "kind": "pseudo_signer", "signer_independent": "estimated",
 "seed": 0, "created": "2026-12-01T10:00:00Z", "source_manifest_sha256": "...",
 "assign": {"include/Seasons/63._Winter/MVI_4997": "train", "...": "test"},
 "leakage": {"group_overlap": {"train-test": 0, "train-val": 0}, "duplicate_overlap": 0, "classes_missing": {"test": []}}}
```

| Split | Construction | Label in reports |
|---|---|---|
| `include-official` | AI4Bharat `include_{train,val,test}.txt` | "official split, **not signer-independent** (same recording session in train and test)" |
| `include-session-cv5` | **Sessions** = runs of consecutive camera file numbers for one word (gap ≤ 3, e.g. MVI_4573–4576). Whole sessions assigned to folds, grouped 5-fold like `splits.signer_kfold` | "session-independent" |
| `include-pseudo-signer-cv5` (**primary**) | Sessions clustered into **pseudo-signers**: per session, a body-proportion descriptor from landmarks (upper-arm/forearm/shoulder-width ratios, hand size/shoulder width, face-subset geometry ratios, all scale-free medians); agglomerative clustering (average linkage, cosine), number of clusters chosen by silhouette in a range 4–12, plus a must-link prior for sessions with nearby file numbers recorded on the same day (number distance ≤ 30 across words). Whole pseudo-signers held out per fold | "estimated signer-independent (pseudo-signers from body proportions; not verified identities)" |
| `oneshot-concept-cv5` | Section 5.5 | "cross-dataset one-shot" |

Leakage checks reuse the logic of `splits.leakage_report` (group overlap, duplicates, missing classes) on the new group keys,
plus the cross-source duplicate check below. The clustering also writes `data/isl/splits/include_groups.json`
(`clip_id → {session, pseudo_signer, silhouette}`) and a short report of cluster sizes; if silhouette < 0.15 the report says
the clustering is weak and the session split becomes the primary one.

Duplicate detection (`backend/app/isl/data/dedup.py`): CISLR is collected from public online ISL videos and may contain the
ISLRTC dictionary videos themselves. Before any evaluation that uses CISLR, pairs across sources with near-identical
landmarks are linked:

```python
def fingerprint(raw: RawSequence) -> np.ndarray: ...   # 16-step resampled wrist tracks + mean hand shapes, normalised
def find_duplicates(clips: list[Clip], threshold: float = 0.02) -> list[tuple[str, str, float]]:
    """Candidate pairs by fingerprint cosine > 0.98 (vectorised), confirmed by DTW distance on the
    body-frame wrist trajectories below `threshold`."""
```

Duplicates are merged into one group (they are always on the same side of every split) and recorded in
`data/isl/derived/duplicates.jsonl`.

### 5.5 Cross-dataset one-shot protocol (open-vocabulary accuracy)

Question answered: "If a word exists only once in the dictionary (one ISLRTC video by one signer), how often does the app
recognise it when **someone else** signs it?"

* Overlap: an early check matched about 200 of INCLUDE's 263 words to an ISLRTC entry by exact normalised name and about
  40 more through synonym lists ("big large", "store or shop", "Pencil" has 7 ISLRTC variants). The vocabulary linking (6.1)
  produces the final list `data/isl/splits/oneshot_concepts.json`; links marked `needs_review` are excluded until reviewed.
* Folds: linked concepts are split into 5 folds (`oneshot-concept-cv5`), stratified by INCLUDE category.
* For fold k:
  * **Encoder training excludes every clip of fold-k concepts** from INCLUDE and CISLR, and excludes the fold-k ISLRTC clips
    from the supervised losses. (They stay in the gallery; they may appear only as an unlabeled SSL view if self-supervised pretraining is used,
    and the report says so.)
  * Gallery = the full ISLRTC index (all ~12k word concepts, all variants, + CISLR when available and deduplicated).
  * Queries = all INCLUDE clips of fold-k concepts (signers from Chennai, never the ISLRTC signers).
  * Report top-1/top-5 over (a) the full gallery and (b) a restricted 263-concept gallery, both "unseen class, unseen signer".
* Also reported: **seen-class retrieval** (concepts whose INCLUDE clips were in training) to show the gap.
* Honest caveat stated with the number: some INCLUDE signers may use a regional form that differs from the ISLRTC form,
  so part of the error is a variant mismatch, not a model error. If a reviewer marks 50 random linked words as
  "same form / different form", the report gives the number on the "same form" subset as well.

### 5.6 Vocabulary linking inputs per source

| Source | Raw gloss example | Parsed |
|---|---|---|
| ISLRTC file name | `Abhor Despise Detest Hate Loathe.mp4`, `Pencil_(Sign_3).mp4`, `AB Testing- English.mp4`, `Breathing_Rate_(Academic).mp4`, `Over_(Indore_Madhya_Pradesh).mp4`, `Numbers/12_Twelve.mp4`, `Minute_(Explanation).mp4` | synonyms; variant 3; drop "- English" marker; domain "Academic"; region "Indore, Madhya Pradesh"; number 12; kind explanation (excluded) |
| INCLUDE folder | `Adjectives/1. loud`, `People/.../big large`, `store or shop`, `race (ethnicity)`, `second (number)`, `you (plural)` | headword; synonyms; sense |
| CISLR `dataset.csv` | word column (format confirmed once access is granted) | same normaliser |

### 5.7 Gated datasets: access check and streaming ingestion

`backend/app/isl/data/remote.py`:

```python
HF_BASE = "https://huggingface.co/datasets"
PROBE_FILES = {"Exploration-Lab/CISLR": "dataset.csv", "Exploration-Lab/iSign": "iSign_v1.1.csv"}

def hf_access(repo: str) -> str:
    """'granted' | 'pending' | 'error:<code>'. Sends a 1-byte Range GET for PROBE_FILES[repo] (README.md is public
    even without access, so it cannot be used). HF auth is injected by the proxy; no token is read, printed or stored."""

class RangeFile(io.RawIOBase):
    """Seekable read-only file over one or more HTTP URLs treated as one concatenated byte stream
    (split archives part_aa, part_ab, ...). Reads with Range requests, retries with backoff, keeps an LRU
    block cache (default 64 MB). Sizes come from the HF tree API."""
    def __init__(self, urls: list[str], sizes: list[int], block: int = 8 << 20, cache_blocks: int = 8): ...
```

`scripts/ingest_isl_gated.py {cislr|isign} [--workers N] [--limit N] [--check]`:

* `--check` prints `granted`/`pending` per repo and exits 0/3 (for a scheduled re-check).
* **CISLR**: download the three CSVs to `data/isl/raw/cislr/` (small). Open the 1.17 GB zip through `RangeFile` +
  `zipfile.ZipFile` (the central directory is read from the end with range requests), then for each member: stream it to a temp
  file, extract with `extract_full` from `scripts/ingest_isl.py` (same extractor, 15 fps), delete the temp file, append to
  `data/isl/manifests/cislr.jsonl` (`id, source, gloss, uid, category, path, n_frames, fps, hands_frac, ok, error,
  prototype|test flag`). Resumable per member through the manifest. Peak disk use: one video.
* **iSign poses** (research profile only): the four parts form one archive. Detect the format from the first bytes of
  `part_aa` (`ustar` at offset 257 → tar; `PK\x03\x04` → zip; `\x1f\x8b` → gzip). Tar: iterate headers through `RangeFile`
  and record `(member name, header offset)` in `data/isl/manifests/isign_offsets.jsonl`, so a restart resumes at the last
  offset; zip: random access from the central directory; gzip: use `indexed_gzip` seek points exported to
  `data/isl/raw/isign/gzip.idx` so a restart can seek instead of re-reading. Each `.pose` member (pose-format, MediaPipe Holistic
  layout) is converted by `pose_to_raw(buf) -> RawSequence`: pose 33 (x, y, z, confidence), hands 21+21, face 468 → `FACE_IDX` subset,
  pixel coordinates divided by the header width/height, `extractor_version = "isign-pose-format"`. Only sentence ids listed in
  the chosen subset (default: 10,000 sentences that contain at least one vocabulary word according to
  `word-presence-dataset_v1.1.csv`) are converted; everything else is skipped without being written to disk.
  The video parts are **not** downloaded.
* Both jobs: `nice -n 10`, at most 2 workers while the other ingestion jobs run, never more than 2 GB of temp data.

Backward-compatible additions to `scripts/ingest_isl.py` (only take effect on the next start; existing manifests stay valid):

* `--only ID [ID ...]`, `--max-s SECONDS`, `--out-subdir NAME`, `--full` (no title-card cut) to re-extract specific long entries
  (the ISLRTC "Alphabet", "Fingerspell", "Numbers_(0-1_Crore)" and regional number videos are longer than the 10 s cap)
  into `data/isl/landmarks/islrtc_full/` with manifest `data/isl/manifests/islrtc_full.jsonl`.
* A `trim_version`-free manifest: trimming lives only in the sidecar (5.3), so the ingestion format does not change.

### 5.8 Implementation notes and deviations (Data Engineer, 25 Nov 2026)

Implemented in `backend/app/isl/{paths,profiles}.py`, `backend/app/isl/vocab/*`, `backend/app/isl/data/*`,
`backend/app/isl/cli.py` (data commands), `scripts/ingest_isl_gated.py` and the new `scripts/ingest_isl.py` options.
Measured numbers and licences are in `docs/DATASETS_ISL.md`. Deviations from sections 5.1-5.7 and 6.1, with reasons:

* **ISLRTC is not single-signer.** Watching sampled dictionary videos showed at least six different signers
  (one per entry). Adapters still put all ISLRTC clips in one group (`islrtc-signer-1`) because identities are
  unknown; this only matters if ISLRTC clips are ever used as test queries (they are not: section 5.5 queries are INCLUDE).
* **Most ISLRTC videos have no title card in the first seconds**: the signer signs the word and then explains it
  without a cut, often without lowering the hands to the sides. Stage 1 (5.3) therefore gained:
  * a `dip` boundary: both wrists drop below the waist line (`dip_below = 0.7` shoulder widths under the shoulder
    line) for `dip_ms = 150` ms - the usual short move between the sign and the explanation;
  * `active_above = 0.75` for the start: hands clasped at the waist (a common resting pose here) are not "up";
  * bounded clips (`stop` = `card`, `eof` or `+posegap`) normally hold the sign only, so they keep everything up to
    the **final** return of the hands (`source_end`); earlier rests/dips become `alternatives` (compound signs such as
    "Account Closing" drop the hands between their parts). If such a clip is longer than 1.25 x the sign window it
    is treated like an unbounded one;
  * a `pause` (both hands still) is only a fallback when no rest/dip exists: on reviewed videos pauses mostly fired
    on held hand shapes inside the sign;
  * multiword names get a longer minimum and maximum sign length (+600 ms / +1.5 s per extra word, capped);
  * methods are `rest | dip | pause | source_end | clip_end | cap | full | empty` (+ `learned` from stage 2).
* **Stage 2 (`boundary.py`)** uses the per-frame activity features only (not the full feature vector) to keep
  training to about a minute on CPU. A smoke run on partial data changed little (see DATASETS_ISL.md), so the sidecar
  keeps stage 1 by default and `trim --learned` is opt-in.
* **Vocabulary linking**: automatic merges are `exact`, `spelling` (British/American pairs in one WordNet synset),
  `qualified` (the dictionary has exactly one domain-qualified entry, e.g. INCLUDE "tight" -> "Tight (Clothes)") and
  `synonym` (member of an ISLRTC synonym-list name); the last two are `needs_review`. **WordNet synonyms are never
  merged** (a trial produced bird -> skirt, I -> one, blind -> screen): they, ambiguous domain matches, token matches
  and different senses ("you (plural)" vs "you") become `see_also` suggestions on a separate concept for review.
  Numbers use the word form as concept id (`Numbers/12_Twelve.mp4` -> `isl.twelve`, `number_value` 12).
  `vocabulary.jsonl` omits defaults/nulls (7.5 MB instead of 11.5 MB); `VocabEntry.model_validate` restores them.
* **INCLUDE ids**: the running ingestion job builds ids from the last path parts, which is wrong for the 27 videos in
  `Extra/` folders. Adapters rebuild canonical ids from the zip member path (`Extra/` removed); `ingest_isl.py` now
  writes the right gloss/category for them but keeps the old id formula so a restart never processes a video twice.
* **Pseudo-signer must-link prior**: chaining every session within 30 camera numbers merged consecutive recording
  passes of different signers. Sessions are now chained into *recording passes* in camera-number order while the gap is
  <= 30 **and no word repeats** (a signer records each word once per pass); passes are clustered. If the clustering is
  weak or gives fewer / very uneven groups than folds, `splits` reports the session split as primary.
* **iSign poses are one ZIP** split into four parts (dataset card), so the primary mode is random access through
  `RangeFile` (central directory from the end of part_ad, one range request per wanted member). The design's tar/gzip
  offset log is kept as the `--mode stream` fallback (piped parts; resumable at recorded offsets for zip/tar).
* **Access check**: verified on 25 Nov 2026 - README returns 206 for a range request on both repos while the probe data
  files return 403, so `hf_access` reports `pending` for both.

---

## 6. Vocabulary, models and fusion (tasks 2 and 4)

### 6.1 Unified ISL vocabulary

Three levels:

* **headword**: the English text shown to the user ("pencil").
* **concept** (`concept_id`): one meaning (headword + optional sense or domain). Recognition output and the Hindi entry are per concept.
* **variant** (`variant_id`): one concrete form of the sign (ISLRTC "Sign 2", a regional form, a user's form, the INCLUDE form).

Normalisation (`backend/app/isl/vocab/normalize.py`):

```python
@dataclass
class ParsedName:
    headword: str                 # "pencil"
    synonyms: list[str]           # ["abhor", "despise", "detest", "hate", "loathe"] (first = headword)
    sense: str | None             # "number" in "second (number)"
    domain: str | None            # "academic" | "math" | "science" | "cricket" | ...
    region: str | None            # "Indore, Madhya Pradesh" (from a gazetteer of Indian states/cities)
    sign_no: int                  # 1 by default; "(Sign_3)" / "(Sign 3)" / "(3)" → 3
    kind: str                     # word | multiword | synonym_list | phrase | number | letter | explanation | example
    number_value: int | None      # Numbers/12_Twelve → 12
    flags: list[str]              # "english_marker", "typo_fixed", "needs_review"

def parse_islrtc_name(rel_path: str) -> ParsedName: ...
def parse_include_word(word_dir: str) -> ParsedName: ...   # "63. Winter" → "winter"
def slugify(text: str) -> str: ...                         # "AC to DC Converter" → "ac-to-dc-converter"
def split_synonyms(tokens: list[str]) -> list[str] | None: ...
```

Rules, in order (counts are from the 13,665-file index):

1. `_` → space, collapse whitespace, Unicode NFC, trim. Keep the original string in `sources`.
2. Variant markers `(Sign_N)`, `(Sign N)`, `(N)`, `(Sign)`, `(Sign_1)` → `sign_no` (more than 1,100 files).
3. Trailing `- English` (about 300 files) → flag only (the clip is the sign for the English word).
4. `(Explanation)`, `(Explanation_2)`, misspelled `(Explaination)`, `(Sign_and_Explanation)`, `(Example)` → kind `explanation` /
   `example` (about 117 + 18 files): excluded from the recognition index, kept in the vocabulary for reference.
5. Parenthesised region from a gazetteer (`Bihar`, `Gujarat`, `Hyderabad`, `Mumbai Maharashtra`, `West Bengal`,
   `Thiruvananthapuram Kerala`, `Indore`, `Indore Madhya Pradesh`, `Andhra Pradesh Telangana`, `Odisha`, ...) → `region`
   (e.g. `Over_(Indore_Madhya_Pradesh)`, `Test_(Indore)`: real Indore variants already in ISLRTC).
6. Other parentheses (about 500 distinct, e.g. `Academic` 183, `Math`/`Maths` 149, `Science`, `Cricket`, `Financial Term`)
   → `domain` (with an alias table: `maths → math`, `admin → administrator`). A domain-tagged entry is its own concept
   (`isl.letter~administrator`), grouped with the plain headword for search and display.
7. Numbers folder: `NN_Words` → kind `number`, `number_value`; non-numeric names in it (`3G`, `404 error`, `120 Volt`) → normal words.
8. Synonym lists: names like `Abhor Despise Detest Hate Loathe`, `Able Can`, `Abort Cancel`, `Afraid Scared`,
   `Minimum Least Lowest`. Decision with WordNet (NLTK data, permissive licence): if the whole string (spaces → `_`) is a WordNet
   lemma or a known multiword term → `multiword`; else if it can be partitioned into ≥ 2 WordNet lemmas that pairwise share a synset
   or are linked by `similar_tos`/hypernym distance ≤ 2 → `synonym_list`; else `multiword`. INCLUDE-style `x or y` is always a synonym list.
9. ≥ 5 tokens and not a synonym list (proverbs such as "A penny saved is a penny earned"; about 500 long names) → `phrase`:
   kept, indexed with the full source clip (no first-sign trim), lower prior.
10. Acronyms stay upper case for display (`ATM`, `AGM`, `ASCII`); everything else lower case for matching, title case for display only where the source was a proper noun (gazetteer/NER list).
11. `overrides.yaml` (tracked) wins over all rules: `{match: "<source path or regex>", set: {...}}`, used for typos and wrong splits.

Concept id: `isl.<slug(headword)>[~<slug(sense|domain)>]`, e.g. `isl.pencil`, `isl.second~number`,
`isl.abhor-despise-detest-hate-loathe`. Ids are stable across rebuilds (a `concept_aliases.json` maps renamed ids).
Variant id: `<source>:<source-specific id>` (`islrtc:P/Pencil_(Sign_3)`, `include:Animals/6._Mouse`, `cislr:<uid>`, `mysign:<uuid>`).

Linking INCLUDE/CISLR words to ISLRTC concepts: exact headword or synonym match → `link: "exact"`; match through a synonym
list or a WordNet synonym → `link: "synonym", needs_review: true`; sense conflicts (`second (number)` vs `second (time)`)
resolved in `overrides.yaml`. INCLUDE clips become their own variant (`include:<word_dir>`) under the linked concept, because
INCLUDE signers (Chennai) may use a different form than ISLRTC. After encoder v1, a variant-consistency report flags concepts
whose INCLUDE centroid is far from every ISLRTC variant (cosine < 0.3): "possible regional variant, check".

`vocabulary.jsonl` (one concept per line, sorted by `concept_id`; row number = `class_index`):

```json
{"concept_id": "isl.pencil", "headword": "pencil", "display": {"en": "pencil"},
 "synonyms": [], "sense": null, "domain": null, "kind": "word", "category": "education",
 "pos": "noun", "fingerspell": "PENCIL",
 "variants": [
   {"variant_id": "islrtc:P/Pencil", "source": "islrtc", "sign_no": 1, "region": null,
    "clips": ["islrtc/P/Pencil.mp4"], "trim": {"method": "rest", "confidence": 0.9}},
   {"variant_id": "islrtc:P/Pencil_(Sign_2)", "source": "islrtc", "sign_no": 2, "region": null, "clips": ["islrtc/P/Pencil_(Sign_2).mp4"]},
   {"variant_id": "include:<Category>/<NN>._Pencil", "source": "include", "region": "Chennai (INCLUDE signers)", "clips_count": 18, "link": "exact"}
 ],
 "sources": [
   {"dataset": "islrtc", "id": "islrtc/P/Pencil.mp4", "licence": "GODL-India",
    "url": "https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded/resolve/main/P/Pencil.mp4"},
   {"dataset": "include", "word_dir": "<Category>/<NN>. Pencil", "licence": "CC-BY-4.0", "url": "https://zenodo.org/records/4010759"}
 ],
 "in_classifier": true, "review": {"status": "auto", "notes": []}}
```

(The INCLUDE category and word-directory number are copied from the source; `<Category>/<NN>` above are placeholders.) Expected size: about 12,000 concepts
(12,073 distinct names after rules 2–6 in a first pass; 1,134 with more than one variant), 4–6 MB, committed under
`models/packs/isl/` because the app needs it without local data. Hindi is **not** stored in this file; it is joined from the
lexicon (section 8) so translations can be reviewed independently.

Category: ISLRTC has no categories; INCLUDE's 15 categories are kept for linked concepts; other concepts get a coarse
category from WordNet lexnames (`noun.animal → animals`, `noun.food → food`, ...) or `other`. Used for filtering and per-category metrics.

### 6.2 Features and canonical handedness

* Same `FeatureConfig` and `build_features` path; the ISL default is `seq_len=48, use_face=true, use_velocity=true` (ISL has many
  face-contact signs; the face block is ablated in phase 2: E1 with and without face).
* New opt-in flag in `FeatureConfig`: `canonical_hand: bool = False`. When true, `build_features` mirrors the body frame
  (same permutation as `augment`'s mirror) whenever the **left** hand is dominant, dominance = larger summed wrist path length
  (ties → no mirror). This removes left/right-handed differences deterministically, which matters for one-shot retrieval
  where augmentation cannot cover the reference side. Default false, so ASL is unchanged and `FeatureConfig.from_dict` of old
  checkpoints still works. The TypeScript port `frontend/src/static/features.ts` must implement it before ISL runs in the static demo.
* Opt-in `hand_size_floor: float = 0.0` (added 26 Nov 2026, 15.14): lower bound, in shoulder widths, on the 2D
  wrist-to-middle-MCP length that normalises the hand-shape block. ISL uses 0.1; 0 keeps the original behaviour and is
  left out of `to_dict()`, so ASL configs and packs built before the change (no key) mean floor 0. The static TypeScript
  port must implement it too before ISL runs there.
* `augment_strong` (`backend/app/isl/augment.py`) composes the existing `augment` with: wider speed (±35%), rotation ±15°,
  scale ±15%, shift ±0.15, **arm-length scaling** about the shoulders (±10%, simulates other body proportions),
  per-finger joint-angle noise (σ 4°), contiguous hand dropout chunks (up to 4 frames), random 0–2 frame time shift between hands,
  face-subset dropout, and mirror p = 0.5 (only when `canonical_hand` is off).

```python
@dataclass(frozen=True)
class StrongAugmentConfig:
    base: AugmentConfig = AugmentConfig(p_mirror=0.0, rotate_deg=15.0, scale=0.15, shift=0.15, crop_frac=0.15,
                                        speed=0.35, landmark_dropout=0.08, jitter=0.012)
    arm_scale: float = 0.10
    finger_angle_deg: float = 4.0
    hand_chunk_drop_p: float = 0.2
    hand_chunk_max: int = 4
    face_drop_p: float = 0.1

def augment_strong(bf: BodyFrame, rng: np.random.Generator, cfg: StrongAugmentConfig = StrongAugmentConfig()) -> BodyFrame: ...
```

### 6.3 Models

**(a) Closed-set core classifier** (`backend/app/isl/training/train_classifier.py`)

* Classes: concepts linked to INCLUDE (about 255 after merging synonyms; exactly the set in `oneshot_concepts.json` plus
  unlinked INCLUDE words as their own concepts) + optional "core+" concepts that reach ≥ 4 clips across sources once CISLR arrives.
* Data: INCLUDE clips + every ISLRTC/CISLR clip of the same concepts (another signer: very useful), with `augment_strong`;
  ISLRTC clips are oversampled ×4 with different trim ends.
* Model: `build_model("gru", dim, n_classes, hidden=192, layers=2)`, i.e. the existing `RecurrentClassifier`. Checkpoint via
  `save_checkpoint` with meta `{..., "class_names": [concept_id...], "language": "isl", "profile": ..., "split": ...,
  "recommended_threshold": ..., "temperature": T}` so `Predictor` loads it unchanged.
* Selection on the pseudo-signer validation fold (macro F1), temperature scaling on validation (`calibration.fit_temperature`).

**(b) Sign encoder + nearest-neighbour index** (`backend/app/isl/models/encoder.py`, `losses.py`, `index.py`)

```python
class SignEncoder(nn.Module):
    """InputProjection → 2-layer BiGRU(hidden) → MaskedAttentionPool → Linear(2*hidden → emb_dim) → L2 normalise."""
    def __init__(self, in_dim: int, hidden: int = 192, layers: int = 2, emb_dim: int = 256, dropout: float = 0.3): ...
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor: ...   # (B, emb_dim), unit norm

class SupConLoss(nn.Module):           # supervised contrastive (Khosla et al. 2020), temperature 0.1
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...

class SubCenterArcFace(nn.Module):     # k sub-centres per concept capture variants; s=30, m=0.2
    def __init__(self, emb_dim: int, n_classes: int, k: int = 3, s: float = 30.0, m: float = 0.2): ...
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
```

* Loss = SupCon + 0.5 × SubCenterArcFace over all training concepts (about 12k classes; the ArcFace weight matrix is 12k × 3 × 256,
  about 9M parameters, fine on CPU).
* Positives: (1) INCLUDE clips of the same concept from other sessions/pseudo-signers (most valuable), (2) cross-source pairs
  (INCLUDE ↔ ISLRTC ↔ CISLR of one linked concept), (3) for single-clip concepts, two strong augmentations of the same clip
  with different trim ends (SimCLR-style). Batch sampler `PKSampler(P=32 concepts, K=2..4)`, half of P drawn from multi-clip concepts.
* Research profile only: warm-start the trunk from (i) WLASL100 local clips (classification pre-training, 524 clips) and
  (ii) a 10,000-sentence iSign pose subset with temporal-crop contrastive pre-training (two crops of the same sentence
  are positives). Recorded in the checkpoint meta `pretraining`.
* Index (`index/` in the pack):

```
index/embeddings.f16.npy   (N, 256) float16, L2-normalised; N ≈ 2 rows × ~12k ISLRTC variants + INCLUDE/CISLR centroids
index/rows.jsonl           {"row": 0, "variant_id": "islrtc:P/Pencil", "concept_id": "isl.pencil", "source": "islrtc",
                            "kind": "first_sign|alt_end|centroid", "weight": 1.0}
index/meta.json            {"encoder_version": "isl-enc-1", "dim": 256, "n": 24311, "created": "...",
                            "feature_config": {...}, "profile": "commercial-safe", "sha256": "...",
                            "contains_eval": true, "encoder_sha256": "..."}   (last two: 15.21)
```

```python
class EmbeddingIndex:
    @classmethod
    def load(cls, folder: Path, expect_encoder_version: str) -> "EmbeddingIndex": ...   # refuses a version mismatch
    def search(self, q: np.ndarray, k: int = 50) -> list[list[tuple[int, float]]]: ...  # q (B, D); cosine via one matmul
    def concept_scores(self, q: np.ndarray, k: int = 50) -> list[dict[str, float]]: ...  # max over rows per concept
```

Brute-force cosine over ~25k × 256 is one small matrix product (well under 5 ms on CPU); no FAISS or other dependency.
Queries are embedded for the 3 engine crops (and the mirrored query when `canonical_hand` is off) and averaged.

**(c) Fingerspelling** (`backend/app/isl/fingerspell/`) - see 6.4.

**(d) Fusion, calibration, abstain** (`backend/app/isl/models/fusion.py`, `calibration.py`)

For one segment:

1. Classifier: `p_c = softmax(logits / T_c)` over the core concepts (mean over crops).
2. Retrieval: concept scores `s_k = max cosine` over the concept's rows; keep the top 50; `p_r = softmax(s / τ_r)`.
3. My signs (searched first, 7.3): user concept scores `u_k` with a bonus `b_u`.
4. Log-linear fusion over the union of candidates:
   `z_k = α·log p_c(k) [core only] + β·log p_r(k) + γ·prior_k`, `p_f = softmax(z)`,
   where `prior_k` is a log-frequency prior (INCLUDE membership, everyday-word list) so common words win close calls.
   `α, β, γ, τ_r, T_c` are fitted on validation by minimising NLL (grid search; tiny).
5. Out-of-vocabulary guard: if the best cosine to any row is below `s_min` (fitted so 95% of in-vocabulary validation
   queries pass), the result is `abstain` with reason `out_of_vocabulary`.
6. Decision:

```python
@dataclass(frozen=True)
class FusionParams:
    alpha: float; beta: float; gamma: float; tau_r: float; t_c: float
    s_min: float
    accept: dict[str, float]      # per policy: {"strict": 0.85, "balanced": 0.7, "lenient": 0.55}  (fitted, not these numbers)
    margin: float                 # p1 - p2
    suggest_min_top5_mass: float  # show "did you mean" only if the top-5 carry at least this much probability
    user_bonus: float
    version: str

def decide(p_sorted: np.ndarray, agree: int, crops: int, params: FusionParams, policy: str) -> tuple[str, str]:
    """accept   if p1 >= accept[policy] and p1 - p2 >= margin and agree >= 2 (of 3)
       suggest  elif sum(top5) >= suggest_min_top5_mass
       abstain  otherwise ("Not sure - please sign again, or spell it")."""
```

Thresholds are chosen on validation so that **accepted words are ≥ 95% correct** for `balanced` (≥ 98% for `strict`), and
`suggest` is shown only when the true word is in the top 5 for ≥ 80% of such cases. The achieved values are stored in
`fusion.json` and shown on the Evaluation page. If a target cannot be reached, the pack ships with the closest achievable value
and the UI shows the real precision, never the target.

`IslRecognizer.recognize(crops)` runs steps 1–6 and returns `Recognition`; target CPU latency per segment < 150 ms
(classifier ~10 ms per crop, encoder ~10 ms per crop, search < 5 ms).

### 6.4 Fingerspelling and spelling mode

Data, honestly scarce:

* ISLRTC "Alphabet" and "Fingerspell" videos, re-extracted full length (5.7), one signer; letters segmented by hold detection
  and labelled **by order** (the alphabet is signed A→Z, so 26 holds are aligned to 26 letters with a monotonic DP that allows
  skipping spurious holds); numbers 1–9 from `Numbers/` and the regional number videos.
* Any openly licensed ISL alphabet dataset found by the Data Engineer (licence recorded; only CC0/CC-BY/MIT-style licences
  enter the commercial-safe profile).
* Guided **alphabet calibration** in My signs (sign A–Z once): per-user prototypes; shared school packs add signers over time.
* Until at least 5 different signers are available, the module is labelled **experimental** in the UI.

Components:

```python
def letter_frame_features(raw: RawSequence, i: int) -> np.ndarray:
    """~150 values: both hands' wrist-relative, size-normalised shapes (2 × 60), right-minus-left wrist vector and
    10 fingertip-to-other-hand distances (two-handed ISL letters touch the other hand), presence flags."""

@dataclass
class Hold:
    start: int; end: int; mean_features: np.ndarray; motion: float

def detect_holds(raw: RawSequence, min_ms: float = 180.0, v_max: float = 0.35) -> list[Hold]: ...

class LetterClassifier(nn.Module):     # MLP 150→128→64→37 (A–Z, 1–9, 0, "noise")
    ...

class LexiconTrie:
    def __init__(self, words: Iterable[tuple[str, float]]): ...      # word → log prior

class SpellDecoder:
    def __init__(self, trie: LexiconTrie, beam: int = 16, ins_pen: float = -2.5, del_pen: float = -2.5): ...
    def decode(self, letter_logprobs: list[np.ndarray], k: int = 5) -> list[tuple[str, float]]:
        """Beam search over the trie with insertion/deletion penalties (missed or extra holds); also returns the
        raw letter string as a candidate marked 'spelled, not in dictionary'."""
```

Lexicon (`models/packs/isl/spelling_lexicon.txt`): vocabulary headwords + a curated list of Indian names, cities and
states (Indore, Bhopal, Madhya Pradesh, ...) + an English word list with a permissive licence (SCOWL size 50) with
frequency ranks. Letters that involve movement (for example J) are recognised from a 0.6 s window around the hold using the
velocity summary; per-letter accuracy is reported so weak letters are visible.

Engine `mode="spell"`: holds → letter posteriors (shown live in `state.spell.letters`); a hands-down gap of `spell_end_ms` ends
the word; the decoder's top-5 become the candidates; decision rules as in 6.3 with their own fitted thresholds. When a segment
is abstained in word mode, the UI offers "Spell it instead".

### 6.5 Per-pack results file

`models/packs/isl/results.json` contains, per protocol (section 9): numbers, n clips, n classes, n signers/groups,
95% intervals, data profile, git commit, date, and a one-line plain-language label. The UI reads only this file for ISL numbers.

### 6.6 Training loop reuse

Extract the epoch loop of `training/trainer.py` into a DB-free function, keeping `run_training` as a thin wrapper:

```python
@dataclass
class FitResult:
    best_epoch: int; best_metric: float; history: list[dict]; checkpoint: Path

def fit(model: nn.Module, train_loader: DataLoader, val_fn: Callable[[nn.Module], dict], loss_fn: Callable,
        cfg: TrainConfig, ckpt_path: Path, meta: dict, select_key: str = "val_macro_f1",
        progress: Callable[[float, str], None] | None = None, should_stop: Callable[[], bool] | None = None) -> FitResult: ...
```

`run_training` behaviour, history fields and checkpoint contents stay identical (covered by the existing tests).
ISL trainers call `fit` with their own datasets (`isl/training/common.py: ClipDataset(clips, fcfg, train, aug)` with an
in-memory cache of `BodyFrame`s in float32; about 1.2 GB for INCLUDE + trimmed ISLRTC).

### 6.7 Implementation notes and deviations (AI Engineer, models and My signs, 25 Nov 2026)

Implemented: `inference/recognizer.py`, the engine's `recognizer=` option, `packs/*` (registry, ASL and ISL packs,
runtime `Vocabulary`), `api/languages.py`, the language-aware WebSocket and analyze routes, `db/migrate.py`,
`training/trainer.fit`, `isl/augment.py`, `isl/models/*`, `isl/fingerspell/*`, `isl/mysigns/*`, a minimal
`isl/text/lexicon.py` (reader + local corrections only), `isl/recognizer.py`, `isl/training/*`, `isl/evaluation/*` and
the model commands of `isl/cli.py` (`train-classifier`, `train-encoder`, `build-index`, `train-letters`, `fit-fusion`,
`evaluate`, `build-pack`, `smoke`). Deviations from sections 4, 6 and 7, with reasons:

* **ASL stays byte-identical.** `FeatureConfig.to_dict()` leaves `canonical_hand` out when it is false (and
  `hand_size_floor` when it is 0), and the engine's
  `describe()` leaves the recogniser-only fields (`policy`, `use_my_signs`, `spell_end_ms`, `max_spell_ms`) out when there
  is no recogniser, so ASL checkpoints, `/api/features/spec`, `hello` and `config_ack` are unchanged (`hello` only gains
  `language` and `languages`). The ASL WebSocket keeps the original predictor engine path; `ClassifierRecognizer` reproduces
  the same rule for the language API and future packs.
* **`canonical_hand`** is applied in one place, `features.build.prepare_body_frame` (used by `build_features`, the training
  cache and the letter features), so training, offline analysis and live inference cannot diverge.
* **Pack ids**: `isl` (commercial-safe) and `isl-research`. `SLR_ISL_PROFILE` is the default profile of the build commands,
  not a runtime switch; both packs are listed when both are built. The tracked `models/packs/isl` holds `pack.json` and the
  vocabulary only (status `missing_models`); `build-pack` writes the full pack to `artifacts/packs/<id>` (git-ignored),
  which overrides it. No smoke weights are committed.
* **`missing_models` still recognises My signs.** Without a classifier and an index the pack builds a recogniser that only
  searches the user's signs (the UI message says so). Because a softmax over a few user signs cannot say "none of them",
  a user sign is accepted only when its cosine passes the sign's own threshold `theta` (lowest leave-one-out
  self-similarity at enrolment minus 0.05, floored at `s_min`); otherwise it is "suggest" or "abstain".
* **My signs without an encoder** use a model-free embedding (`FeatureStatsEmbedder`, version `featstats-1`: per-dimension
  mean and standard deviation of the feature sequence). Takes are re-embedded automatically when the pack's encoder appears.
* **Fusion details** not fixed by 6.3: the softmax is taken over the union of candidates (classifier top-50, retrieval
  top-50, all user signs); concepts outside the classifier get `noncore_logp = log(1e-3)`; concepts with no index row get the
  50th-best retrieval score as a neutral floor; the retrieval softmax includes user signs with `user_bonus` (logit units).
  When a precision target cannot be reached on validation, the threshold with the best measured precision is used, all
  policies may collapse to one value, and the measured numbers are stored in `fusion.json` `achieved` (the UI must show
  these, never the targets).
* **Fingerspelling data does not exist yet.** The ISLRTC `A/Alphabet.mp4` and `F/Fingerspell.mp4` entries are the signs for
  the *words* (about 3.8 s each, re-extracted in full into `data/isl/landmarks/islrtc_full/`), not A-Z sequences. So:
  (1) `train-letters` trains the letter MLP from in-order alphabet recordings when someone records them (monotonic
  alignment of holds to letters); (2) until then spelling uses a **per-user alphabet calibration** in My signs
  (`GET/POST/DELETE /api/languages/{lang}/my-signs/alphabet`, one recording per letter, nearest-prototype letter
  posteriors, at least 5 letters); (3) spelled words are always `suggest` (never auto-accepted) and the decoder prefers a
  lexicon word reachable by one edit over an out-of-lexicon string (`oov_pen`). The spelling lexicon is vocabulary headwords
  plus Indian states/cities with all Madhya Pradesh districts; SCOWL is not added yet.
* **Feedback**: `reject` is stored as a recognition event with `source: "feedback"`; `choose` commits are stored with
  `detail.chosen_by_user: true`. Neither is used for training.
* **Not done here** (owned by later tasks or phase 2): the sentence endpoint and Hindi generation (only the lexicon reader
  exists, so `text.hi` is null until `lexicon_hi.jsonl` is generated); conflict reports on sign-pack import (enrolment does
  check conflicts); research-profile pre-training (WLASL warm start, iSign contrastive); the per-fold held-out-concept
  one-shot protocol (E2 in the smoke run is the "seen-class" variant, labelled as such); E4/E5/E8 runners.
* **Test note**: the backend suite is CPU-bound; while the ingestion jobs saturate the CPUs, run it with
  `SLR_TORCH_THREADS=2` (otherwise torch's default threads fight MediaPipe's and GRU training slows down about 30x).

**Smoke run on partial data** (`python -m app.isl.cli smoke`, 2 threads, `nice -n 10`, 90 s in total including the
incremental trim, while ingestion used the CPUs; split `include-session-cv5-fold0`, session-independent, **not**
signer-independent): INCLUDE 623 clips of 51 words in the split (373 train / 125 val / 125 test; 168 newer clips not yet
in any split), ISLRTC 2,023 dictionary clips (entries A to about B), all but 5 trimmed. Classifier (hidden 64, 1 layer,
8 epochs, 11 s): 43 classes, val top-1 0.40, top-5 0.82. Encoder (hidden 64, 128-d, 4 x 20 batches, 29 s): 1,849
concepts, val recall@1 0.22, @5 0.51. Index: 3,239 rows. Fusion: no precision target is supported by the validation
data (balanced accept threshold 0.73 for every policy). Test fold: E1 top-1 0.50, top-5 0.79, macro F1 0.43 (n = 84);
E2 (INCLUDE vs ISLRTC rows only, seen-class) top-1 0.00 (n = 15); E3 coverage 0.06, accepted precision 0.63 (5 of 8),
silent error rate 0.02, suggest 0.57 (true word in the top 5 for 0.69 of them), abstain 0.37 (n = 125); E6 p50 114 ms,
p95 207 ms per segment under full CPU load (about 15-20 ms when the CPU is free). **These numbers only prove the pipeline
runs end to end; they are not the app's accuracy.** The results file carries the same label.

* **Thresholds use the 95 % Wilson lower bound** of accepted precision, not the point estimate: an earlier smoke run
  picked 0.85 because 5 of 5 validation clips above it were right, and then accepted 1 of 4 correctly on test.
  With the lower bound a target counts as reached only when the amount of data supports it.

---

## 7. Personal and regional signs: "My signs" (task 5)

### 7.1 Storage (local, git-ignored)

```
data/my_signs/<pack_id>/
  signs.jsonl                           one MySign per line (last line per sign_id wins; deletes are tombstones)
  takes/<sign_id>/<n>.npz               RawSequence of each take (landmarks only, never video)
  emb_<encoder_version>.npy             cached embeddings (rows in emb_<encoder_version>.rows.json)
```

```python
@dataclass
class MySign:
    sign_id: str                  # uuid4
    pack: str                     # "isl"
    concept_id: str | None        # existing concept this is a variant of, or None for a new word
    text_en: str
    text_hi: str | None
    region: str | None            # "Indore, MP"
    note: str | None
    n_takes: int
    origin: str                   # "recorded" | "imported:<pack title>"
    author: str | None            # free text, optional
    created: str
    enabled: bool = True
    deleted: bool = False

@dataclass
class AddResult:
    sign: MySign | None           # None for a dry run or when rejected
    take_quality: list[dict]      # sequence_quality() + hand ratio per take
    consistency: float            # mean pairwise cosine between takes
    conflicts: list[Candidate]    # global/user entries that look the same
    self_test: dict               # {"hits": 4, "n": 5}
    accepted: bool
    message: str                  # plain-language advice

class MySignsStore:
    def __init__(self, root: Path, pack_id: str): ...
    def list(self, include_disabled: bool = True) -> list[MySign]: ...
    def add(self, text_en: str, takes: list[RawSequence], *, text_hi: str | None = None, concept_id: str | None = None,
            region: str | None = None, note: str | None = None, dry_run: bool = False) -> AddResult: ...
    def update(self, sign_id: str, **fields) -> MySign: ...
    def delete(self, sign_id: str) -> None: ...
    def user_index(self, encoder: SignEncoder, encoder_version: str) -> EmbeddingIndex: ...   # re-embeds from takes on version change
```

Because embeddings depend on the encoder version, **takes are always stored as landmarks** and re-embedded when the
pack's encoder changes. The cache is rebuilt lazily on first use.

### 7.2 Enrolment rules

* 3 to 5 takes; each take must pass `sequence_quality` (`usable`) with hands in ≥ 50% of frames and 0.3–4 s duration.
* Consistency: mean pairwise cosine between takes ≥ `c_min` (fitted on INCLUDE same-session takes, about the 10th percentile);
  otherwise "Your recordings look different from each other. Please sign the same way each time."
* Conflict check: every take is searched in the global and user indexes. If another concept scores at or above the takes'
  own consistency, the user sees "This looks like '<word>' (ISLRTC dictionary). Save anyway as your own version?"
  Choosing an existing concept stores the sign as a variant of it (`concept_id` set).
* Self-test: leave-one-out, each take queried against the user index + global index; result "recognised 4 of 5".
* Hindi: taken from the lexicon when `concept_id` is set; otherwise typed by the user (marked `source: user`).

### 7.3 Use at recognition time

The user index is searched first. If the best user score ≥ `θ_user` (calibrated: the sign's lowest leave-one-out
self-similarity minus a margin, floored at the global `s_min`) and beats the best global concept by `margin`,
the user sign is the top candidate (`source: "my_signs"`, `origin: "My signs"`). Otherwise user scores join the fusion
with bonus `b_u`. Disabled signs are ignored. Turning `use_my_signs` off in the engine config ignores the whole user index.

### 7.4 Sign packs (export and import)

`GET /api/languages/isl/my-signs/export?ids=a,b,c&title=...&region=...&licence=CC-BY-4.0&include_face=false` → `application/zip`
named `<title>.signpack`:

```
manifest.json      {"format": "sign-pack", "version": 1, "pack": "isl", "title": "St. X School, Indore",
                    "region": "Indore, Madhya Pradesh", "author": "optional free text", "licence": "CC-BY-4.0",
                    "created": "...", "n_signs": 42, "encoder_version": "isl-enc-1", "app_version": "..."}
signs.jsonl        MySign rows (without local paths)
takes/<sign_id>/<n>.npz
embeddings.npy     optional fast path, used only if encoder_version matches
```

Import (`POST .../my-signs/import`, multipart; `mode=merge|replace`):

* Limits: ≤ 50 MB, ≤ 2,000 signs, ≤ 5 takes per sign, ≤ 300 frames per take.
* Safety: member names must match `^(manifest\.json|signs\.jsonl|embeddings\.npy|takes/[0-9a-f-]{36}/[0-9]\.npz)$`
  (no path traversal), `np.load(..., allow_pickle=False)`, every array validated through `RawSequence` shapes and finite values,
  text fields length-limited and stripped of control characters.
* Imported signs get `origin: "imported:<title>"` and the region; conflicts are reported like enrolment conflicts;
  takes are re-embedded if the encoder differs.
* Export consent text (shown before download): "This file contains body and hand movement points of the people who recorded
  these signs (no video). Share it only with their permission." Face points are left out by default (`include_face=false`);
  the recogniser handles missing face like any missing detection (presence flag 0).

Endpoints:

| Method and path | Body / query | Response |
|---|---|---|
| `GET /api/languages/{lang}/my-signs` | | `{"signs": [MySign...]}` |
| `POST /api/languages/{lang}/my-signs` | `{"text_en", "text_hi"?, "concept_id"?, "region"?, "note"?, "dry_run"?, "takes": [{"width","height","frames":[RawFrame...]}]}` | `AddResult` |
| `PATCH /api/languages/{lang}/my-signs/{sign_id}` | `{"text_en"?, "text_hi"?, "region"?, "enabled"?}` | `MySign` |
| `DELETE /api/languages/{lang}/my-signs/{sign_id}` | | `{"deleted": true}` |
| `GET /api/languages/{lang}/my-signs/export` | query as above | zip |
| `POST /api/languages/{lang}/my-signs/import` | multipart `file`, `mode` | `{"imported", "skipped", "conflicts": [...], "reembedded": bool}` |

All write endpoints are disabled when `public_demo` is set. The static (in-browser) demo keeps My signs in IndexedDB in a later phase.

---

## 8. Output: English, Hindi, sentences, speech (task 6)

### 8.1 Hindi lexicon

`models/packs/isl/lexicon_hi.jsonl` (tracked) plus local overrides `data/isl/lexicon_hi.local.jsonl` (written by the review UI):

```json
{"concept_id": "isl.pencil", "hi": "पेंसिल", "hi_alt": [], "translit": "pensil",
 "source": "manual", "model": null, "review": "reviewed", "reviewer": "teacher-1", "updated": "2026-12-03"}
{"concept_id": "isl.abhor-despise-detest-hate-loathe", "hi": "घृणा करना", "hi_alt": ["नफ़रत करना"],
 "source": "indictrans2", "model": "ai4bharat/indictrans2-en-indic-dist-200M", "review": "machine", "reviewer": null, "updated": "..."}
```

`review` values: `reviewed` (a Hindi speaker checked it), `machine` (generated, not checked), `disputed` (a reviewer flagged it), `missing`.

```python
class HindiLexicon:
    def __init__(self, base: Path, local: Path | None): ...      # local rows override base rows per concept_id
    def get(self, concept_id: str) -> dict | None: ...
    def set_local(self, concept_id: str, hi: str, hi_alt: list[str], review: str, reviewer: str | None) -> dict: ...
```

Generation (`python -m app.isl.cli hindi`), offline, phase 2:

1. **Core list first** (about 1,000 concepts: INCLUDE words, numbers, pronouns, greetings, days, family, food, school,
   question words, plus the most frequent everyday words): machine draft, then **human review** by a Hindi speaker through the
   Vocabulary page (owner or a teacher). These are the words people will see most.
2. Everything else: IndicTrans2 (AI4Bharat, en→indic distilled 200M model, run on CPU in batches; confirm its licence at
   integration time and record it in `docs/DATASETS_ISL.md`). Input text carries the context to choose the right sense:
   headword + synonyms + sense/domain, e.g. `"hate (abhor, despise, detest, loathe)"`, `"second (number)"`, `"bank (finance)"`;
   the output is post-processed to drop the context words.
3. Proper nouns (people, places, organisations; detected by the gazetteer/category) are transliterated, not translated.
4. Numbers: generated from a rule table (`12 → बारह`, `100000 → एक लाख`).
5. Nothing is invented at runtime: a concept without an entry shows English plus "Hindi not available yet / हिंदी उपलब्ध नहीं".

The UI shows a small marker next to machine Hindi ("machine translation, not yet checked") and lets a local user correct it
(`PATCH .../lexicon/{concept_id}`). Corrections can be exported and contributed back to the tracked file by a maintainer.

### 8.2 Sentence builder (`backend/app/isl/text/sentence.py`)

ISL word order is typically time–topic–comment, verb-final (subject-object-verb), with question words and negation at the end.
Hindi is also verb-final, so ISL order maps more naturally to Hindi than to English. The builder is rule-based, deterministic,
and **always marked approximate**.

```python
@dataclass
class Token:
    concept_id: str
    en: str
    hi: str | None
    pos: str | None            # from vocabulary "pos" + function_words.yaml
    role: str | None           # "wh" | "neg" | "time" | "pronoun" | None

@dataclass
class SentenceOut:
    text: str
    approximate: bool          # always True
    rules: list[str]           # rules applied, shown on hover ("moved question word to front")
    missing_hi: list[str]      # concepts without Hindi

def build_sentence(tokens: list[Token], lang: str) -> SentenceOut: ...
```

`function_words.yaml` (tracked) lists closed-class words with English and Hindi forms and roles: pronouns
(I/me → मैं/मुझे, you → आप/तुम, my → मेरा, ...), question words (what → क्या, where → कहाँ, who → कौन, when → कब,
why → क्यों, how → कैसे, how many → कितने), negation (not/no → नहीं, never → कभी नहीं), time words (today → आज, yesterday,
tomorrow, now, ...), and a copula (है/हैं).

English rules: (1) a final question word moves to the front and the sentence ends with "?"; (2) a final negation moves before
the verb (`not` / `do not`); (3) time words stay first; (4) with exactly one verb after one or more noun phrases, the verb moves
after the first noun phrase (SOV → SVO); (5) pronoun case (`ME` → "I" at subject position); (6) capital letter and full stop.
Hindi rules: (1) keep ISL order; (2) a final question word moves before the verb, or, with no verb, the copula is added
(`YOUR NAME WHAT` → "आपका नाम क्या है?"); (3) final negation moves before the verb; (4) "?" for questions, "।" otherwise.
If no rule applies, the output is the words joined with spaces. Example outputs and their rules are unit-tested.

### 8.3 Speech (frontend only)

`frontend/src/lib/speech.ts`:

```ts
export function voicesFor(lang: "hi-IN" | "en-IN"): SpeechSynthesisVoice[];
export function speak(text: string, lang: "hi-IN" | "en-IN", opts?: { rate?: number }): Promise<void>;
export function speechAvailable(lang: "hi-IN" | "en-IN"): boolean;
```

Optional and off by default (a "Speak" button per word and per sentence, and an "auto-speak accepted words" toggle).
Since 15.29 an ISL word joins the sentence, and so can be spoken, only after the user taps to confirm it.
If the browser has no `hi-IN` voice, the button explains that and offers `en-IN`. Nothing is sent to a server.

### 8.4 Implementation notes and deviations (AI Engineer, Hindi and sentences, 25 Nov 2026)

Implemented: `isl/text/hindi_build.py` (+ `data/hindi_core.tsv`), `isl/text/numbers_hi.py`, `isl/text/mt.py`
(IndicTrans2 fetch/verify/translate, offline only), the full `isl/text/lexicon.py`, `isl/text/sentence.py`
(+ `data/function_words.yaml`), the CLI command `hindi`, and the API fields below. Deviations, with reasons:

* **Row fields.** Rows carry `english`, `hindi`, `hi_alt`, `source`, `reviewed` (bool), `review`, `model` (the task asked
  for english / hindi / source / reviewed; `review` and `model` keep the richer states of 8.1). `translit` and
  `updated` are left out of generated rows (a rebuild would rewrite every date); local corrections keep `updated` and
  `reviewer`. The reader also accepts the older `hi` field.
* **Result (25 Nov 2026).** Every one of the 12,442 vocabulary concepts has a Hindi row: 870 `manual`/`curated`
  (core list), 20 `rule` (numbers), 11,552 `model`/`machine` (IndicTrans2, 11,829 model inputs, about 40 min on 2
  CPU threads at `nice -n 10` beside the ingestion jobs); 0 `reviewed`. ASL: 100 of 100 WLASL glosses (all core list).
  `lexicon_hi.jsonl` is 2.7 MB. A spot check of 100 random single-word machine rows **by the AI engineer, not a
  native speaker**: about 89 acceptable, 7 clearly wrong (e.g. "pus" -> पस instead of मवाद, "batter (beat)" ->
  बल्लेबाज, "laidback" -> आराम करें, "arraignment" -> नियुक्ति), 4 doubtful. Machine rows are therefore shown with the
  "machine translation, not yet checked" marker, and a teacher review of the most-used words is still needed.
* **New review state `curated`.** Rows from the hand-written core list (about 990 entries: greetings, family, numbers,
  days, months, colours, food, places incl. Indore/MP, school, body, medical/emergency, emotions, question words,
  common verbs and adjectives, and all 263 INCLUDE words) and from the number rules are careful but were written by
  the AI engineer, not confirmed by a native-speaker teacher. Calling them `reviewed` would overstate what was done,
  so `reviewed: true` is reserved for a person's confirmation (in the app, then `hindi --merge-local`).
* **Sources are `manual`, `rule` or `model`** (`rule` for numbers, step 4 of 8.1). Proper nouns are not transliterated
  separately: IndicTrans2 already transliterates names ("Indore" -> इंदौर, "Nagpur" -> नागपुर), and the important
  places (Indore, Bhopal, MP, Delhi, Mumbai, India) are in the core list.
* **Context for the model** (8.1 step 2) is given as English framing, because a bare word comes back as an imperative
  ("go" -> जाओ) and padded batches sometimes run on ("सुंदर । । ।"): verbs (WordNet POS) as "to <verb>" (the purpose form
  "<x>ने के लिए" is turned into the infinitive "<x>ना"), single nouns as "a <noun>" (a leading "एक" is dropped), a
  domain/sense as "<word> (<domain>)" (the echoed parenthesis is dropped), output cut at the first danda or "..",
  output length capped at 2 x input tokens + 8, `no_repeat_ngram_size=4`. A synonym list ("abhor despise detest hate
  loathe") is translated member by member and the majority answer wins (the others go to `hi_alt`): translating the
  whole list in one input produced "घृणा ( घृणा ) घृणा ...".
* **Model download.** The official `ai4bharat/indictrans2-en-indic-dist-200M` repository is "gated: auto" (Hugging Face
  asks the account owner to share contact details first). `mt.fetch_model` tries it first and otherwise downloads a
  public mirror, accepting a file only if its git object id equals the official one pinned in `mt.py` (for LFS files
  the pointer is rebuilt from the SHA-256 of the download), so the bytes are provably the official MIT release. The
  owner can accept the gate on the model page to use the official repository directly. The model's own Python code is
  loaded with `trust_remote_code` from that verified local folder only. `transformers` and `sentencepiece` are offline
  tooling (`backend/requirements-hindi.txt`), not app requirements.
* **ASL gets Hindi too.** `models/packs/asl-wlasl100/lexicon_hi.jsonl` covers all 100 WLASL glosses (all from the core
  list). The ASL pack exposes `lexicon()`; `ClassifierRecognizer` and the legacy predictor path of the engine add
  `text: {en, hi}` and `hi_review` to each top-k item and the commit. These keys are additive; without a lexicon the
  ASL messages are byte-identical to before. ASL `output_languages` is now `["en", "hi"]`.
* **build-pack** copies only the lexicon rows of the pack's vocabulary.
* **Sentence builder additions** beyond 8.2, each recorded in `rules`: topic-first signing ("WATER I WANT" -> "I want
  water"; a later pronoun becomes the subject); "YOU NAME WHAT" reads the pronoun as a possessive ("What is your
  name?"); "to" after a motion verb before a place ("I go to school"); English do-support for questions and negation;
  Hindi present-tense endings by subject (मैं -> ता हूँ, आप -> ते हैं, ...), feminine endings for a female subject
  (mother, sister, she, ... -> "माँ खाना पकाती है"), otherwise masculine assumed. Past and future tense are not
  inferred (ISL marks them with time signs; "YESTERDAY I DOCTOR MEET" stays "Yesterday I meet doctor").
* **API** (all additive): `GET /api/languages/{lang}/lexicon` (coverage, counts by source/review, review labels in
  English and Hindi, the "Hindi not available yet / हिंदी उपलब्ध नहीं" text, `?local=true` for local corrections),
  `GET /api/languages/{lang}/lexicon/{concept_id}`, `POST /api/languages/{lang}/sentence` (`concept_ids` in signing
  order, or `words` `[{concept_id?, en, hi?}]` for My signs and spelled words; returns `en` and `hi` `SentenceOut`,
  the words used and the approximate label; allowed in the read-only demo because it stores nothing); vocabulary rows
  gain `hi_alt`, `hi_source`, `hi_reviewed`; `PATCH .../lexicon/{concept_id}` stores `english` and `reviewed` too.

---

## 9. Evaluation (task 7)

### 9.1 Protocols

| Id | Protocol | Data | Metrics |
|---|---|---|---|
| E1a | INCLUDE closed set, official split | `include-official` | top-1, top-5, macro F1; **labelled not signer-independent** |
| E1b | INCLUDE closed set, **estimated signer-independent** (primary) | `include-pseudo-signer-cv5` (+ `include-session-cv5`) | top-1, top-5, macro F1, per category (15), per class, per pseudo-signer; mean ± spread over folds |
| E2 | Cross-dataset one-shot, open vocabulary | `oneshot-concept-cv5` (5.5) | top-1, top-5, top-10 over full gallery and over a 263-concept gallery; seen-class vs unseen-class |
| E3 | Abstain behaviour (fused system) | E1b test folds + E2 queries | risk–coverage curve, AURC, accepted precision, coverage, suggest rate, top-5 hit rate when suggesting, abstain rate, **silent error rate** (accepted and wrong / all) |
| E4 | My signs, few-shot | INCLUDE sessions: enrol k ∈ {1,3,5} takes of a held-out pseudo-signer, test on their other takes | top-1 gain over no enrolment; false-accept rate of user signs on other words |
| E5 | Fingerspelling | letter holds (leave-one-signer-out when ≥ 2 signers), synthetic spelled words from concatenated holds (labelled synthetic) | per-letter accuracy, confusion, word top-1/top-5 after lexicon decoding |
| E6 | Latency | engine end-to-end on CPU | p50/p95 per segment (feature, classifier, encoder, search, fusion) |
| E7 | ASL regression | bundled WLASL100 model | metrics identical to `models/wlasl100-gru-signer-holdout/results.json`; full existing test suite passes |
| E8 | Trim quality | 5.3 checks | end error on clean clips, method mix, review sheet |

### 9.2 Reporting rules (honesty)

* Every number carries: protocol id, data profile, n clips, n classes, n signers or groups (and whether they are estimated),
  95% interval (bootstrap over **groups**, not clips, 1,000 resamples), git commit, date.
* The official INCLUDE split number is **never shown alone**; it always sits next to E1b with its label.
* Test folds are used once, after all selection on validation (same rule as `docs/EVALUATION.md`).
* Open-vocabulary numbers (E2) are shown as the headline "any dictionary word" accuracy; closed-set numbers as "core words".
* Accuracy is always shown together with coverage and the silent error rate (an accuracy at 40% coverage is not comparable to one at 100%).
* Live webcam accuracy is expected to be lower than all of these (different camera, lighting, signers); the UI says so.
* Planning guesses (not results): core words, signer-independent: somewhere between 60% and 85% top-1; any-dictionary-word one-shot:
  far lower top-1 (likely below 30%) with a much higher top-5. This is why the product relies on "did you mean", abstaining,
  My signs and spelling. Replace these guesses with measured values as soon as E1b and E2 exist.
* `backend/app/isl/evaluation/report.py` writes `results.json` and a Markdown table for `docs/` with the same fields.

---

## 10. Data profiles and licences (task 8, part 1)

### 10.1 Profiles (`backend/app/isl/profiles.py`)

```python
@dataclass(frozen=True)
class DataProfile:
    name: str
    sources: tuple[str, ...]            # adapters allowed for training, index and calibration
    pretraining: tuple[str, ...]        # extra sources used only to initialise weights
    commercial: bool

PROFILES = {
    "commercial-safe": DataProfile("commercial-safe", ("include", "islrtc"), (), True),
    "research":        DataProfile("research", ("include", "islrtc", "cislr", "isign"), ("wlasl", "isign"), False),
}
```

* User data (My signs) never enters the global model in either profile; it lives only in the user's index.
* The profile is written into every checkpoint meta, `index/meta.json`, `fusion.json`, `results.json` and `pack.json`;
  `IslPack` refuses to combine components from different profiles.
* Both packs can be built side by side: `artifacts/packs/isl/` (commercial-safe, default) and `artifacts/packs/isl-research/`.
  Only the commercial-safe pack's small files are committed to `models/packs/isl/` (each file < 50 MB).

### 10.2 Licence table (copy into `docs/DATASETS_ISL.md`, owned by the Data Engineer)

| Source | Licence | Commercial | Attribution / notes |
|---|---|---|---|
| INCLUDE (AI4Bharat), zenodo 4010759 | CC BY 4.0 (zenodo metadata) | yes | cite Sridhar et al., ACM MM 2020; split lists from the AI4Bharat INCLUDE code repository |
| ISLRTC Indian Sign Language Dictionary (data.gov.in) | GODL-India | yes, with attribution | "Indian Sign Language Research and Training Centre (ISLRTC), Government of India, via data.gov.in"; the Hugging Face re-encode (silentone0725, labelled MIT) is a mirror and does not replace the GODL attribution |
| CISLR (Exploration-Lab) | AFL-3.0 (dataset card) | **pending review**: AFL-3.0 itself allows commercial use, but the videos were collected from public online sources whose rights the licence may not cover | research profile only until reviewed; cite Joshi et al., EMNLP 2022 |
| iSign (Exploration-Lab) | CC BY-NC-SA 4.0 (dataset card) | no | research profile only; share-alike applies to derived data |
| WLASL | C-UDA 1.0 | no | already documented in `docs/LICENSING.md` |
| WordNet (via NLTK data) | WordNet licence (permissive) | yes | used for synonym detection and POS |
| SCOWL word list | permissive (MIT-like) | yes | spelling lexicon |
| IndicTrans2 / IndicXlit (AI4Bharat) | confirm at integration (expected MIT) | expected yes | only the generated Hindi text ships, with `source` per row |
| MediaPipe Holistic | Apache-2.0 | yes | unchanged |
| User sign packs | chosen by the author at export (default CC BY 4.0) | per pack | the manifest carries the licence |

---

## 11. Phase-2 training plan and CPU estimates (task 8, part 2)

Basis (measured on this machine on 25 Nov 2026 while ingestion used about 3.5 CPUs): BiGRU hidden 192, T = 48, D = 359,
batch 32: **about 240 ms per training step** with 2 threads; inference batch 1: about 10 ms; strong augmentation + featurisation:
**about 2.3 ms per clip**. After ingestion, with 4 threads, expect roughly 1.5–2× faster. Estimates below use the measured (slow) numbers.

| Step | Command (`python -m app.isl.cli ...`) | Work | CPU wall time |
|---|---|---|---|
| 0 | `vocab` | parse 13.7k names, link INCLUDE/CISLR, overrides | < 5 min |
| 1 | `trim` (stage 1) | 13.6k clips × ~5 ms | 2–5 min |
| 2 | `trim --learned` (stage 2) | synthetic set + 1-layer BiGRU | 10–20 min |
| 3 | `splits` + `dedup` | sessions, pseudo-signer clustering, duplicates | < 15 min |
| 4 | `train-classifier` (pseudo-signer 5-fold CV, then final) | ~3.5k clips, ~110 steps/epoch ≈ 35 s/epoch, 100 epochs | ~1 h per run, ~5–6 h for CV + final |
| 5 | `train-encoder` (commercial-safe) | ~16k clips/epoch (INCLUDE + ISLRTC ×1 + positives), batch 64 ≈ 250 steps × ~0.5 s + 40 s features ≈ 2.7 min/epoch, 60 epochs | ~2.7 h final; one-shot CV with 2 folds × 30 epochs ≈ 2.7 h |
| 6 | `build-index` | ~25k rows, batched | 3–5 min |
| 7 | `train-letters` | a few thousand frames, MLP | < 10 min |
| 8 | `fit-fusion` + calibration | grid search on cached validation outputs | < 10 min |
| 9 | `hindi` | IndicTrans2 distilled, ~12k short inputs on CPU | 0.5–1.5 h (then human review of the core 1,000) |
| 10 | `evaluate` (E1–E8) | cached embeddings/probabilities | ~30 min |
| 11 | `build-pack --profile commercial-safe` | copy + checksums + `pack.json` | minutes |
| **Total commercial-safe** | | | **about 13–17 h** of wall time; run overnight in 2 sessions with `nice -n 10` |
| R1 | research: WLASL warm start | 524 clips, 100 epochs | ~20 min |
| R2 | research: iSign subset ingestion | 10k sentence poses (streamed; about 14 GB transferred, nothing large kept) | network bound, 3–6 h |
| R3 | research: iSign contrastive pre-training | 10k clips × 20 epochs | ~3 h |
| R4 | research: CISLR ingestion + encoder retrain + index | 7k videos (~1 s each to extract) + step 5 again | ~2 h + ~3 h |
| **Total research extra** | | | **about 11–15 h** |

Memory: BodyFrame cache in float32 about 1.2 GB; ArcFace weights about 36 MB; peak training RSS under 4 GB.
Disk: derived files (trims, splits, caches, packs) under 2 GB.

Order of work and acceptance per agent:

1. **Data Engineer**
   * Commit-ready: `isl/vocab/*`, `isl/data/{sources,trim,boundary,splits,dedup,remote,cislr,isign}.py`,
     `scripts/ingest_isl_gated.py`, backward-compatible `ingest_isl.py` options (5.7), `docs/DATASETS_ISL.md`.
   * Accept when: `vocabulary.jsonl` builds from the current partial manifests and from the final ones; tests for name parsing
     (≥ 60 real ISLRTC/INCLUDE names with expected output), trimming on synthetic data, split leakage = 0, id mapping of
     the official INCLUDE lists (every path matched or reported); `hf_access` reports `pending` today for both gated repos
     (README 200 but data 403) and the CISLR job runs end-to-end on a mocked `RangeFile`; trim review sheet generated.
2. **AI Engineer (models)**
   * `inference/recognizer.py`, engine `recognizer=` option, `packs/*`, `api/languages.py` (recognition parts),
     `db/migrate.py`, `training/trainer.fit`, `isl/augment.py`, `isl/models/*`, `isl/fingerspell/*`, `isl/recognizer.py`,
     `isl/training/*`, `isl/evaluation/*`, `isl/cli.py`.
   * Accept when: the full existing backend test suite passes unchanged (E7); smoke runs on partial data (≤ 10 min CPU each)
     produce a loadable pack; WebSocket tests cover `language` query/config, re-hello, `choose`, unknown language error, and
     ASL messages identical to today; fusion `decide()` unit tests for accept/suggest/abstain/out-of-vocabulary; latency E6 measured.
3. **AI Engineer (Hindi + sentences)**
   * `isl/text/*`, `function_words.yaml`, `lexicon_hi.jsonl` generation, `isl/mysigns/*` text fields, sentence endpoint.
   * Accept when: every concept has a lexicon row (possibly `missing`), the core list is marked for review, sentence builder tests
     (≥ 25 example gloss sequences in both languages, each marked approximate), no runtime translation.
4. **Frontend Developer**
   * `state/language.tsx` (`LanguageContext`, persisted in `localStorage` key `sis.language`, read with try/catch),
     `LanguageSwitcher` in `components/Layout.tsx`, Live page: bilingual word (`BilingualWord`, Devanagari font self-hosted via
     `@fontsource/noto-sans-devanagari`, OFL), `CandidateList` ("Did you mean?" 5 buttons sending `choose`), "Not sure" state with
     "sign again / spell it", spell-mode toggle and letter strip, `SentenceBar` with "≈ approximate" label and speak buttons,
     "this was wrong" button; new `MySignsPage` (`/my-signs`): record wizard (3–5 takes using the existing MediaPipe loop and the
     segment boundaries), quality/consistency/conflict feedback, list, enable/disable, export (with consent text) and import;
     Vocabulary page: language-aware, server-side search and paging, filters, Hindi review editing, skeleton playback of the
     reference sign; Evaluation page: ISL `results.json` with labels and intervals. Types in `api/client.ts`: `Candidate`
     (superset of `TopK`), `LanguageInfo`, `Recognition`, `MySign`, `AddResult`.
   * Accept when: typecheck, Vitest and build pass; Playwright fake-camera test switches to ISL and back; ASL page behaviour unchanged;
     keyboard and screen-reader access for the candidate list (the WCAG work already done must not regress).

---

## 12. Risks and open questions

| Risk | Effect | Mitigation |
|---|---|---|
| ISLRTC has one signer per entry | open-vocabulary accuracy for unseen signers is low | strong augmentation, canonical handedness, cross-source positives, top-5 + abstain, My signs, CISLR when available |
| INCLUDE signers (Chennai) and Indore signers use regional forms | correct signs judged "wrong" | variant model, variant-consistency report, regional packs from Indore schools |
| Trimming errors in ISLRTC | bad references | sidecar trims with confidence, learned refinement, two index rows per entry, review sheet |
| Official INCLUDE split leaks sessions | inflated numbers | pseudo-signer and session splits as primary; label on every official number |
| CISLR duplicates ISLRTC videos | inflated one-shot numbers | `dedup.py` before any CISLR evaluation |
| Browser vs Python MediaPipe differences | live accuracy drop | existing parity check extended to ISL clips (CHECKLIST) |
| Machine-translated Hindi wrong for a sense | misleading output | review status shown, core list human-reviewed, sense context in MT input, local corrections |
| iSign pose format / MediaPipe version differs | distribution shift in research pre-training | separate `extractor_version`, used only for pre-training, ablation in E1/E2 |
| Gated access stays pending | no CISLR/iSign | everything works with INCLUDE + ISLRTC; `--check` can be scheduled |
| Personal data in sign packs | privacy / DPDP | landmarks only, face off by default, consent text, local storage |

Open questions for the owner (do not block phase 2): who reviews the core Hindi list (a teacher at an Indore school would be ideal);
whether an Indore school can record the ISL alphabet and 100 everyday words with 5+ signers (the single most valuable dataset for this launch).

---

## 13. Frontend implementation notes and deviations (Frontend Developer, 25 Nov 2026)

Implemented (section 11, task 4): `state/language.tsx`, `i18n/` (English/Hindi interface strings), `lib/speech.ts`,
`components/{LanguageSwitcher,BilingualWord,CandidateList,SignResult,SentenceBar,SkeletonPreview,ErrorBoundary}.tsx`,
`recognition/useLandmarkRecorder.ts`, `pages/{MySignsPage,LanguageVocabulary,LanguageEvaluation}.tsx`, and the language-aware
`LivePage`, `VocabularyPage`, `EvaluationPage` and `Layout`. What the user sees:

* **Sign language switch** in the header: ISL first and selected by default, ASL second; the choice is kept in `localStorage`
  (`sis.language`). The backend default stays `asl-wlasl100` (so old clients and the legacy routes are unchanged); the
  *frontend* picks ISL whenever an ISL pack is listed (`pickLanguage`). The static in-browser demo has only ASL and shows no switch.
* **Live page, ISL**: the result is shown in large type as "WATER / पानी" with the Hindi review marker (checked / word list /
  machine / doubtful / your text) or "Hindi not available yet / हिंदी उपलब्ध नहीं"; `suggest` shows up to five tappable
  English+Hindi chips ("Did you mean…?") that send `choose`; nothing enters the sentence until the user taps; `abstain` shows
  "Not sure – try again or spell it". An accepted word has a "Wrong word? Remove it" button that sends `reject`. The
  ASL threshold sliders are replaced by the pack policy (Strict / Balanced / Lenient – more mistakes since 15.29; default strict) and a
  "Recognise my own signs" switch; the ASL engine still receives exactly the config fields it always did
  (`engineConfigPayload`, unit-tested). Switching language on an open socket sends `config` with `language`, clears the
  sentence and waits for the new `hello`.
* **ASL keeps working** with the same messages. Small visible changes: a rejected ASL sign that still has ranked guesses now
  shows the same "did you mean" chips (the choice is added locally, because the classic engine has no `segment_id`), and
  ASL words also show Hindi when `models/packs/asl-wlasl100/lexicon_hi.jsonl` provides it.
* **Spelling**: the Words / Spell switch appears only when the socket's `hello` lists the `spelling` capability (a trained
  letter model, or a calibrated alphabet). Otherwise the page links to the alphabet calibration on My signs. Live letters
  (`state.spell`) show in a letter strip; the finished word arrives as "did you mean" chips (never auto-accepted), labelled
  experimental.
* **Sentence strip**: the committed words as English+Hindi chips; `POST /api/languages/{lang}/sentence` (`words` form, so My
  signs and spelled words work too) gives an approximate English and Hindi sentence, always labelled approximate, with the
  applied rules on hover. If the endpoint is unreachable the words are shown in signing order. **Speak** uses the browser's
  `speechSynthesis` (`hi-IN` or `en-IN`; off by default; "speak each new word" toggle); with no Hindi voice the page says so
  and English is spoken. Nothing is sent to a server.
* **My signs** (`/my-signs`): camera + MediaPipe in the browser; "Record take N" starts when the hands appear and stops after
  450 ms with hands down (capped at 3.8 s, under the 4 s server limit); 3–5 takes, each previewed as a stick figure; English
  word with dictionary suggestions to link the sign to an existing concept (then Hindi is filled in); Hindi, region
  (default "Indore, Madhya Pradesh") and note. "Check and save" runs `dry_run` first: if it is accepted with no
  look-alikes it saves; with a look-alike it asks "This looks like X. Save anyway as your own version?" (or save as a version
  of X); otherwise it shows the server's message and marks the problem takes. The list allows on/off, edit and delete
  (two-step). Export needs the consent checkbox (text from 7.4 in English and Hindi), face points off by default, licence
  choice; import supports merge/replace. **Alphabet calibration**: A–Z buttons record a 1.2 s hold per letter
  (`POST .../my-signs/alphabet`); "spelling ready" once the server says so (at least 5 letters).
* **Vocabulary, ISL**: server-side paged search over all concepts; a query in Devanagari searches the Hindi lexicon (main
  word and alternatives, nukta-insensitive; small backend addition in `packs/vocabulary.py`); filters for entry type and
  core words; the Hindi coverage line states how much is machine translation and that 0 rows are teacher-checked; detail
  panel with Hindi review state, alternatives (machine alternatives labelled "not checked"), variants and regions, a
  stick-figure preview of the dictionary sign when the landmarks are on this computer, speak buttons and a local Hindi
  correction form (`PATCH .../lexicon/{concept_id}`: "Hindi is correct", "Save correction", "Hindi is wrong" = disputed;
  disabled in the public demo).
* **Evaluation, ISL**: `results.json` per protocol with n, classes, groups and 95% intervals; E2 ("any dictionary word") is
  placed first; E3 shows coverage next to accepted precision, the silent error rate, "did you mean" and "not sure" rates per
  policy; a smoke run is labelled as such. Only measured numbers are shown, never targets.
* **Hindi interface**: an EN / हिंदी toggle in the header (remembered; defaults to Hindi when the browser language is Hindi);
  `<html lang>` follows it. Every English string has a Hindi string with the same placeholders (unit-tested). The Hindi
  interface text was written by the AI developer, **not reviewed by a native speaker**; a teacher should read it.
* **Fonts**: Noto Sans Devanagari (OFL, `@fontsource/noto-sans-devanagari`, Devanagari subset only, loaded only when Hindi is
  on screen); Hindi text carries `lang="hi"` for screen readers.

Backend changes made for the UI (small, tested in `tests/isl/test_vocab_hindi_search.py`): Devanagari queries in
`Vocabulary.search`; `IslPack` falls back to the tracked `models/packs/<id>/lexicon_hi.jsonl` when a local build in
`artifacts/packs/<id>` has none (the smoke pack was built before the lexicon existed, so live results had no Hindi).

Deviations and gaps: the localStorage key for the language is `sis.language` as specified, but the interface language uses
`ui-lang-v1`; the static demo does not offer ISL or My signs (no IndexedDB store yet, 7.4); the Analysis, Studio, Lab and
Dataset pages remain ASL/English only; no "did you mean" keyboard shortcuts beyond normal Tab/Enter; the Playwright tests use
Chromium's synthetic camera (no person), so accept/suggest/abstain rendering is covered by Vitest component tests and a
scripted WebSocket check, not by an end-to-end signing test.


## 14. Accessibility audit for deaf and low-literacy users (Accessibility Auditor, 25 Nov 2026)

**Scope.** Every screen added or changed for ISL: header switches, Live (recognised / did you mean / not sure, sentence
strip, speech controls), My signs (recorder, form, list, share, alphabet), ISL Vocabulary (search, word detail, Hindi
correction), ISL Evaluation, plus the layout, sidebar and footer around them. Checked in the English and Hindi interface,
with ISL and ASL selected, at 1280 px and 320 px (= 400% zoom). Standard: WCAG 2.2 AA. The users in mind are deaf ISL
signers in Indore/MP, many reading Hindi better than English, some with low literacy, some deafblind (braille display),
some using a phone or tablet, all signing with both hands while the app listens.

**Method.** axe-core 4.14 through `@axe-core/playwright` on the preinstalled Playwright Chromium (all 11 pages x 2
interface languages x 2 sign languages); custom browser checks for what axe cannot see (Hindi marked `lang="hi"`, letter
spacing and size of Devanagari, English-only pages marked `lang="en"` in the Hindi interface, sideways scrolling at 320 px,
clipped content); keyboard-only walk-throughs; code review of names, roles, live regions and focus; the three result states
driven through a mocked recognition WebSocket. Screen readers (NVDA, JAWS, VoiceOver, TalkBack) and a real braille display
were **not** available on this machine: screen-reader behaviour below is from the accessibility tree and ARIA semantics,
not from listening to a real reader.

**Automated result.** axe-core already reported 0 violations on every page before this audit (earlier work had fixed the
basics). All the problems below were found by manual review and the custom checks, which is why "axe is green" was not
enough.

| # | Problem found | WCAG | Severity | Fix |
|---|---|---|---|---|
| 1 | Recognised, "did you mean" and "not sure" were told apart mostly by an 11 px label; "did you mean" and "not sure" used the same amber colour | 1.4.1, 1.3.3 (and plain usability for low literacy) | Serious | Each state is now a framed panel with its own colour, border, round icon and a 16-20 px heading: teal + tick (recognised), amber + "?" (did you mean), orange dashed border + "try again" arrow (not sure). `data-state` marks them for tests |
| 2 | A recognised word was not announced to screen-reader / braille users (only the generic status pill was) | 4.1.3 | Serious | `ResultAnnouncer`: polite live region announcing "Recognised: water, पानी" (Hindi in a `lang="hi"` span), "Not sure. 3 possible words are shown…", "Not sure…"; quiet while signing; re-announces the same word signed twice |
| 3 | After "Wrong word? Remove it" the word was still shown large as "recognised" | 3.3.4-style error recovery, clarity | Serious | A rejected word becomes a resolved state "Removed from your sentence. Please sign the word again." (hook tracks `rejected`) |
| 4 | Focus was lost to the page after choosing a "did you mean" word, rejecting a word, removing a sentence word, Edit/Delete in My signs, alphabet reset | 2.4.3 | Serious (keyboard, switch and screen-reader users) | Focus moves to the confirmation message, the word list, the edit form / confirm button, and back to Edit / Delete; it only moves when the pressed control has gone |
| 5 | Vocabulary word detail opened above the grid while focus stayed on the card below | 2.4.3 | Moderate | Detail panel is a labelled region that takes focus; closing returns focus to the card |
| 6 | "Did you mean" buttons used an `aria-label` ("Choose water पानी, 46 percent") that replaced the visible text and its `lang` marks | 2.5.3, 3.1.2 | Moderate | Name now comes from the visible text with explicit spaces: "Choose water पानी 46% sure"; percentage is visible as "46% sure" instead of a bare number |
| 7 | Hindi letter-spaced apart ("श ब्द") in the Live eyebrow and other `tracking-*` labels | 1.4.8-style readability; rendering defect | Serious in the Hindi interface | Global rule: no letter spacing on Devanagari |
| 8 | Hindi captions at 10-11 px (sidebar groups, badges, review markers, alphabet counts at 9 px) lose their matras | 1.4.4 / readability | Moderate | Devanagari never below 12 px; alphabet counts and review markers enlarged |
| 9 | Devanagari outside `lang="hi"` (and in `font-mono`) fell back to whatever system font exists, tofu boxes on some Android/Windows set-ups | robustness of Hindi rendering | Moderate | Bundled Noto Sans Devanagari is in the default sans and mono stacks (its unicode-range keeps Latin in IBM Plex) |
| 10 | "हिंदी" option and button labels had no `lang="hi"` in the English interface | 3.1.2 | Moderate | `Segmented` options take `lang`; all "हिंदी"/"English" labels marked |
| 11 | English-only pages (Overview, Evaluation, Analysis, Studio, Dataset, Lab, Experiments, Research, ASL vocabulary), the footer, demo banner, ASL settings and camera metrics were read with Hindi rules in the Hindi interface (`<html lang="hi">`) | 3.1.2 | Moderate | `pageIsTranslated()`; untranslated page content, footer and English-only blocks carry `lang="en"` |
| 12 | Vocabulary badges showed raw codes ("multiword", "phrase") in the Hindi interface | 3.1.2, understandable | Minor | Translated labels |
| 13 | Dictionary and take previews loop forever with no pause | 2.2.2 | Moderate (vestibular, attention) | Pause/play button on every `SkeletonPreview`; starts paused (middle frame) with reduced motion |
| 14 | Sentence chip buttons 22 px and take-remove buttons 18 px | 2.5.8 | Moderate (motor, tablets) | 32 px targets; did-you-mean actions are full-size buttons |
| 15 | "Check and save" was disabled with no reason given | 3.3.2 | Moderate (low literacy) | "To save: record the sign at least 3 times (2 so far) / type the English word", linked with `aria-describedby` |
| 16 | Recording time inside a live region would be read ten times a second | 4.1.3 (noise) | Moderate | Time is visual only; the region announces phase changes |
| 17 | Header health dot on phones was colour only (text `display:none`) | 1.4.1 | Minor | Text kept for assistive technology at all widths |
| 18 | 320 px: "camera off" panel clipped (Start button cut), My signs scrolled sideways (file input, 16:9 box) | 1.4.10 | Moderate | Minimum height when the camera is off; grid columns `minmax(0,1fr)`, full-width file input |
| 19 | ISL evaluation table: scrollable area not keyboard focusable, no caption or header scope | 2.1.1, 1.3.1 | Minor | `ScrollRegion`, caption, `scope` on headers |

**No reliance on sound.** Confirmed: nothing in the ISL flows uses audio cues; speaking the sentence aloud is optional, off
by default, and its absence is explained on screen. The live camera `<video>` is muted and has no audio track, so captions
(axe "needs review": video-caption) do not apply.

**What was already good and must be kept.** Skip link and focus to `<main>` on navigation; page titles per route; the
mobile drawer is a proper modal with focus trap and Escape; radio-group switches with arrow keys; `role="switch"` toggles;
form fields wrapped in labels; landmark structure; visible focus outline; reduced-motion support; "Hindi not available yet"
instead of invented Hindi; the review markers' long text available to screen readers.

**Tests added.** `frontend/src/components/a11y.test.tsx` (16 Vitest tests: distinct states, axe-core structure rules on
every state in English and Hindi, focus after choose / reject / remove, announcements, names and `lang`, pause control).
`frontend/e2e/a11y.spec.ts` (48 Playwright tests: axe WCAG 2.2 A/AA on 11 pages x EN/HI x ISL/ASL with the Devanagari and
`lang` checks, bundled font and no letter spacing, 320 px reflow, keyboard focus in Vocabulary, the save hint, and the
three live states via a mocked socket with axe, announcements and keyboard choice). Run with the e2e servers up:
`cd frontend && npx playwright test e2e/a11y.spec.ts`. The checks were run against the code before the fixes and caught
the Hindi size/spacing/lang problems, the 320 px overflow and the missing `lang="en"` marks.

**Still open (not fixed here).**
* Real assistive-technology testing: NVDA + Chrome and TalkBack + Chrome on Android (the likely set-up in Indore), and a
  braille display for deafblind users; voice control is irrelevant for most signers but switch access is not.
* Testing with deaf ISL users and teachers in Indore (low literacy, Hindi-first): icon meaning, wording and the order of
  English/Hindi should be validated with them, ideally with an ISL video or picture guide for the three states.
* Overview, Evaluation, Analysis, Studio, Dataset, Lab, Experiments and Research are English only (now correctly marked
  `lang="en"`); the Overview landing page in particular should be translated. Small leftovers in translated pages: "frames"
  in the result footnote, the camera metrics overlay, ASL-only settings, engine reason codes for ASL.
* The Hindi interface text and screen-reader announcements need review by a native Hindi speaker.
* `role="alert"` is used for static warning notices (e.g. the smoke-build notice), which some readers announce on every
  page load; consider `role="note"` for static warnings.
* Older ASL pages (Dataset, Experiments) still have non-focusable horizontal-scroll tables; out of this audit's scope.
* The status pill still blinks (`pulse-dot`) while busy; it stops with reduced motion but has no other pause.


## 15. Phase 1 gate and the phase-2 work list (Reality Checker, 25 Nov 2026)

**Verdict: phase 1 is complete as a code base, NOT as a product.** The ISL pipeline, API, UI, vocabulary and Hindi word list
exist and are tested; **no ISL model has been trained beyond a smoke run, so the app cannot yet recognise ISL dictionary words.**

Checked from a clean clone of `claude/indian-sign-language` (commit `fc1745a`) in `/tmp`, following the README:

* `./setup.sh --cpu --dev` works (1 min 23 s). Backend: 286 passed, 1 skipped (WordNet data not installed in a fresh
  clone), with `SLR_TORCH_THREADS=2`. Frontend: typecheck + production build pass, `build:static` passes, Vitest 57 passed, 1 skipped.
* `./start.sh` serves both languages. ASL (`asl-wlasl100`) is `ready` and the bundled model answers over the WebSocket
  (git shows no change to `models/wlasl100-gru-signer-holdout/`; `FeatureConfig.to_dict()` is unchanged for ASL).
  ISL (`isl`) is `missing_models` in a clean clone: real INCLUDE recordings streamed over the ISL WebSocket end as
  `not_sure`, never a wrong word. The smoke pack exists only locally in `artifacts/packs/isl` (version `2026.11.0-smoke`,
  `results.json` labelled "SMOKE RUN ... NOT the accuracy of the app").
* No accuracy number is presented as the app's accuracy: the smoke numbers in 6.7 are labelled, 9.2 holds planning guesses
  marked as guesses, the frontend has no hard-coded ISL metrics.
* Licences are recorded in `docs/DATASETS_ISL.md`; this gate added the ISLRTC non-profit terms to the pack licence entries
  (`pack.json`, `build-pack`) and corrected the CISLR licence string in research packs to CC-BY-NC-4.0.
* Nothing large or secret is in the branch history since `main` (largest blob `vocabulary.jsonl`, 7.5 MB; no data,
  landmarks, weights, `.env` or tokens). Gated access today: CISLR and iSign `pending`.
* Ingestion jobs still running at 14:15 UTC: INCLUDE 1,921 of 4,292 videos, ISLRTC 4,140 of 13,665. At the last hour's
  rate INCLUDE needs about 4.5 h more and ISLRTC about 10.5 h more (finishing around 01:00 UTC on 26 Nov).

### 15.1 What phase 2 must do (in order)

1. **Confirm ingestion finished** (updated 25 Nov 20:10 UTC, see 15.6): `scripts/phase2_train.sh --profile both
   --check-only` exits 0 (the readiness guard: no ingestion process, every manifest complete, retryable download
   rows retried). INCLUDE is complete when all 44 zips are in `manifests/include_zips_done.txt` (4,284 distinct clips,
   not the published 4,292); `islrtc.jsonl` needs a row for each of the 13,665 index entries. Do not delete `data/isl`.
2. **Refresh the data layer** (from `backend/`, `nice -n 10`, 4 threads once ingestion is done):
   `python -m app.isl.cli wordnet` (if missing), `trim`, `trim-eval`, `trim-review` (look at the sheet), `splits --fetch`,
   `splits`, `dedup`, `vocab --no-trims`, `status`. If concept ids changed, re-run `hindi --translate` then `hindi`, and commit
   the new `vocabulary.jsonl` and `lexicon_hi.jsonl` (the consistency test needs >= 95 % Hindi coverage).
3. **Decide the primary evaluation split** on the full data: if the pseudo-signer clustering is still very uneven, keep
   `include-session-cv5` as primary and say "session-independent, not signer-independent" next to every number.
4. **Trimming**: the synthetic end-error check is 67.7 % within 300 ms against an 80 % target, and compound names are cut
   early. Either improve stage 1 / stage 2 on the full data or accept and record the measured figure; keep low-confidence
   trims down-weighted.
5. **Train the commercial-safe pack at full settings** (section 11, about 13-17 h): `train-classifier` (CV + final),
   `train-encoder`, `build-index --all-include`, `fit-fusion`, `evaluate`, `build-pack`. `train-letters` only once real
   alphabet recordings exist (none in the open data).
6. **Implement the missing evaluation runners** before quoting accuracy: E2 per-fold *held-out concept* one-shot (the
   smoke E2 is the seen-class variant), E4 My-signs few-shot, E5 fingerspelling, E8 trim quality; plus E1a (official split,
   labelled not signer-independent), E1b, E3 risk-coverage with the silent-error rate, E6 latency on free CPUs, E7 ASL
   regression (`models/wlasl100-gru-signer-holdout/results.json` unchanged). Bootstrap intervals over groups; test folds once.
7. **Re-fit the fusion thresholds** on the full validation folds (Wilson lower bound); the UI must show `fusion.json`
   `achieved`, never the targets. Revisit the 0.85 top-5 rule that turns many correct top-1 results into "not sure".
8. **Replace guesses with measurements** in README, `docs/ISL_DESIGN.md` (0, 9.2, this section) and `docs/DATASETS_ISL.md`
   (all counts dated 25 Nov are from partial data). Remove the README "not yet trained" note only when a full pack exists.
9. **Ship decision for the pack**: weights must stay under 50 MB to be committed (commercial-safe pack only; never research
   or iSign-derived weights); otherwise publish it as a release asset and document the download.
10. **Research profile** only if `scripts/ingest_isl_gated.py --check` reports access: ingest CISLR / an iSign subset,
    `dedup` against ISLRTC before any evaluation, verify iSign left/right hands (`--swap-hands`), check CISLR CSV column
    names on the real files, WLASL warm start, build `isl-research` separately.
11. **People, not code**: a native Hindi-speaking teacher reviews the core ~1,000 Hindi words and the Hindi interface text
    (then `hindi --merge-local`); hide unchecked machine `hi_alt`; real-webcam tests of Live and My signs with Indore deaf
    signers; ask an Indore school to record the ISL alphabet and everyday words with 5+ consenting signers.
12. **Smaller gaps**: look-alike conflict report on sign-pack import; Playwright `e2e/isl.spec.ts` against a backend with
    the trained pack; ISL in the static demo (IndexedDB) later; licence decision on ISLRTC before any paid release.

### 15.2 Phase-2 training and evaluation on all sources (AI Engineer, 25 Nov 2026)

**Status: ready to run, not run at full settings.** Everything below is implemented and tested on synthetic data
and smoke-tested on the partial data of 25 Nov; no full training has been started (ingestion is still running).

**One command per profile** (resumable; run it again after any stop and it continues where it was):

```
nohup scripts/phase2_train.sh --profile both >> data/isl/logs/phase2_driver.out 2>&1 &   # commercial-safe, then research
scripts/phase2_train.sh --profile research --list                                      # which steps are done
```

Steps (one stamp file per step in `data/isl/logs/phase2_<profile>.stamps/`, log in `data/isl/logs/phase2_<profile>.log`,
everything under `nice -n 10`, stops before any step if less than 10 GB of disk is free):

| Step | Command | What it does |
|---|---|---|
| trim | `trim` | ISLRTC first-sign trims (incremental); learned end model `artifacts/isl/trim_boundary.pt` when present, `trim_version` says which (15.4) |
| dedup | `dedup --sources include islrtc cislr wslp isl500 mendeley_*` | landmark near-duplicates across sources |
| vocab | `vocab --no-trims --out <work>/vocabulary.jsonl` | refreshed vocabulary in the work folder (review, then commit it to `models/packs/isl/`) |
| splits | `splits --vocab <work>/vocabulary.jsonl` | INCLUDE official / session / pseudo-signer splits, one-shot concept folds |
| link | `link-sources --all-sources` | every labelled word source linked to concepts, signer-held-out splits, duplicates dropped |
| bench | `bench` | CPU seconds per step for the chosen model size (logged, for the time budget) |
| pretrain_dedup | `pretrain-dedup` | OpenHands Sign Library chunks that repeat another chunk or contain an evaluation clip (pose-content matcher, 15.4) |
| pretrain | `pretrain` | self-supervised encoder pre-training on every unlabelled / sentence source |
| encoder | `train-encoder --all-sources --init auto` | fine-tune (SupCon + sub-centre ArcFace) from the pre-trained trunk |
| oneshot | `train-oneshot --folds 0 1` | E2 encoders, each fold's concepts held out of all supervised training |
| classifier | `train-classifier --all-sources` | closed-set classifier over every concept with >= 2 signer groups |
| cv | `train-classifier --split <primary>-fold{1..4} --ckpt <work>/cv/fold<k>/classifier.pt` | E1b mean / spread over folds and a signer-grouped interval pooled over every fold's held-out signers (one fold holds out a single pseudo-signer, so one fold alone cannot give a signer-level interval); `CV_FOLDS=""` skips it |
| e1a | `train-classifier --split include-official --ckpt <work>/e1a/classifier.pt` | the model E1a needs (official split) |
| fingerspell | `train-fingerspell` | letter model from every letter source + leave-one-signer-out E5 |
| index | `build-index --all-sources --out <work>/index` | index over training references only (keeps evaluation clean; 15.21) |
| fusion | `fit-fusion --all-sources` | fusion weights and thresholds on validation; always writes strict / balanced / lenient |
| evaluate | `evaluate --all --all-sources` | E1a-E8 -> `<work>/eval/report.json` + `report.md` and `artifacts/packs/<pack>/eval/` |
| index_final | `build-index --all-sources --all-include --out <work>/index_final` | the shipped index: every reference of every source, in its own folder (15.21) |
| pack | `build-pack` | `artifacts/packs/isl` or `isl-research` (copies `eval/`, lists every source and licence) |

Options: `--from STEP` and `--force S1,S2` (both also delete the training state of the steps they redo; 15.19),
`--only STEP`, `--tag NAME` and `--dry-run` (15.19), `--allow-shared-data` (15.20), `--threads 4` once the
machine is free, `--step-minutes M` (each training call stops after M minutes with its state saved and is re-run with
`--resume` until done: exit code 75 means "stopped, continue later"). Full-setting sizes can be overridden with
environment variables (`HIDDEN LAYERS EMB PRE_EPOCHS PRE_STEPS PRE_BATCH ENC_EPOCHS ENC_STEPS CLS_EPOCHS CLS_STEPS
E1A_STEPS CV_FOLDS ONESHOT_FOLDS ONESHOT_EPOCHS LETTER_EPOCHS`). `--smoke` uses tiny settings, a private ISL root (real manifests and
landmarks through read-only symlinks, its own `derived/` and `splits/`) and writes only under
`artifacts/isl/phase2-smoke/`, so it never changes shared derived files or the packs the app loads.

**Sources per profile** (`backend/app/isl/profiles.py`, re-exported by `training/multisource.py`; WLASL and `isl_csltr_words` are in no profile, 15.4; enforced twice: an allow-list per profile **and**
the adapter's own `commercial` flag, `LicenceError` otherwise; `build-pack` also refuses a commercial-safe pack whose
`sources.json` lists a non-commercial source):

| Role | commercial-safe | research (adds) |
|---|---|---|
| labelled words (classifier, encoder, index, calibration) | INCLUDE, ISLRTC (non-profit terms), Mendeley emergency + agriculture words | ISL500, WSLP, CISLR |
| pre-training only (labels never read) | ISL-CSLTR sentences + the word clips without labels | GloSL-ISL, OpenHands raw, iSign, WSLP unlabelled |
| letters | Mendeley static alphabet | ISL-Fingerspelling, OpenHands fingerspelling, HF alphabet image sets |

My signs are never a training source. A source whose adapter or manifest is missing is simply skipped and reported.

**Multi-source labelled data.** `GlossLinker` maps every source's gloss onto one concept id (adapter's own concept ->
exact headword key, preferring the plain concept over domain-qualified ones -> a *unique* synonym key -> a new
`isl.<slug>` concept shared by all sources with that gloss). New concepts go to `<work>/extra_concepts.jsonl` and are
appended to the pack's vocabulary (no Hindi yet: they show English only until the lexicon covers them). Sources with
real signer ids (ISL500, Mendeley emergency) are split by **whole signers** (about 15 % test, 10 % val; at least 6
signers, else all training) and give E1c. The classifier takes every concept with >= 2 signer groups and >= 2 clips
across the training sources (INCLUDE rule unchanged), class-balanced batches (class drawn with probability ~
count^0.5, inside a class by per-source weight: INCLUDE / ISLRTC / ISL500 1.0, WSLP / CISLR 0.8, Mendeley 0.5;
ISLRTC x4 as before). Augmentation is the existing strong signer augmentation (body proportions, rotation, scale,
speed, hand chunk dropout, finger-angle noise); **left-handed signers are handled by `canonical_hand`** (the features
mirror a left-dominant signer deterministically), so no random mirroring is added (it would create left-dominant
inputs that never occur at inference). The index adds, per (concept, source), a centroid plus up to 3 per-signer
centroids, or one row per clip for one-clip-per-word sources, with per-source row weights (ISLRTC / INCLUDE / ISL500
1.0, Mendeley 0.97, CISLR / WSLP 0.95). Training sets too large to cache (ISL500 has tens of thousands of clips) use
`LazyClipDataset` (bounded LRU of body frames, about 1.3 GB at the default 12,000 items).

**Self-supervised pre-training** (`training/pretrain.py`): same `SignEncoder`, two CPU-feasible objectives:
temporal contrastive NT-Xent between two augmented crops (1-4 s) of the same sequence that are close in time
(whole-clip SimCLR views for short word clips), plus masked-frame modelling (25 % of frames in spans of 3 replaced by a
learned token after the input projection, reconstructed from the BiGRU states; since 26 Nov 2026 with a robust loss,
see 15.14). Sources are sampled ~ 1/sqrt(size)
per clip so one huge corpus does not drown the others. The checkpoint is an ordinary encoder checkpoint
(`meta.pretraining` lists sources, clip counts, config, `labels_used: false`); fine-tuning warm-starts from it
(`--init auto`, recorded in the encoder's meta). **Evaluation clips are excluded from pre-training** (INCLUDE val/test
of the primary split and the other sources' held-out signers), so test videos are not seen even without labels.
Dictionary (ISLRTC) clips are in pre-training; E2's label says so.

**Resumable training** (`training/loop.py`, used by pre-training, encoder and classifier): state (model, optimiser,
scheduler, epoch, step inside the epoch, history, best score, RNG) is written atomically every 5 minutes and every
epoch; batches are drawn from a `(seed, epoch)` RNG so a restart skips the batches already trained, by index, without
loading data (tested: an interrupted run reaches the same weights as an uninterrupted one). A state file from a run
with other settings is refused (delete it, or `--force STEP`).

**Evaluation** (`evaluation/runners.py`, `evaluate --all`): every accuracy is reported with the **not-sure rate**
(abstain), the **wrong-accept rate** (accepted and wrong / all), coverage, n, number of signer groups and a 95 %
bootstrap interval over **signer groups**. E1a (only with the official-split model; labelled not
signer-independent), E1b (primary split, + `cv/fold*/classifier.pt` models when present), E1c (other sources'
held-out signers), E2 (below), E2_seen (the old seen-class retrieval number, labelled as such), E3 (fused system, all
three policies, risk-coverage), E4 (My signs: a held-out signer enrols k = 1/3/5 takes of half of their words; gain
on their other takes and false accepts of user signs on the words they did not enrol), E5 (fingerspelling,
leave-one-signer-out, computed at training time and read from `letters.pt`), E6 (latency), E7 (ASL model files
unchanged in git), E8 (trim report + method mix).
Leakage guards: **E2** holds a fold's concepts **and every concept sharing a headword or synonym key with them** out of
all supervised training (INCLUDE, ISLRTC, ISL500, WSLP, ...); the runner refuses an encoder whose
`excluded_concepts` do not cover that set; the gallery is one ISLRTC clip per dictionary entry; queries are the
fold concepts' INCLUDE clips from the **val / test parts only** (those are also kept out of pre-training, so a query
was never seen in any form); decision thresholds are fitted on *seen-class* validation queries only; queries in a
duplicate group of any training clip are dropped.
**E1** refuses a classifier trained on another split. Duplicate groups (`DedupIndex`) join clips with the same
source-video SHA-256, the same landmark file, or a `duplicates.jsonl` cluster; training clips sharing a group with any
evaluation clip are removed before training (counts in `<work>/multisource.json`).

**Fingerspelling** (`training/train_fingerspell.py`): letter frames from ISL-Fingerspelling (middle half of each
annotated letter; space segments and frames outside letters = noise), OpenHands letter / digit clips and static
alphabet images; the existing `LetterClassifier` trained class-balanced, temperature-scaled on a held-out signer,
same checkpoint format, so `Speller` (dictionary-guided `SpellDecoder`) and the per-user alphabet calibration are
unchanged. `n_signers` counts only known signer ids (3 in ISL-Fingerspelling), so spelling stays labelled
experimental (< 5 signers) and spelled words are still only suggested, never auto-accepted.

**Fusion fix**: `FusionParams` always holds all three policies (missing ones filled from the nearest present one,
kept ordered strict >= balanced >= lenient), `fit_thresholds` always fits all three, and `fit-fusion` asserts it, so a
partial `fusion.json` can no longer make `decide()` raise (HTTP 500).

**CPU time budget (honest estimate, not a measurement of a full run).** `cli bench` (2 threads, model size of the
full run: BiGRU hidden 192, 2 layers, 256-d embedding) measured on 25 Nov **while about 24 processes of the
ingestion jobs shared the 4 CPUs** (load average ~24): pre-training 10-12 s per step (48 pairs), encoder 5.6-7.5 s
per step (64 clips, 12k-class ArcFace), classifier 2.5-4.6 s per step (32 clips), augmentation + features 4-8 ms per
clip. The same classifier step measured 0.24 s on 25 Nov morning with ~3.5 CPUs busy (section 11), so today's numbers
are 10-20x slower than a free machine. Full-run defaults and the expected wall time **once ingestion has finished**
(assuming ~0.25-0.3 s per classifier step, ~0.6 s per encoder step and ~1 s per pre-training step for the model, plus
data loading of ~5 ms per clip; the driver logs `bench` first so the budget can be recomputed as steps x seconds):

| Step | Default size | Steps | Expected wall time |
|---|---|---|---|
| dedup, vocab, splits, link | all sources | - | 0.5-1.5 h (dedup fingerprints every clip) |
| pretrain | 20 epochs x 400 steps x 48 pairs | 8,000 | 3-4 h |
| encoder | 40 x 400, P=32 K=2 (early stop, patience 10) | <= 16,000 | 3-4 h |
| oneshot | 2 folds x 20 x 300 | <= 12,000 | 2.5-3 h |
| classifier | 40 x 300 x 32 (patience 10) | <= 12,000 | 1-1.5 h |
| classifier, research (15.38) | 40 x 800 x 32 (patience 8), quota + worst_source | <= 32,000 | about 2.2 h |
| cv | 4 more folds of the classifier | <= 48,000 | 4-6 h (skip with `CV_FOLDS=""`) |
| cv, research (15.38) | 4 folds x 40 x 800, quota + worst_source | <= 128,000 | about 9.2 h (skip with `CV_FOLDS=""`) |
| e1a | 40 x 120 | <= 4,800 | 0.5 h |
| fingerspell | MLP, 40 epochs + 2-3 leave-one-signer-out refits | - | < 0.5 h (feature extraction dominates) |
| index, fusion, evaluate, index_final, pack | all references | - | 1.5-2.5 h |
| **total per profile** | | | **about 17-24 h with cv, 13-18 h without**; research is the longer one (more clips to index / dedup / pre-train on) |

Run it with `--threads 4` after ingestion ends (about 1.5x faster than 2 threads), and `--step-minutes 120` if the run
has to share the machine (each training call then yields every 2 h and continues). Memory: the lazy dataset keeps at
most 12,000 body frames (~1.3 GB); validation feature caches ~0.4 GB; peak RSS expected under 4 GB. Disk: work folder
and packs well under 1 GB per profile.

**Smoke run of the whole driver (25 Nov, 15:15-16:20 UTC, partial data, load average ~24 from the ingestion jobs).**
`scripts/phase2_train.sh --profile research --smoke` and `--profile commercial-safe --smoke` both completed all 18
steps (trim, dedup, vocab, splits, link, bench, pretrain, encoder, oneshot, classifier, cv, e1a, fingerspell, index,
fusion, evaluate, index_final, pack) with tiny models (hidden 32, 1-2 epochs, a few batches), at most 300 clips per
extra source and 400 ISLRTC clips; about 20-25 min per profile, most of it evaluation and fusion evidence collection.
Data seen: INCLUDE 2,380 clips in the refreshed pseudo-signer split (795 / 550 / 1,035 in research), ISL500 and WSLP
300 each (research), Mendeley agriculture 2, letters: OpenHands fingerspelling 172 clips, ISL-Fingerspelling 4
segments (1 signer), Mendeley static 7 images. What the run proved: every command and hand-over works on real files
for both profiles; the commercial-safe pack lists only commercial sources (INCLUDE, ISLRTC, ISL-CSLTR, Mendeley) and
the research pack every source used; E1b gives a cv-pooled signer-grouped interval; E2 ran with 47 held-out concepts
(after synonym expansion) and 0 duplicate queries dropped; E7 confirmed the ASL model files are unchanged; E1c and E5
reported "not run" with the reason (fewer than 6 ISL500 signers and 1 fingerspelling signer ingested so far).
**All accuracy numbers of the smoke reports are near chance (E1b top-1 about 0.01-0.02) because the models are tiny
and trained for seconds; they are labelled SMOKE RUN and are not the app's accuracy.** Smoke outputs (work folders,
private ISL roots, `pack/` folders) are under `artifacts/isl/phase2-smoke/` (git-ignored, never loaded by the app);
logs `data/isl/logs/phase2_<profile>_smoke.log`.

**Open issues found while building this:**

* **My signs inside the fusion** (done in 15.4): a user sign is not a classifier class, so in the fused softmax it
  gets the classifier's non-core penalty (`noncore_logp`) by default; `fit-fusion` now compares that with a neutral
  classifier term (`user_cls_mode`) on validation few-shot data and keeps the penalty when that data is too small.
* E1c, E5 and the per-signer interval of E1b depend on data still being ingested (ISL500 users, ISL-Fingerspelling
  signers); the runners report "not run" until then.
* New concepts linked only from WSLP / ISL500 / Mendeley glosses have English text only (no Hindi) until the Hindi
  lexicon is regenerated for them (`hindi --translate` on the pack vocabulary, then review).
* The refreshed `vocabulary.jsonl` is written to the work folder; copy it to `models/packs/isl/` and commit it after
  review (the consistency test needs >= 95 % Hindi coverage).

### 15.3 Reality check of the data expansion and the phase-2 driver (Reality Checker, 25 Nov 2026, 16:37-17:30 UTC)

**Verdict: NEEDS WORK (small, can be done while ingestion runs).** The pipeline itself is real: a fresh smoke run of
both profiles at commit `35b4be6` (old smoke outputs moved to `artifacts/isl/phase2-smoke.prev_1638/`, new private ISL
roots built from today's manifests) completed all 18 steps, the commercial-safe pack lists only CC BY / GODL sources,
the shared `data/isl/derived`, `data/isl/splits` and `artifacts/packs/isl` were not touched, and the backend suite passes
(375 passed). Times: commercial-safe 16:38-16:55 UTC, research 16:55-17:07 UTC, 0 failed steps. The research run read
the real pose sources (GloSL 61,196 clips, ISL500 35,809, OpenHands raw 11,786, WSLP 682 at that moment) and its E1c
ran for the first time (ISL500 held-out signer, n = 7). All smoke numbers are near chance and labelled SMOKE RUN. Minor:
the research `pack.json` licence list also names `cislr` and `isign`, which contributed no clips.
What is not ready: the evaluation report can print a "95 % interval" over one or two signer groups, nothing
stops a full (non-smoke) run from starting on partial data, and ingestion ends about 4-9 hours later than the agents reported.

**Evidence, not claims** (measured with `ps`, manifest counts every 5 minutes for 26 minutes, logs, `df`, `dmesg`):

* Running and growing: INCLUDE, ISLRTC, GloSL, ISL500, OpenHands raw (restarted 16:26 after a memory-group OOM kill at
  15:53 that `dmesg` confirms: 6.9 GB resident), the video queue on WSLP. OpenHands fingerspelling finished (172/172).
  No error lines except WSLP HTTP 429s (retried) and ISL500's hands-only clips (failed by design). Counts, rates, ETAs
  and the disk projection are in `docs/DATASETS_ISL.md` section 3.5.
* **ISLRTC is the critical path**: 5.7 videos/min under today's load (17/min this morning), 8,330 left at 17:03, so it
  ends about **04:00-08:00 UTC on 26 Nov**, not 00:20-01:00. The video queue stays on 1 worker until INCLUDE and
  ISLRTC have both ended, so WSLP ends about 03:00-07:00 and the rest of the queue (letters, Mendeley, CSLTR, HF
  alphabets, WSLP unlabelled) about **14:00-18:00 UTC on 26 Nov**.
* Memory: the jobs' memory group sat at 12-14 GB of its 14.3 GB limit (6.0-6.7 GB resident, the rest page cache); a
  second leak like the OpenHands one would again kill a random job of the group, the protected INCLUDE / ISLRTC jobs
  included.
* Licences: every adapter's `commercial` flag read on the real manifests is right (DATASETS_ISL.md section 1, "What
  phase-2 training actually uses"). `profiles.py` still lists only include / islrtc (+ cislr, isign, wlasl); training
  reads `multisource.py`, which is correct. `isl_csltr_words` is ingested but used by no profile.
* Leakage guards that exist and are tested: E1 refuses a classifier trained on another split; E2 refuses an encoder that
  saw held-out concepts (synonyms included); pre-training excludes every evaluation clip and its whole duplicate group;
  `build-pack` refuses non-commercial sources in a commercial-safe pack. Weak spots: `check_classifier_split` passes a
  classifier whose meta has no `split`; `leakage_guards.pretraining_excludes_eval_clips` is written as a constant `True`,
  not checked against the pre-trained checkpoint; the OpenHands Sign Library overlap (up to 6.8 h) is not deduplicated.

**Must fix before the full run reaches its `evaluate` step** (training steps can start meanwhile; the driver resumes):

1. **Intervals over too few groups.** `grouped_bootstrap` (`evaluation/report.py`) returns an interval whenever there are
   2 or more groups. The fresh commercial-safe smoke report prints E1b "95 % CI 0.006-0.007 (cv pooled)" from **2**
   signer groups. At full settings E1b pools 5 pseudo-signers, E1c holds out about 15 % of ISL500's 15 signers (2) and
   of Mendeley's 26 people (about 4): such intervals look precise and are not. Return no interval (or mark it "not
   reliable") below about 5 groups, and print the group count next to every interval in `report.md`.
2. **No partial-data guard.** A non-smoke `scripts/phase2_train.sh` started while ingestion runs would label its report
   "Measured on held-out data" and copy it into the pack. Add a check at the start of non-smoke runs (refuse while
   `pgrep -f 'scripts/ingest_isl(_pose|_video)?\.py '` matches, unless `--allow-partial`, and then write
   `partial_data: true` into `report.json` / `pack.json`). Do not make it wait for `isl_queue.sh` itself: after its list
   the queue only polls CISLR every 6 h for 14 days.
3. After ISLRTC ingestion ends, re-run `python -m app.isl.data.trim_bench run --learned ../artifacts/isl/trim_boundary.pt`
   (from `backend/`) before the driver: the driver's `trim` step is rules-only, so about 8,800 ISLRTC clips would get
   the rules-only end (section 7 of DATASETS_ISL.md).

**Realistic timeline on this 4-CPU machine** (estimates; the per-step times of the full run have never been measured on
a free machine: `bench` at load 18-21 gave pre-training 9.9 s, encoder 4.4 s, classifier 5.4 s per step at full size,
and the 15.2 budget assumes 10-20x faster once the CPUs are free):

| When (UTC) | What |
|---|---|
| 25 Nov ~19:00-21:30 | GloSL, ISL500, INCLUDE done; OpenHands raw reaches its 12 GB budget between ~20:00 and 08:00 next day |
| 26 Nov ~04:00-08:00 | ISLRTC done (critical path); WSLP done about the same time |
| 26 Nov ~14:00-18:00 | rest of the video queue done (then it only polls CISLR) |
| 26 Nov evening | data refresh (trim with the learned end, then the driver's dedup / vocab / splits / link: 0.5-1.5 h) |
| 26 Nov ~20:00 -> 27 Nov ~09:00-20:00 | commercial-safe profile (15.2 estimate 13-24 h) |
| -> 28 Nov ~02:00 to 29 Nov ~02:00 | research profile (15.2 says 17-24 h; allow up to 30 h: more data to dedup, index and pre-train on) |

So the first full-settings, honestly evaluated packs can exist around **28-29 Nov**, if nothing fails and nothing else
loads the CPUs. Setting `CV_FOLDS=""` saves 4-6 h per profile but leaves E1b without a signer-level interval.

### 15.4 Fixes after the reality check (AI Engineer, 25 Nov 2026, 17:40-19:55 UTC)

**Status: every item of the 15.3 "must fix" list and of the code review is fixed and tested offline; nothing was
trained at full settings.** Code commits `f1c188a` and `b27df44` (smoke-run follow-ups). Backend suite: **417 passed**
(`SLR_TORCH_THREADS=2`, nice 10, 101 s while the ingestion jobs loaded the CPUs). New tests:
`backend/tests/isl/test_phase2_guards.py` (19 tests, synthetic data, no network; the driver's guard is tested through
`scripts/phase2_train.sh --check-only` with a fake process table) and extra tests / assertions in `test_eval_runners.py`.

| # | Problem (15.3 / review) | Fix | Where |
|---|---|---|---|
| 1 | "95 % CI" from 2 signer groups | `grouped_bootstrap` returns no interval below **5** groups; JSON keeps `null` plus `interval_reasons` (`"too few signer groups (2)"`) and `interval_groups`; `report.md` prints `0.612-0.701 (7 groups)` or `none: too few signer groups (2)` in every interval cell (E1a/b/c, E2, E3 per policy, E4, E5, cv pooled, phase-1 `results.md`) | `evaluation/report.py` (`MIN_INTERVAL_GROUPS`, `interval_fields`), `runners.py`, `protocols.py`, `train_fingerspell.py` |
| 2 | a full run could start on partial data and say "Measured on held-out data" | `python -m app.isl.cli readiness --profile P`: (a) no process runs `ingest_isl.py`, `ingest_isl_pose.py`, `ingest_isl_video.py` (read-only `status` / `sources` / `probe` ignored) or `isl_queue.sh` - the queue counts only until its log says "all listed sources processed" (afterwards it only polls CISLR, which must not block training: the 15.3 concern); (b) every manifest the profile reads has a row for every expected item (item lists `manifests/<source>_items.json`, `islrtc_index.json`, the GloSL id list, published sizes; gated CISLR / iSign without access = unavailable, not incomplete; OpenHands raw = complete once its budget-limited job ended). The driver runs it before every full run and **refuses (exit 4)** unless `--allow-partial`, which writes `<work>/partial_data.json` (sticky until `--from trim`); `report.json` / `results.json` / `pack.json` then carry `partial_data: true`, the label is "PARTIAL DATA ... NOT the accuracy of the app" and the markdown title ends "- PARTIAL DATA". `--check-only` runs only the guard. Measured at 18:00 UTC: not ready (8 ingestion processes; include 3,401/4,292, islrtc 5,642/13,665, isl500 56,200/86,756, glosl 86,222/126,846, wslp 912/4,398, the rest of the queue barely started) | `training/readiness.py`, `cli.py readiness`, `scripts/phase2_train.sh` |
| 3a | `check_classifier_split` passed a classifier without `split` | refused (`LeakageError`); E1 then reports `run: false, reason: "refused: ..."` instead of a number | `runners.py` |
| 3b | `pretraining_excludes_eval_clips` was a constant `True` | `pretrain` writes the exact clip list `<work>/pretrain_clips.txt` (union over resumed parts; count + SHA-256 in the checkpoint meta); `evaluate` intersects it with every evaluation clip **and its duplicate group** (and with the Sign Library content matches): `true` / `false` (with examples, and a LEAKAGE line in `report.md`) / `null` with the reason (no pre-training, or a checkpoint from before this change) | `multisource.py` (`eval_clip_ids`, `write_pretrain_list`), `runners.pretraining_guard` |
| 4 | two drifting profile definitions; adapters not registered; licence lists naming sources with 0 clips | the source lists live once in `profiles.py` (`WORD_SOURCES`, `PRETRAIN_SOURCES`, `LETTER_SOURCES`; `DataProfile` is built from them, `multisource` re-exports them). **WLASL removed** (it is American Sign Language; no ASL warm start in phase 2). **`isl_csltr_words` not used** (word labels are valid, but each item is one still frame of a moving sign: the word models read sequences, the letter model reads letters; `profiles.NOT_USED` says why). Video and pose adapters registered in `sources.ADAPTERS` (lazily, `register_optional_adapters`: `VIDEO_ADAPTERS` + `pose_sources.register`); a test checks every listed source has an adapter and that commercial-safe agrees with `catalog.profile_sources` / `pose_sources.PROFILE_SOURCES`. `sources.json` now records per role the clips that **really reached** the build (after caps, sampling, dedupe); `build-pack` lists only sources with > 0 (no `cislr` / `isign` while they contribute nothing); INCLUDE / ISLRTC are always named because the vocabulary is built from their word lists (role `vocabulary` when no clip of theirs was used) | `profiles.py`, `data/sources.py`, `cli._record_sources`, `pipeline.pack_licences`, `models/packs/isl/pack.json` |
| 5 | OpenHands Sign Library overlap "up to 6.8 h" | pose-content matcher (`training/pose_dedup.py`, new driver step `pretrain_dedup`), measured first: see below | `pose_dedup.py`, `cli pretrain-dedup`, `cli pretrain` |
| 6 | the driver's trim was rules-only; truncated clips used | `trim` loads `artifacts/isl/trim_boundary.pt` when present (the same `trim_learned.refine_cap` call as `trim_bench run --learned`), records `trim_version = trim-2+trim-learned-1` (rules only: `trim-2`) and redoes every clip whose version or landmark file differs, so new clips are never silently rules-only; `--learned-model none|PATH` overrides. Re-extracted ISLRTC rows (`manifests/islrtc_recut.jsonl`, card-v2 detector, written by the ingestion agent) replace the original file in `trim` and in training (`training.common.resolve_islrtc`; the trim is used only if it was computed on the new file, recorded as `landmarks`); a clip whose trim is `truncated` and that has no re-extracted file is **excluded** (training, index, galleries). `data.summary()` reports `islrtc_reextraction` and `islrtc_trim_versions` | `cli.py trim`, `training/common.py` |
| 7 | E2 did not say what pre-training saw | E2 label: "self-supervised pre-training saw ISLRTC clips WITHOUT labels, including the dictionary reference clips E2 retrieves against (n of N ISLRTC clips were in the pre-training list)"; `pretraining_saw_gallery_unlabelled` in the JSON | `runners.py` |
| 8 | My signs got the classifier's non-core penalty | `FusionParams.user_cls_mode`: `penalty` (default, as before) or `neutral` (a user key's classifier term = the crop's best classifier log-probability: the classifier can neither vote for nor against it). `fit-fusion` runs the E4 protocol on the **validation** signers for both modes (k = 3, else k = 1) and picks `neutral` only if enrolled-word top-1 rises and the user-sign false-accept rate rises by <= 0.01; with < 30 enrolled test takes or < 2 groups it keeps `penalty` and says why (`fit-fusion` report `my_signs`). E4 in the report names the mode | `models/fusion.py`, `training/fit_fusion.py`, `runners.e4_my_signs` |

**Sign Library: measured, then decided (item 5).** The two files of the channel cannot be matched by name, so chunks
are matched by pose content: 7 upper-body points (x, y) per frame; a query chunk copies a reference chunk when, for a
large enough fraction of its consecutive frame pairs, both frames have one of their 4 nearest reference frames within
0.04 (normalised units) *in that same chunk* and the reference index moves forward 0-2 frames (time order separates a
re-cut copy from another video of the same presenter in the same studio).
* Real data (25 Nov, 9,510 `sub_based` chunks against the 341 usable one-minute chunks, nearest frame only, eps 0.05):
  **16 chunks** match with time-order scores 0.5-1.0 at median frame distances of 0.014-0.023 (true copies); every
  other chunk scores <= 0.33 with median distances >= 0.035. So the real overlap is a few minutes, not 6.8 h.
* Calibration (run again inside every `pretrain_dedup` step): 300 synthetic re-cuts of one-minute chunks (a random
  2-5 s window re-sampled half a frame off the stored frames, plus 0.003-0.005 landmark jitter, as a second
  extraction / another stride offset gives) and 300 windows of other videos queried against an index without their
  own video. At eps 0.04: recall 0.977 (jitter 0.003) / 0.967 (0.005) at a score threshold of 0.2 with a
  false-positive rate of 0.010; above 0.4 the false-positive rate is 0.003 (one "negative" window scores 1.0 at every
  threshold: most likely the same video published under two names). The step picks the threshold with the best recall
  among those with a false-positive rate <= 1 % (never below 0.2) and calls the dedupe **reliable only with recall >= 0.95**.
* Decision: reliable -> **drop only the matched chunks** (copies of a one-minute chunk, and any chunk that contains an
  evaluation clip: INCLUDE val/test and the other sources' held-out signers are queried against every Sign Library
  chunk). If the calibration is not reliable on the data of a run, or the step has not run, **the whole Sign Library
  channel is left out of pre-training** (`apply_dedup`).
* Full run on today's data (25 Nov 18:00-18:27 UTC, research profile, split `include-session-cv5-fold0`, every
  ingested Sign Library chunk: 341 one-minute + 9,510 other, 1.52 M frames; 1,474 s at load ~24): calibration
  **reliable** (recall 0.997 at threshold 0.15 -> used 0.2, false-positive rate 0.010); **19** chunks copy a one-minute
  chunk (scores 0.2-0.94, 16 of them >= 0.5, the same 16 as the first search); of **16,474** evaluation clips
  (INCLUDE val/test, ISL500 held-out signers, ...) **1** matched a Sign Library chunk (ISL500 `user007` "Town",
  score 0.25: most likely a false positive near the threshold, dropped anyway). So 20 chunks are dropped. This was a
  one-off run into a scratch work folder; the driver's `pretrain_dedup` step repeats it for every full run.
  In the smoke runs (600 chunks, 100 controls) the calibration had too few controls to show recall >= 0.95 at a
  false-positive rate <= 1 %, so the fallback was exercised: the whole channel was excluded.
* Not checked by content: GloSL, NIOS, ISH News against the evaluation sources (sentence-level news / education
  videos; the evaluation clips are studio recordings that are not published on these channels) and GloSL against
  OpenHands NIOS / ISH News (possible double weight in pre-training, not a leakage into evaluation).

**Smoke re-run of both profiles after the fixes.** The outputs of 16:38 were moved to
`artifacts/isl/phase2-smoke.prev_1758/` (stamps / logs `*.prev_1758`). A first re-run (17:58-19:02 UTC, kept in
`phase2-smoke.run1_1903/`, logs `*.run1_1903`) passed all 19 steps but showed three things to fix (commit `b27df44`):
`sources.json` gave a read-but-unsampled source its read count, the research E2 had no queries (the smoke cap of 400
ISLRTC clips had dropped the held-out concepts' dictionary clips) and fit-fusion spent minutes on E4 evidence with one
validation signer group. The final run, `scripts/phase2_train.sh --profile both --smoke` from fresh work folders and
private ISL roots (manifests of 19:03 UTC, load average 22-26 from the ingestion jobs), **completed all 19 steps of
both profiles with 0 failures**: commercial-safe 19:03:22-19:29:05 UTC (25.7 min), research 19:29:10-19:49:21 UTC
(20.2 min). Seconds per step (commercial-safe / research): trim 323 / 268 (every clip re-trimmed once with the learned
end: 5,896 / 6,078 clips, 474 / 492 learned ends, 22 / 9 still `truncated`), dedup 61 / 48, vocab 19 / 18, splits 19 / 20, link 18 / 23,
bench 29 / 18, pretrain_dedup 25 / 58, pretrain 37 / 38, encoder 50 / 35, oneshot 51 / 35, classifier 55 / 29, cv
38 / 32, e1a 28 / 26, fingerspell 7 / 10, index 37 / 33, fusion 252 / 162, evaluate 443 / 321, index_final 46 / 33,
pack 4 / 4. What the reports show (all numbers near chance, labelled SMOKE RUN, never the app's accuracy):

* every interval cell is either `lo-hi (k groups)` (only E1a, 5 groups) or `none: too few signer groups (k)` (E1b and
  its cv pool: 2, E1c: 1, E2: 2, E3: 1-3, E4: 1); both packs and reports carry `partial_data: true`;
* `pretraining_excludes_eval_clips: true`, computed: 0 of 200 pre-training clips are among the 2,500 / 2,714
  evaluation clips and their duplicates; E2 ran with 173 queries per profile and its label says 14 / 1 of the 400 ISLRTC
  clips were in the pre-training list;
* ISLRTC: 66 / 75 clips read from their re-extracted (`islrtc_recut`) files, 13 / 9 truncated clips without one left
  out, every trimmed clip `trim-2+trim-learned-1`;
* research `pretrain_dedup` with 100 controls was not reliable (recall 0.87 at a false-positive rate of 0.008), so
  the whole Sign Library channel was left out of that smoke pre-training (the fallback); commercial-safe has no
  OpenHands data;
* fit-fusion kept `user_cls_mode = penalty` in both: "insufficient validation few-shot data (1 signer groups with
  repeated takes; need 2)";
* pack licences: commercial-safe `include, islrtc, mendeley_agri, mendeley_static`; research adds `glosl, isl500,
  isl_fingerspelling, openhands_fs, openhands_raw, wslp` with per-role clip counts - no `cislr` / `isign` /
  `mendeley_emergency` (0 clips used).

**Still open:** `docs/DATASETS_ISL.md` section 1 "Data profiles" still shows the old `profiles.py` table (owner: Data
Engineer; the content is now the "What phase-2 training actually uses" table there, minus WLASL). The data guard is a
start-of-run check: a run started with `--allow-partial` keeps its PARTIAL DATA mark even after ingestion ends; rebuild
with `--from trim` for clean numbers.

### 15.5 Adversarial review of the 15.4 fixes (Code Reviewer, 25 Nov 2026, 19:50-20:01 UTC, commits `2c649f9`, `5bb2c19`)

Every claimed fix was re-read and, where possible, checked against the real data. Confirmed: intervals below 5 signer
groups, the readiness guard (exit 4 on real data now), the classifier split guard in E1, the computed pre-training
exclusion, one source list (`profiles.py`), learned trim versions, the ingestion locks / memory guard / process scan,
and the running jobs (manifests growing, no deleted-but-open files, 9.5 GB RAM available). Six real gaps, fixed with
offline tests (`tests/isl/test_review_round2.py`, one more in `test_eval_runners.py`):

| Gap | Effect | Fix |
|---|---|---|
| `IslrtcAdapter` dropped rows with 0 frames / failing quality **before** the `islrtc_recut` row could replace them | the 260 opening-intro / false-card clips that the recut job re-extracted never reached training (real data: 6,014 ISLRTC clips; 6,277 with the fix) | the adapter applies the recut rows first (`overrides`), `load_isl_data` passes them |
| E3 and E4 used the classifier that E1b refused | a classifier from another split, or with no split, still produced E3 / E4 numbers | E3 / E4 are refused with E1b (`classifier_refusal`); `leakage_guards.classifier_refused` and the report say so |
| `apply_dedup` kept every Sign Library chunk not named in `drop_ids` | chunks never checked (ingested after the `pretrain_dedup` step, outside a smoke sample, or all of them when the step found none) went into pre-training | the dedupe records `checked_ids`; unchecked chunks are dropped (`unchecked_dropped`) |
| the readiness guard ignored `ingest_isl_gated.py` | after the queue's list is done (CISLR / iSign polling only), an iSign conversion run would not block a full run | `ingest_isl_gated.py cislr|isign` blocks; its `--check` probe does not |
| readiness expected 4,292 INCLUDE rows (published size) | the INCLUDE job finished at 19:59 UTC with all 44 zips and 4,284 distinct clip ids, so the guard could never turn ready | INCLUDE is complete once all 44 zips are in `include_zips_done.txt` |
| `pack_licences` read every row of a reused `sources.json` | a source removed from the profile (e.g. WLASL) could still be named in a research pack | rows outside the profile are not listed; the commercial-safe refusal still reads every row (fails closed) |

Still open (not fixed here): the Sign Library calibration uses synthetic re-cuts of the *same* OpenHands extraction, so
its recall says nothing about matching our own MediaPipe extractions of evaluation videos against OpenHands poses (a
different extractor and normalisation); `choose_user_mode` picks `neutral` on any top-1 gain
(no noise margin with as few as 30 validation takes) - fixed in 15.6.

### 15.6 Final reality check before the full run (Reality Checker, 25 Nov 2026, 20:00-20:45 UTC)

**Verdict: READY for the commercial-safe full run as soon as its data guard passes (estimate 26 Nov, 12:00-17:00
UTC). The research run is ready in code but its data will not be complete for days unless the OpenHands raw job is
stopped or fixed (one decision, below).** Nothing was trained at full settings; no ingestion job was touched.

**Evidence.**
* Backend suite: 425 passed at `5bb2c19` (20:02), **431 passed** after the fixes below (`SLR_TORCH_THREADS=2`, nice 10).
* Fresh smoke run of both profiles (`scripts/phase2_train.sh --profile both --smoke`; the 19:03 outputs moved to
  `artifacts/isl/phase2-smoke.run2_2002/`, stamps / logs `*.run2_2002*`): **all 19 steps, 0 failures** in both:
  commercial-safe 20:02:16-20:15:42 UTC (13.4 min), research 20:15:46-20:29:03 (13.3 min); load average 15-20.
  Seconds per step (commercial-safe / research): trim 231 / 162, dedup 42 / 40, vocab 15 / 14, splits 20 / 17, link
  16 / 19, bench 20 / 11, pretrain_dedup 15 / 45, pretrain 21 / 27, encoder 39 / 27, oneshot 34 / 26, classifier
  26 / 26, cv 28 / 28, e1a 22 / 21, fingerspell 4 / 5, index 23 / 26, fusion 98 / 118, evaluate 122 / 147,
  index_final 27 / 35, pack 3 / 3. Trim: 6,380 clips `trim-2+trim-learned-1`, 517 learned ends, 344 clips read from
  `islrtc_recut`, 8 truncated clips without a recut left out. Reports: every interval is `lo-hi (k groups)` with
  k >= 5 (E1b cv pool 8 groups, research E3 6 groups) or `none: too few signer groups (k)`; `partial_data: true`
  and the SMOKE label everywhere; pre-training excludes evaluation clips: true (computed); research Sign Library
  dedupe not reliable with 100 controls -> whole channel excluded (the designed fallback). **fit-fusion chose
  `user_cls_mode = neutral` from data for the first time** (4 validation groups, 87 enrolled takes: top-1 with
  enrolment 0.126 -> 0.632, user-sign false accepts +0.008); with a near-chance smoke classifier this mostly says
  the penalty mode buries user signs - the full run decides again. All numbers near chance, never the app's accuracy.
* Data guard on real data: `scripts/phase2_train.sh --profile commercial-safe|research --check-only` -> exit 4 for
  both (running jobs plus incomplete manifests: ISLRTC 6,650 / 13,665, WSLP, the queue's later sources, HF alphabet
  listings, OpenHands raw). Ingestion rates and ETAs measured at 20:03-20:26 are in DATASETS_ISL.md 3.6.

**Found and fixed (commits `a844f63`, `a0f954f`, `e744583`, `b00fe99`; tests in
`backend/tests/isl/test_readiness_retry.py`):**

| Gap | Effect | Fix |
|---|---|---|
| readiness counted every failed row as attempted | a download that failed in the job's last pass (curl / HF 429, written as `ok: false`), or the 3 `superseded` Mendeley rows, would pass as "complete" while the next run would redo them | rows whose last record is `fetch: ...`, `download failed ...` or `superseded ...` count as missing (`retry_pending`); permanent failures (ISL500 without WholeBody) still count |
| a public source with no rows and no item list was "not available" | the two HF alphabet sets, whose listings failed with 429 / 503 in the probe runs: if the queue gives up on them (5 tries) the guard would have passed silently | incomplete ("listing failed or never run"); only CISLR / iSign are "unavailable" |
| any ingestion process blocked every profile | the OpenHands raw job (research only, days from its end, DATASETS 3.6) would have held back the commercial-safe run too | a job of one known source blocks only the profiles that read it; `ingest_isl.py`, the queue and unknown jobs still block all |
| `build-pack` copied `<work>/results.json` | phase-2 work folders have no such file (the report is `eval/report.json`), so a phase-2 pack served no results to the app; and the full run reuses `artifacts/isl/commercial-safe/`, which still holds the **phase-1 smoke** `results.json` of 11:57 - the full pack would have shipped and served it | the pack's `results.json` is the phase-2 report when one exists (verified on both smoke packs: identical to `eval/report.json`) |
| `choose_user_mode` picked `neutral` on any top-1 gain | one lucky take of 30 decided the mode | needs a gain >= 0.02 **and** >= 2 more correct enrolled takes (replaced by a paired sign test in 15.27) |

**Still open (not blocking the commercial-safe run):**
* *(Closed 21:13 UTC, see 15.7 and DATASETS_ISL.md 3.7: the job now reads each member in one forward pass; the
  "30 MB per chunk" was a logging artifact, the real cost was about 16 MB.)* **OpenHands raw reads about 30 MB per 130-frame chunk** (10.7 chunks/min, 5-8 MB/s fetched; 84,129 ISH News
  `sub_based` chunks left, then MBM), so its job, and with it research readiness, would end around 1-2 Dec or later.
  Ingestion owner: make it read in data-address order / keep the inflated window; or the main session stops it
  (SIGINT, resumable) and accepts the 16.7k chunks already ingested (all four channels are represented).
* *(Closed by `7508274` / `45aee49`, see 15.7.)* Frontend (`LanguageEvaluation.tsx`) renders the phase-1 protocol names: with a phase-2 report it shows E2, E3
  and E6 but not E1a/b/c, E4, E5, E7, E8, the interval reasons or a `partial_data` warning (only `smoke` gets a
  notice). A `--allow-partial` pack would therefore not be flagged in the UI; do not ship one.
* Carried over: the Sign Library eval-clip matching is uncalibrated across extractors (15.5); GloSL / NIOS / ISH
  News not content-checked against evaluation clips; a recut that lands after the trim step is used untrimmed until
  the next trim; the learned trim end (`artifacts/isl/trim_boundary.pt`) was trained at 16:20 on 4,832 clips.

**Commands for the main session, in order** (from the repository root; never start a second copy of a running job):

1. Ingestion is complete for a profile when
   `scripts/phase2_train.sh --profile commercial-safe --check-only; echo $?` prints 0 (same for `research`).
   The signs behind it: `data/isl/logs/islrtc.log` ends with `islrtc done: total ok ...`; `islrtc_recut.log` with
   `islrtc-recut: finished`; the queue has no source of the profile left (`queue.log`: every source the profile
   reads logged `finished` / `giving up`, or `all listed sources processed`); `pgrep -af 'scripts/ingest_isl'`
   shows no job for a source of the profile; `artifacts/isl/<profile>/data_readiness.json` lists no reasons.
   *(Since 25 Nov 22:20 UTC, `readiness.queue_progress`.)* A running queue blocks only the profiles that read a source
   it still has to process: its list is the `queue start (pid N): ...` line of that pid, checked against the
   `SOURCES` default of `isl_queue.sh` (or the process's `SOURCES`), progress is its per-source log lines, its running
   child counts, and CISLR / iSign count until they logged `finished` (the queue may start them before any chunk).
   So commercial-safe is no longer held back by the research-only end of the list (`isl_csltr_words`, the two HF
   alphabet sets, WSLP unlabelled). Anything unreadable or inconsistent (no start line for that pid, another list, an
   unknown name or child) blocks both profiles; `processes[].queue` in `data_readiness.json` shows `remaining` or
   the `error`. `ingest_isl_video.py list <source>` blocks only the profiles of that source. A gated source (CISLR,
   iSign) left partly ingested by a stopped job is `partial (gated; job not running)`: not a reason (research uses
   the rows that exist), but listed under `warnings` / `partial_gated` there and in the report's `data.readiness`
   ("Data readiness" section of `report.md`).
   If the queue gave up on a source (`queue.log`: `giving up`), re-run that one source, e.g.
   `nice -n 10 .venv/bin/python scripts/ingest_isl_video.py run hf_akrit_alphabet --workers auto --min-free-gb 10`,
   then check again.
2. Move the phase-1 smoke work folder aside (classifier / encoder / index / results of 25 Nov 11:57):
   `mv artifacts/isl/commercial-safe artifacts/isl/commercial-safe.phase1_1157`
3. Commercial-safe full run (about 13-24 h, 15.2; resumable - re-run the same command after any stop):
   `nohup scripts/phase2_train.sh --profile commercial-safe --threads 4 >> data/isl/logs/phase2_driver.out 2>&1 &`
   Progress: `scripts/phase2_train.sh --profile commercial-safe --list`, `data/isl/logs/phase2_commercial-safe.log`.
4. Research: decide on OpenHands raw (above). If stopping it: `kill -INT $(pgrep -f 'ingest_isl_pose.py openhands-raw')`,
   wait until it has exited, then `scripts/phase2_train.sh --profile research --check-only` must print 0, and, after
   the commercial-safe run has finished (one profile at a time; both write the shared ISLRTC trims):
   `nohup scripts/phase2_train.sh --profile research --threads 4 >> data/isl/logs/phase2_driver.out 2>&1 &`
5. After each run: read `artifacts/isl/<profile>/eval/report.md` (it must not say PARTIAL DATA or SMOKE), then the
   15.1 follow-ups (commit the reviewed vocabulary, Hindi for new concepts, pack size decision). Never use
   `--allow-partial` for a pack that is shipped.

### 15.7 OpenHands speed-up and eval page (Reality Checker, 25 Nov 2026, 21:33-22:10 UTC, at `45aee49`)

Read-only re-check of `0d6e7e8` (HF alphabet listings), `0bccbcf` + `0efd7b7` (OpenHands forward pass), `7508274` +
`45aee49` (Evaluation page) and `f1d92f7`, with live evidence. No job was touched; nothing was trained.

**Verdicts.** Commercial-safe full run: **NEEDS WORK only on data** (code ready; data guard exit 4; estimate **26 Nov
09:00-15:00 UTC**). Research full run: **NEEDS WORK only on data** (same queue, plus the OpenHands job, which now ends
tonight; estimate also **26 Nov 09:00-15:00 UTC**). Both are bounded by the video queue (`isl_queue.sh`), which blocks
every profile until its whole list is done, including the research-only sources at its end.

**OpenHands raw (PID 25287, pre-review code in memory).** Sampled every 60 s from `/proc/25287/{status,io}`, the
manifest, the log and `df`, 21:33-22:00 UTC:

| | before (20:33, old pass 2) | MBM data pass 21:39:50-21:57:11 (17.4 min, measured) |
|---|---|---|
| chunks | 10.7 / min (~130-frame ISH News chunks) | 6,069 chunks of a median 3,729 native frames (~2.5 min; modes 7,500-7,552 = 5 min, so "chunked_1_min" is a name, not the length) = 350 / min |
| download | 2.9 MB/s, ~16 MB per chunk for 0.64 MB of payload | 22.5 MB/s (`rchar` +22.98 GB over 17.0 min); 23.33 GB in total = the member's compressed size, i.e. **every byte fetched once**; 0 jumps, 0 fallbacks, 0 stream errors |
| inflated | - | 59.07 GB in 1,043 s = 56.6 MB/s; 1.19 cores of CPU |
| RSS | 0.69-0.87 GB | **1.11-1.20 GB** (ISH News: 0.40-0.66 GB; MBM chunks are 15-30x larger); 0.96-1.11 GB in the listing passes; guard at 1.5 GB |

The ISH News `sub_based` pass (log): 16,462 chunks in 591 s = 1,670 / min, 0.28 MB downloaded per chunk (old path:
~16 MB). So the claimed ~170x holds for small chunks; per inflated byte the job is now network-and-one-core bound.
Listing passes are the slow part: MBM listed 12,138 datasets in 1,596 s (23.32 GB fetched = the whole member a first
time). Manifest at 21:58: **63,745 rows, all ok, 0 duplicate ids, one key order**; rows 0-18,023 have `.npz` mtimes
before the 20:47:38 restart and rows 18,024-57,675 after it, contiguously, so nothing was redone. Spot checks: 3 new
ISH News rows, 3 MBM rows and 1 old row load with pose (T,33,4), hands (T,21,3), face (T,20,3) all-NaN (by design),
increasing `t_ms`; 500 sampled rows (400 new) all match `n_frames` and `hands_frac` to 0.002. No deleted-but-open
files in any job; `du -xsh /` = `df` used (41 GB at 21:34). Landmarks `openhands_raw` **10.99 GB** allocated at 21:57
(MBM 4.76 GB), 24 GB free disk. Hours ingested at 25 fps: Sign Library 35, NIOS 140, ISH News 174, MBM 273 = 622 h of
the published ~1,129 h.

ETA (estimate): NewzHook (61.24 GB compressed, **159.48 GB raw**) started listing at 21:57:14 at 17-20 MB/s, so its
listing ends about 22:45-23:05; the 12 GB budget then leaves about 1.0 GB of landmarks, i.e. roughly the first 12 GB
(about 8 %) of NewzHook **in file order (not a random sample)**, and the job ends about **22:50-23:30 UTC, 25 Nov**
(DATASETS 3.7 said 00:00-01:00). Risks: (1) the NewzHook listing starts at RSS 1.11 GB with up to 256 MB of metadata
cache, and the memory guard's `relief` still points at the previous member's stream during a listing; a stop there
restarts the 61 GB listing from byte 0 (up to 3 respawns, then the job exits and readiness treats OpenHands as
complete); (2) the running process still has the pre-review `PrefetchReader` (a failed block is never re-fetched),
so a Zenodo outage longer than `range_fetch`'s retries ends the job; restart it with the same command, it resumes.

**HF alphabets.** `manifests/hf_{akrit,hemg}_alphabet_items.json` exist: 10,118 items (33 classes, 300-418 per class,
train + test) and 10,500 (35 x 300), unique ids, `listing_status.json` "complete" for every split. Readiness now says
`hf_hemg_alphabet 0/10500`, `hf_akrit_alphabet 0/10118` (not "listing failed"). Remaining research reasons at 21:34:
five running jobs (islrtc, recut, queue, its WSLP chunk, openhands-raw), islrtc 7,597/13,665, wslp 2,213/4,398,
wslp_unlabelled 3/635, mendeley_static 8/13,800, isl_fingerspelling 4/1,308, isl_csltr 2/687, mendeley_agri 2/78,
mendeley_emergency 0/412 (3 retries), both HF sets 0, openhands_raw (job running). Commercial-safe: islrtc, recut and
the queue running, plus islrtc, mendeley_static, isl_csltr, mendeley_agri, mendeley_emergency. Note: the per-class
cap of 300 means Hemg contributes 10,500 of its 42,745 rows (25 %); "all means all" is not literally met there (most
Hemg rows are probably augmented copies, DATASETS 3.3).

**Rates for the ETA** (manifests, 21:39:59-21:58:16, plus 20:26-21:34): islrtc 12.6-13.9 / min (5,759 left -> about
**05:00-06:00 UTC**, recut final scan 30-60 min later); WSLP 6-8.5 / min on 1 worker (1,979 left -> about 02:00-03:30).
After WSLP the queue runs fingerspelling (1 worker until islrtc ends), then Mendeley static, CSLTR, agriculture,
emergency, CSLTR words, the two HF sets (20,618 images through the rows API, the least certain step) and WSLP
unlabelled on 3 workers: roughly 3.5-7.5 h of work, so the list ends about **09:00-15:00 UTC, 26 Nov**. Commercial-safe
could start 2-4 h earlier if the queue processed its four commercial-safe sources first, or if readiness let a
queue whose remaining sources are all research-only block only the research profile (not changed here).

**Eval page.** `npm run typecheck` pass, `npx vitest run` 76 passed / 1 skipped, `npm run build` pass. Re-took the
screenshots at `45aee49` (the 21:18 one predates the review fix) from the real smoke `report.json` files with
Playwright (`/api` mocked): research and commercial-safe at 1360 px, commercial-safe in Hindi at 390 px; no console
errors, `scrollWidth` = 390 at phone width. Every overview number matches the report (research: E1a 1.0 %, E1b 1.0 %,
E1c 0.0 %, E2 3.3 %, E2_seen 1.0 %, E3 8.2 % 6.1-9.9 % (6 groups), E4 76.2 % at k = 3, E5 not measured with the report's
reason, E6 14 ms, E7 yes, E8 29.3 %; commercial-safe: 0.7 / 0.8 / not measured / 3.3 / 1.0 / 6.4 / 74.1 % / not
measured / 22 ms / yes / 29.2 %); E1b cross-validation pooled 0.7 % (0.3-1.1 %, 8 groups); SMOKE + PARTIAL DATA banner
and badges shown. One misleading row, from the backend: E8's "Learned trim model used: no" is true of
`artifacts/reports/trim_eval.json` (written 11:00 by the rules-only `cli trim-eval`), but the trims the run used are
`trim-2+trim-learned-1` (530 learned ends in the research smoke sidecar). `e8_trim` reads that stale file and the
driver never re-runs `trim-eval`, so the full run's E8 synthetic check would describe an older trimmer.

**Backend suite:** 461 passed (`SLR_TORCH_THREADS=2`, nice 10, 50 s).

**For the main session.**
1. Nothing to do for OpenHands tonight unless its log shows `stopped for memory` 3 times or a traceback: then start
   it again with the same command (resumable). To get more NewzHook, raise `--budget-gb` (owner decision; about
   +1 GB of landmarks per 12 GB of NewzHook inflated, and a restart also loads the review fixes).
2. Before the full runs, make E8 honest: run `cli trim-eval` with the learned model in the driver (or have `e8_trim`
   read `trim_eval_learned.json` when the sidecar's `trim_version` contains `trim-learned`, and record the file's
   time). Until then, read E8's synthetic numbers as the rules-only trimmer's.
3. Then follow the 15.6 commands unchanged; check-only must print 0 for each profile first.

### 15.8 Honest E8: the driver evaluates the trimmer the run uses (AI Engineer, 25 Nov 2026, 22:00-22:30 UTC)

Fixes item 2 of 15.7. Before: `e8_trim` read `artifacts/reports/trim_eval.json` (written 11:00 by the rules-only
`cli trim-eval`), the driver never re-ran `trim-eval`, and `trim-eval --learned` applied the *old* stage-2 model
(`boundary.pt`), not the learned end the trim step uses (`trim_boundary.pt`, `refine_cap`). So E8 and the page said
"Learned trim model used: no" for runs trimmed with `trim-2+trim-learned-1`.

Now:
* **`cli trim-eval`** (`app/isl/data/trim_eval.py`) runs exactly the trimmer of `cli trim`: stage 1, then
  `trim_learned.refine_cap` for non-phrases (phrases keep the full clip), chosen the same way
  (`--learned-model auto|PATH|none`, default auto = `artifacts/isl/trim_boundary.pt` when present). The report
  records `trim_version` (`trim-2` / `trim-2+trim-learned-1`), the model file (path, SHA-256, time), the sidecar's
  versions / record count / time, `learned_ends` in the synthetic set, and `written_at`. `--out FILE` writes it
  elsewhere (atomic). `--learned` (old stage-2 model) is kept for comparison only, as `trim-2+boundary-stage2`.
* **Driver step `trim_eval`** (between `fusion` and `evaluate`; stamped and resumable like every step):
  `cli trim-eval --learned-model auto --out <work>/trim_eval.json` (`--n 60` in smoke runs). When it runs it drops
  the `evaluate` and `pack` stamps, so a report built before it is rebuilt.
* **E8** reads `<work>/trim_eval.json` (never the global file when a work folder is given) and shows its numbers
  only when its `trim_version` equals the single `trim_version` of the run's trim sidecar and the learned model file
  is unchanged (same SHA-256). Otherwise `trim_eval_check.status` is `mismatch` (other version, no version recorded,
  mixed sidecar versions, model file changed or gone) or `missing`, with the reason, the numbers are left out and the
  label says "Synthetic trim check NOT SHOWN (MISMATCH|MISSING): reason" (the page shows the label). The report
  records the file, its `written_at` / file time, the evaluated `trim_version` and the run's versions; `report.md`
  prints them on an explicit "trim-eval:" line. Limit: the sidecar records the trim version, not the model's hash,
  so a model replaced *between* the trim step and `trim_eval` is not detected (nothing retrains the model in a run).
* Evaluation page: the SMOKE badge and the "commit" label are i18n keys (`eval.smokeBadge`, `eval.commit`; Hindi
  "स्मोक रन", "कमिट").

Checks: offline tests `backend/tests/isl/test_trim_eval_e8.py` (statuses; E8 refuses the old rules-only report for
learned trims and reads only the work folder; `trim_one` gives exactly the sidecar cuts of `cli trim` with a learned
model; `trim-eval` applies the learned model and records its version, rules-only gives `trim-2`; the driver's
`--only trim_eval` writes `<work>/trim_eval.json`, drops the evaluate / pack stamps, and the step sits between
`fusion` and `evaluate`). Backend suite 465 passed; frontend typecheck, vitest (77 passed / 1 skipped), build pass.

Smoke check on a scratch copy of `artifacts/isl/phase2-smoke/research` + its private ISL root (the shared folders and
stamps untouched): `trim-eval --learned-model auto --n 60` 9 s, `evaluate --all` 184 s, both exit 0. The report's E8:
`trim_eval_check.status ok`, evaluated `trim-2+trim-learned-1` = the sidecar's 6,521 trims, written 22:09:42 UTC,
model `trim_boundary.pt` (16:20:46, SHA-256 `4c0efb82...`), **Learned trim model used: yes**; synthetic n 60: within
300 ms 0.717, median 0 ms, late 0.083, early 0.200 (3 learned ends); INCLUDE active frames kept 1.000 (smoke-sized
samples, not the full run's numbers). The same sidecar against the old 11:00 `artifacts/reports/trim_eval.json`
gives `mismatch` ("the report records no trim_version ... learned=False").

**Closed: the trim records carry the model hash (Data Engineer, 25 Nov 2026 23:00-23:20 UTC; the 15.9 open issue
(1)).** `cli trim` reads the learned model's bytes once, loads the model from them and stores their SHA-256 in every
trim record (`trim_model_sha256`; `null` for rules only; the run summary prints `learned_model_sha256` and
`model_hashes`). A clip is re-trimmed when that hash differs from the model used now, not only when `trim_version`
changes, so a retrained `trim_boundary.pt` (same version) re-trims everything instead of leaving two models under
one version; records without the field (all sidecars written before this change) are re-trimmed once. The full runs'
trim step re-trims every clip anyway (the shared sidecar is still `trim-2`, 4,832 records); a smoke work folder's
sidecar (6,521 records, no hashes) now gives E8 `mismatch` if `evaluate` is re-run on it alone (re-run `--from trim`).
`trim-eval` records the hash of the bytes it loaded (`learned_model.sha256`; `changed_during_eval` / `gone_after_eval`
if the file moved meanwhile) and counts the sidecar's hashes (`sidecar.model_hashes`; report schema 3). E8
(`trim_eval.check_report`, keys from `trim_model_key`: hash, `none`, `unrecorded`) is `ok` only when the run's records
carry one single hash equal to the model trim-eval evaluated (and the file is unchanged): records without a hash,
mixed hashes or another model's hash -> `mismatch` with the reason; `report.md`'s "trim-eval:" line prints both
hashes. Tests (`backend/tests/isl/test_trim_eval_e8.py`): hash statuses; `cli trim` records the hash, does nothing on
a re-run, re-trims hash-less records once, a retrained model with `--limit 2` gives mixed hashes (E8 mismatch), all
re-trimmed but not re-evaluated -> mismatch, after `trim-eval` -> ok, rules only -> null hashes; `trim-eval` keeps
the loaded model's hash if the file changes or goes during the run.

### 15.9 Adversarial review of `d5b488e`, `0b54985`, `8827aa2` (Code Reviewer, 25 Nov 2026, 22:25-22:45 UTC)

Each change was attacked with targeted offline tests and read-only probes of the real process table, logs and
manifests. No job was stopped or restarted; nothing was trained.

**Held up.** E8: `trim_eval.trim_one` makes the same calls as `cmd_trim` (stage 1, `refine_cap` for non-phrases,
phrases whole); a rules-only, unversioned, mixed-version or model-changed report is refused; the driver step runs with
`$ROOTARG`, so smoke runs read their private sidecar. Queue parsing: a restarted queue reads only lines after its own
`queue start (pid N)` line; an empty `SOURCES` in the environment falls back to the default, as bash does; a
different list, a repeated name, no start line or an unreadable log block both profiles; a stopped queue's left-over
chunk is a process of its own and blocks its source's profiles. The real check at 22:30 UTC: commercial-safe is held
by islrtc, the recut job and the queue's four commercial-safe sources; research also by WSLP and OpenHands raw (PID
861, restarted 22:24). HF rows: one paced request per page (>= 2 s), a refused image refreshes its page at most once
and only when it is older than 2 min, and fetch failures are not counted as progress, so a refusing server ends in
exit 2 and the queue's 5 tries, not an endless loop of chunks.

**Fixed (tests: `backend/tests/isl/test_readiness_review3.py`, `test_hf_fetch_and_headers.py`,
`frontend/src/pages/LanguageEvaluation.test.tsx`):**

| Gap | Effect | Fix |
|---|---|---|
| option values before the source were read as the source | `ingest_isl_pose.py --budget-gb 26 openhands-raw` wrote source "26", `ingest_isl_video.py run --workers auto wslp` source "auto": no profile reads those, so the job blocked nothing (fail open; argparse accepts both orders) | every value-taking option of the four scripts is skipped, and only known names count (catalog / pose choices); anything else is unknown and blocks every profile. `isl500-links` counts as ISL500 (it writes `derived/isl500_links.jsonl`) |
| OpenHands raw counted as complete once its job was gone | the 22:06 crash (h5py "bad symbol table node signature" while listing NewzHook) would have let research start with ~8 % of NewzHook | complete only when the last line of `logs/ingest_pose_openhands_raw.log` is a normal end: `all members done`, the disk guard's `stopping:` (budget / min free) or `KeyboardInterrupt` (the documented `kill -INT`, 15.6 command 4). A traceback, a kill or a memory give-up keeps it incomplete ("did not end normally ...: restart it, it resumes; or --allow-partial") |
| `parse_v1_header` turned `TypeError` / `OverflowError` into `HeaderError` | a reader bug handing it something that is not bytes would have become one quiet failed row per chunk | only `struct.error` / `IndexError` from malformed bytes are converted; a non-bytes argument raises `TypeError` |
| E8 mismatch and the guard's warnings were not on the page | a `mismatch` / `missing` E8 showed only in the English subtitle; a partly ingested gated source (`data.readiness.warnings`) was in `report.md` only | the E8 card says "Synthetic trim check not shown (STATUS)" with the reason, or names the evaluated trimmer; a "Some data was used as it is" notice lists the warnings (English + Hindi) |

**Still open.** (1) (Closed 25 Nov 23:20 UTC, see the note in 15.8.) The trim sidecar records `trim_version`, not the learned model's hash, and `cli trim` re-trims
only on a version change, so a retrained `trim_boundary.pt` would leave a sidecar trimmed by two models under one
version, and E8 would check the newer one; record the model's SHA-256 per trim record and re-trim on change before
retraining that model. (2) HF rows: a page the datasets-server keeps refusing (8 tries, ~20 min) stops the run at that
item; every later chunk starts at the same item, so one persistently broken page (e.g. a 500 for one offset) would end
the source after the queue's 5 tries with the rest of its items not attempted (fail closed: readiness keeps research
blocked; re-run the source, or skip that page's items). Not changed now because `engine.py` is loaded by every new
queue chunk. (3) The disk guard's `stopping:` counts as a normal end for OpenHands raw whether the landmark budget or
`--min-free-gb` stopped it.

### 15.10 Pre-run fixes (25 Nov night): reality check (Reality Checker, 25 Nov 2026, 22:36-23:05 UTC, at `b3f8aac`)

(Asked for as "15.8"; 15.8 and 15.9 were already taken by the E8 fix and its review, so this is 15.10.) Read-only
re-check of `d5b488e` (E8), `0b54985` + `2f7d69e` (readiness), `8827aa2` + `0a5d016` (HF alphabets in full, OpenHands
headers) and `b3f8aac` (eval page), with live evidence. No job was stopped, restarted or signalled; nothing was trained;
the shared smoke work folders and stamps were not touched.

**Verdicts.** Commercial-safe full run: **NEEDS WORK only on data** - code ready, data guard exit 4 and blocked only by
what it reads; estimate **ready 26 Nov 05:30-09:00 UTC**. Research full run: **NEEDS WORK** - code ready, data guard
exit 4, and one new latent OpenHands bug (below) that a restart during the NewzHook listing would hit again; data
estimate **26 Nov 08:00-12:00 UTC** if the OpenHands job is not restarted before its listing ends; it starts after the
commercial-safe run anyway (one profile at a time).

**(1) OpenHands raw (PID 861, restarted 22:24:09 UTC after the 22:06 crash, points file moved aside; `--budget-gb 26
--min-free-gb 8`).**
* Progress (restore points written by the listing pass, `/proc/861/io`): inflated 42.35 GB at 22:36:38, 48.21 at
  22:40:22, 54.17 at 22:41:52, **74.17 GB of 159.48 GB at 22:47:22**, 85.87 GB at 22:50:30; compressed input 28.47 -> 32.96 of 61.24 GB. Rate
  **23.3-25.6 MB/s compressed** (60-66 MB/s inflated). The listing ends about **23:08-23:20 UTC**; the forward data pass
  over 159.48 GB at the MBM rate (56.6 MB/s inflated) takes about 45-75 min, so the job should end about **00:00-00:40
  UTC, 26 Nov** with all of NewzHook (estimate).
* RSS 0.59-0.75 GB (peak 0.80 GB) against the 1.5 GB guard; 13 threads; **0 deleted-but-open files** in any process on
  the machine; `du -xsB1 /` = 43.069 GB = `df` used (43.069 GB), so nothing is hidden. Free disk **30.74 GB**.
* Landmarks `openhands_raw` **10.99 GB** allocated (Sign Library 0.65, NIOS 2.24, ISH News 3.32, MBM 4.77); manifest
  63,745 rows, all ok. Density measured so far: 81 MB (MBM) to 88 MB (ISH News) of landmarks per inflated GB, so
  NewzHook adds about **12.9-14.0 GB -> 23.9-25.0 GB** in total: under the 26 GB budget (by 1-2 GB; if NewzHook is
  denser the budget stops it in its last few percent, which readiness accepts as a normal end). Free disk at the end
  of the job about **16-17 GB**, and about 15-16 GB after the rest of the queue (ISLRTC, WSLP, Mendeley, HF ~ +1.5 GB):
  above both the job's 8 GB floor and the queue's 10 GB.
* **Root cause of the 22:06 crash (found here, not fixed).** The 22:02 restart loaded 158 persisted restore points of
  NewzHook (up to 16.81 GB) and the h5py listing died 4 min later ("bad symbol table node signature"). The points are
  *not* corrupt: all 271 points of the renamed file `raw/openhands_raw/NewzHook.points.bad_2205` are byte-identical
  (same `out`, `inp`, `bits`, window) to the ones the current run re-created. The bug is in `SeekableInflate`'s metadata
  retention: `_step` calls `_scan(out[max(0, scan_pos - out):])`, which assumes the stream never moves ahead of
  `scan_pos`. With persisted points ahead of it, a forward seek (`_goto`: `p - out > every`) restores at a point far
  ahead, and those bytes are filed as metadata pages at the *old* `scan_pos`; later h5py reads of early metadata
  (`readinto` -> `_from_meta`) then get bytes from elsewhere. Reproduced offline (scratch probe, synthetic 12 MB
  member, 20 persisted points): 2 of 7 reads return wrong bytes; the same reads with the points dropped: 0 wrong. It
  only matters while h5py lists a member (the data pass reads with `read_range`, which bypasses the cache, which is
  why the ISH News / MBM restarts with points were byte-identical). Consequence: if PID 861 is restarted or respawned
  by its memory guard **before the NewzHook listing ends (~23:10-23:20)**, it loads the ~400 new points and will very
  likely crash the same way. Workaround: move `NewzHook.zip_NewzHook_newshook_dataset.h5.points` aside before such a
  restart (what was done at 22:24). Fix for the ingestion owner: in `_restore` (or `_step`), when the restored `out`
  is past `scan_pos`, drop `stage` / `_prev` and restart the scan at the next 4 KB page boundary at or after `out`
  (or do not scan at all after a forward jump); plus a test that seeks ahead through persisted points and re-reads
  early metadata. Readiness fails closed here (a traceback keeps research blocked, 15.9), so this cannot let a run
  start early; it can only cost a listing pass (~40 min).
* **Status (Data Engineer, 25 Nov 2026 23:00 UTC): fixed in code, tested offline; PID 861 not touched** (it keeps
  the old code in memory until its next start or memory respawn). Root cause confirmed with the scratch probe (2 of 7
  reads wrong before, 0 after). Fix in `SeekableInflate`: the scan stage now holds exactly the output bytes
  `[stage_at, out)`; every `_restore` (forward jump, backward seek, `ForwardPass` jump and error-handler re-positioning)
  re-aligns it (the partial page before the point comes from the point's 32 KB window), `_scan` drops the stage if a
  piece ever fails to continue it, and scanned pages are tracked as page intervals so each page is classified once,
  when first inflated, at its own offset (pages skipped by a jump are scanned if a later restore inflates them).
  Audit of the other paths that move the inflate position: all go through `_restore` (`_goto` forward / backward,
  `ForwardPass._restore_before` for its start, jumps and `redo` after a stream error); a failed restore changes no
  state; `ForwardPass` and its fallback read with `read_range` (no metadata cache); the forward pass's 16-32 MB window
  already re-based at every restore and now also restarts if a piece does not continue it. Also: the forward pass now
  frees `zero_pages` (up to 4 M ints) with the rest of the listing cache. Existing `*.index.json` listings are all
  from runs without loaded points (the log shows points loaded only at ISH News restarts, whose index existed, and at
  the 22:02 NewzHook run, which crashed before writing one), so none is affected. Tests:
  `backend/tests/isl/test_openhands_meta_cache.py` (31; synthetic zip + deflated HDF5, 260 chunks, metadata spread
  over the file): listing with 7 point subsets (none, all, first 10 %, first half, far ahead only, every 3rd, random)
  equals the listing without points, every cached page equals the bytes at its own offset, 400 random metadata reads
  identical with and without points, a 28-run fuzz over jump patterns (forward / backward, near / far, `_goto`,
  restores at random points, scanning switched off and on, native and zlib-fallback), listing with points then the
  forward pass on the same stream. Against the old class the listing tests fail with the production error ("bad symbol
  table node signature") and 16 of 24 fuzz runs fail. Throughput unchanged (first pass 58-59 MB/s, re-inflation
  118-126 vs 105-133 MB/s old, 96 MB synthetic, nice 10). Backend suite 534 passed (shared working tree). Once the job runs this code, a
  restart no longer needs the `.points` file moved aside.

**(2) Data guard on real data, 22:42 UTC** (`scripts/phase2_train.sh --profile X --check-only`):
* commercial-safe: **exit 4**. Reasons: islrtc job (6785), islrtc-recut (20244), the queue (17314) "still to process
  for commercial-safe: mendeley_static, isl_csltr, mendeley_agri, mendeley_emergency", manifests islrtc 8,607 / 13,665,
  mendeley_emergency 0 / 412 (3 to retry), mendeley_agri 2 / 78, isl_csltr 2 / 687, mendeley_static 8 / 13,800.
  OpenHands raw (861) and the WSLP chunk (30308) are listed as **not blocking**. The profile reads exactly include,
  islrtc, isl_csltr and the three Mendeley sets (`sources` in its `data_readiness.json`), so **it now waits only on
  what it reads**.
* research: **exit 4**. The same plus openhands-raw (861), the WSLP chunk (30308), the queue's research list (wslp,
  isl_fingerspelling, the four above, both HF sets, wslp_unlabelled, cislr, isign), manifests wslp 2,820 / 4,398,
  wslp_unlabelled 3 / 635, isl_fingerspelling 4 / 1,308, hf_hemg_alphabet 0 / **42,745**, hf_akrit_alphabet 0 /
  **10,752**, openhands_raw 63,745 / ? (budget-limited). `isl_csltr_words` is in the queue but in neither profile
  (one still frame per word, `profiles.py`), so it blocks nothing - correct. Cosmetic: the openhands_raw reason still
  says "complete once its ingestion job has ended"; since `2f7d69e` it means "ended normally" (the module docstring
  says so).

**(3) E8 on a scratch copy** (`artifacts/isl/phase2-smoke/research` + `isl_root_research` copied to the session
scratchpad, private logs; `SLR_ISL_WORKDIR` pointed there; the real smoke folder, stamps and
`artifacts/reports/trim_eval*.json` unchanged, mtimes checked):
* no `<work>/trim_eval.json` -> `trim_eval_check.status = missing`, no numbers, label "NOT SHOWN (MISSING)";
* the old 11:00 global `trim_eval.json` copied in -> `mismatch` ("records no trim_version ... learned=False");
* `cli trim-eval --learned-model auto --out <work>/trim_eval.json --n 60` -> exit 0 in **10 s**; report: `trim_version
  trim-2+trim-learned-1`, model `artifacts/isl/trim_boundary.pt` (SHA-256 `4c0efb82...`, 16:20:46), sidecar 6,521
  records all `trim-2+trim-learned-1`; synthetic n 60: within 300 ms **0.717**, median 0 ms, late 0.083, early 0.200,
  **learned_ends 3**; INCLUDE active kept 1.000 (n 60) -> E8 `ok`, numbers shown, label names the trimmer and time;
* the same with a scratch copy of the model -> `ok`; copy changed -> `mismatch` ("changed after trim-eval"); copy
  deleted -> `mismatch` ("no longer exists"); `--learned-model none` -> `mismatch` ("evaluated trim-2 but the run's
  trims are trim-2+trim-learned-1").
* `evaluate --all` on the scratch copy (the driver's smoke arguments): exit 0 in **296 s** (load 17-20). `report.json`
  E8: `trim_eval_check ok`, `trim_version trim-2+trim-learned-1`, run's trims {trim-2+trim-learned-1: 6,521}, model
  SHA-256 recorded, `learned: true`; `report.md` prints "trim-eval: **ok** - file ... (written 22:44:09Z), evaluated
  trim-2+trim-learned-1" and "synthetic end error: within 300 ms 0.717 ... (n 60, learned ends 3)"; the overview row
  names the trimmer. SMOKE and PARTIAL DATA labels kept. So E8 now reports the learned trimmer correctly.
* Side observation (not an E8 problem): with the same models, the re-run changed E3 top-1 from 0.082 to **0.209**
  (same n 890, 6 groups), E2 n 120 -> 154 and E4 n 84 -> 93, because `evaluate` reads the *current* manifests (ISLRTC
  grew from about 6,500 to 8,600 rows since 20:28, i.e. more dictionary references). A work folder's report is
  therefore only reproducible in the data state it was trained in: never re-run `evaluate` alone on a full-run work
  folder after more data has arrived (re-run the driver `--from trim` instead). The data guard keeps this true for the
  full runs, since nothing a profile reads may still be ingesting when it starts.
Note for reading the full run's E8: the synthetic set exercises the learned end on only 3 of 60 clips (it replaces
only `cap` ends), so the synthetic numbers mostly describe the rules; with the full run's default n they cover more
learned ends (default `--n 300`), but the learned model's own quality is in its training report (16:20), not in E8.

**(4) HF item lists.** `manifests/hf_akrit_alphabet_items.json` **10,752** items (train 9,139 + test 1,613, 33 classes,
300-798 per class), `hf_hemg_alphabet_items.json` **42,745** (35 classes, 1,200-1,447), ids unique, written 22:21:18-19;
`listing_status.json` "complete"; readiness expects exactly these. Neither source has started (no manifest); the queue
(17314) is still on WSLP (chunk 30308: 200 / 1,798 this run at 22:40, 6.65-6.99 s per item on 1 worker).

**(5) Checks.** Backend suite **499 passed** (64.6 s, `SLR_TORCH_THREADS=2`, nice 10, clean tree at `b3f8aac`).
Frontend: `npm run typecheck` pass, `npx vitest run` **81 passed / 1 skipped**, `npm run build` pass.

**Readiness ETAs (estimates from the rates measured 21:34-22:47).**
* islrtc: 7,597 (21:34) -> 8,122 (22:12) -> 8,607 (22:42): 15-16 / min, 5,058 left -> **04:00-04:30 UTC**; the recut
  job's final scan 30-60 min later.
* WSLP: 2,213 -> 2,547 -> 2,820: 9 / min, 1,578 left -> **about 01:40-02:00 UTC**.
* then the queue: ISL-Fingerspelling 1,304 videos (8.8 s each; 1 worker until islrtc ends, then 3) -> about
  04:30-05:00; Mendeley static 13,792 images, CSLTR 685, agriculture 76, emergency 412 (3 workers) -> **about
  05:30-07:30**: commercial-safe's last blocker. Then CSLTR words, the HF sets (53,497 images, 1.2-2.5 h, DATASETS 3.8)
  and WSLP unlabelled (632 videos) -> **about 08:00-11:30**: research's last queue blocker.

**For the main session (unchanged from 15.6 except where marked).**
1. Do not restart OpenHands raw before its NewzHook listing has ended (log line `NewzHook.zip:...: <N> datasets
   listed in <s> s`, about 23:10-23:20). If it stops before that, move
   `data/isl/raw/openhands_raw/NewzHook.zip_NewzHook_newshook_dataset.h5.points` aside, then start it with the same
   command from the repository root (`nohup nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py openhands-raw
   --workers 12 --budget-gb 26 --min-free-gb 8 >> data/isl/logs/ingest_pose_openhands_raw.log 2>&1 &`). After the
   listing (`<member>.index.json` written), a restart is safe.
   **Done (23:40 UTC, 15.11 "Live verification"):** the NewzHook listing ended 23:11:05 (30,604 datasets, index
   written), so a restart of PID 861 no longer needs the `.points` file moved aside; the code on disk (`3f2ad35`)
   is also verified against the real member's saved points.
2. Ready check: `scripts/phase2_train.sh --profile commercial-safe --check-only; echo $?` must print 0 (15.6 command 1).
3. `mv artifacts/isl/commercial-safe artifacts/isl/commercial-safe.phase1_1157` (still the phase-1 smoke folder of
   11:57: classifier, encoder, index, results.json, smoke_report.json).
4. `nohup scripts/phase2_train.sh --profile commercial-safe --threads 4 >> data/isl/logs/phase2_driver.out 2>&1 &`
   (13-24 h; now includes the `trim_eval` step before `evaluate`).
5. Research, after the commercial-safe run has finished: `scripts/phase2_train.sh --profile research --check-only`
   must print 0 (it does only after OpenHands raw ends *normally*), then
   `nohup scripts/phase2_train.sh --profile research --threads 4 >> data/isl/logs/phase2_driver.out 2>&1 &`.
   15.6 step 4's "stop OpenHands with SIGINT" is no longer needed (the job now ends tonight).
6. After each run: `artifacts/isl/<profile>/eval/report.md` must not say PARTIAL DATA or SMOKE, and its "trim-eval:" line
   must say the evaluated trimmer equals the run's (`trim_eval_check ok`).
Decided (25 Nov 22:54 UTC, `98b4dbd`): `mendeley_static` is ingested in full (102,470 photos, no per-folder cap;
DATASETS 3.9); the queue now needs about 7-10 h more for it, which moves the commercial-safe estimate above to
about 26 Nov 13:00-19:00 UTC.

### 15.11 Adversarial review of `3f2ad35` and `98b4dbd` (Code Reviewer, 25 Nov 2026, 23:05-23:35 UTC)

Every change was attacked offline (scratch copies, synthetic zips) and with read-only probes; no job was touched.

| Attack | Result |
|---|---|
| `SeekableInflate`: 200 extra fuzz runs (native and zlib fallback; point subsets; `every` 32-97 KB; input chunks 1-16 KB; `meta_cap` 3 pages to 256 MB; LRU of 1-256 pages; reads, `read_range`, restores at any point, scan off/on), then 20 h5py listings after a random pre-positioning | 0 wrong bytes, every cached / zero page at its own offset, listings identical |
| Fetch failures (15 %) inside `_restore` and `_step` during random reads; ForwardPass with injected stream errors (redo restores, jumps through points) on 60 random pair subsets / windows / steps | 0 wrong bytes or arrays; state consistent after every failure (2 tests added) |
| ForwardPass speed (101 MB synthetic member, 3 reps, old `6b54c28` vs new) | listing 1.13-1.38 s vs 1.21-1.44 s, forward pass 0.52-0.63 s vs 0.54-0.67 s, plain stream 0.31-0.33 s vs 0.34-0.39 s: parity |
| `mendeley_static` ids | the old 13,800-item list is a subset of the new 102,470 with identical gloss / meta; the 8 ingested rows are in it; readiness says 8 / 102,470 |
| `mendeley_static` memory under the queue's `--max-rss-gb 1.5` (scratch root, manifest filled to 101k rows) | peak 0.88 GB (1 worker), 0.60 GB (3 workers): fits |
| Trim hash re-trim loop | none: rules-only stores `null` and compares equal; every `cli trim` record path sets the hash |
| E8 false ok / mismatch (`check_report` with unrecorded, mixed, other, rules-only, learned report without hash, changed file) | all as intended |

**Open (not changed here).** (1) 3 byte-identical photo pairs in `mendeley_static` sit in two label folders
(digits 3/4 and 4/5, teenagers / adults full sleeves); the engine gives the copy a `duplicate_of` row that reuses the
original's landmarks under the other label, so 3 photos carry two labels. Drop both rows of such pairs in the letter
set builder, or have the engine fail a duplicate whose gloss differs (engine.py is loaded by every queue chunk).
(2) `cli trim --learned` (legacy stage-2, no run uses it) writes records without `trim_model_sha256`, so every
later `cli trim` re-trims those clips and E8 says `mismatch` (correct, but a loop if someone uses `--learned`).
`trim_bench run` writing the shared sidecar has the same effect once. (3) A sidecar record whose id later leaves
`_islrtc_records` (n_frames 0 on both the original and the recut row) is never re-trimmed, so its old version /
missing hash would keep E8 at `mismatch` (0 such records now; the version check had the same gap before).

**Live verification (Reality Checker, 25 Nov 2026, 23:15-23:45 UTC, at `9a27bf9`; read-only, no job touched).**
Default verdict NEEDS WORK; each line below is a measurement unless marked "estimate".

| Check | Evidence | Result |
|---|---|---|
| Synthetic repro (`probe_meta.py`, 12 MB member, 20 persisted points) | fixed code (working tree = `9a27bf9`, no diff in `pose_sources/`) vs `6b54c28` exported with `git archive` into scratch | **0 / 7** wrong reads fixed, **2 / 7** old: the probe still catches the bug, and the fix removes it |
| Real NewzHook member, byte level: the job's saved `.points` file **copied** to scratch (1,498 points, up to 159.39 of 159.48 GB); read byte 0, then jump to 150 / 95 / 40 / 120 GB, and after each jump do 40 random reads (512 B-4 KB) in the first 48 MB, compared with a fresh stream without points | same script, fixed vs old code | **fixed: 0 of 160 wrong** (71 restores, 1.09 GB fetched); **old: 158 of 160 wrong** (59,583 pages filed wrongly). This is the production bug, reproduced on the real member and gone with the fix |
| Real member, h5py: 30 `links.get_info` lookups (two highest addresses first, then the two lowest, then 26 at random), compared with the job's own `index.json` (listed without points) | fixed code | 30 / 30 addresses equal, 0 errors, 1.18 GB fetched. The old code also passes this test, so it only backs up the byte-level test and does not tell the versions apart |
| Real member, h5py `get_num_objs("keypoints")`: the full B-tree walk that crashed at 22:06, 4 min after start, with points loaded | fixed code | **no error over 14 min and 8 GB fetched**; stopped by the probe's own 8 GB download cap (it exited with 139 because the cap's exit was raised inside an h5py callback), so the walk was not finished. The job's listing ended without points anyway (below) |
| Live `.points` file untouched | SHA-256 `ad5fe2d5...` before and after, size 20,223,146, mtime 23:10:58 | unchanged |
| OpenHands raw PID 861 | log, `/proc/861`, manifest, `du`, `df` | NewzHook listing **ended 23:11:05** (30,604 datasets = 15,302 chunks, 2,806 s, 61.27 GB fetched; `index.json` written). Forward pass 31.49 GB inflated at 23:20:52 -> **94.24 of 159.48 GB at 23:39:19 = 56.7 MB/s inflated**, 5.1-5.3 chunks/s, 20.4-21.4 MB/s downloaded, no slowdown while the probes ran. Manifest 73,021 rows, **every row `ok`** (NewzHook 2,264 at 23:18, hands_frac median 0.91). RSS 1.21 GB (23:18), 1.33 GB (23:39), 1.26 GB (23:40), peak 1.39 GB, against the 1.5 GB guard: a guard respawn (`execv`) is possible near the end. It would load `3f2ad35` and resume from `headers.jsonl` with the index already there, so it is safe and costs only minutes. Landmarks `openhands_raw` 18.57 GB (NewzHook 7.58 GB so far = 80 MB per inflated GB). Free disk 22.47 GB; 0 deleted-but-open files |
| OpenHands ETA (estimate) | 65.2 GB left at 56.7 MB/s | forward pass ends **about 23:58-00:10 UTC, 26 Nov**; NewzHook is the last member, so the job then logs `all members done`. Landmarks about **23.7 GB** (under the 26 GB budget), free disk about 17 GB |
| `mendeley_static` item list | `mendeley_static_items.json` (22:54:18) | **102,470** items, all ids unique, 44 glosses; the 8 ingested rows are all in it. **Queue position:** `isl_queue.sh` (17314) is still on `wslp` (3,271 / 4,398 at 23:32), then `isl_fingerspelling` (4 / 1,308), then `mendeley_static` |
| Data guard (`scripts/phase2_train.sh --profile X --check-only`, 23:20 UTC) | outputs | **commercial-safe exit 4**, blocked only by what it reads: islrtc (6785), the queue (17314, "mendeley_static, isl_csltr, mendeley_agri, mendeley_emergency"), islrtc-recut (20244); islrtc 9,093 / 13,665, mendeley_emergency 0 / 412 (3 to retry), mendeley_agri 2 / 78, isl_csltr 2 / 687, mendeley_static 8 / **102,470**. PID 861 and the WSLP chunk 20872 are not listed. **research exit 4**: the above plus 861, 20872, wslp 3,163 / 4,398, wslp_unlabelled 3 / 635, isl_fingerspelling 4 / 1,308, hf_hemg 0 / 42,745, hf_akrit 0 / 10,752, openhands_raw 66,644 / ? |
| Backend suite | `cd backend && SLR_TORCH_THREADS=2 nice -n 10 ../.venv/bin/python -m pytest -q`, clean tree at `9a27bf9` | **539 passed** (71 s) |

**Rates and readiness ETAs (estimates from these samples).** ISLRTC 8,607 (22:42) -> 9,093 (23:20) -> 9,243 (23:32) =
12.3-12.8 / min (15-16 / min earlier, now under load 17-18). 4,422 left, so it ends **about 04:10-05:30 UTC**, faster once
OpenHands stops at about 00:00; recut's final scan follows 30-60 min later. WSLP 2,820 (22:40) -> 3,271 (23:32) = 8.6-8.9 / min,
so it ends **about 01:40**. ISL-Fingerspelling: 1,304 x 8.8 s on 1 worker, **about 04:50**. `mendeley_static`: 102,462
photos at 0.30-0.41 s (3 workers once islrtc ends; the single fetch thread limits it, DATASETS 3.9) take 8.5-11.7 h, so it
finishes **about 13:20-16:30**. CSLTR + agriculture + emergency (1,173 videos) take 0.8-2.1 h more, so
**commercial-safe data is ready about 14:00-19:00 UTC, 26 Nov**. Research also needs CSLTR words, the HF sets
(1.2-2.5 h) and WSLP unlabelled, so its data is ready **about 26 Nov 16:00 - 27 Nov 00:30 UTC**. Research starts
after the commercial-safe run (13-24 h) in any case, so in practice **27 Nov about 03:00-19:00 UTC**.

**Verdicts.**
* **Commercial-safe: NEEDS WORK.** The code is ready (the guard refuses only on data it reads, the suite passes, and
  the OpenHands fix is verified on the real member). The data is not complete: estimate 26 Nov 14:00-19:00 UTC.
  There is also **one owner decision before anything trained by this profile is shipped**: `mendeley_static`
  (102,470 photos; the **only** letter source of this profile, `LETTER_SOURCES` in `profiles.py`) has a licence field of CC BY 4.0 but a description
  that says "research purposes only ... any other use strictly prohibited", and it shows faces, including children's
  (DATASETS 3.9). The profile still lists it as commercial-safe. Recommendation: training can go ahead, but the
  commercial-safe letter model should not ship until this is settled. Two known model-quality gaps remain open: the letter temperature is fitted on a
  random 10 % of near-duplicate photos, and 3 photo pairs carry two labels (above).
* **Research: NEEDS WORK.** The code is ready. Its data waits on the OpenHands job (estimate: ends about 00:00-00:10,
  normal end) and on the whole queue (estimate: 26 Nov 16:00 - 27 Nov 00:30). It runs after the commercial-safe run.

**Commands for the main session** (from the repository root; none of these is needed before the ETAs above).
1. Leave PID 861 alone; check its end with `grep "all members done" data/isl/logs/ingest_pose_openhands_raw.log`.
   If it stops early for any reason, restart it with the same command (15.10 item 1). The `.points` file can now stay
   where it is.
2. Ready check, repeat until it prints 0: `scripts/phase2_train.sh --profile commercial-safe --check-only; echo $?`.
3. The owner decides on the `mendeley_static` licence (keep it in commercial-safe, or move it to research-only in
   `backend/app/isl/profiles.py` before the run, which also drops it from this profile's data guard).
4. `mv artifacts/isl/commercial-safe artifacts/isl/commercial-safe.phase1_1157`, then
   `nohup scripts/phase2_train.sh --profile commercial-safe --threads 4 >> data/isl/logs/phase2_driver.out 2>&1 &`.
5. After that run finishes: `scripts/phase2_train.sh --profile research --check-only; echo $?` must print 0, then
   `nohup scripts/phase2_train.sh --profile research --threads 4 >> data/isl/logs/phase2_driver.out 2>&1 &`.
6. After each run, `artifacts/isl/<profile>/eval/report.md` must not say PARTIAL DATA or SMOKE, and its trim-eval
   line must say `ok`.

### 15.12 Letter model: content duplicates, grouped hold-out, balanced sampling (AI Engineer, 25-26 Nov 2026, 23:40-00:15 UTC)

Fixes the two letter-model gaps left open in 15.11 ("Open" item 1 and the commercial-safe verdict) before the full
phase-2 run. Code: `backend/app/isl/training/train_fingerspell.py`, E5 runner / report lines in
`backend/app/isl/evaluation/runners.py`; tests `backend/tests/isl/test_fingerspell_groups.py` (11, synthetic
manifests in a temporary data root read through the real adapters) and `test_fingerspell_train.py` (updated). No
ingestion file was touched (`engine.py`, `adapters.py`, `pose_sources/` unchanged); nothing was trained except the
scratch smoke below.

**1. Content duplicates.** `letter_data` now also reads the `duplicate_of` rows (it sets `include_duplicates` on its
own adapter instances), and `dedupe_by_content` groups every item (images, letter clips, fingerspelling clips, across
all sources of the profile) by content: same landmark file + slice (a `duplicate_of` row reuses the original's file,
so a chain is one content) or same SHA-256 of the whole source file (copies extracted twice, or the same photo in two
sources; never used for sliced clips). A content filed under two or more labels is dropped **completely** (all its
items, so neither label is learnt from it); a content repeated under one label is kept once (the first in manifest
order, i.e. the original). Every drop is logged; counts and up to 20 examples (ids + labels) go into the
checkpoint's `data.dedup` (`cross_label_contents`, `cross_label_items`, `same_label_copies`, `per_source`). On the
full Mendeley item list (expected SHA-256) this drops the 3 cross-label pairs of 15.11 (6 items: digits 3/4 once,
4/5 twice) and 105 same-label copies of 93 photos (108 duplicate rows in DATASETS 3.9 = 3 cross-label + 105). Near-copies
that are not byte-identical (Mendeley bursts, Hemg augmentations) are not caught here; the grouped hold-out below is
what keeps them from inflating E5 and the temperature.

**2. Grouped hold-out for E5 and calibration (one logic for both).** Every example has a unit (`unit_of`):
ISL-Fingerspelling = signer (`isl_fingerspelling:signer_00`); Mendeley static = **age group**
(`mendeley_static:adults`; the manifest group is age x sleeves, but one person can be in both sleeve groups, so a
sleeve group is never held out alone); HF alphabets, OpenHands fingerspelling and anything without group ids = the
whole source, which **always stays on the training side** (akrit's own `test` rows are never a test or calibration
set). The held-out units are every unit of a source with at least 2 grouped units: commercial-safe = the 3 Mendeley
age groups; research = 3 ISL-Fingerspelling signers + 3 age groups (6). E5 is leave-one-unit-out over them (each fold
trains on everything else, including the whole-source sets) and reports letter top-1 pooled, per source
(`letter_top1_<source>`) and per group (`folds`, plus the unweighted `letter_top1_group_mean`), each with
`interval_fields` (95 % group bootstrap only with >= 5 groups, else the reason: commercial-safe "too few signer groups
(3)"; research pooled has 6 groups but from two sources, so since 15.13 no pooled interval either; its per-source rows have 3 each, so none), and word decoding for
held-out signers as before. The temperature is fitted on the pooled **out-of-fold logits** of those folds (letters
and noise), so it never sees a photo of a held-out person's group; `meta.calibration` records `status`, the held-out
units, n, NLL and ECE (15 bins) at T = 1 and at the fitted T. Status values: `grouped`; `not grouped` (no source with
2+ grouped units, or no fold with a trainable remainder: T stays 1.0 and the reason is printed; the old random 10 %
fallback is gone); `too few held-out examples` (< 200 held-out examples: T stays 1.0). The fold models are trained
on less data than the final model, so the fitted T is, if anything, slightly too soft (the safe side for a
"suggest" feature). There is still no early stopping or epoch selection (fixed epochs), so nothing else is chosen on
held-out data. With `evaluate=False` the temperature uses one held-out unit per grouped source (the smallest with
>= 200 letter examples). The report (`report.md`, E5 section) now prints one line per letter metric with its interval
cell or reason, the word line, and the calibration line; the E5 label says leave-one-group-out. Old checkpoints show
"calibration: not recorded".

**3. `max_per_class` decision: keep 4,000 per class per epoch, but sample, do not drop.** Each epoch takes up to
4,000 examples per class, split evenly over the class's subgroups (signers; Mendeley age x sleeves; whole sources;
a small subgroup gives all it has, the rest is shared by the others, and the remainder rotates every epoch), read
cyclically through a seeded permutation per subgroup. So over the epochs every example is used (`training.coverage`
in the checkpoint) and the data no longer depends on one random subset; deterministic for a seed. Before, each fit
kept one random 4,000 per class for all epochs. Measured on this machine (2 threads, nice 10, load 19-21 from the
ingestion jobs, 25 Nov 23:55 UTC; synthetic features of the real shapes, `_fit_mlp`, batch 128):

| letter set | examples | per epoch | s / epoch | coverage after 2 epochs |
|---|---|---|---|---|
| commercial-safe (Mendeley letters + digits, 6 subgroups) | 84,600 | 84,600 (cap binds only for E / 9) | 16.1 | 1.00 |
| commercial-safe, no cap | 84,600 | 84,600 | 14.4 | 1.00 |
| research (+ Hemg 42.7k, akrit 10.8k, ISL-FS ~77k frames, OpenHands 2.3k) | 217,944 | 144,000 | 24.7 | 0.92 (1.00 well before 40 epochs) |
| research, no cap | 217,944 | 217,944 | 40.7 | 1.00 |

For commercial-safe the cap changes nothing (only E and 9, ~4,300 photos each, are above it, and they are cycled).
For research it saves 40 % of the time per epoch while still using every example, and it stops the near-duplicate-
heavy sources (Hemg augmentations, ISL-FS consecutive frames) from outvoting the others inside a class. Estimated
letter-step time at 40 epochs, 2 threads under today's load (estimate): commercial-safe 3 age folds (training
remainders 41.7k-83.8k photos) + final fit ~ **30-35 min** (was ~20 min: calibration fit + final) plus feature
loading of 102k files; research 6 folds + final at ~144k per epoch ~ **1.9 h** (was ~1.4 h: 3 signer folds +
calibration fit + final). The full run uses
`--threads 4`, so likely less.

**4. Smoke, scratch copy (25 Nov 23:59 UTC).** `artifacts/isl/phase2-smoke/<profile>` and `isl_root_<profile>`
copied to the session scratchpad, `SLR_ISL_WORKDIR` pointed there, `cli --root <copy> train-fingerspell --smoke
--epochs 3 --threads 2`, nice 10; the real smoke folders were not written (SHA-256 of both `letters.pt` unchanged).
Data on disk at that time: `mendeley_static` 8 rows, `isl_fingerspelling` 4, `openhands_fs` 172, HF sets none.
* commercial-safe: 7 Mendeley items (1 Hindi vowel skipped) -> 6 examples in 2 age groups (adults 4, kids 2); dedup
  0; both folds skipped (remainder < 10), so **E5 not run** ("no held-out group had letter examples and a trainable
  remainder") and **calibration `not grouped`, T 1.0**; 1.3 s.
* research: 183 items -> 2,535 examples (36 letters, 20 noise): ISL-FS signer_00 238 (one signer: not a grouped
  source yet), OpenHands 2,291 (whole source, training side), Mendeley adults 4 / kids 2; dedup 0. E5 over the 2 age
  groups: n 6, letter top-1 0.167 (group mean 0.125), interval "none: too few signer groups (2)";
  **calibration `too few held-out examples`** (6 < 200), T 1.0; 0.13 s per epoch; 2.4 s.
These smoke numbers only show the plumbing; the data guard keeps the full runs from starting before the manifests
are complete.

**Checks.** Backend suite **550 passed** (53 s, `SLR_TORCH_THREADS=2`, nice 10). New tests: unit kinds; dedup through
the real Mendeley adapter (a `duplicate_of` copy under another label: both dropped; same-label copy: kept once; two
extractions with one SHA-256 under two labels: both dropped; slices of one file are distinct); commercial E5 +
calibration by age group (3 groups: no interval, reason printed; coverage 1.0 with a cap of 10); the hold-out without
E5; HF-only research = `not grouped`, T 1.0, akrit `test` rows never held out; whole sources in every fold's training
side; interval at 5 groups and none at 4; water-fill / balanced epochs (even split, rotation, determinism, full
coverage); too few held-out examples; E5 runner + report lines.

**Open.** (1) Near-duplicate clustering (landmark distance or perceptual hash) is still not done; with the age-group
hold-out it is not needed for E5 / calibration, but within-group bursts still weigh more than single photos.
(2) Commercial-safe E5 has 3 groups, so it never gets an interval; more letter signers (own recordings, 7.x) are the
only fix. (3) The `mendeley_static` licence question (15.11) is unchanged.

### 15.13 Adversarial review + reality check of `f11c9c3` (letter model) (Code Reviewer, 26 Nov 2026, 00:05-00:30 UTC)

Every claim of 15.12 was attacked with read-only checks on the real manifests / item lists, synthetic probes and a
scratch smoke run (copies of `artifacts/isl/phase2-smoke/<profile>` and `isl_root_<profile>` in the session
scratchpad; the real smoke `letters.pt` files kept their SHA-256, `cf7f09e0...` / `05871b72...`). No job was touched,
no ingestion file edited, nothing trained beyond the smoke.

| Attack | Evidence | Result |
|---|---|---|
| Cross-label byte copies (full Mendeley item list, 102,470 items, engine semantics: a `duplicate_of` row reuses the original's landmark path, `engine.write_duplicate`) | `dedupe_by_content` on Clips built from the list | 3 contents / 6 items dropped (3/4 once, 4/5 twice), 105 same-label copies dropped, 102,359 kept; 0 kept SHA-256 in two units, 0 kept SHA-256 twice; 0.78 s |
| Byte copies across age groups (would leak through the age-group hold-out) | the 96 SHA-256 groups with > 1 item | 0 cross age group, 0 cross sleeve group (93 same label + group, 3 cross-label in one group) |
| Group ids silently missing | real manifests + item lists | Mendeley: every item `static-<age>-<sleeves>` (6 groups, 9,070-33,417 items); ISL-FS: 3 signers (544 / 266 / 498 clips), 0 videos with two signers; OpenHands FS / HF: no ids -> whole source, training side |
| akrit `test` rows / HF sets in a hold-out | code path + test | whole source, every fold's training side; never held out or calibrated on |
| Cross-source copies (Hemg / akrit vs Mendeley) | catalog notes, item lists | not measurable yet (HF not ingested). Exact copies would be caught by the SHA-256 key and kept in their Mendeley group (Mendeley is read first); Hemg is 128x128 hand crops, Mendeley full photos, so byte copies are unlikely. Near-copies across sources: open |
| Labels across sources | glosses of all four letter sources | all map through `symbol_of` (a-z / A-Z / 0-9 / zero-nine); Mendeley `E` and `9` have 2 variants each (E1/E2, 9a/9b: two handshapes, one label, correct); 8 Hindi vowels skipped as before. Visually close pairs (V / 2, O / 0) are real ISL ambiguity, not label conflicts |
| Profile mixing in training | `LETTER_SOURCES`, `open_adapter(..., profile)` | commercial-safe reads only `mendeley_static`, licence checked per adapter: ok |
| **Profile mixing after training** | `build_pack`, `IslPack.recognizer` | **gap (fixed here):** `build-pack` copied any `letters.pt` and the app's ADR-ISL-6 check compared classifier / encoder / index / fusion profiles but **not the letter model**, so a research letter model in a commercial-safe work folder or pack loaded silently |
| **E5 pooled interval** | `evaluate_grouped` | **gap (fixed here):** research pooled letter top-1 got a "95 % CI (6 groups)" from a bootstrap over 3 ISL-FS signers + 3 Mendeley age groups. With ~90 % of the held-out examples from Mendeley, resamples mostly change the source mix, so the interval measured source difficulty, not person-to-person variation; the report header calls it a bootstrap over signer groups |
| **Sampler wrap** (synthetic research scale, 218k examples, 37 classes, 12 subgroups, cap 4,000) | `BalancedEpochs` | **gap (fixed here):** when a subgroup wrapped, the new permutation could hand out examples just taken: 12,073 of 148,000 draws (8 %) repeated inside one epoch. Speed: plan 0.07 s, 40 epochs of sampling 0.15 s (no blow-up) |
| E5 numbers | smoke checkpoints, report lines | pooled / per-source / group mean recomputed from `folds`: consistent; interval cells print a reason under 5 groups |
| Smoke: `train-fingerspell --smoke --epochs 3`, then `evaluate --all` (driver smoke arguments), then `build-pack --smoke` | scratch copy, 2 threads, nice 10, load 11-15 | all exit 0 for both profiles: letter step 1-5 s, evaluate 88 s (commercial-safe) / 90 s (research); E5 + calibration lines as in 15.12 item 4 |

**Fixes** (small, offline-tested; `test_letter_review_guards.py`, 2 tests, plus one in `test_pipeline_and_api.py`,
each failing on `f11c9c3` and passing now):
1. `pipeline.check_letters_profile`: `build-pack` refuses a `letters.pt` whose `meta.profile` is another profile, or
   whose recorded letter sources (`meta.source`, `data.per_source`) include one the profile may not read (legacy
   `train-letters` checkpoints without a profile, made from own recordings, still pass). Checked live: the scratch
   research smoke model copied into the commercial-safe work folder -> `build-pack` exit 1, nothing written.
2. `app/packs/isl.py`: the ADR-ISL-6 check now includes the trained letter model's profile (`ProfileMismatch` names
   `letters`); the user's own alphabet calibration is not a trained component and is unaffected.
3. E5: the pooled `letter_top1` interval is only computed when the held-out letters come from one source; otherwise
   the reason is "pooled over N sources (...): intervals only per source" (a "too few signer groups" reason, when it
   applies, wins). Per-source rows keep their own intervals (>= 5 groups of that source).
4. `BalancedEpochs`: on a wrap, the examples just taken go to the end of the new pass, so no example is drawn twice
   in one epoch (coverage and determinism unchanged).
5. Log wording: "letter temperature kept at 1.0 (<status>)" (it said "not grouped" also for "too few held-out examples").

**Checks.** Backend suite **553 passed** (63 s, `SLR_TORCH_THREADS=2`, nice 10). One earlier full run had 1 failure,
`test_hf_fetch_and_headers.py::test_memory_guard_drops_the_page_cache` (19 / 19 when that file runs alone, 553 / 553 on
the re-run): a flaky ingestion test (engine memory guard vs. its fetch thread), unrelated to these changes; the
engine is loaded by the running queue, so it is left to the ingestion owner.

**Open (not changed here).** (1) Near-copies across sources and inside Mendeley bursts are not clustered (15.12 open
1); the hold-out units keep them out of E5 / calibration except across sources, which cannot be measured before
the HF sets are ingested. (2) The fingerspell step records the letter sources' `clips` (including the dropped
duplicate rows) as their "used" count in `sources.json`; harmless for the licence list today (every source with
clips also contributes examples), but a source whose items were all skipped would still be listed. (3) The step has
no resume: the research letter step (~1.9 h estimate, 15.12) restarts from scratch if interrupted. (4) The
`mendeley_static` licence question (15.11) is unchanged; it is the commercial-safe letter model's only source.

**Verdicts (letter model).**
* **Commercial-safe: READY to train** once the data guard passes (code and smoke ok; dedup verified on the full item
  list; age-group hold-out has no byte-copy leak; E5 will honestly show no interval with 3 age groups). **Not ready
  to ship** until the owner settles the `mendeley_static` licence (15.11).
* **Research: READY to train** after the commercial-safe run, once its data guard passes (code and smoke ok; E5
  intervals per source only; HF sets whole on the training side). Its E5 numbers for Mendeley age groups assume the
  HF sets hold no near-copies of Mendeley photos (not measurable until they are ingested; a check worth doing on the
  full run's data before quoting the research E5 number).

### 15.14 Hand-size floor (AI Engineer, 26 Nov 2026)

**Bug.** The hand-shape block divides each hand's wrist-relative landmarks by the 2D wrist-to-middle-MCP length
(`_hand_shape` in `backend/app/features/build.py`). When the hand points at the camera (foreshortened), that length
goes to about 0 and the hand-shape features, and the velocity features computed from them, blow up. Over the
commercial-safe pre-training clips the largest feature value was about **2027**, and about **2 %** of clips had a value
above 100. The pre-training loss spikes came from this: in epoch 13 (train loss 42.95, against 2-6 for most epochs) two
steps made up about **95 %** of the epoch's loss. The diagnosis and checks are scratch-only (not in the repo).

**Fix** (commit `697120d`). New `FeatureConfig.hand_size_floor`: the length is bounded from below by the floor (in
shoulder widths, the body-frame unit). `ISL_FEATURES.hand_size_floor = 0.1`. Default 0 = old behaviour, and 0 is left
out of `to_dict()`, so ASL configs and older ISL packs are byte-identical. The value is written into the encoder,
classifier and index metas and into `pack.json`, and every training, evaluation and live path reads it from there,
so a pack is trained and served with the same features. On a 1,500-clip sample with the floor, the median absolute
hand-frame feature value is 0.17, p99 2.3, p99.99 10, max about 120 (rare landmark misdetections).

**The commercial-safe run.** Its pre-training process started (10:39 UTC) before the commit and kept the code it had
loaded, so `artifacts/isl/commercial-safe/pretrained_encoder.pt` was trained on **unfloored** features (its meta
`feature_config` has no `hand_size_floor`; best epoch 18 of 20). It was kept: on the same validation batches, unfloored
vs floored input gives NT-Xent 0.1361 vs 0.1350, view-matching accuracy 0.9911 vs 0.9914, and mean embedding cosine
0.997 between the two. It is only a warm start; the encoder and later steps (started 12:42 UTC) train and evaluate on
floored features, and their metas say so. The fine-tuned encoder's meta does not yet record that its warm start was
unfloored (follow-up 3). The research run pre-trains with the floor.

**Robust MFM loss (pre-training version `isl-enc-pre-2`).** Even with the floor, masked-frame modelling reconstructs
raw, unnormalised features, so one misdetected frame (value ~100) still dominated a plain MSE step. `pretrain.py` now
uses smooth L1 (beta 1) against targets clamped to +-10 (`MFM_CLIP`, about the p99.99 above): quadratic for normal
values, linear above 1, so the gradient per element is bounded. In the quadratic region smooth L1 is half the
squared error, so at the same `mfm_weight` (1) normal frames weigh about half as much against NT-Xent as in
version 1; the research run is the first full run with this balance. The best checkpoint is chosen on validation NT-Xent +
`mfm_weight` x robust MFM (`val`); `val_contrastive`, `val_mfm` (raw MSE) and `val_view_acc` are still reported, and
`val_mfm_robust` is new. `mfm_loss="mse"` keeps the version-1 training loss for comparisons (still stamped `isl-enc-pre-2` and selected on
the robust term). The pre-training resume
signature now includes the pre-training version and the feature config, so a state from the old loss or from
other features is refused (`StateMismatch`; re-run with `--force pretrain`), never continued. Note: with the new code a
commercial-safe `pretrain` re-run without `--force` is refused rather than returning "finished".

**Follow-ups.**
* **C (report only):** `backend/app/isl/fingerspell/features.py` (`_shape`) and `backend/app/isl/data/dedup.py` use
  the same 2D denominator. Letter features have no config, so changing them would silently change the input of the
  shipped `letters.pt`; needs a versioned letter-feature change and a retrain.
* **D:** the encoder and classifier resume signatures (`train_encoder.py`, `train_classifier.py`, via `loop.py`'s
  `extra`) still ignore `feature_config`, so a resume across a feature change would mix features. To be added after the
  run (adding it while a state exists would refuse that state).
* 3: `train_encoder.py` should record the warm start's `feature_config` in `meta.pretraining` and warn when it differs
  in more than `hand_size_floor`.
* My signs: the embedding cache was keyed by the encoder version only (`isl-enc-1` for every ISL encoder), so a new
  pack would have reused the old pack's vectors. Fixed: the key now includes a hash of the encoder weights and the
  feature config (`mysigns/store.py`, `embedder_key`). The per-sign threshold `theta`, computed at enrolment with the
  old encoder, is not recomputed yet.

### 15.15 Feature config in the resume signatures; warm-start feature config in meta (AI Engineer, 26 Nov 2026)

Follow-ups D and 3 of 15.14, added while the research run was still in dedup / pretrain (no
`artifacts/isl/research/**/*.state` existed, so no research state is invalidated).

* **Resume signature.** `train_encoder.py` and `train_classifier.py` now put `"features": fcfg.to_dict()` into the
  `LoopConfig.extra` of the resumable loop. `loop.py` hashes the whole config (minus `save_every_s`, `max_minutes`,
  `patience`), so a `.state` saved under another feature config (e.g. before `hand_size_floor=0.1`) raises
  `StateMismatch` instead of being continued on different inputs. A state written by the older code has no
  `features` key, so its hash differs too: it is refused, never silently resumed. Consequence: the existing
  commercial-safe `encoder` / `oneshot` / `classifier` / `e1a` / `cv` states are refused on a re-run without
  `--force <step>` (steps with a `.done` stamp are skipped anyway).
* **Warm start.** `meta.pretraining` of a fine-tuned encoder now records `init_feature_config` (the init
  checkpoint's `feature_config`, `None` if it has none) and `init_feature_config_diff` (`{field: [init, current]}`,
  defaults filled in, so an unfloored config counts as `hand_size_floor: 0.0`). When the diff is non-empty,
  `train_encoder` logs a warning (logger `app.isl.training.train_encoder`, reaches the driver log via stderr). Every
  difference warns, the hand-size floor included: the commercial-safe warm start from the unfloored pretrain is
  accepted, but it is now visible in the log and the meta. The research run pre-trains with the floor (15.14),
  so its encoder diff is expected to be `{}`.
* Tests: `backend/tests/isl/test_resume_feature_config.py` (resume accepted on the same features and refused on
  others, for encoder and classifier; an old-code state is refused; meta records; warning only on a mismatch).

### 15.16 E1 "not sure" and sliding mode used the fused threshold on classifier probabilities (Model QA, 26 Nov 2026)

**What the commercial-safe report's E1 "not sure" column meant.** `artifacts/isl/commercial-safe/eval/report.md`
(commit `a272419`) shows not sure 0.999 (E1b), 0.952 (E1a), 1.000 (E1c) next to top-1 0.723 / 0.955 / 0.915.
That column was **not** an abstain rate. It was the share of test clips whose raw classifier probability (crop
mean, temperature 1) was below **0.96**, the pack's *fused* balanced accept threshold (`fusion.json`; `e1_closed_set`
at `a272419` `runners.py:151-163`). 0.96 is a value on another scale: the softmax of 0.5 x classifier log-probability
(t_c 0.72) + retrieval (tau 0.1) + 1.0 x prior. The commercial-safe classifier trains with label smoothing, and its raw
top-1 probability never gets that high on E1b (p90 0.881, p99 0.943, reproduced exactly: 1 of 792 clips reaches it).
So the column only said "the classifier probability never reaches a threshold made for another score", and "not
accepted" also counted clips the app would show as "did you mean" suggestions. E3's not sure (0.105) is the
abstain rate only (`decision_block`, `runners.py:66`), so the two numbers in one column measured different things.
The E1 top-1 / top-5 / macro-F1 numbers, and everything in E2-E8, are not affected.

**The same mismatch in the live app.** In sliding mode (`Sliding window` on the Live and Analyze pages) an ISL pack
decides on `IslRecognizer.class_probs` (raw classifier probabilities, `recognizer.py:82` at `a272419`) against
`RecognitionEngine.threshold`, which returned the fused `accept[policy]` (`engine.py:152-160`, used by
`_decision` at `engine.py:241` and `_push_sliding` at `engine.py:488`). With the commercial-safe pack that is 0.96 for
every policy, so sliding mode would almost never commit a word. Segment mode, the default, uses the fused decision
correctly: its numbers are E3.

**Fix.**
* `app/isl/recognizer.py`: `classifier_calibration(predictor)` reads the classifier checkpoint's own `temperature`
  and `recommended_threshold`. `train_classifier` fits both on the classifier's validation part, and the threshold is
  on temperature-scaled probabilities. If a checkpoint has no threshold, the Predictor default 0.6 is used and
  flagged. `class_probs` now returns temperature-scaled probabilities, and `IslRecognizer.classifier_threshold`
  exposes the threshold.
* `app/inference/engine.py`: the threshold for sliding mode with a recogniser is `classifier_threshold`. A threshold
  the user sets still wins. Segment mode keeps the fused thresholds.
* `app/isl/evaluation/runners.py`: E1 (`e1_closed_set` and the new `classifier_alone_decisions`) applies the engine's
  classifier-alone rule: crop-mean temperature-scaled p1 >= the classifier's own threshold, top-1 minus top-2 >=
  `EngineConfig.min_margin` (0.10), and at least 2 crops agree. It reports `coverage`, `accepted_precision`,
  `wrong_accept_rate`, and `not_sure_rate` = 1 - coverage (there is no suggestion step). A new `decision` block
  records the temperature, threshold and margin. Top-1 / top-5 are now computed on the temperature-scaled crop mean,
  the probabilities sliding mode shows (E1b commercial-safe: 0.722 instead of 0.723).
* `markdown_report` has an **accepted** column, and its legend says what not sure means for each protocol: E1 =
  1 - accepted, E2 / E3 = abstain only.
* Tests: `backend/tests/isl/test_e1_classifier_decision.py`.

**Corrected E1, commercial-safe checkpoints, read-only scratch re-run** (not written into the commercial-safe folder,
whose report stays as it is):

| | threshold (T) | accepted | not sure | wrong accept | accepted precision |
|---|---|---|---|---|---|
| E1b (792) | 0.60 (0.72) | 0.710 | 0.290 | 0.104 | 0.854 |
| E1a (816) | 0.35 (0.72) | 0.975 | 0.025 | 0.036 | 0.964 |

For comparison, the classifier alone *through the fused rule* (a pack without an index) on E1b gives accept 0.000 and
suggest 1.000. The fused thresholds are only meaningful together with retrieval.

**Open (not changed here).** (1) The commercial-safe `fusion.json` has accept 0.96 for strict, balanced *and*
lenient: no policy reached its Wilson-lower-bound target on validation, so `fit_thresholds`
(`calibration.py:137-138`) picked the same best-precision threshold for all three, and the app's policy switch has no
effect for that pack. (2) A classifier-alone accept rule reaches only 85 % precision on E1b (10 % silent errors).
Sliding mode uses the classifier's validation threshold, which targets 85 % accuracy (`pick_threshold`), not the
95 % of the fused balanced policy. (3) A pack without an index would apply fused thresholds fitted with retrieval to
classifier-only fused probabilities. Such a pack never accepts in segment mode; it shows suggestions.

### 15.17 Distinct per-policy thresholds: fused, classifier alone (sliding mode) and "did you mean" (AI Engineer, 26 Nov 2026)

Fixes open items (1) and (2) of 15.16. Packs fitted before this change still load and behave exactly as before; the
new behaviour starts with the next `cli fit-fusion` (in the running research run, its `fusion` step).

**Diagnosis (commercial-safe, read-only scratch re-run of fit-fusion's validation evidence, 895 of 902 val clips,
train-only index rebuilt in scratch because `work/index` is now the index_final build).** No policy can reach its
target at this accuracy level (signer-independent top-1 0.72-0.75). This is a precision ceiling, not a Wilson-bound
effect: fused accepted precision stays at 0.89-0.91 for every threshold from 0.85 to 0.99, and the best 95 % lower
bound (LB) is 0.867 (t 0.96, 270 accepts, precision 0.907). The margin, crop-agreement and `s_min` guards change
nothing above t 0.60. The fallback in `fit_thresholds` then gave every policy the same best-LB threshold (0.96 / 0.96
/ 0.96), so the policy switch did nothing and every policy was the most conservative one. The 25 wrong val accepts
at 0.96 are systematic confusions (clothing->dress 5, female->"woman / female" 4, office->blue 3, race->alright 3;
all INCLUDE clips, 13 from signer group ps00). The classifier alone has the same ceiling (best LB 0.865 when every val
clip counts). `suggest_min_top5_mass` 0.0 is what the rule returns (non-accepted val clips already have top-5 hits
0.895), but it was fitted on the balanced pool only. With a lower lenient threshold, lenient's pool keeps only the
hard cases, and its top-5 hit rate drops to 0.750 (val) and 0.743 (test), below the 80 % promise.

**Fix.**
* `calibration.choose_policy_thresholds` (used by `fit_thresholds` and the new `fit_classifier_thresholds`).
  * A policy whose target is reachable keeps the old rule: the smallest threshold with LB >= target.
  * Otherwise the targets keep their spacing below the best achievable bound LB* (ties go to the larger threshold).
    Effective target = LB* - (0.98 - target), so 0 / 0.03 / 0.08 below LB*. When a more lenient target *is*
    reachable, the unreachable ones are spread proportionally between the strictest reachable target and LB*.
  * Each policy takes the smallest threshold whose LB reaches its effective target. The strictest unreachable policy
    takes the LB* threshold, as before.
  * The thresholds are ordered by construction. They differ whenever the data have distinct thresholds at those
    bounds. When precision does not rise with confidence, they stay equal, and the code does not invent a difference.
  * The report (`accept_fit`) and `fusion.json` `achieved[policy]` hold `rule` (target / fallback / default),
    `target`, `target_reached`, `effective_target`, the `precision_lower_bound` actually achieved, precision, accepts
    and coverage, plus a `note` naming the policies whose target was not reachable. The `<policy>_target_reached`
    keys are kept.
* "Did you mean": `suggest_min_top5_mass_by_policy` holds a mass fitted for each policy on its own suggestion pool,
  with the same rule (smallest mass with >= 80 % top-5 hits). `decide` uses `FusionParams.suggest_mass(policy)`.
  `suggest_min_top5_mass` stays as the balanced value, for older readers and for packs without the new field.
  Anyone overriding the single value by hand (as the API test fixture does) must also clear the per-policy field.
* Classifier alone (sliding mode). `fit_fusion.fit_classifier_alone` fits per-policy thresholds on the validation
  evidence fit-fusion already collects. It uses the engine's rule: crop-mean probability at the checkpoint's
  temperature, margin `EngineConfig.min_margin`, >= 2 crops agree. Every val clip counts, and one whose word is not a
  classifier class can only be a wrong accept. The thresholds are stored in `fusion.json` as `classifier_accept`,
  together with `classifier_fit`: the classifier's fingerprint (`recognizer.classifier_fingerprint`, a hash of meta
  fields that change with every training run), the temperature, margin and per-policy report.
  * `recognizer.classifier_policy_thresholds` uses them only when the fingerprint and temperature match the loaded
    classifier. E1a and the CV folds swap other classifiers in next to the main fusion.json.
  * Otherwise, and for every older pack, all policies use the checkpoint's `recommended_threshold`, as before.
  * `IslRecognizer.classifier_threshold_for(policy)` feeds `RecognitionEngine.threshold` in sliding mode. A
    user-set threshold still wins, and segment mode is unchanged.
* E1 (`e1_closed_set`): the summary columns keep their 15.16 meaning (classifier-alone rule, not sure = 1 -
  accepted), now at the policy the app uses by default (`decision.policy` = balanced). `decision` records the
  per-policy thresholds, their source (fusion.json or checkpoint), and why it fell back. `policies` has every
  policy's accepted / not sure / wrong accept / precision.
  * E3 records each policy's threshold, suggest mass and validation target status, and says its summary row is
    balanced (`metrics_policy`). The phase-1 E3 records the thresholds too.
  * `report.md` gets a "Decision policies" table (E1b, E1c, E3 x strict / balanced / lenient).

**What the new fit gives on the commercial-safe evidence** (scratch only; `artifacts/isl/commercial-safe` is
unchanged and keeps 0.96 / 0.96 / 0.96 until it is re-fitted. A re-fit must use the train-only index, see above).

| | strict | balanced | lenient |
|---|---|---|---|
| fused accept threshold (old: 0.96 for all) | 0.96 | 0.73 | 0.46 |
| val LB achieved / effective target (targets 0.98 / 0.95 / 0.90 not reachable) | 0.867 / 0.867 | 0.838 / 0.837 | 0.790 / 0.787 |
| suggest mass (old: 0.0 for all) | 0.0 | 0.0 | 0.65 |
| E3 test (915): accepted / precision / wrong accept | 0.274 / 0.932 / 0.019 | 0.601 / 0.875 / 0.075 | 0.742 / 0.823 / 0.131 |
| E3 test: suggest (top-5 hit) / not sure | 0.621 (0.882) / 0.105 | 0.294 (0.803) / 0.105 | 0.103 (0.851) / 0.155 |
| classifier-alone threshold (old: checkpoint 0.60 for all) | 0.96 | 0.74 | 0.46 |
| E1b (792): accepted / precision / wrong accept (old 0.60: 0.710 / 0.854 / 0.104) | 0.326 / 0.938 / 0.020 | 0.619 / 0.869 / 0.081 | 0.785 / 0.826 / 0.136 |

Fused strict is what every policy was before. In sliding mode, every policy used 0.60 before; balanced is now slightly stricter (0.74). **Note the default changes**: for a re-fitted pack, balanced (the app
default) accepts about 60 % of test queries at 87.5 % precision (7.5 % wrong accepts), instead of 27 % at 93 %.
Users who want the old, conservative behaviour choose strict. The UI must show the achieved numbers, not the targets
(none is reached). Raising precision itself needs better models or label fixes (for example `isl.female`, which is
not a classifier class but is confused with "woman / female"), not threshold tuning.

Tests: `backend/tests/isl/test_policy_thresholds.py`, with:
* distinct ordered thresholds when the targets are unreachable;
* no invented split when precision is flat;
* the reachable / partly reachable cases;
* per-policy suggest masses;
* old-pack loading and behaviour;
* sliding mode per policy, plus its fallbacks (old pack, swapped classifier, other temperature);
* E1 / E3 / report.

`test_pipeline_and_api.py` checks the real fit-fusion output.

### 15.18 Sliding-mode thresholds fitted on the real sliding mode, not the segment proxy (Model QA review of 15.17, 26 Nov 2026)

**Finding (High for the guarantee, Medium for users).** 15.17 fitted the per-policy classifier-alone thresholds
(`classifier_accept`) on a segment-level proxy: the crop-mean probability of three crops of the *whole* clip, with
>= 2 crops agreeing. The app's sliding mode does something else. It classifies one crop per 2 s window every 200 ms,
smooths the windows (EMA 0.5), commits after the same top word for 3 windows in a row, and can commit several times
during one sign (transitions, partial signs). Replaying fit-fusion's own commercial-safe validation clips (895) and
the E1b test clips (792) through the real `RecognitionEngine` in sliding mode (read-only scratch,
`scratchpad/thresholds/review/`):

| | strict | balanced | lenient |
|---|---|---|---|
| 15.17 threshold (proxy fit) | 0.96 | 0.74 | 0.46 |
| validation lower bound stored in fusion.json (proxy) | 0.865 | 0.835 | 0.785 |
| validation lower bound of sliding mode at that threshold | 0.819 | 0.763 | 0.658 |
| E1b test, sliding mode: clips with a commit / precision / clips with a wrong commit | 0.162 / 0.922 / 0.013 | 0.628 / 0.803 / 0.124 | 0.806 / 0.708 / 0.235 |
| 15.17's E1b proxy numbers for the same thresholds | 0.326 / 0.938 / 0.020 | 0.619 / 0.869 / 0.081 | 0.785 / 0.826 / 0.136 |

So the stored bounds were not bounds on sliding mode, and the proxy understated wrong commits by 4-10 points.
Lenient at 0.46 put a wrong word on 23.5 % of test clips, more than the old single 0.60 (17.6 %).

**Fix.**
* `fit_fusion.sliding_windows` replays a recorded clip through `RecognitionEngine` (sliding mode, default
  `EngineConfig`) with a threshold that nothing reaches. It records every classified window: time, EMA probabilities,
  streak, sign boundary (hands gone), and the engine's class key. EMA, streak and boundaries do not depend on
  commits. `calibration.sliding_commits` then gives the commits for any threshold, using the engine's rule: threshold,
  margin, streak >= `agree_n`, no repeat until a boundary, `cooldown_ms`. A test checks that it equals the live
  engine's commits for several thresholds, including a sign boundary.
* `fit_classifier_alone` (called by fit-fusion with its validation clips) fits `classifier_accept` with
  `calibration.fit_sliding_thresholds`. The unit is the **clip**: accepted means sliding mode committed during it,
  and correct means every commit was the clip's word. A word that is not a classifier class can only be wrong. The
  fit uses the same targets and graded fallback as 15.17 (`choose_policy_thresholds`).
  * `classifier_fit` records `rule` ("sliding replay"), the engine settings, the unit and `replay_failed`.
  * If no clip can be replayed, the 15.17 proxy is used and labelled (`rule` "segment proxy", `proxy_note`).
  * Packs fitted with 15.17 keep working; only the label is missing.
* E1b / E1c (main classifier only, not the CV folds or E1a) add `policies.<policy>.sliding_replay`: the same test
  clips through the real sliding mode at that policy's threshold. report.md's "Decision policies" table gets
  "sliding replay" rows. The 15.16 / 15.17 E1 columns keep their meaning, which is the three-crop proxy.

**On commercial-safe (scratch, the new fit on the same validation clips; about 61 s single-threaded):**

| | strict | balanced | lenient |
|---|---|---|---|
| sliding-mode threshold | 0.92 | 0.81 | 0.68 |
| validation lower bound (sliding mode, per clip) / precision | 0.819 / 0.861 | 0.791 / 0.827 | 0.741 / 0.776 |
| E1b test, sliding mode: clips with a commit / precision / clips with a wrong commit | 0.330 / 0.908 / 0.030 | 0.564 / 0.841 / 0.090 | 0.670 / 0.785 / 0.144 |

Every test precision is above its validation bound. All three thresholds are still fallbacks, because sliding mode
cannot reach 0.90 at this accuracy. The research run's `fusion` step uses this code (it starts a fresh process).

**Not changed, for the record (lens: are the reported numbers what their labels say?).**
* The "95 % lower bound" is computed on the same validation data the threshold was chosen from: strict takes the
  maximum over about 80 thresholds. It is optimistic after this selection, not a simultaneous 95 % bound. On
  commercial-safe, test still met every bound.
* The 15.17 fallback has no absolute floor. A weak pack's lenient policy can end up far below 0.90, and the report
  says so (`precision_lower_bound`, `target_reached` false). The "did you mean" 80 % top-5 rule uses the point
  estimate, not a lower bound. Lenient's pool on commercial-safe validation: 64 of 77 = 0.831, lower bound about 0.73.
* The commercial-safe numbers in 15.17 / 15.18 come from `include-session-cv5-fold0`. Validation and test share all
  four pseudo-signer groups (ps00-ps03), so they are **session**-independent, not signer-independent as 15.17's
  diagnosis says.
* Engine quirk (pre-existing): two classifier classes of commercial-safe are not in the pack vocabulary. Their
  `class_index` is -1, so sliding mode's streak and repeat suppression treat them as one word. The replay keeps the
  engine's key, so the fit matches what the engine does.

### 15.19 Driver: a retrain really retrains; scratch runs keep their own stamps (Model QA fix 0.1, 27 Nov 2026)

**The bug.** `--from STEP` removed only the `.done` stamps. `train()` always passes `--resume`, and `run_loop`
returns at once when the saved state says "finished". So a retrain after a code change gave back the OLD weights
when the config signature was unchanged, and failed with `StateMismatch` when it had changed. `--force` took only
one step. And the stamps and log were keyed only by profile under `data/isl/logs/`, while `PHASE2_WORK_BASE` moved
only the work folder: a scratch run with `--profile research` would skip steps on the real run's stamps, or
overwrite or delete them. Its `pack` and `evaluate` steps also wrote into the shipped `artifacts/packs/<id>/`.

**What changed in `scripts/phase2_train.sh`.**
* `--from STEP` now forgets, for STEP and every later step, the stamp AND the training state files:
  `pretrained_encoder.state.pt`, `encoder.pt.state`, `oneshot/fold*/encoder.pt.state`, `classifier.pt.state`,
  `cv/fold*/classifier.pt.state`, `e1a/classifier.pt.state`, plus any half-written `<state>.tmp<pid>`.
  Earlier steps keep theirs. Each removed file is written to the log.
* `--force` takes a comma-separated list (`--force encoder,cv`). Unknown step names in `--from`, `--only` or
  `--force` are refused (exit 2). Before, a typo in `--from` silently ran every unstamped step.
* `--only STEP` is unchanged: it ignores the stamp but resumes from the state. To retrain one step, use
  `--force STEP`.
* `--dry-run` prints the work folder, the stamp folder, the log, the pack folder, every file that would be removed
  and every step that would run or be skipped. It writes nothing and runs nothing (no data guard, no cli call).
* Run tag: `--tag NAME` or env `PHASE2_TAG` (letters, digits, `.`, `_`, `-`).
  * The real run (no tag, default work folder `artifacts/isl`) keeps the old places:
    `data/isl/logs/phase2_<profile>.stamps/` and `.log`. The finished research and commercial-safe runs stay "done".
  * Any other run keeps its stamps, log and readiness output in `<work>/driver/phase2_<profile>[_smoke][_<tag>].*`.
    That covers a tag, and also `PHASE2_WORK_BASE` without a tag. Such a run never reads, skips on or deletes the
    real run's stamps.
  * A tag without `PHASE2_WORK_BASE` gets its own work folder, `artifacts/isl/runs/<tag>/<profile>`. A tag with
    `PHASE2_WORK_BASE` set to `artifacts/isl` is refused (exit 2), because it would share the real state files.
  * A non-default work folder sends `evaluate` and `build-pack` to `<work>/pack`, never to `artifacts/packs/<id>`.
  * Smoke runs without a tag are unchanged. With a tag they use `artifacts/isl/phase2-smoke/runs/<tag>/`.
* The data steps of a scratch run, and how `--from` and a failed run interact, were fixed after review: see 15.20.

**Resume signatures (`training/loop.py` and the three trainers).** A stale state with any changed training option
now raises `StateMismatch` instead of resuming. `LoopConfig.extra` now also holds:
* classifier: the whole `TrainConfig` except `patience` and `device` (dropout, label smoothing, batch size,
  augment, class weighting...), the split name, `min_clips`, `min_signers`, `use_extra`, `islrtc_oversample`,
  `class_power`, the merged source weights, a hash of the class list, and a hash of the training list in sampling
  order (clip id, trim slice, landmark file name, label, weight) and of the validation list;
* encoder and oneshot: the same `TrainConfig` fields, the split, the encoder version, `arc_weight`, `use_extra`, the
  source weights, a hash of the excluded concepts, the SHA-256 of the warm-start file (a re-run pretrain at the same
  path is not resumed on), the training-list hash, and the validation settings and query / gallery hashes;
* pretrain: a hash of the clip list.
`patience`, `max_minutes` and `save_every_s` may still change between parts of one run.
Rule for later work (Batch 1 sampler, quota, selection options): put every new option that changes training into
`extra`. The helpers are `digest`, `clips_digest`, `file_digest` and `train_config_signature` in `loop.py`.

Consequence: every existing `*.state` file was written with the old, shorter signature, so it is refused on
`--resume`. That is safe: a stamped step is skipped anyway, and `--from` / `--force` now delete the state first.
`--only` on a finished training step with such an old state now fails with `StateMismatch` instead of returning the
old weights. On a state written by this code with the same settings, `--only` still returns the finished weights.
To retrain, use `--force STEP`. No training job was running when this landed.

Tests: `backend/tests/isl/test_phase2_guards.py` section 9. A dry run `--from classifier` lists `classifier.pt.state`,
`cv/fold*/classifier.pt.state` and `e1a/...` and keeps the encoder's. A real run with a stand-in python removes them.
`--force a,b` parses, and a bad name is refused. A tagged scratch run leaves the real stamps byte-identical and uses its own.
`run_loop` with a finished state and a changed config raises `StateMismatch`. The classifier and encoder refuse a
state after a change to class power, source weights, ISLRTC oversampling, label smoothing, dropout, batch size,
`use_extra`, `arc_weight`, validation size or the warm-start file. `test_trim_eval_e8.py` now reads the stamps of its
`PHASE2_WORK_BASE` run from `<work>/driver/`.

### 15.20 Driver: scratch runs never rewrite the shared data; `--from` is applied at once (review of 15.19, 27 Nov 2026)

**Bug 1: scratch runs wrote the shared data.** 15.19 said "start a scratch run at `--from bench`". But `--from`
only forgot steps from STEP on. Earlier steps without a stamp still ran, and a new scratch run has an empty stamp
folder. So a scratch `--from bench` ran `trim`, `dedup` and `splits`, which rewrite `data/isl/derived/` and
`data/isl/splits/`. The real run reads those files.

**Fix.** A scratch run is a full run with a tag or its own `PHASE2_WORK_BASE` (smoke runs already have a private
ISL root and are not affected).
* `trim`, `dedup` and `splits` are refused in a scratch run (exit 2), before anything is written. This covers a
  fresh run without `--from`, `--from trim`, `--force splits` and `--only dedup`. `--allow-shared-data` lifts the
  refusal for someone who really means to refresh the shared data.
* With `--from STEP`, the earlier steps that have no stamp are **seeded**, not run. The driver copies what later
  steps read from the real work folder (`artifacts/isl/<profile>`, or `PHASE2_SEED_BASE/<profile>`):
  `vocabulary.jsonl`, `pretrained_encoder.pt` and `pretrain_clips.txt`, `encoder.pt`, `oneshot/`, `classifier.pt`,
  `cv/`, `e1a/`, `letters.pt`, `index/`, `fusion.json`, `trim_eval.json`, and the `link` reports. It never copies a
  training state (`*.state`, `*.state.pt`, `*.tmp*`). A file already in the scratch work folder is kept. Then it
  writes the stamp `seeded from <folder> (scratch run --from STEP)`.
* A required file that is missing (`vocabulary.jsonl`, `pretrained_encoder.pt`, `encoder.pt`, `classifier.pt`, for
  the steps being seeded) stops the run at once (exit 2). The other outputs are copied when present.
* A seed folder marked PARTIAL DATA marks the scratch run too.
* `link` writes only to the work folder (`multisource.json`, `sources.json`, `extra_concepts.jsonl`), so it is
  allowed. I found no `variant_links.json` writer in the code.
* `--dry-run` shows `would seed STEP` and `would copy FILE`, and exits 2 on a refusal, like a real run.
* A refused run writes nothing: the stamp and log folders are made only after these checks.
* The real run (no tag, default work folder) is unchanged: it runs its data steps as before.

**Bug 2: `--from` was applied one step at a time.** Stamps and states were forgotten inside the loop, just before
each step ran. After a failure, the advice was "run the same command again". With `--from` that forgot the finished
steps again and retrained them (hours). Without `--from` it skipped the later steps on their OLD stamps, although
earlier models had changed, and shipped a stale pack.

**Fix.** The driver now plans first. It forgets STEP and every later step (and every `--force` step) in one pass,
after the data guard and before the first step runs. After a failure, run again WITHOUT `--from` / `--force`: the
failed step resumes from its state and every later step runs, and the finished earlier steps are skipped. The
failure message now says exactly that (with `--only`, it says to run the same `--only` command again).

**Smaller fixes.**
* An option that needs a value refuses an empty or missing one (exit 2): `--tag ""`, `--tag` as the last
  argument, `--from ""`, or `--tag --dry-run`. Before, `--tag ""` silently meant the real run.
* `--from pretrain` / `--force pretrain` also delete `<work>/pretrain_clips.txt`. The trainer unions a new clip
  list with an old one on `--resume`, so the recorded list (and its sha256 in the meta) would have included clips
  the new pre-training never saw.
* `--profile both` also passes `--allow-shared-data` to both children (it already passed `--tag` and `--dry-run`).

Tests (`test_phase2_guards.py` section 9): a fresh scratch `--from bench` seeds trim..link and never calls `trim`,
`dedup`, `splits`, `vocab` or `link-sources`, copies `vocabulary.jsonl` and no state file, and is refused without
`--from` (nothing written) or with `--force splits` / `--only dedup`; a missing seed file fails fast; `--from encoder`
failing at the classifier leaves classifier..pack unstamped, and a plain re-run runs classifier..pack and not the
encoder; `--from pretrain` lists `pretrain_clips.txt`; `--profile both` passes the tag and the dry run to both
profiles; empty option values are refused; pre-training refuses a finished state after one clip of the list changed.

### 15.21 A separate final index, and a leak guard on the index (Model QA fix 0.2 / RC8, 27 Nov 2026)

**Bug.** The driver's `index` step built a training-only index in `<work>/index`. Fit-fusion and evaluate read it.
But the later `index_final` step (`build-index --all-include`) wrote to the same folder. It holds every reference,
validation and test clips too. So after a full run, `<work>/index` was the eval-including index. Any later fit-fusion
or evaluate (`--from fusion`, `--only evaluate`, the planned re-runs) would have measured on clips that were in the
index. Both real work folders were in this state (their `index/meta.json` has the `index_final` time).

**Fix: two folders.**
* `build-index` writes to `<work>/index` (training parts only). With `--all-include` it writes to
  `<work>/index_final`. `--out` picks another folder. `--all-include --out <work>/index` is refused (exit 2).
* The driver passes the folders explicitly: `index` → `--out <work>/index`, `index_final` →
  `--all-include --out <work>/index_final`. A scratch run seeds `index/` and `index_final/` separately.
* `build-pack` ships `<work>/index_final` as the pack's `index/`. Without it (an older work folder), it ships
  `<work>/index`. The app may use the eval-including index: the app is not measured with it.
  `pack.json` `components.index` now says `built_from`, `contains_eval` and `encoder_sha256`.

**Two new fields in `index/meta.json`.**
* `contains_eval`: `true` for an `--all-include` build, `false` for a training-only build. An index file written
  before this change has no field and loads as `None` (unknown).
* `encoder_sha256`: the SHA-256 of the encoder file the rows were embedded with. `ENCODER_VERSION` is a constant
  (`isl-enc-1`), so it cannot tell a retrained encoder from the old one. Older files load with `""`.

**The leak guard** (`EmbeddingIndex.check_for_evaluation`, raises `LeakageError`). It runs in `load_recognizer`
(the default mode, used by fit-fusion and every evaluation), at the start of `fit_fusion`, in phase-1 `evaluate`
and in `evaluate --all`. It refuses:
* an index with `contains_eval` true;
* an index with `contains_eval` unknown. **Rule: an old index is never used to fit or evaluate, wherever it lives.**
  A folder name cannot prove it is clean: the old `index_final` step wrote exactly to `<work>/index`. The fix is
  cheap: rebuild it with the `index` step (72 s commercial-safe, 280 s research);
* an index whose `encoder_sha256` differs from the `encoder.pt` in use, or that records none;
* an index with a row whose clip id is one of the clips being fitted or measured (a direct check on the rows that
  carry a clip id: ISLRTC rows and single-clip rows of other sources).
`evaluate --all` writes the check's result to `report.json` `leakage_guards.index`. `LeakageError` is now one class
(`app.isl.models.index`); `app.isl.evaluation.runners.LeakageError` is the same class. The CLI turns it into a
one-line message and exit code 6 (`EX_LEAKAGE`), so the driver log says "step fusion FAILED (exit 6)".

**The app.** `EmbeddingIndex.load(..., encoder_path=...)` compares the SHA-256 when the index records one and raises
`IndexEncoderMismatch` (a kind of `IndexVersionMismatch`). The pack loader (`app/packs/isl.py`) and
`load_recognizer(..., for_app=True)` pass the encoder file. `for_app=True` reads `<work>/index_final` when it exists.
`build-pack` refuses an index embedded with another encoder file. Packs built before this change have no SHA in
their index and load as before. The ASL path is unchanged.

**Other.** `EmbeddingIndex.save` writes into `<folder>.tmp` and then swaps it in. A killed build no longer leaves
new embeddings next to an old `meta.json`.

**Re-runs (0.6).** `--force index,fusion,trim_eval,evaluate,index_final,pack` now does the right thing. `index`
runs first and rebuilds `<work>/index` from the training parts. Fusion and evaluate read it. `index_final` writes
its own folder, and `pack` ships it. Until that re-run, fit-fusion and evaluate on the real work folders are
refused (exit 6), because their `<work>/index` is an old, eval-including index. `--from fusion` alone is refused
for the same reason: start at `--from index` or force `index` too.

Tests (`test_index_leak_guard.py`): an old-format index loads (fields unknown) but is refused for evaluation;
fit-fusion refuses an index with `contains_eval` and one with another encoder SHA; `load_recognizer` refuses the old
driver's layout (final index in `<work>/index`), an old-format index and a retrained `encoder.pt`, and gives the app
`index_final`; the pack ships `index_final`, falls back to `index`, and refuses a stale one; an old pack loads in the
app loader; `build-index --all-include` writes `index_final` and never `index`; `fit-fusion` on the eval-including
index exits 6 and writes no `fusion.json`; the driver passes `--out` for both index steps and the 0.6 re-run order
is index, fusion, trim_eval, evaluate, index_final, pack.

### 15.22 The index leak guard sees every row; stale and missing indexes (review of 15.21, 27 Nov 2026)

**Gap.** `contains_eval = false` only said "built from the training parts of the split assignment in force at build
time". The meta did not say which split or which clips. The direct clip check only saw rows with a `clip` id. Most
rows have none: INCLUDE centroids, multi-clip `centroid` rows and `signer_centroid` rows. Two cases slipped through:
(a) the split assignment changes (Batch 1 changes the ISL500 `signer_split`), and `--force fusion,evaluate` runs
without rebuilding `index`, so old centroids hold clips that are now val / test clips; (b) `fit-fusion` or `evaluate`
runs with `--split <other>` against an index built for the primary split. In both, the guard passed and
`report.json` said `training_parts_only: true`.

**Fix.**
* Two new fields in `index/meta.json`: `split` (the name of the split that chose the clips) and
  `train_clip_ids_sha256` (SHA-256 of the sorted INCLUDE and other-source clip ids given to the build;
  `index.clip_ids_sha256`). `index_final` records them too (there: every reference).
* Every `centroid` and `signer_centroid` row lists its clips in `clips`. `EmbeddingIndex.row_clip_ids()` returns
  `clip` and `clips` of every row, and the clip check uses it. So the check covers every row. The shipped pack's
  `rows.jsonl` grows by the clip ids (about 1 MB per 30,000 clips). The app ignores the field.
* `check_for_evaluation` now also refuses: an index that does not record `split` and `train_clip_ids_sha256`
  (written before this change, including by 15.21's code: rebuild with the `index` step); a different `split`; a
  different training-clip digest. The digest of the current data comes from
  `build_index.index_training_clip_ids(data)`: INCLUDE `train` part plus other sources' `train` part, labelled only.
  This is exactly what the `index` step indexes.
* `fit_fusion(..., data=)` and `check_index_for_fitting(rec, clips, data)` pass the split and the training clips.
  `cli fit-fusion`, the smoke command, phase-1 `evaluate` and `evaluate --all` pass `data`.
* `report.json` `leakage_guards.index` adds `split`, `split_checked` and `train_clip_ids_checked`.
  `training_parts_only` is true only when `contains_eval` is false and both were checked.
* **Rule:** the digest must match exactly. New data (a source finished ingesting) also changes it, even when no clip
  moved between parts. Then fit-fusion and evaluate are refused (exit 6) until `index` is rebuilt. This is on
  purpose: the index then also matches the data the run measures.

**A recogniser with an index but no encoder path** (custom or test callers) is refused by
`check_index_for_fitting` (`LeakageError`). Before, the SHA check was skipped silently.

**Stale `index_final`.** If `index` was rebuilt and `index_final` was not, `index_final` holds older data or older
row logic. Two changes:
* The driver's `index` step, on success, forgets the stamps of `fusion`, `evaluate`, `index_final` and `pack`. They
  used the old index or ship next to it, so they run again.
* `pipeline.shipped_index(work)` picks the shipped index. It is `index_final` unless that one's `created` is older
  than `index`'s. Then it is `index`, with a note. `build-pack` prints the note and writes it to `pack.json`
  `components.index.note`. `load_recognizer(for_app=True)` makes the same choice.

**Missing `work/index`.** In the default (measuring) mode, `load_recognizer` used to build a recogniser without
retrieval when `work/index` was missing. Fit-fusion then fitted classifier-only parameters and evaluate said "no
retrieval index", with no error. Now, when the encoder exists and `work/index` is missing but `index_final`,
`index.old` or `index.tmp` exists, it raises `IndexMissingError` (a `LeakageError`, so the CLI exits 6). `index.old`
and `index.tmp` are what a save killed between its two renames leaves. We do not restore `index.old` automatically:
the message says to run the `index` step. A work folder with no index at all still loads without retrieval, as
before.

Tests: `test_index_leak_guard.py` (split and digest rules, centroid clips, a pre-15.22 index refused, the recogniser
without an encoder path, fit-fusion after one INCLUDE train clip moves to val (refused by the row check and by the
digest), `--split <other>` refused in the API and the CLI, a missing `work/index` next to `index.old` /
`index_final`, `evaluate_all` writes the guard summary, CLI `evaluate --all` and phase-1 `evaluate` exit 6 on the
eval-including index and write no report, the pack falls back from an older `index_final`, the driver's `index` step
forgets the later stamps). `test_eval_runners.py`: other-source centroids list their clips, and every listed clip is
a training clip.

### 15.23 Fit-fusion is fast and small: one compact evidence pass, the weight grid on a subsample (Model QA fix 0.3 / RC11, 27 Nov 2026)

**Problem.** Research `fit-fusion` took 8,144 s and about 9.7 GB RSS for 13,156 validation clips. Measured causes:
the 81-point weight grid ran a full `fuse()` per clip per point (54 % of the time); `choose_user_mode` computed every
clip's evidence a second time for its E4 cache (18 %); each clip kept the full retrieval matrix (3 x 12,440 cosines,
149 KB), a prior dict over about 4,400 keys (104 KB) and, in the sliding replay, about ten full EMA vectors
(about 35 KB each). On a 16 GB machine shared with other jobs, and with Batch 1 doubling the ISL500 validation set,
that does not fit.

**Fix.** Nothing here changes what `fuse()` or fit-fusion decides; only the cost changes, except the weight grid's
subsample (last point).
* **Shared prior.** `FusionInputs.prior` is now one read-only `ConceptPrior` per recogniser (`rec.prior`), not a dict
  per segment. It gives `Vocabulary.prior` for any vocabulary concept and 0 for user signs and unknown keys, as the
  old dict did for every key `fuse()` reads, as long as the recogniser's `top_n` is the fitting `top_n` (see 15.24).
* **Compact retrieval evidence** (`fusion.compact_inputs`, used by fit-fusion only). A clip keeps:
  * the best `top_n` (50) index concepts by crop-mean cosine, already in `fuse()`'s order;
  * the index columns of the classifier's best 2 x `top_n` classes (a classifier candidate keeps its real
    retrieval score, and one without an index row still gets the neutral floor);
  * per crop, the retrieval softmax normaliser (log-sum-exp of `ret / tau` over all index concepts) for each `tau_r`
    of the grid. It is computed with the same expression as `fuse()`, so without user signs the result is bit for
    bit the same. With user signs the normaliser is joined by log-add-exp. It is a per-crop constant of every
    candidate, so it cancels in the softmax up to float rounding (worst seen: 2e-14).
  That is about 150 of 12,440 columns (about 2 KB instead of 149 KB). `fuse()` raises a `ValueError` for a `tau_r`
  without a stored normaliser or a `top_n` larger than the kept top, never a silent wrong answer. The values stay
  float32. The plan said float16, but float16 cosines divided by `tau_r = 0.02` move the fused probabilities by far
  more than the required 1e-6, and the saving (300 bytes per clip) is not worth it.
* **One evidence pass.** `collect_evidence(..., e4_cache=)` stores `(inputs, embedding)` per clip with a word and a
  signer group, and `choose_user_mode(..., cache=)` uses it. The embedding is the one `rec.evidence` already computed
  (or the feature-statistics embedding when there is no encoder), so the E4 numbers are the same.
* **Weight grid on a subsample.** `fit_fusion(weight_clips=3000, weight_seed=0)` runs the 81-point grid on a seeded,
  source-stratified subsample (`fit_fusion.weight_subsample`: each source keeps its share, at least one clip). The
  chosen point is then scored on every clip, and temperature, thresholds, suggest masses, the sliding thresholds and
  the "My signs" choice still use every validation clip. With at most 3,000 clips nothing is subsampled, exactly as
  before. fusion.json gets a new field `weight_fit`: `subsample`, `seed`, `max_clips`, `stratified_by`,
  `subsample_by_source`, `n_full`, `grid_points`, `nll_subsample` and `nll_full`. Older fusion.json files have no
  such field and load with `weight_fit = {}`.
* **Compact sliding replay.** `replay_sliding` keeps each clip's windows as `calibration.PreparedWindows`: per window
  only what the commit rule reads (time, top-1, top-1 minus top-2, class, streak, boundary, key). The EMA vectors are
  dropped as soon as a clip is replayed. Every sliding function accepts both forms and gives identical counts and
  thresholds. The replay itself is unchanged (it is the app's engine, so it stays the reference).

**Also fixed: E4 was not reproducible.** `e4_my_signs` enrolled words in the order of a Python `set` of strings,
and that order changes per process (string hashing). The order decides which takes the seeded generator picks, so
two runs of the same fit-fusion could choose differently (on a 536-clip research sample: top-1 with enrolment 0.936
in one process, 0.925 in another). The words are now enrolled in sorted order. E4 numbers change once (other takes
are drawn), then stay the same in every run.

**Measured** (research models copied to a scratch folder, 2 threads, same 536-clip validation sample as the RC11
check, old = 8abf6f2 with the E4 order fix so both are deterministic):

| 536 clips (513 ISL500, 23 INCLUDE) | old | new |
|---|---|---|
| total | 224 s (0.418 s per clip) | 82 s (0.154 s per clip) |
| weight grid | 127 s | 25 s |
| E4 ("My signs" choice) | 35 s | 2.6 s |
| evidence | 21 s | 15 s |
| sliding replay | 39 s | 38 s |
| peak RSS growth during fit-fusion | 340 MB (0.63 MB per clip) | 56 MB (0.10 MB per clip) |

* Old and new give the same `fusion.json` (every field except the new `weight_fit`) and the same "My signs" report,
  run in two processes with different string-hash seeds.
* On 2,007 clips the full grid and a 450-clip subsample (the same ratio as 3,000 of 13,156) chose the same point
  (alpha 0.5, beta 1.0, gamma 0, tau_r 0.05), so every threshold and achieved number was identical. The grid took
  94 s on every clip and 21 s on the subsample. Peak RSS growth was 316 MB (0.16 MB per clip, with the grid's
  temporaries).
* What each clip still holds: the classifier log-probabilities (3 x 4,354 float32, 52 KB, the largest part), compact
  retrieval (about 1.5 KB), the embedding for E4 (3 KB), the fused result (about 4 KB) and the prepared windows
  (about 2 KB).
* **Expected full research run** (13,156 clips): about 1,700 s at 2 threads on an idle machine (old code: about
  5,500 s on the same basis, and 8,144 s measured in the real run). Scaled by the same factor, the real run should
  take about 0.7 h. Memory should be 1.2-2.1 GB above the 1.05 GB the loaded data and models take, so about 2.3-3.2
  GB peak instead of about 9.7 GB. The sliding replay (about 0.08 s per clip) is now about 60 % of the time. It
  runs the app's own engine, so making it faster means changing the engine.
* **Left open:** the classifier log-probabilities could be kept compact too (best classes plus a per-temperature
  normaliser, as for retrieval). That would roughly halve the remaining memory. It also needs the temperature fit
  and the segment-proxy fallback to read the full rows first, so it was not done here.

Tests: `test_fusion_efficiency.py` (the pre-15.23 `fuse()` kept as a reference: compact inputs plus the shared prior
give the same keys, probabilities within 1e-6 (bit for bit without user signs), crops, cosines and decisions for
every policy on random 300-class / 1,500-concept inputs with tied columns, with and without user signs and both
user modes; refusals; the shared prior; the subsample; a toy fit-fusion gives exactly the parameters and "My signs"
report of the old path with one evidence pass instead of two; a subsampled fit records `weight_fit` and still fits
thresholds on every clip; a fusion.json without the new field loads; prepared windows give identical sliding fits).
`test_eval_runners.py`: fit-fusion's compact evidence fills the E4 cache, and `choose_user_mode` then computes no
evidence and reports what a fresh pass reports.

### 15.24 Fit-fusion review fixes: compact classifier evidence, a faster sliding replay, a safer grid subsample (review of 15.23, 27 Nov 2026)

**Problem.** The review of 15.23 found that fit-fusion still missed the plan's memory target, and that three smaller
points were open:
* Each validation clip still kept its full classifier log-probabilities (3 x 4,354 float32, 52 KB) and its own copy of
  the class list (35 KB). That was about 93 KB per clip, so the full research run would need about 2.3-3.2 GB, and
  about 4-6 GB once Batch 1 doubles the ISL500 validation set. The plan asks for under 2 GB.
* The sliding replay was about 60 % of fit-fusion's time. It called the classifier once per window.
* Only the subsample's best weight point was scored on every clip. A near-tie on the subsample could pick a worse
  point, and nothing would show it.
* `weight_subsample` gave every source at least one clip. With many tiny sources the subsample grew above `n`. It
  also ignored signer groups.

**Fix.**
* **Compact classifier evidence** (`compact_inputs(..., t_cs=...)`, fit-fusion only). A clip keeps:
  * the columns of the classifier's best 2 x `top_n` classes by crop-mean log-probability (best first), plus the
    classes of the kept retrieval columns;
  * per crop and per temperature `t_c`, the two numbers `_log_softmax` computes over all classes (the row maximum and
    the log-sum-exp after it). The temperatures are the 49 points of `fit_temperature`'s grid, 1.0 and the default.
  `fuse()` rebuilds the kept log-probabilities with the same arithmetic, so they are bit for bit the old values. The
  row maximum that the "neutral" user-sign mode reads is `0 - log-sum-exp`, also exact. `fuse()` raises a
  `ValueError` for a `t_c` without stored numbers, a `top_n` above the kept classes, or full retrieval next to a
  compact classifier. It never gives a silent wrong answer.
* **What fit-fusion still needs from the full rows** is taken during the same evidence pass
  (`collect_evidence(stats=...)`):
  * for the temperature fit, crop 0's 49 terms per clip whose word is a class (`calibration.temperature_nll_terms`).
    `fit_temperature_terms` then picks exactly what `fit_temperature` picks;
  * for the segment-proxy fallback of the sliding thresholds, a `ClassifierSummary` per clip at the checkpoint
    temperature: crop-mean top-1 and top-2, the top class and each crop's top class. `fit_classifier_thresholds`
    gives the same result from these summaries as from the full rows.
* **Batched sliding replay** (`fit_fusion.batched_sliding_windows`, the default in `replay_sliding`). The app's engine
  stays the reference. The engine runs once to find the windows it classifies; that never depends on the
  probabilities. Their features go through the classifier as one batch. Then the engine runs again with those
  probabilities injected, so EMA, streak, boundaries and repeat keys still come from the engine's own code.
  A batched call can differ from a one-window call by float rounding (up to about 4e-7 on the research classifier).
  So when any window's EMA top-1 is within 1e-5 of a threshold of the grid, or its top-1 minus top-2 is within 1e-5 of
  the margin or of 0, that clip is replayed window by window. The commits, counts and thresholds are therefore the
  engine's own. The report says how many clips went each way (`classifier_alone.replay_clips`).
* **Grid re-scoring.** The best 5 subsample points (`WEIGHT_FIT_RESCORE`) are scored on every clip, and the best full-set
  NLL wins. `weight_fit` records them (`rescored`, each with `nll_subsample` and `nll_full`) and
  `subsample_best_kept` (false when re-scoring changed the choice).
* **Subsample size and signers.** When the one-clip minimums push the total above `n`, the excess comes back from the
  largest sources, so the subsample has exactly `n` clips unless there are more sources than `n` (then one clip per
  source). Inside each source, signer groups keep their share by largest remainders, so ISL500's four signers are
  balanced. `weight_fit.stratified_by` is now "source, then signer group". With at most 3,000 validation clips nothing
  changes: there is no subsample.
* **The shared prior and `top_n`.** The old per-segment prior dict held only the classifier classes and the
  recogniser's own best `top_n` retrieval concepts; any other candidate got prior 0. Fit-fusion fits with the default
  `top_n` (50). If a recogniser's `top_n` was smaller, the extra retrieval candidates got prior 0 in the old path and
  get their real `Vocabulary.prior` now. That was a latent mismatch between fitting and the app; the shared prior
  makes them consistent. With equal `top_n` (every pack so far) nothing changes.

**Measured** (the research models copied to a scratch folder, 2 threads, old = 15.23 at commit e7a6160):

| | 536 clips, 15.23 | 536 clips, now | 2,007 clips (grid on 450), 15.23 | 2,007 clips (grid on 450), now |
|---|---|---|---|---|
| total | 88 s | 63 s | 278 s | 196 s |
| evidence | 17 s | 18 s | 63 s | 67 s |
| weight grid | 26 s | 18 s | 22 s | 15 s |
| sliding replay | 42 s | 25 s | 176 s | 100 s |
| E4 ("My signs" choice) | 2.6 s | 1.9 s | 10.8 s | 9.1 s |
| peak RSS growth | 94 MB | 38 MB | 355 MB | 48 MB |

* Both runs gave the same `fusion.json` as 15.23 (every field except `weight_fit`), the same sliding thresholds and the
  same "My signs" report. On 2,007 clips the subsample's best point stayed best on every clip (nll 0.3316; the next
  re-scored points were 0.3336-0.3377).
* Evidence kept per clip after the evidence pass (tracemalloc, 300 clips, with the E4 cache): 93 KB before, 12.9 KB
  now. Adding the fused results (about 4 KB, twice while re-scoring) and the prepared windows (about 2 KB), the peak is
  about 23 KB per clip.
* **Expected full research run** (13,156 clips): about 1,300 s at 2 threads on an idle machine (evidence about 440 s,
  replay about 650 s, grid on 3,000 clips plus 5 re-scoring passes about 120 s, E4 about 60 s). Scaled like 15.23 to
  the real run, that is about 0.5 h. Memory: about 0.3 GB above the 1.05 GB of loaded data and models, so about
  1.35 GB peak. With the ISL500 validation set doubled (about 25,600 clips): about 1 h and about 1.65 GB peak. Both are
  under the 2 GB target.

Tests: `test_fusion_efficiency.py`: compact classifier inputs give the old `fuse()` result (bit for bit without user
signs, within 1e-6 with them, same decisions, both user modes); refusals; temperature terms clip by clip equal
`fit_temperature`; summaries give the same segment-proxy thresholds; a toy fit-fusion with a classifier and the proxy
fallback gives exactly the old temperature, weights, thresholds and classifier-alone thresholds; the batched replay
gives exactly the engine's windows on synthetic clips (with a sign boundary) and falls back near an edge; with
batch-dependent rounding the sliding counts and thresholds equal the reference; the subsample with many 1-clip
sources has exactly `n` clips and balanced signers; a subsampled fit records the re-scored points and keeps the best
full-set NLL.

### 15.25 Honest calibration: per-source certification and confirm-only policies (Model QA fix 0.4, 27 Nov 2026)

**Problem.** Thresholds were fitted on all validation clips pooled, with a Wilson bound that counts every clip as
independent. Research validation is 93 % ISL500 from 2 signers. The pooled bound said "strict", but INCLUDE got 0.918
(val) / 0.938 (test), and new ISL500 signers 0.969. When no target was reachable, the graded fallback (15.17) still
let every policy auto-accept. Nothing in the app read `target_reached`.

**What changed** (`models/calibration.py`, `models/fusion.py`, `training/fit_fusion.py`, `recognizer.py`,
`inference/engine.py`, `evaluation/runners.py`, `packs/isl.py`, `training/pipeline.py`):

1. **Per-source certification.** Every fit gets each clip's `source` and signer `group`. A source is *major* when it
   has at least 100 clips, at least 30 accepts at the lowest threshold and at least 2 groups (15.26; on research
   validation that is INCLUDE and ISL500, not Mendeley with 39 clips). At each threshold, the bound is the **worst
   major source's** Wilson lower bound on `n / DEFF`. `DEFF = max(1, phi)`, one value per source: phi is the median
   Pearson dispersion of accepted errors over that source's groups, over the thresholds with at least 3 groups and 10
   errors (15.26). Without such a threshold, phi comes from `phi_prior` (for example cross-validation groups) or is
   5.0 (the ISL500 4-signer value), never 1. No cross-fitting and no Bonferroni: threshold selection bias was measured
   at 0.0001-0.0008.
2. **Finer grid.** `THRESHOLD_GRID` is 0.20-0.99 by 0.01, then 0.991-0.999 by 0.001 (fused and sliding). Strict can
   now sit above balanced when only the top 1 % is precise enough.
3. **Floors and the confirm-only rule.** Each policy aims at `min(target, best worst-source bound)` and takes the
   smallest threshold that reaches it. It is `certified` only when that bound is at least its floor: strict 0.95,
   balanced 0.90, lenient 0.85. Otherwise it is `confirm_only`. This replaces the graded fallback of 15.17.
   * `decide()` never returns `accept` for a confirm-only policy. A query that passes the accept rule becomes
     "did you mean" (`suggest` / `confirm_only`), whatever its top-5 mass.
   * A direct "My signs" match (7.3) is a "did you mean" too under a confirm-only policy (15.26). Under a certified
     policy it keeps its own per-sign rule.
   * The rule lives in `fusion.json` `achieved.<policy>.rule` (with `by_source`) plus a `certification` record
     (version `cert-1`). `FusionParams.policy_rule()` reads it; the recogniser and the engine use it.
   * Sliding mode: the engine never auto-commits with a confirm-only policy. The word becomes a "did you mean"
     `segment_result` the user can choose (15.26). It is certified only when `classifier_fit.rule` is "sliding replay" for this very classifier and that
     policy's worst source reached its floor. The segment proxy is never certified.
4. **Out-of-vocabulary guard.** `s_min` is the minimum over major sources of each source's 5 % quantile of the best
   cosine (it was pooled). So at least 95 % of each major source's in-vocabulary queries pass. E2 now also measures
   held-out-concept false accepts: the held-out queries against the gallery **without** their concepts, where every
   accept is wrong (`E2.oov_false_accept`, per policy, with the gate and with the threshold alone).
5. **"Did you mean" mass.** The smallest mass at which the **worst** major source's top-5 hit rate in that policy's
   pool is at least 0.80. No mass may cut a major source below 20 % of its pool. Sources with fewer than 3 suggestions
   are not judged.
6. **Open-vocabulary words** (outside the classifier, retrieval only; every word in a pack without a classifier) are
   always confirm-only unless the policy is certified **and** their own accepts reach the strict floor (0.95) on every
   major source (`certification.open_vocabulary`). `decide(..., open_vocabulary=True)` returns `suggest` /
   `open_vocabulary` otherwise. The recogniser and E3 pass the flag.
7. **Reporting.** `achieved.<policy>` holds `rule`, `certified_precision` (the worst-source bound when certified),
   `worst_source`, `floor`, `pooled_lower_bound` and `by_source` (precision, lower bound, coverage, n, n_groups, DEFF).
   E1 and E3 emit per-source rows (`by_source`, `worst_source`), what the app accepts by itself, and `threshold_rule`
   (what the threshold alone accepts, that is the candidate a confirm tap shows). `report.md` opens with a headline
   table: the INCLUDE row and the worst-source row per policy, for E3 and sliding mode.

**What the app shows** (critic point 15). The floors are below the 0.98 goal. So:
* The recogniser's `detail.certification` (and `engine.describe()["certification"]`, and the pack info
  `components.fusion.certification`) gives the rule, the certified worst-source precision (never the target), the
  worst source, the floor, the target and `confirm_tap: true`.
* The app should keep a confirm tap before a word is spoken or sent, even when strict is certified. Since 15.26
  every `segment_result` and `commit` of a language pack carries `certification` and `needs_confirm`.
* **Open item (frontend follow-up).** The frontend does not read these fields yet. It should show the certified
  precision next to a word and require a tap before a committed word with `needs_confirm: true` is spoken or sent.
  (Done in 15.29.)

**Old packs.** A `fusion.json` without the `certification` record loads as `{}`: uncertified, every policy
confirm-only, never a crash. That includes the commercial-safe file with accept 0.96 x 3 and no classifier
thresholds, and the research file. An old `achieved` block (rule "target" or "fallback", no `by_source`) is not a
certification. A pack without `fusion.json` gets uncertified defaults too. Parameters built in code (tests, tools)
have `certification=None` and are not gated. The ASL path is unchanged (the engine gate needs a language-pack
recogniser and no predictor).

**TODO (critic point 8, not done here).** The clips that certify are the validation clips that also selected the
models (early stopping, temperature). The cheaper and stronger fix is to calibrate INCLUDE on out-of-fold predictions
of the cross-validation folds (about 4k clips over all sessions), or to hold one INCLUDE session out for calibration
only. The API is ready: calibration clips are columns with a source and a group (`certify_scan`), and a per-source
`phi_prior` can come from the CV groups. `certification.selection_note` says this in every `fusion.json`.

**Next step.** Refit commercial-safe and research with this code (plan 0.6). Until then both packs load as
confirm-only.

Tests: `test_policy_thresholds.py` (pooled passes but the worst source fails; a small source is not major; DEFF lowers
the bound only for heterogeneous groups, the 5.0 fallback and the prior; a weak scan is confirm-only everywhere and
`decide()` never accepts; the fine grid lets strict sit above balanced; `s_min` is the per-source minimum; the suggest
mass is chosen on the worst source; an old `fusion.json` is confirm-only; open-vocabulary words stay confirm-only;
the recogniser exposes the certified precision; E3 per-source rows and the headline; E2 held-out false accepts) and
`test_sliding_replay_thresholds.py` (sliding thresholds certified on the worst source; sliding mode never
auto-commits with a confirm-only policy, an old pack or a proxy fit; fit-fusion certifies the replay per source; E1
reports the rule and per-source sliding rows).

### 15.26 Calibration review fixes: which sources count, a DEFF that does not vanish, confirmable sliding words (review of 15.25, 27 Nov 2026)

**Problems found in review of 15.25.**
* A source counted as major with only 30 accepts. On research validation Mendeley (39 clips, 3 signers) was major.
  A source of n clips can never have a bound above `wilson(n, n)`, which is 0.91 for 39 clips. So strict could
  never be certified, every policy landed on 0.96 with Mendeley as the worst source, and it also set `s_min`
  (0.734 instead of INCLUDE's 0.756).
* DEFF was re-estimated at each threshold. With 0 accepted errors it was set to 1, and with 1 error it is about 1 by
  construction. INCLUDE's DEFF was 2.1-2.8 at 0.80-0.98 but 1.0 from 0.993 up, and balanced and lenient were
  certified only because of that drop.
* In sliding mode a confirm-only word was only a provisional list. It had no `segment_id`, so the user could not
  confirm it. With both shipped packs uncertified, sliding mode could never produce text.
* `segment_result` and `commit` did not say which rule applied. A direct "My signs" match was still accepted under
  a confirm-only policy.

**What changed** (`models/calibration.py`, `inference/engine.py`, `isl/recognizer.py`, `evaluation/runners.py`):
1. **Major sources.** A source is major only with at least 100 clips (`MAJOR_MIN_CLIPS`), 30 accepts at the lowest
   threshold and 2 signer groups (`is_major_size`). 100 is above the 73 clips that `wilson(n, n) >= 0.95` needs, so
   a major source can reach the strict floor. A smaller source is still reported in `by_source`, with
   `major: false` and the note "too small to certify". It does not block or certify a policy, does not set `s_min`,
   is not judged for the "did you mean" mass and is not the E1/E3 worst-source headline. A caller can pass an
   explicit `major_sources` list (for example from a profile) instead of the size rule.
2. **One DEFF per source.** phi is computed at every threshold that has at least 3 groups with accepts and at
   least 10 accepted errors (`PHI_MIN_ERRORS`). Accepts and errors only shrink as the threshold rises, so these are
   the low thresholds. The source's phi is the median of those values, and `DEFF = max(1, phi)` at every
   threshold. So the clustering measured where errors exist also applies where they run out. The median keeps one
   noisy threshold from setting it. Without any such threshold: the prior, else 5.0. Never 1 for lack of errors.
   The open-vocabulary check uses each source's DEFF from the full scan as its prior.
3. **Sliding mode, confirm-only.** The word becomes a `segment_result` with `decision: suggest`,
   `reason: confirm_only` and a `segment_id`. The candidates are stored, so the existing `choose` message commits
   it. The repeat and cooldown rules apply, so one held sign gives one suggestion.
4. **Certification on every word** (critic 15). For a language pack, each `segment_result` and `commit` carries
   `certification` (rule, certified precision, worst source, floor) and `needs_confirm`. `needs_confirm` is true
   for a word the app accepted by itself, and false for a word the user picked. The ASL path is unchanged.
5. **My signs.** Under a confirm-only policy a direct "My signs" match is `suggest` / `confirm_only`, with the user
   sign on top. Under a certified policy, or with in-code parameters, it keeps its own rule.

**Effect on research validation** (evidence from the Model QA run, current research models, train-only index):
Mendeley is no longer major and `s_min` is INCLUDE's 0.756. INCLUDE's DEFF is 3.45 (median of 2.1-5.4). The best
INCLUDE bound is 0.818 at 0.97, so all three policies are confirm-only. That is the honest result: 52 of 52 at 0.994
was certified before only because DEFF dropped to 1 there.

**Open items.**
* Frontend: show `certification.certified_precision` next to a word, and require a tap before a word with
  `needs_confirm: true` is spoken or sent (see 15.25). (Done in 15.29.)
* Out-of-fold CV calibration (15.25 TODO) would add INCLUDE groups and clips, and is the main way to certify
  INCLUDE at all.

Tests: `test_policy_thresholds.py` (a 39-clip, 3-group source at 0.6 precision next to two strong sources is
reported but not major, strict certifies, it sets neither `s_min` nor the mass, and an explicit list overrides; DEFF
stays at the low-threshold phi where errors vanish, stays 1 for homogeneous errors, and is 5.0 with no errors; a
user sign is never auto-accepted under a confirm-only policy), `test_sliding_replay_thresholds.py` (a sliding
confirm-only word is a suggestion that `choose` commits; a certified commit has `needs_confirm`) and
`test_engine_recognizer.py` (segment results and commits carry the certification).

### 15.27 Protocol and engine fixes: E4 measures the app, a paired test for the user-sign mode, like-for-like E1b (Model QA fix 0.5, 27 Nov 2026)

**Problems found by the Model QA review.**
* E4 (My signs) did not measure what the app does. It enrolled 1, 3 and 5 takes, but the app needs 3 to 5. Its
  direct "My signs" match had no crop-agreement check. A user sign on top of the fused list was accepted even
  without the direct match, while the app only suggests it. It enrolled the mean of three crop embeddings, while
  the app's store embeds the whole take.
* E4's real regression was precision, and no report showed it well. Research k3 has a user-sign false-accept rate of
  0.0148, commercial-safe 0.0. The two packs also used different populations: research pooled INCLUDE and ISL500
  test signers (n 2,808), commercial-safe had only INCLUDE (n 89).
* `choose_user_mode` needed an absolute top-1 gain of 0.02. On research validation (4,689 takes) neutral won 84
  takes and lost none, a gain of 0.018, so penalty stayed. With 30 takes, +3 of 30 passed.
* Segment mode with a pack echoed `EngineConfig.threshold` in its messages, but the recogniser never reads it: it
  uses the pack's fused threshold for the policy.
* Sliding mode keyed its streak and repeat rule on the vocabulary index of the top class. Classifier classes that are
  missing from the vocabulary all have index -1 (7 research classes), so two different such words formed one streak
  and the second was suppressed as a repeat.
* fit-fusion and evaluation loaded `work/vocabulary.jsonl` without the link step's `extra_concepts.jsonl`. The pack
  appends those concepts, so while measuring they had prior 0 and index -1, and in the app prior 0.5.
* A user sign's `theta` was stored at enrolment, as a cosine of that day's encoder. After a pack update the takes
  are re-embedded but the old theta was still used.
* E1b of the two packs is not like for like: commercial-safe has 792 clips and 270 classes, research 852 clips and
  more classes. The report did not say which clips it measured.

**What changed.**
1. **One decision function.** `recognizer.user_sign_decision` holds the recogniser's decision after fusion: the
   fused rule, the direct "My signs" match (theta, margin over the dictionary, and the user sign must be the best
   user sign in at least min(min_agree, crops) crops), the confirm-only demotion, the "user sign on top without the
   direct match is only a suggestion" rule, and the user-signs-only rule. `IslRecognizer.recognize` and E4 both
   call it.
2. **E4 = the app.** `runners.E4_KS = (3, 5)`, as the store's 3 to 5 takes. Enrolment vectors are whole-take
   embeddings (crop 0), as `MySignsStore.user_index` makes them. Theta is `store.theta_from_embeddings` (lowest
   leave-one-out cosine minus 0.05; none, so never auto-accepted, when the takes are inconsistent).
3. **E4 false accepts first.** Each k reports `user_sign_false_accept_rate` (what the app accepts by itself; 0 under
   a confirm-only policy) and `user_sign_false_accept_rate_threshold_rule` (the same rules without the
   certification gate: the user sign the app puts first and asks to confirm), with counts and intervals. report.md
   shows both in bold, before the top-1 columns.
4. **E4 populations.** `evaluate --all` runs E4 on INCLUDE test signers only; that is the `E4` entry and the row to
   compare packs on. `E4.all_sources` adds the other sources' held-out signers. The evidence cache is shared.
   A gate for a new pack should be "INCLUDE-only false accept (threshold rule) <= commercial-safe's".
5. **User-sign mode by a paired sign test.** On the paired enrolled takes, b = takes only neutral gets right and
   c = takes only penalty gets right. Neutral is chosen when the one-sided exact sign test P(X >= b),
   X ~ Binomial(b + c, 1/2), is below 0.05, and the threshold-rule false-accept rate rises by at most
   `MAX_FA_INCREASE` (0.01). The threshold-rule rate is used because the app's own rate is 0 under a confirm-only
   policy. k = 3 first, then 5. So: 4,689 takes with +84 and none lost gives neutral (p about 5e-26); 30 takes with
   +1 gives penalty (p 0.5); +3 of 30 is still penalty (p 0.125); +84 net from 1,500 won and 1,416 lost is penalty
   (p 0.06). The report's `my_signs.sign_test` has b, c and p.
6. **Engine threshold.** In segment and spell mode with a pack, `threshold` in messages and `describe()` is the
   pack's fused accept threshold for the policy. Sliding mode still uses `EngineConfig.threshold` when it is set.
7. **Engine sliding key.** With a pack, the streak and the repeat rule use the classifier's own top class index.
   Commits still carry the vocabulary index in `class_index`. The ASL path (a predictor) is unchanged. fit-fusion's
   sliding replay records the engine's key, so it follows.
8. **Classifier thresholds cached.** `classifier_policy_thresholds` is computed once per (classifier, fusion) pair
   (`IslRecognizer._classifier_policy`, since 15.25); now tested.
9. **Extra concepts while measuring.** `pipeline.load_recognizer` appends the rows of `<work>/extra_concepts.jsonl`
   that the vocabulary does not have, in file order, exactly as `build_pack` does (`extra_concept_rows`). Nothing is
   written to disk. We chose this over writing the rows into `work/vocabulary.jsonl` in the link step: that file is
   the vocab step's output, and with the extra rows in it a second link run would link those glosses as "exact",
   write an empty `extra_concepts.jsonl` and change the link report.
10. **Theta follows the encoder.** `MySignsStore.user_index` recomputes every sign's theta from the vectors of the
    current embedder (`theta_from_embeddings`). `MySignsStore.user_theta(sign_id, embedder)` returns it, and the
    recogniser uses it (floors still apply). The stored `theta` is only a fallback for stores without this method.
11. **E1b records its clips.** Every E1 entry has `split`, `clip_ids` and `class_ids_sha256`. E1b and E1c have
    `per_clip.hit1` / `hit5` (one "0"/"1" per clip), and E1b has `class_ids`.
12. **E1b like for like.** `cli evaluate --all --compare-to <report.json>` (for example
    `artifacts/isl/commercial-safe/eval/report.json`) adds `E1b.like_for_like`: top-1 / top-5 with scores masked to
    the reference's classes (clips of those classes only), on the clip ids both reports measured, and both
    together. The reference's own top-1 / top-5 on those clips and the discordant counts are next to them. The
    split must match. A report from before this change has no clip or class ids: the class ids are then read from
    `classifier.pt` next to the report folder, and the common-clip rows say the reference must be re-run. Labels are
    strict; alias scoring is not part of this change (QA Batch 1).

**Backward compatibility.** Old fusion.json files load as before (an old one without a certification record is
uncertified, so E4's app false-accept rate is 0 and the threshold-rule rate is the one to read). Old stores keep
working: theta is recomputed from the takes. Old reports load in `--compare-to`. The ASL engine is unchanged.

**Effect.** E4 numbers change: k1 is gone, the enrolment vector and theta follow the store, and the headline is
INCLUDE only. Fit-fusion may now choose `neutral` for research. Re-run fit-fusion and evaluate (QA 0.6) to get the
new numbers.

**Open items.**
* Frontend: the E4 card shows the INCLUDE-only entry; it does not yet show `all_sources` or the threshold-rule
  false-accept rate.
* `in_classifier` is false on extra concepts, so their prior is 0.5, not 1, even when the classifier has them.

Tests: `test_eval_runners.py` (E4 and `IslRecognizer.recognize` with a real `MySignsStore` give the same decision
and top word for every test take, gated and ungated; the crop-agreement, below-theta and confirm-only rules of
`user_sign_decision`; E1b records clip and class ids and compares with a reference report, an other-split report
and a pre-15.27 report; the measuring recogniser sees the extra concepts), `test_readiness_retry.py` and
`test_phase2_guards.py` (the sign-test rule and its p-values), `test_mysigns.py` (an encoder change changes theta,
and the recogniser uses it) and `test_engine_recognizer.py` (segment mode reports the pack's threshold; two classes
with vocabulary index -1 commit as two words; the classifier thresholds are computed once per pair).

### 15.28 Protocol fixes review: false accepts in the app, a like-for-like E4 check, the pack's enrolment rule (review of 15.27, 27 Nov 2026)

**Problems found by the code review of 15.27.**
* The app's evaluation page showed only the E4 "auto" false-accept rate. Under a confirm-only policy that rate is 0
  by design, and every current pack is confirm-only. So the page would show 0.0 % and hide the research regression
  (0.0148). The new threshold-rule headline metric had no label and no interval. `E4.all_sources` was never shown.
* Section 15.27 said "INCLUDE-only false accept <= commercial-safe's", but nothing computed it.
* The recomputed theta used the default c_min 0.5. The app's enrolment uses the pack's `fusion.enrol["c_min"]`. With
  another c_min, a sign accepted at enrolment could get no theta later, or the other way round. E4 did the same.
* E4 kept a word as enrolled even when its takes failed the store's consistency rule. The app refuses that
  enrolment ("Your recordings look different"), so the word is never a user sign there.
* `choose_user_mode` ran the sign test pooled over all validation sources. On research, ISL500 is about 94 % of those
  takes, so neutral could win on ISL500 while INCLUDE got worse.
* For a pre-15.27 reference, class ids came from the `classifier.pt` next to the report, with no check that this
  checkpoint made the report. A reference with no split was accepted silently.
* `cli evaluate --compare-to` without `--all` was silently ignored.
* In spell mode the engine reported the fused accept threshold, but spelling never accepts on a threshold.

**What changed.**
1. **Evaluation page.** The E4 card has both false-accept columns next to each other, with counts and intervals:
   "shown without asking" (the app's own rate) and "a user sign would win" (the threshold rule). A note says why the
   first one is 0 under confirm-only. The INCLUDE-only table is labelled as the one to compare packs on.
   `all_sources` has its own table, labelled "not comparable across packs". If the INCLUDE-only entry did not run,
   the card still shows its reason and the all-sources table. The headline metric
   `user_sign_false_accept_rate_threshold_rule_k3` has a label (English and Hindi) and its interval.
2. **E4 like for like.** With `evaluate --all --compare-to <report.json>`, `E4.like_for_like` holds the reference's
   and this pack's threshold-rule false-accept rate (INCLUDE only, k = 3) and `pass` = this <= reference. A
   reference from before 15.27 has no threshold-rule rate; its `user_sign_false_accept_rate` is used and a note says
   so. report.md prints PASS or FAIL. It does not block a pack yet.
3. **The pack's enrolment rule.** `MySignsStore.user_theta(sign_id, embedder, c_min=..., theta_margin=...)`
   computes theta from the cached vectors with the given setting and caches it per (sign, c_min, theta_margin).
   The recogniser passes `fusion.enrol["c_min"]`. E4 uses the same c_min.
4. **Refused enrolments in E4.** For k >= 2, a word whose takes fail the consistency rule is not enrolled. All its
   takes are then tested as a word the signer did not enrol (false-accept rows). A signer with no accepted
   enrolment is skipped. Each k reports `enrolments` and `enrolment_refused`; E4 has `enrolment` and `enrol_c_min`.
   The clip-for-clip test now enrols with `MySignsStore.add`, so the app's refusals are mirrored, for k = 3 and 5.
5. **Sign test per source.** `my_signs.sign_test.by_source` has n, b, c and both one-sided p values for each source
   of the enrolled validation takes, and for INCLUDE against the rest. Neutral is now chosen only when (a) the
   pooled one-sided sign test gives p < 0.05, (b) the threshold-rule false-accept rate rises by at most 0.01, and
   (c) penalty is not better on INCLUDE takes alone (one-sided sign test of penalty over neutral, p < 0.05). E4 keeps
   the source of each enrolled take (`_with_src`, with `keep_hits`).
6. **Reference checks.** When the class ids come from the `classifier.pt` next to a reference report: a checkpoint
   trained on another split is not used (a note says so); a checkpoint newer than the report's `created` time
   (more than 60 s) gets a WARNING note. A reference report with no split gets a note that the split could not be
   verified.
7. **CLI.** `evaluate --compare-to` without `--all` exits with code 2 and says it needs `--all`.
8. **Engine threshold source.** Segment mode with a pack still reports the fused accept threshold. Spell mode keeps
   the value it had before 15.27 (`EngineConfig.threshold`, else the fused accept). `describe()` for a pack now has
   `threshold_source`: `pack_fused_accept`, `unused` (spell mode), `config`, `classifier` or `model`. The ASL
   `describe()` output is unchanged. The live page has no threshold control for packs, so nothing is greyed out.

**Backward compatibility.** Old reports and fusion.json files load as before. A fusion.json without `enrol` uses
c_min 0.5. A reference report without E4, or with E4 not run, gives `like_for_like.run = false` with a reason.
Reports before 15.27 render on the evaluation page with the old generic E4 table title.

**Effect.** E4 numbers can change slightly where enrolments are refused (those takes move to the false-accept rows).
Fit-fusion may keep `penalty` where 15.27 picked `neutral`, if INCLUDE gets worse. Re-run fit-fusion and evaluate
(QA 0.6) for the new numbers.

Tests: `test_eval_runners.py` (evaluate_all runs the INCLUDE-only E4 with repeated INCLUDE takes and compares with a
reference report; like-for-like with a pre-15.27 report; refused enrolments, also refused by `MySignsStore.add`;
real paired E4 hit lists feed an unpatched `choose_user_mode` with per-source counts; penalty stays when INCLUDE gets
worse; a rebuilt pack's checkpoint is flagged; `--compare-to` without `--all`; clip-for-clip for k = 3 and 5),
`test_mysigns.py` (a non-default c_min in the store and the recogniser), `test_engine_recognizer.py`
(`threshold_source` and the spell-mode value) and `LanguageEvaluation.test.tsx` (an auto rate of 0 next to a
threshold-rule rate above 0, the all-sources table, the like-for-like line, and an INCLUDE entry that did not run).

### 15.29 The app default: strict, sign by sign, a confirm tap before every word (QA plan "Recommended app default", critic 15, 27 Nov 2026)

**Problem.** Since 15.25 and 15.26 the backend sends `certification` and `needs_confirm` with every word, but the live
page did not read them. It added an accepted word to the sentence at once, and auto-speak could say it aloud. The
default policy was balanced. The certified floors (strict 0.95, balanced 0.90, lenient 0.85) are below the owner's
"without any fail" goal of 0.98, so a word should never be spoken or sent before the user taps it (critic 15).

**What the app does now** (`frontend/src/recognition/live.ts`, `useLiveRecognition.ts`, `components/ConfirmWord.tsx`,
`components/SignResult.tsx`, `pages/LivePage.tsx`, `i18n/strings.ts`):
1. **Default.** Sign by sign (segment mode), strict policy. Settings are saved under a new key (`live-settings-v2`).
   Old v1 settings are carried over, except the policy: v1 saved "balanced" for everyone, so the new default is
   strict. Continuous signing starts off.
2. **A tap before every word.** A commit with `needs_confirm: true` (or `open_vocabulary: true`) is not added to the
   sentence. It waits as a "pending" word. Only "Add" (or "Yes, add") puts it in the sentence, and only then can it
   be spoken or copied. "Wrong word" / "No" never adds it and sends `reject` for that sign (logged as feedback).
   The next sign that brings a word (an accepted word, or a "did you mean" with guesses) replaces a word that was
   never confirmed; a too-short movement, an abstain or "not sure" leaves it on show (15.30). A word the user picked
   from "did you mean"
   (`chosen_by_user`) has had its tap and is added at once. ASL commits carry no certification and are added as
   before.
3. **Two looks.** "Recognised – add it?" (teal, a tick) only when the word's policy is certified, the word is in the
   classifier, and, in continuous mode, sliding strict is certified. Everything else is "Did you mean this word?"
   (amber, a question mark) with "Yes, add X" / "No, sign again" and the other guesses as small buttons. That
   includes every word under a confirm-only policy (`suggest` / `confirm_only`) and every dictionary-only word
   (`open_vocabulary`). A doubtful sign (`ambiguous`, `low_confidence`) keeps the usual five-chip list.
4. **The certified precision, never the target.** Next to the word and under the policy setting the app shows the
   certified worst-source precision of the active policy, rounded down: "Strict: right about 95 of 100 times when it
   answers". A confirm-only policy says "not tested well enough yet, so every word is shown as 'Did you mean…?'". A
   dictionary-only word shows no precision, because the policy's number does not cover it.
5. **Policies.** The labels are Strict, Balanced and "Lenient – more mistakes" (Hindi: सख़्त, संतुलित, "ढीला –
   ज़्यादा गलतियाँ"). The hint says the user always taps to add a word.
6. **Continuous signing (sliding mode).** Off by default. The toggle appears only when the pack has a classifier.
   With it on, the app sends `mode: "sliding"`. Words that are not certified are guesses the user can confirm.
   A word is shown as recognised only when sliding strict is certified, and it still needs the tap.
7. **Accessibility.** The confirm card is a labelled region with real buttons (keyboard: Tab and Enter). It stays on
   screen while the next sign is in progress. The live region announces "Recognised, please confirm: water, पानी"
   or "Did you mean: water", with the Hindi word marked `lang="hi"`. All new strings have Hindi versions. The
   English and Hindi word output and the sentence builder are unchanged.

**Backend** (`inference/engine.py`, `isl/recognizer.py`):
* `describe()` of a language pack (so `hello` and `config_ack`) adds `certification_by_policy`: the certification of
  each policy for `segment` and `sliding` mode. `sliding` is null when the pack has no classifier. The ASL
  `describe()` is unchanged.
* Each `segment_result` and `commit` of a language pack with a certification now also says `open_vocabulary` (the
  word is outside the classifier; a "My signs" word never counts). Messages without a certification are unchanged.
* `IslRecognizer.sliding_rule(policy)` is "certified" only when that policy **and strict** are certified for the
  sliding replay on every major source. A certified balanced or lenient alone no longer lets sliding mode
  auto-commit. The certification then says why (`reason`). E1's sliding `rule` column follows the same rule.

**Old packs.** A fusion.json without a certification record gives confirm-only for every policy (15.25), so every word
is "Did you mean this word?" and the precision line says it is not tested well enough. Nothing crashes. Today both
shipped packs are in this state until they are refit (QA 0.6).

**This closes** the frontend open items of 15.25 and 15.26.

Tests: `backend/tests/isl/test_app_defaults.py` (sliding auto-commits only when sliding strict is certified;
`certification_by_policy` for segment and sliding; no sliding without a classifier; old packs confirm-only; the ASL
`describe()` is unchanged; `open_vocabulary` on results and commits), `frontend/src/recognition/live.test.ts`
(defaults, the v1 to v2 migration, the sliding payload, the precision rounding, the two looks),
`frontend/src/recognition/useLiveRecognition.test.tsx` (a fake socket: a pending word is added only after the tap;
"wrong" sends `reject` and adds nothing; the next sign replaces it; continuous commits; ASL unchanged),
`frontend/src/components/confirm.test.tsx` (both looks, the precision line, dictionary-only words, Hindi, keyboard
buttons, axe-core, announcements) and a new step in `frontend/e2e/a11y.spec.ts` (not run here: no browser).

### 15.30 App default review fixes: log only confirmed words, keep the waiting word, safer taps (review of 15.29, 27 Nov 2026)

**Problems** (code review of 15.29):
1. The server logged a word that waits for a tap as `accept` the moment it arrived. History and
   `/api/recognition/summary` counted words the user ignored, rejected, or replaced by another guess. If the user
   picked another guess, both words were logged.
2. Any new sign result cleared the waiting word, even an abstain or "not sure". A stray hand movement dropped the word
   without telling the user.
3. A "My signs" word showed the policy's certified precision. User signs follow their own per-sign rule (15.25), so
   that number does not cover them.
4. When the next sign replaced the waiting word, React kept the same buttons. A tap or Enter meant for the old word
   could add the new one.
5. In spelling mode the spelling panel hid the waiting word. Sliding mode was offered through state copied by an
   effect. An ungated pack said "Start recognition to see…" even while recognition ran.

**What changed:**
1. **Only confirmed words are logged** (`backend/app/api/recognition.py`). A pack commit with `needs_confirm` or
   `open_vocabulary` (and not `chosen_by_user`) is no longer logged when it arrives. The socket keeps the last 32 of
   them. A new client message `{"type": "confirm", "segment_id": N}` (or `{"type": "confirm", "t": ms}` for a
   continuous-mode word, which has no segment id) logs the word with `decision: "confirmed_by_user"` and
   `needs_confirm: true`, and answers `confirm_ack` (`confirmed`: the concept id, or null for a second tap or a word
   the server no longer holds). A word that is rejected, replaced by another guess (`choose`), reset or left alone
   is never logged as a word. A rejected word still gives its `rejected_by_user` feedback event. Offline video
   analysis has no tap: its pack commits are logged as before, now with `needs_confirm` in the detail. ASL commits
   are logged exactly as before.
2. **The waiting word stays** (`live.ts` `bringsWord`, `useLiveRecognition.ts`). Only a result that brings a word
   replaces it: `accept`, or `suggest` with at least one guess. A too-short movement, an abstain or "not sure" keeps
   it on show. The app now sends `confirm` when the user taps "Add" (not in the static demo).
3. **User signs get their own line** (`SignResult.tsx`). A waiting word or a confirm-only suggestion whose word comes
   from "My signs" shows "Your own sign: the tested figure does not cover it, so check it before you add it." instead
   of the policy's precision. Dictionary-only words still show no number.
4. **Safer taps** (`ConfirmWord.tsx`, `SignResult.tsx`). The confirm card is keyed on the word (segment id and time),
   so a new word gets new buttons. After a replacement, taps are ignored for 350 ms. If focus was inside the old card,
   it moves to the new card itself (not to "Add"), so the new word is read first. The first word of a series is not
   delayed.
5. **Smaller fixes.** While spelling, the waiting word is shown above the spelling panel. Sliding mode is offered
   when the server's certification has sliding rows: the hook and the page derive this, with no copied state, and
   the hook sends a `config` message only when the engine settings really change. An ungated pack (developer
   parameters) says "{policy}: not calibrated (developer settings), so there is no tested figure".

Tests: `backend/tests/isl/test_app_defaults.py` (a "My signs" top word is never dictionary-only; the store of waiting
words; `needs_confirm` in the logged detail; over the WebSocket: logged only after `confirm`, once, never after
`reject` or `choose`, bad `confirm` messages are errors), `frontend/src/recognition/useLiveRecognition.test.tsx`
(abstain and empty "not sure" keep the waiting word, guesses or a new word replace it; `confirm` by segment id or by
time; sliding from the hello), `frontend/src/components/confirm.test.tsx` (the user-sign line, the replacement guard
and focus, no delay for the first word, the card while spelling) and the new `frontend/src/pages/LivePage.test.tsx`
(a fake socket hello: the strict precision in the settings and on the model card, confirm-only and ungated lines, the
continuous toggle shown only with sliding rows and sending `mode: "sliding"`). The e2e step of 15.29 in
`frontend/e2e/a11y.spec.ts` now ran (system Chrome, `scripts/e2e_env.sh`) and passed.

### 15.31 Batch 0 final check: the driver can compare like for like (QA plan 0.6, 27 Nov 2026)

**Problem.** E1b and E4 like for like (15.27, 15.28) need `evaluate --all --compare-to <report.json>`. The driver's
evaluate step never passed it. So the QA 0.6 refresh of research, and the fallback run, would have written reports
without the rows that compare them with commercial-safe (critic point 4: user-sign false accepts no worse than
commercial-safe). Getting them would have meant a second evaluate run (about 1.3 h).

**Fix** (`scripts/phase2_train.sh`). The environment variable `EVAL_COMPARE_TO=<report.json>` makes the evaluate step
run with `--compare-to <absolute path>`. The driver refuses a missing file before anything runs (exit 2). It also
refuses `--profile both`: commercial-safe would compare with its own report, which its evaluate then replaces. Without
the variable nothing changes. The dry run and the start line of the log name the report.

**QA 0.6 commands** (no training; each one from the repo root, one at a time):
1. Commercial-safe first, so its report has clip ids and class ids:
   `scripts/phase2_train.sh --profile commercial-safe --from index --threads 4`
   (index train-only, fusion, trim_eval, evaluate, index_final, pack).
2. Research, compared with the new commercial-safe report:
   `EVAL_COMPARE_TO=artifacts/isl/commercial-safe/eval/report.json scripts/phase2_train.sh --profile research --from index --threads 4`
3. Fallback (research classifier, commercial-safe encoder) in a scratch run. Its stamps, log and pack stay under
   `artifacts/isl/runs/fallback-csenc/research/` and never touch `data/isl/logs` or `artifacts/packs`. Put the
   encoder in first; seeding never overwrites a file that is already there:
   `mkdir -p artifacts/isl/runs/fallback-csenc/research && cp -p artifacts/isl/commercial-safe/encoder.pt artifacts/isl/runs/fallback-csenc/research/`
   then
   `EVAL_COMPARE_TO=artifacts/isl/commercial-safe/eval/report.json scripts/phase2_train.sh --profile research --tag fallback-csenc --from index --threads 4`.
   The other inputs (vocabulary, classifier, cv, oneshot, letters, link outputs) are seeded from
   `artifacts/isl/research`. E2 one-shot still uses the research one-shot encoders. Its pack is for measuring only:
   the app refuses a pack whose encoder records another data profile (`ProfileMismatch`, ADR-ISL-6).

Use `--dry-run` first to see what each command removes, seeds and runs.

Test: `backend/tests/isl/test_phase2_guards.py::test_evaluate_step_passes_eval_compare_to`.

### 15.32 WSLP is a second view of the ISLRTC dictionary, not a signer (QA plan 1.1, 27 Nov 2026)

**Finding.** WSLP's labelled word clips are the ISLRTC dictionary videos again, re-encoded. A WSLP clip and its
ISLRTC clip have the same timing and lie a median 0.062 shoulder widths apart (retakes by one signer: 0.163).
The dedup step does not catch them (`data/dedup.py:83`). WSLP clips have no signer id, so each one counted as one
more "signer". CISLR is the collection WSLP stands in for and is treated the same way.

**Why it matters (critic 2).** The fix is justified by the class space and the encoder, not by E1b. Masking the
research classifier to commercial-safe's classes moves E1b top-1 only from 0.592 to 0.606; most research errors are
confusions inside the shared classes. E1b gains are expected from 1.3 and 1.4, not from this item.
* **Class space.** A concept with one ISLRTC row and one WSLP twin had "2 signers" and became a classifier class.
  That is about 3,770 of research's 4,354 classes: words the classifier saw from one signer only, in two views.
  With one dictionary group the verifier measured 581 classes. The dictionary-only words are served by retrieval
  (confirm-only under 15.25).
* **Encoder.** Each WSLP twin made an "INCLUDE <-> ISLRTC <-> WSLP" positive pair that is really the same video. The
  pair is almost free for the loss and teaches no signer invariance, and it takes P x K slots away from real
  cross-signer pairs. The research encoder never learned INCLUDE <-> ISLRTC (true-pair cosine 0.368, the same as the
  pretrained encoder).
* **Index.** 4,274 of the 4,359 WSLP rows repeat an ISLRTC row of the same concept. They add hub rows and nothing new.

**What changed.** Every change is opt-in. With no new flag, training, `build-index`, fit-fusion, evaluate and
`build-pack` behave as in Batch 0.
* `training/common.py`: `DICTIONARY_SOURCES = ("islrtc", "wslp", "cislr")`. `multi_signer_counts(clips,
  dictionary_group=True)` counts all their clips as the one group `dictionary`. Default False: the old grouping.
* Classifier: `classifier_classes(..., dictionary_group=)`, `train_classifier(..., dictionary_group=)` and
  `train-classifier --dictionary-group`. WSLP clips still train the classes that remain.
* Encoder: `encoder_training_clips(..., exclude_sources=)`, `train_encoder(..., exclude_sources=)` and
  `--exclude-sources wslp` on `train-encoder` and `train-oneshot`. Excluded sources leave the training list, the
  validation queries and the validation gallery.
* Index: `build-index --wslp-twins keep|drop|auto` (default `keep`). `drop` leaves out every WSLP clip whose concept
  has an ISLRTC row in the index. `auto` drops them only when `encoder.pt` was trained without WSLP, so the index of an
  older encoder is built exactly as before. WSLP-only words keep their rows. `index/meta.json` has a new field
  `row_filters` (`{}` when nothing was left out, else `{"wslp_twins": {"dropped": n, ...}}`). The training-clip
  digest of 15.22 still covers every clip given to the build, so the leak guard compares like with like.

**The class space is explicit in meta.** Old and new checkpoints are never mixed silently.
* `classifier.pt` meta `class_space`: `signer_grouping` (`source` or `dictionary`), `min_signers`, `min_clips`,
  `use_extra`, `n_classes`, `classes_sha256`. `encoder.pt` meta `class_space`: `excluded_sources`, `use_extra`,
  `n_concepts`, `concepts_sha256`. A checkpoint without the field reads as `source` / no excluded source
  (`common.class_space_of`).
* Resume signature (15.19): `dictionary_group` and `exclude_sources` go into `LoopConfig.extra` when they are set.
  A state of the default settings written before this change still resumes.
* `evaluate --all`: a cv fold whose classifier used another signer grouping than the main classifier is not pooled
  into the CV number. It is listed in `report.json` `leakage_guards.class_space.cv_folds_not_pooled` and in
  `report.md`. E2 rebuilds each fold encoder's training list with that encoder's own excluded sources; a fold
  trained with other sources than the main encoder is listed in `oneshot_folds_other_sources`.
* `build-pack`: `pack.json` `components.classifier.class_space` and `components.encoder.class_space` when the
  checkpoints record one, and `components.index.row_filters` when rows were left out.

**Measured on the real research data** (read-only, current split): classifier classes 4,354 (old grouping) -> 581
(dictionary group); index rows of WSLP 4,359 -> 85 (4,274 twins dropped); encoder training list 80,221 -> 75,862 clips
and 12,439 -> 12,355 concepts (the 84 concepts only WSLP knows leave encoder supervision but keep their index rows).

**Driver** (`scripts/phase2_train.sh`). New environment variables, research defaults on:
* `CLS_DICTIONARY_GROUP=1|0`: `--dictionary-group` for classifier, cv and e1a. Research 1, commercial-safe 0.
* `ENC_EXCLUDE_SOURCES=wslp`: `--exclude-sources` for encoder and oneshot. Research `wslp`, commercial-safe empty.
* `INDEX_WSLP_TWINS=keep|drop|auto`: for index and index_final. Research `auto`, commercial-safe `keep`.
Commercial-safe has no WSLP or CISLR, so it passes no new flag and nothing changes there. The dry run prints the three
settings. The QA 0.6 research refresh (`--from index`) runs no training step, and with the current research encoder
`auto` keeps every row, so its numbers are unchanged. The Batch 1 full run (`--from link`) gets all three.

Tests: `backend/tests/isl/test_wslp_dictionary.py`. One WSLP clip plus one ISLRTC row give no class (ISL500 as a
second signer still does); the grouping changes nothing without dictionary views; the encoder list drops WSLP only
when asked; a toy encoder trained without WSLP records it, and its `auto` index has no WSLP twin rows, keeps the
WSLP-only word and keeps the clip digest; `keep` indexes every WSLP clip; the classifier meta records the grouping
and a dictionary-group state is refused without it; the driver passes the flags for research only and refuses a bad
value.

### 15.33 WSLP / dictionary review fixes: class spaces visible everywhere (review of 15.32, 27 Nov 2026)

The review of 15.32 found no blocker. These fixes make a class-space change hard to miss and hard to mix.

* **CV number.** When `evaluate --all` leaves a cv fold out (another signer grouping), `E1b.cv.not_pooled` lists it
  next to the CV number, and `report.md` prints a "NOT pooled" line under it. When every fold is left out,
  `E1b.cv` is `{"run": false, "reason": "cv folds trained with another signer grouping", "not_pooled": [...]}`
  instead of being missing. Anyone reading the Batch 1 gate ("E1b and CV pooled >= 0.72") sees the dropped folds.
* **Source names are checked.** `--exclude-sources` takes only word sources of a profile (`isl500`, `wslp`, `cislr`,
  `mendeley_emergency`, `mendeley_agri`). A typo such as `wlsp` is refused (argparse, exit 2), so WSLP is never
  trained in by mistake. `include` and `islrtc` are refused too: the encoder's validation queries and gallery are
  built from them. `train_encoder` checks the same rule (`check_exclude_sources`). The driver refuses `include` and
  `islrtc` in `ENC_EXCLUDE_SOURCES` before anything runs; an unknown name fails at the encoder step.
* **Effective grouping.** `--dictionary-group` without `--all-sources` (the driver's E1a) changes no class, so the
  classifier records `signer_grouping: "source"` and the resume signature gets no `dictionary_group` key. The
  driver still passes the flag to e1a, so an E1a run with other sources would use the same grouping.
* **fusion.json says what it was fitted on.** `fit-fusion` writes `fitted_on`: the classifier's signer grouping,
  class count and class-name digest, the encoder's left-out sources, and the kinds of index rows left out (only the
  kinds: `index` and `index_final` drop the same kind of rows from different clip counts). `evaluate --all` flags a
  fusion fitted on another class space (`leakage_guards.class_space.fusion_mismatch`, a bold line in `report.md`).
  `build-pack` refuses it. A fusion.json written before this has no `fitted_on` and passes as before, so the running
  0.6 refresh is not affected. This catches `--only classifier` followed by `--only evaluate` without fit-fusion.
* **Run log.** The driver's start line ends with `dict_group=… enc_exclude=… wslp_twins=…`. Each step's stamp
  records the setting it ran with (classifier / cv / e1a: `dict_group`, encoder / oneshot: `enc_exclude`, index /
  index_final: `wslp_twins`), so the "skip … (done …)" lines of a later partial run show it.
* `report.md` lists excluded sources as plain words ("trained without wslp"), not as a Python list.

Tests: `test_wslp_dictionary.py` (unknown and core sources refused in code and on the command line; with default
arguments `LoopConfig.extra` has no new key, so Batch 0 states resume; an encoder state without the exclusion is not
resumed with it; E1a records the effective grouping; `fusion_fit_space` and its mismatch; the driver passes
`--dictionary-group` to cv and e1a and writes the settings into the start line and the stamps) and
`test_eval_runners.py` (a cv fold of another grouping is not pooled and is named in `E1b.cv`, `report.json` and
`report.md`; with every fold left out `E1b.cv` says why; a one-shot fold encoder trained without WSLP is listed; a
fusion fitted on another classifier is flagged by evaluate and refused by build-pack).

### 15.34 Linker numbering and alias groups (QA plan 1.2, critic 12, 27 Nov 2026)

**Finding.** WSLP numbers repeated words: "Adult(1)". The linker kept the "(1)", so 26 WSLP words became new
concepts (`isl.adult-1`, `isl.bank-1`, ...) next to the real ones. Some twins are also kept apart only because their
source names differ: `isl.female` and `isl.woman-female` (both hold an INCLUDE class: "female" and "Woman"; see
15.35), `isl.big` and `isl.large`, `isl.mouse` and `isl.rat-mouse`. (`isl.race~ethnicity` and `isl.race` were in the
first table; 15.35 took them out.) A model that picks the twin is not wrong for the user, and
the fused probability of the word is split over two keys. The plan expects a small gain (+0.35 pp). The groups are
also what 1.3 (variant forms) builds on.

**What changed.** Everything is opt-in. With no new flag and no new file, linking, training, fit-fusion, evaluate and
`build-pack` behave as in Batch 0.
* `app/isl/vocab/aliases.py` (new): `strip_sign_number` removes a trailing "(digits)" only ("(academic)" and a bare
  "(1)" stay). `TWIN_ALIASES` maps alias -> root; the root is the INCLUDE concept, because INCLUDE labels E1b.
  `NEVER_ALIAS` keeps `isl.you` and `isl.you~plural` apart (a real sense difference); `resolve_aliases` refuses it,
  follows chains to the final root, refuses cycles, and keeps only pairs whose two ids are known.
* `GlossLinker(strip_numbering=, aliases=)`. With `strip_numbering`, "adult(1)" links to `isl.adult` (exact). A word
  the vocabulary does not know gets one new concept for all its numbers (`zebra(1)`, `zebra(2)` -> `isl.zebra`). The
  clip's `variant_id` keeps the number (`wslp:adult1`). `alias_of` gives the groups of known concepts. `expand` (the
  held-out concept guard) adds the whole alias group.
* **One switch, at the link step.** `link-sources --link-aliases` turns on both. The link step always writes
  `<work>/link_settings.json` (on or off) with the resolved `alias_of` and its sha256. Every later step of that work
  folder (`_model_ctx`: training, index, fit-fusion, evaluate) reads the file and links the same way. No file (a work
  folder linked before this change): the old linking. So the running 0.6 refresh, which starts at `index`, is not
  affected. Relinked clips change the training labels; the training-list hash of 15.19 holds the labels, so an old
  training state is refused, not resumed.
* `FusionParams.aliases` (alias -> root, stored in fusion.json). When set, `fuse` sums the per-crop fused
  probabilities of each group before the accept rule. The word shown is the group member with the highest
  probability; `FusionResult.roots` records the root. Crop agreement is counted on the summed groups. User signs are
  never in a group. A fusion.json without the field loads with `{}`: strict keys, as before. The app gets this from
  the pack's fusion.json; `decide()` itself is unchanged. Sliding mode (classifier alone) stays per class.
* `fit_fusion(..., aliases=)`. `cli fit-fusion` passes the link step's map (none without the file). The weights are
  fitted on strict keys as before. Then every validation clip is fused once more with the groups summed, and the
  thresholds and the certification are fitted on that, with **strict** labels: the shown word must be the clip's own
  concept id. So the certified precision describes what the app does, and an alias does not make a word "correct".
  `rep["aliases"]` lists the groups.

**Evaluation (critic 12).** Strict labels stay the metrics, the headlines and every gate. Alias numbers are reported
next to them, never instead.
* E1b and E1c: `alias_labels` with top-1 / top-5 after summing the classifier's probabilities over each group, the
  strict numbers of the same clips, and the clips that change (`top1_only_alias`, `top1_only_strict`).
* E3: `alias_labels` with top-1 / top-5 and, per policy, accepted precision and wrong-accept rate on strict and on
  alias labels. The decisions are the app's (with the fusion's groups when it has them).
* `report.json` `aliases`: which map was used (the link step's, else fusion.json's), whether fusion sums the groups,
  and `mismatch` when fusion.json was fitted with another map than the link step's (run fit-fusion again).
* `report.md`: an "Alias labels ... NOT a gate" block under the E1b rows. E1b like for like stays strict.

**Driver.** `LINK_ALIASES=1|0`: research 1, commercial-safe 0. Only the link step passes `--link-aliases`. The value is
in the start line (`link_aliases=`), in the link stamp and in the dry run. A scratch run seeds `link_settings.json`
with the other link outputs. A bad value is refused (exit 2). The Batch 1 run (`--from link`) gets it.

**Measured on the real research vocabulary** (read-only): all 26 numbered WSLP glosses link exactly to an existing
concept with stripping. All four twin pairs exist in both the research and the commercial-safe vocabularies.

Tests: `backend/tests/isl/test_linker_aliases.py`. `link('adult(1)', 'wslp')` gives `isl.adult` only with
stripping; summing female 0.45 + woman-female 0.40 turns "not accepted" into "accept" (and can change the winner);
crop votes use the groups; you / you~plural are refused; an old fusion.json has no aliases; the app's
`user_sign_decision` accepts the summed group; E1 / E3 alias rates count a twin as correct and keep the strict
number; fit-fusion keeps the strict weights and fits thresholds on the groups; `evaluate --all` keeps strict metrics,
adds alias rows and flags a fusion fitted with another map; `link-sources` writes the settings and a later step reads
them; the driver passes `--link-aliases` for research only.

### 15.35 Alias review fixes: evidence for every pair, a frozen map, one gate per group (review of 15.34, 27 Nov 2026)

**Race is not a twin.** ISLRTC has three "race" rows: Race, Race (Competition) and Race (Social Science). We checked
which one INCLUDE's "Race (ethnicity)" matches, read-only, with the encoders and indexes of 27 Nov (train clips only:
the INCLUDE centroid against the 13,461 ISLRTC dictionary rows). Research / commercial-safe ranks: Race 4813 / 781,
Race (Competition) 766 / 1510, Race (Social Science) 10981 / 2469. None is close, so no race pair is aliased (the
social-science row is not either). `ALIAS_VERSION` is now `isl-alias-2`.

**Evidence next to every pair.** Each pair in `TWIN_ALIASES` has a comment with its numbers:
* `isl.woman-female` -> `isl.female`: INCLUDE "Woman" ranks the Woman/Female row 62 / 2; ISL500 "Female" ranks it 1.
* `isl.large` -> `isl.big`: INCLUDE "big / large" ranks Large 2 / 14 and Big 6 / 1; ISL500 "Large" ranks Large 2,
  Big 5.
* `isl.rat-mouse` -> `isl.mouse`: INCLUDE "Mouse" ranks Rat/Mouse 35 / 5; ISL500 "Mouse" ranks ISLRTC Mouse 1. These
  are two forms of the one word "mouse", both used for the animal.

**Woman and female are two INCLUDE classes.** 15.34 said the group joins INCLUDE's `isl.female` and ISLRTC's
`isl.woman-female`. In fact `isl.woman-female` also holds INCLUDE's own "People/65. Woman" class (20 clips, in the
classifier). So this group joins two INCLUDE classifier classes, and the word shown can be "woman / female" or
"female". The alias numbers of E1b therefore count woman <-> female confusions as right. That is allowed (critic 12:
reported, never a gate). E1b / E1c `alias_labels` now have `by_group`: per group, how many clips are right only on
alias labels, so a reader sees how much of the gain comes from woman / female. report.md prints them.

**Later steps use the frozen map.** `linker_options` takes `alias_of` from `<work>/link_settings.json` when the file
has it. Only the link step itself reads `TWIN_ALIASES`. So when the table in the code changes (plan 1.3), every step
of a work folder linked earlier still uses the same groups: the held-out guard and one-shot exclusions (`expand`),
fit-fusion and evaluate.

**Numbering only for WSLP.** `strip_sign_number` now runs only for sources in `STRIP_NUMBERING_SOURCES` (`wslp`).
Today only WSLP has "(N)" glosses, so nothing changes. A future source with meaningful numbers is not merged silently.
The count is also kept per source (`numbering_stripped:wslp`).

**One open-vocabulary rule per group.** With aliases, the fused decision shows the group's best member. That member
can be retrieval only (`isl.large`, `isl.rat-mouse`) while the summed mass holds the root's classifier probability.
Before, such a group got the confirm-only open-vocabulary gate only when that member led. Now
`aliases.closed_with_aliases` adds every member of a group that has a class to the closed set. fit-fusion, E3 and the
app (`IslRecognizer.is_open_vocabulary`, so also the engine) use it. The group is a classifier word whichever member
leads. We kept "show the best member" (not "always show the root"): showing the root would make every INCLUDE "Woman"
clip strictly wrong.

**Reports.** `report.json` `aliases` has `stale` (the link step turned aliases on, but fusion.json has none: run
fit-fusion again) and the link step's `link_alias_version`. E3 `alias_labels` has `top5_strict`. report.md prints
strict and alias top-5 and the stale warning.

**Driver.** When the link step is not in this run, the start line shows the folder's real state, for example
`link_aliases=1 (link step not run; folder: none)`. `none` / `off` / `on` come from `<work>/link_settings.json` (of
the seed folder when link is seeded). A warning is logged when `LINK_ALIASES` and the folder disagree. The dry run
prints the same. The script was replaced by a new file (not edited in place), so the running driver keeps its copy.

Defaults are unchanged: with no `--link-aliases` and no link_settings.json, linking, fit-fusion, evaluate and the app
behave as in Batch 0. Tests: `backend/tests/isl/test_linker_aliases.py` (the race pair is gone and every pair has a
comment; WSLP-only stripping; a settings file whose map differs from the code's; a group with a class is closed;
`fit_thresholds` sees the summed groups, roots, strict truths and the group-closed set; per-group counts, top5_strict,
stale; the driver's start line).

### 15.36 Variant forms instead of forced merges: the variant audit (QA plan 1.3, critics 7 and 13, 27 Nov 2026)

*15.37 changed parts of this section: the dictionary rule, the file names, the split check, the gates and the driver's seeding. Read 15.37 with it.*

**Finding.** ISL500 and INCLUDE sometimes sign one word in two different ways. The QA review found 40 of 160 linked
classes with cross-dataset agreement below 0.3 (for example short, tight~clothes, hot, child, pink, lock). The linker
gives both forms one concept id. So the encoder is pulled to put two different signs together, and the classifier has
one class for two forms.

**What changed.** Everything is opt-in. With no audit file, linking, training, fit-fusion, evaluate and `build-pack`
behave as before.
* `app/isl/training/variant_audit.py` (new) and `cli variant-audit` (research only; refused for commercial-safe,
  which has no ISL500, exit 2). It reads the research data with the base linking (no twins, no variants) and scores
  **ISL500 train clips only** (no validation or test clip) with the reference models of `--reference-work`
  (default: the commercial-safe work folder). The reference classifier must come from a profile that never trains on
  ISL500 (checked from its meta, else exit 2).
* **Step 1, classifier agreement.** For each ISL500 word the reference classifier knows (or a twin of it), the share
  of train clips whose top 5 holds the word or a twin. Words with at least `--min-clips` (20) clips and agreement
  below `--max-agreement` (0.3) are candidates.
* **Step 2, dictionary cross-check (critic 13).** Low agreement can be signer shift: new signers, same sign. Moving
  those clips away would take supervision from INCLUDE. So, with the commercial-safe encoder, the centroid of the
  word's ISL500 train clips and the centroid of its INCLUDE train clips both retrieve against the ISLRTC rows of the
  reference index (concept-level rank: each concept scored by its best row). The verdict:
  * `different_form`: INCLUDE finds the word's dictionary row within rank 20 (`--dict-include-max-rank`), and ISL500
    finds it only below max(10, 3 x INCLUDE's rank) (`--dict-rank-floor`, `--dict-rank-ratio`);
  * `no_difference`: ISL500 finds it about as well (signer shift);
  * `inconclusive`: no dictionary row, no INCLUDE train clip, or INCLUDE does not find the row either.
* **Step 3, reverse direction (optional, `--reverse-classifier`).** A classifier trained on ISL500 scores INCLUDE
  clips of the word (`--reverse-part val` by default, never test; `train` is refused for a classifier trained on that
  split). Agreement of 0.3 or more means the forms are compatible. Every classifier we have also trains on INCLUDE,
  so this veto is weak evidence; the file says so.
* **Decision.** Relink only on evidence of another form: the dictionary says `different_form` and the reverse check
  (when run) does not say `no_difference`; or the dictionary is inconclusive and the reverse check says
  `different_form`. Without a dictionary and without a reverse check, nothing is relinked.
* **Evidence per word** in the file: clip and signer counts, top-1 and top-5 agreement, what the clips were predicted
  as, both dictionary ranks, the top 3 dictionary concepts of each centroid, the centroid cosine, the reverse
  agreement, the decision and its reason. The summary also counts how often the dictionary rule says "another form"
  on words the classifier agrees on: a high rate means the cross-check cannot tell forms apart with that encoder.
* **The relink.** Every ISL500 clip of a relinked word (train, val and test, so ISL500's own labels stay consistent)
  gets `isl.<word>~isl500`. That concept is an alias of `isl.<word>` (docs 15.34): the fused decision sums the group,
  so the word shown is the same. Its row in `extra_concepts.jsonl` copies the word's headword and display and has
  `alias_of`; `build-pack` gives it the word's Hindi row. INCLUDE, ISLRTC and the other sources keep the word. A
  variant id that already is a vocabulary concept is never used.

**Scope (critic 7).** The audit is written to `<work>/variant_links.json` of the profile's work folder, or `--out`.
A path inside `data/isl/derived` is refused. `link-sources --variant-links auto|none|FILE`: `auto` (default) reads
`<work>/variant_links.json` only when it exists in that work folder; a file of another profile is refused (exit 2).
The link step freezes the relinks into `link_settings.json` (`variants`, and `variant_links` with the file name and
its SHA-256), adds the variant pairs to the frozen `alias_of`, and counts the moved clips (`variant_relinked`). Later
steps read only that frozen copy, so a re-audit changes nothing until the link step runs again. A scratch run that
runs its own link step has no audit file and links as before. Relinked clips change the training labels, so the
training-list hash of 15.19 refuses an old state.

**Evaluation.** Strict labels stay the metrics and the gates (critic 12). An INCLUDE "hot" clip predicted as
`isl.hot~isl500` is wrong on strict labels and right on alias labels; E1b `alias_labels.by_group` shows how many clips
each variant group moves. The one-shot guard (`expand`) holds out a word's variant with the word.

**Driver.** `VARIANT_LINKS=auto|none` (default `auto` for both profiles): `none` passes `--variant-links none` to the
link step. The start line, the link stamp and the dry run show `variant_links=` (`none`, `off`, `sha256:<16 hex>`, or
the folder's frozen state when the link step does not run). `variant_links.json` is a link output for seeding. The
audit itself is not a driver step: run it before the Batch 1 run, for example
`python -m app.isl.cli variant-audit --profile research --reference-work ../artifacts/isl/commercial-safe --threads 1`,
read the evidence, then run `scripts/phase2_train.sh --profile research --from link`. No audit has been run on the
real data yet, so no work folder has the file and the running chain is not affected.

Tests: `backend/tests/isl/test_variant_audit.py`. A fake classifier gives the expected relink ("hot", another form)
with its evidence; a word with low agreement whose ISL500 centroid still finds its dictionary row (signer shift) is
not relinked; without a dictionary nothing is relinked; the reverse direction can veto; the dictionary rule's
thresholds; the file is checked for profile and form and hashed; the linker relinks only that source and word, writes
the variant row, freezes the settings, and a later step links the same way; alias-scored E1b counts a variant
prediction as correct and strict does not; `link-sources` reads only this work folder's file, keeps the frozen copy
after the file changes, ignores it with `none`, and refuses another profile's file; `variant-audit` end to end with
tiny commercial-safe models scores ISL500 train clips only and refuses a commercial-safe run, a reference trained on
ISL500 and an output in `data/isl/derived`; the pack gives a variant the word's Hindi; the driver's setting.

**First look at the real data** (read-only, 27 Nov 2026, output in a scratch folder, not in any work folder; 20 train
clips per word, 1 thread, 99 s). 191 of the 499 ISL500 words have a commercial-safe class. Top-5 agreement below 0.3:
13 words; 8 relinked by the dictionary rule (clean-tidy, friend, gun, market, mean, monday, paint, she), 2 kept as
signer shift (doctor, short), 3 kept as inconclusive (i, light, thief: no dictionary row). The QA count of "40 of 160"
matches **top-1** agreement (37 of 191 below 0.3 here); `--top-k 1` gives that reading. The plan's examples mostly
agree at top 5 (hot 0.50, lock 0.85, pink 0.45, child 0.35). The dictionary rule says "another form" for 36 of the 178
words the classifier agrees on (20 %), so on its own it is noisy; it only ever narrows the classifier's candidates.
Read the evidence before linking with the file. `--top-k` was added for this choice (default 5, as planned).

### 15.37 Variant audit review fixes: a calibrated dictionary rule, a proposal file, word labels (review of 15.36, 27 Nov 2026)

**Dictionary rule, calibrated on signer shift.** The rule of 15.36 compared the ISL500 rank with the INCLUDE rank.
The commercial-safe encoder was trained on those INCLUDE train clips, so the INCLUDE rank is in-sample: on the real
sample its 25th, 50th and 75th percentiles were all 1. The rule then reduced to "ISL500 rank above 10", and it fired on
20 % of the words the classifier agrees on. Now the yardstick is the ISL500 rank of the words the classifier agrees
on (same form, new signers, same encoder). A low-agreement word is `different_form` only when its ISL500 rank is
above the 95th percentile of that spread (`--dict-calib-percentile`, and above `--dict-rank-floor`). The INCLUDE rank
is only a sanity check that the row exists and is reachable (`--dict-include-max-rank`). `--dict-rank-ratio` is gone.
* The summary has `dictionary_calibration`: the number of agreeing words, the threshold, the 50/80/90/95th
  percentiles, and the false-alarm rate on agreeing words (leave-one-out: each word against the threshold of the
  others).
* `dictionary_rule_usable` is false with fewer than 20 agreeing words (`--dict-calib-min-words`; then every verdict
  is inconclusive) or a false-alarm rate above 0.10 (`--dict-max-false-alarm`). `link-sources` refuses such a file
  with relinks (exit 2) unless `--accept-noisy`; that is recorded (`accepted_noisy`).
* The audit version is now `isl-variant-audit-2`. The link step refuses a version-1 file.
* The INCLUDE centroid is embedded with the encoder only (no classifier pass).

**Real data, read-only** (27 Nov 2026, 20 train clips per word, 1 thread, 85 s per run, output in a scratch folder).
Top-5: 13 low-agreement words, 4 relinked (mean 278, monday 211, paint 1502, she 731). Threshold 130 (178 agreeing
words, p80 12, p90 41), false alarms 9 of 178 (0.051), usable. gun 22, market 27, friend 47 and clean-tidy 101 are
now kept as signer shift. Top-1: 37 low-agreement words, 8 relinked (clean-tidy, mean, monday, paint, she, spring,
tight~clothes, ugly). Threshold 90 (154 agreeing words), false alarms 0.058, usable.
**Open: the orchestrator picks `--top-k` for the Batch 1 run and records it here.** The plan's count (40 of 160) and
its examples fit top-1; with the calibrated rule, `--top-k 1` is a defensible reading.

**A proposal, not an accepted file.** `variant-audit` now writes `<work>/variant_audit.json`. The link step never
reads it. A person reads the evidence and approves it by copying it to `<work>/variant_links.json`, or names it with
`link-sources --variant-links FILE`. `auto` reads only `variant_links.json`. A file whose `work` is another folder
(for example a copy) is ignored in `auto` mode with a log line; a file named explicitly is allowed. `link-sources`
prints the relink count, the false-alarm rate and the evidence scope.

**Split check.** The audit records `isl500_split`: the seed, the split parameters (`multisource.SIGNER_SPLIT`) and
the sorted ISL500 train signer groups with their SHA-256. `link-sources` recomputes the current ISL500 train groups.
When an audited group is no longer train (for example QA plan 1.6 moves ISL500 signers to val), it refuses the file
(exit 2). `--allow-split-change` links anyway and records `split_check.override` and the moved groups. A file without
the record is refused too. Re-run the audit after any split change.

**Evidence scope.** The audit records `parts_used` (ISL500 `train`; INCLUDE `train` for the dictionary, plus the
reverse check's part, `val` by default). `link_settings.json` keeps it in `variant_links`. The evaluate report
(`aliases.variant_links`, and a "Variant links" paragraph in report.md) and the pack manifest (`variant_links`) carry
the file, its SHA-256, the relinks, the moved clip count and `parts_used`. The report warns when validation clips
were evidence, when a noisy rule was accepted, and when the split check was overridden.

**Word labels; the gates.** Strict labels count an INCLUDE clip predicted as `isl.<word>~isl500` as wrong, and an
ISL500 test clip predicted as `isl.<word>` as wrong too, although the app shows the same word. That biases gates (a)
and (d) against a variant run. Word labels collapse only the variant concepts to their word (from the frozen
`variants`); twins such as female / woman-female stay strict (critic 12). E1b, E1c and E3 report `word_labels` next
to strict and alias labels, and E1b like-for-like has `common_clips_word_labels`. In a run with variant links, **gate
(a) uses the word-label common-clip number (or the masked `reference_classes_common_clips` number, which masks the
variant column), and gate (d) uses the ISL500 row of `word_labels.top1_by_source` in E1c (classifier) and E3
(fused)**. Without variants word labels equal
strict labels and nothing new is written.

**Driver.** `variant_links.json` is no longer a link output for seeding, so a scratch run never gets the real
folder's audit; a scratch `--from link` links without variants. `--only link` is refused (exit 2) when the file's
state differs from the variant state frozen in the folder's `link_settings.json`: the later steps would keep stamps
on models trained with the old labels. Use `--from link`. `VARIANT_ACCEPT_NOISY=1` and `VARIANT_ALLOW_SPLIT_CHANGE=1`
pass the two overrides to the link step.

Tests (`backend/tests/isl/test_variant_audit.py`): the calibrated rule keeps a low-agreement word whose ISL500 rank
is inside the agreeing words' spread although its in-sample INCLUDE rank is 1, and relinks one far above it; a noisy
or uncalibrated rule; refusal of a noisy, old-version or unchecked-split file; a split change between audit and link
is refused and the override is recorded; `auto` ignores a file of another work folder and never reads the proposal;
word labels count a variant as its word while a twin stays wrong, also on common clips; the report and the pack
carry the audit and warn on validation evidence; the driver refuses `--only link` on a changed state and does not
seed the audit file.

### 15.38 Classifier: a source quota sampler and selection on the worst source (QA plan 1.4, critics 2, 9 and 10, 27 Nov 2026)

**Why.** The research classifier draws a class by `count ** 0.5`, then a clip of that class by source weight. ISL500
has 59,463 training clips and INCLUDE 2,566, so inside a shared class almost every draw is an ISL500 clip. INCLUDE
clips are seen less often than in the commercial-safe run, and the critic showed that the E1b drop is confusion
inside the shared classes (critic 2). Selection on the pooled validation macro F1 has the same bias: about 94 % of
the research validation clips are ISL500.

**The sampler (`loop.balanced_batches`, new `groups` and `group_quota`).** With a quota every draw has two stages:
1. a quota group, with probability equal to its share;
2. a class among that group's clips, by `count_in_group ** class_power`, then a clip of that class and group by its
   per-clip weight (the ISLRTC x4 and the WSLP 0.8 still apply inside the dictionary group).
Without a quota the draws are exactly the old ones (a test compares them with the old code).

**Quota groups (`training/source_quota.py`).** A key is a source name or a family: `dictionary` (islrtc, wslp,
cislr) and `mendeley` (every `mendeley_*`). An exact source name wins over its family. Groups with no training clip
are dropped and the rest renormalised, so one quota fits every split. A training source with no share is refused
(a new source never silently gets no draws), and so is an unknown name (a typo). Research default:
`include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08`.

**Selection (`train_classifier(select_on=...)`).**
* `macro_f1` (default): as before, the pooled validation macro F1.
* `worst_source`: the epoch with the highest `val_worst_major_f1`, the lowest validation macro F1 of INCLUDE and
  ISL500 (those present). With neither present it falls back to the pooled number. The temperature is then fitted on
  source-balanced validation: every source weighs the same in total, a source with fewer than 20 clips weighs
  `clips / 20` (`fit_temperature(weights=...)`; without weights the fit is unchanged). Changed in 15.39: the
  weights now follow the quota groups and their shares.
* Every history row now has `val_f1_<source>` and `val_acc_<source>` per validation source and `val_worst_major_f1`,
  in both modes.

**Recorded.** The checkpoint meta has `source_quota` (requested shares, source -> group, renormalised shares, sources
never drawn), `select_on`, `select_key`, `temperature_fit`, `val_by_source` (clips and top-1 per source after
calibration) and `data.expected_draw_share` (expected share of draws per source; the first log line prints it). The
quota and the selection mode are in `LoopConfig.extra` only when on (rule of 15.19), so a stale state of other
settings is refused and an old default state still resumes.

**CLI.** `train-classifier --source-quota SPEC|default|none` and `--select-on macro_f1|worst_source`. A bad quota exits 2.

**Driver.** `CLS_SOURCE_QUOTA` (empty or `none` = off) and `CLS_SELECT` (`macro_f1` / `worst_source`).
* Research defaults: the quota above, `worst_source`, `CLS_STEPS=800` and a new `CLS_PATIENCE=8` (full runs).
* Commercial-safe defaults: no quota, `macro_f1`, 300 steps, patience 10. Its commands are unchanged.
* The quota and selection go to the classifier and cv steps. e1a (INCLUDE official split) never gets them and keeps
  patience 10.
* The classifier and cv stamps record `quota=` and `select=`; the start line and `--dry-run` print them.

**How to read the results (critics 2, 9, 10).**
* Run the quota ablation alone first (critic 2), at the plan's 1a setting (primary split, 300 steps, about 0.75 h
  per arm). Use the direct command, not the driver (see 15.39 for the exact command and why).
  Credit an E1b gain to 1.4 only from that run.
* Choose the quota on INCLUDE validation, never on test (critic 9). The per-source history makes that possible
  without a test pass.
* QA plan 1.6 moves two ISL500 signers to validation. Selection then sees a different ISL500 validation set, and an
  ablation on the old split does not measure that change (critic 10). Compare runs on the same split only.
* Open (critic 8): worst-source selection selects on the same INCLUDE validation clips that later certify the
  thresholds. Out-of-fold calibration (cv folds) is still the cleaner fix.
* Not done (plan 1.4): an encoder-initialised classifier. It needs a new architecture and pack changes (Batch 2).

Tests (`backend/tests/isl/test_source_quota.py`): 10,000 draws give every group its share within 2 %; the expected
share formula matches the draws; no quota draws exactly as the old code; class power inside a group; parsing,
renormalisation and refusals; the CLI refuses a bad quota; source-balanced weights and the weighted temperature fit;
the classifier's `val_worst_major_f1` is the minimum of the INCLUDE and ISL500 F1 in every history row and picks the
best epoch; meta records the quota; a changed quota or selection raises `StateMismatch`; the default run is unchanged
but logs per-source rows; the driver's research and commercial-safe defaults; e1a gets no quota.

### 15.39 Quota review fixes: checked before the run, a fair temperature, a cheap ablation (review of 15.38, 27 Nov 2026)

**The driver checks the quota before anything runs.** Before, the driver only checked the syntax of
`CLS_SOURCE_QUOTA`. A typo (`incldue=0.45`) or a quota without the dictionary failed only when the classifier step
started, after link, bench, pretrain, encoder and oneshot. Now, when the classifier or cv step is in the run, the
driver calls `parse_source_quota` and `resolve_source_quota` against every word source of the profile
(`profiles.WORD_SOURCES`). A bad quota exits 2 and nothing runs, `--dry-run` included. This is a little stricter than
the sources a run really has, which is fine for a pre-flight. The check is one `python -c` line calling
`source_quota.check_source_quota`. That module now imports no torch (it keeps its own copy of the dictionary family,
`DICTIONARY_FAMILY`, and a test checks it equals `common.DICTIONARY_SOURCES`), so the check takes well under a second.
`CLS_SOURCE_QUOTA=default` now means the research default quota, as `--source-quota default` does.

**A stamp of other settings is said, not hidden.** A resume without `--from` skips a classifier or cv step that has a
stamp. If the stamp's settings (`dict_group`, `quota`, `select`) differ from this run's, the driver now logs
`warning: classifier stamp settings differ from the current ones (...); use --from classifier to retrain` (also in
`--dry-run`). It never refuses, so a running chain behaves as before. Every research stamp written before 15.38 gets
this warning.

**Temperature weights by quota group.** In 15.38 every validation source weighed the same. On the real research
data that gave 46 Mendeley emergency clips the same weight as 856 INCLUDE clips, and WSLP plus CISLR would give the
dictionary two votes. Now:
* the units are quota groups: the dictionary family and the Mendeley family count once each;
* each unit weighs its quota share (the run's quota, else the research default one; a source of no group gets the
  smallest share);
* a unit with fewer than 20 validation clips gets `clips / 20` of its share, as before.
So INCLUDE weighs 0.45, ISL500 0.35, Mendeley 0.08 (renormalised over the groups present). The meta records the
total weight per source in `temperature_weight_by_source` (in the plain fit: each source's share of the clips).

**The temperature fit is its own option.** `train-classifier --temperature-fit auto|all|source_balanced`. `auto`
(the default) keeps 15.38: source-balanced with `--select-on worst_source`, all rows alike otherwise. So a quota run
with `macro_f1` selection can still ask for the balanced fit. A value other than `auto` goes into `LoopConfig.extra`
(rule of 15.19). The driver passes nothing new.

**Cost of the research defaults.** The research classifier now runs 40 x 800 steps (patience 8, about 2.2 h). The cv
step inherits the quota, the selection and the 800 steps for all 4 folds: about 9.2 h, the largest new cost of 15.38
(table in 15.2). For a cheaper run: `CLS_SOURCE_QUOTA=none`, `CLS_STEPS=300` or `CV_FOLDS=` (no cv).

**The quota ablation (plan 1a), the cheap way.** The driver example of 15.38 ran a full chain per arm (800 steps,
cv, index, fusion, evaluate, pack: over 12 h per arm). Use the classifier command alone, from `backend/`, with the
same settings for all 3 arms and only the quota changed:

    nice -n 10 ../.venv/bin/python -m app.isl.cli train-classifier --profile research --all-sources --threads 4 \
      --vocab ../artifacts/isl/research/vocabulary.jsonl --hidden 192 --layers 2 --epochs 40 --batches 300 \
      --patience 10 --dictionary-group --select-on worst_source \
      --source-quota include=0.30,isl500=0.50,dictionary=0.12,mendeley=0.08 \
      --ckpt ../artifacts/isl/runs/q30/classifier.pt

* Arms: `include=0.30,isl500=0.50,...`, the default `0.45 / 0.35`, and `include=0.60,isl500=0.20,...`. Same
  `--batches`, `--epochs`, `--patience` and split for all 3.
* About 0.75 h per arm on 4 threads. It writes only the `--ckpt` file and its `.state`.
* Record for each arm, from the checkpoint meta: `val_by_source.include.top1` and `val_f1_include` of the best
  epoch's history row. Choose on those, never on test.
* With the driver instead: `PHASE2_TAG=q30 CLS_STEPS=300 CLS_PATIENCE=10 CV_FOLDS= CLS_SOURCE_QUOTA=...
  scripts/phase2_train.sh --profile research --from classifier`, and stop it after the classifier step.

**Still open (critic 8, validation reuse).** With `worst_source`, INCLUDE validation (856 clips, 4 signer groups)
now chooses the quota, the epoch and the temperature, and later certifies the 0.4 threshold. The selection bias is
larger than the 0.0001-0.0008 the plan quotes for thresholds alone. Test data is never used. Follow-up for Batch 1:
out-of-fold INCLUDE calibration from cv folds 1-4, or one INCLUDE session held out for calibration only, and a
warning in the evaluation report when `meta.select_on` is `worst_source`.

Tests (`test_source_quota.py`): the driver refuses `incldue=1`, `include=0.5,isl500=0.5` and all-zero shares under
research in `--dry-run` (exit 2) and skips the check when neither classifier nor cv runs; `default` works;
classifier and cv stamps of other settings give the warning, current ones do not; the weights follow the quota
groups on the review's real counts; the classifier's temperature is the weighted fit of the best checkpoint's
validation logits; `--temperature-fit` works with `macro_f1` and is in the resume signature; with only Mendeley
validation clips `val_worst_major_f1` falls back to the pooled macro F1.

### 15.40 Encoder: a dictionary-anchored sampler and a validation that sees the dictionary (QA plan 1.5, critics 3, 11 and 14, 27 Nov 2026)

**Why.** The research encoder never learned INCLUDE <-> ISLRTC: the mean true-pair cosine stayed at the pretrained
level (critic 1). The P x K sampler draws a concept and then two clips by source weight. A concept with 200 ISL500
clips and one ISLRTC row almost never gives an INCLUDE + ISLRTC pair (about 13 per epoch in total). And the old
`val_r1` is about 94 % ISL500 queries against a gallery that also holds training clips, so it cannot see the
dictionary either.

**The bridge sampler (`loop.pk_batches`, `bridge_plan`; `train-encoder` / `train-oneshot --bridge-p N`).**
* A bridge concept has an ISLRTC row and at least one signer clip (INCLUDE, ISL500 or Mendeley) in the training list.
* Every batch first takes `N` bridge concepts (plan: 12 of P = 32). Each gives one ISLRTC row plus K-1 signer clips.
* The signer source is drawn by `--source-weights` (default `include=0.5,isl500=0.35,mendeley=0.15`), renormalised over
  the sources that concept has. `mendeley` means every `mendeley_*` source. A dictionary source is refused (it is the
  anchor side), and so is an unknown name.
* Then "multi" concepts fill the batch up to half of P (the bridge concepts count toward that half). A multi concept
  now needs two recording groups: two ISLRTC rows of one word are one group, the dictionary
  (`signer_group_key(dictionary_group=True)`). The rest is uniform, as before.
* Without `--bridge-p` the draws are exactly the old ones (a test compares them with the old code).
* The meta has `sampler`: the bridge concepts per signer source, the multi concepts, and the expected anchor <-> source
  pairs per concept over the planned run (`expected_pairs_per_label`). The plan's target is at least 50 INCLUDE pairs
  per INCLUDE concept. The first log line prints the same.

**The dictionary validation (`--val-protocol dictionary`; default `mixed` = the old one).**
* Queries: every INCLUDE validation clip, ISL500 validation clips stratified at 300 per signer
  (`--dict-val-per-signer`, 0 = all) and every Mendeley validation clip. A query whose word has no ISLRTC row is left
  out (it cannot be retrieved). The counts before and after are in meta `val_dictionary_sets`.
* Gallery: the ISLRTC rows only, no training clip.
* Held-out concepts of a one-shot fold are in neither the queries nor the gallery (critic 11). So fold selection never
  sees a held-out concept. The same code serves the encoder and the oneshot folds.
* Features are cached once (float16); the old and the new sets share one cache, so each clip is embedded once per epoch.
* Every history row has `val_dict_r1_<src>`, `val_dict_r5_<src>` and `val_dict_n_<src>` for `include`, `isl500` and
  `mendeley`, the pooled `val_dict_r1`, and the old `val_r1` / `val_r5` as a secondary. Each epoch logs one line:
  `val dictionary: include->islrtc r1 ... isl500->islrtc r1 ...; select ...; old val_r1 ...`.
* The epoch is selected on `val_dict_select = 0.5 * r1_include + 0.5 * mean(r1 of the other sources)`. With one side
  missing it is that side alone. The meta records `val_protocol`, `select_key` and the best epoch's `val_dict`.

**Resume signature.** `LoopConfig.extra` gets `bridge` (N, weights, anchor, the bridge concepts' hash) and
`val_protocol` / `val_dict` (per-signer count and the query and gallery hashes) only when they are on (rule of 15.19).
A state of the old settings still resumes with the old options; a change of either is refused (`StateMismatch`).

**Measuring any encoder: `cli encoder-probe` (critic 3).** It runs one encoder checkpoint on exactly this protocol.
* Read-only: it loads the data like the training steps, writes nothing into the work folder, and prints the result
  (`--out FILE` also writes it). The split defaults to the encoder's own.
* The encoder's `excluded_concepts` and excluded sources are left out, as in training.
* `--bar FILE|R1`: a previous probe JSON (its `include_r1`) or a number. Below it, the result has `warning_bar` and the
  command prints a warning. This is the plan's "warn when INCLUDE -> ISLRTC r1 falls below the previous value".
* `--work DIR` reads that folder's `link_settings.json` (default: the profile's work folder).
* Smoke on a copy of the commercial-safe encoder (300 ISLRTC rows, 1 thread): 15 s, 0.86 GB. The full protocol is
  about 15k clips, about 1 min at 1 thread and about 1.5 GB.

**The encoder gate bar (critic 3).** The plan's bar of 0.352 came from another protocol. Measure the commercial-safe
encoder on this protocol and use its number. Run both encoders on the same data (the research profile, so ISL500 is
measured too), from `backend/`:

    nice -n 10 ../.venv/bin/python -m app.isl.cli encoder-probe --profile research \
      --vocab ../artifacts/isl/research/vocabulary.jsonl --threads 2 \
      --encoder ../artifacts/isl/commercial-safe/encoder.pt --out <scratch>/probe_cs.json
    nice -n 10 ../.venv/bin/python -m app.isl.cli encoder-probe --profile research \
      --vocab ../artifacts/isl/research/vocabulary.jsonl --threads 2 \
      --encoder <new encoder.pt> --bar <scratch>/probe_cs.json --out <scratch>/probe_new.json

Gate: `include_r1` of the new encoder >= `include_r1` of `probe_cs.json`, and `by_source.isl500.r1` >= 0.25 (plan).
This gate is biased toward a dictionary-selected encoder and is a coin flip at parity; 15.41 makes it paired, with a
margin, and confirmed on E2_seen.

**A short encoder probe run first (critic 14).** The driver has a new `--until STEP`: stop after STEP; later steps are
neither run nor forgotten. Two scratch arms of about 3 epochs, each logging INCLUDE -> ISLRTC r1 per epoch:

    PHASE2_TAG=encprobe-bridge ENC_EPOCHS=3 ENC_STEPS=400 scripts/phase2_train.sh --profile research --from encoder --until encoder
    PHASE2_TAG=encprobe-plain ENC_EPOCHS=3 ENC_STEPS=400 ENC_BRIDGE_P=0 scripts/phase2_train.sh --profile research --from encoder --until encoder

* Both use the dictionary validation (research default), so the per-epoch lines compare directly.
* Each is a scratch run (15.19-15.20): own work folder `artifacts/isl/runs/<tag>/research`, the earlier steps seeded
  from `artifacts/isl/research`. About 3 x 400 x 0.54 s plus validation: about 20-25 min per arm at 4 threads.
* The research work folder has no `link_settings.json` yet (it was linked before 15.34), so a probe seeded from it links
  without aliases. That is fine for this comparison; the full run relinks with `--from link`.
* `encoder-probe` on each arm's `encoder.pt` gives the same numbers as the gate.

**Driver.**
* `ENC_BRIDGE_P` (research 12, commercial-safe 0), `ENC_SOURCE_WEIGHTS` (empty = the default weights; needs
  `ENC_BRIDGE_P` > 0) and `ENC_VAL` (research `dictionary`, commercial-safe `mixed`). They go to the encoder AND the
  oneshot step. A bad value exits 2 before anything runs.
* Full research runs now default to 800 steps per epoch for the encoder AND oneshot (the plan; oneshot read
  `ENC_STEPS` before too). `ONESHOT_STEPS` sets oneshot alone. Commercial-safe keeps 400 and 300: its commands are
  unchanged.
* The encoder and oneshot stamps and the start line record `bridge_p=... source_weights=... val=...`; `--dry-run`
  prints an `encoder:` line with the settings.
* Cost: research encoder 40 x 800 steps is about 5.8 h, oneshot 2 folds x 20 x 800 about 5.5 h (plan table).

**Not changed.** The running Batch 0.6 chain: the driver file was replaced, not edited in place, so the running bash
keeps its own copy; fit-fusion, evaluate, build-index and build-pack import nothing new. ASL is untouched.

**Open.**
* The selection key gives Mendeley (about 12 validation clips with a dictionary row) the same weight as ISL500
  (600). That is the plan's formula; it is noisy. A `clips / 20` damping, as in the temperature fit (15.39), would be
  a small follow-up if the per-epoch curves jump.
* The INCLUDE validation clips now also choose the encoder epoch. That adds to the selection reuse of critic 8.

Tests (`backend/tests/isl/test_encoder_bridge.py`): the default draws equal the old code's; in a toy world every batch
of P = 32 has at least 12 ISLRTC <-> signer pairs (the old sampler under 6), also with K = 3; the signer source shares
follow the weights within 2 % and renormalise per concept; two ISLRTC rows are no multi concept; the expected pairs
match the draws; weight parsing and the CLI refuse a dictionary source and bad values (exit 2); the validation gallery
has only ISLRTC rows, held-out concepts are in neither set, ISL500 is stratified per signer; the per-source metrics and
the selection key; a toy training run has the per-source keys in every history row, selects on the key and refuses a
changed bridge or protocol on resume; the default run adds no resume key; `encoder-probe` writes nothing into the work
folder, matches the encoder's own best epoch, warns below a bar and leaves out a fold encoder's held-out concepts; the
driver's research defaults, commercial-safe unchanged, bad values refused, and `--from encoder --until encoder` runs
and forgets only the encoder.

### 15.41 Encoder bridge review fixes: a bar that measures the same thing, a paired gate, no silent options (review of 15.40, 27 Nov 2026)

**A probe bar must measure the same thing.** `encoder-probe` now records what it measured: `profile`, `split`,
`per_signer`, `data_seed` (the `--seed` the data was loaded with; it sets the signer split of the extra sources),
`link` (hash of the link settings), `query_digest`, `include_query_digest` and `gallery_digest`.
* With `--bar FILE`, the probe compares `profile`, `split`, `per_signer`, `data_seed`, `link`, `include_query_digest`
  and `gallery_digest` with the bar's. Any difference gives `warning_bar_mismatch` and a printed warning. A bar JSON
  written before this change has none of the keys, so it is always flagged. A plain number is not checked
  (`bar.checked = []`).
* `--profile` now defaults to the encoder's own profile (its meta), then the settings default, as `_model_ctx` does.
  So a commercial-safe encoder is probed on commercial-safe data unless `--profile research` is given. The gate in
  15.40 passes `--profile research` to both probes, so both bars measure the same queries.
* A new dictionary-validated encoder records the hashes of its own validation sets (meta `val_dictionary_sets`). The
  probe reports `same_sets_as_training` (true / false; null when not recorded or another per-signer value) and warns
  when they differ: another data seed, link settings or data.

**The gate is paired and has a margin.** The research encoder's epoch is chosen on `val_dict_select`, half of which is
INCLUDE r1 on these same INCLUDE validation clips, over up to 40 epochs. So its probe number is optimistic against
the commercial-safe encoder, which was chosen on the mixed `val_r1`. The probe says so in `note_selection`. At parity
the comparison is a coin flip: about 850 INCLUDE clips give a standard error of about 0.017, and they come from only
4 session groups.
* The probe JSON now holds `hits`: hit@1 of every query (`<source>|<clip id>` -> 0 / 1).
* With a probe JSON bar that has `hits`, `bar.paired` compares the two encoders on their common INCLUDE queries:
  `new_only`, `bar_only`, `delta_r1` and the exact McNemar p-value. Clips of one session are not independent, so the
  p-value is optimistic too.
* `bar.clears` = new INCLUDE r1 >= bar + `--bar-margin` (default 0.02, about one clip standard error).
* Gate: `bar.clears` is true, `bar.paired.new_only > bar.paired.bar_only`, and no `warning_bar_mismatch`. Then confirm
  once on E2_seen (test), after the selection is final. `warning_bar` (below the bar) keeps its meaning.

**No silent options.** `train-encoder` and `train-oneshot` exit 2 before loading data, and `train_encoder` raises
`ValueError`, when an option would be ignored: `--source-weights` without `--bridge-p`, a negative
`--dict-val-per-signer`, or a non-default `--dict-val-per-signer` with `--val-protocol mixed`. `train-oneshot --ckpt`
exits 2: oneshot always writes `<work>/oneshot/foldK/encoder.pt`; use a `--tag` scratch run for another folder.

**Driver.**
* `stamp_warn` now covers `encoder` and `oneshot`. It warns when this run asks for the bridge sampler, the dictionary
  validation or source weights (or the stamp has them) and the stamp's settings differ. A stamp from before 15.40
  matches the default settings, so commercial-safe stays quiet. A research resume that skips the old encoder now says
  `warning: encoder stamp settings differ ...; use --from encoder to retrain`.
* `ENC_BAR=<probe.json>` (default off): after the encoder step, the driver runs `encoder-probe --bar <probe.json>
  --out <work>/eval/encoder_probe.json` with the run's profile, vocabulary and work folder. A warning goes to the log
  when the probe is below the bar or not comparable. A probe failure never fails the encoder step. A missing file
  exits 2 before anything runs. The encoder step deletes an old `encoder_probe.json` first.
* `evaluate --all` copies `<work>/eval/encoder_probe.json` into `report.json` as `encoder_probe` (numbers, bar, paired
  result, warnings, `stale` when the probed encoder is not the current `encoder.pt`) and adds an "Encoder probe"
  section to `report.md`. Without the file, the report has no new key.

**Not changed.** Defaults: no new resume key, no new flag in the commercial-safe commands, evaluate unchanged without
the probe file. The running Batch 0.6 chain keeps its own copy of the driver (the file was replaced, not edited in
place). ASL is untouched.

Tests (`backend/tests/isl/test_encoder_bridge.py`): the ignored-option refusals (library and CLI) and
`train-oneshot --ckpt`; the probe records per-signer, data seed and per-query hits, defaults to the encoder's profile,
gives a paired result against a matching bar and `warning_bar_mismatch` against a bar of another per-signer value or
an old bar; McNemar and the paired counts; per-query hits in the metrics; the probe in the evaluation report and
`stale`; the driver's encoder / oneshot stamp warning (research on, commercial-safe quiet) and `ENC_BAR`.

### 15.42 More calibration signers (signer split v2) and out-of-fold INCLUDE calibration (QA plan 1.6, critics 8 and 10, 27 Nov 2026)

**Problem 1: two ISL500 validation signers.** The signer split put 11 ISL500 signers in train, 2 in validation and 2
in test. Two validation signers cannot estimate how much precision varies between signers. The two test signers then
missed 0.98.

**Fix 1: signer split v2** (`training/multisource.py`, `cli link-sources --signer-split`).
* `signer_split` takes a per-source validation share. v2 = ISL500 0.27, every other source 0.1 as before.
* On the real manifests (read-only check): v1 = 11 / 2 / 2 signers (59,463 / 12,261 / 8,432 clips); v2 = 9 / 4 / 2
  (49,714 / 22,010 / 8,432 clips). Mendeley is the same in both.
* The shuffle and the seed are unchanged. The first two signers are still test: user005 and user015. So test numbers
  stay comparable. The two v1 validation signers (user002, user014) stay validation.
* **Critic 10: v2 moves two ISL500 signers from train to validation** (user009 and user013). ISL500 training drops
  from 11 to 9 signers (about 16 % fewer ISL500 training clips). So a v2 run differs from a v1 run in two ways at
  once: more calibration signers and less signer diversity in training. A gate "no ISL500 test regression" can fail
  because of v2 alone, and a 1a ablation on the v1 split does not measure this change. Run the ablation on v2, or read
  gate (d) with this in mind.
* Opt-in. The link step writes `signer_split: "v2"` (and its parameters) into `link_settings.json`. Every later step
  reads the folder's settings. A folder without the key is v1, as before. For v1 the settings file is byte-identical
  to before.
* Driver: `SIGNER_SPLIT=v1|v2`, research default v2, commercial-safe v1. Only the link step gets it. When the link step
  is not run, the start line shows the folder's split and warns when it differs. To change the split, run
  `--from link`.
* Resume signatures: the classifier and the encoder add `signer_split` to `LoopConfig.extra` only when it is not v1
  (rule of 15.19). The data summary in checkpoint meta says `signer_split` only when not v1.
* Pre-training. `pretrain_dedup` and `pretrain` read the split through `eval_clip_ids`, so after a v2 link step they
  leave the new validation signers out. `--from link` re-runs both. If someone links v2 and keeps an old pre-training,
  `train-encoder` and `train-oneshot` refuse the warm start `pretrained_encoder.pt` when `pretrain_clips.txt` holds a
  v2 validation or test clip (exit 2; `--init none` skips it). The report's pre-training guard also reports the
  overlap. v1 is not checked by the new refusal, exactly as before.
* Variant links. An audit made on v1 recorded the v1 train signers. A v2 link step refuses it (two of its train
  signers are now validation). Re-run `cli variant-audit --signer-split v2` first.

**Problem 2 (critic 8): INCLUDE certified on the selection clips.** fit-fusion certified INCLUDE on the primary
validation part: 856 clips in 4 pseudo-signer groups. These clips also chose the classifier's epoch, temperature and
quota. And 856 clips could not reach a 0.95 lower bound.

**Fix 2: out-of-fold calibration** (`training/oof_calibration.py`, `fit-fusion --oof-calibration`, driver
`FUSION_OOF=1`).
* The cv classifiers `cv/fold1-4/classifier.pt` never trained or selected on their own fold's test part. Those four
  test parts together are exactly the primary train and validation INCLUDE clips. On the real data: 3,422 clips, 1,070
  sessions, the 4 pseudo-signer groups. Fold 1's test part is the primary validation part; folds 2-4 are primary train.
* Each clip is scored by ITS fold's classifier. Encoder, fusion weights, temperature and aliases are the primary
  ones. The clips certify under source `include` with their own signer group. They replace the primary validation
  INCLUDE clips they cover (fold 1), so no clip counts twice. The other sources certify on the primary validation
  clips as before.
* What still uses the primary validation clips: the weights, the temperature, the sliding-mode thresholds and the
  "My signs" mode choice.
* **Leak checks.** No out-of-fold clip, and no duplicate of one, may be in the primary split's test part (the E1b
  test): `LeakageError`, before any work, in the cli and again in `fit_fusion`. Fold 0 itself is refused. Parts must
  not overlap.
* **Assumption: the fold classifiers are exchangeable.** The app uses the fold-0 classifier. A fold-K classifier is
  the same code and settings on another 3/5 of the sessions, so we take its accuracy on unseen sessions to be the
  same. `check_fold_classifier` refuses a fold classifier of another split, architecture, size, features, train
  config, class grouping, source quota, selection or signer split. On the real research folder all four pass. The class
  lists differ a little (fold 2: 4,352 of its 4,353 classes are classes of the 4,354-class fold-0 model).
* **The encoder is not out of fold.** There is one encoder, trained on the primary train part, so it has seen the
  clips of folds 2-4. Two defences:
  1. Leave-fold-out gallery. Fold K is scored against the training-only index with every row built from a fold-K
     clip (or a duplicate) removed. INCLUDE centroids are recomputed from their other clips. First the code rebuilds
     every stored centroid from the clips and refuses when one does not match (cosine < 0.995). So no clip retrieves
     itself.
  2. In-sample screen. DiD = (retrieval top-1 on folds 2-4 minus fold 1) minus (fold-classifier top-1 on folds 2-4
     minus fold 1). The classifier difference controls for fold difficulty, since every fold classifier is out of
     sample. 95 % interval from a session-cluster bootstrap. When the DiD is above `--oof-max-encoder-gain` (0.03), the
     in-sample folds are not used: only fold 1 certifies (then INCLUDE still gains an out-of-fold classifier, not more
     clips). `--oof-keep-in-sample` (driver `FUSION_OOF_KEEP_IN_SAMPLE=1`) uses them anyway, recorded.
  * The screen is not a proof. Passing means no advantage above 0.03 was seen. The interval says how large it could
    still be. Fold 1 also chose the encoder's epoch and the fusion weights, which makes the out-of-sample arm a little
    optimistic and the screen a little less sensitive. A per-fold encoder would remove this assumption.
* Report and record. `fit-fusion` prints `oof_calibration`: per fold the classifier and its SHA-256, clips, groups,
  classifier / retrieval / fused top-1, the gallery changes, the screen, the folds used and dropped, and the
  certification clips per source. `fusion.json` `certification.calibration_sets` records the clips, the folds, the
  assumption and the screen. `selection_note` then says INCLUDE was certified out of fold.
* Cost: one extra evidence pass over about 3,400 clips plus about 2,600 INCLUDE embeddings for the centroids.
* Opt-in for both profiles. Without the flag, fit-fusion is exactly as before. The fusion step needs the cv step
  (fit-fusion exits 2 when a cv classifier is missing).

Tests: `backend/tests/isl/test_signer_split_oof.py`. With the same seed, v1 and v2 have the same test signers (user005,
user015 by name) and v2 has 4 val groups; other sources are unchanged; v1 link settings are unchanged; the resume
signature names v2 only; `attach_all_sources` and `link-sources --signer-split v2` follow the settings; the report's
pre-training guard and the encoder's warm-start check catch a v1 pre-training under v2; a v1 audit fails the v2 split
check; the driver passes `--signer-split v2` for research and `--oof-calibration` only with `FUSION_OOF=1`. Out of fold:
the parts never hold a fold-0 test clip or a duplicate of one; fold 0 and a missing classifier are refused; the
leave-fold-out gallery holds no row of the fold and rebuilds the centroids; a fold classifier of other settings is
refused; certification uses each out-of-fold clip with source `include` and its own group, replaces the primary
validation INCLUDE clips, and leaves the weights unchanged; the screen drops in-sample folds when the encoder
memorised them, not when a fold is just easier for every model.
