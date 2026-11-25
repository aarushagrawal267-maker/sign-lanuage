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
| `FeatureConfig`, raw/feature layout | `backend/app/features/spec.py` | Unchanged layout. Two **opt-in** flags added (`canonical_hand`, default off) so ASL dims and behaviour stay identical. |
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
  "feature_config": {"seq_len": 48, "use_face": true, "use_velocity": true, "max_gap": 5, "canonical_hand": true},
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
                            "feature_config": {...}, "profile": "commercial-safe", "sha256": "..."}
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

* **ASL stays byte-identical.** `FeatureConfig.to_dict()` leaves `canonical_hand` out when it is false, and the engine's
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
  ASL threshold sliders are replaced by the pack policy (Careful / Balanced / More words = strict / balanced / lenient) and a
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

1. **Confirm ingestion finished**: `pgrep -af 'ingest_isl.py '` empty (the bare `ingest_isl` pattern also matches the pose / video ingesters); `include.jsonl` has 4,292 rows and `islrtc.jsonl` 13,665
   (count `ok: false` and 0-frame rows; re-run the script, which resumes, for any failures). Do not delete `data/isl`.
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
| index | `build-index --all-sources` | index over training references only (keeps evaluation clean) |
| fusion | `fit-fusion --all-sources` | fusion weights and thresholds on validation; always writes strict / balanced / lenient |
| evaluate | `evaluate --all --all-sources` | E1a-E8 -> `<work>/eval/report.json` + `report.md` and `artifacts/packs/<pack>/eval/` |
| index_final | `build-index --all-sources --all-include` | the shipped index: every reference of every source |
| pack | `build-pack` | `artifacts/packs/isl` or `isl-research` (copies `eval/`, lists every source and licence) |

Options: `--from STEP`, `--only STEP`, `--force STEP` (also deletes that step's training state), `--threads 4` once the
machine is free, `--step-minutes M` (each training call stops after M minutes with its state saved and is re-run with
`--resume` until done: exit code 75 means "stopped, continue later"). Full-setting sizes can be overridden with
environment variables (`HIDDEN LAYERS EMB PRE_EPOCHS PRE_STEPS PRE_BATCH ENC_EPOCHS ENC_STEPS CLS_EPOCHS CLS_STEPS
E1A_STEPS CV_FOLDS ONESHOT_FOLDS ONESHOT_EPOCHS LETTER_EPOCHS`). `--smoke` uses tiny settings, a private ISL root (real manifests and
landmarks through read-only symlinks, its own `derived/` and `splits/`) and writes only under
`artifacts/isl/phase2-smoke/`, so it never changes shared derived files or the packs the app loads.

**Sources per profile** (`backend/app/isl/training/multisource.py`; enforced twice: an allow-list per profile **and**
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
learned token after the input projection, reconstructed from the BiGRU states). Sources are sampled ~ 1/sqrt(size)
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
| cv | 4 more folds of the classifier | <= 48,000 | 4-6 h (skip with `CV_FOLDS=""`) |
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
