# Indian Sign Language (ISL) support: architecture and design

Status: **Proposed** (phase 1 design, 25 Nov 2026). Branch `claude/indian-sign-language`.
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
* Hindi user interface (menus in Hindi). Output text is bilingual; UI translation can follow later.
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
Context: INCLUDE gives about 16 clips per word for 263 words (multi-signer); ISLRTC gives **one** clip per entry for about 12k concepts (single signer).
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
  * Queries = all INCLUDE clips of fold-k concepts (signers from Chennai, never the ISLRTC signer).
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
