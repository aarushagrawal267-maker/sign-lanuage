# Indian Sign Language (ISL) datasets: sources, licences and data pipeline

Owner of this file: Data Engineer (docs/ISL_DESIGN.md section 10.2). Status: 25 Nov 2026, phase 1 (ingestion
running, partial data). Every number below says when and on how much data it was measured.

---

## 1. Sources and licences

| Source | What | Licence | Commercial use | Profile | Access |
|---|---|---|---|---|---|
| **INCLUDE** (AI4Bharat), zenodo record 4010759 | 263 ISL words, 4,292 videos, several signers (Chennai), 15 categories | CC BY 4.0 (zenodo metadata) | yes, with attribution | commercial-safe + research | public |
| INCLUDE split lists (AI4Bharat INCLUDE code repository, `train_test_paths/`) | official train/val/test file lists (3,127 / 348 / 817) | MIT (repository `LICENSE`, checked 25 Nov 2026); downloaded at run time, not redistributed | yes | both | public |
| **ISLRTC Indian Sign Language Dictionary** (Indian Sign Language Research and Training Centre, Government of India, data.gov.in) | 13,665 videos: about 12,400 dictionary concepts, numbers, phrases; one video per entry, several different signers across entries | Government Open Data Licence - India (GODL-India); ISLRTC's own FAQ (Q.23) adds: not to be resold or used for profiteering, acknowledge ISLRTC / DEPwD / MoSJE | **free, non-profit use only** with acknowledgement; any paid or ad-funded use needs written permission from ISLRTC (see note below) | both | public |
| ISLRTC Hugging Face re-encode (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`) | HEVC mirror of the above, used for downloading | labelled MIT by the uploader | mirror only: does not replace the GODL attribution | both | public |
| **CISLR** (Exploration-Lab) | about 4,700 ISL words, 7,050 videos collected from online ISL videos | AFL-3.0 in the Hugging Face metadata, but the dataset README (HF and GitHub) says CC BY-NC 4.0, "do not use our dataset for any commercial purposes" | **no**: treat as non-commercial (the README wins over the metadata tag) | research only | gated (`pending` on 25 Nov 2026) |
| **iSign** (Exploration-Lab) | 118k ISL video - English sentence pairs; MediaPipe Holistic poses in pose-format | CC BY-NC-SA 4.0; the access form adds "research use only, no commercial use, no redistribution" | **no** | research only (pre-training) | gated (`pending` on 25 Nov 2026) |
| **WSLP 2025 word recognition** (Exploration-Lab, `Exploration-Lab/WSLP`, `Shared_task_WR/`) <!-- video-sources --> | 4,398 labelled single-word clips, 4,361 words (train split; CISLR-like, our CISLR substitute); 635 unlabelled val/test clips | CC BY-NC-ND 4.0 (dataset card) | **no** (and no redistribution of derivatives) | research only | public |
| **ISL-Fingerspelling** (`kirandevraj/ISL-Fingerspelling`) | 1,308 continuous fingerspelling segments, 14,685 letters with frame boundaries, 3 signers, signer-independent split | CC BY-NC 4.0 | **no** | research only | public |
| **ISL-CSLTR** (Mendeley `kcmpdxky7p` v1, Elakkiya & Natarajan) | 687 sentence videos of 100 sentences (7 signers) with gloss sequences; 1,036 word-level still frames | CC BY 4.0 | yes, with attribution | commercial-safe + research | public |
| **ISL emergency words** (Mendeley `2vfdm42337` v1, Adithya & Rajesh) | 8 words x 26 people x 2 takes (412 raw videos), hands only (signer behind a board) | CC BY 4.0 | yes, with attribution | commercial-safe + research | public |
| **ISL agricultural words** (Mendeley `6v53kfxf46` v1, Adithya & Rajesh) | 13 words, 3 deaf farmers outdoors; 78 original videos (the crop1/2/3 folders are crops of the same recordings, not used) | CC BY 4.0 | yes, with attribution | commercial-safe + research | public |
| **Static ISL alphabet, Hindi vowels, numerals** (Mendeley `7tsw22y96w` v1, Singh et al.) | 102,470 photos, 46 classes (A-Z with two E signs, 8 Hindi vowels, 0-9 with two 9 signs), kids / teens / adults; we use a balanced sample of 300 per class (13,800) | CC BY 4.0 | yes, with attribution | commercial-safe + research | public |
| `Hemg/Indian_sign_language_dataset`, `akritRihal/Indian_Sign_Language_dataset` (Hugging Face) | alphabet + digit images (42,745 hand crops / 10,752 photos); 300 per class sampled | **none stated**, no provenance | **no** (unknown) | research only | public |
| MediaPipe HandLandmarker (`hand_landmarker.task`) | hand landmarks where Holistic finds none (hands-only videos, still images) | Apache-2.0 | yes | both | downloaded to `assets/mediapipe/` (git-ignored) |
| WLASL (existing ASL model) | ASL, 100 words | C-UDA 1.0 | **no** | research only (warm start) | see `docs/LICENSING.md` |
| WordNet 3.0 (via NLTK data) | synonym lists, part of speech, coarse categories | WordNet licence (permissive) | yes | both | public |
| MediaPipe Holistic | landmark extractor | Apache-2.0 | yes | both | bundled |
| pose-format (only for reading iSign; our reader is self-contained) | `.pose` file layout | MIT | yes | research | optional |
| **GloSL-ISL** (`pedroodb/glosl-isl`, Hugging Face) <!-- pose-sources --> | iSign / ISLTranslate MediaPipe Holistic poses re-packaged per clip: 126,846 sentence clips, 269 h, English text, source-video-disjoint splits (101,481 / 12,683 / 12,682) | CC BY-NC 4.0 (card); iSign author's permission to include the poses recorded in `card.json` (8 Oct 2026: poses only, no video clips, cite five papers) | **no** | research only (pre-training, sentence data, negatives) | public |
| **ISL500 / ISL-DATA** (`ISL500/ISL-DATA`, Hugging Face) <!-- pose-sources --> | 500 words x 15 signers (Goa); 86,760 landmark clips (MediaPipe hands `.h5` + MMPose COCO-WholeBody `.npy`), 7,500 MP4s | "research and academic use only; contact the authors for commercial use" (card), no SPDX licence | **no** | research only | public |
| **OpenHands raw ISL poses** (AI4Bharat, zenodo record 5558897) <!-- pose-sources --> | about 1,129 h unlabelled MediaPipe Holistic poses (pose + hands, no face) from five Indian deaf-community YouTube channels (Sign Library, NIOS, ISH News, MBM Vadodara, NewzHook) | CC BY 4.0 (zenodo metadata), but derived from third-party YouTube videos | undecided: **research only until the owner decides** | research only (pre-training) | public |
| **OpenHands Indian fingerspelling** (AI4Bharat, zenodo record 6731224, `Indian.zip`) <!-- pose-sources --> | 172 clips of 36 glosses (a-z, zero-nine), 123 train / 49 test | CC BY 4.0 (zenodo metadata); origin of the clips undocumented | undecided: research only until the origin is known | research only (fingerspelling smoke tests) | public |
| h5py <!-- pose-sources --> | reads the ISL500 / OpenHands HDF5 files (`scripts/ingest_isl_pose.py`) | BSD-3-Clause | yes | both | pip (`.venv`) |
| **IndicTrans2** en-indic distilled 200M (AI4Bharat), `ai4bharat/indictrans2-en-indic-dist-200M` | offline English -> Hindi drafts for `lexicon_hi.jsonl` (section 10); never shipped with or run by the app | MIT (model card + `LICENSE`, checked 25 Nov 2026) | yes; the Hindi rows it drafts are ours to license with the repository | both | official repo is "gated: auto" (contact details); fetched from a byte-identical public mirror verified against the official git object ids (`backend/app/isl/text/mt.py`) |
| Hindi core word list (`backend/app/isl/text/data/hindi_core.tsv`) | about 990 hand-written everyday words and senses (870 match an ISL concept, all 100 WLASL glosses) | project licence (written for this repository) | yes | both | tracked |
| User sign packs ("My signs") | user recordings as landmarks | chosen by the author at export (default CC BY 4.0) | per pack | never in global training | local |

**What "commercial-safe" means.** The `commercial-safe` profile leaves out every dataset with a
non-commercial licence (iSign, WLASL, CISLR, and every other source marked "no" or "undecided" in the table above). It still uses the ISLRTC dictionary, whose publisher allows research,
teaching and ISL technology but not resale or profiteering (ISLRTC FAQ Q.23, https://islrtc.nic.in/faq/, found by
the dataset survey on 25 Nov 2026, `docs/ISL_DATASET_SURVEY.md` section 4). So a `commercial-safe` pack is fine for a
free, non-profit app with acknowledgement; before any paid, subscription or ad-funded release, get written
permission from ISLRTC or retrain without ISLRTC data. The ISLRTC Hugging Face mirror's "MIT" label is the uploader's
claim and does not change this.

### Required attribution (copy into the app's About / Research page and any publication)

* "Indian Sign Language Research and Training Centre (ISLRTC), Department of Empowerment of Persons with
  Disabilities (DEPwD), Ministry of Social Justice and Empowerment (MoSJE), Government of India - Indian Sign Language
  Dictionary, published on data.gov.in under the Government Open Data Licence - India (GODL)."
* "INCLUDE dataset: A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. INCLUDE: A Large Scale Dataset for Indian Sign
  Language Recognition. ACM Multimedia 2020. CC BY 4.0, https://zenodo.org/records/4010759."
* Hindi machine drafts: "IndicTrans2: J. Gala et al., IndicTrans2: Towards High-Quality and Accessible Machine
  Translation Models for all 22 Scheduled Indian Languages, TMLR 2023. AI4Bharat, MIT licence." (the lexicon rows with
  `source: "model"`).
* Research profile only: "CISLR: A. Joshi et al., EMNLP 2022" and "iSign: A. Joshi et al., 2024 (arXiv 2407.05404),
  CC BY-NC-SA 4.0". iSign-derived data (landmarks, models pre-trained on it) inherit NC and share-alike terms.

* Commercial-safe video sources (CC BY 4.0, attribution required): "R. Elakkiya, B. Natarajan, ISL-CSLTR: Indian Sign
  Language Dataset for Continuous Sign Language Translation and Recognition, Mendeley Data V1, doi:10.17632/kcmpdxky7p.1";
  "V. Adithya, R. Rajesh, A Video Dataset of the Hand Gestures of Indian Sign Language Words used in Emergency Situations,
  Mendeley Data V1, doi:10.17632/2vfdm42337.1"; "V. Adithya, R. Rajesh, A video dataset on agricultural words in Indian
  sign language, Mendeley Data V1, doi:10.17632/6v53kfxf46.1"; "A. Singh, S. K. Singh, A. Mittal, B. B. Gupta, Static
  gestures of Indian Sign Language (ISL) for English Alphabet, Hindi Vowels and Numerals, Mendeley Data V1,
  doi:10.17632/7tsw22y96w.1".
* Research profile only (video sources): "WSLP 2025 shared tasks, Exploration-Lab (IIT Kanpur), CC BY-NC-ND 4.0,
  https://huggingface.co/datasets/Exploration-Lab/WSLP"; "Continuous Fingerspelling Dataset for Indian Sign Language,
  WSLP @ AACL-IJCNLP 2025, CC BY-NC 4.0, https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling"; the two Hugging
  Face alphabet image sets have no licence, so nothing derived from them may leave the research profile.

* Research profile only (pose-native sources) <!-- pose-sources -->: "GloSL-ISL (pedroodb/glosl-isl), CC BY-NC 4.0: iSign
  (A. Joshi et al., Findings of ACL 2024), ISLTranslate (A. Joshi et al., Findings of ACL 2023), CISLR (EMNLP 2022),
  PoseStitch-SLT (EMNLP 2025), WSLP 2025 overview"; "ISL-DATA / ISL500, https://huggingface.co/datasets/ISL500/ISL-DATA,
  research and academic use only"; "P. Selvaraj, G. NC, P. Kumar, M. Khapra, OpenHands: Making Sign Language Recognition
  Accessible with Pose-based Pretrained Models across Languages, ACL 2022; raw ISL poses https://zenodo.org/records/5558897
  and fingerspelling poses https://zenodo.org/records/6731224, CC BY 4.0 (zenodo label)". GloSL inherits iSign's
  non-commercial terms; models pre-trained on it stay research-only.

### Data profiles (`backend/app/isl/profiles.py`)

| Profile | Training / index / calibration sources | Pre-training only | Commercial |
|---|---|---|---|
| `commercial-safe` (default) | INCLUDE, ISLRTC | none | yes |
| `research` | INCLUDE, ISLRTC, CISLR, iSign | WLASL, iSign | no |

User data never enters a global model in either profile. The profile name is written into every derived artifact.

**What phase-2 training actually uses** (checked 25 Nov 2026 16:40 UTC by the Reality Checker): the table above is
still the content of `profiles.py`, which only the phase-1 commands read. The phase-2 commands (`--all-sources`,
`scripts/phase2_train.sh`) take their source lists from `backend/app/isl/training/multisource.py`
(`WORD_SOURCES`, `PRETRAIN_SOURCES`, `LETTER_SOURCES`) and check each adapter's own `commercial` flag as well:

| Role | `commercial-safe` | `research` |
|---|---|---|
| labelled words | include, islrtc, mendeley_emergency, mendeley_agri | + isl500, wslp, cislr |
| pre-training only | isl_csltr | + glosl, openhands_raw, isign, wslp_unlabelled |
| letters | mendeley_static | + isl_fingerspelling, openhands_fs, hf_hemg_alphabet, hf_akrit_alphabet |

Every adapter's flag was read on the real manifests: `commercial = True` only for include, islrtc, isl_csltr,
isl_csltr_words, mendeley_emergency, mendeley_agri and mendeley_static; every NC / ND / research-only / no-licence /
pending source is `False`, and asking for one in `commercial-safe` raises `LicenceError`. `isl_csltr_words` (1,036
CC BY still frames) is ingested but is in neither profile's lists, so no model uses it yet. WLASL is listed in
`profiles.py` but not in `multisource.py`. `profiles.py` should be brought in line with `multisource.py` (model owner).

Pose-native sources (section 3.4) are not yet registered in `profiles.py` / `sources.ADAPTERS` (that is the model
owner's call); `app.isl.data.pose_sources.PROFILE_SOURCES` says where they belong <!-- pose-sources -->: `research`
gains `isl500` and `openhands_fs` (labelled) and `glosl`, `openhands_raw` (pre-training only); `commercial-safe` gains
nothing. `pose_sources.register(ADAPTERS)` adds the adapters to a registry.

---

## 2. What is on disk (all under `data/isl/`, git-ignored)

| Path | Written by | Content |
|---|---|---|
| `manifests/include.jsonl` | `scripts/ingest_isl.py include` (running) | one row per INCLUDE video: id, gloss, category, zip, member, path, n_frames, fps, hands_frac, ok, error |
| `manifests/islrtc_index.json` | `scripts/ingest_isl.py islrtc` | the 13,665 dictionary file names and sizes |
| `manifests/islrtc.jsonl` | `scripts/ingest_isl.py islrtc` (running) | one row per dictionary video: + stop (card / eof / max_s / +posegap), duration_s (approximate; `t_ms` in the npz is authoritative) |
| `landmarks/{include,islrtc}/*.npz` | ingestion | `RawSequence` (pose 33x4, hands 21x3, face 20x3, t_ms; NaN = not detected). Videos are deleted after extraction |
| `derived/islrtc_trim.jsonl` | `python -m app.isl.cli trim` or `python -m app.isl.data.trim_bench run [--learned ...]` | first-sign frame range per ISLRTC clip + `second` cut (sidecar; npz untouched; section 7) |
| `derived/vocab_report.json`, `derived/duplicates.jsonl` | `vocab`, `dedup` | build report, near-duplicate clusters |
| `splits/official/*.txt` | `splits --fetch` | AI4Bharat lists (SHA-256 pinned in `splits.py`) |
| `splits/*.json`, `splits/include_groups.json`, `splits/oneshot_concepts.json`, `splits/splits_report.json` | `splits` | split assignments with leakage reports, session / pseudo-signer groups, one-shot folds |
| `raw/{cislr,isign}/*.csv` | `scripts/ingest_isl_gated.py` | the gated datasets' small CSV files |
| `nltk_data/` | `python -m app.isl.cli wordnet` | WordNet (10 MB) |
| `manifests/islrtc_full.jsonl`, `landmarks/islrtc_full/*.npz` | `scripts/ingest_isl.py islrtc --only ... --full --max-s 120 --out-subdir islrtc_full` (25 Nov, 7 entries) | full-length re-extraction of `A/Alphabet`, `F/Fingerspell` and the five `Numbers_*` compilation videos. Finding: `Alphabet` and `Fingerspell` (3.8 s / 3.4 s) are the signs for those *words*, not A-Z sequences, so the open data has **no fingerspelling letters**; see ISL_DESIGN.md 6.7 |
| `manifests/<source>.jsonl`, `landmarks/<source>/*.npz`, `manifests/<source>_items.json`, `raw/<source>/`, `raw/mendeley/` | `scripts/ingest_isl_video.py run <source>` (queue: `scripts/isl_queue.sh`) | video / image sources of section 3.3 (`wslp`, `isl_fingerspelling`, `isl_csltr`, `isl_csltr_words`, `mendeley_emergency`, `mendeley_agri`, `mendeley_static`, `hf_hemg_alphabet`, `hf_akrit_alphabet`; `cislr` shares `cislr.jsonl`). `_items.json` = the exact item list (sample) of the last run; `raw/` = cached CSVs and listings |
| `logs/queue.log`, `logs/video_<source>.{log,out}` | `scripts/isl_queue.sh` | one line per chunk; per-source progress |
| `manifests/{glosl,isl500,openhands_raw,openhands_fs}.jsonl`, `landmarks/<source>/**/*.npz`, `raw/<source>/`, `derived/glosl_isign_uids.txt`, `derived/isl500_links.jsonl`, `manifests/openhands_raw_members_done.txt`, `logs/ingest_pose_<source>.log` <!-- pose-sources --> | `scripts/ingest_isl_pose.py` (section 3.4) | pose-native sources converted to `RawSequence` (float16, about 15 fps); `raw/` holds the small CSV/JSON files, the HF revision used and (OpenHands) per-member dataset indexes |
| `../my_signs/<pack>/` | the app (My signs) | users' own signs as landmarks, `alphabet.npz` letter calibration; local only |

Model build outputs (git-ignored, `artifacts/`): `artifacts/isl/<profile>/` (classifier.pt, encoder.pt, index/,
fusion.json, results.json, smoke_report.json) and `artifacts/packs/<isl|isl-research>/` (the pack the app loads).

Tracked in git: `models/packs/isl/vocabulary.jsonl` (built from file names and split lists only, 7.5 MB), the code,
`backend/app/isl/vocab/overrides.yaml`. Never committed: videos, landmarks, manifests, splits, model weights.

---

## 3. Ingestion

### 3.1 INCLUDE and ISLRTC (`scripts/ingest_isl.py`, running since 25 Nov 2026 10:05)

Downloads one zip / video at a time, extracts landmarks with the app's own extractor, deletes the video.
Backward-compatible options added for the next start (default behaviour, ids and manifest format unchanged):

```
python scripts/ingest_isl.py islrtc --only A/Alphabet.mp4 F/Fingerspell.mp4 "Numbers/Numbers_(0-1_Crore).mp4" \
    --full --max-s 120 --out-subdir islrtc_full        # long reference videos, full length
```

INCLUDE quirk: 27 videos sit one level deeper in an `Extra/` folder; the running job gave them the gloss "Extra".
The adapters rebuild canonical ids from the zip member path, and the script now writes the right gloss/category.

Partial data on 25 Nov 2026 (about 12:15): INCLUDE 413 of 4,292 videos (all `ok`, all pass the quality filter);
ISLRTC 1,054 of 13,665 videos ingested (all `ok`; 50 with 0 frames because the title card came first; 1,002 pass the
quality filter `n_frames >= 8, hands_frac >= 0.2`). Stop reasons: max_s 619, eof 298, max_s+posegap 72, card 65.

### 3.2 Gated datasets (`scripts/ingest_isl_gated.py`)

```
python scripts/ingest_isl_gated.py --check           # exit 0 = granted, 3 = pending (cron-friendly)
python scripts/ingest_isl_gated.py cislr --workers 2
python scripts/ingest_isl_gated.py isign --subset 10000          # random access into the split zip
python scripts/ingest_isl_gated.py isign --mode stream           # fallback: parts piped one after another
```

* **Access check**: a 1-byte range request for a data file (`dataset.csv`, `iSign_v1.1.csv`). The dataset cards
  (`README.md`) answer 206 even without access, so they cannot be used. 25 Nov 2026: both repos `pending` (data 403).
  Credentials come from the egress proxy (or `HF_TOKEN` elsewhere); never printed or stored.
* **CISLR**: the three CSVs are downloaded (about 500 KB). The 1.17 GB video zip is opened through `RangeFile`
  (seekable HTTP range reads, 8 MB blocks, LRU cache); members are streamed one by one to a temp file, extracted with
  `extract_full` (15 fps, same as INCLUDE), deleted. Resumable per member; peak disk = one video per worker; at most
  2 workers and 2 GB of temp data. Manifest `manifests/cislr.jsonl` adds `uid`, `member`, `split` (prototype / test).
* **iSign**: the dataset card says the four pose parts (3 x 48.3 GB + 25.3 GB) are one **zip**. Default mode reads its
  central directory from the end of part_ad and fetches only the selected members (`.pose` files); the 58 GB video
  parts are never touched. The subset (default 10,000 sentences) is chosen from `iSign_v1.1.csv`: sentences containing
  the most vocabulary words first. Stream mode walks the piped parts sequentially (zip local headers or tar/tar.gz)
  and logs member offsets in `manifests/isign_offsets.jsonl` so an uncompressed archive resumes with a range request;
  a compressed stream has to be re-read from the start (an `indexed_gzip` seek index would avoid that; not needed
  while the archive is a zip).
* **iSign pose mapping** (`backend/app/isl/data/isign.py`, self-contained pose-format v0.1/v0.2 reader):
  `POSE_LANDMARKS` (33, pixels) -> `pose` x/W, y/H, z as stored (raw MediaPipe z, like our extractor; the earlier
  z/W scaling was removed in commit 514be97), confidence as visibility; `LEFT_HAND_LANDMARKS` /
  `RIGHT_HAND_LANDMARKS` (21) -> `left` / `right` (confidence 0 -> missing frame); `FACE_LANDMARKS` (468/478) ->
  the app's 20-point `FACE_IDX` subset; `POSE_WORLD_LANDMARKS` ignored; `t_ms` from the file fps;
  `extractor_version = "isign-pose-format"`. **Open check once data arrives**: whether the older Holistic used by
  iSign labels hands from the camera's view (then run with `--swap-hands`); compare one iSign clip with our own
  extraction of the same video before pre-training.

### 3.3 Video and image sources with our own extraction (`scripts/ingest_isl_video.py`, queue `scripts/isl_queue.sh`)

Owner: Data Engineer, 25 Nov 2026. Code: `backend/app/isl/data/video_sources/` (catalogue, listings, fetch, extract,
engine, adapters); tests `backend/tests/isl/test_video_sources.py` (no network).

```
python scripts/ingest_isl_video.py sources | status | probe [SOURCE ...]
python scripts/ingest_isl_video.py run SOURCE [--workers N|auto] [--limit N] [--max-minutes M] [--per-class N]
python scripts/ingest_isl_video.py islrtc-full            # ISLRTC alphabet / fingerspell / number compilations, full length
setsid nohup bash scripts/isl_queue.sh >> data/isl/logs/queue.log 2>&1 < /dev/null &   # everything, in order
```

**How it works.** A zip is never downloaded whole: its central directory is read with HTTP range requests
(`RangeFile`), members are streamed one at a time in archive order into a `mkstemp` temp file (descriptor closed at once,
SHA-256 computed on the way), converted, deleted. Single files (Mendeley, ISL-Fingerspelling) and Hugging Face dataset
rows (signed image URLs from the datasets-server rows API) work the same way. Member names are only labels: names that
are absolute or contain `..` are skipped, members over 2 GB or with a zip-bomb ratio are refused. A manifest row per item
(`id, source, gloss, licence, profile, sha256, bytes, [crc32], path, n_frames, fps, hands_frac, extractor_version, ok,
error, secs` + source fields) makes every source resumable; failed items are retried on the next run. Byte-identical
files (same SHA-256) are extracted once; later copies get a row with `duplicate_of` (adapters skip them by default). The
free disk under `data/isl` must stay above 10 GB (checked before every item; the queue waits otherwise).

**Extraction modes** (`video_sources/extract.py`):

| Mode | Used for | What |
|---|---|---|
| `full` | WSLP, CISLR, ISL-CSLTR sentences, ISL-Fingerspelling | `scripts/ingest_isl.py extract_full` itself (HolisticExtractor, max side 640): identical to INCLUDE. 15 fps, max 20 s (CSLTR: 60 s). Fingerspelling: **every frame** (30 fps source, step 1, max 60 s) so the letter boundaries map 1:1 onto `t_ms` |
| `hands` | Mendeley emergency and agricultural words | `full` + MediaPipe HandLandmarker (VIDEO mode, 2 hands, confidences 0.5) for frames where Holistic found no hand of that side. Needed because Holistic finds hands only through the body pose: on the emergency videos (signer behind a black board, only forearms visible) Holistic found a hand in 0-5% of frames. A HandLandmarker hand at the position of a Holistic hand is never copied to the other side. Handedness check on sample clips: where both models saw the same hand, the left/right labels agreed in 42 of 42 frames (per clip: `hl_agree`). `extractor_version` = `holistic-v1+hands` when hands were filled (`hands_filled` frames) |
| `image` | static alphabet / digit photos, CSLTR word frames, HF image sets | one-frame `RawSequence` (`t_ms = [0]`): Holistic in IMAGE mode (same confidences) for pose/face/hands, HandLandmarker (IMAGE mode) for the hands Holistic missed. On 15 sample photos Holistic alone found a hand in 2, HandLandmarker in 11. `extractor_version = image-holistic-v1+hands`; letter features are computed from frame 0 (`letter_frame_features`) |

**Sources** (sizes checked 25 Nov 2026; `profile` = where the data may be used):

| Source (`run` name) | Items | Labels | Profile | Notes |
|---|---|---|---|---|
| `wslp` | 4,398 clips (train.zip 5.01 GB, ~1.1 MB/clip) | 4,361 words (`train_WR.csv` `uid,word`) | research | CISLR-like word clips: our CISLR substitute until CISLR access. **Dedupe against CISLR** later by gloss + SHA-256 (stored now) and CRC32 |
| `wslp_unlabelled` | 635 clips (val 109 + test 526) | none published | research | same manifest `wslp.jsonl`, `labelled: false`; pre-training only |
| `cislr` | ~7,050 clips (1.17 GB zip) | `dataset.csv` | research | gated (pending); probed by the queue (1-byte request for the zip) and ingested as soon as it answers; same ids/paths/manifest as `scripts/ingest_isl_gated.py cislr`, plus `sha256`/`crc32` |
| `isl_fingerspelling` | 1,308 segments (1.3 GB) | letters (authoritative letter annotations), per-letter `start_frame`/`end_frame`, signer, signer-independent and standard splits | research | `adapters.letter_samples()` maps letter boundaries to our frame indices for the letter classifier |
| `isl_csltr` | 687 sentence videos (7.57 GB inside one 8.9 GB zip) | sentence (folder) + sign-gloss sequence (`ISL Corpus sign glosses.csv`) | commercial-safe | continuous signing, 100 sentences, 7 signers; some folders hold word clips (`free (2).MP4`): `file_hint` keeps the file name |
| `isl_csltr_words` | 1,036 still frames | word (folder) | commercial-safe | one frame of a mostly dynamic sign: hand-shape hints only |
| `mendeley_emergency` | 412 raw videos (8 zips, 2.58 GB; the cropped copies are not used) | 8 words | commercial-safe | hands only. Some takes are byte-identical (e.g. `call_001_01` = `call_001_02`): kept once (`duplicate_of`) |
| `mendeley_agri` | 78 videos (the `Original` folders; crop1/2/3 are crops of the same recordings) | 13 words (`Cococnut` -> coconut) | commercial-safe | outdoor, faces often cut off at the top -> `hands` mode. Subject 1's `Original` folder actually holds the `crop1` files (same names and sizes as its `cropped1` folder) |
| `mendeley_static` | 13,800 of 102,470 photos (300 per folder, round-robin over kids/teens/adults x full/half sleeves, seed 13) | A-Z (`E1`/`E2` -> E, kept as `variant`), 8 Hindi vowels, 0-9 (`9a`/`9b` -> 9) | commercial-safe | `--per-class N` for more; the selected list is in `manifests/mendeley_static_items.json`; Mendeley SHA-256 verified per file |
| `hf_akrit_alphabet` | up to 300 per class of 10,752 (33 classes, no h/j/v) | letters, digits | research | licence unknown |
| `hf_hemg_alphabet` | up to 300 per class of 42,745 (35 classes, 128x128 hand crops) | letters 1-9 + A-Z | research | licence unknown; probably augmented copies of few originals |
| `islrtc-full` | ISLRTC compilations not yet in `islrtc_full.jsonl`: `A/Ancient_Alphabets`, `G/Greek_Alphabets` (the 7 others were done on 25 Nov) | as ISLRTC | as ISLRTC | runs `scripts/ingest_isl.py`'s own `run_islrtc` (`--only --full --max-s 180 --out-subdir islrtc_full`) in-process, so no second `ingest_isl.py islrtc` process appears |

**Decided not to ingest.** WSLP `Shared_task_WPP` query videos: all 2,541 (train 1,410, val 576, test 555) are
byte-identical copies (same CRC32 and size, same uid) of WR clips, so they add no word labels; the WPP sentence videos
have only "is this word present" labels (no text) and stay out. WSLP `Shared_task_MT` (sentence videos + poses) is
the iSign-style translation task: left for the pose-source owner. Cropped copies of the Mendeley videos (duplicates).

**Throughput measured 25 Nov 2026** (1 worker at nice 10 while INCLUDE, ISLRTC and other agents' jobs kept the 4 CPUs
busy, load average about 10; first few items of each source): WSLP 7.1 s/clip (11.7 s/clip at load 18-21 later
that day, section 3.5), ISL-Fingerspelling 8.9 s/segment
(every frame), emergency 3.9 s, agriculture 6.4 s, static photos 0.26 s + download, CSLTR word frames 0.3 s; CSLTR
sentence videos not measured on the large 1080p files yet (estimate 20-40 s). Landmarks: about 110 KB per WSLP clip.

**Queue** (`scripts/isl_queue.sh`, started 25 Nov 2026 detached): order `islrtc-full, wslp, isl_fingerspelling,
mendeley_static, isl_csltr, mendeley_agri, mendeley_emergency, isl_csltr_words, hf_akrit_alphabet, hf_hemg_alphabet,
wslp_unlabelled`, then CISLR polling every 6 h for 14 days. Each source runs in 45-minute chunks with `--workers auto`
(1 worker while `ingest_isl.py include` or `islrtc` runs, 3 once both have finished), so the long INCLUDE/ISLRTC jobs
keep their CPUs. Only one queue runs at a time (`flock` on `data/isl/logs/queue.lock`; PID in `queue.pid`).

**Adapters** (`video_sources/adapters.py`, `VIDEO_ADAPTERS`): one `Clip` reader per manifest (quality filter
`n_frames >= 8` for videos, `>= 1` for images, `hands_frac >= 0.2`; unlabelled and `duplicate_of` rows skipped by
default). They are **not yet registered** in `sources.ADAPTERS` / `profiles.py` (owned by the model code): add the
commercial-safe ones (`isl_csltr`, `isl_csltr_words`, `mendeley_*`) to both profiles and the research ones (`wslp`,
`isl_fingerspelling`, `hf_*_alphabet`) to `research` only; `catalog.profile_sources()` returns these lists.

Open checks: CSLTR sentence-level glosses need alignment before they can label single signs; whether the mostly
hand-only emergency clips help a full-body webcam model is untested; the HF image sets are likely augmented duplicates
(SHA-256 dedupe only catches byte copies).

### 3.4 Pose-native sources (`scripts/ingest_isl_pose.py`) <!-- pose-sources -->

Owner: Data Engineer, 25 Nov 2026. Code: `backend/app/isl/data/pose_sources/` (`common` storage / quality / disk
guard, `posefmt` pose-format reader, `glosl`, `isl500`, `openhands`, `fingerspell`; the package is not called
`sources/` because `app/isl/data/sources.py` exists). Tests: `backend/tests/isl/test_pose_sources.py` (synthetic, no
network). These sources ship landmarks, so ingestion is a format conversion with no MediaPipe run (except the tiny
fingerspelling set) and little CPU; the jobs are network-bound.

```
nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py glosl --workers 8
nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py isl500 --workers 6
nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py openhands-raw --workers 12 --budget-gb 12
nice -n 10 .venv/bin/python scripts/ingest_isl_pose.py openhands-fs
.venv/bin/python scripts/ingest_isl_pose.py isl500-links   # ISL500 glosses -> vocabulary (derived/isl500_links.jsonl)
.venv/bin/python scripts/ingest_isl_pose.py status        # counts, frames, disk per source
```

Conventions shared with `scripts/ingest_isl.py`: manifest `manifests/<source>.jsonl` (id, source, gloss, path relative
to the repo, n_frames, fps, hands_frac, ok, error, secs, plus per-source fields), resumable (ok ids are skipped),
failures retried on the next run. Every job started from the committed code holds a lock (`manifests/<source>.lock`) so a
second copy exits (the `glosl` and `isl500` processes started at 14:45 UTC on 25 Nov ran pre-commit code and hold none:
never start a second copy while they run), and
stops cleanly (exit code 3) before free disk drops under 12 GB (`--min-free-gb`). Nothing is downloaded to disk except
small CSV/JSON files and the 28 MB fingerspelling zip; pose files are converted in memory.

**Storage.** Landmark arrays are saved as **float16** (`--dtype float32` keeps full precision) and strided to about
**15 fps** (`--target-fps`, same rule as INCLUDE: step = round(fps / 15); 25/29/30 fps -> step 2). `t_ms` stays
float64 and keeps the real timestamps. The `.npz` keys are the usual `RawSequence` keys (plus `storage`), so
`RawSequence.load` reads them; the adapters yield `PoseClip`s whose `load()` upcasts to float32. float16 error on a
real GloSL clip: at most 0.0009 (normalised units) on pose points, 0.0002 on hands. Measured size: 531 bytes per
stored frame (GloSL), about 360 bytes (OpenHands, no face).

| Source | Input (inspected 25 Nov 2026) | Conversion -> `RawSequence` | Parity check | `extractor_version` |
|---|---|---|---|---|
| `glosl` | `poses/<NNNN>/<id>.pose`, pose-format v0.2, 543 points (pose 33, face 468, hands 21+21), x/y pixels of the header size, **z raw**, missing = (0,0,0) conf 0, fps 25 (57,042 clips) / 29 (69,490) / 30 / 12 | `posefmt.holistic_pose_to_raw`: x/W, y/H, z copied (pose-format's Holistic writer does not scale z), visibility = confidence, a part is present in a frame when any of its points has confidence > 0, face = `FACE_IDX` of the 468 mesh points (same first 468 as our 478), `t_ms = i * 1000 / fps` | real clips: the left hand's wrist lies on pose point 15 (median 0.017 vs 0.33 for point 16), as in our INCLUDE extractions (0.011 vs 0.18), so no hand swap; z ranges (pose median -0.72, hands -0.03..0.004, face -0.04..0.08) are raw MediaPipe values like ours, not pixels. Most headers say 300x300; the hand-bone aspect estimate on 14 of them gives 0.85-1.07, i.e. square signer crops, and 0.82 / 0.85 on two files with real crop sizes 620x712 / 584x672 (0.87); kept as is, per-clip `aspect_est` in the manifest. No GloSL video is available to compare frame by frame (the card forbids using the clips) | `glosl-pose-format` |
| `isl500` | `Landmarks/MediaPipe/ISL_DATA_USER0NN/*.h5` (86,760 files, 2.5 GB): one dataset `intermediate` (T, 2, 21, 3) = MediaPipe **hands only** (index 0 = signer's left), 30 fps; `Landmarks/Pose/User0NN/[Round1/]*.npy` (about 21 GB): MMPose COCO-WholeBody (T, 133, 3) px + score | hands from the h5 as they are; pose 33 from WholeBody body/feet points mapped to MediaPipe indices (eyes inner/outer and mouth from the 68-point face, pinky/index/thumb from the WholeBody hand knuckles), z = 0, visibility = score, present when nose or a shoulder score >= 0.3; face `FACE_IDX` from iBUG-68 equivalents (z = 0); frame size from the two views of the same hand (WholeBody px / MediaPipe normalised) snapped to a standard size | our `HolisticExtractor` on the released MP4 `USER001/Absent__session106__clip014` (2560x1440, same 166 frames; errors in shoulder widths): hands 0.012-0.019 at lag 0, nose / shoulders / elbows 0.02-0.05, wrists 0.07-0.09 when the hand is visible (0.4 when off-screen: both models guess), face points 0.004-0.035; size estimate 2556x1442 -> 2560x1440. Close, not identical: pose z is 0 and some face points are approximations, so the version tag keeps the clips separable | `isl500-mphands+mmpose-wb` |
| `openhands_raw` | 5 zips (3.0 / 11.2 / 15.7 / 23.3 / 61.2 GB), each with one or two deflated HDF5 files; groups `keypoints/<chunk>` (T, 75, 3) float64 and `visibility/<chunk>` (T, 75), contiguous, in file order; 75 = pose 33 + left hand + right hand, normalised x/y, raw z, missing = zeros; no fps, no frame size, no face | `openhands.openhands_to_raw`: pose / hands copied, visibility as pose channel 4, face all missing, fps 25 (what OpenHands' own sampler assumes; 1-minute chunks hold 1,536 frames = 25.6 fps), width/height from the per-chunk hand-bone aspect snapped to 16:9 / 4:3 / 1:1 / 3:4 / 9:16 when within 20 % (Sign Library: estimates 1.5-1.8) | left hand on pose point 15 (median 0.018 vs 0.21), same convention as ours | `openhands-raw-mp-holistic` |
| `openhands_fs` | `Indian.zip`: MP4 + pickled poses | the pickles are never loaded (untrusted pickle = code execution); our own extractor runs on the 172 MP4s (`ingest_isl.extract_full`, 15 fps, float32) | exact (same extractor) | `holistic-v1` |

**Streaming the OpenHands archives** (`openhands.SeekableInflate`): the zip member is inflated from HTTP range
requests with parallel read-ahead (Zenodo gives about 0.8 MB/s per connection and sometimes stalls a connection
without closing it; 8 parallel 8 MB blocks gave 4-8 MB/s; a block slower than 250 KB/s after 15 s is retried).
While inflating, a `zlib` snapshot is kept every max(4 MB, size / 3000) of output so backward seeks restart nearby,
and 4 KB pages that look like HDF5 metadata (a block signature, or under 99 % plausible float64 words) are kept in
memory with their neighbours, plus the indexes of all-zero pages (Sign Library 1-minute file: 446 of 357,076 pages,
enough for h5py to list all 748 datasets without a restart; without the neighbours the 6.1 GB `sub_based` file
needed hundreds of restarts, which is why the first run was restarted at 14:57 UTC). Pass 1 lists
the datasets and their object-header addresses with h5py (one streaming pass; the list is cached in
`raw/openhands_raw/<member>.index.json`), pass 2 parses each small version-1 object header itself and reads the data
in address order. Network cost about twice the compressed size; no temp files; peak memory about 0.5 GB. Chunks of a
video already ingested from another file of the same channel are skipped. Sign Library ships two files: `1_min_splits`
(374 chunks of 76 named YouTube videos, 6.8 h at 25 fps, ingested first) and `sub_based` (9,582 chunks named by
hashes, about 28 h): the names cannot be matched, so up to 6.8 h may be ingested twice; deduplicate by pose content
before pre-training (open issue). Zips are processed smallest first until `landmarks/openhands_raw/` reaches `--budget-gb` (12 GB): at about
360 B per stored frame that is roughly Sign Library + NIOS + ISH News and part of MBM Vadodara (estimate).

**GloSL and iSign overlap.** The GloSL `id` is the iSign / ISLTranslate `uid`: 30,648 of the 31,221 public
ISLTranslate uids (`ISLTranslate.csv`, GitHub) are GloSL ids (iSign itself is still gated, so the rest is
unverified). The manifest keeps `isign_uid` and `in_isltranslate`; `derived/glosl_isign_uids.txt` lists all 126,846
ids. Any future iSign ingest must drop `isign/<uid>` for those uids (`glosl.isign_duplicates`), otherwise the same
poses enter pre-training twice. GloSL also ships `diagnostics/joint_train_exclude.csv` (flag `joint_train_exclude`).

**ISL500 details.** Ids `isl500/<userNNN>/<clip>`; some files carry an export-path prefix
(`_mnt_..._R2_Clips_R2_user001_Absent__session7__clip000`, `R2_..._user002_...`), stripped by `isl500.clip_key` (4 ids
collide after stripping: the first file wins). Gloss = file name before `__` without an `R2_` prefix (500 glosses).
`group = isl500-userNNN` (true signer ids: 15 signers) so splits can be signer-independent; the shipped
`PersonDependentTrain/Test.csv` split is kept as `pd_split`. 6,391 clips have no WholeBody file and are skipped
(hands only, no body to normalise). Hands are visible in only about 30 % of the raw frames (long rests), so each
record stores `active_start/active_end` (first..last hand frame +- 2) and the adapter slices to it.
Vocabulary links (`isl500-links`, vocabulary linker of `vocab/build.py`, 25 Nov 2026): 483 exact, 8 synonym-list
(Author, Cheap, Clean, Cry, House, Marry, Upset, Woman), 3 domain-qualified (Old, Rain, Tight), 6 unlinked (Fine,
Little: ambiguous; Photgraph: source typo; Wanted; Washface; Washhands). Plan if the hybrid features prove too
different: re-extract the 7,500 released MP4s (one per word and signer, about 21 MB each, 159 GB) one file at a time
with our extractor once the CPU is free (estimate 8-10 CPU hours).

**Jobs started 25 Nov 2026** (detached, `nice -n 10`, logs `data/isl/logs/ingest_pose_<source>.log`): `glosl`
(14:45 UTC, 8 download threads, 8.5-12.6 clips/s depending on machine load, ETA about 19:00 UTC), `isl500` (14:45,
6 threads, 4.5-5.8 clips/s, ETA about 20:10 UTC), `openhands-raw` (restarted 15:02 with 12 read-ahead after stream
fixes, 4-8 MB/s, budget 12 GB; Zenodo allows 133 requests per minute and answers 429 above that, which the reader
waits out; **killed by the kernel OOM killer at 15:53** while streaming NIOS because the read cache stopped evicting
after backward seeks, fixed in b4a29a2 and restarted at 16:26 UTC as PID 26183, which re-reads NIOS from the start
before reaching new chunks), `openhands-fs` (done 14:58: 172 of 172 clips ok). Rates measured on the first 3-7k
items; the measured status at 16:40 UTC is in section 3.5; every count must be refreshed when the jobs finish.

**Not done / open**: none of these sources is registered in `profiles.py` yet; no model has been trained on them, so
nothing here says anything about accuracy.

### 3.5 Measured status of every ingestion job (Reality Checker, 25 Nov 2026 16:37-17:07 UTC)

Measured from the processes (`ps`), manifest line counts taken every 5 minutes from 16:37 to 17:03, the logs and `df`,
not from the agents' reports. Load average 18-21 on 4 CPUs the whole time; memory group of the jobs at 12-14 of 14.3 GB
(6.0-6.7 GB resident, the rest page cache). No leaked deleted files in any job. Free disk under `data/isl`: 33.6 GB at 16:37, 32.3 GB
at 16:48, 31.3 GB at 17:03 (an INCLUDE zip download sits in `data/isl/tmp` while it is converted).

| Job (PID) | Done at 16:48 | Total | Rate (16:37-17:03) | Errors | ETA (UTC, estimate) |
|---|---|---|---|---|---|
| `ingest_isl.py include` (1632) | 2,895 rows, all ok | 4,292 | 6.7 videos/min | none | 20:00-21:30, 25 Nov |
| `ingest_isl.py islrtc` (6785) | 5,239 rows, all ok | 13,665 | **5.7 videos/min** (17/min this morning on a quieter machine) | none | **04:00-08:00, 26 Nov** (needs 12-17/min after the pose jobs end; the 00:20-01:00 ETA given earlier assumed the morning rate) |
| `ingest_isl_pose.py glosl` (21834, no lock) | 57,633 ok | 126,846 | 7.9 clips/s | none | about 19:20, 25 Nov |
| `ingest_isl_pose.py isl500` (21835, no lock) | 36,589 rows (about 4,260 failed by design: no WholeBody file; 1 frame-count mismatch) | 86,756 | 4.6 clips/s | by design only | about 19:50, 25 Nov |
| `ingest_isl_pose.py openhands-raw` (26183, lock) | 11,812 chunks (Sign Library 9,956 done; NIOS about 1,850 of 1,936) | until `landmarks/openhands_raw` = 12 GB (2.7 GB at 16:49) | 4.5 GB of landmarks per hour on NIOS, 0.7 GB/h on ISH News (from 17:00; 2.9 GB at 17:07) | OOM kill at 15:53 (fixed, see 3.4) | budget reached between about 20:00, 25 Nov and 08:00, 26 Nov (depends on Zenodo speed and chunk sizes) |
| `isl_queue.sh` (18382) -> `ingest_isl_video.py run wslp` | 619 rows (4 HTTP 429 fetch failures, retried next chunk) | 4,398 | **11.7 s/clip on 1 worker** (7.1 s measured earlier on the first items) | 429 only | WSLP about 03:00-07:00, 26 Nov (1 worker until INCLUDE and ISLRTC both end) |
| rest of the queue (fingerspelling, Mendeley static / CSLTR / agriculture / emergency, CSLTR words, HF alphabets, WSLP unlabelled) | a few probe rows each | see 3.3 | not started | - | about 6-10 h after WSLP on 3 workers: **about 14:00-18:00, 26 Nov** |
| `ingest_isl_pose.py openhands-fs` | 172 of 172 ok | 172 | - | none | done 14:58, 25 Nov |
| CISLR / iSign | gated (403) | - | - | - | the queue polls CISLR every 6 h for 14 days |

Projected disk: GloSL about 6.5 GB in total (3.0 GB at 16:50), ISL500 about 2 GB, OpenHands up to the 12 GB budget, ISLRTC
about 1.3 GB, WSLP about 0.6 GB, the queue's remaining sources about 1 GB: roughly 15 GB free when everything has
finished, above the 10 GB floor but with less margin than the 39-41 GB seen at 15:08. About 8 GB more can be freed
outside the repository (an old clean clone in `/tmp/rc_isl`, 2.5 GB, and agents' scratch virtualenvs, about 5.8 GB).


---

## 4. Vocabulary (`python -m app.isl.cli vocab`)

Built on 25 Nov 2026 from all 13,665 ISLRTC names + the 263 INCLUDE word folders (official lists), WordNet on,
without trim information (so the committed file depends only on names and lists):

| | count |
|---|---|
| ISLRTC files | 13,665 |
| ISLRTC concepts | 12,398 (1,086 with more than one indexable variant) |
| concepts in total | 12,442 (12,402 indexable; 263 in the core classifier = have INCLUDE clips) |
| name kinds | word 7,321, multiword 5,421, synonym list 393, phrase 349, explanation 122, example 19, number 32, compilation 8 |
| INCLUDE links | exact 206, spelling 3 (neighbour/neighbor ...), qualified 4, synonym-list member 6, new concept 44 |
| concepts needing review | 41 (all INCLUDE links other than exact, and words with `see_also` suggestions) |
| regional variants | Indore, Madhya Pradesh (`Over`, `Test`); Andhra Pradesh/Telangana; Bihar; Munger, Bihar; Gujarat; Hyderabad; Meghalaya; Mumbai; Odisha; Thiruvananthapuram, Kerala; West Bengal |

Rules (design 6.1 plus the deviations in design 5.8): variant markers `(Sign_N)`, `(N)` -> `sign_no`;
`- English` -> flag; `(Explanation)` / misspellings / `(Example)` -> not indexed; places from a gazetteer -> `region`;
other parentheses -> `domain` (own concept, e.g. `isl.light~electricity`); acronyms in parentheses matching the initials
-> synonym; `Numbers/12_Twelve` -> `isl.twelve` with `number_value`; synonym lists detected with WordNet (missed ones stay
`multiword`, which only affects search synonyms: the concept id is the same). WordNet synonyms, several domain
candidates and different senses are **never merged automatically**; they become `see_also` suggestions for review.
Manual fixes go to `backend/app/isl/vocab/overrides.yaml`.

---

## 5. Splits (`python -m app.isl.cli splits`)

* `include-official`: AI4Bharat lists. Path mapping: last three parts with `Extra/` removed, spaces = `_`. On the
  partial data every ingested clip (413) matched a list path; the other 3,879 list paths are reported as not ingested
  yet, none unparsable, none in two lists. **Not signer-independent**: on these 413 clips, 67 recording sessions
  appear in both train and test (31 train-val, 12 val-test).
* `include-session-cv5`: sessions = runs of camera file numbers of one word with gaps <= 3 (115 sessions so far).
* `include-pseudo-signer-cv5`: sessions are chained into recording passes (camera-number order, gap <= 30, no
  repeated word) and passes are clustered on scale-free body proportions (arm/hand lengths, face geometry over
  shoulder width). Partial data: 10 passes, 4 pseudo-signers, silhouette 0.48, but very uneven (198 / 161 / 42 / 12
  clips), so the report currently names `include-session-cv5` as primary. Re-run after ingestion.
* Leakage checks (group overlap, duplicates split across partitions, missing classes) pass for every session and
  pseudo-signer fold.
* `oneshot-concept-cv5`: 218 concepts have both INCLUDE clips and an indexable ISLRTC variant; 205 are usable (exact
  link, no review); 5 folds stratified by INCLUDE category.

## 6. Near-duplicates (`python -m app.isl.cli dedup`)

1,418 clips (INCLUDE + ISLRTC, 25 Nov 2026): one duplicate pair, `A_Line_Silhouette` / `A_Line_Skirt` (the dictionary
reuses one video for two entries). Duplicates are kept on the same side of every split.

---

## 7. ISLRTC first-sign trimming: quality

Current version **`trim-2`** (25 Nov 2026), `backend/app/isl/data/trim.py`; checks and tooling in
`trim_bench.py` (labels, sampling, synthetic checks, full run), `trim_sheet.py` (contact sheets), `trim_synth.py`
(splice / join helpers), `trim_learned.py` (optional learned end for `cap` clips). Labels:
`backend/app/isl/data/trim_labels_islrtc.jsonl`.

### 7.1 By-eye labels (ground truth)

167 random ISLRTC dictionary videos in three stratified samples (stop reason x name kind: one word, 2 words,
3+ words, synonym list; phrases/explanations excluded): seed 2026 (64 clips, alternating `dev`/`test`), seed 7
(63, all `test`) and seed 99 (40, `holdout`, labelled after the last rule change). Each video was downloaded,
rendered as a **blind** contact sheet (one tile per 200 ms, no cut shown, signer crop or full frame) and the
end of the first sign was written down as the first tile where the hands leave the sign. Videos were deleted
after rendering.

| status | bounded (`eof`/`card`) | `max_s+posegap` | `max_s` | all |
|---|---|---|---|---|
| `ok`: end clearly visible | 46 | 27 | 23 | 96 |
| `guess`: plausible end | 0 | 0 | 5 | 5 |
| `unclear`: no visible boundary | 13 | 0 | 52 | 65 |
| not a sign | 1 | 0 | 0 | 1 |

**65% of plain `max_s` clips (52 of 80) have no boundary a person can see on 5 fps frames**: the signer goes
from the sign straight into the explanation with the hands in signing space. These clips cannot be scored by
eye; section 7.2 measures them on synthetic data. The 13 unclear bounded clips are 6 clips cut mid-sign by a
false title-card stop and 7 `eof` clips with long continuous signing.

End error (trim end time - label), share within 300 ms, on the 96 `ok` clips:

| | dev (n=22) | test (n=56) | holdout (n=18) | all (n=96) |
|---|---|---|---|---|
| trim-1 | 81.8% | 94.6% | 94.4% | 91.7% |
| **trim-2**, main cut | 81.8% | 94.6% | 94.4% | **91.7%** (88/96) |
| trim-2, better of main and `second` cut | 90.9% | 98.2% | 94.4% | 95.8% |

With the 5 guesses: trim-2 89.1% (90/101). "Usable" (between 300 ms early and 600 ms late): trim-2 95.8%,
trim-1 94.8%. **On clips with a visible boundary both versions meet the 80% target; they differ on the
continuous-signing clips, which by-eye labels cannot score.** Remaining errors (8): a second signer after a
real title card (`Bermuda Shorts`, now `truncated`), one rest at the waist inside a compound (`Business
Minded`), one `cap` 1.1 s early, four ends 360-560 ms late and one compound 0.9 s late (`Advocate General`).

### 7.2 Synthetic checks (known boundary, no labels needed)

`python -m app.isl.data.trim_bench synthetic --half report`. Clips are split into two stable halves by
crc32 of the id; every rule and parameter was tuned on the `tune` half (and the `dev` labels) only. Numbers on
the `report` half, clip set pinned to the 4,619 clips ingested at 15:30 UTC:

* **spliced** (the hard case): an ISLRTC `max_s` clip whose sign ends in an unambiguous hands-down run has
  those frames cut out and bridged with 2 interpolated frames, so the same signer runs straight from the sign
  into the explanation. True end = last frame before the hands went down. n = 506.
* **joined**: a bounded clip (whole clip = sign + trailing rest) followed by 2-6 s of another signer's
  explanation moved into the same body frame. n = 743.

| within 300 ms | spliced | spliced, 1 word | 2 words | 3+ words | spliced `cap` ends | joined |
|---|---|---|---|---|---|---|
| trim-1 | 23.9% | 32.0% | 13.8% | 10.0% | 35.1% (n=322) | 59.9% |
| **trim-2** | **43.5%** | 48.0% | 41.3% | 24.0% | **65.8%** (n=322) | 65.8% |
| trim-2, better of 2 cuts | 62.1% | | | | | 75.4% |
| trim-2 + learned `cap` end (`trim_learned`, trained on `tune`) | 51.7% (n=520, stage 1 alone 43.8%) | | | | | |

The old synthetic number (67.7%, section history) used tails cut at the trimmer's own end and is not
comparable. **INCLUDE** (one sign per clip, 400 clips): trim-2 keeps 100% of the active frames on 100% of clips.

### 7.3 What changed in trim-2 and what was tried

Kept (each improved the tune half / dev labels and did not hurt the others):

* **`cap` = sign-length prior**: with no rest/dip/pause, the end is the best frame in [0.6, 1.8] x the expected
  length (2.2 s for one word, 2.8 s for multiword names; medians of 1,400 bounded clips), scored by a log-length
  prior plus hand height. trim-1 took the lowest hand in the last 40% of a 4 s window, i.e. usually 3-4 s.
  Spliced `cap` ends within 300 ms: 35% -> 66%.
* **two candidate cuts per entry**: each record has `second` (also in `alternatives`, so the existing
  training crops and the second index row pick it up) at least 400 ms from the main cut: the next rest/dip
  after a rest/dip cut (compounds that drop the hands between parts), the last inner rest/dip of a bounded
  clip, or the best prior frame; when the first rest/dip comes later than 1.3 x the expected length, the
  second cut comes from the prior instead. Best of two: 91.7% -> 95.8% on labels, 43.5% -> 62.1% spliced.
* **bounded clips keep long compounds**: the whole sign is kept up to 2 x the sign window (was 1.25 x);
  `Beekeeper`-type 4-5 s signs were cut at a pause.
* **`truncated` card clips**: the ingestion title-card detector (`_cut_at_card`, near-uniform left part of the
  frame) also fires on plain grey backgrounds. Of 10 `card` clips viewed, 6 stopped mid-sign (0.8-3.8 s into
  the video), 1 was a real card after a second signer, 3 were right. A `card` clip whose last frames still
  have a hand up is now `truncated` (confidence 0.1; 53 of the 71 card clips in the final run). **These should
  be re-extracted with
  `scripts/ingest_isl.py islrtc --only ... --full --max-s 14 --out-subdir islrtc_full`** (not done here: the
  ingestion jobs belong to the main session).
* **learned end for `cap` clips** (`trim_learned.py`, optional): the stage-2 BiGRU of `boundary.py` trained on
  same-signer spliced clips (not cross-signer joins). Spliced report half: 43.8% -> 51.7% main cut, 65.4% best
  of 2. It changes only `cap` ends; applied to every method it reached 70% on spliced data but lost on the
  labels (91.7% -> 90.6%), because real first rests/dips are mostly right. Train with
  `python -m app.isl.data.trim_learned train` (about 15 min on the shared 4 CPUs,
  `artifacts/isl/trim_boundary.pt`), apply with `trim_bench run --learned artifacts/isl/trim_boundary.pt`.
  The default `cli trim` does not use it.

Tried and dropped (measured, did not help):

* end at the onset of the hand drop instead of where the hands reach the waist: dev labels +9 pt, test -9 pt
  (people mark the first frame after the sign, where the hands are already dropping);
* shot / signer change detector (shoulder jump between frames): fixed 1 labelled clip, broke 1;
* longer `dip_ms` / `rest_ms`, larger minimum sign length per extra word (300-1000 ms), larger compound
  offsets (1.2-1.8 s), motion-minimum weight in the prior: no gain or worse on the tune half;
* the existing cross-signer stage-2 model (`cli trim --learned`): not re-trained here; the spliced model above
  replaces it for `cap` clips.

Number of words: about half of the labelled 2-3 word names are signed as **one** continuous sign (`Chop in Table
Tennis`, `Bank Account Number`), so the word count only shifts the expected length (+0.6 s), it does not set a
sign count. Synonym lists (`Almost Nearly`, `Customer Client`) are one sign.

### 7.4 Method mix and confidence (all ingested clips)

Run on all 4,832 ISLRTC clips ingested at 16:20 UTC (`trim_bench run --learned artifacts/isl/trim_boundary.pt`,
sidecar `derived/islrtc_trim.jsonl`, `trim_version` `trim-2`; `.npz` files only read): source_end 1,950, cap
1,341, dip 662, rest 653, full (phrases) 125, truncated 53, pause 31, clip_end 10, empty 7; 357 `cap` ends
taken from the learned model (trained on all ingested `max_s` clips, 3,984 spliced examples, 15 min CPU);
4,429 records have a `second` cut. `cap` clips keep 2.8 s (median; p10 1.8 s, p90 4.3 s) instead of 3.9 s.
`cap` is still 28% of clips: they are the continuous-signing videos, not a failure of the boundary rules.
Clips ingested later and trimmed by `python -m app.isl.cli trim` get stage 1 only (same `trim_version`);
re-run `trim_bench run --learned ...` after ingestion to refine them.
Confidence by method (unchanged): rest 0.9, source_end 0.8, dip 0.75, pause 0.7, cap 0.3, truncated 0.1.
Measured: rest 14/16 and source_end 68/72 on labels (all dip labels right, n=6); `cap` 66% on spliced data,
so 0.3 is conservative (kept: an existing data-layer test pins `cap <= 0.3`).

**Estimate for all dictionary clips** (not a measurement): weighting the strata by their share of the corpus
(bounded about 35%, `max_s+posegap` 7%, `max_s` 57%, of which about 65% continuous signing) with the label
accuracy for clips with a visible end and the spliced accuracy for continuous signing gives roughly **70-75%
within 300 ms for the main cut and about 80% for the better of the two cuts**. The design check in
ISL_DESIGN 5.3 (bounded `card`/`eof` clips within 300 ms for >= 80%) is met on the labels: 91.3% of 46
bounded clips (100% with the better of two cuts; `max_s+posegap` 96.3% of 27, `max_s` with a visible end 87.0%
of 23);
continuous signing remains the main error, and the second index row / training crop (`second`) is the
mitigation.

Review tooling: `python -m app.isl.data.trim_bench sheets SAMPLE.json OUT [--marked] [--full-frame]` renders
contact sheets (download one video at a time, deleted after rendering); `--marked` shows the kept range and
cuts. `python -m app.isl.cli trim-review` still draws the wrist-height plots.

---

## 8. Commands (from `backend/`)

```
python -m app.isl.cli status            # counts, trims, vocabulary, splits, gated access
python -m app.isl.cli wordnet           # once
python -m app.isl.cli vocab [--no-trims]
nice -n 10 python -m app.isl.cli trim [--redo] [--learned]
python -m app.isl.cli trim-review | trim-eval
python -m app.isl.data.trim_bench labels | synthetic --half report | include | run [--learned MODEL]
python -m app.isl.data.trim_learned train | eval --half report    # learned end for cap clips (section 7.3)
python -m app.isl.cli splits [--fetch] | dedup | access | profiles
nice -n 10 python -m app.isl.cli hindi --translate   # Hindi lexicon drafts (offline, CPU); then: hindi
python -m app.isl.cli hindi --merge-local            # fold reviewed local Hindi corrections into the tracked file
```

## 9. Open issues

* INCLUDE and ISLRTC ingestion are partial (25 Nov 16:40 UTC: 2,839 of 4,292 and 5,191 of 13,665; section 3.5);
  every count above must be refreshed after ingestion.
* Pseudo-signers are estimates; INCLUDE signer identities are not published in the manifest or the split lists.
* Trimming (trim-2): 91.7% within 300 ms on 96 by-eye labelled clips with a visible end, but 65% of plain `max_s`
  videos run straight from the sign into the explanation and cannot be labelled by eye; on synthetic versions of
  that case 44-52% (62-65% with the second cut). 53 `card` clips stopped mid-sign need a `--full` re-extraction.
* 41 vocabulary concepts need a human review (`review.status == "needs_review"`), plus the synonym-list detection.
* CISLR licence review (videos collected from third-party online sources) before any commercial use.
* iSign hand labelling (left/right) to be verified on real data.
* Hindi lexicon: curated rows await a native-speaker teacher; machine rows are unreviewed drafts (section 10).
* Pose-native sources (3.4) <!-- pose-sources -->: register them in `profiles.py` / `sources.ADAPTERS`; decide the
  OpenHands raw corpus (CC BY label on third-party YouTube content) and fingerspelling set before any commercial use;
  ISL500 signs were recorded in Goa and 14 of its 500 words link only through a synonym / domain rule or not at all
  (`derived/isl500_links.jsonl`), so check sign forms against ISLRTC before mixing them into one class; frame aspect
  of GloSL / OpenHands is a hand-bone estimate (about +-10 %); Sign Library's two OpenHands files may overlap by up
  to 6.8 h (hash-named chunks in `sub_based`), deduplicate by content before pre-training.
* Fixed in commit 514be97: `isign.pose_to_raw` used to divide z by the frame width; it now keeps raw MediaPipe z like
  pose-format's Holistic writer and our extractor. No iSign data exists yet (still gated), so nothing needs re-running.

---

## 10. Hindi lexicon (`python -m app.isl.cli hindi`)

`models/packs/isl/lexicon_hi.jsonl` (tracked, one row per vocabulary concept) and `models/packs/asl-wlasl100/lexicon_hi.jsonl`
(the 100 WLASL glosses). Row: `{"concept_id", "english", "hindi", "hi_alt", "source", "reviewed", "review", "model"}`.

| `source` | `review` | Where it comes from |
|---|---|---|
| `manual` | `curated` | hand-written core list `backend/app/isl/text/data/hindi_core.tsv` (greetings, family, numbers, days, months, colours, food, places incl. Indore/MP, school, body, medical/emergency, emotions, questions, common verbs/adjectives, all 263 INCLUDE words). Written and checked by the AI engineer, **not yet confirmed by a native-speaker teacher**. |
| `rule` | `curated` | numbers from the rule table (`numbers_hi.py`, Indian system: सौ, हज़ार, लाख, करोड़) |
| `model` | `machine` | IndicTrans2 draft (beam 4, CPU). Input carries the sense: "to <verb>", "a <noun>", "<word> (<domain>)"; each member of a synonym list is translated and the majority wins |
| `manual` | `reviewed` | a Hindi speaker confirmed it (local correction in the app, folded in with `hindi --merge-local`); `reviewed: true` only here |

Status 25 Nov 2026: 12,442 of 12,442 ISL concepts have Hindi (870 curated, 20 rule, 11,552 machine, 0 reviewed);
ASL 100 of 100. An AI spot check of 100 random machine word rows found about 89 acceptable, 7 clearly wrong, 4
doubtful (not a native-speaker review).

Steps: `nice -n 10 python -m app.isl.cli hindi --translate` (downloads and verifies the model into
`artifacts/hf/`, translates what is not in `data/isl/derived/hindi_mt_cache.jsonl`, resumable; about 40 min on 2 threads
while ingestion runs) and then `python -m app.isl.cli hindi` (rebuild from the cache, seconds). Reviewed rows are never
overwritten by a rebuild. Extra Python packages for `--translate` only: `backend/requirements-hindi.txt`.
