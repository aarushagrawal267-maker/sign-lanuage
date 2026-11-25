# Indian Sign Language (ISL) datasets: sources, licences and data pipeline

Owner of this file: Data Engineer (docs/ISL_DESIGN.md section 10.2). Status: 25 Nov 2026, phase 1 (ingestion
running, partial data). Every number below says when and on how much data it was measured.

---

## 1. Sources and licences

| Source | What | Licence | Commercial use | Profile | Access |
|---|---|---|---|---|---|
| **INCLUDE** (AI4Bharat), zenodo record 4010759 | 263 ISL words, 4,292 videos, several signers (Chennai), 15 categories | CC BY 4.0 (zenodo metadata) | yes, with attribution | commercial-safe + research | public |
| INCLUDE split lists (AI4Bharat INCLUDE code repository, `train_test_paths/`) | official train/val/test file lists (3,127 / 348 / 817) | MIT (repository `LICENSE`, checked 25 Nov 2026); downloaded at run time, not redistributed | yes | both | public |
| **ISLRTC Indian Sign Language Dictionary** (Indian Sign Language Research and Training Centre, Government of India, data.gov.in) | 13,665 videos: about 12,400 dictionary concepts, numbers, phrases; one video per entry, several different signers across entries | Government Open Data Licence - India (GODL-India) | yes, with attribution | both | public |
| ISLRTC Hugging Face re-encode (`silentone0725/Indian_Sign_Language_Data.gov_Rencoded`) | HEVC mirror of the above, used for downloading | labelled MIT by the uploader | mirror only: does not replace the GODL attribution | both | public |
| **CISLR** (Exploration-Lab) | about 4,700 ISL words, 7,050 videos collected from online ISL videos | AFL-3.0 on the dataset card | **pending review**: AFL-3.0 allows commercial use, but the videos come from third-party online sources whose rights the licence may not cover | research only until reviewed | gated (`pending` on 25 Nov 2026) |
| **iSign** (Exploration-Lab) | 118k ISL video - English sentence pairs; MediaPipe Holistic poses in pose-format | CC BY-NC-SA 4.0; the access form adds "research use only, no commercial use, no redistribution" | **no** | research only (pre-training) | gated (`pending` on 25 Nov 2026) |
| WLASL (existing ASL model) | ASL, 100 words | C-UDA 1.0 | **no** | research only (warm start) | see `docs/LICENSING.md` |
| WordNet 3.0 (via NLTK data) | synonym lists, part of speech, coarse categories | WordNet licence (permissive) | yes | both | public |
| MediaPipe Holistic | landmark extractor | Apache-2.0 | yes | both | bundled |
| pose-format (only for reading iSign; our reader is self-contained) | `.pose` file layout | MIT | yes | research | optional |
| User sign packs ("My signs") | user recordings as landmarks | chosen by the author at export (default CC BY 4.0) | per pack | never in global training | local |

### Required attribution (copy into the app's About / Research page and any publication)

* "Indian Sign Language Research and Training Centre (ISLRTC), Government of India - Indian Sign Language
  Dictionary, published on data.gov.in under the Government Open Data Licence - India (GODL)."
* "INCLUDE dataset: A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. INCLUDE: A Large Scale Dataset for Indian Sign
  Language Recognition. ACM Multimedia 2020. CC BY 4.0, https://zenodo.org/records/4010759."
* Research profile only: "CISLR: A. Joshi et al., EMNLP 2022" and "iSign: A. Joshi et al., 2024 (arXiv 2407.05404),
  CC BY-NC-SA 4.0". iSign-derived data (landmarks, models pre-trained on it) inherit NC and share-alike terms.

### Data profiles (`backend/app/isl/profiles.py`)

| Profile | Training / index / calibration sources | Pre-training only | Commercial |
|---|---|---|---|
| `commercial-safe` (default) | INCLUDE, ISLRTC | none | yes |
| `research` | INCLUDE, ISLRTC, CISLR, iSign | WLASL, iSign | no |

User data never enters a global model in either profile. The profile name is written into every derived artifact.

---

## 2. What is on disk (all under `data/isl/`, git-ignored)

| Path | Written by | Content |
|---|---|---|
| `manifests/include.jsonl` | `scripts/ingest_isl.py include` (running) | one row per INCLUDE video: id, gloss, category, zip, member, path, n_frames, fps, hands_frac, ok, error |
| `manifests/islrtc_index.json` | `scripts/ingest_isl.py islrtc` | the 13,665 dictionary file names and sizes |
| `manifests/islrtc.jsonl` | `scripts/ingest_isl.py islrtc` (running) | one row per dictionary video: + stop (card / eof / max_s / +posegap), duration_s (approximate; `t_ms` in the npz is authoritative) |
| `landmarks/{include,islrtc}/*.npz` | ingestion | `RawSequence` (pose 33x4, hands 21x3, face 20x3, t_ms; NaN = not detected). Videos are deleted after extraction |
| `derived/islrtc_trim.jsonl` | `python -m app.isl.cli trim` | first-sign frame range per ISLRTC clip (sidecar; npz untouched) |
| `derived/vocab_report.json`, `derived/duplicates.jsonl` | `vocab`, `dedup` | build report, near-duplicate clusters |
| `splits/official/*.txt` | `splits --fetch` | AI4Bharat lists (SHA-256 pinned in `splits.py`) |
| `splits/*.json`, `splits/include_groups.json`, `splits/oneshot_concepts.json`, `splits/splits_report.json` | `splits` | split assignments with leakage reports, session / pseudo-signer groups, one-shot folds |
| `raw/{cislr,isign}/*.csv` | `scripts/ingest_isl_gated.py` | the gated datasets' small CSV files |
| `nltk_data/` | `python -m app.isl.cli wordnet` | WordNet (10 MB) |

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
  `POSE_LANDMARKS` (33, pixels) -> `pose` x/W, y/H, z/W, confidence as visibility; `LEFT_HAND_LANDMARKS` /
  `RIGHT_HAND_LANDMARKS` (21) -> `left` / `right` (confidence 0 -> missing frame); `FACE_LANDMARKS` (468/478) ->
  the app's 20-point `FACE_IDX` subset; `POSE_WORLD_LANDMARKS` ignored; `t_ms` from the file fps;
  `extractor_version = "isign-pose-format"`. **Open check once data arrives**: whether the older Holistic used by
  iSign labels hands from the camera's view (then run with `--swap-hands`); compare one iSign clip with our own
  extraction of the same video before pre-training.

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

Stage 1 on 1,010 ISLRTC clips (25 Nov 2026): source_end 331, cap 346, dip 158, rest 130, full (phrases) 31,
pause 6, clip_end 5, empty 3. `cap` (confidence 0.3) means no boundary was found within the sign window: the signer
went straight on into the explanation.

How often is it right? There is no ground truth, so three checks:

1. **Video check (by eye)**: the dictionary videos of 12 random clips (seed 29, 3 per method) were downloaded and
   shown as 4 fps contact sheets with the cut marked, next to the wrist-height plots. Correct 7, wrong 2 (both compound
   names, "Amniotic fluid" and "Adobe Creative Suite", cut after the first part), unclear 2 (continuous signing, no
   visible boundary), 1 explanation entry (not indexed). That is **7 of 9 judgeable (about 78%)**, a small sample.
   Earlier samples (24 clips) drove the rule changes listed in design 5.8 and are not counted.
2. **Synthetic end error**: 331 clean bounded clips each followed by 2-6 s of another clip's explanation, aligned to
   one body frame, trimmed as if `stop=max_s`: **67.7% within 300 ms** (design target 80%), median error 80 ms,
   18.7% too late, 13.6% too early.
3. **INCLUDE**: on 400 INCLUDE clips (one sign each) the trimmer keeps 100% of the active frames.

Stage 2 (learned boundary, smoke run on partial data: 698 clean signs, 112 tails, 10 epochs, ~1 min CPU): 88.7% of
synthetic validation joins within 2 frames, but on the synthetic end-error check it scored 71.1% against 69.3% for
stage 1 at that moment (about 270 clips), with leakage (the same clips in training and check). Not used by default.

Review sheet: `python -m app.isl.cli trim-review` writes `artifacts/reports/trim_review/*.png` + `index.md`.
Known failure modes: compound multiword names in `max_s` clips are cut after their first part; continuous signing
without any drop of the hands ends at the 4 s cap. Mitigations already in the design: alternatives as extra training
crops and a second index row per entry, low trim confidence lowers the weight of these references.

---

## 8. Commands (from `backend/`)

```
python -m app.isl.cli status            # counts, trims, vocabulary, splits, gated access
python -m app.isl.cli wordnet           # once
python -m app.isl.cli vocab [--no-trims]
nice -n 10 python -m app.isl.cli trim [--redo] [--learned]
python -m app.isl.cli trim-review | trim-eval
python -m app.isl.cli splits [--fetch] | dedup | access | profiles
```

## 9. Open issues

* INCLUDE and ISLRTC ingestion are partial (about 10% and 8%); every count above must be refreshed after ingestion.
* Pseudo-signers are estimates; INCLUDE signer identities are not published in the manifest or the split lists.
* Trimming accuracy is about 70-78% by two rough measures; compound signs are the main error.
* 41 vocabulary concepts need a human review (`review.status == "needs_review"`), plus the synonym-list detection.
* CISLR licence review (videos collected from third-party online sources) before any commercial use.
* iSign hand labelling (left/right) to be verified on real data.
