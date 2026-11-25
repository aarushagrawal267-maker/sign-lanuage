# ISL dataset and resource survey

Survey date: 25 Nov 2026. Scope: every public (or requestable) resource that could improve
the Indian Sign Language (ISL) word recogniser, beyond the four sources already planned
(INCLUDE, ISLRTC dictionary, CISLR, iSign). Those four are listed briefly at the end for
completeness, with corrections found during this survey.

Everything marked **verified** was checked from this machine on the survey date: an HF API
call, a Mendeley or Zenodo API call, a file-list read, or a page fetch. Everything marked
**reported** comes from a web page or search snippet that could not be opened
directly, so check it before you rely on it. Nothing here was invented. If a fact could not
be found, the table says "not stated".

---

## 1. Search boundaries (what was and was not searched)

| Source | Method | Queries |
|---|---|---|
| Hugging Face datasets | `https://huggingface.co/api/datasets?search=` + per-repo API, tree and README fetch | indian sign language, ISL, indian-sign, sign language india, CISLR, iSign, hindi sign, sign language, fingerspelling, wlasl, how2sign, asl, csl, lsfb, autsl, popsign, youtube-sl, slovo, asl-citizen, islr, isl-, isl_, holistic, sign pose, and others; authors Exploration-Lab, pedroodb, ai4bharat |
| Mendeley Data | `data.mendeley.com/api/research-data/search?query=` (5 queries x 5 pages) + `public-api/datasets/<id>` | indian sign language, ISL sign, hindi sign language, ISL gesture, indian sign |
| Zenodo | `zenodo.org/api/records?q=` (resource_type=dataset, 2 pages each) | "indian sign language", ISL sign language dataset, "sign language" hindi, isolated sign language dataset, sign language pose keypoints dataset |
| IEEE DataPort | page fetch of 3 listings found by web search | Indian Sign Language |
| Kaggle | **web search only**. No Kaggle API key on this machine, and Kaggle pages render with JavaScript, so nothing on Kaggle could be verified | Indian Sign Language dataset ISL words video landmarks |
| GitHub | `raw.githubusercontent.com` only (api.github.com and github.com return 403 through the proxy) | AI4Bharat/INCLUDE, AI4Bharat/OpenHands, Exploration-Lab/ISLTranslate, Exploration-Lab/CISLR, hukenovs/slovo |
| Project / government sites | page fetch | cs.rkmvu.ac.in/~isl (FDMSE-ISL), islrtc.nic.in (FAQ, directories), ciet.ncert.gov.in/sign, openhands.ai4bharat.org |
| General web | WebSearch (US index) | Indore and MP deaf organisations, ISLRTC Hindi dictionary, YouTube-SL-25, ASL Citizen, Google ISLR |

**Date range:** no lower bound; newest item found is from Nov 2026.
**Language:** English-language pages only. No Hindi-language search was done, which is a known gap (see section 9).
**Inclusion:** the resource contains ISL signs (any modality), or it is a large multilingual
sign corpus usable to pre-train a pose/landmark encoder, or it is an organisation that could
supply consented regional (Indore/MP) data.
**Exclusion:** empty HF repos (README only, no data), traffic-sign and other false positives,
Israeli Sign Language (also abbreviated "ISL", e.g. `tal1569/unisign-isl`), duplicate mirrors (noted, not counted
twice), and model repos.

**Screening counts (approximate):** about 300 HF repo hits screened, 26 opened, 17 included.
11 Mendeley ISL records, all opened, 11 included. About 55 Zenodo sign records screened, 8
opened, 7 included. 3 IEEE DataPort listings included (reported, subscription-gated). 10 Kaggle
listings recorded (unverified).

---

## 2. Legend

- **Licence class:** `C` = commercial use allowed (CC BY, CC0, MIT, Apache). `NC` = non-commercial or research-only.
  `?` = no licence stated, or the statements conflict. Treat `?` as NC until the authors confirm.
- **Access:** `direct` = anonymous HTTP download works from this machine (verified).
  `HF-gated` = HF auto-gate; returns 403 until the owner's account accepts.
  `request` = email or form to the authors. `subscription` = IEEE DataPort paid login.
  `kaggle-key` = needs a Kaggle API token and acceptance of the dataset or competition rules.
- **Usefulness:** high / med / low for *this* app: live webcam, MediaPipe Holistic landmarks,
  isolated ISL words, ISLRTC standard, many words, signer-independent.

---

## 3. Top 10 recommended new sources (ranked)

| # | Source | Why | Licence | Access |
|---|---|---|---|---|
| 1 | **FDMSE-ISL** (RKMVERI): 2,002 words, 40,033 videos, 20 deaf native signers, signer-independent split. https://cs.rkmvu.ac.in/~isl/dataset.html | Largest multi-signer isolated ISL set. It would give about 20 signers per word for 2,002 words. That is what turns one-signer dictionary coverage into real accuracy. | Research use on request (`NC`) | `request` (form linked from the page; contacts listed on https://cs.rkmvu.ac.in/~isl/islr/) |
| 2 | **ISL500 / ISL-DATA** (HF): 500 words x 15 signers, about 7,500 videos + MediaPipe `.h5` landmarks. https://huggingface.co/datasets/ISL500/ISL-DATA | Multi-signer, landmarks already extracted (about 34 MB per user), and it can be downloaded today. Signer IDs are in the paths, so a signer-independent split can be built. | "Research and academic use only" (`NC`) | `direct` (ungated, verified) |
| 3 | **OpenHands raw ISL pose corpus**: about 1,129 h of unlabelled ISL (ISH News, MBM Vadodara, NewzHook, NIOS, Sign Library) as MediaPipe Holistic poses. https://zenodo.org/records/5558897 | Self-supervised pre-training of the encoder on real Indian signing from many signers. Paper reports gains in low-resource fine-tuning. | Zenodo record says CC BY 4.0 (`C`?); poses derived from third-party YouTube channels, see note in section 4 | `direct` (Zenodo; HTTP range verified, so it can be streamed per zip member) |
| 4 | **WSLP 2025 shared task, Word Recognition**: 4,398 train clips covering 4,361 distinct ISL words + 270 MB test. https://huggingface.co/datasets/Exploration-Lab/WSLP | Ungated CISLR-style one-clip-per-word ISL vocabulary, usable **now** while the CISLR gate is pending. | CC BY-NC-ND 4.0 (`NC`, no-derivatives) | `direct` (ungated, verified; remote-zip listing verified) |
| 5 | **GloSL-ISL** (iSign/ISLTranslate poses re-packaged): 126,846 clips, 269 h, per-clip MediaPipe Holistic `.pose` files, video-disjoint splits. https://huggingface.co/datasets/pedroodb/glosl-isl | The iSign pose data without the 170 GB split archive. Files are per clip (`poses/<NNNN>/<id>.pose`), so they can be streamed one by one. Card records the iSign author's permission (8 Oct 2026) to include pose files. | CC BY-NC 4.0 (`NC`) | `direct` (ungated, verified) |
| 6 | **ISL-CSLTR**: 100 sentences x 7 signers, 700 videos + 1,036 word-level images. https://data.mendeley.com/datasets/kcmpdxky7p/1 | Commercial-safe continuous ISL. Gives transitions, non-sign "rest" segments for the "not sure" class, and a sentence test set. | CC BY 4.0 (`C`) | `direct` (single 8.9 GB zip; Mendeley S3 supports Range, verified) |
| 7 | **INCLUDE official splits + OpenHands INCLUDE poses**. https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/ and https://zenodo.org/records/6674324 | The official train/val/test path lists (3,126 / 347 / 816 for INCLUDE; 688 / 76 / 191 for INCLUDE-50) are needed by the running ingestion. They are on GitHub, not in the Zenodo zips. | CC BY 4.0 (`C`) | `direct` (verified) |
| 8 | **Mendeley ISL word-video sets** (Central Univ. of Kerala): emergency words (8 words, 26 signers) https://data.mendeley.com/datasets/2vfdm42337/1, and agricultural words (13 words) https://data.mendeley.com/datasets/6v53kfxf46/1 | Small, but commercial-safe and multi-signer. "Help", "doctor", "pain", "accident" and "thief" are high-stakes words to get right. | CC BY 4.0 (`C`) | `direct` |
| 9 | **YouTube-SL-25 ISL slice**: about 209 h, 3,023 videos, 8 channels. https://github.com/google-research/google-research/tree/master/youtube_sl_25 | More unlabelled ISL for pre-training (overlaps OpenHands channels). It is only a list of video IDs, so you have to download and extract the videos yourself. | IDs CC BY 4.0 (reported); videos belong to their channel owners (`?`) | video-ID list `direct`; videos need YouTube download (not attempted) |
| 10 | **Indore partner organisations for consented data**: Anand Service Society (https://anandservicesociety.org), Mook Badhir Sangathan / Indore Deaf Bilingual Academy, ISLRTC's MP deaf-teacher directory | The **only** route to Indore/MP regional variants. No public Indore/MP ISL dataset exists (section 7). | You set the terms: use a consent form that allows commercial use | partnership |

Runners-up: ISLVT (sentences, CC BY 4.0), ISL continuous fingerspelling (CC BY-NC), Mendeley
static ISL alphabet with Hindi vowels (CC BY 4.0), Google ISLR and PopSign (ASL, for encoder
pre-training in the commercial-safe config).

---

## 4. Licence findings that change earlier assumptions (read first)

1. **ISLRTC dictionary is not unconditionally commercial.** ISLRTC's own FAQ (Q.23,
   https://islrtc.nic.in/faq/, verified) says the dictionary may be used for research, teaching and
   "development of ISL-related technology" on two conditions. The data must "not be resold or used for any
   profiteering purposes", and ISLRTC / DEPwD / MoSJE must be acknowledged. The data.gov.in GODL-India
   listing does not override the publisher's stated conditions in practice. The HF mirror's "MIT" licence
   (`silentone0725/...`, and duplicates `Vignesh3816/...`, `suneetpaul/...`) is the uploader's
   claim and has no authority. **Recommendation:** a free, non-profit app with acknowledgement fits the
   FAQ. Any paid or ad-funded use needs written permission from ISLRTC. Record this in `docs/DATASETS_ISL.md`.
2. **CISLR is non-commercial.** The HF metadata says `afl-3.0`, but both the HF README and the GitHub README
   say "CC BY-NC 4.0 ... do not use our dataset for any commercial purposes" (verified). Treat it as `NC`
   and keep it out of the commercial-safe config.
3. **iSign** is `cc-by-nc-sa-4.0` on HF (verified) and "free for research use but not for commercial
   purposes". Two ungated third-party copies exist. `cpajitha/iSign_videos` is an unlicensed re-upload of
   video part `ab` (25.7 GB), so do not use it. `pedroodb/glosl-isl` holds poses only and records author
   permission for the poses. The licence stays NC either way.
4. **OpenHands raw ISL poses** (Zenodo 5558897) are labelled CC BY 4.0 by AI4Bharat. They are derived from
   videos owned by YouTube channels (ISH News, NewzHook, NIOS, MBM Vadodara, Sign Library), and the
   channels did not license them. Landmark coordinates are not the video, but the legal position is
   untested. Keep them in the research config. Use them in the commercial config only after the owner's
   lead accepts the risk.
5. **Silent Voice** (IEEE DataPort, INCLUDE-derived landmarks) shows CC BY 4.0 on the page but
   CC BY-NC-SA 4.0 in its description. It is redundant anyway, because we extract INCLUDE ourselves.
6. **Microsoft ASL Citizen** is non-commercial, with no redistribution of the data or modifications
   (https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/, reported).

---

## 5. Catalogue by category

### 5A. Isolated-word ISL (video or landmarks)

| Dataset | URL | Size | Signs / videos / signers | Format | Licence | Access | Use | Reason |
|---|---|---|---|---|---|---|---|---|
| FDMSE-ISL (RKMVERI Belur + FDMSE Coimbatore; Patra et al., Pattern Anal. Appl. 2025) | https://cs.rkmvu.ac.in/~isl/dataset.html ; paper https://arxiv.org/abs/2407.14224 ; code https://github.com/suvajit-patra/sl-hwgat | about 36 h frontal 60 fps; size not stated | 2,002 words / 40,033 videos / 20 deaf native signers (10 F, 10 M, southern India); 57 categories; split 10/2/8 signers (20,016 / 4,003 / 16,014) | 3 RGB views 1080p60 + Kinect RGB-D; green screen | Research use, on request (`NC`) | `request`. The request page `isl-dataset/` returned 503 on the survey date. Contacts: suvajit.patra.cs20(at)gm.rkmvu.ac.in, soumitra.samanta(at)gm.rkmvu.ac.in (from their public demo page) | **high** | Best multi-signer coverage. Caveat: vocabulary comes from the FDMSE dictionary, so some signs may differ from ISLRTC/MP forms; map glosses and flag variants. They also run a live 2,002-word ISLR demo (https://cs.rkmvu.ac.in/~isl/islr/), a useful benchmark. |
| ISL500 / ISL-DATA | https://huggingface.co/datasets/ISL500/ISL-DATA | 159 GB total; MediaPipe landmarks about 34 MB/user (about 0.5 GB for all 15) | 500 words / about 7,500 videos / 15 users (1 video per word per user; more sessions in places) | MP4 + MediaPipe `.h5` + MMPose `.npy`; `PersonDependentTrain/Test.csv` | "Research and academic use only; contact authors for commercial" (`NC`); no SPDX licence | `direct` (ungated, verified) | **high** | 15 signers per word, downloadable now. Paths say `ISL_Goa_Data`, so likely recorded in Goa. Origin, authors and sign standard are undocumented, so check the glosses against ISLRTC. Build a signer-held-out split, because the shipped split is person-dependent. Prefer re-extracting from MP4 with our own extractor for feature parity, or validate the `.h5` layout first. |
| WSLP 2025 shared task, Word Recognition (Exploration-Lab) | https://huggingface.co/datasets/Exploration-Lab/WSLP (`Shared_task_WR/`) | train.zip 5.0 GB, test.zip 0.27 GB | 4,398 clips / 4,361 distinct words (about 1 per word); signers not stated | MP4 + CSV (`uid,word`) | CC BY-NC-ND 4.0 (`NC`) | `direct` (ungated; remote zip listing via HTTP Range verified) | **high** (research config) | Same style as CISLR, available today. ND means do not redistribute derived data; internal training for research is the intended use. Expect heavy overlap with CISLR, so dedupe by gloss and video hash when CISLR arrives. |
| Vāksetu ISL Feature Dataset v1.0 (Goa College of Engineering) | https://zenodo.org/records/23004815 | 2.2 GB HDF5 | 300 classes; 89,883 main samples (only 49,081 unique arrays); mostly 1 signer (author) via webcam; 920 from INCLUDE; 3,915 negative "reject" samples in 23 categories | fixed (20, 506) engineered features, not raw landmarks | CC BY 4.0 (`C`) | `direct` | **low-med** | Its features are incompatible with our RawSequence. Its **negative/reject category design** (transitions, idle, incomplete signs, other people) is a good template for the "not sure" class. Also an example of a deployment-feedback adapter. |
| Mendeley: ISL emergency words (Central Univ. of Kerala, 2021) | https://data.mendeley.com/datasets/2vfdm42337/1 | 3.4 GB (raw + cropped zips) | 8 words (accident, call, doctor, help, hot, lose, pain, thief) / 2 per signer / 26 signers | AVI in zips | CC BY 4.0 (`C`) | `direct` | **med** | Commercial-safe, 26 signers on safety-critical words. A related Data in Brief article (doi 10.1016/j.dib.2020.106016) reports 824 videos, so check whether it is a different version. |
| Mendeley: ISL agricultural words (Central Univ. of Kerala, 2021) | https://data.mendeley.com/datasets/6v53kfxf46/1 | 7.9 GB, about 450 AVI | 13 words (Animal, Banana, Coconut, Cut, Fire, Help, Insect, Monkey, Rat, Rice, Snake, Tiger, Wheat); signers not stated | AVI (multiple crops per clip) | CC BY 4.0 (`C`) | `direct` (per-file URLs) | **med-low** | Commercial-safe extra signers for common words. |
| Kaggle: "Indian Dynamic sign Language dataset" | https://www.kaggle.com/datasets/pratikshadewarde/indian-dynamic-sign-language-dataset | not verified | reported 11 classes / 4,224 videos | video | not verified | `kaggle-key` | low | Small vocabulary; licence unknown. |
| Kaggle: "Indian Sign Language words with Landmarks" | https://www.kaggle.com/datasets/kaushikyh/indian-sign-language-words-with-landmarks | not verified | INCLUDE-derived | landmarks | not verified | `kaggle-key` | low | Redundant with our own INCLUDE extraction. |
| Kaggle: "Indian Sign Language Video Dataset" (prasadshet) | https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset | not verified | not verified (reported: includes MediaPipe Holistic landmarks) | video + landmarks | not verified | `kaggle-key` | unknown | Check when a Kaggle key exists. |
| IEEE DataPort: ISL-50 (Maheshwari & Sinhal, 2026) | https://ieee-dataport.org/documents/indian-sign-language-dataset-isl-50 (doi 10.21227/s3r7-9t93) | 22.8 GB | 50 words / 800 clips (16 per class) / signers not stated; split 480/160/160 | 4K MP4 | CC BY 4.0 (`C`) | `subscription` | low-med | Commercial licence, but small and paywalled. |
| danielelvs/multilingual-islr-mediapipe (INCLUDE-50 subset) | https://huggingface.co/datasets/danielelvs/multilingual-islr-mediapipe | 26.6 GB (14 GB frames.csv) | INCLUDE-50 958 samples + KSL, MINDS-Libras, LIBRAS-UFOP | MediaPipe Holistic 543 x 3 per frame, CSV | MIT on the repo; sources keep their own licences | `direct` | low | Redundant for ISL. Its INCLUDE-50 split is copied from the official one. |
| Mendeley: "Indian Sign Language_Dataset" (2024) | https://data.mendeley.com/datasets/yx7kdssfjp/1 | 281 MB | not stated (generic description) | zip, video + MediaPipe (reported) | CC BY 4.0 (`C`) | `direct` | low | Description is boilerplate and contents unverified; inspect before use. |
| Mendeley: ISL real-life words (static) | https://data.mendeley.com/datasets/s6kgb6r3ss/2 | 529 MB | 20 words, 18,000 images, 8 signers | JPG (static only) | CC BY 4.0 (`C`) | `direct` | low | Static images of signs that are dynamic in ISLRTC. |
| Mendeley: ISL depth data (week days) | https://data.mendeley.com/datasets/smt3f5bkj8/1 | 6.9 GB | 4 day-signs present (Fri, Mon, Sat, Sun), 160 each | RealSense depth CSV | CC BY 4.0 (`C`) | `direct` | low | Depth only; incompatible modality. |

### 5B. Fingerspelling, alphabet and numbers (ISL uses a two-handed manual alphabet)

| Dataset | URL | Size | Content | Format | Licence | Access | Use | Reason |
|---|---|---|---|---|---|---|---|---|
| ISL continuous fingerspelling (WSLP @ AACL-IJCNLP 2025) | https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling (also `salik2610/ISL-Fingerspelling`) | 1.8 GB | 1,308 segments, 70.85 min, about 14.7k characters with per-letter frame boundaries; from YouTube | MP4 + CSV/JSON letter timestamps | CC BY-NC 4.0 (`NC`) | `direct` | **med-high** (research) | The only real-world, letter-timed ISL fingerspelling set. Enables spelling out names and places that are not in the vocabulary, a key fallback for "any word". |
| Mendeley: static ISL English alphabet + 8 Hindi vowels + numerals | https://data.mendeley.com/datasets/7tsw22y96w/1 | 6.6 GB, 102,470 images | A-Z (two variants of E and 9), Hindi vowels, 0-9; kids/teen/adult signers; varied backgrounds | JPG | CC BY 4.0 (`C`) | `direct` (about 100k individual files, so slow) | **med** | Commercial-safe; Hindi vowels; age diversity. Static frames only. |
| Mendeley: multimodal ISL English + Marathi alphabet + digits (2026) | https://data.mendeley.com/datasets/2tj6c7pzfk/2 | 1.3 GB | 79 classes (26 English, 43 Marathi/Devanagari, 10 digits); 4 participants; 1,580 images per modality | RGB + MediaPipe overlay + skin mask | CC BY 4.0 (`C`) | `direct` | low-med | Devanagari letters (shared with Hindi). Small. |
| OpenHands Indian fingerspelling poses | https://zenodo.org/records/6731224 (`Indian.zip`, 28 MB) | 46 MB unzipped | 36 glosses (letters + digits); 160 train / 86 test clips (verified listing) | MP4 + pickled poses | CC BY 4.0 label (source dataset not documented) | `direct` | low | Tiny; useful only as a smoke test. |
| HF `Hemg/Indian_sign_language_dataset` (mirror of Kaggle prathumarikeri "Indian Sign Language (ISL)") | https://huggingface.co/datasets/Hemg/Indian_sign_language_dataset ; https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl | 292 MB parquet | 42,745 images, 35 classes (1-9, A-Z) | images | none stated (`?`) | `direct` | low | Widely used, but no licence and static only. The HF copy looks like a re-upload of the Kaggle set (same classes); provenance unverified. |
| HF `akritRihal/Indian_Sign_Language_dataset` | https://huggingface.co/datasets/akritRihal/Indian_Sign_Language_dataset | 516 MB | 10,752 images, 33 classes (digits + letters, some missing) | images | none (`?`) | `direct` | low | No licence. |
| HF `Vedant3907/Hindi-Sign-Language-Dataset` | https://huggingface.co/datasets/Vedant3907/Hindi-Sign-Language-Dataset | 439 MB | 16,177 images, 29 Devanagari consonant classes | images | MIT (`C`) | `direct` | low-med | Hindi-letter signs, commercial-safe. Which signing system it follows is not documented. |
| Kaggle `krishbhagat/indian-sign-language-hindi` (HF mirrors `KRISH09bha/...`, `SandhyaPitchika/...` are empty, README only) | https://www.kaggle.com/datasets/krishbhagat/indian-sign-language-hindi/ | reported 48,000 images | reported 40 Hindi letter classes | images | Apache-2.0 on the HF stub (unverified for Kaggle) | `kaggle-key` | low-med | Hindi alphabet; unverified. |
| HF `VinayHajare/Marathi-Sign-Language` | https://huggingface.co/datasets/VinayHajare/Marathi-Sign-Language | 313 MB | 50,099 images, 43 Marathi letters, 128x128 | images | Apache-2.0 (`C`) | `direct` | low | Marathi script; partly overlaps Hindi Devanagari letters. |
| Zenodo NITJ-ISLC-CB (child signers, deaf school) | https://zenodo.org/records/21713512 | 19 MB public subset | 36 ISL character classes; faces removed; full set on request | images | CC BY 4.0 (`C`) | `direct`; full set `request` (nimratveerkb.cs.19@nitj.ac.in, from the record) | low | Child signers are rare; small. |
| IEEE DataPort Samvaad_ISL, alphabet part | see 5E | 1.06 GB | Devanagari Varnamala | images | CC BY 4.0 | `subscription` | low-med | see 5E |

### 5C. Continuous and sentence-level ISL

| Dataset | URL | Size | Content | Format | Licence | Access | Use | Reason |
|---|---|---|---|---|---|---|---|---|
| ISL-CSLTR (Elakkiya R., SASTRA; 2021) | https://data.mendeley.com/datasets/kcmpdxky7p/1 (doi 10.17632/kcmpdxky7p.1) | 8.9 GB single zip | 100 sentences / 700 videos / 7 signers (2 native signers from Navajeevan school AP + 4 SASTRA volunteers); 18,863 sentence frames, 1,036 word images | video + frames | CC BY 4.0 (`C`) | `direct` | **med** | Commercial-safe. Use it for (a) negative/transition samples for the "not sure" class, (b) testing a sliding-window recogniser, and (c) word clips cut from sentences. Pre-extracted features without a licence also exist at `axijit/isl_csltr_extracted_features` (HF). |
| ISLVT (Cummins College of Engineering for Women, Pune; 2024) | https://data.mendeley.com/datasets/98mzk82wbb/1 | 2.9 GB, 153 MOV | about 76 sentences x 2 takes, 1 professional signer, green screen; gloss file names | MOV + gloss | CC BY 4.0 (`C`) | `direct` | low-med | Commercial-safe sentences with glosses. Description mixes ISL and Marathi sign language. |
| iSign / ISLTranslate poses via GloSL | https://huggingface.co/datasets/pedroodb/glosl-isl | 232 GB (per-clip files) | 126,846 clips, 269 h, English text, video-disjoint topic-stratified splits | `.pose` (MediaPipe Holistic 543 pts, pixel coordinates) | CC BY-NC 4.0 (`NC`) | `direct` (ungated) | **high** (research) | Practical streaming route for iSign poses. For word recognition, use it for encoder pre-training and to mine word clips through the English text. Needs the `pose-format` reader and a mapping into RawSequence. |
| Exploration-Lab PoseStitch-ISL | https://huggingface.co/datasets/Exploration-Lab/PoseStitch-ISL | 24.7 GB CSV | synthetic stitched pose sequences (BPCC/BLIMP text to ISL/ASL) | CSV | CC BY-NC-ND 4.0 (`NC`); research-only gate fields in the card | `direct` (gate fields listed but repo shows `gated: False`) | low | Synthetic poses for translation; not real signing. |
| WSLP 2025, MT and word-presence tasks | https://huggingface.co/datasets/Exploration-Lab/WSLP | part of 24.9 GB | sentence videos + text | MP4 + CSV | CC BY-NC-ND 4.0 (`NC`) | `direct` | low-med | Overlaps iSign. |
| Kaggle "Indian Sign Language Video and Text Dataset" | https://www.kaggle.com/datasets/manojkumarcs28/indian-sign-language-video-and-text-dataset | reported 152 videos / 76 sentence pairs | appears to be ISLVT re-uploaded | | not verified | `kaggle-key` | none | Use the Mendeley original. |

### 5D. Unlabelled / pose-only ISL for pre-training

| Dataset | URL | Size | Content | Licence | Access | Use | Reason |
|---|---|---|---|---|---|---|---|
| OpenHands v1 raw ISL pose datasets | https://zenodo.org/records/5558897 | 114.5 GB in 5 zips: NewzHook 61.2, MBM_Vadodara 23.3, ISH_News 15.7, NIOS 11.2, Sign_Library 3.0 | about 1,129 h MediaPipe Holistic poses from 5 Indian deaf channels; HDF5 per channel | CC BY 4.0 label (see section 4 item 4) | `direct` (Range verified) | **high** | Self-supervised encoder pre-training on Indian signing, many signers. Paper: https://aclanthology.org/2022.acl-long.150. Start with Sign_Library (3 GB) as a pilot. Pretrained DPC checkpoints: https://github.com/AI4Bharat/OpenHands/releases/tag/checkpoints_v1 (Apache-2.0 code; check the weights licence). |
| OpenHands v1 labelled ISLR poses (incl. INCLUDE) | https://zenodo.org/records/6674324 | INCLUDE.zip 0.66 GB (plus WLASL, MS-ASL, AUTSL, GSL, LSA64, ASLLVD) | ready-made INCLUDE poses for cross-checking our extraction | CC BY 4.0 label; underlying sets keep their own licences | `direct` | med | Cross-check our INCLUDE features. Multilingual labelled poses for pre-training (research config). |
| YouTube-SL-25 ISL slice | https://github.com/google-research/google-research/tree/master/youtube_sl_25 ; paper https://arxiv.org/abs/2407.11144 | 209 h ISL, 3,023 videos, 8 channels (paper table, verified) | video IDs + captions | IDs CC BY 4.0 (reported); videos belong to their channel owners | IDs `direct`; videos need YouTube download | med | More data, but likely overlaps OpenHands channels, and downloading is costly on this machine. |

### 5E. Hindi-labelled ISL

| Resource | URL | What it gives | Licence | Access | Use |
|---|---|---|---|---|---|
| ISLRTC ISL Dictionary (official): 10,000 terms with **English and Hindi equivalents** (FAQ Q.18) | https://islrtc.nic.in/faq/ ; YouTube https://www.youtube.com/channel/UC3AcGIlqVI4nJWCwHgHFXtg ; Google Drive https://drive.google.com/drive/folders/1U-Pr4r1-cupgNOOq9NH_uTsQnPSVEKco ; DIKSHA ; site https://divyangjan.depwd.gov.in/islrtc/ (unreachable from this machine) | The authoritative Hindi equivalent for each gloss. The data.gov.in files are named in English only, so the Hindi mapping must come from the dictionary website, app or DIKSHA. | ISLRTC conditions (section 4 item 1) | web; Drive folder not machine-listed (needs an API key or browser) | **high**: the cleanest source for the Hindi output column. Fallback: machine translation reviewed by a human. |
| Samvaad_ISL (UPES Dehradun + NIT Jalandhar, Oct 2026) | https://ieee-dataport.org/documents/samvaadisl-multi-modal-indian-sign-language-dataset (doi 10.21227/r7s3-2a79) | Word videos with **on-screen Hindi captions** (about 150-200 words per English initial letter, reported), checked against the ISL portal; plus Devanagari alphabet images. 1.29 + 1.06 GB | CC BY 4.0 (`C`) | `subscription` | med: Hindi labels + commercial licence; signer count not stated. Worth one subscription month if the owner's lead approves. |
| NCERT CIET ISL resources (textbooks Class I-V in ISL, ISL dictionary PDF/flipbook, video programmes) | https://ciet.ncert.gov.in/sign | Hindi/English school vocabulary in ISL, as taught in schools | Government content; terms not stated | web | med: a vocabulary list for "school words" in Hindi; check whether signs match ISLRTC. |
| Hindi alphabet image sets | see 5B (Mendeley 7tsw22y96w, Vedant3907, Kaggle krishbhagat) | Devanagari letter signs | mixed | mixed | low-med (fingerspelling Hindi names) |

### 5F. Madhya Pradesh / Indore

**No public dataset of Indore or MP regional ISL was found** (searched HF, Mendeley, Zenodo and the
web). The ISLRTC dictionary records some regional number and colour signs (Hyderabad, Mumbai,
Kolkata, Munger, Manipur) but none from MP (FAQ Q.18). An NIOS ISL course module
(https://www.nios.ac.in/media/documents/230-ISL/Ch-4.pdf, reported) cites research estimating that about 25% of the
ISL lexicon varies by region. Regional MP data must therefore be **collected with consent**.
The candidate partners below were found during the survey. Contact them through official
channels only; this file does not copy personal phone numbers or emails.

| Organisation | Evidence / URL | Why relevant |
|---|---|---|
| Anand Service Society, Indore (Gyanendra and Monica Purohit) | https://anandservicesociety.org ; news coverage in Free Press Journal (Indore police deaf help centre at Tukoganj, police ISL training) | Long-running Indore deaf organisation; hostels and day-care in Indore, Dhar, Alirajpur, Khandwa; created the sign-language national anthem (2003). Best-known local ISL experts. |
| Mook Badhir Sangathan (founded 1974) and Indore Deaf Bilingual Academy (IDBA) | Gallaudet MP deaf-organisation list (page blocked for automated fetch); DeafNation feature https://deafnation.com/nobarriers/nb-asia/nb-india/india-indore-deaf-bilingual-academy ; ISH News "Math Mela at Deaf School in Indore" https://ishnews.tv/videos/deaf_buzz/math-mela-at-deaf-school-in-indore | A bilingual (ISL + written language) school in Indore. Native young signers; suited to recording school vocabulary and local variants. No official website verified. |
| ISLRTC Directory of ISL Teachers | https://cdnbbsr.s3waas.gov.in/s3f7e2b2b75b04175610e5a00c1e221ebb/uploads/2025/02/202502211294399903.pdf (verified) | The MP section lists about 15 deaf ISL teachers (Indore, Bhopal, Gwalior, Burhanpur, Betul, Dhar and others), including 2 in Indore. These are qualified deaf signers who could validate or record MP variants. |
| ISLRTC Directory of Deaf Schools | https://islrtc.nic.in/directory-of-deaf-schools/ | Lists schools by state, which helps find MP government deaf schools. |
| Indore Police deaf help centre / Indore Bilingual Academy volunteers | Free Press Journal coverage (reported) | Local institutions already using ISL; possible pilot users for the "did you mean" flow. |

**Suggested collection protocol (for the phase-2 owner):** written informed consent in Hindi and ISL
video, allowing commercial use and withdrawal. Record each word 3-5 times on the signer's own
phone (this matches the deployment domain), store landmarks only, and pay signers. Start with
the 500 most frequent words, comparing ISLRTC and FDMSE glosses against what Indore signers
actually use.

### 5G. Multilingual corpora for encoder pre-training

| Dataset | URL | Size / content | Licence | Access | Use | Reason |
|---|---|---|---|---|---|---|
| Google Isolated Sign Language Recognition (Kaggle `asl-signs`) | https://www.kaggle.com/competitions/asl-signs | about 94k sequences, 250 ASL signs, MediaPipe Holistic landmarks (parquet); about 40 GB archive (reported) | reported CC BY 4.0 (not verifiable here) | `kaggle-key` + accept competition rules | **med-high** (commercial config) | Same landmark family as ours, many signers, smartphone-captured. A good commercial-safe encoder warm-up if CC BY is confirmed. |
| Google ASL Fingerspelling (Kaggle) | https://www.kaggle.com/competitions/asl-fingerspelling | landmark sequences, phrases (reported) | reported CC BY 4.0 | `kaggle-key` | med | Pre-training hand-shape features for fingerspelling. |
| PopSign ASL (images on HF) | https://huggingface.co/datasets/sign/popsign-images | 91 GB frames; game and non-game subsets; 250 signs (reported) | CC BY 4.0 (`C`, verified) | `direct` | med | Smartphone captures, commercial-safe. Frames would need landmark extraction (costly on CPU). |
| ASL Citizen (Microsoft) | https://www.microsoft.com/en-us/research/project/asl-citizen/ ; poses mirror https://huggingface.co/datasets/SorensenAI/asl-citizen-poses (81 GB, no licence on mirror) | 83,399 videos, 2,731 signs, 52 signers | MS research licence, `NC`, no redistribution | official download; the mirror's status is doubtful | med (research) | Large webcam ISLR; good for pre-training. Do not use the third-party mirror. |
| WLASL (already in app) | Voxel51/WLASL etc. | 2,000 signs | C-UDA (`NC`) | | | Keep for ASL mode only. |
| AUTSL (Turkish) | https://chalearnlap.cvc.uab.es/dataset/40/description/ ; poses in Zenodo 6674324 | 226 signs, 43 signers, about 38k videos | research-only (reported) | `direct` (poses) | med (research) | Multi-signer ISLR pre-training. |
| LSA64 (Argentine) | http://facundoq.github.io/datasets/lsa64/ | 64 signs, 10 signers, 3,200 videos | CC BY-NC-SA 4.0 (verified on page) | `direct` | low | Small. |
| Slovo (Russian) | https://github.com/hukenovs/slovo ; landmarks https://huggingface.co/datasets/karinversan/slovo-full-landmarks (8.9 GB) | 20k videos, 1,000 classes (reported); smartphone/webcam | "variant of CC BY-SA 4.0" with a custom licence PDF (verified README) | `direct` | med | Large, crowdsourced, webcam-like. Read the licence PDF before commercial use. |
| LSFB-ISOL (French Belgian) | https://huggingface.co/datasets/ppoitier/lsfb-isol | 34.5 GB WebDataset; vocab variants 500/750/2000/all | CC BY-NC-SA 4.0 (`NC`) | `direct` | low-med | Research pre-training. |
| OpenHands multilingual raw poses | https://zenodo.org/records/6673685 (Chinese, Greek, Korean, Russian, Turkish; 152 GB) ; https://zenodo.org/records/6990583 (American, British, Australian, Spanish; 162 GB) | thousands of hours of unlabelled MediaPipe poses | CC BY 4.0 label (derived from YouTube) | `direct` | med | Only needed if ISL-only pre-training is not enough. Disk-heavy. |
| GloSL family (pedroodb) | https://huggingface.co/pedroodb | 20 continuous corpora in one pose format (How2Sign, CSL-Daily, Phoenix14T, LSFB, ...) | mostly NC | `direct` | low | One reader for many corpora; NC. |

---

## 6. Already-planned sources: corrections and helpful extras

| Source | Finding |
|---|---|
| INCLUDE (Zenodo 4010759, CC BY 4.0, 56.8 GB in 46 zips, verified) | **Official splits** are on GitHub, not Zenodo: `https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/{include,include50}_{train,val,test}.txt` (verified: 3,126 / 347 / 816 and 688 / 76 / 191 lines). Paths look like `Adjectives/1. loud/MVI_9289.MOV`, so match on category + folder + file to manifest `zip`/`member`. The Zenodo README mentions a `Train_Test_Split` folder, but no such zip is listed. Signer IDs are not published, so these splits are **not guaranteed signer-independent**. |
| ISLRTC dictionary (data.gov.in; HF mirror silentone0725, about 77.9 GB) | Licence caveat in section 4 item 1. Duplicate HF mirrors: `Vignesh3816/...`, `suneetpaul/...`. Hindi equivalents exist in the official dictionary but not in the file names. |
| CISLR (HF gated) | Still **403** for data files on 25 Nov 2026 (README returns 200 even without access, so check a data file such as `test.csv`, not README.md, to detect access). Licence is CC BY-NC 4.0 per README (section 4 item 2). Interim substitute: WSLP-WR (5A). |
| iSign (HF gated) | Still **403** for `word-presence-dataset_v1.1.csv`. Same README caveat applies to the access check. Ungated pose route: GloSL-ISL (5C). |

**Access-check correction:** for gated HF repos, `resolve/main/README.md` returns 200 without access,
so a README check will wrongly report that access was granted. Test a data file with a 1-byte
Range GET instead (e.g. `Exploration-Lab/CISLR/resolve/main/test.csv`,
`Exploration-Lab/iSign/resolve/main/iSign_v1.1.csv`): 403 = no access, 200/206/302 = granted.

---

## 7. Evidence synthesis map

```text
CLAIM: Enough public ISL data exists to recognise "as many ISL words as possible" reliably.
Well-established:  Vocabulary breadth exists. The ISLRTC dictionary (about 10k terms, 1 signer),
                   CISLR/WSLP (about 4.4-4.7k words, about 1 clip each) and FDMSE (2,002 words) are
                   independent recordings from different institutions.
Contested:         Which sign is "correct" for a word. FDMSE (Coimbatore), ISLRTC (Delhi) and INCLUDE
                   (Chennai-area students) were recorded in different regions. Regional variation is
                   estimated at about 25% of lexicon (NIOS teaching module citing older surveys, so
                   weak evidence).
Single-source:     ISL500 (15 signers x 500 words) has no paper and no documented origin. The
                   Vāksetu numbers are mostly one signer.
Evidence gap:      (1) No public MP/Indore regional data. (2) No multi-signer data for the long
                   tail beyond about 2k words (FDMSE is the ceiling, and only on request).
                   (3) No public Hindi-captioned multi-signer word set except Samvaad_ISL
                   (paywalled). (4) Kaggle sets unverified (no key).
Confidence:        Moderate that about 2,000-2,500 words can reach usable top-5 accuracy with
                   FDMSE + ISL500 + INCLUDE + pre-training. Low for the long tail (4k-10k words),
                   which rests on one example per word, so the "did you mean / not sure" UI and
                   user-added variants are required, not optional.
```

---

## 8. Practical ingestion notes (verified techniques)

- **Remote-zip streaming works** on HF resolve URLs and Mendeley S3: an HTTP Range reader plus
  Python `zipfile` listed the 4,399-entry WSLP `train.zip` without downloading it. Use the same pattern for
  CISLR (`CISLR_v1.5-a_videos`), the WSLP zips, the OpenHands Zenodo zips and the Mendeley zips. Read one
  member, extract landmarks, delete it, move on. Zenodo returns 206 for Range requests.
- Mendeley file URLs come from `https://data.mendeley.com/public-api/datasets/<id>` → `files[].content_details.download_url`
  (anonymous; redirects to S3).
- GloSL `.pose` files need `pose-format` (`uv pip install pose-format`). Coordinates are in pixels
  with width and height in the header. Normalise to MediaPipe's 0-1 range before converting to RawSequence.
- ISL500 `.h5` landmark layout is undocumented; inspect one file before writing the adapter.
- Commercial-safe config candidates: INCLUDE, ISL-CSLTR, ISLVT, the Mendeley CC BY sets, PopSign, and
  Google ISLR (if CC BY is confirmed). ISLRTC is fine for non-profit use (see section 4). The OpenHands raw poses
  need a risk decision.
  Research/best config adds: FDMSE-ISL, ISL500, WSLP, CISLR, iSign/GloSL, ISL-Fingerspelling, ASL Citizen,
  AUTSL, LSFB and WLASL.

## 9. Known gaps in this survey

- Kaggle could not be checked (no API key; JavaScript pages). The 10 listings are from search snippets only.
- No Hindi-language web search was done; Hindi-medium ISL resources (e.g. MP state portals) may be missed.
- GitHub search API is blocked; GitHub-only datasets were found only through web search.
- The FDMSE-ISL request page returned HTTP 503 on the survey date; its terms are taken from the
  overview page ("research use upon request").
- The ISLRTC dictionary website (divyangjan.depwd.gov.in) and the data.gov.in catalogue page were
  unreachable from this machine; the Gallaudet MP directory page returned 403.
