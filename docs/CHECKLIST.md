# Development checklist & acceptance status

Legend: ✅ implemented and verified here · 🟡 implemented, verification pending on the target machine / with real data · ⏳ waiting on a decision · ❌ not done

Verification environment for this build: Linux container (4 CPU cores, no GPU), Python 3.12, Node 22, headless Chromium with a fake camera, PowerShell 7.5 for the scripts. The WLASL video hosts were not reachable from it, except the S3 host (one clip, see below).

## Deliverables

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Frontend | ✅ | 10 pages, typecheck + production build pass, Playwright renders every page without console errors |
| 2 | Backend | ✅ | FastAPI, docs at `/docs`, 36 pytest tests |
| 3 | WLASL100 acquisition pipeline | ✅/🟡 | YouTube/HTTP/ASLPro paths; failures recorded (verified with real URLs: 1 valid, 5 blocked-host failures recorded). Full download not run here (network) |
| 4 | Dataset inventory | ✅ | per-gloss inventory API/page/CSV; verified on the real annotation file |
| 5 | Video validation | ✅ | decode check, measured fps/resolution/duration/hash; tests with valid + corrupt files |
| 6 | MediaPipe extraction | ✅ | Python Holistic on a real WLASL clip (07068 BOOK): pose 100%, face 100%, hands 56% of frames |
| 7–9 | Hand / body / face processing | ✅ | features with tests (invariances, missing handling) |
| 10 | Feature normalisation | ✅ | translation/scale/aspect invariance tests |
| 11 | Temporal sequences | ✅ | timestamp resampling, masks, augmentation |
| 12 | Trained LSTM or GRU model **on WLASL100** | ✅ | GRU on 524 real clips: val top-1 62.0%, held-out test (7 unseen signers) top-1 44.3% / top-5 72.2% (`models/`) |
| 13 | Baseline comparisons | ✅ | val top-1 / macro F1: GRU 62.0/58.1, GRU no-face 62.0/58.6, TCN 60.8/56.7, LSTM 55.7/53.4 |
| 14 | Signer-independent evaluation | ✅ | signer-holdout + 5-fold signer CV built from real metadata, all pass leakage checks |
| 15 | Automated leakage checks | ✅ | signer/hash/source/duplicate/class checks + tests; detected and fixed a shared-source signer pair (5 & 64) |
| 16 | Live webcam recognition | ✅/🟡 | verified with a real signing clip as fake camera: waiting → collecting → inference → recognised; physical webcam on Windows pending |
| 17 | Real-time text output | ✅ | transcript, copy/clear |
| 18 | Prediction smoothing | ✅ | segment + sliding modes; tests: ≤1 word per sign, repeat suppression, rejection |
| 19 | Vocabulary explorer | ✅ | |
| 20 | Landmark visualisation | ✅ | 2D overlay, trails, 3D hand, trajectories, presence strip, coordinates |
| 21 | Video analysis | ✅ | upload → Python extraction → whole-clip + streaming results, temp file deleted |
| 22 | Model laboratory | ✅ | real progress from the training loop, cancel, presets, compare |
| 23 | Evaluation dashboard | ✅ | val/test separated, confusion matrix, per-class, per-signer, calibration |
| 24 | Experiment history | ✅ | full configs, CSV export |
| 25 | SQLite database | ✅ | |
| 26 | API documentation | ✅ | `/docs` (HTTP 200 verified) |
| 27 | Automated tests | ✅ | pytest 36, Vitest 8, Playwright 2 |
| 28–30 | setup / start / stop scripts | ✅/🟡 | all three run successfully under PowerShell 7 (Linux); Windows-only branches (`taskkill`, `Scripts\python.exe`, `npm.cmd`, CUDA wheel) need a run on Windows |
| 31–36 | README, dataset docs, references, licensing, limitations, reproducibility | ✅ | `README.md`, `docs/` |

## Final acceptance criteria

| Criterion | Status |
|---|---|
| Starts successfully on Windows | 🟡 run `setup.ps1` + `start.ps1` |
| Frontend and backend communicate | ✅ |
| Webcam permission works | ✅ (fake camera, Chromium) · 🟡 physical webcam |
| Webcam feed displays | ✅ (fake camera) · 🟡 physical webcam |
| MediaPipe detects required landmarks | ✅ (photo and real WLASL clip, browser and Python) |
| Two-hand tracking where needed | ✅ both hands detected in the real BOOK clip |
| Relevant body information captured | ✅ |
| Landmark features match training config | ✅ one feature module; spec served to the browser; parity on visible landmarks ≤0.01 · 🟡 parity on real signing video |
| WLASL100 samples can be processed | ✅ (one real sample end to end) · ⏳ full set |
| Labels correctly associated with videos | ✅ cross-check vs `nslt_100.json`, 0 mismatches |
| Sequence construction works | ✅ |
| Model can be trained locally | ✅ (CPU, synthetic) · ⏳ WLASL100 |
| Signer-independent evaluation where metadata permits | ✅ |
| Model can classify actual signs | ✅ 44.3% top-1 on unseen signers (modest: ~5 training clips per sign) |
| Live predictions from webcam input | ✅ (pipeline, synthetic model) · ⏳ meaningful predictions need a WLASL model |
| Smoothing prevents repeated unstable text | ✅ (tests; real-clip loop produced ≈1 word per repetition) |
| Unknown / uncertain signs handled | ✅ threshold, margin, agreement, explicit "uncertain" state |
| Real evaluation results displayed | ✅ from stored experiment records (currently synthetic-flagged) |
| Experiment history saved | ✅ |
| Important workflows pass automated tests | ✅ |
| Documentation explains the methodology | ✅ |

## Data acquired (2026-11-15)

524 of 2,038 WLASL100 instances usable, all 100 glosses covered (min 2 per gloss). YouTube (685) refuses cloud downloads; signingsavvy/handspeak links are mostly dead (404); aslpro, aslsearch, aslsignbank, aslbricks hosts unreachable. Missing videos were requested from the WLASL authors.

## Deployment

Online demo (static, browser-only): https://huggingface.co/spaces/agrawalaarush08/sign-language-studio — verified live: camera, MediaPipe, ONNX model, recognition statuses. TS feature port matches Python on real clips (Vitest parity test).

## Open decisions (from the Stage 1 blueprint)

1. **Where data acquisition and training run** — allow the video hosts in this environment, or run `acquire`/`extract`/`train` on your Windows machine.
2. **Missing videos** — submit the WLASL authors' request form, or proceed with what downloads.
3. **Committing derived data** — whether a trained checkpoint should be committed (landmarks stay local).

## Notes

* During the acquisition test, one WLASL video (07068, BOOK, from the S3 host that this environment can reach) was downloaded into a temporary directory to verify extraction on real signing. It was not committed or redistributed.
* Synthetic landmark data (procedurally generated stick-figure trajectories) is used only by automated tests and the e2e harness, in isolated data directories, and every model trained on it is flagged "synthetic" in the UI.
