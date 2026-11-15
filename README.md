# Sign Intelligence Studio

**Real-time isolated American Sign Language (ASL) sign recognition on the WLASL100 vocabulary**, using MediaPipe Holistic landmarks, recurrent sequence models (GRU / LSTM, with a temporal-CNN baseline) and a live webcam pipeline.

> **Educational and research implementation.** It recognises *one isolated sign at a time* from a fixed 100-word vocabulary and outputs an ASL **gloss label**. It does not translate ASL sentences, does not model ASL grammar, and must not be used in place of a qualified interpreter. WLASL data is licensed for **non-commercial, computational use only** (C-UDA).

**Online demo (runs entirely in your browser):** https://huggingface.co/spaces/agrawalaarush08/sign-language-studio

**Current model** (`models/wlasl100-gru-signer-holdout`): bidirectional GRU trained on the 524 WLASL100 clips that could be downloaded (~5 per sign). Signer-independent results — validation top-1 62.0%; held-out test on 7 unseen signers: **top-1 44.3%, top-5 72.2%**. Live webcam accuracy will be lower.

---

## Contents

1. [What it does](#what-it-does)
2. [Quick start (Windows)](#quick-start-windows)
3. [Using the application](#using-the-application)
4. [Architecture](#architecture)
5. [Command line](#command-line)
6. [Testing](#testing)
7. [Project structure](#project-structure)
8. [Documentation](#documentation)
9. [Licensing, attribution & citation](#licensing-attribution--citation)
10. [Status](#status)

## What it does

| Area | Summary |
|---|---|
| Dataset | Fetches the official WLASL annotations, selects **WLASL100** (first 100 glosses, cross-checked against the authors' `nslt_100.json`: 2,038 instances, 97 signers), downloads only those videos, validates them, and records every failure. |
| Landmarks | MediaPipe **Holistic**: 33 pose, 2 × 21 hand and a 20-point face subset per frame. The same model runs in Python (dataset/offline) and in the browser (live webcam). |
| Features | Aspect-corrected, shoulder-centred, scale-normalised body frame; wrist-relative hand shape; hand position; face subset; presence flags; velocities. Missing detections stay missing (flagged), never invented. |
| Models | Bidirectional **GRU** (primary), **LSTM**, **temporal CNN** with masked attention pooling (PyTorch, CPU or CUDA). |
| Evaluation | Official split (labelled *not* signer-independent) plus **signer-holdout** and **5-fold signer-grouped CV** splits; automated leakage checks; test set evaluated only on explicit request. |
| Live | Webcam → MediaPipe in the browser → landmark coordinates over a localhost WebSocket → sign-boundary detection, temporal-crop agreement, confidence/margin thresholds, cooldown and repeat suppression → gloss text. |
| UI | Ten pages: Overview, Live Recognition, Video Analysis, Vocabulary, Landmark Studio, Dataset Explorer, Model Laboratory, Evaluation, Experiment History, Research & Docs. |

## Quick start (Windows)

Requirements: **Windows 10/11**, **Python 3.12 or 3.11**, **Node.js 20+**, Chrome or Edge, a webcam. Optional: an NVIDIA GPU (CUDA), `ffmpeg` (needed only for the 162 ASLPro `.swf` videos).

```bash
winget install Python.Python.3.12
```

```bash
winget install OpenJS.NodeJS.LTS
```

```bash
winget install Gyan.FFmpeg
```

From the project folder, in PowerShell:

```bash
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

```bash
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

| | |
|---|---|
| Frontend | http://localhost:5173 |
| Backend | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |

```bash
powershell -ExecutionPolicy Bypass -File .\stop.ps1
```

* `setup.ps1` creates `.venv`, installs PyTorch (CUDA build when `nvidia-smi` finds a GPU, otherwise CPU; `-Cpu` forces CPU), backend and frontend dependencies, downloads the 13 MB MediaPipe model and initialises SQLite. It does **not** download WLASL or start training. `-Dev` adds test tools.
* `start.ps1` launches both servers in the background (logs in `.run/`), waits until they respond, and opens the browser (`-NoBrowser` to skip).
* `stop.ps1` stops only the processes `start.ps1` recorded.

## Using the application

1. **Dataset Explorer → Load annotations.** Downloads the two official annotation files, records their SHA-256, builds the WLASL100 inventory and the splits.
2. **Download & validate.** Start with a limit (e.g. 50) to see how your network fares; the full set is 2,038 instances. Expired links are recorded as *failed*. Videos the WLASL authors send you on request go in `data/videos/provided/<video_id>.mp4`, then **Scan local files**.
3. **Extract landmarks.** Runs MediaPipe Holistic over each validated clip (CPU; roughly 10–30 ms per frame depending on the machine).
4. **Model Laboratory → Start training.** The *Quick check* preset trains in minutes on a CPU; *Standard* is the recommended configuration. Curves stream from the real training loop.
5. **Activate** the best run, compare on **Evaluation**, and run the **held-out test** evaluation once you have finished selecting models.
6. **Live Recognition → Start camera → Start recognition.** Sit with head and shoulders in frame, raise your hands, sign one word, lower your hands. Settings let you change the mode (sign boundaries / sliding window), thresholds, processing resolution, MediaPipe processor (GPU/CPU) and overlays.

## Architecture

```
Browser (React + TypeScript)                       Local backend (FastAPI, Python)
┌─────────────────────────────────────┐            ┌─────────────────────────────────────────────┐
│ webcam ─► MediaPipe Holistic (WASM) │  landmarks │ WebSocket /api/ws/recognize                 │
│          GPU or CPU, local only     │ ─────────► │   RecognitionEngine (buffer, boundaries,    │
│ overlay, status, transcript         │ ◄───────── │   smoothing, thresholds) ─► Predictor (torch)│
└─────────────────────────────────────┘   states   │ REST: dataset, splits, experiments, jobs,   │
                                                    │   vocabulary, analysis, exports             │
                                                    │ Jobs: acquire ▸ validate ▸ extract ▸ train  │
                                                    │ SQLite (metadata/results) · data/ (videos,  │
                                                    │ landmarks) · artifacts/ (checkpoints)       │
                                                    └─────────────────────────────────────────────┘
```

* **One feature implementation** (`backend/app/features/build.py`) serves training, offline analysis and live inference. The browser sends raw landmarks in the layout published at `/api/features/spec`.
* **Privacy:** no webcam image ever leaves the browser; only coordinates go to `localhost`. Recognitions are stored as gloss + score only.
* **Why landmarks in the browser and the model on the backend:** browser MediaPipe avoids streaming video, while one Python model/feature path guarantees training–inference parity.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full method.

## Command line

All pipeline steps are also available without the UI (run inside `backend/` with the virtual environment active):

```bash
python -m app.cli annotations
```

```bash
python -m app.cli acquire --limit 100
```

```bash
python -m app.cli extract
```

```bash
python -m app.cli splits
```

```bash
python -m app.cli train --arch gru --split signer-holdout --epochs 80 --seq-len 48 --activate
```

```bash
python -m app.cli evaluate-test 1
```

Environment variables (prefix `SLR_`): `SLR_DATA_DIR`, `SLR_ARTIFACTS_DIR`, `SLR_DEVICE` (`auto`/`cpu`/`cuda`), `SLR_TORCH_THREADS`, `SLR_DOWNLOAD_WORKERS`, `SLR_DOWNLOAD_TIMEOUT_S`.

## Testing

```bash
cd backend; ..\.venv\Scripts\python -m pytest -q
```

```bash
cd frontend; npm test
```

With the app running (`start.ps1`), browser end-to-end tests use Chromium's fake camera:

```bash
cd frontend; npx playwright install chromium; npx playwright test
```

`E2E_FAKE_VIDEO=<file.y4m>` feeds a video file as the fake camera.

Backend tests run against an isolated temporary data directory with **synthetic landmark sequences** (procedurally generated, clearly flagged). They verify the pipeline's wiring and logic, not ASL accuracy.

## Project structure

```
backend/
  app/
    api/          FastAPI routers (dataset, experiments, recognition + WebSocket, system)
    core/         settings, background jobs
    db/           SQLAlchemy models + session (SQLite)
    dataset/      annotations (WLASL100 selection), acquisition/validation, synthetic test data
    landmarks/    MediaPipe Holistic extractor, raw sequence format, batch pipeline
    features/     feature spec + normalisation/resampling/augmentation
    splits/       official / signer-holdout / signer-CV splits, leakage checks
    models/       GRU / LSTM / TCN networks, checkpoint + predictor
    training/     dataset loading, training loop, test evaluation
    evaluation/   metrics (top-k, macro F1, per-class, confusion, calibration curve)
    inference/    streaming recognition engine, active-model registry
    cli.py        command-line interface
    main.py       app factory (docs at /docs)
  tests/          pytest suite
frontend/
  src/pages/      the ten pages
  src/recognition live state machine, WebSocket client, status model
  src/mediapipe/  browser Holistic loader, overlay drawing
  src/components/ UI kit and SVG/canvas charts
  e2e/            Playwright tests
scripts/fetch_models.py   MediaPipe model download
docs/                     dataset, methodology, licensing, limitations, reproducibility, checklist
data/  artifacts/  assets/   local only (git-ignored)
setup.ps1  start.ps1  stop.ps1
```

## Documentation

* [docs/DATASET.md](docs/DATASET.md) — WLASL100 specification, acquisition, inventory, splits
* [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — landmarks, features, models, training, live engine
* [docs/EVALUATION.md](docs/EVALUATION.md) — signer-independent protocol and leakage checks
* [docs/LICENSING.md](docs/LICENSING.md) — C-UDA terms, attribution, third-party licences
* [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — scientific and practical limitations
* [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) — exact steps to reproduce results
* [docs/CHECKLIST.md](docs/CHECKLIST.md) — development checklist and acceptance-criteria status

## Licensing, attribution & citation

The **code** in this repository is provided for education and research. The **WLASL dataset** is © its authors and licensed under the [Computational Use of Data Agreement (C-UDA)](https://github.com/dxli94/WLASL): academic and computational use only, **no commercial use**. This repository contains no WLASL videos; each user downloads them from the original sources. Please cite:

```
@inproceedings{li2020word,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1459--1469},
  year={2020}
}
```

MediaPipe © Google (Apache-2.0). Fonts: IBM Plex (SIL OFL 1.1). Icons: Lucide (ISC).

## Deployment (Hugging Face)

The online demo is a free **static** Space: the GRU is exported to ONNX and run with ONNX Runtime Web, and the feature pipeline and recognition engine have TypeScript ports (`frontend/src/static/`) whose output is tested against the Python implementation.

```bash
python scripts/export_static.py
```

```bash
cd frontend; npm run build:static
```

```bash
python scripts/deploy_hf_static.py <user>/<space>
```

A full-backend Docker Space (`deploy/huggingface`, `scripts/build_hf_space.py`) is also provided; Hugging Face requires a PRO plan for Docker Spaces.

## Status

See [docs/CHECKLIST.md](docs/CHECKLIST.md) for what has been verified and what still needs verification on a Windows machine with a real webcam and the WLASL videos.
