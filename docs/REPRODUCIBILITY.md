# Reproducibility

1. **Environment** — `setup.ps1` (Python 3.12/3.11 venv, PyTorch CPU or CUDA, pinned-minimum requirements in `backend/requirements.txt`, `frontend/package-lock.json`, MediaPipe `holistic_landmarker.task`). The System card on the Overview page shows the versions in use.
2. **Annotations** — Dataset → *Load annotations* (or `python -m app.cli annotations`). The SHA-256 of `WLASL_v0.3.json` becomes the dataset version recorded with each experiment (`WLASL_v0.3/WLASL100@<sha12>`).
3. **Videos** — *Download & validate*. The availability of each video at your time of download is recorded (`download_attempts`). Export the inventory (CSV) to document what you trained on.
4. **Landmarks** — *Extract landmarks*. Extractor version `holistic-v1` is stored with each file.
5. **Splits** — built deterministically (seed 0) when the annotations are loaded; `python -m app.cli splits` prints sizes and leakage-check results.
6. **Training** — every run stores: architecture, model sizes, optimiser settings, epochs/patience, batch size, augmentation flag, seed, device, feature configuration (sequence length, face, velocity, gap length), split name, dataset version, data summary (videos/signers/classes per partition, exclusions), per-epoch history, checkpoint path. Re-running with the same seed on the same device and data reproduces the run up to nondeterminism in some CUDA kernels.
7. **Evaluation** — validation metrics at the end of training; test metrics only via the explicit action. Export with `/api/experiments/export.csv` and `/api/experiments/{id}/per-class.csv`.
8. **Robust estimate** — train one model per `signer-cv5-fold*` split with identical settings and report mean ± standard deviation of test top-1 and macro F1.

Command-line sequence:

```bash
cd backend
python -m app.cli annotations
python -m app.cli acquire
python -m app.cli extract
python -m app.cli train --arch gru --split signer-holdout --seq-len 48 --epochs 80 --seed 42
python -m app.cli evaluate-test <experiment id>
```
