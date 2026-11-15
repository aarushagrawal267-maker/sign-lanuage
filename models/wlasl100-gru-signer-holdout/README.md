# WLASL100 GRU — signer-holdout

Bidirectional GRU (hidden 128, 2 layers, attention pooling) over MediaPipe Holistic landmark features
(T = 64, velocity on, face subset off — the face ablation scored equal/higher on validation).

| Partition | Signers | Videos | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|---|---|
| Validation (model selection) | 2 | 79 | 62.0% | 82.3% | 58.6% |
| Held-out test (run once) | 7 | 79 | 44.3% | 72.2% | 44.0% |

Trained on the 524 WLASL100 clips that were downloadable on 2026-11-15 (no YouTube; ~5 per sign),
`signer-holdout-available` split (no signer in more than one partition, leakage checks passed).
Full numbers and configuration: `results.json`.

Files: `checkpoint.pt` (PyTorch, loaded by the backend), `model.onnx` (softmax output, used by the
browser build), `model.json` (class names, feature config, live threshold).

Derived from WLASL (C-UDA): academic / computational use only, no commercial use.
To use locally: copy `checkpoint.pt` to `artifacts/checkpoints/` and train/activate from the Laboratory,
or retrain with `python -m app.cli train --arch gru --split signer-holdout-available --seq-len 64 --hidden 128 --epochs 150 --patience 40 --no-face`.
