# Methodology

## 1. Landmark extraction

MediaPipe **Holistic Landmarker** (`holistic_landmarker.task`, float16) in VIDEO mode. A fresh landmarker per video prevents tracking state leaking between clips. Frames larger than 960 px are downscaled before detection (coordinates are normalised, so this does not change them).

Stored per frame (`RawSequence`, `.npz`): pose 33×(x, y, z, visibility), left hand 21×3, right hand 21×3, face subset 20×3 (mesh indices in `features/spec.py`), timestamp, image size. **Missing detections are NaN**, so "not detected" stays distinguishable from a coordinate.

Left/right are the signer's own hands. Frames are always processed unmirrored; the webcam preview is mirrored on screen only.

## 2. Feature pipeline (`features/build.py`)

Shared by training, offline analysis and live inference.

1. **Aspect correction**: x and hand-z × (width / height), making distances isotropic for any camera aspect.
2. **Gap filling**: interior gaps ≤ `max_gap` (5) frames are linearly interpolated per component; leading/trailing and longer gaps stay missing.
3. **Body frame**: origin = median shoulder midpoint, unit = median shoulder width over the sequence. No pose in any frame → the sequence is rejected (live: "Body not visible").
4. **Augmentation** (training only): mirror with hand swap (p = 0.5), rotation ±10°, scale ±10%, shift ±0.1, temporal crop ≤15% per end, smooth time-warp ±20%, jitter σ = 0.01, per-frame hand dropout 5%.
5. **Resampling** to `seq_len` steps on the timestamp axis. Linear only when both neighbours are detected, nearest-neighbour otherwise — a detection is never blended with a missing frame.
6. **Per-step features** (default 359 values):
   * upper body (nose, shoulders, elbows, wrists) x, y — 14
   * each hand: 20 wrist-relative points × 3, divided by wrist→middle-MCP length — 60 ×2
   * each hand: wrist position in the body frame — 2 ×2
   * face subset x, y in the body frame — 40 (optional)
   * velocities of all of the above (first differences, zero across a missing frame) — 178 (optional)
   * presence flags: left hand, right hand, face — 3

Verified properties (tests): invariance to translation and scale, invariance to camera aspect ratio, missing hands produce zeros + flag 0, mirror augmentation is an involution, browser JSON round-trip reproduces features.

## 3. Models (`models/networks.py`)

| | Structure |
|---|---|
| GRU (primary) | LayerNorm → Linear(D→H) + GELU + dropout → 2-layer bidirectional GRU (H) → masked attention pooling → LayerNorm → dropout → Linear(→100) |
| LSTM | same with LSTM cells |
| Temporal CNN | same projection → 4 residual blocks of dilated Conv1d (k = 5, dilation 1, 2, 4, 8) with BatchNorm → attention pooling → classifier |

The attention pool ignores steps with no detected hand. Bidirectionality is valid live because the engine classifies a complete window.

## 4. Training (`training/trainer.py`)

AdamW (lr 1e-3, weight decay 1e-2), linear warm-up (5% of steps) then cosine decay, cross-entropy with label smoothing 0.1 (optionally class-weighted for imbalance), gradient clipping at 1.0, batch 32, up to 80 epochs, early stopping on **validation macro F1** (patience 20), best checkpoint saved. Seeds fixed; the full configuration, feature config, class list and split name are stored in the checkpoint and the experiment record.

After training, the best checkpoint is evaluated on validation; the **rejection threshold** used live is chosen from the validation accuracy–coverage curve (lowest confidence threshold whose accepted predictions reach 85% accuracy, minimum 0.35). The test partition is never touched by training.

## 5. Live recognition engine (`inference/engine.py`)

Input: one raw landmark frame at a time (≈3 KB JSON) over `ws://localhost:8000/api/ws/recognize`.

**Sign-boundary mode (default)**
* start: hands detected in 3 consecutive frames (240 ms pre-roll included)
* end: no hands for `end_absent_ms` (400 ms) or segment ≥ 4 s
* while active: provisional top-5 every 200 ms (displayed, never emitted)
* at end: classify the full segment plus two 10%-trimmed crops; emit only if mean top-1 ≥ threshold, margin over runner-up ≥ 10%, ≥ 2 of 3 crops agree, outside the 700 ms cooldown. Otherwise "uncertain" with the reason.

**Sliding-window mode**
* every 200 ms classify the last 2 s (≥ 50% of frames with hands), EMA (α = 0.5) over windows
* emit after 3 consecutive agreeing windows above threshold and margin; suppress the same gloss until it changes or the hands drop; cooldown applies

Statuses surfaced in the UI: camera inactive/starting/error, initialising MediaPipe, detecting landmarks, connecting, recognition unavailable, body not visible, waiting for hands, collecting sequence, model inference, recognised sign, uncertain sign, paused.

## 6. Browser pipeline

`@mediapipe/tasks-vision` Holistic with WASM and model served locally (no CDN). Delegate: GPU (WebGL) by default, falling back to CPU; "auto" chooses CPU when WebGL is software-rendered (measured here: 8.4 fps CPU vs 1.0 fps software-GPU in a headless 4-core container). Frames can be downscaled (default 640 px wide) and rate-capped (default 25 fps). Back-pressure: frames are dropped, not queued, if more than 8 are in flight.

Parity check performed (photo of a person, same model in both runtimes): face landmarks agree within 0.0005 and nose/shoulders within 0.01 (normalised units); joints hidden or cropped in that photo (elbows, wrists) differed by up to 0.16 — both runtimes guess for occluded joints. A parity check on real signing footage is part of the acceptance steps (see CHECKLIST.md).
