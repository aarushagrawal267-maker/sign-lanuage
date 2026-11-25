"""Worker-side landmark extraction for the video / image sources (runs in spawned processes).

* ``full``  calls ``scripts/ingest_isl.py extract_full`` itself (same HolisticExtractor, max side 640, same
  sampling code), so these clips are directly comparable with INCLUDE.
* ``hands`` = ``full`` + MediaPipe HandLandmarker (VIDEO mode, 2 hands, same 0.5 confidences) for frames where
  Holistic found no hand of that side. HandLandmarker reports handedness for a *mirrored* (selfie) image; for the
  un-mirrored recordings used here its "Left"/"Right" agreed with Holistic's left/right on the frames where both
  saw a hand (the per-clip agreement is recorded as ``hl_agree`` so this stays checkable). ``extractor_version`` is
  ``holistic-v1+hands`` when at least one hand was filled in, otherwise plain ``holistic-v1``.
* ``image`` = one still -> one-frame ``RawSequence``: Holistic in IMAGE mode (pose, face, hands) and HandLandmarker
  for the hands Holistic missed (``extractor_version = "image-holistic-v1+hands"``). ``t_ms = [0]``. The AI
  engineer's letter features (``letter_frame_features``) read frame 0 of these files.

Model files: ``assets/mediapipe/holistic_landmarker.task`` (already used by the app) and
``assets/mediapipe/hand_landmarker.task`` (Apache-2.0, fetched by ``scripts/ingest_isl_video.py`` on first use).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
                  "hand_landmarker.task")
MAX_SIDE = 640  # same as scripts/ingest_isl.py (HolisticExtractor(max_side=640))

_HOLISTIC_IMAGE = None
_HANDS_IMAGE = None


def hand_model_path() -> Path:
    from app.core.config import get_settings

    return get_settings().assets_dir / "mediapipe" / "hand_landmarker.task"


def _hand_landmarker(video: bool):
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path()), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def _holistic_image():
    """Holistic with the app's confidences (HolisticExtractor._create) but IMAGE mode (no tracking state)."""
    global _HOLISTIC_IMAGE
    if _HOLISTIC_IMAGE is None:
        from mediapipe.tasks.python import BaseOptions, vision

        from app.core.config import get_settings

        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(get_settings().holistic_model_path),
                                     delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5, min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5, min_face_detection_confidence=0.5,
        )
        _HOLISTIC_IMAGE = vision.HolisticLandmarker.create_from_options(opts)
    return _HOLISTIC_IMAGE


def _hands_image():
    global _HANDS_IMAGE
    if _HANDS_IMAGE is None:
        _HANDS_IMAGE = _hand_landmarker(video=False)
    return _HANDS_IMAGE


def _resize(bgr):
    import cv2

    h, w = bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def _mp_image(bgr):
    import cv2
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def hand_results(res) -> list[tuple[str, float, np.ndarray]]:
    """HandLandmarker result -> [(\"left\"|\"right\", score, (21, 3) array)]."""
    out = []
    for lms, hd in zip(res.hand_landmarks or [], res.handedness or []):
        if not hd or len(lms) != 21:
            continue
        side = "left" if hd[0].category_name.lower() == "left" else "right"
        out.append((side, float(hd[0].score or 0.0), np.array([[p.x, p.y, p.z] for p in lms], np.float32)))
    return out


def fill_hands(left: np.ndarray, right: np.ndarray, i: int, found: list[tuple[str, float, np.ndarray]],
               stats: dict) -> None:
    """Put HandLandmarker hands into frame ``i`` where that side is missing.

    A HandLandmarker hand whose wrist lies within 0.05 (image units) of a hand Holistic already found is the same
    hand: it is counted (``both``) and, if the labels match, ``agree``; it is never copied to the other side (no
    duplicated hands). On the sample videos checked on 25 Nov 2026 the labels matched in 42 of 42 such frames."""
    best: dict[str, tuple[float, np.ndarray]] = {}
    for side, score, arr in found:
        if side not in best or score > best[side][0]:
            best[side] = (score, arr)
    had = {"left": not np.isnan(left[i, 0, 0]), "right": not np.isnan(right[i, 0, 0])}
    for side, (_, arr) in best.items():
        near = [s for s, h in (("left", left), ("right", right))
                if had[s] and float(np.linalg.norm(h[i, 0, :2] - arr[0, :2])) < 0.05]
        if near:
            stats["both"] += 1
            stats["agree"] += int(side in near)
            continue
        dst = left if side == "left" else right
        if not had[side]:
            dst[i] = arr
            stats["filled"] += 1


def extract_hands_video(path: str, out: str, step_fps: float = 15.0, max_s: float = 20.0) -> dict:
    """``extract_full`` (Holistic, identical sampling) + HandLandmarker for missing hands."""
    from app.isl.data.ingest_bridge import ingest_module
    from app.landmarks.raw import RawSequence

    m = ingest_module()
    cap, fps, step, w, h = m._frames(path, step_fps, max_s)
    frames = []
    try:
        i = 0
        while True:
            ok, f = cap.read()
            if not ok or i / fps > max_s:
                break
            if i % step == 0:
                frames.append((_resize(f), i * 1000.0 / fps))  # the extractor would resize to 640 anyway
            i += 1
    finally:
        cap.release()
    seq = m._extractor().extract_frames(iter(frames), w, h)
    stats = {"filled": 0, "both": 0, "agree": 0}
    if len(seq):
        left, right = seq.left.copy(), seq.right.copy()
        hl = _hand_landmarker(video=True)
        try:
            last = -1
            for k, (f, t) in enumerate(frames):
                ts = max(int(round(t)), last + 1)
                last = ts
                fill_hands(left, right, k, hand_results(hl.detect_for_video(_mp_image(f), ts)), stats)
        finally:
            try:
                hl.close()
            except Exception:  # noqa: BLE001
                pass
        ver = seq.extractor_version + ("+hands" if stats["filled"] else "")
        seq = RawSequence(seq.pose, left, right, seq.face, seq.t_ms, seq.width, seq.height, ver)
    seq.save(Path(out))
    pr = seq.presence()
    return {"n_frames": len(seq), "fps": fps / step,
            "hands_frac": float((pr["left"] | pr["right"]).mean()) if len(seq) else 0.0,
            "pose_frac": float(pr["pose"].mean()) if len(seq) else 0.0,
            "hands_filled": stats["filled"], "hl_agree": round(stats["agree"] / stats["both"], 3) if stats["both"] else None,
            "extractor_version": seq.extractor_version}


def extract_image(path: str, out: str) -> dict:
    import cv2

    from app.landmarks.extractor import _face, _hand, _pose
    from app.landmarks.raw import RawSequence

    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if bgr is None:
        raise ValueError("cannot decode image")
    h0, w0 = bgr.shape[:2]
    img = _mp_image(_resize(bgr))
    res = _holistic_image().detect(img)
    pose, left, right, face = (_pose(res.pose_landmarks)[None], _hand(res.left_hand_landmarks)[None],
                               _hand(res.right_hand_landmarks)[None], _face(res.face_landmarks)[None])
    holistic_hands = int(not np.isnan(left[0, 0, 0])) + int(not np.isnan(right[0, 0, 0]))
    stats = {"filled": 0, "both": 0, "agree": 0}
    fill_hands(left, right, 0, hand_results(_hands_image().detect(img)), stats)
    seq = RawSequence(pose, left, right, face, np.zeros((1,), np.float64), w0, h0, "image-holistic-v1+hands")
    seq.save(Path(out))
    pr = seq.presence()
    return {"n_frames": 1, "fps": None, "hands_frac": float((pr["left"] | pr["right"]).mean()),
            "n_hands": int(pr["left"][0]) + int(pr["right"][0]), "holistic_hands": holistic_hands,
            "hands_filled": stats["filled"], "pose_frac": float(pr["pose"].mean()),
            "width": w0, "height": h0, "extractor_version": seq.extractor_version}


def work(mode: str, src: str, out: str, step_fps: float = 15.0, max_s: float = 20.0) -> dict:
    """Process entry point: never raises (failures become manifest rows)."""
    t0 = time.time()
    try:
        if mode == "full":
            from app.isl.data.ingest_bridge import ingest_module

            info = ingest_module().extract_full(src, out, step_fps=step_fps, max_s=max_s)
            info["extractor_version"] = "holistic-v1"
        elif mode == "hands":
            info = extract_hands_video(src, out, step_fps=step_fps, max_s=max_s)
        elif mode == "image":
            info = extract_image(src, out)
        else:
            raise ValueError(f"unknown mode {mode}")
        info["ok"] = True
    except Exception as e:  # noqa: BLE001
        info = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    info["secs"] = round(time.time() - t0, 2)
    return info
