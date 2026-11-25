"""MediaPipe Holistic landmark extraction for video files (offline path).

The browser runs the same ``holistic_landmarker.task`` model through the
MediaPipe Tasks Vision WASM build for live webcam input.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.features.spec import FACE_IDX, N_FACE, N_HAND, N_POSE
from app.landmarks.raw import RawSequence

log = logging.getLogger(__name__)
os.environ.setdefault("GLOG_minloglevel", "2")


class ExtractorUnavailable(RuntimeError):
    pass


class HolisticExtractor:
    """Wraps MediaPipe's HolisticLandmarker in VIDEO mode.

    A fresh landmarker is created per video so tracking state never leaks
    between clips.
    """

    def __init__(self, model_path: Path | None = None, max_side: int = 960):
        self.model_path = Path(model_path or get_settings().holistic_model_path)
        if not self.model_path.exists():
            raise ExtractorUnavailable(
                f"MediaPipe model not found at {self.model_path}. Run setup.ps1 (or scripts/fetch_models.py)."
            )
        self.max_side = max_side
        try:
            from mediapipe.tasks.python import BaseOptions, vision  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ExtractorUnavailable(f"mediapipe is not importable: {e}") from e

    def _create(self):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            min_face_detection_confidence=0.5,
        )
        try:
            return vision.HolisticLandmarker.create_from_options(opts)
        except OSError as e:  # native library missing a system dependency (e.g. libEGL on Linux)
            raise ExtractorUnavailable(
                f"MediaPipe could not load its native library ({e}). On Debian/Ubuntu install the system "
                "packages with:  sudo apt install libegl1 libgles2 libgl1"
            ) from e

    def extract_frames(
        self,
        frames: Iterator[tuple[np.ndarray, float]],
        width: int,
        height: int,
        n_hint: int = 0,
        progress: Callable[[int], None] | None = None,
    ) -> RawSequence:
        """``frames`` yields (BGR image, timestamp_ms)."""
        import mediapipe as mp

        poses, lefts, rights, faces, ts = [], [], [], [], []
        lm = self._create()
        try:
            last_ts = -1
            for i, (bgr, t_ms) in enumerate(frames):
                h, w = bgr.shape[:2]
                scale = self.max_side / max(h, w)
                if scale < 1:
                    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ts_int = max(int(round(t_ms)), last_ts + 1)  # must be strictly increasing
                last_ts = ts_int
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts_int)
                poses.append(_pose(res.pose_landmarks))
                lefts.append(_hand(res.left_hand_landmarks))
                rights.append(_hand(res.right_hand_landmarks))
                faces.append(_face(res.face_landmarks))
                ts.append(float(t_ms))
                if progress:
                    progress(i + 1)
        finally:
            try:
                lm.close()
            except Exception:  # pragma: no cover
                pass
        if not ts:
            return RawSequence.empty(width, height)
        return RawSequence(
            pose=np.stack(poses), left=np.stack(lefts), right=np.stack(rights), face=np.stack(faces),
            t_ms=np.asarray(ts), width=width, height=height,
        )

    def extract_video(self, path: Path, start: int = 0, end: int | None = None,
                      progress: Callable[[int], None] | None = None) -> tuple[RawSequence, float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps > 240:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        def gen():
            idx = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx >= start and (end is None or idx < end):
                        yield frame, (idx - start) * 1000.0 / fps
                    idx += 1
                    if end is not None and idx >= end:
                        break
            finally:
                cap.release()

        seq = self.extract_frames(gen(), w, h, progress=progress)
        return seq, fps


def _nan(shape):
    return np.full(shape, np.nan, np.float32)


def _pose(lms) -> np.ndarray:
    if not lms:
        return _nan((N_POSE, 4))
    lms = lms[0] if isinstance(lms[0], list) else lms
    out = np.array([[p.x, p.y, p.z, p.visibility if p.visibility is not None else 1.0] for p in lms], np.float32)
    return out if out.shape == (N_POSE, 4) else _nan((N_POSE, 4))


def _hand(lms) -> np.ndarray:
    if not lms:
        return _nan((N_HAND, 3))
    lms = lms[0] if isinstance(lms[0], list) else lms
    out = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
    return out if out.shape == (N_HAND, 3) else _nan((N_HAND, 3))


def _face(lms) -> np.ndarray:
    if not lms:
        return _nan((N_FACE, 3))
    lms = lms[0] if isinstance(lms[0], list) else lms
    if len(lms) <= max(FACE_IDX):
        return _nan((N_FACE, 3))
    return np.array([[lms[i].x, lms[i].y, lms[i].z] for i in FACE_IDX], np.float32)
