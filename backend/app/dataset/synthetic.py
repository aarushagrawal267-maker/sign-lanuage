"""Synthetic landmark sequences - FOR AUTOMATED TESTS ONLY.

These are procedurally generated stick-figure landmark trajectories (no video,
no real signing) used to verify that the pipeline is wired correctly: features
have the right shape, a model can learn class-dependent motion, the streaming
engine segments and emits words, and the API works end to end. Anything trained
on them is flagged ``is_synthetic`` and must never be presented as an ASL result.
"""

from __future__ import annotations

import numpy as np

from app.features.spec import FACE_IDX, N_FACE, N_HAND, N_POSE
from app.landmarks.raw import RawSequence


def _hand_template(curl: float, spread: float) -> np.ndarray:
    pts = np.zeros((N_HAND, 3), np.float32)
    base_angles = np.deg2rad(np.array([-50, -20, 0, 20, 40]) * spread)
    for f in range(5):
        ang = base_angles[f]
        r = 0.0
        for j in range(4):
            seg = 0.035 if f else 0.03
            bend = curl * (j + 1) * 0.35
            r += seg
            pts[1 + f * 4 + j] = [np.sin(ang) * r * (1 - bend * 0.5), -np.cos(ang) * r * (1 - bend), -0.01 * j * curl]
    return pts


def make_sequence(class_index: int, signer: int = 0, n_frames: int | None = None, fps: float = 25.0,
                  rng: np.random.Generator | None = None, width: int = 640, height: int = 480,
                  rest_frames: int = 0, n_classes: int = 10) -> RawSequence:
    rng = rng or np.random.default_rng(class_index * 1000 + signer)
    srng = np.random.default_rng(10_000 + signer)  # signer-specific body
    T = n_frames or int(rng.integers(35, 70))
    body_shift = srng.uniform(-0.06, 0.06, 2)
    body_scale = srng.uniform(0.85, 1.15)
    speed = rng.uniform(0.85, 1.15)
    cr = np.random.default_rng(class_index)  # class-defining parameters
    amp = cr.uniform(0.04, 0.12, 2)
    freq = cr.uniform(0.6, 2.2)
    phase = cr.uniform(0, 2 * np.pi)
    base = np.array([cr.uniform(0.38, 0.62), cr.uniform(0.32, 0.6)])
    curl = cr.uniform(0, 1)
    spread = cr.uniform(0.5, 1.5)
    two_hands = cr.random() < 0.5

    pose = np.full((T, N_POSE, 4), np.nan, np.float32)
    left = np.full((T, N_HAND, 3), np.nan, np.float32)
    right = np.full((T, N_HAND, 3), np.nan, np.float32)
    face = np.full((T, N_FACE, 3), np.nan, np.float32)
    center = np.array([0.5, 0.62]) + body_shift
    sw = 0.22 * body_scale
    for i in range(T):
        u = i / max(1, T - 1)
        active = rest_frames <= i < T - rest_frames
        tt = u * speed * 2 * np.pi * freq + phase
        p = np.zeros((N_POSE, 4), np.float32)
        p[:, 3] = 0.99
        p[:, :2] = center
        p[0, :2] = center + [0, -0.22 * body_scale]
        p[11, :2] = center + [sw / 2, 0]   # signer's left shoulder appears on image right
        p[12, :2] = center + [-sw / 2, 0]
        p[23, :2] = center + [sw / 2.5, 0.35]
        p[24, :2] = center + [-sw / 2.5, 0.35]
        hand_r = base + body_shift + amp * [np.cos(tt), np.sin(tt)]
        hand_l = np.array([1 - hand_r[0] + 2 * body_shift[0], hand_r[1]])
        p[16, :2] = hand_r if active else center + [-sw / 2, 0.3]
        p[15, :2] = hand_l if (active and two_hands) else center + [sw / 2, 0.3]
        p[14, :2] = (p[12, :2] + p[16, :2]) / 2 + [-0.03, 0.05]
        p[13, :2] = (p[11, :2] + p[15, :2]) / 2 + [0.03, 0.05]
        p[:, :2] += rng.normal(0, 0.003, (N_POSE, 2))
        pose[i] = p
        if active:
            h = _hand_template(np.clip(curl + 0.15 * np.sin(tt), 0, 1), spread) * body_scale
            right[i] = h + [*hand_r, 0]
            right[i, :, :2] += rng.normal(0, 0.002, (N_HAND, 2))
            if two_hands:
                hl = h.copy()
                hl[:, 0] *= -1
                left[i] = hl + [*hand_l, 0]
        nose = p[0, :2]
        offs = np.linspace(-0.04, 0.04, N_FACE)
        face[i, :, 0] = nose[0] + offs
        face[i, :, 1] = nose[1] + 0.02 * np.cos(np.arange(N_FACE))
        face[i, :, 2] = 0
    # random short dropouts of the right hand
    for _ in range(int(rng.integers(0, 3))):
        a = int(rng.integers(0, T))
        right[a : a + int(rng.integers(1, 4))] = np.nan
    # store x as fraction of width (undo aspect so features re-apply it)
    asp = width / height
    for arr in (pose, left, right, face):
        arr[..., 0] = (arr[..., 0] - 0.5) / asp + 0.5
    for arr in (left, right):  # MediaPipe hand z shares the width-normalized x scale
        arr[..., 2] = arr[..., 2] / asp
    t_ms = np.arange(T) * 1000.0 / fps
    assert len(FACE_IDX) == N_FACE
    return RawSequence(pose, left, right, face, t_ms, width, height)


def frames_for_stream(seq: RawSequence, t0: float = 0.0) -> list[dict]:
    out = []
    for i in range(len(seq)):
        f = seq.frame_to_json(i)
        f["t"] = t0 + float(seq.t_ms[i])
        out.append(f)
    return out


def idle_frames(n: int, t0: float, fps: float = 25.0, width: int = 640, height: int = 480) -> list[dict]:
    """Body visible, no hands - used to separate signs in streaming tests."""
    seq = make_sequence(0, n_frames=max(2, n), rest_frames=n, width=width, height=height)
    frames = frames_for_stream(seq, t0)[:n]
    for k, f in enumerate(frames):
        f["left"] = None
        f["right"] = None
        f["t"] = t0 + k * 1000.0 / fps
    return frames


def populate_synthetic_db(n_classes: int = 6, n_signers: int = 8, per_signer: int = 2, seed: int = 0) -> dict:
    """Create a self-contained synthetic dataset in the CURRENT data dir (tests only)."""
    from app.core.config import get_settings
    from app.db.models import Gloss, LandmarkFile, MetaKV, Video
    from app.db.session import session_scope
    from app.features.spec import EXTRACTOR_VERSION

    s = get_settings()
    rng = np.random.default_rng(seed)
    with session_scope() as db:
        for c in range(n_classes):
            db.add(Gloss(class_index=c, gloss=f"synthetic_{c:02d}"))
        db.flush()
        n = 0
        for sg in range(n_signers):
            split = "train" if sg < n_signers - 2 else ("val" if sg == n_signers - 2 else "test")
            for c in range(n_classes):
                for k in range(per_signer):
                    vid = f"syn{sg:02d}{c:02d}{k}"
                    seq = make_sequence(c, sg, rng=rng, n_classes=n_classes)
                    path = s.landmarks_dir / f"{vid}.npz"
                    seq.save(path)
                    rates = seq.detection_rates()
                    db.add(Video(video_id=vid, class_index=c, gloss=f"synthetic_{c:02d}", signer_id=sg, split=split,
                                 source="synthetic", url="synthetic://", annotated_fps=25, frame_start=1, frame_end=-1,
                                 status="valid", clip_frames=len(seq), duration_s=len(seq) / 25.0, measured_fps=25.0,
                                 width=seq.width, height=seq.height))
                    db.flush()
                    db.add(LandmarkFile(video_id=vid, path=str(path), n_frames=len(seq), pose_rate=rates["pose"],
                                        left_rate=rates["left"], right_rate=rates["right"],
                                        any_hand_rate=rates["any_hand"], face_rate=rates["face"],
                                        extractor_version=EXTRACTOR_VERSION))
                    n += 1
        db.add(MetaKV(key="synthetic", value={"warning": "synthetic test data", "videos": n}))
    return {"videos": n}
