"""pose-format (MediaPipe Holistic) -> ``RawSequence``, matching ``HolisticExtractor`` (app/landmarks/extractor.py).

Reuses the dependency-free reader ``app.isl.data.isign.read_pose`` (pose-format v0.1 / v0.2). Same mapping as
``isign.pose_to_raw`` (whose z/W scaling was fixed to raw z in the phase-2 review), plus:

* **z is not divided by the width.** pose-format's own Holistic writer (``pose_format.utils.holistic``)
  stores ``x * width, y * height, z`` (raw MediaPipe z), and the GloSL card says "z raw". Our extractor stores
  raw MediaPipe z too, so z is copied as is (``z_mode="raw"``). ``z_mode="width"`` divides by the width
  for files whose z was scaled like x.
* Per-part presence is decided per frame from the confidence channel (missing = (0, 0, 0) with confidence 0):
  pose frame present when any of the 33 body points has confidence > 0 (the visibility becomes channel 4,
  like ``p.visibility`` in the extractor); a hand / the face is present when any of its points has
  confidence > 0; absent parts are all-NaN, as the extractor writes them.
* Face: the app's 20-point ``FACE_IDX`` subset of the 468-point mesh (the extractor's 478-point mesh has the
  same first 468 points; the extra 10 are irises, unused).
* Hands: pose-format / legacy Holistic ``LEFT_HAND_LANDMARKS`` is the signer's left hand (checked on GloSL:
  its wrist lies on pose point 15, the left wrist, as in our extractor's output), so no swap.
* ``t_ms = i * 1000 / fps`` like the extractor's frame timestamps (before any stride).
"""

from __future__ import annotations

import numpy as np

from app.features.spec import FACE_IDX, N_FACE, N_HAND, N_POSE
from app.isl.data.isign import PoseFormatError, read_pose
from app.landmarks.raw import RawSequence


def _locate(components: list[dict], name: str) -> tuple[int, int] | None:
    off = 0
    for c in components:
        if c["name"].upper() == name:
            return off, len(c["points"])
        off += len(c["points"])
    return None


def parsed_pose_to_raw(p: dict, extractor_version: str, z_mode: str = "raw", person: int = 0) -> RawSequence:
    """Convert the dict returned by ``read_pose`` (pixel coordinates + confidence)."""
    if z_mode not in ("raw", "width"):
        raise ValueError(f"z_mode must be 'raw' or 'width', not {z_mode!r}")
    W, H = max(1, int(p["width"])), max(1, int(p["height"]))
    data, conf = p["data"], p["confidence"]
    F = int(data.shape[0])
    if F == 0:
        raise PoseFormatError("pose file has no frames")
    person = min(person, data.shape[1] - 1)
    d = np.asarray(data[:, person], dtype=np.float64)
    c = np.asarray(conf[:, person], dtype=np.float64)
    dims = d.shape[-1]
    zdiv = float(W) if z_mode == "width" else 1.0

    def block(name: str, n_needed: int):
        loc = _locate(p["components"], name)
        if loc is None:
            return None, None
        off, n = loc
        if n < n_needed:
            raise PoseFormatError(f"{name} has {n} points, expected >= {n_needed}")
        xyz = np.zeros((F, n, 3))
        xyz[..., 0] = d[:, off:off + n, 0] / W
        xyz[..., 1] = d[:, off:off + n, 1] / H
        if dims >= 3:
            xyz[..., 2] = d[:, off:off + n, 2] / zdiv
        return xyz, c[:, off:off + n]

    pxyz, pc = block("POSE_LANDMARKS", N_POSE)
    if pxyz is None:
        raise PoseFormatError("no POSE_LANDMARKS component")
    pose = np.full((F, N_POSE, 4), np.nan, np.float32)
    present = pc[:, :N_POSE].max(axis=1) > 0
    pose[present, :, :3] = pxyz[present, :N_POSE]
    pose[present, :, 3] = pc[present, :N_POSE]

    hands = []
    for name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
        hxyz, hc = block(name, N_HAND)
        h = np.full((F, N_HAND, 3), np.nan, np.float32)
        if hxyz is not None:
            ok = hc[:, :N_HAND].max(axis=1) > 0
            h[ok] = hxyz[ok, :N_HAND]
        hands.append(h)

    face = np.full((F, N_FACE, 3), np.nan, np.float32)
    fxyz, fc = block("FACE_LANDMARKS", max(FACE_IDX) + 1)
    if fxyz is not None:
        ok = fc.max(axis=1) > 0
        face[ok] = fxyz[ok][:, FACE_IDX]

    fps = float(p["fps"]) if p.get("fps") and p["fps"] > 0 else 25.0
    t_ms = np.arange(F, dtype=np.float64) * 1000.0 / fps
    return RawSequence(pose, hands[0], hands[1], face, t_ms, W, H, extractor_version)


def holistic_pose_to_raw(buf: bytes, extractor_version: str, z_mode: str = "raw") -> tuple[RawSequence, dict]:
    """Bytes of a ``.pose`` file -> (RawSequence at native fps, header info {fps, width, height, version})."""
    p = read_pose(buf)
    seq = parsed_pose_to_raw(p, extractor_version, z_mode=z_mode)
    return seq, {"fps": float(p["fps"]), "width": int(p["width"]), "height": int(p["height"]),
                 "version": p["version"], "components": [c["name"] for c in p["components"]]}
