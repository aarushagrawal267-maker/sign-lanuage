"""Single source of truth for the raw landmark layout and the feature layout.

The browser fetches this spec from ``/api/features/spec`` so live webcam frames
are sent in exactly the layout that offline extraction stores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

EXTRACTOR_VERSION = "holistic-v1"

N_POSE = 33
N_HAND = 21

# MediaPipe Pose indices
POSE_NOSE = 0
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_ELBOW, POSE_R_ELBOW = 13, 14
POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_L_HIP, POSE_R_HIP = 23, 24

# Upper-body points used as features (hips are frequently outside a laptop
# webcam frame, so they are deliberately excluded).
POSE_FEATURE_IDX = [POSE_NOSE, POSE_L_SHOULDER, POSE_R_SHOULDER, POSE_L_ELBOW, POSE_R_ELBOW, POSE_L_WRIST, POSE_R_WRIST]

# Face-mesh subset (indices into the 478-point MediaPipe face mesh): mouth,
# eyebrows, eye corners, nose tip, chin and cheek contour. Non-manual signals
# (mouth shape, brow raise) and head location relative to the hands.
FACE_IDX = [
    1,  # nose tip
    152,  # chin
    234, 454,  # cheek contour left/right
    61, 291, 0, 17, 13, 14,  # mouth corners, outer upper/lower lip, inner lips
    70, 105, 107, 336, 334, 300,  # eyebrows
    33, 133, 362, 263,  # eye corners
]
N_FACE = len(FACE_IDX)

# Hand landmark indices
HAND_WRIST = 0
HAND_MIDDLE_MCP = 9


@dataclass(frozen=True)
class FeatureConfig:
    """Options that change the feature vector. Stored with every experiment."""

    seq_len: int = 48
    use_face: bool = True
    use_velocity: bool = True
    max_gap: int = 5  # interpolate hand/pose gaps up to this many frames

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> FeatureConfig:
        d = d or {}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    # --- layout -----------------------------------------------------------
    def static_blocks(self) -> list[tuple[str, int]]:
        blocks = [
            ("pose_xy", len(POSE_FEATURE_IDX) * 2),
            ("left_shape", (N_HAND - 1) * 3),
            ("left_pos", 2),
            ("right_shape", (N_HAND - 1) * 3),
            ("right_pos", 2),
        ]
        if self.use_face:
            blocks.append(("face_xy", N_FACE * 2))
        return blocks

    def flag_blocks(self) -> list[tuple[str, int]]:
        blocks = [("left_present", 1), ("right_present", 1)]
        if self.use_face:
            blocks.append(("face_present", 1))
        return blocks

    @property
    def static_dim(self) -> int:
        return sum(n for _, n in self.static_blocks())

    @property
    def flag_dim(self) -> int:
        return sum(n for _, n in self.flag_blocks())

    @property
    def dim(self) -> int:
        return self.static_dim * (2 if self.use_velocity else 1) + self.flag_dim

    def layout(self) -> list[dict]:
        out, off = [], 0
        blocks = list(self.static_blocks())
        if self.use_velocity:
            blocks += [(f"d_{n}", k) for n, k in self.static_blocks()]
        blocks += self.flag_blocks()
        for name, n in blocks:
            out.append({"name": name, "start": off, "size": n})
            off += n
        return out


def raw_spec() -> dict:
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "pose": {"count": N_POSE, "values": ["x", "y", "z", "visibility"]},
        "hand": {"count": N_HAND, "values": ["x", "y", "z"]},
        "face_indices": FACE_IDX,
        "face": {"count": N_FACE, "values": ["x", "y", "z"]},
        "coordinates": "MediaPipe normalized image coordinates of the UNMIRRORED frame; "
        "left/right are the signer's own hands as reported by Holistic",
        "pose_feature_indices": POSE_FEATURE_IDX,
    }
