import numpy as np
import pytest

from app.dataset.synthetic import make_sequence
from app.features.build import (
    AugmentConfig,
    FeatureError,
    augment,
    build_features,
    featurize,
    fill_gaps,
    to_body_frame,
)
from app.features.spec import FeatureConfig
from app.landmarks.raw import RawSequence


def _transform(seq: RawSequence, s: float, dx: float, dy: float) -> RawSequence:
    out = RawSequence(seq.pose.copy(), seq.left.copy(), seq.right.copy(), seq.face.copy(), seq.t_ms.copy(), seq.width, seq.height)
    for a in (out.pose, out.left, out.right, out.face):
        a[..., 0] = a[..., 0] * s + dx
        a[..., 1] = a[..., 1] * s + dy
    out.left[..., 2] *= s
    out.right[..., 2] *= s
    return out


@pytest.mark.parametrize("cfg", [FeatureConfig(), FeatureConfig(seq_len=32, use_face=False, use_velocity=False)])
def test_shape_and_layout(cfg):
    x = build_features(make_sequence(2, 1), cfg)
    assert x.shape == (cfg.seq_len, cfg.dim)
    assert np.isfinite(x).all()
    layout = cfg.layout()
    assert layout[-1]["start"] + layout[-1]["size"] == cfg.dim


def test_translation_and_scale_invariance():
    cfg = FeatureConfig()
    seq = make_sequence(4, 2)
    a = build_features(seq, cfg)
    b = build_features(_transform(seq, 0.7, 0.1, -0.05), cfg)
    assert np.abs(a - b).max() < 1e-4


def test_aspect_ratio_is_corrected():
    """Same physical scene seen through 4:3 and 16:9 frames yields the same features."""
    cfg = FeatureConfig()
    a = make_sequence(1, 1, width=640, height=480)
    b = make_sequence(1, 1, width=1280, height=720)
    assert np.abs(build_features(a, cfg) - build_features(b, cfg)).max() < 1e-3


def test_fill_gaps_only_short_interior_gaps():
    arr = np.arange(10, dtype=np.float64).reshape(10, 1, 1).repeat(2, axis=2)
    arr[[0, 3, 4, 7, 8, 9]] = np.nan  # leading, interior(2), trailing(3)
    out = fill_gaps(arr, max_gap=2)
    assert np.isnan(out[0, 0, 0])  # leading stays missing
    assert np.allclose(out[3:5, 0, 0], [3, 4])  # interior interpolated
    assert np.isnan(out[7:, 0, 0]).all()  # trailing stays missing
    out2 = fill_gaps(arr, max_gap=1)
    assert np.isnan(out2[3:5, 0, 0]).all()  # gap too long


def test_missing_hand_is_flagged_not_invented():
    cfg = FeatureConfig(seq_len=16)
    seq = make_sequence(0, 0)
    seq.left[:] = np.nan
    x = build_features(seq, cfg)
    flags = {b["name"]: b for b in cfg.layout()}
    lp = flags["left_present"]["start"]
    assert (x[:, lp] == 0).all()
    ls = flags["left_shape"]
    assert (x[:, ls["start"] : ls["start"] + ls["size"]] == 0).all()


def test_no_pose_raises():
    seq = make_sequence(0, 0)
    seq.pose[:] = np.nan
    with pytest.raises(FeatureError):
        build_features(seq, FeatureConfig())


def test_mirror_augmentation_swaps_hands():
    seq = make_sequence(3, 0)
    bf = to_body_frame(seq)
    cfg = AugmentConfig(p_mirror=1.0, rotate_deg=0, scale=0, shift=0, crop_frac=0, speed=0, landmark_dropout=0, jitter=0)
    m = augment(bf, np.random.default_rng(0), cfg)
    assert np.allclose(m.left[..., 0], -bf.right[..., 0])
    assert (m.present["left"] == bf.present["right"]).all()
    mm = augment(m, np.random.default_rng(0), cfg)
    assert np.allclose(featurize(mm, FeatureConfig()), featurize(bf, FeatureConfig()), atol=1e-5)


def test_random_augmentation_is_finite():
    rng = np.random.default_rng(0)
    for c in range(5):
        x = build_features(make_sequence(c, c), FeatureConfig(), rng=rng)
        assert np.isfinite(x).all()


def test_browser_frame_roundtrip():
    seq = make_sequence(2, 3)
    frames = [seq.frame_to_json(i) for i in range(len(seq))]
    back = RawSequence.from_frames(frames, seq.width, seq.height)
    cfg = FeatureConfig()
    assert np.abs(build_features(back, cfg) - build_features(seq, cfg)).max() < 5e-3


def test_from_frames_rejects_bad_shapes():
    with pytest.raises(ValueError):
        RawSequence.from_frames([{"t": 0, "pose": [[0, 0, 0]] * 33}], 640, 480)
