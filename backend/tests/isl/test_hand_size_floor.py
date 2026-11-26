"""FeatureConfig.hand_size_floor (26 Nov 2026): a foreshortened hand (wrist-to-middle-MCP axis along the camera axis)
has a ~0 2D size, which blew the hand-shape features up 10-1000x and caused the pre-training loss spikes."""

import numpy as np

from app.features.build import _hand_shape
from app.features.spec import HAND_MIDDLE_MCP, HAND_WRIST, FeatureConfig
from app.isl.training.common import ISL_FEATURES


def _hand(size2d: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.05, size=(4, 21, 3)).astype(np.float32)
    h[:, HAND_WRIST] = 0.0
    h[:, HAND_MIDDLE_MCP] = [size2d, 0.0, 0.3]  # long in z, tiny in 2D: foreshortened
    return h


def test_floor_bounds_foreshortened_hand():
    pres = np.ones(4, dtype=bool)
    raw, _ = _hand_shape(_hand(0.001), pres)
    floored, _ = _hand_shape(_hand(0.001), pres, 0.1)
    assert np.abs(raw).max() > 100  # the bug
    assert np.abs(floored).max() <= 5.0


def test_floor_zero_is_byte_identical_and_normal_hands_unchanged():
    pres = np.array([True, True, False, True])
    h = _hand(0.3, seed=1)
    a, pa = _hand_shape(h, pres)
    b, pb = _hand_shape(h, pres, 0.0)
    assert np.array_equal(a, b) and np.array_equal(pa, pb)
    c, _ = _hand_shape(h, pres, 0.1)  # size 0.3 is above the floor: same values
    assert np.array_equal(a, c)


def test_config_round_trip_and_old_configs_unchanged():
    assert "hand_size_floor" not in FeatureConfig().to_dict()  # ASL / old ISL configs stay byte-identical
    assert FeatureConfig.from_dict(FeatureConfig().to_dict()).hand_size_floor == 0.0
    assert ISL_FEATURES.hand_size_floor == 0.1
    assert FeatureConfig.from_dict(ISL_FEATURES.to_dict()) == ISL_FEATURES
