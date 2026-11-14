import cv2
import numpy as np
import pytest

from app.core.config import get_settings


@pytest.fixture
def extractor():
    from app.landmarks.extractor import ExtractorUnavailable, HolisticExtractor

    try:
        return HolisticExtractor()
    except ExtractorUnavailable as e:
        pytest.skip(str(e))


def test_blank_video_yields_missing_landmarks(tmp_path, extractor):
    p = tmp_path / "blank.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 120))
    for _ in range(12):
        vw.write(np.zeros((120, 160, 3), np.uint8))
    vw.release()
    seq, fps = extractor.extract_video(p, 2, 10)
    assert len(seq) == 8 and abs(fps - 25) < 0.5
    assert np.isnan(seq.pose).all() and np.isnan(seq.left).all()  # nothing invented
    assert seq.detection_rates()["pose"] == 0.0
    out = tmp_path / "x.npz"
    seq.save(out)
    from app.landmarks.raw import RawSequence

    back = RawSequence.load(out)
    assert back.width == 160 and len(back) == 8


def test_model_asset_location():
    assert get_settings().holistic_model_path.name == "holistic_landmarker.task"
