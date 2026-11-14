import json

import pytest

from app.dataset.annotations import AnnotationError, parse_wlasl100
from app.splits.splits import VideoMeta, leakage_report, official_assignment, signer_groups, signer_holdout, signer_kfold


def _fake_annotations(tmp_path, n_glosses=105, per=6, mismatch=False):
    data, nslt = [], {}
    vid = 0
    for g in range(n_glosses):
        inst = []
        for k in range(per):
            vid += 1
            split = ["train", "train", "train", "train", "val", "test"][k]
            inst.append({"video_id": f"{vid:05d}", "url": f"https://www.youtube.com/watch?v=abcdefghi{g:02d}" if k == 0 else f"https://x.org/{vid}.mp4",
                         "signer_id": (g + k) % 12, "split": split, "fps": 25, "frame_start": 1, "frame_end": -1,
                         "bbox": [0, 0, 10, 10], "source": "x", "variation_id": 0, "instance_id": k})
            if g < 100:
                nslt[f"{vid:05d}"] = {"subset": split, "action": [g, 1, 50]}
        data.append({"gloss": f"g{g}", "instances": inst})
    if mismatch:
        first = next(iter(nslt))
        nslt[first]["subset"] = "test" if nslt[first]["subset"] != "test" else "train"
    a, b = tmp_path / "ann.json", tmp_path / "nslt.json"
    a.write_text(json.dumps(data))
    b.write_text(json.dumps(nslt))
    return a, b


def test_wlasl100_is_first_100_glosses_and_cross_checked(tmp_path):
    a, b = _fake_annotations(tmp_path)
    d = parse_wlasl100(a, b)
    assert len(d.glosses) == 100 and d.glosses[0] == "g0" and d.glosses[-1] == "g99"
    assert len(d.instances) == 600
    assert d.checks["consistent"] is True
    assert {i["class_index"] for i in d.instances} == set(range(100))


def test_inconsistent_split_is_detected(tmp_path):
    a, b = _fake_annotations(tmp_path, mismatch=True)
    d = parse_wlasl100(a, b)
    assert d.checks["consistent"] is False and len(d.checks["split_mismatch"]) == 1


def test_too_few_glosses_rejected(tmp_path):
    a, b = _fake_annotations(tmp_path, n_glosses=50)
    with pytest.raises(AnnotationError):
        parse_wlasl100(a, b)


def _meta(n_signers=20, per=10, classes=10):
    out = []
    for s in range(n_signers):
        for k in range(per):
            out.append(VideoMeta(f"{s:03d}{k:03d}", (s + k) % classes, s, "train" if k < 7 else ("val" if k < 9 else "test"),
                                 source_key=f"{s:03d}{k:03d}"))
    return out


def test_official_style_split_is_flagged_as_leaky():
    meta = _meta()
    rep = leakage_report(official_assignment(meta), meta, n_classes=10)
    assert rep["signer_independent"] is False
    assert rep["passed"] is False
    assert rep["eval_videos_with_train_signers"]["test"]["videos_with_train_signer"] > 0


def test_signer_holdout_has_no_signer_overlap():
    meta = _meta()
    a = signer_holdout(meta, seed=3, trials=200, n_classes=10)
    rep = leakage_report(a, meta, n_classes=10)
    assert rep["signer_independent"] and rep["passed"]
    assert all(rep["n_videos"][p] > 0 for p in ("train", "val", "test"))
    # every signer's videos in one partition
    parts = {}
    for m in meta:
        parts.setdefault(m.signer_id, set()).add(a[m.video_id])
    assert all(len(v) == 1 for v in parts.values())


def test_kfold_every_signer_tested_exactly_once():
    meta = _meta()
    folds = signer_kfold(meta, k=5)
    tested = {}
    for a in folds:
        assert leakage_report(a, meta, 10)["signer_independent"]
        for m in meta:
            if a[m.video_id] == "test":
                tested.setdefault(m.signer_id, set()).add(id(a))
    assert len(tested) == 20 and all(len(v) == 1 for v in tested.values())


def test_signers_sharing_a_source_file_are_grouped():
    meta = _meta(n_signers=6, per=4)
    meta[0].source_key = "shared"
    meta[-1].source_key = "shared"  # signer 0 and signer 5 share a raw file
    g = signer_groups(meta)
    assert g[0] == g[5]
    a = signer_holdout(meta, seed=0, trials=50, n_classes=10)
    assert a[meta[0].video_id] == a[meta[-1].video_id]
    assert leakage_report(a, meta, 10)["source_file_overlap"] == {"train-val": 0, "train-test": 0, "val-test": 0}
