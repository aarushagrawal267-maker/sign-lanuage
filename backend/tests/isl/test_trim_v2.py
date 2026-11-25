"""trim-2: cap fallback, second cut, truncated title-card clips, splice/join helpers, contact sheets and the
trim bench (labels file, sampling, scoring). Synthetic data only, no network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.isl.data.trim import (
    METHOD_CONFIDENCE,
    TrimConfig,
    activity_signal,
    cap_scores,
    find_first_sign,
    trim_record_for,
)
from app.isl.data.trim_bench import (
    LABELS_PATH,
    load_labels,
    score_errors,
    stop_group,
    stratified_sample,
    word_group,
)
from app.isl.data.trim_synth import align_concat, splice_out

from .helpers import make_trim_sequence


def test_continuous_signing_uses_cap_prior_and_second_cut():
    # sign straight into the explanation (no rest, no dip, no pause): the cap fallback must stay near the expected
    # sign length (~2.2 s) and not run to the 4 s window edge; a second cut is offered at least 400 ms away
    raw = make_trim_sequence([("rest", 6), ("sign", 26), ("expl", 70)])
    res = find_first_sign(raw, "max_s")
    assert res.method == "cap" and res.confidence == METHOD_CONFIDENCE["cap"]
    t = raw.t_ms
    kept_s = (t[res.end - 1] - t[res.start]) / 1000
    assert 1.0 <= kept_s <= 4.2
    assert res.second is not None and res.second in res.alternatives
    assert abs(t[res.second - 1] - t[res.end - 1]) >= TrimConfig().second_sep_ms
    rec = trim_record_for("islrtc/A/x.mp4", raw, "max_s", "word")
    assert rec["second"] == res.second and rec["trim_version"] == TrimConfig().version


def test_cap_scores_prefer_low_hands():
    raw = make_trim_sequence([("rest", 6), ("sign", 20), ("low", 3), ("expl", 60)])
    sig = activity_signal(raw)
    t0 = sig["t"][6]
    lo, sc = cap_scores(sig, t0, 1)
    assert len(sc) > 0
    best = lo + int(np.argmax(sc))
    assert 26 <= best <= 29  # the 'low' frames


def test_rest_cut_keeps_next_rest_as_second():
    # compound: part, short rest, part, rest, explanation -> main cut at the first rest, second at the next
    raw = make_trim_sequence([("rest", 6), ("sign", 14), ("rest", 4), ("sign", 14), ("rest", 6), ("expl", 30)])
    res = find_first_sign(raw, "max_s", n_words=1)
    assert res.method == "rest" and abs(res.end - 22) <= 2
    assert res.second is not None and abs(res.second - 40) <= 3


def test_bounded_long_compound_kept():
    # an eof clip with a 6 s sign (long compounds exist: 5-7 s) is kept whole instead of being cut at a dip
    raw = make_trim_sequence([("rest", 6), ("sign", 30), ("low", 3), ("sign", 40), ("rest", 6)])
    res = find_first_sign(raw, "eof")
    assert res.method == "source_end" and abs(res.end - 81) <= 2
    assert res.second is not None and abs(res.second - 38) <= 3  # the dip between the parts


def test_card_clip_cut_mid_sign_is_flagged():
    raw = make_trim_sequence([("rest", 8), ("sign", 10)])  # ends with the hands up
    res = find_first_sign(raw, "card")
    assert res.method == "truncated" and res.confidence <= 0.1 and res.end == len(raw)
    ok = find_first_sign(make_trim_sequence([("rest", 8), ("sign", 16), ("rest", 4)]), "card")
    assert ok.method == "source_end"
    # eof clips that end with the hands up are not flagged (the video simply ends)
    assert find_first_sign(raw, "eof").method == "source_end"


def test_splice_out_and_align_concat():
    raw = make_trim_sequence([("rest", 6), ("sign", 20), ("rest", 8), ("expl", 30)])
    x = splice_out(raw, 26, 34, bridge=2)
    assert len(x) == len(raw) - 8 + 2
    assert np.allclose(np.diff(x.t_ms), np.diff(raw.t_ms)[0])
    np.testing.assert_array_equal(x.pose[:26], raw.pose[:26])
    np.testing.assert_array_equal(x.pose[28:], raw.pose[34:])
    mid = x.pose[26, 15, :2]
    assert np.all(np.isfinite(mid))
    with pytest.raises(ValueError):
        splice_out(raw, 0, 5)
    # spliced: no rest left -> the trimmer must not end at a rest; end within ~1.5 s of the true end
    res = find_first_sign(x, "max_s")
    assert res.method != "rest"
    a = make_trim_sequence([("rest", 4), ("sign", 20), ("rest", 4)], seed=1)
    b = make_trim_sequence([("expl", 30)], seed=2, center=np.array([0.3, 0.5]), sw=0.3)
    j = align_concat(a, b, gap_frames=2)
    assert len(j) == len(a) + 2 + len(b)
    # B was moved into A's body frame: shoulder centres agree
    ca = np.nanmedian((a.pose[:, 11, :2] + a.pose[:, 12, :2]) / 2, axis=0)
    cb = np.nanmedian((j.pose[-30:, 11, :2] + j.pose[-30:, 12, :2]) / 2, axis=0)
    assert np.allclose(ca, cb, atol=1e-3)


def test_contact_sheet_from_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from app.isl.data.trim_sheet import SheetConfig, contact_sheet

    vid = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 36))
    if not w.isOpened():
        pytest.skip("no MJPG writer")
    for i in range(30):  # 3 s, brightness ramps so tiles differ
        w.write(np.full((36, 64, 3), 8 * i, np.uint8))
    w.release()
    out = tmp_path / "s.png"
    info = contact_sheet(vid, out, "t", SheetConfig(step_ms=500, cols=4, tile_w=32), kept=(500, 1500),
                         marks={"cut": [1500.0], "label": [2000.0]})
    assert out.stat().st_size > 500
    assert info["tiles"] == 6 and info["times_ms"][:2] == [0.0, 500.0]
    img = cv2.imread(str(out))
    assert img.shape[1] == 4 * (32 + 2)


def test_labels_file_is_valid():
    labels = load_labels(LABELS_PATH)
    assert len(labels) >= 80
    for cid, r in labels.items():
        assert cid.startswith("islrtc/") and r["status"] in ("ok", "unclear", "guess", "not_a_sign")
        assert (r["end_ms"] is None) == (r["status"] in ("unclear", "not_a_sign"))
        assert r["split"] in ("dev", "test", "holdout")
    assert sum(r["status"] == "ok" for r in labels.values()) >= 40


def test_sampling_and_scoring():
    recs = []
    for i in range(40):
        stop = ["max_s", "eof", "max_s+posegap", "card"][i % 4]
        name = ["Able", "Bank_Account", "Big_Red_Bus", "Abroad_Overseas"][i // 10]
        recs.append({"id": f"islrtc/A/{name}_{i}.mp4", "stop": stop, "n_frames": 50, "duration_s": 4.0})
    # strata and splits are assigned; a clip is drawn at most once even if it is in two strata (card)
    s = stratified_sample(recs, seed=1, alloc={("max_s", "1w"): 2, ("bounded", "1w"): 2, ("bounded", "_card"): 1})
    assert 1 <= len(s) <= 5 and all(x["split"] in ("dev", "test") for x in s)
    assert len({x["id"] for x in s}) == len(s)
    assert stop_group("max_s") == "max_s" and stop_group("max_s+posegap") == "max_s+posegap"
    assert stop_group("eof") == stop_group("card+posegap") == "bounded"
    assert word_group("synonym_list", "a b") == "syn" and word_group("multiword", "a b c") == "3w+"
    sc = score_errors([0, 100, -400, 500, 900])
    assert sc["n"] == 5 and sc["within_300ms"] == 0.4 and sc["early"] == 0.2 and sc["late"] == 0.4
    assert sc["usable"] == 0.6
    assert score_errors([]) == {"n": 0}


def test_bench_cli_labels_json(tmp_path, monkeypatch):
    # the bench CLI runs on an empty data root without network (no clips -> empty scores)
    from app.isl.data import trim_bench

    monkeypatch.setenv("SLR_ISL_DIR", str(tmp_path))
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "islrtc.jsonl").write_text("")
    lab = tmp_path / "labels.jsonl"
    lab.write_text(json.dumps({"id": "islrtc/A/x.mp4", "end_ms": 1000, "status": "ok", "split": "test"}) + "\n")
    monkeypatch.setattr(trim_bench, "LABELS_PATH", Path(lab))
    out = trim_bench.eval_labels(trim_bench._paths(), trim_bench.load_labels(Path(lab)))
    assert out["all"] == {"n": 0}


def test_learned_refine_cap_smoke(tmp_path):
    pytest.importorskip("torch")
    from app.isl.data import trim_learned as tl

    raws = [make_trim_sequence([("rest", 6), ("sign", 20 + i % 4), ("low", 6), ("expl", 50)], seed=i) for i in range(6)]
    assert tl.splice_point(raws[0]) is not None
    assert tl.splice_point(make_trim_sequence([("rest", 6), ("sign", 60)])) is None
    ex = tl.spliced_examples(raws, bridges=(0, 2))
    assert len(ex) == 6 * 3 and all(0 < e < len(x) for x, e in ex)
    model = tl.train(ex, epochs=2, threads=1)
    path = tmp_path / "m.pt"
    tl.save(model, path, {"half": "tune"})
    model2, meta = tl.load(path)
    assert meta["version"] == tl.MODEL_VERSION and meta["half"] == "tune"
    cont = make_trim_sequence([("rest", 6), ("sign", 26), ("expl", 70)])
    st1 = find_first_sign(cont, "max_s")
    out = tl.refine_cap(cont, st1, model2, threshold=0.0)  # threshold 0: always take the learned peak
    assert out.method == "cap" and out.start == st1.start and out.end > out.start
    if out.end != st1.end:
        assert any("trim-learned" in n for n in out.notes)
    rest = find_first_sign(make_trim_sequence([("rest", 6), ("sign", 20), ("rest", 6), ("expl", 30)]), "max_s")
    assert tl.refine_cap(cont, rest, model2) is rest  # non-cap results are untouched
