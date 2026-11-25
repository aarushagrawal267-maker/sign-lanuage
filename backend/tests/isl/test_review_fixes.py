"""Regression tests for fixes made in the phase-2 code review (no network)."""

from types import SimpleNamespace

from app.isl.cli import _eval_clip_ids
from app.isl.training.multisource import DedupIndex


def _clip(cid: str):
    return SimpleNamespace(clip_id=cid)


def test_eval_clip_ids_cover_whole_duplicate_groups():
    """Pre-training must not see a copy of an evaluation clip that another source carries under another id."""
    d = DedupIndex()
    d.union("include/test_a", "wslp/copy_of_a", "same_video")
    d.union("isl500/user009/x", "isl500/user009/x_dup", "same_file")
    d._find("include/train_b")
    d._find("wslp/other")
    data = SimpleNamespace(
        include=[_clip("include/test_a"), _clip("include/train_b")],
        assign={"include/test_a": "test", "include/train_b": "train"},
        extra=[_clip("isl500/user009/x")],
        extra_assign={"isl500/user009/x": "test"},
        dedup=d,
    )
    ids = _eval_clip_ids(data)
    assert {"include/test_a", "wslp/copy_of_a", "isl500/user009/x", "isl500/user009/x_dup"} <= ids
    assert "include/train_b" not in ids and "wslp/other" not in ids


def test_eval_clip_ids_without_dedup_index():
    data = SimpleNamespace(include=[_clip("a")], assign={"a": "val"}, extra=[], extra_assign={})
    assert _eval_clip_ids(data) == {"a"}
