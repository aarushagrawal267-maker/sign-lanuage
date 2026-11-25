"""ISL500 / ISL-DATA (``ISL500/ISL-DATA`` on Hugging Face): 500 ISL words x 15 signers, "research and academic
use only" -> **research profile**. Recorded in Goa (paths say ``ISL_Goa_Data``); no paper, origin undocumented.

Layout, inspected 25 Nov 2026 (the README's tree is out of date):

* ``Landmarks/MediaPipe/ISL_DATA_USER0NN/<name>.h5``: **86,760 files** (2.5 GB), far more than the 7,500
  videos. Each file has one dataset ``intermediate`` (T, 2, 21, 3) float32 and attrs ``native_fps``,
  ``target_fps`` (30), ``num_frames``. It holds **MediaPipe hands only**: index 0 = the signer's left hand,
  1 = right (checked against MMPose's left/right hands and against our own extraction), normalised x/y, raw z,
  missing hand = all zeros. No body, no face.
* ``Landmarks/Pose/User0NN/[Round1/]<name>.npy``: MMPose **COCO-WholeBody** (T, 133, 3) = x px, y px, score
  (17 body, 6 feet, 68 face in iBUG-300W order, 21 left hand, 21 right hand). 80,365 of the h5 clips have one
  (same name after :func:`clip_key`); the 6,391 others are hands-only and are skipped (our features need the
  body: shoulders set the scale).
* ``Videos/User0NN/<name>.mp4``: 500 per signer (one per word), 2560x1440 ~29.6 fps, about 21 MB each (159 GB).

Conversion (:func:`isl500_to_raw`, ``extractor_version = "isl500-mphands+mmpose-wb"``):

* hands: the h5 MediaPipe hands as they are (same model family as ours);
* pose (33, MediaPipe order): COCO-WholeBody body/feet points mapped to their MediaPipe equivalents, the
  MediaPipe eye inner/outer and mouth points from the 68-point face, the pose "pinky / index / thumb" points from
  the WholeBody hand knuckles; z = 0 (MMPose is 2-D; our features only use pose x/y); visibility = MMPose score;
  frame present when nose or a shoulder has score >= 0.3;
* face: the app's 20 ``FACE_IDX`` mesh points from their iBUG-68 equivalents (z = 0), present when the mean
  face score >= 0.3;
* pixel -> normalised with the frame size estimated per clip from the two hand estimates of the same hand
  (WholeBody pixels / MediaPipe normalised), snapped to a standard size within 6 %, else the signer's usual
  size, else 2560x1440 (the released videos' size). The fallback is deterministic (:func:`signer_size`: the
  signer's most common estimate over all their clips, applied in a second pass after every estimate is known).

Parity check on ``USER001/Absent__session106__clip014`` (our ``HolisticExtractor`` on the released MP4, same
166 frames, error in shoulder widths): hands 0.012-0.019; nose / shoulders / elbows 0.02-0.05; wrists
0.07-0.09 when the hand is visible (both models guess differently when it is out of frame); the 20 face
points 0.004-0.035. Close but not identical; the version tag keeps these clips separable.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np

from app.features.spec import FACE_IDX, N_FACE, N_HAND, N_POSE
from app.isl.data.pose_sources.common import PoseClip
from app.isl.data.sources import MIN_FRAMES, MIN_HANDS_FRAC, _Base
from app.landmarks.raw import RawSequence

REPO = "ISL500/ISL-DATA"
LICENCE = "research-only (ISL-DATA card: research and academic use only)"
URL = "https://huggingface.co/datasets/ISL500/ISL-DATA"
EXTRACTOR_VERSION = "isl500-mphands+mmpose-wb"
DEFAULT_SIZE = (2560, 1440)
STANDARD_SIZES = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (960, 540), (640, 480), (1440, 1080),
                  (2160, 3840), (1440, 2560), (1080, 1920), (720, 1280), (1080, 1080)]
SCORE_MIN = 0.3

# COCO-WholeBody indices
WB_FACE0 = 23  # iBUG-68 point i is WB index 23 + i
WB_LHAND0, WB_RHAND0 = 91, 112

# MediaPipe pose index -> ("body", WB index) | ("face", iBUG index) | ("lhand"/"rhand", hand index)
POSE_MAP: dict[int, tuple[str, int]] = {
    0: ("body", 0),
    1: ("face", 42), 2: ("body", 1), 3: ("face", 45),  # left eye inner / centre / outer (signer's left)
    4: ("face", 39), 5: ("body", 2), 6: ("face", 36),  # right eye inner / centre / outer
    7: ("body", 3), 8: ("body", 4),  # ears
    9: ("face", 54), 10: ("face", 48),  # mouth left / right (signer's)
    11: ("body", 5), 12: ("body", 6), 13: ("body", 7), 14: ("body", 8), 15: ("body", 9), 16: ("body", 10),
    17: ("lhand", 17), 18: ("rhand", 17), 19: ("lhand", 5), 20: ("rhand", 5), 21: ("lhand", 2), 22: ("rhand", 2),
    23: ("body", 11), 24: ("body", 12), 25: ("body", 13), 26: ("body", 14), 27: ("body", 15), 28: ("body", 16),
    29: ("body", 19), 30: ("body", 22), 31: ("body", 17), 32: ("body", 20),  # heels, big toes
}
# MediaPipe face-mesh index (FACE_IDX entries) -> iBUG-68 index (errors measured in the module docstring)
FACE_MAP: dict[int, int] = {1: 30, 152: 8, 234: 1, 454: 15, 61: 48, 291: 54, 0: 51, 17: 57, 13: 62, 14: 66,
                            70: 17, 105: 19, 107: 21, 336: 22, 334: 24, 300: 26, 33: 36, 133: 39, 362: 42, 263: 45}
assert all(i in FACE_MAP for i in FACE_IDX) and len(POSE_MAP) == N_POSE

_USER_PREFIX = re.compile(r".*_user\d{3}_(.+)$", re.I)


def clip_key(stem: str) -> str:
    """Canonical clip name: drop the export-path prefix some files carry
    (``_mnt_..._R2_Clips_R2_user001_Absent__session7__clip000`` -> ``Absent__session7__clip000``)."""
    m = _USER_PREFIX.match(stem)
    return m.group(1) if m else stem


def gloss_of(key: str) -> str:
    """``Absent__session7__clip000`` / ``R2_Absent__000001`` -> ``Absent``."""
    g = key.split("__", 1)[0]
    g = re.sub(r"^R\d+_", "", g)
    return g.replace("_", " ").strip()


def user_of(folder: str) -> str:
    """``ISL_DATA_USER007`` / ``User007`` -> ``user007``."""
    m = re.search(r"user\s*0*(\d+)", folder, re.I)
    if not m:
        raise ValueError(f"no user id in {folder!r}")
    return f"user{int(m.group(1)):03d}"


def clip_id(user: str, key: str) -> str:
    return f"isl500/{user}/{key}"


def estimate_size(hands: np.ndarray, wb: np.ndarray, min_frames: int = 3) -> tuple[int, int] | None:
    """Frame size (W, H) from the same hands seen by MediaPipe (normalised) and MMPose (pixels)."""
    rx, ry = [], []
    for k, off in ((0, WB_LHAND0), (1, WB_RHAND0)):
        mp_h = hands[:, k]
        present = ~np.all(mp_h == 0, axis=(1, 2))
        wbh = wb[:, off:off + N_HAND]
        sel = present & (wbh[:, :, 2].mean(axis=1) > 0.5)
        if sel.sum() < min_frames:
            continue
        x, y = mp_h[sel, :, 0], mp_h[sel, :, 1]
        ok = (x > 0.05) & (y > 0.05)
        rx.append(wbh[sel, :, 0][ok] / x[ok])
        ry.append(wbh[sel, :, 1][ok] / y[ok])
    if not rx:
        return None
    w, h = float(np.median(np.concatenate(rx))), float(np.median(np.concatenate(ry)))
    if not (np.isfinite(w) and np.isfinite(h)) or w < 100 or h < 100:
        return None
    for sw, sh in STANDARD_SIZES:
        if abs(w / sw - 1) < 0.06 and abs(h / sh - 1) < 0.06:
            return sw, sh
    return None


def signer_size(counts: dict[tuple[int, int], int]) -> tuple[int, int]:
    """Deterministic per-signer fallback frame size: the most common hand-based estimate; ties go to the size
    listed first in ``STANDARD_SIZES`` (then the smaller size); :data:`DEFAULT_SIZE` without estimates."""
    if not counts:
        return DEFAULT_SIZE
    rank = {sz: i for i, sz in enumerate(STANDARD_SIZES)}
    return min(counts, key=lambda sz: (-counts[sz], rank.get(tuple(sz), len(rank)), tuple(sz)))


class SignerSizes:
    """Thread-safe counts of hand-based size estimates per signer; :meth:`fallback` -> (size, ``size_from``)."""

    def __init__(self):
        import threading
        from collections import Counter

        self._c: dict[str, Counter] = {}
        self._lock = threading.Lock()
        self._Counter = Counter

    def add(self, user: str, size: tuple[int, int]) -> None:
        with self._lock:
            self._c.setdefault(user, self._Counter())[(int(size[0]), int(size[1]))] += 1

    def fallback(self, user: str) -> tuple[tuple[int, int], str]:
        with self._lock:
            counts = dict(self._c.get(user) or {})
        return (signer_size(counts), "user") if counts else (DEFAULT_SIZE, "default")


def isl500_to_raw(hands: np.ndarray, wb: np.ndarray, size: tuple[int, int], fps: float = 30.0) -> RawSequence:
    """h5 hands (T, 2, 21, 3) + COCO-WholeBody (T, 133, 3) -> RawSequence (see module docstring)."""
    hands = np.asarray(hands, dtype=np.float64)
    wb = np.asarray(wb, dtype=np.float64)
    T = min(len(hands), len(wb))
    hands, wb = hands[:T], wb[:T]
    W, H = size
    xy = wb[..., :2] / np.array([W, H], dtype=np.float64)
    sc = wb[..., 2]

    def src(kind: str, i: int) -> int:
        return {"body": i, "face": WB_FACE0 + i, "lhand": WB_LHAND0 + i, "rhand": WB_RHAND0 + i}[kind]

    pose = np.full((T, N_POSE, 4), np.nan, np.float32)
    idx = np.array([src(*POSE_MAP[j]) for j in range(N_POSE)])
    present = np.max(sc[:, [0, 5, 6]], axis=1) >= SCORE_MIN
    pose[present, :, :2] = xy[present][:, idx]
    pose[present, :, 2] = 0.0
    pose[present, :, 3] = sc[present][:, idx]

    face = np.full((T, N_FACE, 3), np.nan, np.float32)
    fidx = np.array([WB_FACE0 + FACE_MAP[i] for i in FACE_IDX])
    fpresent = sc[:, WB_FACE0:WB_FACE0 + 68].mean(axis=1) >= SCORE_MIN
    face[fpresent, :, :2] = xy[fpresent][:, fidx]
    face[fpresent, :, 2] = 0.0

    out_h = []
    for k in (0, 1):
        h = np.full((T, N_HAND, 3), np.nan, np.float32)
        ok = ~np.all(hands[:, k] == 0, axis=(1, 2)) & np.all(np.isfinite(hands[:, k]), axis=(1, 2))
        h[ok] = hands[ok, k]
        out_h.append(h)
    t_ms = np.arange(T, dtype=np.float64) * 1000.0 / (fps or 30.0)
    return RawSequence(pose, out_h[0], out_h[1], face, t_ms, int(W), int(H), EXTRACTOR_VERSION)


# --------------------------------------------------------------------------------------------- vocabulary

def link_gloss(word: str, linker) -> tuple[str | None, str, bool, list[str]]:
    from app.isl.vocab.normalize import parse_cislr_word

    return linker.link(parse_cislr_word(word))


def make_linker(vocab):
    """A vocabulary ``_Linker`` over a ``VocabIndex`` (same rules as INCLUDE/CISLR linking in vocab/build.py)."""
    from app.isl.vocab.build import _Linker

    return _Linker({e.concept_id: e for e in vocab.entries})


class Isl500Adapter(_Base):
    """ISL500 word clips with true signer ids (``group = isl500-userNNN``), so a signer-independent split is
    possible. Each clip is sliced to its hand-active window (``active_start/end`` in the manifest: hands are
    visible in only ~30 % of the raw frames) and the quality filter uses the hand fraction inside it. ``concept_id``: the vocabulary's own ``isl500:<gloss>`` variant when a later vocabulary build adds
    one, else the vocabulary linker (exact / qualified / synonym / spelling merges, as for INCLUDE)."""

    name = "isl500"
    licence = LICENCE
    commercial = False
    manifest_name = "isl500"
    profiles = ("research",)
    pretraining_only = False

    def __init__(self, paths=None, vocab=None):
        super().__init__(paths, vocab)
        self._links: dict[str, tuple[str | None, str, bool]] = {}
        self._linker = None

    def link(self, gloss: str) -> tuple[str | None, str, bool]:
        if gloss in self._links:
            return self._links[gloss]
        res: tuple[str | None, str, bool] = (None, "none", False)
        if self.vocab is not None:
            c = self.vocab.concept_of_variant(f"isl500:{gloss}") if hasattr(self.vocab, "concept_of_variant") else None
            if c:
                res = (c, "vocabulary", False)
            elif hasattr(self.vocab, "entries"):
                if self._linker is None:
                    self._linker = make_linker(self.vocab)
                concept, kind, review, _ = link_gloss(gloss, self._linker)
                res = (concept, kind, review)
        self._links[gloss] = res
        return res

    def clips(self, *, only_ok: bool = True) -> Iterator[PoseClip]:
        for rec in self.records(only_ok=only_ok):
            start = int(rec.get("active_start") or 0)
            end = rec.get("active_end")
            n = (int(end) if end is not None else int(rec.get("n_frames") or 0)) - start
            hf = rec.get("hands_frac_active", rec.get("hands_frac"))
            if only_ok and (n < MIN_FRAMES or float(hf or 0.0) < MIN_HANDS_FRAC):
                continue
            gloss = rec.get("gloss") or ""
            concept, kind, review = self.link(gloss)
            yield PoseClip(
                clip_id=rec["id"], source="isl500", raw_gloss=gloss, concept_id=concept,
                variant_id=f"isl500:{gloss}", category=None, path=self.paths.resolve(rec["path"]),
                start=start, end=int(end) if end is not None else None,
                group=f"isl500-{rec.get('user')}", fps=rec.get("fps"), licence=self.licence, profiles=self.profiles,
                quality={"n_frames": rec.get("n_frames"), "hands_frac": rec.get("hands_frac"),
                         "hands_frac_active": rec.get("hands_frac_active"), "link": kind,
                         "needs_review": review, "pd_split": rec.get("pd_split"), "size_from": rec.get("size_from"),
                         "extractor_version": rec.get("extractor_version")},
            )
