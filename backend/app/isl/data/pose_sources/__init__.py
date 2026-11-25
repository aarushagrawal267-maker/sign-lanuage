"""Pose-native ISL sources: converters and read-only adapters (docs/DATASETS_ISL.md, "Pose-native sources").

(The package is ``pose_sources`` rather than ``sources/`` because ``app/isl/data/sources.py`` already exists and a
package of the same name would shadow it.)

=================  ===================================  ======================  ================================
adapter            data                                 licence                 profile / use
=================  ===================================  ======================  ================================
``glosl``          iSign/ISLTranslate sentence poses    CC BY-NC 4.0            research, pre-training only
``isl500``         500 words x 15 signers (hands+body)  research-only           research (labelled, signer ids)
``openhands_raw``  ~1,129 h unlabelled ISL poses        CC BY 4.0 label (*)     research, pre-training only
``openhands_fs``   36 letters/digits, 172 clips         CC BY 4.0 label (*)     research (fingerspelling tests)
=================  ===================================  ======================  ================================

(*) kept out of ``commercial-safe`` until the owner decides (third-party YouTube origin / undocumented origin).

The adapters follow the :class:`~app.isl.data.sources.SourceAdapter` protocol and yield
:class:`~app.isl.data.pose_sources.common.PoseClip` (a ``Clip`` whose ``load`` upcasts float16 storage).
Like ``video_sources``, they are listed here and not yet registered in ``sources.ADAPTERS`` / ``profiles.PROFILES``;
:func:`register` does that for a caller that wants them (``register(ADAPTERS)``), and :data:`PROFILE_SOURCES`
says which profile each belongs to.
"""

from __future__ import annotations

from app.isl.data.pose_sources.fingerspell import OpenHandsFingerspellAdapter
from app.isl.data.pose_sources.glosl import GloslAdapter
from app.isl.data.pose_sources.isl500 import Isl500Adapter
from app.isl.data.pose_sources.openhands import OpenHandsRawAdapter

POSE_ADAPTERS = {
    "glosl": GloslAdapter,
    "isl500": Isl500Adapter,
    "openhands_raw": OpenHandsRawAdapter,
    "openhands_fs": OpenHandsFingerspellAdapter,
}

# profile -> {"sources": labelled sources for training/index/calibration, "pretraining": weight init only}
PROFILE_SOURCES = {
    "commercial-safe": {"sources": (), "pretraining": ()},
    "research": {"sources": ("isl500", "openhands_fs"), "pretraining": ("glosl", "openhands_raw")},
}


def register(adapters: dict) -> dict:
    """Add the pose adapters to an adapter registry (e.g. ``app.isl.data.sources.ADAPTERS``) without
    overwriting existing names. Returns the registry."""
    for k, v in POSE_ADAPTERS.items():
        adapters.setdefault(k, v)
    return adapters


__all__ = ["POSE_ADAPTERS", "PROFILE_SOURCES", "register", "GloslAdapter", "Isl500Adapter", "OpenHandsRawAdapter",
           "OpenHandsFingerspellAdapter"]
