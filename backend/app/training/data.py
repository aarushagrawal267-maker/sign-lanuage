"""Training data: landmark files + split assignment -> feature tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch.utils.data import Dataset

from app.db.models import Gloss, LandmarkFile, Video
from app.features.build import AugmentConfig, BodyFrame, FeatureError, augment, featurize, to_body_frame
from app.features.spec import FeatureConfig
from app.landmarks.raw import RawSequence
from app.splits.splits import get_assignment


@dataclass
class Sample:
    video_id: str
    label: int
    signer_id: int
    partition: str
    bf: BodyFrame
    n_frames: int


@dataclass
class LoadedData:
    samples: dict[str, list[Sample]]
    class_names: list[str]
    skipped: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        out = {"skipped": self.skipped}
        for p, ss in self.samples.items():
            out[p] = {
                "videos": len(ss),
                "signers": len({s.signer_id for s in ss}),
                "classes": len({s.label for s in ss}),
                "median_frames": float(np.median([s.n_frames for s in ss])) if ss else None,
            }
        return out


def load_samples(db: Session, split_id: int, cfg: FeatureConfig, min_hand_rate: float = 0.1) -> LoadedData:
    assign = get_assignment(db, split_id)
    class_names = [g.gloss for g in db.scalars(select(Gloss).order_by(Gloss.class_index))]
    rows = db.execute(
        select(Video.video_id, Video.class_index, Video.signer_id, LandmarkFile.path, LandmarkFile.any_hand_rate)
        .join(LandmarkFile, LandmarkFile.video_id == Video.video_id)
        .where(LandmarkFile.status == "ok")
    ).all()
    samples: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    skipped = {"no_landmarks": len(assign) - len(rows), "low_hand_detection": 0, "no_body": 0, "unassigned": 0}
    for vid, ci, sid, path, hand_rate in rows:
        part = assign.get(vid)
        if part is None:
            skipped["unassigned"] += 1
            continue
        if hand_rate < min_hand_rate:
            skipped["low_hand_detection"] += 1
            continue
        try:
            raw = RawSequence.load(Path(path))
            bf = to_body_frame(raw, cfg.max_gap)
        except (FeatureError, OSError, KeyError, ValueError):
            skipped["no_body"] += 1
            continue
        samples[part].append(Sample(vid, ci, sid, part, bf, len(raw)))
    return LoadedData(samples, class_names, skipped)


class SequenceDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg: FeatureConfig, train: bool, seed: int = 0,
                 aug: AugmentConfig | None = None):
        self.samples = samples
        self.cfg = cfg
        self.train = train
        self.aug = aug or AugmentConfig()
        self.rng = np.random.default_rng(seed)
        # Evaluation features are deterministic: compute once.
        self._cache = None if train else [featurize(s.bf, cfg) for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        if self._cache is not None:
            x = self._cache[i]
        else:
            x = featurize(augment(s.bf, self.rng, self.aug), self.cfg)
        return torch.from_numpy(x), s.label


def class_weights(samples: list[Sample], n_classes: int) -> torch.Tensor:
    counts = np.bincount([s.label for s in samples], minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)
