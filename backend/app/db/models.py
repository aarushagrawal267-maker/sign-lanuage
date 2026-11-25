"""SQLAlchemy ORM models.

The database stores metadata, provenance and results only. Videos and landmark
arrays live on disk under the data directory; webcam footage is never stored.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Gloss(Base):
    __tablename__ = "glosses"

    class_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    gloss: Mapped[str] = mapped_column(String(64), unique=True, index=True)


class Video(Base):
    """One WLASL100 instance, with the official annotation preserved verbatim."""

    __tablename__ = "videos"

    video_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    class_index: Mapped[int] = mapped_column(ForeignKey("glosses.class_index"), index=True)
    gloss: Mapped[str] = mapped_column(String(64), index=True)
    signer_id: Mapped[int] = mapped_column(Integer, index=True)
    split: Mapped[str] = mapped_column(String(8), index=True)  # official: train/val/test
    source: Mapped[str] = mapped_column(String(64))
    url: Mapped[str] = mapped_column(Text)
    annotated_fps: Mapped[int] = mapped_column(Integer)
    frame_start: Mapped[int] = mapped_column(Integer)
    frame_end: Mapped[int] = mapped_column(Integer)  # -1 = until end of video
    bbox: Mapped[list | None] = mapped_column(JSON, nullable=True)
    variation_id: Mapped[int] = mapped_column(Integer, default=0)
    instance_id: Mapped[int] = mapped_column(Integer, default=0)

    # Acquisition / validation state
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    # pending | downloaded | failed | valid | invalid
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    acquired_via: Mapped[str | None] = mapped_column(String(32), nullable=True)
    measured_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    clip_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class DownloadAttempt(Base):
    __tablename__ = "download_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), index=True)
    method: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class LandmarkFile(Base):
    __tablename__ = "landmark_files"

    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok | failed | low_quality
    n_frames: Mapped[int] = mapped_column(Integer, default=0)
    pose_rate: Mapped[float] = mapped_column(Float, default=0.0)
    left_rate: Mapped[float] = mapped_column(Float, default=0.0)
    right_rate: Mapped[float] = mapped_column(Float, default=0.0)
    any_hand_rate: Mapped[float] = mapped_column(Float, default=0.0)
    face_rate: Mapped[float] = mapped_column(Float, default=0.0)
    extractor_version: Mapped[str] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Split(Base):
    __tablename__ = "splits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(32))  # official | signer_grouped
    signer_independent: Mapped[bool] = mapped_column(Boolean, default=False)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    leakage_report: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SplitAssignment(Base):
    __tablename__ = "split_assignments"

    split_id: Mapped[int] = mapped_column(ForeignKey("splits.id", ondelete="CASCADE"), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.video_id"), primary_key=True)
    partition: Mapped[str] = mapped_column(String(8))  # train | val | test


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    # queued | running | completed | failed | cancelled
    architecture: Mapped[str] = mapped_column(String(16))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_config: Mapped[dict] = mapped_column(JSON, default=dict)
    split_id: Mapped[int | None] = mapped_column(ForeignKey("splits.id"), nullable=True)
    split_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    data_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    history: Mapped[list] = mapped_column(JSON, default=list)
    val_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    test_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    device: Mapped[str | None] = mapped_column(String(16), nullable=True)
    n_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RecognitionEvent(Base):
    """A committed recognition. Only the gloss and score are stored - no video."""

    __tablename__ = "recognition_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    source: Mapped[str] = mapped_column(String(16))  # live | video
    gloss: Mapped[str] = mapped_column(String(64))
    class_index: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    experiment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Language pack id and pack-specific details (concept_id, text, source, decision, chosen_by_user).
    # Added by db/migrate.py to databases created before language packs existed.
    language: Mapped[str] = mapped_column(String(32), default="asl-wlasl100", server_default="asl-wlasl100")
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class MetaKV(Base):
    """Dataset provenance and other key/value metadata."""

    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
