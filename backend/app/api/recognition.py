"""Live recognition (WebSocket), offline video analysis, vocabulary and events.

Privacy: the browser runs MediaPipe locally and sends only landmark
coordinates over a localhost WebSocket. No webcam image ever reaches the
backend. Uploaded videos for offline analysis are processed from a temporary
file that is deleted immediately afterwards.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import Experiment, Gloss, LandmarkFile, RecognitionEvent, Video
from app.db.session import get_db, session_scope
from app.features.build import FeatureError, build_features, sequence_quality
from app.inference import registry
from app.inference.engine import EngineConfig, RecognitionEngine
from app.landmarks.raw import RawSequence

router = APIRouter(tags=["recognition"])

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _record(gloss: str, class_index: int, score: float, source: str, exp_id: int | None) -> None:
    with session_scope() as db:
        db.add(RecognitionEvent(gloss=gloss, class_index=class_index, score=score, source=source, experiment_id=exp_id))


@router.get("/api/vocabulary", summary="WLASL100 vocabulary with data and per-class performance")
def vocabulary(db: Session = Depends(get_db)):
    glosses = list(db.scalars(select(Gloss).order_by(Gloss.class_index)))
    vids = list(db.scalars(select(Video)))
    lm_ok = {lf.video_id for lf in db.scalars(select(LandmarkFile).where(LandmarkFile.status == "ok"))}
    active = db.scalar(select(Experiment).where(Experiment.is_active.is_(True)))
    per_val = {r["class_index"]: r for r in ((active.val_metrics or {}).get("per_class") or [])} if active else {}
    per_test = {r["class_index"]: r for r in ((active.test_metrics or {}).get("per_class") or [])} if active else {}
    by_cls: dict[int, list[Video]] = {}
    for v in vids:
        by_cls.setdefault(v.class_index, []).append(v)
    out = []
    for g in glosses:
        vs = by_cls.get(g.class_index, [])
        out.append({
            "class_index": g.class_index, "gloss": g.gloss,
            "annotated": len(vs), "valid": sum(v.status == "valid" for v in vs),
            "with_landmarks": sum(v.video_id in lm_ok for v in vs),
            "train_official": sum(v.split == "train" for v in vs),
            "signers": len({v.signer_id for v in vs}),
            "variations": len({v.variation_id for v in vs}),
            "local_example": next((v.video_id for v in vs if v.status == "valid" and v.file_path and Path(v.file_path).exists()), None),
            "val": per_val.get(g.class_index), "test": per_test.get(g.class_index),
        })
    if not out and active is not None:
        # Annotations not loaded yet (fresh install with the pretrained model): list the glosses
        # the active model recognises so the vocabulary is still browsable.
        try:
            p, _ = registry.get_predictor()
        except Exception:  # noqa: BLE001 - an unreadable checkpoint just means no fallback list
            p = None
        for i, name in enumerate(p.class_names if p is not None else []):
            out.append({"class_index": i, "gloss": name, "annotated": 0, "valid": 0, "with_landmarks": 0,
                        "train_official": 0, "signers": 0, "variations": 0, "local_example": None,
                        "val": per_val.get(i), "test": per_test.get(i)})
    return {"model": None if active is None else {"id": active.id, "name": active.name, "split": active.split_name,
                                                   "is_synthetic": active.is_synthetic},
            "glosses": out}


@router.get("/api/recognition/events", summary="Recently recognised signs (gloss + score only)")
def events(limit: int = 50, db: Session = Depends(get_db)):
    rows = list(db.scalars(select(RecognitionEvent).order_by(RecognitionEvent.id.desc()).limit(min(limit, 500))))
    return [{"id": e.id, "gloss": e.gloss, "class_index": e.class_index, "score": e.score, "source": e.source,
             "experiment_id": e.experiment_id, "created_at": e.created_at} for e in rows]


@router.delete("/api/recognition/events", summary="Clear recognition history")
def clear_events(db: Session = Depends(get_db)):
    n = db.execute(delete(RecognitionEvent)).rowcount
    return {"deleted": n}


@router.websocket("/api/ws/recognize")
async def ws_recognize(ws: WebSocket):
    """Protocol (JSON):
    client -> {"type":"config","width":W,"height":H,"engine":{...}}
              {"type":"frame","t":ms,"pose":[[x,y,z,v]*33]|null,"left":[[x,y,z]*21]|null,"right":...,"face":[[x,y,z]*K]|null}
              {"type":"reset"} | {"type":"ping","t":ms}
    server -> {"type":"hello",...} | {"type":"state",...} | {"type":"pong"} | {"type":"error","message":...}
    """
    await ws.accept()
    predictor, info, problem = await asyncio.to_thread(registry.load_active)
    engine = RecognitionEngine(predictor)
    await ws.send_json({"type": "hello", "model": info, "engine": engine.describe(), "message": problem})
    exp_id = info["experiment_id"] if info else None
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except (ValueError, KeyError):  # not JSON (JSONDecodeError is a ValueError) or a binary frame
                await ws.send_json({"type": "error", "message": "messages must be JSON objects"})
                continue
            if not isinstance(msg, dict):
                await ws.send_json({"type": "error", "message": "messages must be JSON objects"})
                continue
            kind = msg.get("type")
            if kind == "frame":
                t0 = time.perf_counter()
                try:
                    state = await asyncio.to_thread(engine.push, msg)
                except (ValueError, TypeError) as e:
                    await ws.send_json({"type": "error", "message": f"invalid frame: {e}"})
                    continue
                state["server_ms"] = (time.perf_counter() - t0) * 1000
                commit = state.get("commit")
                if commit:
                    await asyncio.to_thread(_record, commit["gloss"], commit["class_index"], commit["score"], "live", exp_id)
                await ws.send_json(state)
            elif kind == "config":
                try:
                    if "width" in msg and "height" in msg:
                        engine.set_size(int(msg["width"]), int(msg["height"]))
                    if "engine" in msg:
                        engine.cfg = EngineConfig.from_dict({**engine.describe()["config"], **(msg["engine"] or {})})
                        engine.reset()
                    await ws.send_json({"type": "config_ack", "engine": engine.describe()})
                except (ValueError, TypeError) as e:
                    await ws.send_json({"type": "error", "message": f"invalid config: {e}"})
            elif kind == "reset":
                engine.reset()
                await ws.send_json({"type": "reset_ack"})
            elif kind == "ping":
                await ws.send_json({"type": "pong", "t": msg.get("t")})
            else:
                await ws.send_json({"type": "error", "message": f"unknown message type {kind!r}"})
    except WebSocketDisconnect:
        return


def _classify_whole(predictor, seq: RawSequence) -> dict:
    try:
        feats = build_features(seq, predictor.feature_config)
    except FeatureError as e:
        return {"error": str(e)}
    probs, ms = predictor.predict(feats)
    idx = np.argsort(-probs)[:5]
    return {"top": [{"class_index": int(i), "gloss": predictor.class_names[i], "prob": float(probs[i])} for i in idx],
            "inference_ms": ms}


def _run_engine(predictor, seq: RawSequence, engine_cfg: dict | None) -> dict:
    eng = RecognitionEngine(predictor, seq.width, seq.height, EngineConfig.from_dict(engine_cfg))
    segments, commits = [], []
    for i in range(len(seq)):
        st = eng.push(seq.frame_to_json(i))
        if "segment_result" in st:
            segments.append({**st["segment_result"], "end_t": st["t"]})
        if "commit" in st:
            commits.append(st["commit"])
    st = eng.flush()
    if "segment_result" in st:
        segments.append({**st["segment_result"], "end_t": float(seq.t_ms[-1]) if len(seq) else 0.0})
    if "commit" in st:
        commits.append(st["commit"])
    return {"segments": segments, "commits": commits, "threshold": eng.threshold, "mode": eng.cfg.mode}


@router.post("/api/analyze/video", summary="Offline analysis of a local video file (not live recognition)")
async def analyze_video(file: UploadFile = File(...), mode: str = Form("segment"), threshold: float | None = Form(None),
                        whole_clip: bool = Form(True)):
    predictor, info, problem = await asyncio.to_thread(registry.load_active)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file larger than 200 MB")
                f.write(chunk)
        from app.landmarks.extractor import ExtractorUnavailable, HolisticExtractor

        try:
            ex = HolisticExtractor()
        except ExtractorUnavailable as e:
            raise HTTPException(400, str(e)) from e
        t0 = time.perf_counter()
        try:
            seq, fps = await asyncio.to_thread(ex.extract_video, Path(tmp))
        except ExtractorUnavailable as e:
            raise HTTPException(503, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, f"could not read video: {e}") from e
        extract_s = time.perf_counter() - t0
    finally:
        try:
            os.unlink(tmp)  # never keep uploaded footage
        except OSError:
            pass
    if len(seq) == 0:
        raise HTTPException(400, "no frames could be decoded")
    step = max(1, int(np.ceil(len(seq) / 600)))
    result = {
        "filename": file.filename, "frames": len(seq), "fps": fps, "width": seq.width, "height": seq.height,
        "duration_s": len(seq) / fps, "extract_seconds": extract_s, "extract_ms_per_frame": extract_s * 1000 / len(seq),
        "quality": sequence_quality(seq), "landmarks": {"frame_step": step, "frames": [seq.frame_to_json(i) for i in range(0, len(seq), step)]},
        "model": info, "whole_clip": None, "stream": None,
    }
    if predictor is None:
        result["message"] = f"Landmarks extracted, but no model is available for classification. {problem}"
        return result
    if whole_clip:
        result["whole_clip"] = await asyncio.to_thread(_classify_whole, predictor, seq)
    cfg = {"mode": mode}
    if threshold is not None:
        cfg["threshold"] = threshold
    try:
        result["stream"] = await asyncio.to_thread(_run_engine, predictor, seq, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    for c in result["stream"]["commits"]:
        _record(c["gloss"], c["class_index"], c["score"], "video", info["experiment_id"])
    return result


class LandmarkClip(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: list[dict]


@router.post("/api/analyze/landmarks", summary="Classify a landmark clip recorded in the browser")
def analyze_landmarks(clip: LandmarkClip):
    predictor, info, problem = registry.load_active()
    if predictor is None:
        raise HTTPException(409, problem)
    try:
        seq = RawSequence.from_frames(clip.frames, clip.width, clip.height)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    res = _classify_whole(predictor, seq)
    res["quality"] = sequence_quality(seq)
    res["model"] = info
    return res


@router.get("/api/recognition/summary", summary="Counts of recognised glosses")
def recognition_summary(db: Session = Depends(get_db)):
    rows = list(db.scalars(select(RecognitionEvent)))
    return {"total": len(rows), "by_gloss": Counter(r.gloss for r in rows).most_common(20),
            "by_source": dict(Counter(r.source for r in rows))}
