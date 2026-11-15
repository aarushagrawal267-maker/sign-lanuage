"""Export everything the fully client-side (static) build needs.

  python scripts/export_static.py            # writes frontend/public/static-data/

* model.onnx      - the active model (softmax probabilities), verified against PyTorch
* api/*.json      - snapshots of the read-only API responses (results, vocabulary, dataset metadata)
* parity.json     - (into frontend/src/static/__fixtures__) raw landmark sequences from real clips with
                    the Python features and probabilities, used by Vitest to prove the TypeScript port matches.

Never exported: WLASL videos or landmark files (C-UDA). The parity fixture holds
features of 3 clips only, for testing.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
OUT = ROOT / "frontend" / "public" / "static-data"
FIXTURE = ROOT / "frontend" / "src" / "static" / "__fixtures__" / "parity.json"


def key_for(path: str) -> str:
    p = path.split("?")[0].strip("/")
    p = p[4:] if p.startswith("api/") else p
    return p.replace("/", "__") or "root"


def main() -> None:
    import numpy as np
    import onnxruntime as ort
    import torch
    from fastapi.testclient import TestClient
    from sqlalchemy import select

    from app.db.models import Experiment, LandmarkFile
    from app.db.session import session_scope
    from app.features.build import build_features
    from app.inference.registry import get_predictor
    from app.landmarks.raw import RawSequence
    from app.main import app
    from app.models.checkpoint import frame_mask

    predictor, info = get_predictor()
    if predictor is None:
        sys.exit("no active model - activate one first")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "api").mkdir(parents=True)

    # ---- ONNX model
    fc = predictor.feature_config

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, mask):
            return torch.softmax(self.m(x, mask), -1)

    w = Wrap(predictor.model).eval()
    x1 = torch.randn(1, fc.seq_len, fc.dim)
    torch.onnx.export(w, (x1, frame_mask(x1, fc)), str(OUT / "model.onnx"), input_names=["x", "mask"],
                      output_names=["probs"], dynamic_axes={"x": {0: "b"}, "mask": {0: "b"}, "probs": {0: "b"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(OUT / "model.onnx"))
    for b in (1, 3):
        x = torch.randn(b, fc.seq_len, fc.dim)
        m = frame_mask(x, fc)
        diff = np.abs(sess.run(None, {"x": x.numpy(), "mask": m.numpy()})[0] - w(x, m).detach().numpy()).max()
        assert diff < 1e-4, f"ONNX mismatch at batch {b}: {diff}"
        print(f"onnx parity batch {b}: max diff {diff:.2e}")

    # ---- API snapshots
    client = TestClient(app)
    with session_scope() as db:
        exp_ids = [e.id for e in db.scalars(select(Experiment).where(Experiment.status == "completed"))]
    paths = ["/api/health", "/api/system", "/api/overview", "/api/models/active", "/api/features/spec",
             "/api/vocabulary", "/api/dataset/status", "/api/dataset/summary", "/api/dataset/inventory", "/api/splits",
             "/api/experiments", "/api/recognition/events", "/api/jobs"] + [f"/api/experiments/{i}" for i in exp_ids]
    for p in paths:
        r = client.get(p)
        r.raise_for_status()
        data = r.json()
        if p == "/api/system":
            data.update(public_demo=True, static=True, data_dir="(static build)")
        if p == "/api/jobs":
            data = []
        if p == "/api/vocabulary":
            for g in data["glosses"]:
                g["local_example"] = None  # WLASL videos are never redistributed
        (OUT / "api" / f"{key_for(p)}.json").write_text(json.dumps(data, default=str))
    vids = client.get("/api/dataset/videos?limit=500&offset=0").json()
    allv = vids["items"]
    off = 500
    while off < vids["total"]:
        allv += client.get(f"/api/dataset/videos?limit=500&offset={off}").json()["items"]
        off += 500
    for v in allv:
        v["has_file"] = False
    (OUT / "api" / "dataset__videos.json").write_text(json.dumps(allv))
    meta = {"model": info, "threshold": predictor.recommended_threshold, "feature_config": fc.to_dict(),
            "class_names": predictor.class_names, "feature_dim": fc.dim}
    (OUT / "model.json").write_text(json.dumps(meta, default=str))
    print(f"api snapshots: {len(paths) + 1} files, {len(allv)} videos")

    shutil.copy(ROOT / "assets" / "mediapipe" / "holistic_landmarker.task", OUT / "holistic_landmarker.task")

    # ---- parity fixture (3 real clips)
    with session_scope() as db:
        lfs = [lf.path for lf in db.scalars(select(LandmarkFile).where(LandmarkFile.status == "ok").limit(3))]
    cases = []
    for path in lfs:
        raw = RawSequence.load(Path(path))
        raw = raw.slice(0, min(len(raw), 60))
        frames = [raw.frame_to_json(i) for i in range(len(raw))]
        raw = RawSequence.from_frames(frames, raw.width, raw.height)  # same rounding the TS test sees
        feats = build_features(raw, fc)
        probs, _ = predictor.predict(feats)
        cases.append({"width": raw.width, "height": raw.height,
                      "frames": frames,
                      "features": np.round(feats, 6).tolist(), "probs": probs.tolist()})
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"feature_config": fc.to_dict(), "cases": cases}))
    print(f"parity fixture: {len(cases)} clips -> {FIXTURE.relative_to(ROOT)}")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"static data: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    os.environ.setdefault("GLOG_minloglevel", "2")
    main()
