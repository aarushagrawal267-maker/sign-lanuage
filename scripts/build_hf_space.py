"""Stage (and optionally upload) the Hugging Face Docker Space.

  python scripts/build_hf_space.py --out build/hf-space
  python scripts/build_hf_space.py --out build/hf-space --push <user>/<space-name>   # needs HF_TOKEN

What is included: backend code, built frontend (frontend/dist), MediaPipe model,
checkpoints of completed experiments, and a copy of the results database with
transient tables cleared. What is NOT included: WLASL videos, extracted
landmark files, download logs (C-UDA: no redistribution of the data).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stage(out: Path, data_dir: Path, artifacts_dir: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(ROOT / "backend" / "app", out / "backend" / "app", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "backend" / "requirements.txt", out / "backend" / "requirements.txt")
    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").exists():
        sys.exit("frontend/dist missing - run `npm run build` in frontend/ first")
    shutil.copytree(dist, out / "frontend" / "dist")
    model = ROOT / "assets" / "mediapipe" / "holistic_landmarker.task"
    (out / "assets" / "mediapipe").mkdir(parents=True)
    shutil.copy(model, out / "assets" / "mediapipe" / model.name)
    for f in ("Dockerfile", "README.md"):
        shutil.copy(ROOT / "deploy" / "huggingface" / f, out / f)

    # Results database: keep metadata + experiments, drop transient/raw-data tables.
    (out / "data").mkdir()
    db_out = out / "data" / "app.db"
    src = sqlite3.connect(data_dir / "app.db")
    dst = sqlite3.connect(db_out)
    src.backup(dst)
    src.close()
    cur = dst.cursor()
    for table in ("jobs", "recognition_events", "download_attempts"):
        cur.execute(f"DELETE FROM {table}")
    cur.execute("UPDATE videos SET file_path = NULL")
    cur.execute("UPDATE landmark_files SET path = ''")
    keep = [r[0] for r in cur.execute("SELECT checkpoint_path FROM experiments WHERE status='completed' AND checkpoint_path IS NOT NULL")]
    cur.execute("DELETE FROM experiments WHERE status != 'completed'")
    dst.commit()
    cur.execute("VACUUM")
    dst.close()

    (out / "artifacts" / "checkpoints").mkdir(parents=True)
    for p in keep:
        f = Path(p)
        if not f.exists():
            f = artifacts_dir / "checkpoints" / f.name
        if f.exists():
            shutil.copy(f, out / "artifacts" / "checkpoints" / f.name)
    print(f"staged {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB, {len(keep)} checkpoints)")


def push(out: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    # HF_TOKEN if set; otherwise rely on an authenticating proxy / cached login.
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Sign Intelligence Studio")
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "hf-space"))
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--push", metavar="USER/SPACE")
    a = ap.parse_args()
    out = Path(a.out)
    stage(out, Path(a.data_dir), Path(a.artifacts_dir))
    if a.push:
        print("deployed:", push(out, a.push))


if __name__ == "__main__":
    main()
