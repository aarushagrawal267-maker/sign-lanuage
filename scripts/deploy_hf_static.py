"""Deploy the browser-only build to a Hugging Face *static* Space (free tier).

  python scripts/export_static.py                 # model.onnx + data snapshots
  (cd frontend && npm run build:static)           # -> frontend/dist-static
  python scripts/deploy_hf_static.py <user>/<space>
Auth: HF_TOKEN env var, cached login, or an authenticating proxy.
"""

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    repo_id = sys.argv[1]
    dist = ROOT / "frontend" / "dist-static"
    if not (dist / "index.html").exists() or not (dist / "static-data" / "model.onnx").exists():
        sys.exit("build missing: run scripts/export_static.py and `npm run build:static` first")
    shutil.copy(ROOT / "deploy" / "huggingface-static" / "README.md", dist / "README.md")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(dist), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Sign Intelligence Studio (browser-only build)")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
