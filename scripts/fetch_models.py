"""Download the MediaPipe Holistic model used by both the backend and the browser.

Usage:  python scripts/fetch_models.py [--force]
Stored at assets/mediapipe/holistic_landmarker.task (served to the browser by the backend).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
DEST = ROOT / "assets" / "mediapipe" / "holistic_landmarker.task"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if DEST.exists() and DEST.stat().st_size > 1_000_000 and not args.force:
        print(f"model present: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"downloading {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as e:  # noqa: BLE001
        print(f"download failed: {e}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return 1
    if tmp.stat().st_size < 1_000_000:
        print("downloaded file is unexpectedly small; aborting", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return 1
    tmp.replace(DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
