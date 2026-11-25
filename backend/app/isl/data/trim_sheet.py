"""Contact sheets of ISLRTC dictionary videos for by-eye checks of the first-sign trim (docs/DATASETS_ISL.md 7).

A sheet shows the signer (centre crop of the frame) every ``step_ms`` with the video time under each tile. Two modes:

* **blind** (no ``marks``): used to label the true end of the first sign without seeing the trimmer's cut;
* **marked**: tiles inside the kept range get a green bar, the others grey, and named marks (cut, alternatives,
  label) are drawn as coloured ticks on the tile nearest to them.

The video file is untrusted input: it is only decoded with OpenCV, never executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MARK_COLORS = {"cut": (220, 40, 40), "alt": (112, 72, 232), "label": (20, 140, 60), "start": (40, 110, 220)}


@dataclass(frozen=True)
class SheetConfig:
    step_ms: float = 200.0
    max_ms: float = 14200.0
    cols: int = 10
    tile_w: int = 128
    crop_x: tuple[float, float] = (0.28, 0.82)  # signer stands in the centre of the 16:9 frame
    crop_y: tuple[float, float] = (0.12, 1.0)


def read_frames(video: Path | str, times_ms: list[float]) -> list[np.ndarray | None]:
    """BGR frames nearest to each requested time (sequential decode: robust for HEVC, no seeking)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 240:
        fps = 25.0
    want = sorted(range(len(times_ms)), key=lambda k: times_ms[k])
    out: list[np.ndarray | None] = [None] * len(times_ms)
    i, w = 0, 0
    try:
        while w < len(want):
            ok, f = cap.read()
            if not ok:
                break
            t = i * 1000.0 / fps
            # take this frame for every wanted time it is nearest to
            while w < len(want) and times_ms[want[w]] < t + 500.0 / fps:
                out[want[w]] = f
                w += 1
            i += 1
    finally:
        cap.release()
    return out


def _tile(frame: np.ndarray | None, cfg: SheetConfig) -> np.ndarray:
    import cv2

    tile_h = int(round(cfg.tile_w * 1.05))
    if frame is None:
        return np.full((tile_h, cfg.tile_w, 3), 255, np.uint8)
    h, w = frame.shape[:2]
    x0, x1 = int(cfg.crop_x[0] * w), int(cfg.crop_x[1] * w)
    y0, y1 = int(cfg.crop_y[0] * h), int(cfg.crop_y[1] * h)
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (cfg.tile_w, tile_h), interpolation=cv2.INTER_AREA)


def contact_sheet(video: Path | str, out_png: Path | str, title: str = "", cfg: SheetConfig = SheetConfig(),
                  duration_ms: float | None = None, kept: tuple[float, float] | None = None,
                  marks: dict[str, list[float]] | None = None) -> dict:
    """Render the sheet. ``kept`` = (start_ms, end_ms) of the trim; ``marks`` = {kind: [ms, ...]} with kind in
    MARK_COLORS. Returns {"tiles": n, "times_ms": [...]}"""
    import cv2

    end_ms = min(cfg.max_ms, duration_ms) if duration_ms else cfg.max_ms
    times = [float(x) for x in np.arange(0.0, end_ms + 1e-6, cfg.step_ms)]
    frames = read_frames(video, times)
    # drop trailing tiles past the end of the video
    while times and frames[-1] is None:
        times.pop()
        frames.pop()
    if not times:
        raise ValueError(f"no frames decoded from {video}")
    label_h, bar_h = 14, 5
    tiles = []
    for t, f in zip(times, frames):
        tl = _tile(f, cfg)
        cell = np.full((tl.shape[0] + label_h + bar_h, tl.shape[1], 3), 255, np.uint8)
        cell[: tl.shape[0]] = tl
        y_bar = tl.shape[0]
        if kept is not None:
            inside = kept[0] - cfg.step_ms / 2 <= t <= kept[1] + cfg.step_ms / 2
            cell[y_bar: y_bar + bar_h] = (60, 180, 60) if inside else (200, 200, 200)
        for kind, vals in (marks or {}).items():
            col = MARK_COLORS.get(kind, (0, 0, 0))
            for v in vals:
                if abs(v - t) <= cfg.step_ms / 2:
                    bgr = (col[2], col[1], col[0])
                    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, tl.shape[0] - 1), bgr, 3)
        cv2.putText(cell, f"{t / 1000:.1f}", (3, tl.shape[0] + bar_h + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(cell)
    ch, cw = tiles[0].shape[:2]
    rows = (len(tiles) + cfg.cols - 1) // cfg.cols
    head = 22
    sheet = np.full((head + rows * (ch + 2), cfg.cols * (cw + 2), 3), 255, np.uint8)
    for k, cell in enumerate(tiles):
        r, c = divmod(k, cfg.cols)
        y, x = head + r * (ch + 2), c * (cw + 2)
        sheet[y: y + ch, x: x + cw] = cell
    cv2.putText(sheet, title[:150], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_png), sheet):
        raise OSError(f"could not write {out_png}")
    return {"tiles": len(tiles), "times_ms": times}
