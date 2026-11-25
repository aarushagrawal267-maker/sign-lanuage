import { useEffect, useRef } from "react";
import type { RawFrame } from "../api/client";
import { drawFrame } from "../mediapipe/draw";

/**
 * Loops a landmark clip as a stick figure (body, hands, face points). Used for dictionary previews and recorded
 * takes: only landmark points exist, never video.
 */
export function SkeletonPreview({ frames, width, height, fps = 12, mirror = false, label, className }: {
  frames: RawFrame[]; width: number; height: number; fps?: number; mirror?: boolean; label: string; className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !frames.length) return;
    const reduce = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let i = 0;
    const draw = () => {
      const w = c.clientWidth || 240, h = c.clientHeight || 180;
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== Math.round(w * dpr)) {
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
      }
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawFrame(ctx, frames[i % frames.length], w, h, { pose: true, hands: true, face: true, mirror });
      i += 1;
    };
    draw();
    if (reduce) return; // show a still frame only
    const id = window.setInterval(draw, 1000 / fps);
    return () => window.clearInterval(id);
  }, [frames, fps, mirror]);
  return (
    <canvas ref={ref} role="img" aria-label={label} className={className ?? "w-full rounded-xl border border-ink-700 bg-black"}
      style={{ aspectRatio: `${width} / ${height}` }} />
  );
}
