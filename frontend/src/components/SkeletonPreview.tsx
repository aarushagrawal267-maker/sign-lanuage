import { useEffect, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import type { RawFrame } from "../api/client";
import { drawFrame } from "../mediapipe/draw";
import { useT } from "../i18n";

const prefersReducedMotion = () => typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Loops a landmark clip as a stick figure (body, hands, face points). Used for dictionary previews and recorded
 * takes: only landmark points exist, never video. The loop never ends, so it has a pause button (WCAG 2.2.2) and
 * starts paused (a still frame) when the system asks for reduced motion.
 */
export function SkeletonPreview({ frames, width, height, fps = 12, mirror = false, label, className }: {
  frames: RawFrame[]; width: number; height: number; fps?: number; mirror?: boolean; label: string; className?: string;
}) {
  const { t } = useT();
  const ref = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const frameRef = useRef(0);
  useEffect(() => {
    const c = ref.current;
    if (!c || !frames.length) return;
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
      drawFrame(ctx, frames[frameRef.current % frames.length], w, h, { pose: true, hands: true, face: true, mirror });
    };
    if (!playing) {
      // A still frame from the middle of the sign shows the hand shape better than the (resting) first frame.
      if (frameRef.current === 0) frameRef.current = Math.floor(frames.length / 2);
      draw();
      return;
    }
    draw();
    const id = window.setInterval(() => { frameRef.current += 1; draw(); }, 1000 / fps);
    return () => window.clearInterval(id);
  }, [frames, fps, mirror, playing]);
  return (
    <div className="relative">
      <canvas ref={ref} role="img" aria-label={label} className={className ?? "w-full rounded-xl border border-ink-700 bg-black"}
        style={{ aspectRatio: `${width} / ${height}` }} />
      {frames.length > 1 && (
        <button type="button" onClick={() => setPlaying((v) => !v)}
          aria-label={playing ? t("preview.pause") : t("preview.play")} title={playing ? t("preview.pause") : t("preview.play")}
          className="absolute bottom-1.5 right-1.5 inline-flex size-8 items-center justify-center rounded-md border border-ink-600 bg-ink-900/85 text-fg hover:bg-ink-800">
          {playing ? <Pause className="size-4" aria-hidden /> : <Play className="size-4" aria-hidden />}
        </button>
      )}
    </div>
  );
}
