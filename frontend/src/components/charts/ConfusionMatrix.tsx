import { useEffect, useRef, useState } from "react";

// Sequential blue ramp (reference palette, steps 100 -> 700) for magnitude.
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

function hexToRgb(h: string) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Row-normalised confusion matrix on canvas (100x100 is too dense for SVG), with hover readout. */
export function ConfusionMatrix({ matrix, labels, size = 520 }: { matrix: number[][]; labels: string[]; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const n = labels.length;
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0));

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size * dpr;
    c.height = size * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#111317";
    ctx.fillRect(0, 0, size, size);
    const cell = size / n;
    const stops = RAMP.map(hexToRgb);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = rowSums[i] ? matrix[i][j] / rowSums[i] : 0;
        if (v <= 0) continue;
        // dark UI: low values recede toward the surface, high values bright
        const t = Math.min(1, v);
        const idx = (1 - t) * (stops.length - 1);
        const a = stops[Math.floor(idx)], b = stops[Math.ceil(idx)];
        const f = idx - Math.floor(idx);
        const rgb = a.map((x, k) => Math.round(x + (b[k] - x) * f));
        ctx.globalAlpha = 0.25 + 0.75 * t;
        ctx.fillStyle = `rgb(${rgb.join(",")})`;
        ctx.fillRect(j * cell, i * cell, Math.max(1, cell - (n > 40 ? 0 : 1)), Math.max(1, cell - (n > 40 ? 0 : 1)));
      }
    }
    ctx.globalAlpha = 1;
  }, [matrix, n, size, rowSums]);

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const j = Math.floor(((e.clientX - r.left) / r.width) * n);
    const i = Math.floor(((e.clientY - r.top) / r.height) * n);
    if (i >= 0 && j >= 0 && i < n && j < n) setHover({ i, j });
  };

  return (
    <div className="flex flex-col gap-3 lg:flex-row lg:items-start">
      <div className="relative w-full max-w-[560px]">
        <div className="mb-1 text-right text-[11px] text-fg-3">predicted →</div>
        <canvas ref={ref} style={{ width: "100%", aspectRatio: "1 / 1" }} className="rounded-md border border-ink-700"
          onPointerMove={onMove} onPointerLeave={() => setHover(null)}
          role="img" aria-label={`Row-normalised confusion matrix over ${n} classes. Rows are true classes, columns predicted.`} />
        <div className="mt-1 text-[11px] text-fg-3">↑ true class (rows normalised to recall)</div>
      </div>
      <div className="min-w-56 flex-1 text-sm">
        <div className="mb-2 flex items-center gap-2 text-[11px] text-fg-3">
          0%
          <span className="h-2 w-32 rounded-full" style={{ background: `linear-gradient(90deg, #111317, ${RAMP.slice().reverse().join(",")})` }} />
          100% of true class
        </div>
        {hover ? (
          <div className="rounded-xl border border-ink-700 bg-ink-900 p-3">
            <div className="font-mono text-lg text-fg">{matrix[hover.i][hover.j]} <span className="text-sm text-fg-3">/ {rowSums[hover.i]}</span></div>
            <div className="text-fg-2">true <span className="font-medium text-fg">{labels[hover.i]}</span></div>
            <div className="text-fg-2">predicted <span className="font-medium text-fg">{labels[hover.j]}</span></div>
          </div>
        ) : (
          <p className="text-xs text-fg-3">Hover a cell to see counts. The diagonal is correct predictions; off-diagonal cells are confusions. Exact values are listed in the per-class table.</p>
        )}
      </div>
    </div>
  );
}
