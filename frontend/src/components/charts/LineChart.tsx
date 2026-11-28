import { useEffect, useMemo, useRef, useState } from "react";

export type Series = { key: string; label: string; color: string; values: (number | null)[]; dashed?: boolean };

type Props = {
  x: number[];
  series: Series[];
  height?: number;
  yDomain?: [number, number];
  xLabel?: string;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  ariaLabel: string;
};

const PAD = { l: 44, r: 16, t: 12, b: 28 };

function niceTicks(lo: number, hi: number, n = 4): number[] {
  if (hi === lo) return [lo];
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n)));
  const err = ((hi - lo) / n) / step;
  const s = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(10));
  return out;
}

/** Multi-series line chart with a snapping crosshair tooltip. One y-axis only. */
export function LineChart({ x, series, height = 220, yDomain, xLabel, yFormat = (v) => v.toFixed(2), xFormat = (v) => String(v), ariaLabel }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [w, setW] = useState(640);
  const [hover, setHover] = useState<number | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = boxRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([entry]) => setW(Math.max(240, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const [y0, y1] = useMemo(() => {
    if (yDomain) return yDomain;
    const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null && Number.isFinite(v)));
    if (!all.length) return [0, 1];
    const lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo || 1) * 0.08;
    return [Math.max(0, lo - pad), hi + pad];
  }, [series, yDomain]);

  const x0 = x.length ? x[0] : 0;
  const x1 = x.length > 1 ? x[x.length - 1] : x0 + 1;
  const iw = Math.max(10, w - PAD.l - PAD.r);
  const ih = height - PAD.t - PAD.b;
  const sx = (v: number) => PAD.l + ((v - x0) / (x1 - x0 || 1)) * iw;
  const sy = (v: number) => PAD.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;
  const yt = niceTicks(y0, y1);
  const xt = niceTicks(x0, x1, Math.min(6, Math.max(1, x.length - 1))).filter((v) => Number.isInteger(v) || x1 - x0 < 6);

  const path = (vals: (number | null)[]) => {
    let d = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (v === null || !Number.isFinite(v)) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || !x.length) return;
    const px = ((e.clientX - rect.left) / rect.width) * w;
    let best = 0;
    let bd = Infinity;
    x.forEach((xv, i) => {
      const d = Math.abs(sx(xv) - px);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    setHover(best);
  };

  return (
    <div className="relative" ref={boxRef}>
      {series.length > 1 && (
        <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-2" aria-hidden>
          {series.map((s) => (
            <span key={s.key} className="inline-flex items-center gap-1.5">
              <svg width="16" height="6"><line x1="0" y1="3" x2="16" y2="3" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "4 3" : undefined} /></svg>
              {s.label}
            </span>
          ))}
        </div>
      )}
      <svg ref={ref} viewBox={`0 0 ${w} ${height}`} width="100%" height={height} role="img" aria-label={ariaLabel}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)} className="touch-none select-none">
        {yt.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={w - PAD.r} y1={sy(t)} y2={sy(t)} stroke="var(--color-ink-700)" strokeWidth={1} />
            <text x={PAD.l - 8} y={sy(t) + 4} textAnchor="end" className="fill-fg-3 font-mono text-2xs">{yFormat(t)}</text>
          </g>
        ))}
        {xt.map((t) => (
          <text key={t} x={sx(t)} y={height - 8} textAnchor="middle" className="fill-fg-3 font-mono text-2xs">{xFormat(t)}</text>
        ))}
        {series.map((s) => (
          <path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" strokeDasharray={s.dashed ? "5 4" : undefined} />
        ))}
        {hover !== null && (
          <g>
            <line x1={sx(x[hover])} x2={sx(x[hover])} y1={PAD.t} y2={PAD.t + ih} stroke="var(--color-fg-3)" strokeWidth={1} />
            {series.map((s) => {
              const v = s.values[hover];
              return v === null || !Number.isFinite(v) ? null : (
                <circle key={s.key} cx={sx(x[hover])} cy={sy(v)} r={4} fill={s.color} stroke="var(--color-ink-850)" strokeWidth={2} />
              );
            })}
          </g>
        )}
      </svg>
      {xLabel && <div className="-mt-1 text-right text-2xs text-fg-3">{xLabel}</div>}
      {hover !== null && (
        <div className="pointer-events-none absolute top-6 z-10 min-w-36 rounded-lg border border-ink-600 bg-ink-900/95 px-3 py-2 text-xs shadow-xl"
          style={{ left: Math.min(Math.max(0, (sx(x[hover]) / w) * 100), 70) + "%" }}>
          <div className="mb-1 text-fg-3">{xLabel ?? "x"} {xFormat(x[hover])}</div>
          {series.map((s) => (
            <div key={s.key} className="flex items-center justify-between gap-3">
              <span className="inline-flex items-center gap-1.5 text-fg-2">
                <svg width="10" height="4"><line x1="0" y1="2" x2="10" y2="2" stroke={s.color} strokeWidth="2" /></svg>
                {s.label}
              </span>
              <span className="font-mono font-medium text-fg">{s.values[hover] === null ? "—" : yFormat(s.values[hover] as number)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
