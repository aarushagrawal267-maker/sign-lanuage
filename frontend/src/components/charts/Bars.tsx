import { useState } from "react";

export type Bar = { label: string; value: number; secondary?: number; hint?: string };

/** Vertical bar chart (single measure, optional lighter "secondary" overlay = a subset of value). */
export function ColumnChart({ bars, height = 160, color = "var(--color-series-1)", secondaryColor = "var(--color-series-3)", valueFormat = (v: number) => String(v), ariaLabel, legend }: {
  bars: Bar[]; height?: number; color?: string; secondaryColor?: string; valueFormat?: (v: number) => string; ariaLabel: string;
  legend?: { primary: string; secondary?: string };
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1, ...bars.map((b) => b.value));
  const n = bars.length;
  const gap = n > 60 ? 1 : 2;
  return (
    <div className="relative">
      {legend && (
        <div className="mb-2 flex flex-wrap gap-x-4 text-xs text-fg-2">
          <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm" style={{ background: color }} />{legend.primary}</span>
          {legend.secondary && <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm" style={{ background: secondaryColor }} />{legend.secondary}</span>}
        </div>
      )}
      <div className="flex items-end" style={{ height, gap }} role="img" aria-label={ariaLabel} onPointerLeave={() => setHover(null)}>
        {bars.map((b, i) => {
          const h = (b.value / max) * (height - 4);
          const hs = b.secondary !== undefined ? (b.secondary / max) * (height - 4) : 0;
          return (
            <div key={i} className="relative flex h-full flex-1 cursor-default items-end" onPointerEnter={() => setHover(i)} tabIndex={-1}>
              <div className="relative w-full rounded-t-[3px] transition-opacity" style={{ height: Math.max(h, b.value > 0 ? 2 : 0), background: color, opacity: hover === null || hover === i ? 1 : 0.55 }}>
                {b.secondary !== undefined && b.secondary > 0 && (
                  <div className="absolute bottom-0 left-0 w-full rounded-t-[3px]" style={{ height: Math.max(hs, 2), background: secondaryColor }} />
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-1 h-px bg-ink-600" />
      {hover !== null && bars[hover] && (
        <div className="pointer-events-none absolute -top-2 z-10 rounded-lg border border-ink-600 bg-ink-900/95 px-3 py-2 text-xs shadow-xl"
          style={{ left: `${Math.min(75, (hover / n) * 100)}%` }}>
          <div className="font-mono text-sm font-medium text-fg">{valueFormat(bars[hover].value)}{bars[hover].secondary !== undefined && <span className="text-fg-3"> · {valueFormat(bars[hover].secondary!)}</span>}</div>
          <div className="text-fg-2">{bars[hover].label}</div>
          {bars[hover].hint && <div className="text-fg-3">{bars[hover].hint}</div>}
        </div>
      )}
    </div>
  );
}

/** Horizontal labelled bars - used for top-k predictions and ranked lists. */
export function HBars({ items, max = 1, format = (v: number) => `${(v * 100).toFixed(1)}%`, highlightFirst = false, color = "var(--color-accent)" }: {
  items: { label: string; value: number; sub?: string }[]; max?: number; format?: (v: number) => string; highlightFirst?: boolean; color?: string;
}) {
  return (
    <ul className="flex flex-col gap-2">
      {items.map((it, i) => (
        <li key={it.label + i} className="grid grid-cols-[minmax(0,7rem)_1fr_3.5rem] items-center gap-3 text-sm">
          <span className={`truncate font-medium ${i === 0 && highlightFirst ? "text-fg" : "text-fg-2"}`} title={it.label}>{it.label}</span>
          <span className="h-2 overflow-hidden rounded-full bg-ink-700">
            <span className="block h-full rounded-full" style={{ width: `${Math.max(0, Math.min(1, it.value / max)) * 100}%`, background: i === 0 && highlightFirst ? "var(--color-teal)" : color, opacity: i === 0 || !highlightFirst ? 1 : 0.6 }} />
          </span>
          <span className="text-right font-mono text-xs text-fg-2 tabular">{format(it.value)}</span>
        </li>
      ))}
    </ul>
  );
}
