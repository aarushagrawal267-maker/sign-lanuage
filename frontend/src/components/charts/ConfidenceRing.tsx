/** Circular confidence meter with a marker at the acceptance threshold. */
export function ConfidenceRing({ value, threshold, size = 120, tone = "accent" }: { value: number | null; threshold?: number; size?: number; tone?: "accent" | "teal" | "muted" }) {
  const r = size / 2 - 8;
  const c = 2 * Math.PI * r;
  const v = value === null ? 0 : Math.max(0, Math.min(1, value));
  const color = tone === "teal" ? "var(--color-teal)" : tone === "muted" ? "var(--color-fg-3)" : "var(--color-accent)";
  const th = threshold !== undefined ? threshold * 2 * Math.PI - Math.PI / 2 : null;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={value === null ? "no confidence" : `confidence ${(v * 100).toFixed(0)} percent`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-ink-700)" strokeWidth={6} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={6} strokeLinecap="round"
        strokeDasharray={`${v * c} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: "stroke-dasharray 200ms ease-out" }} />
      {th !== null && (
        <line x1={size / 2 + (r - 7) * Math.cos(th)} y1={size / 2 + (r - 7) * Math.sin(th)} x2={size / 2 + (r + 7) * Math.cos(th)} y2={size / 2 + (r + 7) * Math.sin(th)}
          stroke="var(--color-fg-2)" strokeWidth={2} strokeLinecap="round" />
      )}
      <text x="50%" y="50%" dominantBaseline="central" textAnchor="middle" className="fill-fg font-mono text-[20px] font-medium">
        {value === null ? "—" : `${Math.round(v * 100)}%`}
      </text>
    </svg>
  );
}
