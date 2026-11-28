import type { ButtonHTMLAttributes, ReactNode } from "react";
import { AlertTriangle, Info, Loader2 } from "lucide-react";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="max-w-3xl">
        {eyebrow && <div className="mb-1 text-xs text-fg-3">{eyebrow}</div>}
        <h1 className="text-2xl font-medium text-fg">{title}</h1>
        {description && <p className="mt-2 text-sm leading-relaxed text-fg-2">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Card({ title, subtitle, actions, children, className, padded = true }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; padded?: boolean;
}) {
  return (
    <section className={cx("rounded-xl border border-ink-700 bg-ink-850", className)}>
      {(title || actions) && (
        <div className="flex items-start justify-between gap-3 border-b border-ink-700/70 px-5 py-4">
          <div>
            {title && <h2 className="text-sm font-medium text-fg">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-fg-3">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={padded ? "p-5" : undefined}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "accent" | "teal" | "warn" | "danger" }) {
  const toneCls = tone === "teal" ? "text-teal" : tone === "accent" ? "text-accent-fg" : tone === "warn" ? "text-warn" : tone === "danger" ? "text-danger" : "text-fg";
  return (
    <div className="rounded-xl border border-ink-700 bg-ink-850 px-5 py-4">
      <div className="text-xs font-medium text-fg-3">{label}</div>
      <div className={cx("mt-2 font-mono text-2xl font-medium tabular", toneCls)}>{value}</div>
      {sub && <div className="mt-1 text-xs text-fg-3">{sub}</div>}
    </div>
  );
}

type BtnVariant = "primary" | "secondary" | "ghost" | "danger" | "teal";
export function Button({ variant = "secondary", size = "md", icon, loading, children, className, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: BtnVariant; size?: "sm" | "md" | "lg"; icon?: ReactNode; loading?: boolean;
}) {
  const v = {
    primary: "bg-accent text-white hover:bg-accent-hover border-transparent",
    teal: "bg-teal text-ink-950 hover:bg-teal-hover border-transparent",
    secondary: "bg-ink-800 text-fg hover:bg-ink-700 border-ink-700",
    ghost: "bg-transparent text-fg-2 hover:bg-ink-800 hover:text-fg border-transparent",
    danger: "bg-transparent text-danger hover:bg-danger/10 border-danger/40",
  }[variant];
  const s = { sm: "h-8 px-3 text-xs gap-1.5", md: "h-9 px-3.5 text-sm gap-2", lg: "h-11 px-5 text-sm gap-2" }[size];
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={cx("inline-flex select-none items-center justify-center rounded-lg border font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45", v, s, className)}
    >
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
}

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: "neutral" | "accent" | "teal" | "warn" | "danger" | "good"; className?: string }) {
  const t = {
    neutral: "border-ink-600 bg-ink-800 text-fg-2",
    accent: "border-accent/50 bg-accent-soft/60 text-accent-fg",
    teal: "border-teal/40 bg-teal-soft text-teal",
    warn: "border-warn/40 bg-warn/10 text-warn",
    danger: "border-danger/40 bg-danger/10 text-danger",
    good: "border-good/50 bg-good/10 text-good-fg",
  }[tone];
  return <span className={cx("inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-2xs font-medium", t, className)}>{children}</span>;
}

export function Notice({ tone = "info", title, children }: { tone?: "info" | "warn" | "danger"; title?: string; children: ReactNode }) {
  const t = tone === "warn" ? "border-warn/35 bg-warn/[0.06]" : tone === "danger" ? "border-danger/40 bg-danger/[0.07]" : "border-accent/30 bg-accent/[0.06]";
  const Icon = tone === "info" ? Info : AlertTriangle;
  const ic = tone === "warn" ? "text-warn" : tone === "danger" ? "text-danger" : "text-accent-fg";
  return (
    <div role={tone === "info" ? "note" : "alert"} className={cx("flex gap-3 rounded-lg border px-4 py-3", t)}>
      <Icon className={cx("mt-0.5 size-4 shrink-0", ic)} aria-hidden />
      <div className="text-sm leading-relaxed text-fg-2">
        {title && <div className="mb-0.5 font-medium text-fg">{title}</div>}
        {children}
      </div>
    </div>
  );
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      {icon && <div className="mb-1 text-fg-3">{icon}</div>}
      <div className="text-sm font-medium text-fg">{title}</div>
      {children && <div className="max-w-md text-sm text-fg-3">{children}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function Progress({ value, tone = "accent" }: { value: number; tone?: "accent" | "teal" }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-700" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={cx("h-full origin-left transition-transform duration-300 ease-out", tone === "teal" ? "bg-teal" : "bg-accent")} style={{ transform: `scaleX(${Math.max(0, Math.min(1, value))})` }} />
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-fg-2">{label}</span>
      {children}
      {hint && <span className="text-2xs text-fg-3">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "h-9 w-full rounded-lg border border-ink-700 bg-ink-900 px-3 text-sm text-fg placeholder:text-fg-3 focus:border-accent focus:outline-none";

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)} className="flex items-center gap-2.5 text-left text-sm text-fg-2">
      <span className={cx("relative inline-flex h-5 w-9 shrink-0 rounded-full border transition-colors", checked ? "border-accent bg-accent" : "border-ink-600 bg-ink-800")}>
        <span className={cx("absolute top-0.5 size-3.5 rounded-full bg-white transition-transform", checked ? "translate-x-4" : "translate-x-0.5")} />
      </span>
      {label}
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange, ariaLabel }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; ariaLabel: string }) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-lg border border-ink-700 bg-ink-900 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx("rounded-md px-3 py-1 text-xs font-medium transition-colors", value === o.value ? "bg-ink-700 text-fg" : "text-fg-3 hover:text-fg")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ErrorText({ error }: { error: string | null | undefined }) {
  if (!error) return null;
  return <p className="text-sm text-danger" role="alert">{error}</p>;
}

export function Kv({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-ink-700/60 py-2 text-sm last:border-0">
      <span className="text-fg-3">{k}</span>
      <span className="text-right font-mono text-sm text-fg tabular">{v}</span>
    </div>
  );
}

export function SyntheticBadge() {
  return <Badge tone="warn">synthetic test model</Badge>;
}
