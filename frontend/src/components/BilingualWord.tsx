import type { HiReview } from "../api/client";
import { useT } from "../i18n";
import { cx } from "./ui";

/** Short marker for how trustworthy the Hindi word is (docs/ISL_DESIGN.md 8.1). */
export function HindiReviewMark({ review, className }: { review: HiReview | undefined; className?: string }) {
  const { t } = useT();
  if (!review || review === "missing") return null;
  const meta: Record<string, { short: string; long: string; cls: string }> = {
    reviewed: { short: t("hi.reviewed.short"), long: t("hi.reviewed.long"), cls: "border-good/50 text-[#5fd35f]" },
    curated: { short: t("hi.curated.short"), long: t("hi.curated.long"), cls: "border-ink-600 text-fg-3" },
    machine: { short: t("hi.machine.short"), long: t("hi.machine.long"), cls: "border-warn/50 text-warn" },
    disputed: { short: t("hi.disputed.short"), long: t("hi.disputed.long"), cls: "border-danger/50 text-danger" },
    user: { short: t("hi.user.short"), long: t("hi.user.long"), cls: "border-accent/40 text-[#a9c8ff]" },
  };
  const m = meta[review];
  if (!m) return null;
  return (
    <span title={m.long} className={cx("inline-flex items-center whitespace-nowrap rounded-full border px-1.5 py-px text-[11px] font-medium leading-tight", m.cls, className)}>
      {m.short}
      <span className="sr-only">: {m.long}</span>
    </span>
  );
}

type Size = "sm" | "md" | "lg" | "xl";
const EN_SIZE: Record<Size, string> = { sm: "text-sm", md: "text-lg", lg: "text-2xl", xl: "text-4xl sm:text-5xl" };
const HI_SIZE: Record<Size, string> = { sm: "text-sm", md: "text-lg", lg: "text-2xl", xl: "text-4xl sm:text-5xl" };

/**
 * A recognised word in English AND Hindi, e.g. "WATER / पानी". Hindi is marked with its review state; a word with
 * no Hindi says so instead of inventing one.
 */
export function BilingualWord({ en, hi, review, size = "md", inline = false, tone = "fg", showMark = true, className }: {
  en: string; hi: string | null | undefined; review?: HiReview; size?: Size; inline?: boolean;
  tone?: "fg" | "teal" | "muted"; showMark?: boolean; className?: string;
}) {
  const toneCls = tone === "teal" ? "text-teal" : tone === "muted" ? "text-fg-3" : "text-fg";
  const big = size === "xl" || size === "lg";
  return (
    <span className={cx(inline ? "inline-flex flex-wrap items-baseline gap-x-2" : "flex flex-col", "min-w-0", className)} data-testid="bilingual-word">
      <span lang="en" className={cx("break-words font-semibold uppercase tracking-tight", EN_SIZE[size], toneCls)}>{en}</span>
      {/* Spaces are explicit text nodes (ignored by flex layout) so accessible names read "water पानी", not "waterपानी". */}
      {" "}
      {inline && <span aria-hidden className={cx("text-fg-3", EN_SIZE[size])}>/</span>}
      {" "}
      {hi ? (
        <span className={cx("inline-flex flex-wrap items-center gap-2", !inline && big && "mt-1")}>
          <span lang="hi" className={cx("font-hi break-words font-medium", HI_SIZE[size], toneCls)}>{hi}</span>
          {showMark && <HindiReviewMark review={review} />}
        </span>
      ) : (
        <span className={cx("text-fg-3", size === "sm" ? "text-[11px]" : "text-xs")}>
          <span lang="en">Hindi not available yet</span> / <span lang="hi" className="font-hi">हिंदी उपलब्ध नहीं</span>
        </span>
      )}
    </span>
  );
}
