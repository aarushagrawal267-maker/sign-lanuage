import { useId } from "react";
import { HelpCircle, RotateCcw, SpellCheck } from "lucide-react";
import type { Candidate } from "../api/client";
import { useT } from "../i18n";
import { wordText } from "../recognition/live";
import { BilingualWord } from "./BilingualWord";
import { Button, cx } from "./ui";

/** Where a candidate came from, in a few plain words (shown under the word). */
export function useOriginLabel() {
  const { t } = useT();
  return (c: Candidate): string | null => {
    if (c.source === "my_signs") return c.origin && c.origin !== "My signs" ? c.origin.replace(/^My signs/, t("origin.mySigns")) : t("origin.mySigns");
    if (c.source === "spelling") return c.origin?.includes("not in dictionary") ? t("origin.spelledNew") : t("origin.spelled");
    if (c.region) return t("origin.region", { region: c.region });
    return null;
  };
}

/**
 * "Did you mean…?": the top candidates as large tappable chips (English + Hindi). Tapping one confirms it; nothing
 * is added to the text until the person chooses. The panel has its own colour, border and question-mark icon so it
 * can never be mistaken for a recognised word (amber + "?" here, teal + tick for recognised, dashed + retry arrow for
 * "not sure"), also for people who cannot read the heading.
 */
export function CandidateList({ candidates, onChoose, onDismiss, onSpell, spellAvailable, disabled }: {
  candidates: Candidate[]; onChoose: (c: Candidate) => void; onDismiss?: () => void; onSpell?: () => void;
  spellAvailable?: boolean; disabled?: boolean;
}) {
  const { t } = useT();
  const origin = useOriginLabel();
  const list = candidates.slice(0, 5);
  const headingId = useId();
  return (
    <div data-testid="did-you-mean" data-state="suggest" className="rounded-xl border-2 border-warn/60 bg-warn/[0.06] p-3.5">
      <div className="mb-1 flex items-center gap-2.5 text-warn">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-warn text-ink-950" aria-hidden>
          <HelpCircle className="size-5" strokeWidth={2.5} />
        </span>
        <h3 id={headingId} className="text-lg font-semibold leading-tight">{t("live.didYouMean")}</h3>
      </div>
      <p className="mb-3 text-sm text-fg-2">{t("live.didYouMeanHelp")}</p>
      <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2" aria-labelledby={headingId}>
        {list.map((c, i) => {
          const w = wordText(c);
          const o = origin(c);
          const pct = Math.round(c.prob * 100);
          return (
            <li key={`${c.concept_id ?? c.gloss}-${i}`}>
              {/* The accessible name comes from the visible text (WCAG 2.5.3), with the Hindi word in its own lang="hi"
                  span so screen readers switch voice for it (3.1.2); no aria-label that would hide both. */}
              <button
                type="button"
                disabled={disabled}
                onClick={() => onChoose(c)}
                className={cx(
                  "flex min-h-14 w-full items-center justify-between gap-3 rounded-xl border-2 px-3.5 py-2.5 text-left transition-colors",
                  "border-ink-600 bg-ink-800 hover:border-accent hover:bg-ink-700 disabled:cursor-not-allowed disabled:opacity-50",
                  i === 0 && "border-accent/60",
                )}
              >
                <span className="min-w-0">
                  <span className="sr-only">{t("live.choose")}</span>{" "}
                  <BilingualWord en={w.en} hi={w.hi} review={c.hi_review} size="md" inline showMark={false} />
                  {o && <>{" "}<span className="mt-0.5 block truncate text-xs text-fg-3">{o}</span></>}
                </span>
                {" "}
                <span className="shrink-0 font-mono text-xs text-fg-3">{t("live.sure", { pct })}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <div className="mt-3 flex flex-wrap gap-2">
        {onDismiss && <Button size="md" variant="secondary" icon={<RotateCcw className="size-4" />} onClick={onDismiss}>{t("live.noneOfThese")}</Button>}
        {onSpell && spellAvailable && <Button size="md" variant="secondary" icon={<SpellCheck className="size-4" />} onClick={onSpell}>{t("live.spellInstead")}</Button>}
      </div>
    </div>
  );
}

/**
 * "Not sure - please sign again or spell it". Shown instead of guessing when nothing is close enough. Dashed border,
 * orange and a "try again" arrow: visibly different from both "recognised" and "did you mean".
 */
export function NotSure({ onSpell, spellAvailable, spellHint }: { onSpell?: () => void; spellAvailable?: boolean; spellHint?: React.ReactNode }) {
  const { t } = useT();
  return (
    <div data-testid="not-sure" data-state="not-sure" className="rounded-xl border-2 border-dashed border-serious/70 bg-serious/[0.06] px-4 py-4">
      <div className="flex items-center gap-2.5 text-serious">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-full border-2 border-serious" aria-hidden>
          <RotateCcw className="size-4" strokeWidth={2.5} />
        </span>
        <h3 className="text-xl font-semibold leading-tight">{t("live.notSure")}</h3>
      </div>
      <p className="mt-2 text-sm text-fg-2">{t("live.notSureHelp")}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {onSpell && spellAvailable && <Button size="md" variant="secondary" icon={<SpellCheck className="size-4" />} onClick={onSpell}>{t("live.spellIt")}</Button>}
        {!spellAvailable && spellHint}
      </div>
    </div>
  );
}
