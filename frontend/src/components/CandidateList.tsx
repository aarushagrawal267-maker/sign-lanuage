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
 * is added to the text until the person chooses.
 */
export function CandidateList({ candidates, onChoose, onDismiss, onSpell, spellAvailable, disabled }: {
  candidates: Candidate[]; onChoose: (c: Candidate) => void; onDismiss?: () => void; onSpell?: () => void;
  spellAvailable?: boolean; disabled?: boolean;
}) {
  const { t } = useT();
  const origin = useOriginLabel();
  const list = candidates.slice(0, 5);
  return (
    <div data-testid="did-you-mean">
      <div className="mb-2.5 flex items-center gap-2 text-sm font-medium text-warn">
        <HelpCircle className="size-4" aria-hidden />
        <span>{t("live.didYouMean")}</span>
      </div>
      <p className="mb-3 text-xs text-fg-3">{t("live.didYouMeanHelp")}</p>
      <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2" aria-label={t("live.didYouMean")}>
        {list.map((c, i) => {
          const w = wordText(c);
          const o = origin(c);
          return (
            <li key={`${c.concept_id ?? c.gloss}-${i}`}>
              <button
                type="button"
                disabled={disabled}
                onClick={() => onChoose(c)}
                aria-label={t("live.chooseWord", { en: w.en, hi: w.hi ?? "", pct: Math.round(c.prob * 100) })}
                className={cx(
                  "flex min-h-14 w-full items-center justify-between gap-3 rounded-xl border px-3.5 py-2.5 text-left transition-colors",
                  "border-ink-600 bg-ink-800 hover:border-accent hover:bg-ink-700 disabled:cursor-not-allowed disabled:opacity-50",
                  i === 0 && "border-accent/60",
                )}
              >
                <span className="min-w-0">
                  <BilingualWord en={w.en} hi={w.hi} review={c.hi_review} size="md" inline showMark={false} />
                  {o && <span className="mt-0.5 block truncate text-[11px] text-fg-3">{o}</span>}
                </span>
                <span className="shrink-0 font-mono text-xs text-fg-3" aria-hidden>{Math.round(c.prob * 100)}%</span>
              </button>
            </li>
          );
        })}
      </ul>
      <div className="mt-3 flex flex-wrap gap-2">
        {onDismiss && <Button size="sm" variant="ghost" icon={<RotateCcw className="size-3.5" />} onClick={onDismiss}>{t("live.noneOfThese")}</Button>}
        {onSpell && spellAvailable && <Button size="sm" variant="ghost" icon={<SpellCheck className="size-3.5" />} onClick={onSpell}>{t("live.spellInstead")}</Button>}
      </div>
    </div>
  );
}

/** "Not sure - please sign again or spell it". Shown instead of guessing when nothing is close enough. */
export function NotSure({ onSpell, spellAvailable, spellHint }: { onSpell?: () => void; spellAvailable?: boolean; spellHint?: React.ReactNode }) {
  const { t } = useT();
  return (
    <div data-testid="not-sure" className="rounded-xl border border-warn/40 bg-warn/[0.06] px-4 py-4">
      <div className="text-xl font-semibold text-warn">{t("live.notSure")}</div>
      <p className="mt-1 text-sm text-fg-2">{t("live.notSureHelp")}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {onSpell && spellAvailable && <Button size="sm" variant="secondary" icon={<SpellCheck className="size-3.5" />} onClick={onSpell}>{t("live.spellIt")}</Button>}
        {!spellAvailable && spellHint}
      </div>
    </div>
  );
}
