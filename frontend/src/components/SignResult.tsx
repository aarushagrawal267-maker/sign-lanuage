// The "current sign" area of the live page: the word in English + Hindi when it is accepted, "did you mean" chips when
// the recogniser is not sure enough, "not sure - try again or spell it" when it has no good guess, and the live
// letters in spelling mode. A word is never shown as recognised unless the engine accepted it or the user chose it.
import type { ReactNode } from "react";
import { CheckCircle2, SpellCheck, X } from "lucide-react";
import type { Candidate } from "../api/client";
import { useT } from "../i18n";
import { segmentDecision, wordText, type SegmentResult, type SpellState, type TranscriptItem } from "../recognition/live";
import { BilingualWord } from "./BilingualWord";
import { CandidateList, NotSure, useOriginLabel } from "./CandidateList";
import { ConfidenceRing } from "./charts/ConfidenceRing";
import { Badge, Button } from "./ui";

/** Live letters while fingerspelling: the letters seen so far and the closest dictionary words. */
export function SpellPanel({ spell, experimental }: { spell: SpellState | null; experimental: boolean }) {
  const { t } = useT();
  return (
    <div data-testid="spell-panel" className="rounded-xl border border-accent/40 bg-accent/[0.05] p-4">
      <div className="mb-2 flex items-center gap-2 text-xs text-fg-2">
        <SpellCheck className="size-4 text-accent" aria-hidden />
        <span className="font-medium">{t("spell.title")}</span>
        {experimental && <Badge tone="warn">{t("spell.experimental")}</Badge>}
      </div>
      <div className="min-h-12 font-mono text-4xl font-semibold tracking-[0.2em] text-fg" aria-live="polite" aria-label={t("spell.letters")}>
        {spell?.buffer || <span className="text-base tracking-normal text-fg-3">{t("spell.waiting")}</span>}
      </div>
      {spell && spell.letters.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {spell.letters.map((l, i) => (
            <span key={i} className="rounded border border-ink-600 bg-ink-800 px-1.5 py-0.5 font-mono text-[11px] text-fg-2">
              {l.letter} <span className="text-fg-3">{Math.round(l.prob * 100)}</span>
            </span>
          ))}
        </div>
      )}
      {spell && spell.suggestions.length > 0 && (
        <div className="mt-3 text-xs text-fg-3">
          {t("spell.closest")}: {spell.suggestions.slice(0, 5).map((s) => s.word.toUpperCase()).join(" · ")}
        </div>
      )}
      <p className="mt-3 text-xs text-fg-3">{t("spell.help")}</p>
    </div>
  );
}

export type SignResultProps = {
  lastResult: SegmentResult | null;
  resolved: { segment: SegmentResult; how: "chosen" | "dismissed" } | null;
  inProgress: boolean;
  provisionalTop: Candidate | null;
  recognizedNow: boolean;
  pack: boolean;
  threshold: number | null;
  spellOn: boolean;
  spell: SpellState | null;
  spellAvailable: boolean;
  spellExperimental: boolean;
  acceptedItem: TranscriptItem | null;
  spellHint?: ReactNode;
  onChoose: (segment: SegmentResult, c: Candidate) => void;
  onDismiss: (segment: SegmentResult) => void;
  onReject: (item: TranscriptItem) => void;
  onSpell: () => void;
};

export function SignResult(p: SignResultProps) {
  const { t } = useT();
  const origin = useOriginLabel();
  const { lastResult } = p;
  const decision = segmentDecision(lastResult);
  const isResolved = !!p.resolved && p.resolved.segment === lastResult;
  const settledTop = lastResult?.top?.[0] ?? null;

  if (p.spellOn && (p.inProgress || !lastResult)) return <SpellPanel spell={p.spell} experimental={p.spellExperimental} />;

  if (p.inProgress) {
    const w = p.provisionalTop ? wordText(p.provisionalTop) : null;
    return (
      <div>
        <div className="text-[11px] uppercase tracking-[0.14em] text-fg-3">{p.provisionalTop ? t("live.provisional") : t("live.collecting")}</div>
        <div data-testid="current-gloss" className="mt-1">
          {w ? <BilingualWord en={w.en} hi={w.hi} review={p.provisionalTop!.hi_review} size="lg" tone="muted" showMark={false} /> : <span className="text-3xl text-fg-3">…</span>}
        </div>
      </div>
    );
  }

  if (lastResult && decision === "accept" && settledTop) {
    const w = wordText(settledTop);
    const o = origin(settledTop);
    return (
      <div data-testid="accepted">
        <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.14em] text-teal">
          <CheckCircle2 className="size-3.5" aria-hidden />{p.recognizedNow ? t("live.recognised") : t("live.lastRecognised")}
        </div>
        <div className="mt-2 flex items-start gap-4">
          <div data-testid="current-gloss" className="min-w-0 flex-1">
            <BilingualWord en={w.en} hi={w.hi} review={settledTop.hi_review} size="xl" tone="teal" />
          </div>
          <div className="hidden shrink-0 sm:block">
            <ConfidenceRing value={lastResult.confidence ?? settledTop.prob} threshold={p.pack ? undefined : p.threshold ?? undefined} size={84} tone="teal" />
          </div>
        </div>
        {o && <div className="mt-1 text-xs text-fg-3">{o}</div>}
        {p.acceptedItem && (
          <div className="mt-3">
            <Button size="sm" variant="danger" icon={<X className="size-3.5" />} onClick={() => p.onReject(p.acceptedItem!)}>{t("live.wrongWord")}</Button>
          </div>
        )}
      </div>
    );
  }

  if (lastResult && decision === "suggest" && lastResult.top?.length && !isResolved) {
    return (
      <CandidateList candidates={lastResult.top} onChoose={(c) => p.onChoose(lastResult, c)} onDismiss={() => p.onDismiss(lastResult)}
        onSpell={p.onSpell} spellAvailable={p.spellAvailable} />
    );
  }

  if (lastResult && isResolved) {
    return (
      <div className="rounded-xl border border-ink-700 bg-ink-900 px-4 py-4 text-sm text-fg-2" data-testid="resolved">
        {p.resolved!.how === "chosen" ? t("live.added") : t("live.signAgain")}
      </div>
    );
  }

  if (lastResult && (decision === "abstain" || decision === "suggest")) {
    return <NotSure onSpell={p.onSpell} spellAvailable={p.spellAvailable} spellHint={p.spellHint} />;
  }

  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.14em] text-fg-3">{t("live.word")}</div>
      <div data-testid="current-gloss" className="mt-1 text-3xl text-fg-3">—</div>
    </div>
  );
}
