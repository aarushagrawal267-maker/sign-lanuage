// The "current sign" area of the live page: the word in English + Hindi when it is accepted, "did you mean" chips when
// the recogniser is not sure enough, "not sure - try again or spell it" when it has no good guess, and the live
// letters in spelling mode. A word is never shown as recognised unless the engine accepted it or the user chose it.
import { useEffect, useRef, type ReactNode } from "react";
import { Check, RotateCcw, SpellCheck, X } from "lucide-react";
import type { Candidate } from "../api/client";
import { useT } from "../i18n";
import {
  candidateFromCommit, confirmKind, isConfirmOnlySuggestion, segmentDecision, wordText,
  type Certification, type SegmentResult, type SpellState, type TranscriptItem,
} from "../recognition/live";
import type { PendingWord, ResolvedHow } from "../recognition/useLiveRecognition";
import { BilingualWord } from "./BilingualWord";
import { CandidateList, NotSure, useOriginLabel } from "./CandidateList";
import { ConfirmWord } from "./ConfirmWord";
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
  resolved: { segment: SegmentResult; how: ResolvedHow } | null;
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
  /** Language packs (docs/ISL_DESIGN.md 15.29): a word the server accepted that waits for the user's tap. */
  pending?: PendingWord | null;
  /** Continuous signing is on, and whether sliding strict is certified (only then a word is shown as recognised). */
  continuous?: boolean;
  slidingStrict?: boolean;
  /** The active policy's certified precision in words, for the confirm card. */
  precisionNote?: (c: Certification | undefined) => string | null;
  onConfirmPending?: () => void;
  onRejectPending?: () => void;
};

/** A small "sign in progress: WORD" line under a card that waits for a tap. */
function ProgressLine({ top }: { top: Candidate | null }) {
  const { t } = useT();
  const w = top ? wordText(top) : null;
  return (
    <div className="flex items-baseline gap-2 text-xs text-fg-3">
      <span className="font-medium uppercase tracking-[0.14em]">{top ? t("live.provisional") : t("live.collecting")}</span>
      {w && <BilingualWord en={w.en} hi={w.hi} review={top!.hi_review} size="sm" inline tone="muted" showMark={false} />}
    </div>
  );
}

/**
 * The line under a word that waits for a tap. The policy's certified precision covers classifier and dictionary words
 * of the pack, so it is not shown for a dictionary-only word (open vocabulary) nor for one of the user's own signs,
 * which follow their own per-sign rule (docs/ISL_DESIGN.md 15.25, 15.30); a user sign gets its own line instead.
 */
function wordNote(word: Candidate, openVocabulary: boolean, cert: Certification | undefined,
  precisionNote: SignResultProps["precisionNote"], ownSign: string): string | null {
  if (word.source === "my_signs") return ownSign;
  if (openVocabulary) return null;
  return precisionNote?.(cert) ?? null;
}

/** How long taps are ignored after the next sign replaced the word on show. */
const REPLACE_GUARD_MS = 350;

/**
 * The confirm card for the word waiting for a tap. It is keyed on the word (segment id and time), so when the next
 * sign replaces the word React mounts new buttons: a key press or tap meant for the old word cannot add the new one.
 * Taps are also ignored for a moment after such a replacement, and focus returns to the card if it was inside the old
 * one (docs/ISL_DESIGN.md 15.30).
 */
function PendingConfirm({ pending, slidingStrict, precisionNote, onConfirm, onReject, onChoose }: {
  pending: PendingWord; slidingStrict?: boolean; precisionNote?: SignResultProps["precisionNote"];
  onConfirm: () => void; onReject: () => void; onChoose: (segment: SegmentResult, c: Candidate) => void;
}) {
  const { t } = useT();
  const c = pending.commit;
  const seg = pending.segment;
  const key = `${c.segment_id ?? "c"}-${c.t}`;
  const last = useRef<{ key: string; replaced: boolean } | null>(null);
  if (last.current?.key !== key) last.current = { key, replaced: last.current !== null };
  const focusInside = useRef(false);
  const word = candidateFromCommit(c);
  const kind = confirmKind(c, { continuous: pending.continuous, slidingStrict });
  return (
    <div onFocus={() => { focusInside.current = true; }}
      onBlur={(e) => { if (e.relatedTarget && !e.currentTarget.contains(e.relatedTarget as Node)) focusInside.current = false; }}>
      <ConfirmWord key={key} word={word} kind={kind} note={wordNote(word, !!c.open_vocabulary, c.certification, precisionNote, t("live.precisionOwnSign"))}
        dictionaryOnly={!!c.open_vocabulary} others={kind === "did_you_mean" ? seg.top : undefined}
        armMs={last.current.replaced ? REPLACE_GUARD_MS : 0} takeFocus={last.current.replaced && focusInside.current}
        onConfirm={onConfirm} onReject={onReject} onChooseOther={(x) => onChoose(seg, x)} />
    </div>
  );
}

export function SignResult(p: SignResultProps) {
  const { t } = useT();
  const ownSign = t("live.precisionOwnSign");
  const origin = useOriginLabel();
  const { lastResult } = p;
  const decision = segmentDecision(lastResult);
  const isResolved = !!p.resolved && p.resolved.segment === lastResult;
  const settledTop = lastResult?.top?.[0] ?? null;

  const spelling = p.spellOn && (p.inProgress || !lastResult);

  // A word waiting for the user's tap stays on show while the next sign is in progress, also while spelling
  // (docs/ISL_DESIGN.md 15.29, 15.30).
  if (p.pending && p.onConfirmPending && p.onRejectPending) {
    return (
      <div className="flex flex-col gap-3">
        <PendingConfirm pending={p.pending} slidingStrict={p.slidingStrict} precisionNote={p.precisionNote}
          onConfirm={p.onConfirmPending} onReject={p.onRejectPending} onChoose={p.onChoose} />
        {spelling ? <SpellPanel spell={p.spell} experimental={p.spellExperimental} /> : p.inProgress && <ProgressLine top={p.provisionalTop} />}
      </div>
    );
  }

  if (spelling) return <SpellPanel spell={p.spell} experimental={p.spellExperimental} />;

  // "Did you mean X?" because of the rules (an uncertified policy, a dictionary-only word), not because of doubt.
  if (lastResult && isConfirmOnlySuggestion(lastResult) && !isResolved) {
    const top = lastResult.top![0];
    const ov = lastResult.reason === "open_vocabulary" || !!lastResult.open_vocabulary;
    return (
      <div className="flex flex-col gap-3">
        <ConfirmWord word={top} kind="did_you_mean" note={wordNote(top, ov, lastResult.certification, p.precisionNote, ownSign)} dictionaryOnly={ov}
          others={lastResult.top} onConfirm={() => p.onChoose(lastResult, top)} onReject={() => p.onDismiss(lastResult)}
          onChooseOther={(x) => p.onChoose(lastResult, x)} />
        {p.inProgress && <ProgressLine top={p.provisionalTop} />}
      </div>
    );
  }

  if (p.inProgress) {
    const w = p.provisionalTop ? wordText(p.provisionalTop) : null;
    return (
      <div>
        <div className="text-xs font-medium uppercase tracking-[0.14em] text-fg-3">{p.provisionalTop ? t("live.provisional") : t("live.collecting")}</div>
        <div data-testid="current-gloss" className="mt-1">
          {w ? <BilingualWord en={w.en} hi={w.hi} review={p.provisionalTop!.hi_review} size="lg" tone="muted" showMark={false} /> : <span className="text-3xl text-fg-3">…</span>}
        </div>
      </div>
    );
  }

  // A word the user has just settled (chosen, dismissed or rejected as wrong) is never shown as "recognised" again.
  if (lastResult && isResolved) {
    return <ResolvedMessage how={p.resolved!.how} />;
  }

  if (lastResult && decision === "accept" && settledTop) {
    const w = wordText(settledTop);
    const o = origin(settledTop);
    return (
      <div data-testid="accepted" data-state="accepted" className="rounded-xl border-2 border-teal/60 bg-teal-soft/40 p-3.5">
        <div className="flex items-center gap-2.5 text-teal">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-teal text-ink-950" aria-hidden>
            <Check className="size-5" strokeWidth={3} />
          </span>
          <span className="text-base font-semibold first-letter:uppercase">{p.recognizedNow ? t("live.recognised") : t("live.lastRecognised")}</span>
        </div>
        <div className="mt-2 flex items-start gap-4">
          <div data-testid="current-gloss" className="min-w-0 flex-1">
            <BilingualWord en={w.en} hi={w.hi} review={settledTop.hi_review} size="xl" tone="teal" />
          </div>
          <div className="hidden shrink-0 sm:block">
            <ConfidenceRing value={lastResult.confidence ?? settledTop.prob} threshold={p.pack ? undefined : p.threshold ?? undefined} size={84} tone="teal" />
          </div>
        </div>
        {o && <div className="mt-1 text-xs text-fg-2">{o}</div>}
        {p.acceptedItem && (
          <div className="mt-3">
            <Button size="md" variant="danger" icon={<X className="size-3.5" />} onClick={() => p.onReject(p.acceptedItem!)}>{t("live.wrongWord")}</Button>
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

  if (lastResult && (decision === "abstain" || decision === "suggest")) {
    return <NotSure onSpell={p.onSpell} spellAvailable={p.spellAvailable} spellHint={p.spellHint} />;
  }

  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-[0.14em] text-fg-3">{t("live.word")}</div>
      <div data-testid="current-gloss" className="mt-1 text-3xl text-fg-3">—</div>
    </div>
  );
}

/**
 * What happened after the user acted on a result. Takes focus only when the control the user pressed has gone and
 * focus fell back to the page (WCAG 2.4.3), so keyboard and screen-reader users land on the confirmation.
 */
function ResolvedMessage({ how }: { how: ResolvedHow }) {
  const { t } = useT();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const active = document.activeElement;
    if (!active || active === document.body) ref.current?.focus();
  }, [how]);
  const chosen = how === "chosen";
  return (
    <div ref={ref} tabIndex={-1} data-testid="resolved" data-state={`resolved-${how}`}
      className="flex items-center gap-2.5 rounded-xl border border-ink-600 bg-ink-900 px-4 py-4 text-sm text-fg">
      <span className={chosen ? "text-teal" : "text-fg-2"} aria-hidden>{chosen ? <Check className="size-5" /> : <RotateCcw className="size-5" />}</span>
      <span>{chosen ? t("live.added") : how === "rejected" ? t("live.removedSignAgain") : t("live.signAgain")}</span>
    </div>
  );
}

/**
 * Screen-reader / braille announcement of each finished sign (deafblind users read the result on a braille display).
 * Always mounted (a live region must exist before its text changes); the key forces a fresh announcement even when
 * the same word is signed twice. The Hindi word sits in a lang="hi" span so the reader switches voice or braille table.
 */
export function ResultAnnouncer({ lastResult, resolved, inProgress, pending, slidingStrict }: {
  lastResult: SegmentResult | null; resolved: { segment: SegmentResult; how: ResolvedHow } | null; inProgress: boolean;
  pending?: PendingWord | null; slidingStrict?: boolean;
}) {
  const { t } = useT();
  let msg: ReactNode = null;
  const decision = segmentDecision(lastResult);
  const top = lastResult?.top?.[0] ?? null;
  const word = (en: string, hi: string | null) => <><span lang="en">{en}</span>{hi && <>, <span lang="hi">{hi}</span></>}</>;
  let key = `${lastResult?.segment_id ?? ""}-${lastResult ? "r" : "n"}-${decision ?? ""}`;
  if (pending) {
    // a word waiting for a tap: announced once, never as "recognised" alone (docs/ISL_DESIGN.md 15.29)
    const w = wordText(candidateFromCommit(pending.commit));
    const kind = confirmKind(pending.commit, { continuous: pending.continuous, slidingStrict });
    msg = <>{kind === "recognised" ? t("live.announceConfirm") : t("live.announceDidYouMeanOne")} {word(w.en, w.hi)}</>;
    key = `p-${pending.commit.segment_id ?? ""}-${pending.commit.t}`;
  } else if (lastResult && isConfirmOnlySuggestion(lastResult) && !(resolved && resolved.segment === lastResult)) {
    const w = wordText(top!);
    msg = <>{t("live.announceDidYouMeanOne")} {word(w.en, w.hi)}</>;
  } else if (lastResult && !inProgress && !(resolved && resolved.segment === lastResult)) {
    if (decision === "accept" && top) {
      const w = wordText(top);
      msg = <>{t("live.announceAccepted")} {word(w.en, w.hi)}</>;
    } else if (decision === "suggest" && lastResult.top?.length) {
      msg = t("live.announceSuggest", { n: Math.min(5, lastResult.top.length) });
    } else if (decision === "abstain" || decision === "suggest") {
      msg = t("live.notSure");
    }
  }
  return (
    <div role="log" aria-live="polite" aria-label={t("live.resultLog")} className="sr-only" data-testid="result-announcer">
      {msg && <p key={key}>{msg}</p>}
    </div>
  );
}
