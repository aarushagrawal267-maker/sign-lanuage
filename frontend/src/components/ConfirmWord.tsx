// A word that waits for the user's tap (docs/ISL_DESIGN.md 15.29). The app default for ISL keeps a confirmation tap
// before a word is added to the sentence, spoken or sent, even when the policy is certified: the certified precision
// (about 95 of 100 for strict) is below the "without any fail" goal. Two looks:
//  * "recognised": the policy is certified for this word. Teal, a tick, "Add water" / "Wrong word".
//  * "did you mean": the policy is not certified (confirm-only) or the word is dictionary-only. Amber, a question
//    mark, "Yes, add water" / "No, sign again", and the other guesses as smaller buttons.
import { useId } from "react";
import { Check, HelpCircle, RotateCcw, X } from "lucide-react";
import type { Candidate } from "../api/client";
import { useT } from "../i18n";
import { wordText } from "../recognition/live";
import { BilingualWord } from "./BilingualWord";
import { useOriginLabel } from "./CandidateList";
import { Button, cx } from "./ui";

export type ConfirmKind = "recognised" | "did_you_mean";

export function ConfirmWord({ word, kind, note, dictionaryOnly, others, onConfirm, onReject, onChooseOther }: {
  word: Candidate;
  kind: ConfirmKind;
  /** The certified precision of the active policy, in words ("Strict: right about 95 of 100 times when it answers"). */
  note?: string | null;
  dictionaryOnly?: boolean;
  others?: Candidate[];
  onConfirm: () => void;
  onReject: () => void;
  onChooseOther?: (c: Candidate) => void;
}) {
  const { t } = useT();
  const origin = useOriginLabel();
  const headingId = useId();
  const helpId = useId();
  const w = wordText(word);
  const o = origin(word);
  const recognised = kind === "recognised";
  const rest = (others ?? []).filter((c) => c.concept_id !== word.concept_id).slice(0, 4);
  return (
    <section data-testid="confirm-word" data-state={recognised ? "confirm-recognised" : "confirm-did-you-mean"}
      aria-labelledby={headingId} aria-describedby={helpId}
      className={cx("rounded-xl border-2 p-3.5", recognised ? "border-teal/60 bg-teal-soft/40" : "border-warn/60 bg-warn/[0.06]")}>
      <div className={cx("flex items-center gap-2.5", recognised ? "text-teal" : "text-warn")}>
        <span className={cx("flex size-8 shrink-0 items-center justify-center rounded-full text-ink-950", recognised ? "bg-teal" : "bg-warn")} aria-hidden>
          {recognised ? <Check className="size-5" strokeWidth={3} /> : <HelpCircle className="size-5" strokeWidth={2.5} />}
        </span>
        <h3 id={headingId} className="text-lg font-semibold leading-tight">{recognised ? t("live.confirmRecognised") : t("live.didYouMeanOne")}</h3>
      </div>
      <div data-testid="confirm-gloss" className="mt-2">
        <BilingualWord en={w.en} hi={w.hi} review={word.hi_review} size="xl" tone={recognised ? "teal" : "fg"} />
      </div>
      {o && <div className="mt-1 text-xs text-fg-2">{o}</div>}
      {note && <p data-testid="confirm-precision" className="mt-2 text-sm text-fg-2">{note}</p>}
      {dictionaryOnly && <p className="mt-1 text-xs text-fg-3">{t("live.dictionaryOnly")}</p>}
      <p id={helpId} className="mt-1 text-xs text-fg-3">{t("live.confirmHelp")}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button size="md" variant="primary" icon={<Check className="size-4" />} onClick={onConfirm}>
          {recognised ? t("live.confirmAdd", { word: w.en }) : t("live.confirmYes", { word: w.en })}
        </Button>
        <Button size="md" variant={recognised ? "danger" : "secondary"} icon={recognised ? <X className="size-4" /> : <RotateCcw className="size-4" />} onClick={onReject}>
          {recognised ? t("live.confirmWrong") : t("live.confirmNo")}
        </Button>
      </div>
      {rest.length > 0 && onChooseOther && (
        <div className="mt-3">
          <div className="mb-1.5 text-xs text-fg-3">{t("live.otherGuesses")}</div>
          <ul className="flex flex-wrap gap-2">
            {rest.map((c, i) => {
              const cw = wordText(c);
              return (
                <li key={`${c.concept_id ?? c.gloss}-${i}`}>
                  {/* named by its visible text (WCAG 2.5.3); the Hindi word keeps its lang="hi" span */}
                  <button type="button" onClick={() => onChooseOther(c)}
                    className="inline-flex min-h-11 items-center gap-1 rounded-lg border-2 border-ink-600 bg-ink-800 px-3 py-1.5 text-left hover:border-accent hover:bg-ink-700">
                    <span className="sr-only">{t("live.choose")}</span>{" "}
                    <BilingualWord en={cw.en} hi={cw.hi} review={c.hi_review} size="sm" inline showMark={false} />
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
