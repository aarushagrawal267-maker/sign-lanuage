import { useEffect, useRef, useState } from "react";
import { Copy, Eraser, Info, Undo2, Volume2, X } from "lucide-react";
import { IS_STATIC, api, type SentenceResponse } from "../api/client";
import { useT } from "../i18n";
import { useLocalStorage } from "../lib/hooks";
import { onVoicesChanged, speak, speechAvailable, type SpeechLang } from "../lib/speech";
import { sentenceWords, wordText, type TranscriptItem } from "../recognition/live";
import { BilingualWord } from "./BilingualWord";
import { Button, Card, Segmented, Toggle, cx } from "./ui";

type SpeechPrefs = { auto: boolean; lang: SpeechLang };

/** Re-render when the browser's voice list arrives (Chrome loads voices asynchronously). */
function useVoicesVersion() {
  const [v, setV] = useState(0);
  useEffect(() => onVoicesChanged(() => setV((x) => x + 1)), []);
  return v;
}

/**
 * The words signed so far (English + Hindi chips) and an approximate English and Hindi sentence built by the
 * backend's rule-based sentence builder (docs/ISL_DESIGN.md 8.2), with optional speak-aloud (8.3).
 */
export function SentenceBar({ items, language, onClear, onRemove, onWrong }: {
  items: TranscriptItem[]; language: string; onClear: () => void; onRemove: (id: number) => void; onWrong: (item: TranscriptItem) => void;
}) {
  const { t } = useT();
  const [sentence, setSentence] = useState<SentenceResponse | null>(null);
  const [sentenceError, setSentenceError] = useState(false);
  const [prefs, setPrefs] = useLocalStorage<SpeechPrefs>("speech-prefs-v1", { auto: false, lang: "hi-IN" });
  useVoicesVersion();
  const hiVoice = speechAvailable("hi-IN");
  const enVoice = speechAvailable("en-IN");
  const anyVoice = hiVoice || enVoice;
  const speakLang: SpeechLang = prefs.lang === "hi-IN" && hiVoice ? "hi-IN" : "en-IN";

  // Build the sentence whenever the words change (debounced; nothing is stored on the server).
  useEffect(() => {
    if (items.length === 0 || IS_STATIC) {
      setSentence(null);
      return;
    }
    let cancelled = false;
    const id = window.setTimeout(() => {
      api.post<SentenceResponse>(`/api/languages/${encodeURIComponent(language)}/sentence`, { words: sentenceWords(items) })
        .then((r) => { if (!cancelled) { setSentence(r); setSentenceError(false); } })
        .catch(() => { if (!cancelled) { setSentence(null); setSentenceError(true); } });
    }, 250);
    return () => { cancelled = true; window.clearTimeout(id); };
  }, [items, language]);

  // Auto-speak each newly added word (off by default).
  const lastSpoken = useRef<number | null>(items.length ? items[items.length - 1].id : null);
  useEffect(() => {
    const last = items[items.length - 1];
    if (!last || last.id === lastSpoken.current) return;
    lastSpoken.current = last.id;
    if (!prefs.auto) return;
    const w = wordText(last);
    if (speakLang === "hi-IN" && w.hi) speak(w.hi, "hi-IN");
    else speak(w.en, "en-IN");
  }, [items, prefs.auto, speakLang]);

  // Removing a word removes the button that had focus: keep focus in the word list instead of losing it (WCAG 2.4.3).
  const listRef = useRef<HTMLDivElement>(null);
  const keepFocus = (fn: () => void) => () => { listRef.current?.focus(); fn(); };

  const fallbackEn = items.map((x) => wordText(x).en).join(" ");
  const fallbackHi = items.map((x) => wordText(x).hi ?? `[${wordText(x).en}]`).join(" ");
  const enText = sentence?.en.text ?? fallbackEn;
  const hiText = sentence?.hi.text ?? fallbackHi;
  const copyText = `${enText}\n${hiText}`;

  const speakSentence = () => {
    if (speakLang === "hi-IN") speak(hiText, "hi-IN");
    else speak(enText, "en-IN");
  };

  return (
    <Card
      title={t("sentence.title")}
      subtitle={t("sentence.subtitle")}
      actions={<>
        <Button size="sm" variant="ghost" icon={<Copy className="size-3.5" />} disabled={!items.length} onClick={() => navigator.clipboard?.writeText(copyText)}>{t("common.copy")}</Button>
        <Button size="sm" variant="ghost" icon={<Eraser className="size-3.5" />} disabled={!items.length} onClick={keepFocus(onClear)}>{t("common.clear")}</Button>
      </>}
    >
      <div data-testid="transcript" ref={listRef} tabIndex={-1} role="group" aria-label={t("sentence.words")}
        className="flex min-h-16 flex-wrap content-start gap-2 rounded-lg">
        {items.length === 0 ? (
          <span className="text-sm text-fg-3">{t("sentence.empty")}</span>
        ) : (
          items.map((x, i) => {
            const w = wordText(x);
            const isLast = i === items.length - 1;
            return (
              <span key={x.id} className={cx("fade-in inline-flex items-center gap-1.5 rounded-lg border py-1 pl-2.5 pr-1",
                x.chosen ? "border-accent/40 bg-accent-soft/40" : "border-teal/30 bg-teal-soft/50")}>
                <BilingualWord en={w.en} hi={w.hi} review={x.hi_review} size="sm" inline tone={x.chosen ? "fg" : "teal"} showMark={false} />
                {isLast && x.segment_id != null && !x.chosen ? (
                  <button type="button" onClick={keepFocus(() => onWrong(x))} className="inline-flex size-8 items-center justify-center rounded-md text-fg-3 hover:bg-ink-700 hover:text-danger" aria-label={t("sentence.wrong", { word: w.en })} title={t("sentence.wrong", { word: w.en })}>
                    <X className="size-4" aria-hidden />
                  </button>
                ) : (
                  <button type="button" onClick={keepFocus(() => onRemove(x.id))} className="inline-flex size-8 items-center justify-center rounded-md text-fg-3 hover:bg-ink-700 hover:text-fg" aria-label={t("sentence.remove", { word: w.en })} title={t("sentence.remove", { word: w.en })}>
                    <Undo2 className="size-4" aria-hidden />
                  </button>
                )}
              </span>
            );
          })
        )}
      </div>

      {items.length > 0 && (
        <div className="mt-4 rounded-xl border border-ink-700 bg-ink-900 p-3.5" data-testid="sentence">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-warn">
            <Info className="size-3.5" aria-hidden />
            <span>{t("sentence.approx")}</span>
            {sentence?.en.rules?.length ? <span className="text-fg-3" title={[...sentence.en.rules, ...sentence.hi.rules].join("\n")}>· {t("sentence.rules", { n: sentence.en.rules.length + sentence.hi.rules.length })}</span> : null}
          </div>
          <p lang="en" className="text-xl font-medium leading-snug text-fg" data-testid="sentence-en">{enText}</p>
          <p lang="hi" className="font-hi mt-1 text-xl font-medium leading-snug text-fg" data-testid="sentence-hi">{hiText}</p>
          {sentenceError && <p className="mt-1 text-[11px] text-fg-3">{t("sentence.offline")}</p>}
          {sentence?.note && <p className="mt-1 text-[11px] text-fg-3">{sentence.note}</p>}
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-ink-700/70 pt-3">
        <Button size="sm" variant="secondary" icon={<Volume2 className="size-3.5" />} disabled={!items.length || !anyVoice} onClick={speakSentence}>
          {t("sentence.speak")}
        </Button>
        <Segmented ariaLabel={t("sentence.voice")} value={prefs.lang} onChange={(v) => setPrefs({ ...prefs, lang: v })}
          options={[{ value: "hi-IN", label: "हिंदी", lang: "hi" }, { value: "en-IN", label: "English", lang: "en" }]} />
        <Toggle label={t("sentence.autoSpeak")} checked={prefs.auto} onChange={(v) => setPrefs({ ...prefs, auto: v })} />
      </div>
      {!anyVoice ? (
        <p className="mt-2 text-[11px] text-fg-3">{t("sentence.noVoices")}</p>
      ) : prefs.lang === "hi-IN" && !hiVoice ? (
        <p className="mt-2 text-[11px] text-fg-3" data-testid="no-hindi-voice">{t("sentence.noHindiVoice")}</p>
      ) : null}
    </Card>
  );
}
