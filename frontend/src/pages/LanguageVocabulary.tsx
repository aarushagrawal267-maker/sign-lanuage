// Vocabulary of a language pack (ISL): every dictionary concept, searchable in English and Hindi, with the Hindi
// review state, a landmark preview of the dictionary sign and a local Hindi correction form (docs/ISL_DESIGN.md 4.4, 8.1).
import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Flag, Search, Volume2, X } from "lucide-react";
import {
  IS_STATIC, api, type LanguageVocabEntry, type LanguageVocabRow, type SystemInfo,
} from "../api/client";
import { useT, type StringKey } from "../i18n";
import { useApi } from "../lib/hooks";
import { speak, speechAvailable } from "../lib/speech";
import { useSignLanguage } from "../state/language";
import { BilingualWord, HindiReviewMark } from "../components/BilingualWord";
import { SkeletonPreview } from "../components/SkeletonPreview";
import { Badge, Button, Card, Empty, ErrorText, Field, Kv, Notice, PageHeader, Segmented, Toggle, cx, inputCls } from "../components/ui";

const PAGE = 60;
type Kind = "" | "word" | "multiword" | "phrase" | "number";
// Entry-type badges in the interface language (they were raw English codes such as "multiword" in the Hindi interface).
const KIND_KEYS: Record<string, StringKey> = { word: "vocab.kindWord", multiword: "vocab.kindMulti", phrase: "vocab.kindPhrase", number: "vocab.kindNumber" };
function useKindLabel() {
  const { t } = useT();
  return (k: string) => (KIND_KEYS[k] ? t(KIND_KEYS[k]) : k);
}

type LexiconSummary = {
  available: boolean; concepts: number; entries: number; coverage?: number; reviewed?: number;
  by_source?: Record<string, number>; by_review?: Record<string, number>;
};

function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}

function hiOf(e: LanguageVocabEntry): string | null {
  return e.hindi?.hi ?? e.hindi?.hindi ?? null;
}

function HindiEditor({ lang, entry, onSaved, readOnly }: { lang: string; entry: LanguageVocabEntry; onSaved: () => void; readOnly: boolean }) {
  const { t } = useT();
  const current = hiOf(entry) ?? "";
  const [hi, setHi] = useState(current);
  const [reviewer, setReviewer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  useEffect(() => { setHi(current); setDone(null); setError(null); }, [current, entry.concept_id]);
  if (readOnly) return <p className="text-xs text-fg-3">{t("vocab.correctionDisabled")}</p>;
  const save = async (review: "reviewed" | "disputed", text = hi) => {
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.patch(`/api/languages/${encodeURIComponent(lang)}/lexicon/${encodeURIComponent(entry.concept_id)}`,
        { hi: text.trim(), hi_alt: entry.hindi?.hi_alt ?? [], review, reviewer: reviewer.trim() || null });
      setDone(review === "reviewed" ? t("vocab.savedReviewed") : t("vocab.savedDisputed"));
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-fg-3">{t("vocab.correctionHelp")}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label={t("vocab.hindiWord")}>
          <input lang="hi" className={cx(inputCls, "font-hi text-base")} value={hi} onChange={(e) => setHi(e.target.value)} maxLength={120} />
        </Field>
        <Field label={t("vocab.reviewer")}>
          <input className={inputCls} value={reviewer} onChange={(e) => setReviewer(e.target.value)} maxLength={80} placeholder={t("vocab.reviewerPlaceholder")} />
        </Field>
      </div>
      <div className="flex flex-wrap gap-2">
        {current && hi.trim() === current ? (
          <Button size="sm" variant="teal" icon={<Check className="size-3.5" />} loading={busy} onClick={() => save("reviewed")}>{t("vocab.confirm")}</Button>
        ) : (
          <Button size="sm" variant="primary" icon={<Check className="size-3.5" />} loading={busy} disabled={!hi.trim()} onClick={() => save("reviewed")}>{t("vocab.saveCorrection")}</Button>
        )}
        {current && <Button size="sm" variant="danger" icon={<Flag className="size-3.5" />} disabled={busy} onClick={() => save("disputed", current)}>{t("vocab.flagWrong")}</Button>}
      </div>
      {done && <p className="text-xs text-teal" role="status">{done}</p>}
      <ErrorText error={error} />
    </div>
  );
}

function DetailPanel({ lang, conceptId, onClose, readOnly }: { lang: string; conceptId: string; onClose: () => void; readOnly: boolean }) {
  const { t } = useT();
  const kindLabel = useKindLabel();
  const { data: e, error, reload } = useApi<LanguageVocabEntry>(`/api/languages/${encodeURIComponent(lang)}/vocabulary/${encodeURIComponent(conceptId)}`);
  // The panel opens above the word grid: move focus to it so keyboard and screen-reader users find it (WCAG 2.4.3).
  const regionRef = useRef<HTMLDivElement>(null);
  const loaded = !!e;
  useEffect(() => { if (loaded) regionRef.current?.focus({ preventScroll: false }); }, [loaded, conceptId]);
  if (error) return <Card><Notice tone="danger">{error}</Notice></Card>;
  if (!e) return <Card><p className="text-sm text-fg-3">{t("common.loading")}</p></Card>;
  const en = e.display?.en || e.headword;
  const hi = hiOf(e);
  const review = e.hindi?.review ?? null;
  const alts = e.hindi?.hi_alt ?? [];
  const trustedAlts = review === "reviewed" || review === "curated";
  return (
    <div ref={regionRef} tabIndex={-1} role="region" aria-label={`${t("vocab.detailTitle")}: ${en}`} className="rounded-2xl outline-none" data-testid="word-detail">
    <Card
      title={t("vocab.detailTitle")}
      subtitle={e.concept_id}
      actions={<button type="button" onClick={onClose} className="inline-flex size-9 items-center justify-center rounded-md text-fg-3 hover:bg-ink-800 hover:text-fg" aria-label={t("common.close")}><X className="size-5" aria-hidden /></button>}
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="flex min-w-0 flex-col gap-4">
          <BilingualWord en={en} hi={hi} review={review} size="xl" />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="ghost" icon={<Volume2 className="size-3.5" />} disabled={!speechAvailable("en-IN")} onClick={() => speak(en, "en-IN")} lang="en">English</Button>
            <Button size="sm" variant="ghost" icon={<Volume2 className="size-3.5" />} disabled={!hi || !speechAvailable("hi-IN")} onClick={() => hi && speak(hi, "hi-IN")}
              title={speechAvailable("hi-IN") ? undefined : t("sentence.noHindiVoice")} lang="hi">हिंदी</Button>
          </div>
          {hi && (
            <div className="text-xs text-fg-3">
              <div className="flex flex-wrap items-center gap-2">
                <HindiReviewMark review={review} />
                <span>{review === "machine" ? t("hi.machine.long") : review === "curated" ? t("hi.curated.long") : review === "reviewed" ? t("hi.reviewed.long") : review === "disputed" ? t("hi.disputed.long") : ""}</span>
              </div>
              {alts.length > 0 && (
                <div className="mt-1.5">
                  {trustedAlts ? t("vocab.alsoSaid") : t("vocab.otherSuggestions")}: <span lang="hi" className="font-hi text-fg-2">{alts.join(", ")}</span>
                </div>
              )}
            </div>
          )}
          <div>
            {e.in_classifier && <Badge tone="teal">{t("vocab.core")}</Badge>}
            {e.kind && e.kind !== "word" && <Badge className="ml-1">{kindLabel(e.kind)}</Badge>}
          </div>
          <div>
            <Kv k={t("vocab.category")} v={e.category ?? "—"} />
            {e.pos && <Kv k={t("vocab.pos")} v={e.pos} />}
            {e.synonyms && e.synonyms.length > 0 && <Kv k={t("vocab.synonyms")} v={e.synonyms.join(", ")} />}
            {e.fingerspell && <Kv k={t("vocab.fingerspell")} v={e.fingerspell} />}
          </div>
          <div>
            <div className="mb-1.5 text-xs font-medium text-fg-2">{t("vocab.variants", { n: e.variants?.length ?? 0 })}</div>
            <ul className="flex flex-col gap-1 text-xs text-fg-3">
              {(e.variants ?? []).map((v) => (
                <li key={v.variant_id} className="flex flex-wrap items-center gap-2">
                  <Badge>{v.source === "islrtc" ? "ISLRTC" : v.source === "include" ? "INCLUDE" : v.source}</Badge>
                  <span className="font-mono">{v.variant_id}</span>
                  {v.region && <Badge tone="accent">{v.region}</Badge>}
                  {v.clips && <span>· {t("vocab.clips", { n: v.clips.length })}</span>}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="mb-1.5 text-xs font-medium text-fg-2">{t("vocab.correctHindi")}</div>
            <HindiEditor lang={lang} entry={e} onSaved={reload} readOnly={readOnly} />
          </div>
        </div>
        <div>
          <div className="mb-1.5 text-xs font-medium text-fg-2">{t("vocab.preview")}</div>
          {e.preview?.frames?.length ? (
            <>
              <SkeletonPreview frames={e.preview.frames} width={e.preview.width} height={e.preview.height} label={t("vocab.previewLabel", { word: en })} />
              <p className="mt-1.5 text-[11px] text-fg-3">{t("vocab.previewNote")}</p>
            </>
          ) : (
            <p className="text-sm text-fg-3">{t("vocab.noPreview")}</p>
          )}
        </div>
      </div>
    </Card>
    </div>
  );
}

export function LanguageVocabulary() {
  const { t } = useT();
  const kindLabel = useKindLabel();
  const { id, info } = useSignLanguage();
  const [q, setQ] = useState("");
  const dq = useDebounced(q.trim());
  const [kind, setKind] = useState<Kind>("");
  const [coreOnly, setCoreOnly] = useState(false);
  const [items, setItems] = useState<LanguageVocabRow[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState<string | null>(null);
  const { data: lex } = useApi<LexiconSummary>(`/api/languages/${encodeURIComponent(id)}/lexicon`);
  const { data: sys } = useApi<SystemInfo>("/api/system");
  const readOnly = IS_STATIC || !!sys?.public_demo;

  const query = useMemo(() => {
    const p = new URLSearchParams({ limit: String(PAGE) });
    if (dq) p.set("q", dq);
    if (kind) p.set("kind", kind);
    if (coreOnly) p.set("in_classifier", "true");
    return p;
  }, [dq, kind, coreOnly]);

  // First page whenever the query changes; "show more" appends.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api.get<{ total: number; items: LanguageVocabRow[] }>(`/api/languages/${encodeURIComponent(id)}/vocabulary?${query}`)
      .then((r) => { if (!cancelled) { setItems(r.items); setTotal(r.total); setError(null); } })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [id, query]);

  const more = async () => {
    const p = new URLSearchParams(query);
    p.set("offset", String(items.length));
    setLoading(true);
    try {
      const r = await api.get<{ total: number; items: LanguageVocabRow[] }>(`/api/languages/${encodeURIComponent(id)}/vocabulary?${p}`);
      setItems((xs) => [...xs, ...r.items]);
      setTotal(r.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const by = lex?.by_review ?? {};
  const closeDetail = () => {
    const opener = sel;
    setSel(null);
    if (opener) window.requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-concept="${CSS.escape(opener)}"]`)?.focus());
  };
  return (
    <div className="fade-in">
      <PageHeader eyebrow={t("vocab.eyebrow", { lang: info?.sign_language ?? "" })} title={t("vocab.title")}
        description={t("vocab.descIsl", { n: (lex?.concepts ?? total ?? 0).toLocaleString() })} />

      <div className="mb-5 flex flex-col gap-3">
        <Notice>{t("vocab.noticeIsl")}</Notice>
        {lex?.available && lex.entries > 0 && (
          <p className="text-xs text-fg-3" data-testid="lexicon-summary">
            {t("vocab.hindiCoverage", {
              n: lex.entries.toLocaleString(), total: lex.concepts.toLocaleString(), curated: (by.curated ?? 0).toLocaleString(),
              machine: (by.machine ?? 0).toLocaleString(), reviewed: (lex.reviewed ?? by.reviewed ?? 0).toLocaleString(),
            })}
          </p>
        )}
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative w-full max-w-sm">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-3" aria-hidden />
          <input className={cx(inputCls, "h-11 pl-9 text-base")} placeholder={t("vocab.searchPlaceholder")} value={q} onChange={(e) => setQ(e.target.value)} aria-label={t("vocab.search")} />
        </div>
        <Segmented<Kind> ariaLabel={t("vocab.kind")} value={kind} onChange={setKind} options={[
          { value: "", label: t("vocab.kindAll") }, { value: "word", label: t("vocab.kindWord") }, { value: "multiword", label: t("vocab.kindMulti") },
          { value: "phrase", label: t("vocab.kindPhrase") }, { value: "number", label: t("vocab.kindNumber") }]} />
        <Toggle label={t("vocab.coreOnly")} checked={coreOnly} onChange={setCoreOnly} />
        <span className="text-xs text-fg-3" aria-live="polite">{total === null ? "" : t("vocab.count", { shown: items.length.toLocaleString(), total: total.toLocaleString() })}</span>
      </div>

      {error && <div className="mb-4"><Notice tone="danger">{error}</Notice></div>}
      {sel && <div className="mb-5"><DetailPanel lang={id} conceptId={sel} onClose={closeDetail} readOnly={readOnly} /></div>}
      {total === 0 && !loading && <Card><Empty title={t("vocab.noResults")}>{t("vocab.noResultsHelp")}</Empty></Card>}

      <ul className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4">
        {items.map((r) => (
          <li key={r.concept_id}>
            <button type="button" data-concept={r.concept_id} onClick={() => (r.concept_id === sel ? closeDetail() : setSel(r.concept_id))} aria-pressed={sel === r.concept_id}
              className={cx("flex w-full flex-col gap-1.5 rounded-xl border bg-ink-850 p-3 text-left transition-colors hover:border-ink-600",
                sel === r.concept_id ? "border-accent" : "border-ink-700")}>
              <BilingualWord en={r.text.en} hi={r.text.hi} review={r.hi_review} size="md" />
              <div className="flex flex-wrap items-center gap-1">
                {r.in_classifier && <Badge tone="teal">{t("vocab.coreShort")}</Badge>}
                {r.kind !== "word" && <Badge>{kindLabel(r.kind)}</Badge>}
                {r.regions.map((g) => <Badge key={g} tone="accent">{g}</Badge>)}
              </div>
            </button>
          </li>
        ))}
      </ul>
      {total !== null && items.length < total && (
        <div className="mt-5 flex justify-center">
          <Button variant="secondary" loading={loading} onClick={more}>{t("vocab.more")}</Button>
        </div>
      )}
    </div>
  );
}
