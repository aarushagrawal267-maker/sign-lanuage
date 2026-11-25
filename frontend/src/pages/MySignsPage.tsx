// "My signs" (docs/ISL_DESIGN.md section 7): record 3-5 takes of a personal or regional (e.g. Indore) sign, give it an
// English and Hindi word, check it, and share signs as a .signpack file. Also the per-user alphabet calibration that
// spelling mode needs (6.7). Only landmark points are stored, locally; never video.
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Camera, CameraOff, Check, Download, Link2, Pencil, Trash2, Upload, Video, X } from "lucide-react";
import {
  IS_STATIC, api, type AddSignResult, type AlphabetStatus, type LanguageVocabRow, type MySign, type SystemInfo,
} from "../api/client";
import { useT } from "../i18n";
import { useApi } from "../lib/hooks";
import { useSignLanguage } from "../state/language";
import { takeSummary, useLandmarkRecorder, type Take } from "../recognition/useLandmarkRecorder";
import { BilingualWord } from "../components/BilingualWord";
import { SkeletonPreview } from "../components/SkeletonPreview";
import { Badge, Button, Card, Empty, ErrorText, Field, Notice, PageHeader, Segmented, Toggle, cx, inputCls } from "../components/ui";

const MIN_TAKES = 3;
const MAX_TAKES = 5;
const DEFAULT_REGION = "Indore, Madhya Pradesh";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

// ------------------------------------------------------------------------------------------- camera stage
function CameraStage({ rec }: { rec: ReturnType<typeof useLandmarkRecorder> }) {
  const { t } = useT();
  return (
    <div className={cx("relative w-full overflow-hidden rounded-2xl border border-ink-700 bg-black", rec.camera !== "active" && "min-h-56")} style={{ aspectRatio: "16 / 9" }}>
      <video ref={rec.videoRef} playsInline muted className={cx("absolute inset-0 size-full -scale-x-100 object-contain", rec.camera !== "active" && "opacity-0")} aria-label={t("live.webcam")} />
      <canvas ref={rec.canvasRef} className="pointer-events-none absolute inset-0 size-full" aria-hidden />
      {rec.camera !== "active" ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
          {rec.camera === "error" ? <CameraOff className="size-6 text-danger" /> : <Camera className="size-6 text-accent" />}
          <p className="max-w-sm text-sm text-fg-3">{rec.cameraError ?? t("live.cameraNote")}</p>
          <Button variant="primary" icon={<Camera className="size-4" />} onClick={rec.start} disabled={rec.camera === "starting"}>{t("live.startCamera")}</Button>
        </div>
      ) : (
        <div className="absolute left-3 top-3 flex flex-wrap gap-2" role="status" aria-live="polite">
          <span data-testid="recorder-status" className={cx("rounded-full border px-3 py-1 text-xs font-medium backdrop-blur",
            rec.phase === "recording" ? "border-danger/60 bg-[#2d1316]/90 text-danger" : rec.phase === "waiting" ? "border-warn/50 bg-[#2b2210]/90 text-warn" : "border-ink-600 bg-ink-800/90 text-fg-2")}>
            {rec.mediapipe !== "ready" ? t("status.mediapipe_loading")
              // The running time is visual only: inside this live region it would be read out ten times a second.
              : rec.phase === "recording" ? <>{t("my.recordingNow")} <span aria-hidden>{(rec.elapsed / 1000).toFixed(1)} s</span></>
              : rec.phase === "waiting" ? t("my.raiseHands")
              : rec.body ? t("my.ready") : t("status.no_body")}
          </span>
          <span className={cx("rounded-full border px-2.5 py-1 text-[11px]", rec.hands ? "border-teal/50 text-teal" : "border-ink-600 text-fg-3")}>
            {rec.hands ? t("my.handsSeen") : t("my.noHands")}
          </span>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- new sign form
type Linked = { concept_id: string; en: string; hi: string | null };

function NewSign({ lang, rec, readOnly, onSaved }: { lang: string; rec: ReturnType<typeof useLandmarkRecorder>; readOnly: boolean; onSaved: () => void }) {
  const { t } = useT();
  const [takes, setTakes] = useState<Take[]>([]);
  const [en, setEn] = useState("");
  const [hi, setHi] = useState("");
  const [region, setRegion] = useState(DEFAULT_REGION);
  const [note, setNote] = useState("");
  const [linked, setLinked] = useState<Linked | null>(null);
  const [suggest, setSuggest] = useState<LanguageVocabRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [check, setCheck] = useState<AddSignResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [takeMsg, setTakeMsg] = useState<string | null>(null);

  // Dictionary suggestions for the English word (link the sign to an existing word = a regional variant of it).
  useEffect(() => {
    const q = en.trim();
    if (!q || linked || q.length < 2) { setSuggest([]); return; }
    let cancelled = false;
    const id = window.setTimeout(() => {
      api.get<{ items: LanguageVocabRow[] }>(`/api/languages/${encodeURIComponent(lang)}/vocabulary?q=${encodeURIComponent(q)}&limit=5`)
        .then((r) => { if (!cancelled) setSuggest(r.items); }).catch(() => {});
    }, 250);
    return () => { cancelled = true; window.clearTimeout(id); };
  }, [en, lang, linked]);

  const recordTake = async () => {
    setTakeMsg(null);
    setCheck(null);
    const take = await rec.record({ kind: "sign" });
    if (!take) return;
    if (take.frames.length < 6) { setTakeMsg(t("my.takeTooShort")); return; }
    setTakes((ts) => [...ts, take].slice(0, MAX_TAKES));
  };

  const reset = () => { setTakes([]); setEn(""); setHi(""); setNote(""); setLinked(null); setCheck(null); setError(null); };

  const submit = async (opts: { dryRun: boolean; conceptId?: string | null }) => {
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      const conceptId = opts.conceptId !== undefined ? opts.conceptId : linked?.concept_id ?? null;
      const body = {
        text_en: en.trim(), text_hi: hi.trim() || null, concept_id: conceptId, region: region.trim() || null, note: note.trim() || null,
        dry_run: opts.dryRun, takes,
      };
      const r = await api.post<AddSignResult>(`/api/languages/${encodeURIComponent(lang)}/my-signs`, body);
      if (opts.dryRun) {
        setCheck(r);
        if (r.accepted && r.conflicts.length === 0) await submit({ dryRun: false, conceptId });
      } else if (r.accepted && r.sign) {
        setSaved(t("my.saved", { word: r.sign.text_en, n: r.sign.n_takes }));
        reset();
        onSaved();
      } else {
        setCheck(r);
      }
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const canSave = !readOnly && takes.length >= MIN_TAKES && en.trim().length > 0 && rec.phase === "idle";
  // Say why "Check and save" is not available yet instead of leaving a silently disabled button (WCAG 3.3.2).
  const missing = readOnly ? [] : [
    ...(takes.length < MIN_TAKES ? [t("my.needTakes", { min: MIN_TAKES, n: takes.length })] : []),
    ...(en.trim() ? [] : [t("my.needWord")]),
  ];
  return (
    <Card title={t("my.newTitle")} subtitle={t("my.newSubtitle")}>
      <div className="flex flex-col gap-5">
        <ol className="list-decimal space-y-1 pl-5 text-sm text-fg-2">
          <li>{t("my.step1")}</li>
          <li>{t("my.step2", { min: MIN_TAKES, max: MAX_TAKES })}</li>
          <li>{t("my.step3")}</li>
        </ol>
        <CameraStage rec={rec} />

        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {rec.phase === "idle" ? (
              <Button variant="primary" icon={<Video className="size-4" />} onClick={recordTake}
                disabled={readOnly || rec.camera !== "active" || rec.mediapipe !== "ready" || takes.length >= MAX_TAKES}>
                {t("my.recordTake", { n: takes.length + 1 })}
              </Button>
            ) : (
              <Button variant="secondary" icon={<X className="size-4" />} onClick={rec.cancel}>{t("common.cancel")}</Button>
            )}
            <span className="text-xs text-fg-3" data-testid="take-count">{t("my.takesCount", { n: takes.length, min: MIN_TAKES, max: MAX_TAKES })}</span>
          </div>
          {takeMsg && <p className="mb-2 text-xs text-warn" role="alert">{takeMsg}</p>}
          {takes.length > 0 && (
            <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
              {takes.map((tk, i) => {
                const s = takeSummary(tk);
                const problems = check?.take_quality?.[i]?.problems ?? [];
                return (
                  <li key={i} className={cx("rounded-xl border bg-ink-900 p-2", problems.length ? "border-warn/60" : "border-ink-700")}>
                    <SkeletonPreview frames={tk.frames} width={tk.width} height={tk.height} mirror fps={20} label={t("my.takeLabel", { n: i + 1 })}
                      className="w-full rounded-lg bg-black" />
                    <div className="mt-1.5 flex items-center justify-between gap-1 text-[11px] text-fg-3">
                      <span>#{i + 1} · {(s.durationMs / 1000).toFixed(1)} s · {t("my.handsPct", { n: Math.round(s.handRate * 100) })}</span>
                      <button type="button" className="inline-flex size-8 shrink-0 items-center justify-center rounded-md hover:bg-ink-700 hover:text-danger" aria-label={t("my.removeTake", { n: i + 1 })}
                        onClick={() => { setTakes((ts) => ts.filter((_, j) => j !== i)); setCheck(null); }}>
                        <X className="size-4" aria-hidden />
                      </button>
                    </div>
                    {problems.length > 0 && <p className="mt-1 text-[11px] text-warn">{problems.join("; ")}</p>}
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <Field label={t("my.englishWord")}>
              <input className={inputCls} value={en} maxLength={80} onChange={(e) => { setEn(e.target.value); if (linked) setLinked(null); }} placeholder={t("my.englishPlaceholder")} />
            </Field>
            {linked ? (
              <div className="flex items-center gap-2 text-xs text-fg-2">
                <Link2 className="size-3.5 text-accent" aria-hidden />
                <span>{t("my.linkedTo")}</span>
                <BilingualWord en={linked.en} hi={linked.hi} size="sm" inline showMark={false} />
                <button type="button" className="text-fg-3 underline" onClick={() => setLinked(null)}>{t("my.unlink")}</button>
              </div>
            ) : suggest.length > 0 && (
              <div className="rounded-lg border border-ink-700 bg-ink-900 p-2">
                <div className="mb-1 text-[11px] text-fg-3">{t("my.linkHelp")}</div>
                <ul className="flex flex-wrap gap-1.5">
                  {suggest.map((s) => (
                    <li key={s.concept_id}>
                      <button type="button" className="rounded-md border border-ink-600 px-2 py-1 text-xs hover:border-accent"
                        onClick={() => { setLinked({ concept_id: s.concept_id, en: s.text.en, hi: s.text.hi }); setEn(s.text.en); if (!hi && s.text.hi) setHi(s.text.hi); }}>
                        {s.text.en}{s.text.hi && <span lang="hi" className="font-hi"> / {s.text.hi}</span>}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <Field label={t("my.hindiWord")} hint={t("my.hindiHint")}>
            <input lang="hi" className={cx(inputCls, "font-hi text-base")} value={hi} maxLength={80} onChange={(e) => setHi(e.target.value)} placeholder="उदाहरण: पानी" />
          </Field>
          <Field label={t("my.region")} hint={t("my.regionHint")}>
            <input className={inputCls} value={region} maxLength={80} onChange={(e) => setRegion(e.target.value)} />
          </Field>
          <Field label={t("my.note")}>
            <input className={inputCls} value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} placeholder={t("my.notePlaceholder")} />
          </Field>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button variant="teal" icon={<Check className="size-4" />} loading={busy} disabled={!canSave} onClick={() => submit({ dryRun: true })}
            aria-describedby={missing.length ? "save-needs" : undefined}>{t("my.checkSave")}</Button>
          {takes.length > 0 && <Button variant="ghost" onClick={reset} disabled={busy}>{t("my.startOver")}</Button>}
        </div>
        {missing.length > 0 && (
          <div id="save-needs" className="-mt-2 text-sm text-fg-2" data-testid="save-needs">
            <span className="font-medium">{t("my.saveNeeds")}</span>
            <ul className="mt-1 list-disc pl-5">{missing.map((m) => <li key={m}>{m}</li>)}</ul>
          </div>
        )}
        {saved && <p className="text-sm text-teal" role="status">{saved}</p>}
        <ErrorText error={error} />

        {check && (
          <div data-testid="sign-check" className={cx("rounded-xl border p-4 text-sm", check.accepted ? "border-ink-700 bg-ink-900" : "border-warn/50 bg-warn/[0.06]")}>
            <p className={check.accepted ? "text-fg" : "text-warn"}>{check.message}</p>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-3">
              {check.consistency !== null && <span>{t("my.consistency", { n: Math.round(check.consistency * 100) })}</span>}
              {check.self_test?.n ? <span>{t("my.selfTest", { hits: check.self_test.hits ?? 0, n: check.self_test.n })}</span> : null}
            </div>
            {check.accepted && check.conflicts.length > 0 && (
              <div className="mt-3 rounded-lg border border-warn/50 bg-warn/[0.06] p-3">
                <p className="text-fg">{t("my.conflict", { word: check.conflicts[0].name })}</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Button size="sm" variant="primary" loading={busy} onClick={() => submit({ dryRun: false })}>{t("my.saveAnyway")}</Button>
                  {!check.conflicts[0].concept_id.startsWith("mysign:") && (
                    <Button size="sm" variant="secondary" loading={busy} onClick={() => submit({ dryRun: false, conceptId: check.conflicts[0].concept_id })}>
                      {t("my.saveAsVariant", { word: check.conflicts[0].name })}
                    </Button>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}

// ------------------------------------------------------------------------------------------- sign list
function SignRow({ s, lang, readOnly, selected, onSelect, onChanged }: {
  s: MySign; lang: string; readOnly: boolean; selected: boolean; onSelect: (v: boolean) => void; onChanged: () => void;
}) {
  const { t } = useT();
  const [edit, setEdit] = useState(false);
  const [en, setEn] = useState(s.text_en);
  const [hi, setHi] = useState(s.text_hi ?? "");
  const [region, setRegion] = useState(s.region ?? "");
  const [confirmDel, setConfirmDel] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Buttons that swap places (Edit -> form, Delete -> confirm) move focus with them so keyboard users never lose
  // their place (WCAG 2.4.3).
  const focusNext = useRef<"edit" | "form" | "confirm" | "delete" | null>(null);
  const editBtn = useRef<HTMLButtonElement>(null);
  const firstInput = useRef<HTMLInputElement>(null);
  const confirmBtn = useRef<HTMLButtonElement>(null);
  const deleteBtn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const target = { edit: editBtn, form: firstInput, confirm: confirmBtn, delete: deleteBtn }[focusNext.current ?? "edit"];
    if (focusNext.current) target.current?.focus();
    focusNext.current = null;
  }, [edit, confirmDel]);
  const go = (next: "edit" | "form" | "confirm" | "delete", fn: () => void) => { focusNext.current = next; fn(); };
  const base = `/api/languages/${encodeURIComponent(lang)}/my-signs/${encodeURIComponent(s.sign_id)}`;
  const patch = async (body: Record<string, unknown>) => {
    setError(null);
    try { await api.patch(base, body); onChanged(); if (edit) go("edit", () => setEdit(false)); } catch (e) { setError(errText(e)); }
  };
  const del = async () => {
    setError(null);
    try { await api.del(base); onChanged(); } catch (e) { setError(errText(e)); }
  };
  const origin = s.origin === "recorded" ? t("my.originRecorded") : t("my.originImported", { title: s.origin.replace(/^imported:/, "") });
  return (
    <li className={cx("rounded-xl border bg-ink-900 p-3", s.enabled ? "border-ink-700" : "border-ink-700 opacity-60")} data-testid="my-sign">
      {edit ? (
        <div className="grid gap-2 sm:grid-cols-3">
          <input ref={firstInput} className={inputCls} value={en} onChange={(e) => setEn(e.target.value)} aria-label={t("my.englishWord")} maxLength={80} />
          <input lang="hi" className={cx(inputCls, "font-hi")} value={hi} onChange={(e) => setHi(e.target.value)} aria-label={t("my.hindiWord")} maxLength={80} />
          <input className={inputCls} value={region} onChange={(e) => setRegion(e.target.value)} aria-label={t("my.region")} maxLength={80} />
          <div className="flex gap-2 sm:col-span-3">
            <Button size="sm" variant="primary" disabled={!en.trim()} onClick={() => patch({ text_en: en.trim(), text_hi: hi.trim() || null, region: region.trim() || null })}>{t("common.save")}</Button>
            <Button size="sm" variant="ghost" onClick={() => go("edit", () => setEdit(false))}>{t("common.cancel")}</Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-3">
            <input type="checkbox" className="mt-1.5 size-4 accent-[var(--color-accent)]" checked={selected} onChange={(e) => onSelect(e.target.checked)}
              aria-label={t("my.includeInExport", { word: s.text_en })} />
            <div className="min-w-0">
              <BilingualWord en={s.text_en} hi={s.text_hi} review={s.text_hi ? "user" : null} size="md" inline showMark={false} />
              <div className="mt-1 flex flex-wrap gap-1.5 text-[11px] text-fg-3">
                {s.region && <Badge tone="accent">{s.region}</Badge>}
                <span>{t("my.nTakes", { n: s.n_takes })}</span>
                <span>· {origin}</span>
                {s.concept_id && !s.concept_id.startsWith("mysign:") && <span>· {t("my.variantOf", { id: s.concept_id })}</span>}
              </div>
            </div>
          </div>
          {!readOnly && (
            <div className="flex flex-wrap items-center gap-1.5">
              <Toggle label={t("my.enabled")} checked={s.enabled} onChange={(v) => patch({ enabled: v })} />
              <Button ref={editBtn} size="sm" variant="ghost" icon={<Pencil className="size-3.5" />} onClick={() => go("form", () => setEdit(true))}>{t("common.edit")}</Button>
              {confirmDel ? (
                <>
                  <Button ref={confirmBtn} size="sm" variant="danger" onClick={del}>{t("my.confirmDelete")}</Button>
                  <Button size="sm" variant="ghost" onClick={() => go("delete", () => setConfirmDel(false))}>{t("common.cancel")}</Button>
                </>
              ) : (
                <Button ref={deleteBtn} size="sm" variant="ghost" icon={<Trash2 className="size-3.5" />} onClick={() => go("confirm", () => setConfirmDel(true))}>{t("common.delete")}</Button>
              )}
            </div>
          )}
        </div>
      )}
      <ErrorText error={error} />
    </li>
  );
}

// ------------------------------------------------------------------------------------------- share
function SharePanel({ lang, signs, selected, readOnly, onImported }: { lang: string; signs: MySign[]; selected: string[]; readOnly: boolean; onImported: () => void }) {
  const { t } = useT();
  const [title, setTitle] = useState("My signs");
  const [region, setRegion] = useState(DEFAULT_REGION);
  const [author, setAuthor] = useState("");
  const [licence, setLicence] = useState("CC-BY-4.0");
  const [face, setFace] = useState(false);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"merge" | "replace">("merge");
  const [importMsg, setImportMsg] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const doExport = async () => {
    setBusy(true);
    setError(null);
    try {
      const p = new URLSearchParams({ title: title.trim() || "My signs", licence, include_face: String(face) });
      if (region.trim()) p.set("region", region.trim());
      if (author.trim()) p.set("author", author.trim());
      if (selected.length && selected.length < signs.length) p.set("ids", selected.join(","));
      const { blob, filename } = await api.blob(`/api/languages/${encodeURIComponent(lang)}/my-signs/export?${p}`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename ?? `${title.trim() || "signs"}.signpack`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const doImport = async () => {
    const f = fileRef.current?.files?.[0];
    if (!f) return;
    setBusy(true);
    setError(null);
    setImportMsg(null);
    try {
      const form = new FormData();
      form.append("file", f);
      form.append("mode", mode);
      const r = await api.upload<{ imported: number; skipped: number; title?: string; licence?: string }>(`/api/languages/${encodeURIComponent(lang)}/my-signs/import`, form);
      setImportMsg(t("my.imported", { n: r.imported, skipped: r.skipped, title: r.title ?? f.name, licence: r.licence ?? "?" }));
      if (fileRef.current) fileRef.current.value = "";
      onImported();
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const nExport = selected.length || signs.length;
  return (
    <Card title={t("my.shareTitle")} subtitle={t("my.shareSubtitle")}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-3">
          <div className="text-xs font-medium text-fg-2">{t("my.export")}</div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label={t("my.packTitle")}><input className={inputCls} value={title} maxLength={60} onChange={(e) => setTitle(e.target.value)} /></Field>
            <Field label={t("my.region")}><input className={inputCls} value={region} maxLength={80} onChange={(e) => setRegion(e.target.value)} /></Field>
            <Field label={t("my.author")}><input className={inputCls} value={author} maxLength={80} onChange={(e) => setAuthor(e.target.value)} placeholder={t("my.authorPlaceholder")} /></Field>
            <Field label={t("my.licence")}>
              <select className={inputCls} value={licence} onChange={(e) => setLicence(e.target.value)}>
                <option value="CC-BY-4.0">CC BY 4.0 ({t("my.licenceBy")})</option>
                <option value="CC-BY-SA-4.0">CC BY-SA 4.0</option>
                <option value="CC0-1.0">CC0 ({t("my.licenceCc0")})</option>
              </select>
            </Field>
          </div>
          <Toggle label={t("my.includeFace")} checked={face} onChange={setFace} />
          <label className="flex items-start gap-2.5 rounded-lg border border-warn/40 bg-warn/[0.05] p-3 text-sm text-fg-2">
            <input type="checkbox" className="mt-1 size-4 accent-[var(--color-accent)]" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>
              <span lang="en">This file contains body and hand movement points of the people who recorded these signs (no video). Share it only with their permission.</span>
              <span lang="hi" className="font-hi mt-1 block">इस फ़ाइल में संकेत रिकॉर्ड करने वाले लोगों के शरीर और हाथों की गति के बिंदु हैं (वीडियो नहीं)। इसे केवल उनकी अनुमति से साझा करें।</span>
            </span>
          </label>
          <div>
            <Button variant="primary" icon={<Download className="size-4" />} loading={busy} disabled={readOnly || !consent || signs.length === 0} onClick={doExport}>
              {t("my.exportButton", { n: nExport })}
            </Button>
          </div>
        </div>
        <div className="flex min-w-0 flex-col gap-3">
          <div className="text-xs font-medium text-fg-2">{t("my.import")}</div>
          <p className="text-xs text-fg-3">{t("my.importHelp")}</p>
          <input ref={fileRef} type="file" accept=".signpack,application/zip" className="w-full max-w-full text-sm text-fg-2 file:mr-3 file:rounded-lg file:border file:border-ink-700 file:bg-ink-800 file:px-3 file:py-1.5 file:text-fg"
            aria-label={t("my.chooseFile")} />
          <Field label={t("my.importMode")} group hint={mode === "replace" ? t("my.replaceWarning") : t("my.mergeHelp")}>
            <Segmented<"merge" | "replace"> ariaLabel={t("my.importMode")} value={mode} onChange={setMode}
              options={[{ value: "merge", label: t("my.merge") }, { value: "replace", label: t("my.replace") }]} />
          </Field>
          <div>
            <Button variant="secondary" icon={<Upload className="size-4" />} loading={busy} disabled={readOnly} onClick={doImport}>{t("my.importButton")}</Button>
          </div>
          {importMsg && <p className="text-sm text-teal" role="status">{importMsg}</p>}
        </div>
      </div>
      <div className="mt-3"><ErrorText error={error} /></div>
    </Card>
  );
}

// ------------------------------------------------------------------------------------------- alphabet
function AlphabetPanel({ lang, rec, readOnly, onChanged }: { lang: string; rec: ReturnType<typeof useLandmarkRecorder>; readOnly: boolean; onChanged: () => void }) {
  const { t } = useT();
  const path = `/api/languages/${encodeURIComponent(lang)}/my-signs/alphabet`;
  const { data, error, reload } = useApi<AlphabetStatus>(path);
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const confirmResetBtn = useRef<HTMLButtonElement>(null);
  useEffect(() => { if (confirmReset) confirmResetBtn.current?.focus(); }, [confirmReset]);

  const recordLetter = async (letter: string) => {
    setMsg(null);
    setErr(null);
    setBusy(letter);
    const take = await rec.record({ kind: "hold", holdMs: 1200 });
    if (!take) { setBusy(null); return; }
    try {
      const r = await api.post<{ letter: string; message: string; ready: boolean }>(path, { letter, take });
      setMsg(`${r.letter}: ${r.message}`);
      reload();
      onChanged();
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(null);
    }
  };
  const resetAll = async () => {
    setConfirmReset(false);
    try { await api.del(path); reload(); onChanged(); } catch (e) { setErr(errText(e)); }
  };

  const counts = data?.letters ?? {};
  const n = Object.keys(counts).length;
  return (
    <Card title={<span id="alphabet">{t("my.alphabetTitle")}</span>} subtitle={t("my.alphabetSubtitle")}
      actions={data ? (data.ready ? <Badge tone="teal">{t("my.alphabetReady")}</Badge> : <Badge tone="warn">{t("my.alphabetNeed", { n: data.min_letters })}</Badge>) : undefined}>
      {error && <Notice tone="danger">{error}</Notice>}
      <p className="mb-3 text-sm text-fg-2">{t("my.alphabetHelp")}</p>
      <ul className="grid grid-cols-7 gap-1.5 sm:grid-cols-9 md:grid-cols-13">
        {LETTERS.map((L) => (
          <li key={L}>
            <button type="button" disabled={readOnly || rec.camera !== "active" || rec.mediapipe !== "ready" || (busy !== null && busy !== L)}
              onClick={() => (busy === L ? rec.cancel() : recordLetter(L))}
              aria-label={busy === L ? t("my.letterRecording", { l: L }) : t("my.recordLetter", { l: L, n: counts[L] ?? 0 })}
              className={cx("flex h-12 w-full flex-col items-center justify-center rounded-lg border font-mono text-base font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-45",
                busy === L ? "border-danger text-danger" : counts[L] ? "border-teal/50 bg-teal-soft/40 text-teal" : "border-ink-600 bg-ink-800 text-fg hover:border-accent")}>
              {L}
              {counts[L] ? <span className="text-[11px] font-normal">{counts[L]}</span> : null}
            </button>
          </li>
        ))}
      </ul>
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-fg-3">
        <span>{t("my.lettersDone", { n, total: LETTERS.length })}</span>
        {data?.trained_model && <Badge tone="accent">{t("my.trainedLetters")}</Badge>}
        {n > 0 && !readOnly && (confirmReset ? (
          <>
            <Button ref={confirmResetBtn} size="sm" variant="danger" onClick={resetAll}>{t("my.confirmReset")}</Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmReset(false)}>{t("common.cancel")}</Button>
          </>
        ) : <Button size="sm" variant="ghost" onClick={() => setConfirmReset(true)}>{t("my.resetAlphabet")}</Button>)}
        {data?.ready && <Link to="/live" className="text-accent underline underline-offset-2">{t("my.trySpelling")}</Link>}
      </div>
      {msg && <p className="mt-2 text-sm text-teal" role="status">{msg}</p>}
      <ErrorText error={err} />
      <p className="mt-3 text-[11px] text-fg-3">{t("my.alphabetExperimental")}</p>
    </Card>
  );
}

// ------------------------------------------------------------------------------------------- page
export default function MySignsPage() {
  const { t } = useT();
  const sign = useSignLanguage();
  const rec = useLandmarkRecorder();
  const { data: sys } = useApi<SystemInfo>(IS_STATIC ? null : "/api/system");
  const readOnly = IS_STATIC || !!sys?.public_demo;
  const supported = !!sign.info && sign.info.capabilities.includes("my_signs");
  const lang = sign.id;
  const { data, error, reload } = useApi<{ signs: MySign[] }>(supported && !IS_STATIC ? `/api/languages/${encodeURIComponent(lang)}/my-signs` : null);
  const signs = (data?.signs ?? []).filter((s) => !s.deleted);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const selected = signs.filter((s) => !excluded.has(s.sign_id)).map((s) => s.sign_id);
  const isl = sign.languages.find((l) => l.capabilities.includes("my_signs"));

  return (
    <div className="fade-in">
      <PageHeader eyebrow={t("my.eyebrow")} title={t("my.title")} description={t("my.description")} />
      <div className="mb-5 flex flex-col gap-3">
        <Notice>{t("my.privacy")}</Notice>
        {IS_STATIC && <Notice tone="warn">{t("my.staticNote")}</Notice>}
        {sys?.public_demo && !IS_STATIC && <Notice tone="warn">{t("my.demoNote")}</Notice>}
      </div>

      {sign.loading ? (
        <p className="text-sm text-fg-3">{t("common.loading")}</p>
      ) : !supported ? (
        <Card>
          <Empty title={t("my.unsupported", { lang: sign.info?.sign_language ?? "" })}
            action={isl ? <Button variant="primary" onClick={() => sign.setId(isl.id)}>{t("my.switchTo", { lang: isl.display_name.en })}</Button> : undefined}>
            {t("my.unsupportedHelp")}
          </Empty>
        </Card>
      ) : (
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-[minmax(0,1fr)] gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
            <NewSign lang={lang} rec={rec} readOnly={readOnly} onSaved={reload} />
            <Card title={t("my.listTitle", { n: signs.length })} subtitle={t("my.listSubtitle")}>
              {error && <Notice tone="danger">{error}</Notice>}
              {signs.length === 0 ? (
                <p className="text-sm text-fg-3" data-testid="my-signs-empty">{t("my.listEmpty")}</p>
              ) : (
                <ul className="flex flex-col gap-2">
                  {signs.map((s) => (
                    <SignRow key={s.sign_id} s={s} lang={lang} readOnly={readOnly} selected={!excluded.has(s.sign_id)} onChanged={reload}
                      onSelect={(v) => setExcluded((ex) => { const n = new Set(ex); if (v) n.delete(s.sign_id); else n.add(s.sign_id); return n; })} />
                  ))}
                </ul>
              )}
            </Card>
          </div>
          <SharePanel lang={lang} signs={signs} selected={selected} readOnly={readOnly} onImported={reload} />
          <AlphabetPanel lang={lang} rec={rec} readOnly={readOnly} onChanged={sign.reload} />
        </div>
      )}
    </div>
  );
}
