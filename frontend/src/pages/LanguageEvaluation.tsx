// Evaluation results of a language pack (ISL results.json, docs/ISL_DESIGN.md 6.5, 9.2 and 15). A phase-2 pack serves
// the phase-2 evaluation report (backend/app/isl/evaluation/runners.py, schema 2): protocols E1a/E1b/E1c, E2, E3, E4,
// E5, E6, E7 and E8, each measured, not run (with its reason) or refused by a leakage guard. Every number is shown with
// its 95 % interval, or with the report's reason why there is none ("none: too few signer groups (4)"); the leakage
// guards are shown next to the numbers; smoke and partial-data reports carry a warning banner. Phase-1 files (schema 1:
// E1, E2, E3, E6) still render. No target numbers are shown, only measured ones. Text that comes from the report itself
// (labels, reasons) is English and marked lang="en"; the page's own words follow the interface language.
import type { ReactNode } from "react";
import { useT, type StringKey } from "../i18n";
import { useApi } from "../lib/hooks";
import { ms, num, pct } from "../lib/format";
import { useSignLanguage } from "../state/language";
import { Badge, Card, Empty, Kv, Notice, PageHeader, ScrollRegion } from "../components/ui";

export type Interval = [number, number] | null;
export type IntervalBlock = {
  metrics?: Record<string, number | null>;
  intervals?: Record<string, Interval>;
  interval_reasons?: Record<string, string>;
  interval_groups?: Record<string, number>;
};
type Policy = Record<string, number | string | Record<string, unknown>> & IntervalBlock;
type Cv = {
  folds?: number; top1_mean?: number; top1_sd?: number; pooled_n?: number; pooled_top1?: number;
  pooled_top1_interval?: Interval; pooled_top1_interval_reason?: string | null; pooled_groups?: number;
  label?: string; per_fold?: { split?: string; n?: number; top1?: number }[];
};
type E2Fold = { fold: number; run?: boolean; reason?: string; leakage?: string; held_out_concepts?: number; gallery_concepts?: number };
type ByK = {
  n?: number; n_groups?: number; top1_without_enrolment?: number; top1_with_enrolment?: number; gain?: number;
  interval_with?: Interval; interval_with_reason?: string | null; n_other_words?: number;
  user_sign_false_accept_rate?: number; interval_false_accept?: Interval; interval_false_accept_reason?: string | null;
};
type TrimEval = {
  synthetic?: { n?: number; within_300ms?: number; median_abs_err_ms?: number; late_frac?: number; early_frac?: number };
  include_active_kept?: { n?: number; mean?: number; frac_ge_90pct?: number };
  learned?: boolean;
};
export type Protocol = IntervalBlock & {
  label?: string; kind?: string; run?: boolean; reason?: string; leakage?: string;
  n?: number; n_classes?: number; n_groups?: number | null; gallery_concepts?: number;
  decision_note?: string; interval_note?: string; per_category_top1?: Record<string, number>; cv?: Cv;
  policies?: Record<string, Policy>; aurc?: number; fusion_fitted?: boolean;
  top1_by_source?: Record<string, { n?: number; top1?: number }>;
  restricted_gallery?: (IntervalBlock & { n?: number; n_groups?: number }) | null;
  folds?: E2Fold[]; pretraining_saw_gallery_unlabelled?: boolean | null;
  user_cls_mode?: string; by_k?: Record<string, ByK>;
  data?: { examples?: number; letters?: number; signers?: number; words?: number; per_source?: Record<string, number> } | null;
  load_average?: number; stages_p50_ms?: Record<string, number>;
  sha256?: string; unchanged_in_git?: boolean;
  trims?: number; method_mix?: Record<string, number>; low_confidence_frac?: number; trim_eval?: TrimEval;
};
type PretrainCheck = {
  excludes_eval_clips?: boolean | null; reason?: string; pretrain_clips?: number; eval_clips_and_duplicates?: number;
  overlap?: number; overlap_examples?: string[]; islrtc_clips_seen_unlabelled?: number; islrtc_clips?: number;
  sign_library?: { reliable?: boolean; drop_channels?: string[]; eval_content_matches?: number; matched_chunks_in_pretraining?: number };
};
export type LeakageGuards = {
  split?: string; classifier_split_checked?: boolean; classifier_refused?: string | null;
  dedup?: { dropped_train_extra?: number; dropped_train_islrtc?: number; dropped_eval_duplicates?: number } | null;
  pretraining_excludes_eval_clips?: boolean | null; pretraining_check?: PretrainCheck;
};
export type Results = {
  schema?: number; language: string; profile: string; created?: string; git_commit?: string | null; smoke?: boolean;
  partial_data?: boolean; label?: string;
  data?: Record<string, unknown> & {
    split?: string; split_label?: string; include?: Record<string, number>; islrtc?: number; include_concepts?: number;
    extra_sources?: Record<string, Record<string, number>>;
  };
  leakage_guards?: LeakageGuards;
  protocols?: Record<string, Protocol>;
};

/** Protocols in reading order. Phase 2 reports must have the ones in PHASE2; phase 1 had E1, E2, E3 and E6. */
const ORDER = ["E1a", "E1b", "E1c", "E1", "E2", "E2_seen", "E3", "E4", "E5", "E6", "E7", "E8"];
const PHASE2 = ["E1a", "E1b", "E1c", "E2", "E3", "E4", "E5", "E6", "E7", "E8"];
const TITLES: Record<string, StringKey> = {
  E1: "eval.p.E1", E1a: "eval.p.E1a", E1b: "eval.p.E1b", E1c: "eval.p.E1c", E2: "eval.p.E2", E2_seen: "eval.p.E2_seen",
  E3: "eval.p.E3", E4: "eval.p.E4", E5: "eval.p.E5", E6: "eval.p.E6", E7: "eval.p.E7", E8: "eval.p.E8",
};
/** The one number per protocol shown in the overview table (default: top-1, else the first metric). */
const MAIN: Record<string, string> = { E6: "p50_ms", E7: "passed", E8: "low_confidence_frac" };
const METRICS: Record<string, StringKey> = {
  top1: "eval.m.top1", top5: "eval.m.top5", top10: "eval.m.top10", macro_f1: "eval.m.macro_f1",
  coverage: "eval.m.coverage", accepted_precision: "eval.m.accepted_precision", not_sure_rate: "eval.m.not_sure_rate",
  abstain_rate: "eval.m.not_sure_rate", wrong_accept_rate: "eval.m.wrong_accept_rate", silent_error_rate: "eval.m.wrong_accept_rate",
  suggest_rate: "eval.m.suggest_rate", suggest_top5_hit: "eval.m.suggest_top5_hit", p50_ms: "eval.m.p50_ms", p95_ms: "eval.m.p95_ms",
  passed: "eval.m.passed", trims: "eval.m.trims", low_confidence_frac: "eval.m.low_confidence_frac", cap_frac: "eval.m.cap_frac",
  top1_with_enrolment: "eval.m.top1_with_enrolment", gain: "eval.m.gain", user_sign_false_accept_rate: "eval.m.user_sign_false_accept_rate",
};

export type Status = "measured" | "not_run" | "refused" | "missing";
export function statusOf(p: Protocol | undefined): Status {
  if (!p) return "missing";
  // E2 refused fold by fold (each encoder failed its leakage check) carries the refusals only in its folds
  const foldsRefused = !!p.folds?.length && p.folds.every((f) => f.run === false) && p.folds.some((f) => f.leakage);
  if (p.run === false) return p.leakage || foldsRefused || /^refused/i.test(p.reason ?? "") ? "refused" : "not_run";
  return "measured";
}
const reasonOf = (p: Protocol) => p.reason ?? (p.leakage ? `refused: ${p.leakage}` : undefined);

/** Text that comes from the report (English only). */
const En = ({ children }: { children: ReactNode }) => <span lang="en">{children}</span>;

function useFmt() {
  const { t } = useT();
  const metricName = (key: string) => {
    const m = /^(.*)_k(\d+)$/.exec(key);
    const base = m ? m[1] : key;
    const name = METRICS[base] ? t(METRICS[base]) : base.replace(/_/g, " ");
    return m ? `${name} (k = ${m[2]})` : name;
  };
  const value = (key: string, v: number | null | undefined) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (key.endsWith("_ms")) return ms(v);
    if (key === "trims" || key === "n") return num(v);
    if (key === "passed") return v >= 1 ? t("eval.yes") : t("eval.no");
    return pct(v);
  };
  return { t, metricName, value };
}

/** "6.1–9.9% (6 groups)", or "none: too few signer groups (4)", or "—" when the report has neither. */
function IntervalText({ ci, groups, reason, fmt }: { ci?: Interval; groups?: number | null; reason?: string | null; fmt: (v: number) => string }) {
  const { t } = useT();
  if (ci) {
    return (
      <span>
        {fmt(ci[0])}–{fmt(ci[1])}
        {groups !== undefined && groups !== null && <span className="text-fg-3"> ({t("eval.groupsN", { n: groups })})</span>}
      </span>
    );
  }
  if (reason) return <span className="text-fg-2">{t("eval.ciNone")} <En>{reason}</En></span>;
  return <span className="text-fg-3">—</span>;
}

/** Interval of metric `k`: from the block itself, else from `fallback` (E3's top-level metrics are its balanced
 * policy, whose wrong-accept and not-sure intervals live under policies.balanced). */
function BlockInterval({ block, k, fallback }: { block: IntervalBlock; k: string; fallback?: IntervalBlock }) {
  const { value } = useFmt();
  const own = block.intervals?.[k] || block.interval_reasons?.[k];
  const b = own || !fallback ? block : fallback;
  return <IntervalText ci={b.intervals?.[k]} groups={b.interval_groups?.[k]} reason={b.interval_reasons?.[k]} fmt={(v) => value(k, v)} />;
}

/** E4's headline metrics carry no intervals; its per-k rows do (top1_with_enrolment_k3 -> by_k.k3.interval_with). */
function e4Interval(p: Protocol, key: string): IntervalBlock | undefined {
  const m = /^(top1_with_enrolment|user_sign_false_accept_rate)_k(\d+)$/.exec(key);
  const r = m ? p.by_k?.[`k${m[2]}`] : undefined;
  if (!m || !r) return undefined;
  const fa = m[1] !== "top1_with_enrolment";
  const ci = fa ? r.interval_false_accept : r.interval_with;
  const why = fa ? r.interval_false_accept_reason : r.interval_with_reason;
  return { intervals: { [key]: ci ?? null }, interval_reasons: why ? { [key]: why } : {}, interval_groups: r.n_groups !== undefined ? { [key]: r.n_groups } : {} };
}

/** Where the intervals of a protocol's headline metrics are, besides the protocol itself. */
function intervalFallback(id: string, p: Protocol, key: string): IntervalBlock | undefined {
  if (id === "E3") return p.policies?.balanced;
  if (id === "E4") return e4Interval(p, key);
  return undefined;
}

function StatusBadge({ s }: { s: Status }) {
  const { t } = useT();
  const tone = s === "measured" ? "good" : s === "refused" ? "danger" : s === "not_run" ? "warn" : "neutral";
  return <Badge tone={tone}>{t(`eval.status.${s}` as StringKey)}</Badge>;
}

function GuardValue({ v }: { v: boolean | null | undefined }) {
  const { t } = useT();
  if (v === true) return <Badge tone="good">{t("eval.yes")}</Badge>;
  if (v === false) return <Badge tone="danger">{t("eval.no")}</Badge>;
  return <Badge tone="warn">{t("eval.unknown")}</Badge>;
}

const th = "py-2 pr-3 font-medium";
const td = "py-2 pr-3";

/** Metric | value | 95 % interval table of one protocol (or of a sub-result such as E2's smaller dictionary). */
function MetricTable({ caption, block, keys, fallback }: { caption: string; block: IntervalBlock; keys?: string[]; fallback?: (k: string) => IntervalBlock | undefined }) {
  const { t, metricName, value } = useFmt();
  const ks = keys ?? Object.keys(block.metrics ?? {});
  if (!ks.length) return null;
  const hasCi = Object.keys(block.intervals ?? {}).length > 0 || Object.keys(block.interval_reasons ?? {}).length > 0
    || ks.some((k) => fallback?.(k));
  return (
    <ScrollRegion label={caption} className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead className="text-xs text-fg-3">
          <tr>
            <th scope="col" className={th}>{t("eval.colMetric")}</th>
            <th scope="col" className={th}>{t("eval.colValue")}</th>
            {hasCi && <th scope="col" className={th}>{t("eval.colInterval")}</th>}
          </tr>
        </thead>
        <tbody>
          {ks.map((k) => (
            <tr key={k} className="border-t border-ink-700/60">
              <th scope="row" className={`${td} font-normal text-fg-2`}>{metricName(k)}</th>
              <td className={`${td} font-mono text-[13px] tabular text-fg`}>{value(k, block.metrics?.[k])}</td>
              {hasCi && <td className={`${td} font-mono text-[13px] tabular`}><BlockInterval block={block} k={k} fallback={fallback?.(k)} /></td>}
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollRegion>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mt-4">
      <h3 className="mb-1 text-xs font-semibold text-fg-2">{title}</h3>
      {children}
    </div>
  );
}

function SizeRows({ p }: { p: Protocol }) {
  const { t } = useT();
  return (
    <>
      {p.n !== undefined && <Kv k={t("eval.n")} v={num(p.n)} />}
      {p.n_classes !== undefined && <Kv k={t("eval.classes")} v={num(p.n_classes)} />}
      {p.gallery_concepts !== undefined && <Kv k={t("eval.galleryConcepts")} v={num(p.gallery_concepts)} />}
      {p.n_groups !== undefined && p.n_groups !== null && <Kv k={t("eval.groups")} v={p.n_groups} />}
    </>
  );
}

function ClosedSetDetails({ p }: { p: Protocol }) {
  const { t, value } = useFmt();
  const cv = p.cv;
  const cats = Object.entries(p.per_category_top1 ?? {});
  return (
    <>
      {p.decision_note && <p className="mt-3 text-xs text-fg-3"><span className="font-medium text-fg-2">{t("eval.decision")}:</span> <En>{p.decision_note}</En></p>}
      {p.interval_note && <p className="mt-1 text-xs text-fg-3"><En>{p.interval_note}</En></p>}
      {cv && (
        <Section title={t("eval.cvTitle")}>
          {cv.label && <p className="mb-1 text-xs text-fg-3"><En>{cv.label}</En></p>}
          {cv.folds !== undefined && <Kv k={t("eval.cvFolds")} v={cv.folds} />}
          {cv.top1_mean !== undefined && <Kv k={t("eval.cvMean")} v={`${pct(cv.top1_mean)} ± ${pct(cv.top1_sd)}`} />}
          {cv.pooled_top1 !== undefined && (
            <Kv k={t("eval.cvPooled", { n: cv.pooled_n ?? "?" })} v={<>{pct(cv.pooled_top1)} · <IntervalText ci={cv.pooled_top1_interval} groups={cv.pooled_groups} reason={cv.pooled_top1_interval_reason} fmt={(v) => value("top1", v)} /></>} />
          )}
          {(cv.per_fold ?? []).map((f, i) => (
            <Kv key={i} k={`${t("eval.fold")} ${i}`} v={`${pct(f.top1)} (n = ${num(f.n)})`} />
          ))}
        </Section>
      )}
      {cats.length > 0 && (
        <details className="mt-4 text-sm">
          <summary className="cursor-pointer text-xs font-semibold text-fg-2">{t("eval.perCategory")}</summary>
          <table className="mt-2 w-full text-left text-sm">
            <caption className="sr-only">{t("eval.perCategory")}</caption>
            <thead className="text-xs text-fg-3"><tr><th scope="col" className={th}>{t("eval.category")}</th><th scope="col" className={th}>{t("eval.m.top1")}</th></tr></thead>
            <tbody>
              {cats.map(([c, v]) => (
                <tr key={c} className="border-t border-ink-700/60"><th scope="row" className={`${td} font-normal text-fg-2`}><En>{c.replace(/_/g, " ")}</En></th><td className={`${td} font-mono text-[13px] tabular`}>{pct(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </>
  );
}

function E2Details({ p }: { p: Protocol }) {
  const { t } = useT();
  return (
    <>
      {p.pretraining_saw_gallery_unlabelled && <p className="mt-3 text-xs text-warn">{t("eval.galleryUnlabelled")}</p>}
      {p.restricted_gallery?.metrics && (
        <Section title={t("eval.restricted")}>
          <MetricTable caption={`E2 · ${t("eval.restricted")}`} block={p.restricted_gallery} />
        </Section>
      )}
      {(p.folds ?? []).length > 0 && (
        <Section title={t("eval.foldsTitle")}>
          <ul className="text-xs text-fg-3">
            {p.folds!.map((f) => (
              <li key={f.fold} className={`py-0.5 ${f.leakage ? "text-danger" : ""}`}>
                {f.run === false
                  ? <>{t("eval.foldNotRun", { k: f.fold })} <En>{f.leakage ?? f.reason ?? ""}</En></>
                  : t("eval.foldLine", { k: f.fold, held: f.held_out_concepts ?? "?", gallery: f.gallery_concepts ?? "?" })}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </>
  );
}

function E3Details({ p }: { p: Protocol }) {
  const { t, metricName, value } = useFmt();
  const pol = (k: string) => (k === "strict" || k === "balanced" || k === "lenient" ? t(`eval.policy.${k}` as StringKey) : k);
  const cols = ["coverage", "accepted_precision", "silent_error_rate", "suggest_rate", "suggest_top5_hit", "abstain_rate"];
  const ciKey: Record<string, string> = { silent_error_rate: "wrong_accept_rate", abstain_rate: "not_sure_rate" };
  return (
    <>
      {p.policies && (
        <Section title={t("eval.policyTitle")}>
          <ScrollRegion label={t("eval.policyTitle")} className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <caption className="sr-only">{t("eval.policyCaption")}</caption>
              <thead className="text-xs text-fg-3">
                <tr>
                  <th scope="col" className={th}>{t("eval.policy")}</th>
                  {cols.map((c) => <th key={c} scope="col" className={th}>{metricName(c)}</th>)}
                </tr>
              </thead>
              <tbody className="font-mono text-[13px] tabular">
                {Object.entries(p.policies).map(([k, v]) => (
                  <tr key={k} className="border-t border-ink-700/60">
                    <th scope="row" className={`${td} font-sans font-normal`}>{pol(k)}</th>
                    {cols.map((c) => {
                      const ck = ciKey[c];
                      const has = ck && (v.intervals?.[ck] || v.interval_reasons?.[ck]);
                      return (
                        <td key={c} className={`${td} align-top`}>
                          {value(c, v[c] as number)}{c === "coverage" && typeof v.n_accepted === "number" ? ` (${v.n_accepted})` : ""}
                          {has && <div className="text-[11px] text-fg-3"><IntervalText ci={v.intervals?.[ck]} groups={v.interval_groups?.[ck]} reason={v.interval_reasons?.[ck]} fmt={(x) => value(c, x)} /></div>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollRegion>
          <p className="mt-2 text-[11px] text-fg-3">{t("eval.e3Note", { n: p.n ?? "?" })}</p>
        </Section>
      )}
      <div className="mt-3">
        {p.aurc !== undefined && <Kv k={t("eval.aurc")} v={p.aurc.toFixed(3)} />}
        {p.fusion_fitted !== undefined && <Kv k={t("eval.fusionFitted")} v={<GuardValue v={p.fusion_fitted} />} />}
        {Object.entries(p.top1_by_source ?? {}).map(([s, v]) => (
          <Kv key={s} k={<>{t("eval.bySource")}: <En>{s}</En></>} v={`${pct(v.top1)} (n = ${num(v.n)})`} />
        ))}
      </div>
    </>
  );
}

function E4Details({ p }: { p: Protocol }) {
  const { t, value } = useFmt();
  const rows = Object.entries(p.by_k ?? {});
  return (
    <>
      {p.user_cls_mode && <div className="mt-3"><Kv k={t("eval.userMode")} v={<En>{p.user_cls_mode}</En>} /></div>}
      {rows.length > 0 && (
        <Section title={t("eval.byKTitle")}>
          <ScrollRegion label={t("eval.byKTitle")} className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <caption className="sr-only">{t("eval.byKTitle")}</caption>
              <thead className="text-xs text-fg-3">
                <tr>
                  <th scope="col" className={th}>{t("eval.k")}</th>
                  <th scope="col" className={th}>{t("eval.n")}</th>
                  <th scope="col" className={th}>{t("eval.colGroups")}</th>
                  <th scope="col" className={th}>{t("eval.without")}</th>
                  <th scope="col" className={th}>{t("eval.m.top1_with_enrolment")}</th>
                  <th scope="col" className={th}>{t("eval.m.gain")}</th>
                  <th scope="col" className={th}>{t("eval.m.user_sign_false_accept_rate")}</th>
                </tr>
              </thead>
              <tbody className="font-mono text-[13px] tabular">
                {rows.map(([k, v]) => (
                  <tr key={k} className="border-t border-ink-700/60 align-top">
                    <th scope="row" className={`${td} font-sans font-normal`}>{k.replace(/^k/, "")}</th>
                    <td className={td}>{num(v.n)}</td>
                    <td className={td}>{v.n_groups ?? "—"}</td>
                    <td className={td}>{pct(v.top1_without_enrolment)}</td>
                    <td className={td}>{pct(v.top1_with_enrolment)}<div className="text-[11px]"><IntervalText ci={v.interval_with} groups={v.n_groups} reason={v.interval_with_reason} fmt={(x) => value("top1", x)} /></div></td>
                    <td className={td}>{pct(v.gain)}</td>
                    <td className={td}>{pct(v.user_sign_false_accept_rate)}{v.n_other_words !== undefined && <span className="text-fg-3"> / {num(v.n_other_words)}</span>}<div className="text-[11px]"><IntervalText ci={v.interval_false_accept} groups={v.n_groups} reason={v.interval_false_accept_reason} fmt={(x) => value("top1", x)} /></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollRegion>
        </Section>
      )}
    </>
  );
}

function E5Details({ p }: { p: Protocol }) {
  const { t } = useT();
  const d = p.data;
  if (!d) return null;
  return (
    <Section title={t("eval.lettersData")}>
      {d.examples !== undefined && <Kv k={t("eval.examples")} v={num(d.examples)} />}
      {d.letters !== undefined && <Kv k={t("eval.letters")} v={num(d.letters)} />}
      {d.signers !== undefined && <Kv k={t("eval.signers")} v={num(d.signers)} />}
      {Object.entries(d.per_source ?? {}).map(([s, n]) => <Kv key={s} k={<>{t("eval.perSource")}: <En>{s}</En></>} v={num(n)} />)}
    </Section>
  );
}

function E6Details({ p }: { p: Protocol }) {
  const { t } = useT();
  return (
    <div className="mt-3">
      {p.load_average !== undefined && <Kv k={t("eval.loadAverage")} v={p.load_average.toFixed(1)} />}
      {Object.entries(p.stages_p50_ms ?? {}).map(([s, v]) => <Kv key={s} k={<>{t("eval.stage")}: <En>{s.replace(/_ms$/, "")}</En></>} v={ms(v)} />)}
    </div>
  );
}

function E7Details({ p }: { p: Protocol }) {
  const { t } = useT();
  return (
    <div className="mt-3">
      {p.unchanged_in_git !== undefined && <Kv k={t("eval.unchanged")} v={<GuardValue v={p.unchanged_in_git} />} />}
      {p.sha256 && <Kv k={t("eval.sha")} v={<span title={p.sha256}>{p.sha256.slice(0, 12)}…</span>} />}
    </div>
  );
}

function E8Details({ p }: { p: Protocol }) {
  const { t } = useT();
  const s = p.trim_eval?.synthetic;
  const a = p.trim_eval?.include_active_kept;
  const mix = Object.entries(p.method_mix ?? {}).sort((x, y) => y[1] - x[1]);
  return (
    <>
      {s && (
        <Section title={t("eval.synthetic")}>
          {s.n !== undefined && <Kv k={t("eval.n")} v={num(s.n)} />}
          <Kv k={t("eval.within300")} v={pct(s.within_300ms)} />
          <Kv k={t("eval.medianErr")} v={ms(s.median_abs_err_ms)} />
          <Kv k={t("eval.late")} v={pct(s.late_frac)} />
          <Kv k={t("eval.early")} v={pct(s.early_frac)} />
        </Section>
      )}
      {(a || p.trim_eval?.learned !== undefined) && (
        <div className="mt-2">
          {a && <Kv k={t("eval.activeKept")} v={pct(a.mean)} />}
          {p.trim_eval?.learned !== undefined && <Kv k={t("eval.learnedTrim")} v={<GuardValue v={p.trim_eval.learned} />} />}
        </div>
      )}
      {mix.length > 0 && (
        <details className="mt-4 text-sm">
          <summary className="cursor-pointer text-xs font-semibold text-fg-2">{t("eval.methodMix")}</summary>
          <div className="mt-1">{mix.map(([m, n]) => <Kv key={m} k={<En>{m.replace(/_/g, " ")}</En>} v={num(n)} />)}</div>
        </details>
      )}
    </>
  );
}

function ProtocolCard({ id, p }: { id: string; p: Protocol }) {
  const { t } = useT();
  const s = statusOf(p);
  const title = TITLES[id] ? t(TITLES[id]) : id;
  const wide = id === "E3" || id === "E4";
  return (
    <Card className={wide ? "min-w-0 lg:col-span-2" : "min-w-0"}
      title={<span className="flex flex-wrap items-center gap-1.5">{title} <Badge>{id}</Badge> <StatusBadge s={s} /></span>}
      subtitle={p.label ? <En>{p.label}</En> : undefined}>
      {s !== "measured" ? (
        <p className={`text-sm ${s === "refused" ? "text-danger" : "text-fg-2"}`}>
          <span className="font-medium">{t("eval.reason")}</span> <En>{reasonOf(p) ?? "—"}</En>
        </p>
      ) : (
        <>
          <MetricTable caption={t("eval.metricsCaption", { id })} block={p} fallback={(k) => intervalFallback(id, p, k)} />
          <div className="mt-3"><SizeRows p={p} /></div>
          {(id.startsWith("E1")) && <ClosedSetDetails p={p} />}
          {id === "E2" && <E2Details p={p} />}
          {id === "E3" && <E3Details p={p} />}
          {id === "E4" && <E4Details p={p} />}
          {id === "E6" && <E6Details p={p} />}
          {id === "E7" && <E7Details p={p} />}
          {id === "E8" && <E8Details p={p} />}
        </>
      )}
      {id === "E5" && <E5Details p={p} />}
    </Card>
  );
}

function mainKey(id: string, p: Protocol): string | undefined {
  const ks = Object.keys(p.metrics ?? {});
  const want = MAIN[id] ?? "top1";
  return ks.includes(want) ? want : ks[0];
}

function SummaryTable({ ids, P }: { ids: string[]; P: Record<string, Protocol> }) {
  const { t, metricName, value } = useFmt();
  return (
    <Card className="min-w-0" title={t("eval.summaryTitle")} subtitle={t("eval.summarySubtitle")}>
      <ScrollRegion label={t("eval.summaryTitle")} className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-sm">
          <caption className="sr-only">{t("eval.summaryCaption")}</caption>
          <thead className="text-xs text-fg-3">
            <tr>
              <th scope="col" className={th}>{t("eval.colProtocol")}</th>
              <th scope="col" className={th}>{t("eval.colStatus")}</th>
              <th scope="col" className={th}>{t("eval.colMain")}</th>
              <th scope="col" className={th}>{t("eval.colValue")}</th>
              <th scope="col" className={th}>{t("eval.colInterval")}</th>
              <th scope="col" className={th}>{t("eval.n")}</th>
              <th scope="col" className={th}>{t("eval.colGroups")}</th>
            </tr>
          </thead>
          <tbody>
            {ids.map((id) => {
              const p = P[id];
              const s = statusOf(p);
              const k = p && s === "measured" ? mainKey(id, p) : undefined;
              return (
                <tr key={id} className="border-t border-ink-700/60 align-top" data-protocol={id}>
                  <th scope="row" className={`${td} font-normal`}>
                    <span className="font-mono text-[13px] text-fg">{id}</span>{" "}
                    <span className="block text-xs text-fg-3">{TITLES[id] ? t(TITLES[id]) : ""}</span>
                  </th>
                  <td className={td}><StatusBadge s={s} /></td>
                  {s === "measured" ? (
                    <>
                      <td className={`${td} text-fg-2`}>{k ? metricName(k) : "—"}</td>
                      <td className={`${td} font-mono text-[13px] tabular text-fg`}>{k ? value(k, p!.metrics?.[k]) : "—"}</td>
                      <td className={`${td} font-mono text-[13px] tabular`}>{k ? <BlockInterval block={p!} k={k} fallback={intervalFallback(id, p!, k)} /> : "—"}</td>
                    </>
                  ) : (
                    <td colSpan={3} className={`${td} text-xs ${s === "refused" ? "text-danger" : "text-fg-2"}`}>
                      {s === "missing" ? t("eval.missingHelp") : <En>{reasonOf(p!) ?? "—"}</En>}
                    </td>
                  )}
                  <td className={`${td} font-mono text-[13px] tabular`}>{p?.n !== undefined && s === "measured" ? num(p.n) : "—"}</td>
                  <td className={`${td} font-mono text-[13px] tabular`}>{p?.n_groups ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </ScrollRegion>
    </Card>
  );
}

function LeakageCard({ lg }: { lg: LeakageGuards | undefined }) {
  const { t } = useT();
  if (!lg) {
    return <Card title={t("eval.guardsTitle")}><p className="text-sm text-fg-3">{t("eval.guardsMissing")}</p></Card>;
  }
  const pc = lg.pretraining_check ?? {};
  const ex = lg.pretraining_excludes_eval_clips ?? pc.excludes_eval_clips ?? null;
  const sl = pc.sign_library;
  const d = lg.dedup;
  return (
    <Card className="min-w-0" title={t("eval.guardsTitle")} subtitle={t("eval.guardsSubtitle")}>
      {ex === false && (
        <div className="mb-3">
          <Notice tone="danger" title={t("eval.leakTitle")}>
            {t("eval.leakBody")}{pc.overlap_examples?.length ? <> <En>{pc.overlap_examples.join(", ")}</En></> : null}
          </Notice>
        </div>
      )}
      <Kv k={t("eval.classifierChecked")} v={<GuardValue v={lg.classifier_split_checked} />} />
      <Kv k={t("eval.classifierRefused")} v={lg.classifier_refused ? <span className="text-danger"><En>{lg.classifier_refused}</En></span> : t("eval.no")} />
      <Kv k={t("eval.pretrainExcludes")} v={<GuardValue v={ex} />} />
      <p className="-mt-1 border-b border-ink-700/60 pb-2 text-xs text-fg-3">
        {pc.pretrain_clips !== undefined
          ? t("eval.pretrainComputed", { overlap: pc.overlap ?? "?", total: num(pc.pretrain_clips) })
          : <>{t("eval.reason")} <En>{pc.reason ?? "—"}</En></>}
      </p>
      {sl && (
        <Kv k={t("eval.signLibrary")} v={sl.reliable ? t("eval.reliable")
          : t("eval.notReliable", { channels: (sl.drop_channels ?? []).join(", ") || "—" })} />
      )}
      {pc.islrtc_clips_seen_unlabelled !== undefined && (
        <Kv k={t("eval.seenUnlabelled")} v={t("eval.ofN", { n: num(pc.islrtc_clips_seen_unlabelled), total: num(pc.islrtc_clips) })} />
      )}
      {d && <Kv k={t("eval.dedup")} v={`${num(d.dropped_train_extra)} / ${num(d.dropped_train_islrtc)} / ${num(d.dropped_eval_duplicates)}`} />}
    </Card>
  );
}

function DataCard({ data }: { data: NonNullable<Results["data"]> }) {
  const { t } = useT();
  const inc = data.include;
  const extra = Object.entries(data.extra_sources ?? {});
  return (
    <Card className="min-w-0" title={t("eval.dataTitle")}>
      {inc && <Kv k={t("eval.includeParts")} v={`${num(inc.train)} / ${num(inc.val)} / ${num(inc.test)}`} />}
      {data.include_concepts !== undefined && <Kv k={t("eval.concepts")} v={num(data.include_concepts)} />}
      {data.islrtc !== undefined && <Kv k={t("eval.islrtc")} v={num(data.islrtc)} />}
      {extra.map(([s, parts]) => (
        <Kv key={s} k={<>{t("eval.extraSource")}: <En>{s}</En></>} v={<En>{Object.entries(parts).map(([k, n]) => `${k} ${num(n)}`).join(" · ")}</En>} />
      ))}
    </Card>
  );
}

/** The whole report (exported for tests; the page below fetches it). */
export function IslReport({ r }: { r: Results }) {
  const { t } = useT();
  const P = r.protocols ?? {};
  const phase1 = (r.schema ?? 1) < 2;
  const ids = [...ORDER.filter((k) => k in P || (!phase1 && PHASE2.includes(k))), ...Object.keys(P).filter((k) => !ORDER.includes(k))];
  const partialOnly = !!r.partial_data && !r.smoke;
  return (
    <div className="flex min-w-0 flex-col gap-5">
      {r.smoke && (
        <Notice tone="warn" title={t("eval.smokeTitle")}>
          <span className="block">{t("eval.smokeBody")}</span>
          {r.label && <span className="mt-1 block font-medium text-fg"><En>{r.label}</En></span>}
          {r.partial_data && <span className="mt-1 block">{t("eval.partialAlso")}</span>}
        </Notice>
      )}
      {partialOnly && (
        <Notice tone="danger" title={t("eval.partialTitle")}>
          <span className="block">{t("eval.partialBody")}</span>
          {r.label && <span className="mt-1 block font-medium text-fg"><En>{r.label}</En></span>}
        </Notice>
      )}
      {phase1 && <Notice tone="info" title={t("eval.phase1Title")}>{t("eval.phase1Body")}</Notice>}
      <div className="flex flex-wrap items-center gap-2 text-xs text-fg-3">
        <Badge tone={r.profile === "commercial-safe" ? "teal" : "warn"}><En>{r.profile}</En></Badge>
        {r.smoke && <Badge tone="warn">SMOKE</Badge>}
        {r.partial_data && <Badge tone="danger">{t("eval.partialBadge")}</Badge>}
        {r.data?.split && <Badge>{t("eval.split")} <En>{r.data.split}</En></Badge>}
        {r.data?.split_label && <Badge><En>{String(r.data.split_label)}</En></Badge>}
        {r.created && <span>{r.created}</span>}
        {r.git_commit && <span className="font-mono">commit {r.git_commit}</span>}
      </div>
      <SummaryTable ids={ids} P={P} />
      <div className="grid gap-5 lg:grid-cols-2">
        <LeakageCard lg={r.leakage_guards} />
        {r.data && <DataCard data={r.data} />}
      </div>
      <h2 className="sr-only">{t("eval.detailsHeading")}</h2>
      <div className="grid gap-5 lg:grid-cols-2">
        {ids.filter((id) => P[id]).map((id) => <ProtocolCard key={id} id={id} p={P[id]} />)}
      </div>
    </div>
  );
}

export function LanguageEvaluation() {
  const { t } = useT();
  const { id, info } = useSignLanguage();
  const { data: r, error } = useApi<Results>(`/api/languages/${encodeURIComponent(id)}/results`);
  return (
    <div className="fade-in">
      <PageHeader eyebrow={t("eval.eyebrow", { lang: info?.sign_language ?? id })} title={t("eval.title")} description={t("eval.description")} />
      {error ? (
        <Card><Empty title={t("eval.noResults")}>{error.includes("404") || error.includes("no evaluation") ? t("eval.noResultsHelp") : <En>{error}</En>}</Empty></Card>
      ) : !r ? (
        <p className="text-sm text-fg-3">{t("common.loading")}</p>
      ) : (
        <IslReport r={r} />
      )}
    </div>
  );
}
