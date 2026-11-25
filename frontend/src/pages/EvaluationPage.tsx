import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CheckCircle2, Download, Lock, ShieldAlert } from "lucide-react";
import { api, type ExperimentDetail, type ExperimentSummary, type Metrics } from "../api/client";
import { useApi } from "../lib/hooks";
import { dateTime, downloadCsv, ms, pct } from "../lib/format";
import { Badge, Button, Card, Empty, ErrorText, Notice, PageHeader, Segmented, Stat, inputCls } from "../components/ui";
import { useSignLanguage } from "../state/language";
import { LanguageEvaluation } from "./LanguageEvaluation";
import { ConfusionMatrix } from "../components/charts/ConfusionMatrix";
import { LineChart } from "../components/charts/LineChart";
import { ColumnChart } from "../components/charts/Bars";

type SortKey = "class_index" | "f1" | "precision" | "recall" | "support";

function PerClassTable({ m }: { m: Metrics }) {
  const [sort, setSort] = useState<SortKey>("f1");
  const [asc, setAsc] = useState(true);
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const r = m.per_class.filter((c) => c.support > 0 && c.gloss.toLowerCase().includes(q.toLowerCase()));
    r.sort((a, b) => (asc ? 1 : -1) * ((a[sort] as number) - (b[sort] as number)));
    return r;
  }, [m, sort, asc, q]);
  const head = (k: SortKey, label: string) => (
    <th className="py-2 pr-3 font-medium" aria-sort={sort === k ? (asc ? "ascending" : "descending") : "none"}>
      <button className="hover:text-fg" onClick={() => (sort === k ? setAsc(!asc) : (setSort(k), setAsc(k === "class_index")))}>
        {label}{sort === k ? (asc ? " ↑" : " ↓") : ""}
      </button>
    </th>
  );
  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <input className={inputCls + " max-w-60"} placeholder="Filter glosses…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter glosses" />
        <Button size="sm" variant="ghost" icon={<Download className="size-3.5" />} onClick={() => downloadCsv(`per_class_${m.partition ?? "eval"}.csv`, m.per_class)}>CSV</Button>
      </div>
      <div className="max-h-[480px] overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-ink-850"><tr className="border-b border-ink-700 text-left text-xs text-fg-3">
            {head("class_index", "Class")}<th className="py-2 pr-3 font-medium">Gloss</th>{head("precision", "Precision")}{head("recall", "Recall")}{head("f1", "F1")}{head("support", "Support")}
          </tr></thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.class_index} className="border-b border-ink-700/40 font-mono text-xs">
                <td className="py-1.5 pr-3 text-fg-3">{c.class_index}</td>
                <td className="py-1.5 pr-3 text-fg">{c.gloss.toUpperCase()}</td>
                <td className="py-1.5 pr-3">{pct(c.precision, 0)}</td>
                <td className="py-1.5 pr-3">{pct(c.recall, 0)}</td>
                <td className="py-1.5 pr-3"><span className="inline-flex items-center gap-2"><span className="h-1.5 w-12 overflow-hidden rounded-full bg-ink-700"><span className="block h-full bg-[var(--color-series-1)]" style={{ width: `${c.f1 * 100}%` }} /></span>{pct(c.f1, 0)}</span></td>
                <td className="py-1.5 pr-3 text-fg-3">{c.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-fg-3">{m.per_class.filter((c) => c.support === 0).length} classes have no samples in this partition and cannot be scored. Weak classes are shown, not hidden — sort by F1 ascending to see them first.</p>
    </div>
  );
}

function SummaryMetricsView({ m, exp }: { m: Metrics; exp: ExperimentDetail }) {
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Top-1 accuracy" value={pct(m.top1)} tone="accent" sub={`${m.n} sequences`} />
        <Stat label="Top-5 accuracy" value={pct(m.top5)} />
        <Stat label="Macro F1" value={pct(m.macro_f1)} sub={m.classes_evaluated ? `over ${m.classes_evaluated} classes present` : undefined} />
        <Stat label="Signers evaluated" value={exp.split?.n_signers?.[m.partition === "test" ? "test" : "val"] ?? "—"}
          sub={exp.split?.signer_independent ? "unseen during training" : undefined} tone="teal" />
        <Stat label="Live threshold" value={m.recommended_threshold !== undefined ? pct(m.recommended_threshold, 0) : "—"} />
      </div>
      <Notice tone="info" title="Pretrained model — summary metrics">
        These numbers were recorded when the bundled model was trained (see <span className="font-mono">models/</span>). The per-class table,
        confusion matrix and calibration curve need the WLASL landmark files, which are not redistributed. Download the dataset and
        train a model in the Model Laboratory to see the full breakdown.
      </Notice>
    </div>
  );
}

function MetricsView({ m, exp }: { m: Metrics; exp: ExperimentDetail }) {
  if (!m.per_class || !m.confusion_matrix) return <SummaryMetricsView m={m} exp={exp} />;
  const labels = m.per_class.map((c) => c.gloss.toUpperCase());
  const signers = Object.entries(m.per_signer).sort((a, b) => b[1].n - a[1].n);
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Top-1 accuracy" value={pct(m.top1)} tone="accent" sub={`${m.n} sequences`} />
        <Stat label="Top-5 accuracy" value={pct(m.top5)} />
        <Stat label="Macro F1" value={pct(m.macro_f1)} sub={`over ${m.classes_evaluated} classes present`} />
        <Stat label="Model inference" value={ms(m.latency?.mean_ms)} sub={m.latency ? `p95 ${ms(m.latency.p95_ms)} · ${m.latency.device}, batch 1` : undefined} />
        <Stat label="Signers evaluated" value={signers.length} sub={exp.split?.signer_independent ? "unseen during training" : "overlap with training signers"} tone={exp.split?.signer_independent ? "teal" : "warn"} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr]">
        <Card title="Confusion matrix" subtitle="Rows: true gloss · columns: predicted gloss">
          <ConfusionMatrix matrix={m.confusion_matrix} labels={labels} />
        </Card>
        <Card title="Per-class performance"><PerClassTable m={m} /></Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <Card title="Most frequent confusions">
          {m.top_confusions.length ? (
            <ul className="flex flex-col gap-1.5 text-sm">
              {m.top_confusions.map((c, i) => (
                <li key={i} className="flex items-center justify-between border-b border-ink-700/50 py-1">
                  <span className="font-mono text-xs"><span className="text-fg">{c.true.toUpperCase()}</span> <span className="text-fg-3">→</span> <span className="text-fg-2">{c.pred.toUpperCase()}</span></span>
                  <span className="font-mono text-xs text-fg-3">{c.count}×</span>
                </li>
              ))}
            </ul>
          ) : <p className="text-sm text-fg-3">No confusions.</p>}
        </Card>
        <Card title="Accuracy vs coverage" subtitle="Effect of the rejection threshold (confidence ≥ x)">
          <LineChart ariaLabel="Accuracy of accepted predictions and coverage as a function of confidence threshold" height={190}
            x={m.confidence_curve.map((r) => r.threshold)} xLabel="threshold" yDomain={[0, 1]} xFormat={(v) => v.toFixed(1)} yFormat={(v) => `${Math.round(v * 100)}%`}
            series={[{ key: "a", label: "accuracy of accepted", color: "var(--color-series-1)", values: m.confidence_curve.map((r) => r.accuracy) },
              { key: "c", label: "coverage", color: "var(--color-series-2)", values: m.confidence_curve.map((r) => r.coverage) }]} />
          {m.recommended_threshold !== undefined && <p className="mt-2 text-xs text-fg-3">Validation-derived live threshold: <span className="font-mono text-fg">{pct(m.recommended_threshold, 0)}</span></p>}
        </Card>
        <Card title="Per-signer accuracy" subtitle="Top-1 for each evaluated signer">
          <ColumnChart ariaLabel="Top-1 accuracy per signer" height={150} valueFormat={(v) => pct(v)}
            bars={signers.map(([sid, v]) => ({ label: `signer ${sid}`, value: v.top1, hint: `${v.n} sequences` }))} />
          <p className="mt-2 text-xs text-fg-3">Spread across signers indicates how much performance depends on who is signing.</p>
        </Card>
      </div>
    </div>
  );
}

function AslEvaluation() {
  const [params, setParams] = useSearchParams();
  const exps = useApi<ExperimentSummary[]>("/api/experiments");
  const completed = (exps.data ?? []).filter((e) => e.status === "completed");
  const defaultId = completed.find((e) => e.is_active)?.id ?? completed[0]?.id;
  const id = Number(params.get("exp")) || defaultId || null;
  const detail = useApi<ExperimentDetail>(id ? `/api/experiments/${id}` : null);
  const [part, setPart] = useState<"val" | "test">("val");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const exp = detail.data;
  const vm = exp?.val_metrics && "top1" in exp.val_metrics ? (exp.val_metrics as Metrics) : null;
  const tm = exp?.test_metrics && "top1" in exp.test_metrics ? (exp.test_metrics as Metrics) : null;

  const runTest = async () => {
    if (!id) return;
    if (!window.confirm("Evaluate on the held-out TEST partition?\n\nDo this only after you have finished choosing models and settings using validation results. Using test results to tune the model invalidates them.")) return;
    setBusy(true);
    setErr(null);
    try {
      await api.post(`/api/experiments/${id}/evaluate-test`);
      detail.reload();
      setPart("test");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fade-in">
      <PageHeader eyebrow="Metrics from stored experiment records" title="Model Evaluation"
        description="Validation results guide model selection. Held-out test results are computed only on explicit request and are reported separately."
        actions={completed.length > 0 && (
          <select className={inputCls + " w-72"} value={id ?? ""} onChange={(e) => setParams({ exp: e.target.value })} aria-label="Experiment">
            {completed.map((e) => <option key={e.id} value={e.id}>#{e.id} · {e.name}{e.is_active ? " (active)" : ""}</option>)}
          </select>
        )} />

      {!completed.length ? (
        <Card><Empty title="No completed experiments">Train a model in the Model Laboratory. Evaluation metrics are computed from real predictions on held-out signers.</Empty></Card>
      ) : !exp ? (
        <p className="text-sm text-fg-3">Loading…</p>
      ) : (
        <div className="flex flex-col gap-5">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="accent">{exp.architecture.toUpperCase()}</Badge>
            <Badge>split: {exp.split_name}</Badge>
            {exp.split?.signer_independent ? <Badge tone="good"><CheckCircle2 className="size-3" />signer-independent evaluation</Badge>
              : <Badge tone="warn"><ShieldAlert className="size-3" />signers overlap with training</Badge>}
            {exp.is_synthetic && <Badge tone="warn">synthetic data — not ASL performance</Badge>}
            <span className="text-xs text-fg-3">dataset {exp.dataset_version ?? "—"} · finished {dateTime(exp.finished_at)}</span>
          </div>
          {!exp.split?.signer_independent && (
            <Notice tone="warn" title="Not a signer-independent result">The official WLASL split shares most signers between train and test, which inflates accuracy. Use a signer-grouped split for the headline result.</Notice>
          )}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Segmented ariaLabel="Partition" value={part} onChange={setPart} options={[{ value: "val", label: "Validation" }, { value: "test", label: "Held-out test" }]} />
            {part === "test" && !tm && <Button variant="secondary" icon={<Lock className="size-4" />} loading={busy} onClick={runTest}>Run held-out test evaluation</Button>}
            {part === "test" && tm && <span className="text-xs text-fg-3">Test evaluated {dateTime(tm.evaluated_at)}</span>}
          </div>
          <ErrorText error={err} />
          {part === "val" ? (vm ? <MetricsView m={vm} exp={exp} /> : <Empty title="No validation metrics recorded" />)
            : tm ? <MetricsView m={tm} exp={exp} />
              : <Card><Empty icon={<Lock className="size-6" />} title="Test set not evaluated yet">The test partition stays untouched during training and model selection. Run it once you have settled on a model.</Empty></Card>}
        </div>
      )}
    </div>
  );
}

export default function EvaluationPage() {
  const { isAsl, loading, id } = useSignLanguage();
  if (loading) return <p className="text-sm text-fg-3">Loading…</p>;
  return isAsl ? <AslEvaluation /> : <LanguageEvaluation key={id} />;
}
