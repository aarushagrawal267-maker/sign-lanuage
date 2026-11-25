import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { FlaskConical, Play, Square, Zap } from "lucide-react";
import { api, type ExperimentDetail, type ExperimentSummary, type SplitInfo, type DatasetSummary } from "../api/client";
import { useApi } from "../lib/hooks";
import { num, pct } from "../lib/format";
import { Badge, Button, Card, Empty, ErrorText, Field, Kv, Notice, PageHeader, Progress, ScrollRegion, Segmented, Toggle, cx, inputCls } from "../components/ui";
import { LineChart } from "../components/charts/LineChart";

type Form = {
  name: string;
  split_name: string;
  architecture: "gru" | "lstm" | "tcn";
  hidden: number;
  layers: number;
  bidirectional: boolean;
  dropout: number;
  lr: number;
  weight_decay: number;
  batch_size: number;
  epochs: number;
  patience: number;
  label_smoothing: number;
  class_weighting: "none" | "balanced";
  augment: boolean;
  seed: number;
  device: "auto" | "cpu" | "cuda";
  features: { seq_len: number; use_face: boolean; use_velocity: boolean; max_gap: number };
};

const DEFAULT: Form = {
  name: "", split_name: "signer-holdout", architecture: "gru", hidden: 192, layers: 2, bidirectional: true, dropout: 0.3,
  lr: 0.001, weight_decay: 0.01, batch_size: 32, epochs: 80, patience: 20, label_smoothing: 0.1, class_weighting: "none",
  augment: true, seed: 42, device: "auto", features: { seq_len: 48, use_face: true, use_velocity: true, max_gap: 5 },
};

const PRESETS: { key: string; label: string; hint: string; patch: Partial<Form> }[] = [
  { key: "quick", label: "Quick check", hint: "small model, 20 epochs — minutes on CPU", patch: { hidden: 96, layers: 1, epochs: 20, patience: 10 } },
  { key: "standard", label: "Standard", hint: "recommended starting point", patch: { hidden: 192, layers: 2, epochs: 80, patience: 20 } },
  { key: "large", label: "Large", hint: "slower; benefits from a GPU", patch: { hidden: 256, layers: 3, epochs: 150, patience: 30 } },
];

function NumField({ label, value, onChange, step = 1, min, max, hint }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <input type="number" className={inputCls} value={value} step={step} min={min} max={max} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}

export function TrainingCurves({ exp }: { exp: ExperimentDetail }) {
  const h = exp.history ?? [];
  const x = h.map((r) => r.epoch);
  if (!h.length) return <Empty title="Waiting for the first epoch">Curves appear as soon as the training loop reports an epoch.</Empty>;
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div>
        <div className="mb-2 text-xs font-medium text-fg-2">Loss</div>
        <LineChart ariaLabel="Training and validation loss per epoch" x={x} xLabel="epoch"
          series={[{ key: "tl", label: "train", color: "var(--color-series-1)", values: h.map((r) => r.train_loss) },
            { key: "vl", label: "validation", color: "var(--color-series-2)", values: h.map((r) => r.val_loss) }]} />
      </div>
      <div>
        <div className="mb-2 text-xs font-medium text-fg-2">Accuracy & macro F1</div>
        <LineChart ariaLabel="Training accuracy, validation accuracy and validation macro F1 per epoch" x={x} xLabel="epoch" yDomain={[0, 1]} yFormat={(v) => `${Math.round(v * 100)}%`}
          series={[{ key: "ta", label: "train acc", color: "var(--color-series-1)", values: h.map((r) => r.train_acc) },
            { key: "va", label: "val acc", color: "var(--color-series-2)", values: h.map((r) => r.val_acc) },
            { key: "vf", label: "val macro F1", color: "var(--color-series-3)", values: h.map((r) => r.val_macro_f1), dashed: true }]} />
      </div>
    </div>
  );
}

function RunMonitor({ id, onDone }: { id: number; onDone: () => void }) {
  const { data: exp } = useApi<ExperimentDetail>(`/api/experiments/${id}`, 1500);
  const running = exp && (exp.status === "running" || exp.status === "queued");
  useEffect(() => {
    if (exp && !running) onDone();
  }, [exp, running, onDone]);
  if (!exp) return null;
  const last = exp.history[exp.history.length - 1];
  const sec = exp.history.reduce((a, b) => a + b.seconds, 0);
  return (
    <Card title={<span className="flex items-center gap-2">Run #{exp.id} · {exp.name} {exp.is_synthetic && <Badge tone="warn">synthetic data</Badge>}</span>}
      subtitle={`${exp.architecture.toUpperCase()} · split ${exp.split_name} · ${exp.device ?? "…"} · ${num(exp.n_params)} parameters`}
      actions={running ? <Button size="sm" variant="danger" icon={<Square className="size-3.5" />} onClick={() => api.post(`/api/experiments/${id}/cancel`)}>Stop</Button>
        : <Badge tone={exp.status === "completed" ? "good" : exp.status === "failed" ? "danger" : "neutral"}>{exp.status}</Badge>}>
      {running && (
        <div className="mb-4">
          <div className="mb-1.5 flex justify-between text-xs text-fg-3"><span>epoch {exp.history.length} / {String(exp.config.epochs)} (early stopping may end sooner)</span><span>{sec.toFixed(0)} s elapsed</span></div>
          <Progress value={exp.history.length / Number(exp.config.epochs || 1)} label="Training progress (epochs)" />
        </div>
      )}
      {exp.error && <div className="mb-3"><Notice tone="danger" title="Training failed">{exp.error}</Notice></div>}
      <TrainingCurves exp={exp} />
      {last && (
        <div className="mt-4 grid grid-cols-2 gap-x-6 md:grid-cols-4">
          <Kv k="val accuracy" v={pct(last.val_acc)} />
          <Kv k="val macro F1" v={pct(last.val_macro_f1)} />
          <Kv k="val loss" v={last.val_loss.toFixed(3)} />
          <Kv k="sec / epoch" v={last.seconds.toFixed(1)} />
        </div>
      )}
      {exp.status === "completed" && (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <span className="text-sm text-fg-2">Best epoch {exp.val_metrics && "best_epoch" in exp.val_metrics ? exp.val_metrics.best_epoch : "—"} · validation top-1 {pct(exp.val?.top1)}</span>
          <Button size="sm" variant="primary" icon={<Zap className="size-3.5" />} onClick={() => api.post(`/api/experiments/${id}/activate`).then(onDone)} disabled={exp.is_active}>{exp.is_active ? "Active model" : "Activate for recognition"}</Button>
          <Link to={`/evaluation?exp=${id}`} className="text-sm text-accent hover:underline">Full evaluation</Link>
        </div>
      )}
    </Card>
  );
}

function Compare({ exps }: { exps: ExperimentSummary[] }) {
  const done = exps.filter((e) => e.status === "completed");
  const [sel, setSel] = useState<number[]>([]);
  const ids = sel.length ? sel : done.slice(0, 3).map((e) => e.id);
  const key = ids.join(",");
  const [loaded, setLoaded] = useState<ExperimentDetail[]>([]);
  useEffect(() => {
    if (!key) return;
    let alive = true;
    Promise.all(key.split(",").map((id) => api.get<ExperimentDetail>(`/api/experiments/${id}`)))
      .then((d) => alive && setLoaded(d))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [key]);
  if (done.length < 2) return null;
  const colors = ["var(--color-series-1)", "var(--color-series-2)", "var(--color-series-3)"];
  const maxEpoch = Math.max(1, ...loaded.map((d) => d.history.length));
  return (
    <Card title="Compare experiments" subtitle="Validation macro F1 per epoch (up to three runs)">
      <div className="mb-4 flex flex-wrap gap-2">
        {done.slice(0, 12).map((e) => {
          const on = ids.includes(e.id);
          return (
            <button key={e.id} onClick={() => setSel(on ? ids.filter((i) => i !== e.id) : [...ids, e.id].slice(-3))}
              className={cx("rounded-lg border px-2.5 py-1 text-xs", on ? "border-accent bg-accent-soft/50 text-fg" : "border-ink-600 text-fg-2 hover:text-fg")} aria-pressed={on}>
              #{e.id} {e.architecture.toUpperCase()} · {pct(e.val?.macro_f1, 0)}
            </button>
          );
        })}
      </div>
      <LineChart ariaLabel="Validation macro F1 per epoch for selected experiments" x={Array.from({ length: maxEpoch }, (_, i) => i + 1)} xLabel="epoch" yDomain={[0, 1]}
        yFormat={(v) => `${Math.round(v * 100)}%`}
        series={loaded.map((d, i) => ({ key: String(d.id), label: `#${d.id} ${d.architecture.toUpperCase()}`, color: colors[i % 3], values: Array.from({ length: maxEpoch }, (_, k) => d.history[k]?.val_macro_f1 ?? null) }))} />
      <ScrollRegion label="Experiment comparison table" className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead><tr className="border-b border-ink-700 text-left text-xs text-fg-3">{["Run", "Arch", "Split", "T", "Params", "Val top-1", "Val top-5", "Val macro F1", "Latency"].map((h) => <th key={h} className="py-2 pr-3 font-medium">{h}</th>)}</tr></thead>
          <tbody>{loaded.map((d) => {
            const vm = d.val_metrics as ExperimentDetail["val_metrics"] & { latency?: { mean_ms: number } };
            return (
              <tr key={d.id} className="border-b border-ink-700/50 font-mono text-xs">
                <td className="py-2 pr-3">#{d.id}</td><td className="py-2 pr-3">{d.architecture.toUpperCase()}</td><td className="py-2 pr-3">{d.split_name}</td>
                <td className="py-2 pr-3">{d.seq_len}</td><td className="py-2 pr-3">{num(d.n_params)}</td><td className="py-2 pr-3">{pct(d.val?.top1)}</td>
                <td className="py-2 pr-3">{pct(d.val?.top5)}</td><td className="py-2 pr-3">{pct(d.val?.macro_f1)}</td><td className="py-2 pr-3">{vm?.latency ? `${vm.latency.mean_ms.toFixed(1)} ms` : "—"}</td>
              </tr>
            );
          })}</tbody>
        </table>
      </ScrollRegion>
    </Card>
  );
}

export default function LabPage() {
  const [f, setF] = useState<Form>(DEFAULT);
  const [err, setErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [runId, setRunId] = useState<number | null>(null);
  const splits = useApi<SplitInfo[]>("/api/splits");
  const exps = useApi<ExperimentSummary[]>("/api/experiments", 4000);
  const summary = useApi<DatasetSummary>("/api/dataset/summary");
  const running = useMemo(() => (exps.data ?? []).find((e) => e.status === "running" || e.status === "queued"), [exps.data]);
  const monitorId = runId ?? running?.id ?? exps.data?.[0]?.id ?? null;
  const rec = summary.data?.sequence_length?.recommended_seq_len;
  const selectedSplit = (splits.data ?? []).find((s) => s.name === f.split_name);
  const set = (patch: Partial<Form>) => setF((x) => ({ ...x, ...patch }));
  const setFeat = (patch: Partial<Form["features"]>) => setF((x) => ({ ...x, features: { ...x.features, ...patch } }));

  const submit = async () => {
    setErr(null);
    setSubmitting(true);
    try {
      const r = await api.post<{ experiment_id: number }>("/api/experiments", { ...f, name: f.name || null });
      setRunId(r.experiment_id);
      exps.reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fade-in">
      <PageHeader eyebrow="Sequence models · PyTorch" title="Model Laboratory"
        description="Configure and train GRU, LSTM or temporal-CNN classifiers on extracted landmark sequences. Model selection uses validation data only; the held-out test set is evaluated separately on the Evaluation page." />

      <div className="grid gap-5 xl:grid-cols-[420px_minmax(0,1fr)]">
        <Card title="New experiment" subtitle="All settings are stored with the run for reproducibility">
          <div className="flex flex-col gap-4">
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button key={p.key} onClick={() => set(p.patch)} className="rounded-lg border border-ink-600 px-2.5 py-1.5 text-left text-xs hover:border-accent" title={p.hint}>
                  <div className="font-medium text-fg">{p.label}</div><div className="text-[10px] text-fg-3">{p.hint}</div>
                </button>
              ))}
            </div>
            <Field label="Architecture" group>
              <Segmented ariaLabel="Architecture" value={f.architecture} onChange={(v) => set({ architecture: v })}
                options={[{ value: "gru", label: "GRU" }, { value: "lstm", label: "LSTM" }, { value: "tcn", label: "Temporal CNN" }]} />
            </Field>
            <Field label="Evaluation split" hint={selectedSplit ? (selectedSplit.signer_independent ? "Signer-independent: no signer appears in more than one partition." : "Not signer-independent — results will be optimistic.") : undefined}>
              <select className={inputCls} value={f.split_name} onChange={(e) => set({ split_name: e.target.value })}>
                {(splits.data ?? []).map((s) => <option key={s.id} value={s.name}>{s.name}{s.signer_independent ? " (signer-independent)" : " (signer overlap)"}</option>)}
              </select>
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <NumField label="Sequence length" value={f.features.seq_len} min={8} max={128} onChange={(v) => setFeat({ seq_len: v })} hint={rec ? `recommended from data: ${rec}` : undefined} />
              <NumField label="Hidden size" value={f.hidden} min={16} max={1024} step={16} onChange={(v) => set({ hidden: v })} />
              <NumField label={f.architecture === "tcn" ? "Depth (blocks ÷ 2)" : "Recurrent layers"} value={f.layers} min={1} max={6} onChange={(v) => set({ layers: v })} />
              <NumField label="Dropout" value={f.dropout} step={0.05} min={0} max={0.9} onChange={(v) => set({ dropout: v })} />
              <NumField label="Epochs (max)" value={f.epochs} min={1} max={500} onChange={(v) => set({ epochs: v })} />
              <NumField label="Early-stop patience" value={f.patience} min={1} max={500} onChange={(v) => set({ patience: v })} />
              <NumField label="Batch size" value={f.batch_size} min={2} max={512} onChange={(v) => set({ batch_size: v })} />
              <NumField label="Learning rate" value={f.lr} step={0.0001} min={0.00001} onChange={(v) => set({ lr: v })} />
              <NumField label="Weight decay" value={f.weight_decay} step={0.005} min={0} onChange={(v) => set({ weight_decay: v })} />
              <NumField label="Label smoothing" value={f.label_smoothing} step={0.05} min={0} max={0.5} onChange={(v) => set({ label_smoothing: v })} />
              <NumField label="Seed" value={f.seed} onChange={(v) => set({ seed: v })} />
              <Field label="Device"><select className={inputCls} value={f.device} onChange={(e) => set({ device: e.target.value as Form["device"] })}><option value="auto">auto (CUDA if available)</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></Field>
            </div>
            <Field label="Class imbalance" group>
              <Segmented ariaLabel="Class weighting" value={f.class_weighting} onChange={(v) => set({ class_weighting: v })} options={[{ value: "none", label: "Unweighted loss" }, { value: "balanced", label: "Class-weighted loss" }]} />
            </Field>
            <div className="flex flex-col gap-2.5">
              {f.architecture !== "tcn" && <Toggle label="Bidirectional recurrence" checked={f.bidirectional} onChange={(v) => set({ bidirectional: v })} />}
              <Toggle label="Training augmentation (mirror, rotation, scale, temporal crop/warp, dropout)" checked={f.augment} onChange={(v) => set({ augment: v })} />
              <Toggle label="Facial landmark subset" checked={f.features.use_face} onChange={(v) => setFeat({ use_face: v })} />
              <Toggle label="Velocity features" checked={f.features.use_velocity} onChange={(v) => setFeat({ use_velocity: v })} />
            </div>
            <Field label="Run name (optional)"><input className={inputCls} value={f.name} onChange={(e) => set({ name: e.target.value })} placeholder="auto" maxLength={120} /></Field>
            <ErrorText error={err} />
            <Button variant="primary" size="lg" icon={<Play className="size-4" />} loading={submitting} disabled={!!running} onClick={submit}>
              {running ? `Run #${running.id} in progress` : "Start training"}
            </Button>
          </div>
        </Card>

        <div className="flex min-w-0 flex-col gap-5">
          {!splits.data?.length && <Notice tone="warn" title="No dataset splits yet">Load the WLASL100 annotations and extract landmarks on the Dataset page before training.</Notice>}
          {monitorId ? (
            <RunMonitor id={monitorId} onDone={() => exps.reload()} />
          ) : (
            <Card><Empty icon={<FlaskConical className="size-6" />} title="No run selected">Start a training run — its loss and accuracy curves will stream here from the real training loop.</Empty></Card>
          )}
          <Compare exps={exps.data ?? []} />
        </div>
      </div>
    </div>
  );
}
