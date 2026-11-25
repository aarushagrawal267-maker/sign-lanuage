import { Fragment, useState } from "react";
import { Link } from "react-router-dom";
import { ChevronDown, ChevronRight, Download, Trash2, Zap } from "lucide-react";
import { IS_STATIC, api, type ExperimentDetail, type ExperimentSummary } from "../api/client";
import { useApi } from "../lib/hooks";
import { dateTime, num, pct } from "../lib/format";
import { Badge, Button, Card, Empty, Kv, PageHeader, buttonClass } from "../components/ui";
import { TrainingCurves } from "./LabPage";

function Detail({ id, onChange }: { id: number; onChange: () => void }) {
  const { data: e } = useApi<ExperimentDetail>(`/api/experiments/${id}`);
  if (!e) return <p className="p-4 text-sm text-fg-3">Loading…</p>;
  const ds = e.data_summary as Record<string, { videos?: number; signers?: number; classes?: number; median_frames?: number }> & { skipped?: Record<string, number> };
  return (
    <div className="grid gap-5 border-t border-ink-700 bg-ink-900/50 p-5 lg:grid-cols-[320px_1fr]">
      <div>
        <div className="mb-2 text-xs font-medium text-fg-2">Configuration</div>
        {Object.entries(e.config).filter(([k]) => k !== "features").map(([k, v]) => <Kv key={k} k={k.replace(/_/g, " ")} v={String(v)} />)}
        <div className="mb-2 mt-4 text-xs font-medium text-fg-2">Features</div>
        {Object.entries(e.feature_config).map(([k, v]) => <Kv key={k} k={k.replace(/_/g, " ")} v={String(v)} />)}
        <div className="mb-2 mt-4 text-xs font-medium text-fg-2">Signer partitions</div>
        {(["train", "val", "test"] as const).map((p) => ds?.[p] && <Kv key={p} k={p} v={`${ds[p]?.videos} seq · ${ds[p]?.signers} signers · ${ds[p]?.classes} classes`} />)}
        {ds?.skipped && <Kv k="excluded" v={Object.entries(ds.skipped).filter(([, n]) => n).map(([k, n]) => `${k} ${n}`).join(", ") || "none"} />}
        <div className="mb-2 mt-4 text-xs font-medium text-fg-2">Provenance</div>
        <Kv k="dataset" v={e.dataset_version ?? "—"} />
        <Kv k="checkpoint" v={<span className="break-all text-[11px]">{e.checkpoint_path ?? "—"}</span>} />
        <Kv k="device" v={e.device ?? "—"} />
        <div className="mt-4 flex flex-wrap gap-2">
          <Button size="sm" variant="primary" icon={<Zap className="size-3.5" />} disabled={e.status !== "completed" || e.is_active} onClick={() => api.post(`/api/experiments/${id}/activate`).then(onChange)}>
            {e.is_active ? "Active" : "Activate"}
          </Button>
          <Link to={`/evaluation?exp=${id}`} className={buttonClass("secondary", "sm")}>Evaluation</Link>
          <Button size="sm" variant="danger" icon={<Trash2 className="size-3.5" />} disabled={e.is_active || e.status === "running"}
            onClick={() => window.confirm(`Delete experiment #${id} and its checkpoint?`) && api.del(`/api/experiments/${id}`).then(onChange)}>Delete</Button>
        </div>
      </div>
      <div className="min-w-0">
        <TrainingCurves exp={e} />
        {e.error && <p className="mt-3 text-sm text-danger">{e.error}</p>}
      </div>
    </div>
  );
}

export default function ExperimentsPage() {
  const exps = useApi<ExperimentSummary[]>("/api/experiments", 5000);
  const [open, setOpen] = useState<number | null>(null);
  const rows = exps.data ?? [];
  return (
    <div className="fade-in">
      <PageHeader eyebrow="Reproducible runs" title="Experiment History"
        description="Every training run with its dataset version, vocabulary, architecture, features, sequence length, parameters, signer partitions, metrics, checkpoint and timestamp."
        actions={rows.length > 0 && !IS_STATIC && <a href="/api/experiments/export.csv" download className={buttonClass("ghost")}><Download className="size-4" aria-hidden />Export CSV</a>} />
      <Card padded={false}>
        {!rows.length ? (
          <Empty title="No experiments yet">Runs started in the Model Laboratory are recorded here automatically.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[980px] text-sm">
              <thead><tr className="border-b border-ink-700 text-left text-xs text-fg-3">
                {["Details", "Run", "Status", "Architecture", "Split", "T", "Params", "Epochs", "Val top-1", "Val F1", "Test top-1", "Created"].map((h, i) => <th key={i} className="px-3 py-2.5 font-medium">{i === 0 ? <span className="sr-only">{h}</span> : h}</th>)}
              </tr></thead>
              <tbody>
                {rows.map((r) => (
                  <Fragment key={r.id}>
                    <tr className="cursor-pointer border-b border-ink-700/60 hover:bg-ink-800/50" onClick={() => setOpen(open === r.id ? null : r.id)}>
                      <td className="px-3 py-2.5 text-fg-3">
                        {/* The whole row is clickable for mouse users; this button is the keyboard and screen-reader control. */}
                        <button type="button" className="rounded p-0.5 hover:text-fg" aria-expanded={open === r.id} aria-label={`Details for run #${r.id} ${r.name}`}
                          onClick={(e) => { e.stopPropagation(); setOpen(open === r.id ? null : r.id); }}>
                          {open === r.id ? <ChevronDown className="size-4" aria-hidden /> : <ChevronRight className="size-4" aria-hidden />}
                        </button>
                      </td>
                      <td className="px-3 py-2.5">
                        <div className="flex items-center gap-2"><span className="font-mono text-xs text-fg-3">#{r.id}</span><span className="text-fg">{r.name}</span></div>
                        <div className="mt-0.5 flex gap-1.5">{r.is_active && <Badge tone="teal">active</Badge>}{r.is_synthetic && <Badge tone="warn">synthetic</Badge>}</div>
                      </td>
                      <td className="px-3 py-2.5"><Badge tone={r.status === "completed" ? "good" : r.status === "failed" ? "danger" : r.status === "running" ? "accent" : "neutral"}>{r.status}</Badge></td>
                      <td className="px-3 py-2.5 font-mono text-xs">{r.architecture.toUpperCase()}</td>
                      <td className="px-3 py-2.5 text-xs text-fg-2">{r.split_name}</td>
                      <td className="px-3 py-2.5 font-mono text-xs">{r.seq_len ?? "—"}</td>
                      <td className="px-3 py-2.5 font-mono text-xs">{num(r.n_params)}</td>
                      <td className="px-3 py-2.5 font-mono text-xs">{r.epochs_run}/{r.epochs_planned ?? "—"}</td>
                      <td className="px-3 py-2.5 font-mono text-xs">{pct(r.val?.top1)}</td>
                      <td className="px-3 py-2.5 font-mono text-xs">{pct(r.val?.macro_f1)}</td>
                      <td className="px-3 py-2.5 font-mono text-xs">{r.test ? pct(r.test.top1) : <span className="text-fg-3">—</span>}</td>
                      <td className="px-3 py-2.5 text-xs text-fg-3">{dateTime(r.created_at)}</td>
                    </tr>
                    {open === r.id && (
                      <tr><td colSpan={12} className="p-0"><Detail id={r.id} onChange={() => exps.reload()} /></td></tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
