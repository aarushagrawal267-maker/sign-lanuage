import { useMemo, useState } from "react";
import { CheckCircle2, CloudDownload, Download, FileCheck2, Play, ScanFace, ShieldAlert, XCircle } from "lucide-react";
import { api, type DatasetStatus, type DatasetSummary, type InventoryRow, type Job, type SplitInfo, type VideoRow } from "../api/client";
import { useApi } from "../lib/hooks";
import { dateTime, downloadCsv, num, pct } from "../lib/format";
import { Badge, Button, Card, Empty, ErrorText, Field, Kv, Notice, PageHeader, Progress, Segmented, Stat, cx, inputCls } from "../components/ui";
import { ColumnChart } from "../components/charts/Bars";

const STATUS_TONE: Record<string, "good" | "danger" | "warn" | "neutral" | "accent"> = {
  valid: "good", failed: "danger", invalid: "danger", pending: "neutral", downloaded: "accent",
};

function JobPanel({ jobs, onCancel }: { jobs: Job[]; onCancel: (id: number) => void }) {
  const recent = jobs.filter((j) => ["acquire", "scan", "extract"].includes(j.kind)).slice(0, 4);
  if (!recent.length) return null;
  return (
    <Card title="Data jobs" subtitle="Progress is reported by the running job itself">
      <ul className="flex flex-col gap-4">
        {recent.map((j) => (
          <li key={j.id}>
            <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2 text-sm">
              <span className="flex items-center gap-2">
                <span className="font-medium text-fg">{j.kind}</span>
                <span className="font-mono text-xs text-fg-3">#{j.id}</span>
                <Badge tone={j.status === "completed" ? "good" : j.status === "failed" ? "danger" : j.status === "running" ? "accent" : "neutral"}>{j.status}</Badge>
              </span>
              <span className="flex items-center gap-3 text-xs text-fg-3">
                {j.message}
                {(j.status === "running" || j.status === "queued") && <Button size="sm" variant="ghost" onClick={() => onCancel(j.id)}>Cancel</Button>}
              </span>
            </div>
            {(j.status === "running" || j.status === "queued") && <Progress value={j.progress} />}
            {j.status !== "running" && Object.keys(j.result || {}).length > 0 && (
              <div className="font-mono text-2xs text-fg-3">{Object.entries(j.result).map(([k, v]) => `${k}: ${String(v)}`).join(" · ")}</div>
            )}
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Actions({ status, onChanged }: { status: DatasetStatus | null; onChanged: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [limit, setLimit] = useState<string>("50");
  const [classes, setClasses] = useState<string>("");
  const [retry, setRetry] = useState(false);
  const run = async (key: string, fn: () => Promise<unknown>) => {
    setBusy(key);
    setErr(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const classList = classes.split(",").map((s) => s.trim()).filter(Boolean).map(Number).filter((n) => Number.isInteger(n) && n >= 0 && n < 100);
  const loaded = !!status?.annotations_loaded;
  return (
    <Card title="Data acquisition" subtitle="Only WLASL100 instances are ever requested. Failures are recorded, never substituted.">
      <div className="grid gap-5 lg:grid-cols-3">
        <div className="flex flex-col gap-2">
          <div className="text-sm font-medium text-fg">1 · Official annotations</div>
          <p className="text-xs text-fg-3">Downloads WLASL_v0.3.json and nslt_100.json from the official GitHub repository, selects the first 100 glosses, cross-checks them and builds the splits.</p>
          <Button variant={loaded ? "secondary" : "primary"} icon={<FileCheck2 className="size-4" />} loading={busy === "ann"} onClick={() => run("ann", () => api.post("/api/dataset/annotations"))}>
            {loaded ? "Re-verify annotations" : "Load annotations"}
          </Button>
        </div>
        <div className="flex flex-col gap-2">
          <div className="text-sm font-medium text-fg">2 · Videos</div>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Max videos this run"><input className={inputCls} inputMode="numeric" value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="all" /></Field>
            <Field label="Class ids (optional)"><input className={inputCls} value={classes} onChange={(e) => setClasses(e.target.value)} placeholder="e.g. 0,1,2" /></Field>
          </div>
          <label className="flex items-center gap-2 text-xs text-fg-2"><input type="checkbox" checked={retry} onChange={(e) => setRetry(e.target.checked)} /> Retry previously failed</label>
          <div className="flex flex-wrap gap-2">
            <Button variant="primary" icon={<CloudDownload className="size-4" />} disabled={!loaded} loading={busy === "acq"}
              onClick={() => run("acq", () => api.post("/api/dataset/acquire", { limit: limit ? Number(limit) : null, class_indices: classList.length ? classList : null, retry_failed: retry }))}>
              Download &amp; validate
            </Button>
            <Button variant="secondary" disabled={!loaded} loading={busy === "scan"} onClick={() => run("scan", () => api.post("/api/dataset/scan"))}>Scan local files</Button>
          </div>
          <p className="text-2xs text-fg-3">Videos provided by the WLASL authors go in <code className="font-mono">data/videos/provided/&lt;video_id&gt;.mp4</code>, then “Scan local files”.</p>
        </div>
        <div className="flex flex-col gap-2">
          <div className="text-sm font-medium text-fg">3 · Landmarks</div>
          <p className="text-xs text-fg-3">Runs MediaPipe Holistic over each validated clip (annotated frame range only) and stores raw landmark sequences.</p>
          <Button variant="primary" icon={<ScanFace className="size-4" />} disabled={!loaded || !(status?.status_counts.valid ?? 0)} loading={busy === "ext"}
            onClick={() => run("ext", () => api.post("/api/dataset/extract", { limit: null, redo: false }))}>
            Extract landmarks
          </Button>
        </div>
      </div>
      <div className="mt-3"><ErrorText error={err} /></div>
    </Card>
  );
}

function SplitsCard({ splits }: { splits: SplitInfo[] }) {
  if (!splits.length) return null;
  return (
    <Card title="Partitions & leakage checks" subtitle="Automated checks: signer overlap, duplicate content (hash), shared source files, class coverage">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead>
            <tr className="border-b border-ink-700 text-left text-xs text-fg-3">
              <th className="py-2 pr-3 font-medium">Split</th>
              <th className="py-2 pr-3 font-medium">Videos train / val / test</th>
              <th className="py-2 pr-3 font-medium">Signers train / val / test</th>
              <th className="py-2 pr-3 font-medium">Signer overlap</th>
              <th className="py-2 pr-3 font-medium">Test videos with seen signer</th>
              <th className="py-2 pr-3 font-medium">Missing classes val / test</th>
              <th className="py-2 font-medium">Checks</th>
            </tr>
          </thead>
          <tbody>
            {splits.map((s) => {
              const r = s.leakage_report;
              const ov = Object.values(r.signer_overlap).reduce((a, b) => a + b.count, 0);
              const t = r.eval_videos_with_train_signers.test;
              return (
                <tr key={s.id} className="border-b border-ink-700/60">
                  <td className="py-2.5 pr-3"><div className="font-medium text-fg">{s.name}</div><div className="text-2xs text-fg-3">{s.kind}</div></td>
                  <td className="py-2.5 pr-3 font-mono text-xs">{r.n_videos.train} / {r.n_videos.val} / {r.n_videos.test}</td>
                  <td className="py-2.5 pr-3 font-mono text-xs">{r.n_signers.train} / {r.n_signers.val} / {r.n_signers.test}</td>
                  <td className="py-2.5 pr-3 font-mono text-xs">{ov}</td>
                  <td className="py-2.5 pr-3 font-mono text-xs">{t.videos_with_train_signer} / {t.videos}</td>
                  <td className="py-2.5 pr-3 font-mono text-xs">{r.classes_missing.val.length} / {r.classes_missing.test.length}</td>
                  <td className="py-2.5">
                    {r.passed ? <Badge tone="good"><CheckCircle2 className="size-3" />signer-independent</Badge>
                      : <Badge tone="warn"><ShieldAlert className="size-3" />{r.signer_independent ? "source overlap" : "not signer-independent"}</Badge>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-fg-3">The official WLASL split is kept for comparability with published results but is labelled as not signer-independent. Signers who share a source video are grouped so the same footage never crosses partitions.</p>
    </Card>
  );
}

function VideoTable({ inventory }: { inventory: InventoryRow[] }) {
  const [cls, setCls] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [split, setSplit] = useState<string>("");
  const [page, setPage] = useState(0);
  const PAGE = 25;
  const q = new URLSearchParams({ offset: String(page * PAGE), limit: String(PAGE) });
  if (cls) q.set("class_index", cls);
  if (status) q.set("status", status);
  if (split) q.set("split", split);
  const { data } = useApi<{ total: number; items: VideoRow[] }>(`/api/dataset/videos?${q}`);
  const total = data?.total ?? 0;
  return (
    <Card title="Video instances" subtitle="Official annotation fields are shown as published; measured values come from validation">
      <div className="mb-3 flex flex-wrap gap-2">
        <select aria-label="Filter by gloss" className={cx(inputCls, "w-48")} value={cls} onChange={(e) => { setCls(e.target.value); setPage(0); }}>
          <option value="">All glosses</option>
          {inventory.map((g) => <option key={g.class_index} value={g.class_index}>{g.class_index} · {g.gloss}</option>)}
        </select>
        <select aria-label="Filter by status" className={cx(inputCls, "w-40")} value={status} onChange={(e) => { setStatus(e.target.value); setPage(0); }}>
          <option value="">Any status</option>
          {["pending", "valid", "failed", "invalid", "downloaded"].map((s) => <option key={s}>{s}</option>)}
        </select>
        <select aria-label="Filter by split" className={cx(inputCls, "w-36")} value={split} onChange={(e) => { setSplit(e.target.value); setPage(0); }}>
          <option value="">Any split</option>
          {["train", "val", "test"].map((s) => <option key={s}>{s}</option>)}
        </select>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-sm">
          <thead>
            <tr className="border-b border-ink-700 text-left text-xs text-fg-3">
              {["Video", "Gloss", "Signer", "Split", "Host", "Frames (annot.)", "Status", "Measured", "Landmarks"].map((h) => <th key={h} className="py-2 pr-3 font-medium">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {(data?.items ?? []).map((v) => (
              <tr key={v.video_id} className="border-b border-ink-700/50 align-top">
                <td className="py-2 pr-3 font-mono text-xs text-fg">{v.video_id}{v.variation_id > 0 && <span className="ml-1 text-fg-3">v{v.variation_id}</span>}</td>
                <td className="py-2 pr-3 font-mono text-xs">{v.gloss.toUpperCase()}</td>
                <td className="py-2 pr-3 font-mono text-xs">{v.signer_id}</td>
                <td className="py-2 pr-3 text-xs">{v.split}</td>
                <td className="py-2 pr-3 text-xs text-fg-2"><a href={v.url} target="_blank" rel="noreferrer noopener" className="hover:text-accent-fg">{v.host}</a></td>
                <td className="py-2 pr-3 font-mono text-xs">{v.frame_start}–{v.frame_end === -1 ? "end" : v.frame_end}</td>
                <td className="py-2 pr-3"><Badge tone={STATUS_TONE[v.status] ?? "neutral"}>{v.status}</Badge>{v.error && <div className="mt-1 max-w-56 truncate text-2xs text-fg-3" title={v.error}>{v.error}</div>}</td>
                <td className="py-2 pr-3 font-mono text-2xs text-fg-2">{v.width ? `${v.width}×${v.height} · ${v.fps?.toFixed(0)} fps · ${v.clip_frames} fr` : "—"}</td>
                <td className="py-2 font-mono text-2xs text-fg-2">{v.landmarks ? `${v.landmarks.status} · hands ${pct(v.landmarks.any_hand_rate, 0)}` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-fg-3">
        <span>{num(total)} instances</span>
        <span className="flex items-center gap-2">
          <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
          <span className="font-mono">{page + 1} / {Math.max(1, Math.ceil(total / PAGE))}</span>
          <Button size="sm" variant="ghost" disabled={(page + 1) * PAGE >= total} onClick={() => setPage(page + 1)}>Next</Button>
        </span>
      </div>
    </Card>
  );
}

export default function DatasetPage() {
  const status = useApi<DatasetStatus>("/api/dataset/status", 4000);
  const summary = useApi<DatasetSummary>("/api/dataset/summary", 8000);
  const inv = useApi<{ glosses: InventoryRow[] }>("/api/dataset/inventory", 8000);
  const splits = useApi<SplitInfo[]>("/api/splits", 15000);
  const jobs = useApi<Job[]>("/api/jobs?limit=10", 2000);
  const [sortBy, setSortBy] = useState<"class" | "annotated" | "valid">("class");

  const glosses = useMemo(() => {
    const g = [...(inv.data?.glosses ?? [])];
    if (sortBy === "annotated") g.sort((a, b) => b.annotated - a.annotated);
    if (sortBy === "valid") g.sort((a, b) => b.valid - a.valid);
    return g;
  }, [inv.data, sortBy]);

  const s = summary.data;
  const st = status.data;
  const reloadAll = () => [status, summary, inv, splits, jobs].forEach((x) => x.reload());
  const prov = st?.provenance;

  return (
    <div className="fade-in">
      <PageHeader eyebrow="WLASL100 · official annotations" title="Dataset explorer"
        description="Inventory, availability, signers, splits and landmark extraction status for the WLASL100 subset. Videos stay on this machine and are never redistributed (C-UDA)."
        actions={glosses.length > 0 && <Button variant="ghost" icon={<Download className="size-4" />} onClick={() => downloadCsv("wlasl100_inventory.csv", glosses.map((g) => ({ ...g, signers: g.signers.join(" "), split: JSON.stringify(g.split), resolutions: JSON.stringify(g.resolutions), fps: JSON.stringify(g.fps) })))}>Export inventory</Button>} />

      {st?.synthetic && <div className="mb-5"><Notice tone="warn" title="Synthetic test data present">This data directory contains synthetic test sequences, not WLASL videos.</Notice></div>}

      <div className="flex flex-col gap-5">
        <Actions status={st} onChanged={reloadAll} />
        <JobPanel jobs={jobs.data ?? []} onCancel={(id) => api.post(`/api/jobs/${id}/cancel`).then(() => jobs.reload())} />

        {!st?.annotations_loaded ? (
          <Card><Empty icon={<Play className="size-6" />} title="No annotations loaded yet">Load the official annotation files to build the WLASL100 inventory. Nothing else is downloaded until you start acquisition.</Empty></Card>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
              <Stat label="Annotated instances" value={num(s?.videos)} sub={s ? `train ${s.official_split.train ?? 0} · val ${s.official_split.val ?? 0} · test ${s.official_split.test ?? 0}` : undefined} />
              <Stat label="Valid videos" value={num(s?.status_counts.valid ?? 0)} />
              <Stat label="Unavailable" value={num((s?.status_counts.failed ?? 0) + (s?.status_counts.invalid ?? 0))} tone={(s?.status_counts.failed ?? 0) > 0 ? "warn" : undefined} sub="dead links or unreadable files" />
              <Stat label="Pending" value={num(s?.status_counts.pending ?? 0)} />
              <Stat label="Landmark sequences" value={num(s?.landmarks.status_counts.ok ?? 0)} sub={s?.landmarks.mean_any_hand_rate !== null && s?.landmarks.mean_any_hand_rate !== undefined ? `mean hand detection ${pct(s.landmarks.mean_any_hand_rate, 0)}` : undefined} />
            </div>

            <Card title="Class distribution" subtitle="Annotated instances per gloss, with valid videos overlaid"
              actions={<Segmented ariaLabel="Sort classes" value={sortBy} onChange={setSortBy} options={[{ value: "class", label: "Class id" }, { value: "annotated", label: "Annotated" }, { value: "valid", label: "Valid" }]} />}>
              <ColumnChart ariaLabel="Annotated and valid videos per gloss" height={170} legend={{ primary: "annotated", secondary: "valid" }}
                bars={glosses.map((g) => ({ label: `${g.class_index} · ${g.gloss.toUpperCase()}`, value: g.annotated, secondary: g.valid, hint: `${g.unavailable} unavailable · ${g.landmarks} with landmarks · ${g.signers.length} signers` }))} />
              <p className="mt-2 text-xs text-fg-3">Classes have between {Math.min(...glosses.map((g) => g.annotated))} and {Math.max(...glosses.map((g) => g.annotated))} annotated instances — the vocabulary is not balanced.</p>
            </Card>

            <div className="grid gap-5 lg:grid-cols-2">
              <Card title="Availability by host" subtitle="Where the official URLs point">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="border-b border-ink-700 text-left text-xs text-fg-3"><th className="py-2 font-medium">Host</th><th className="py-2 text-right font-medium">Total</th><th className="py-2 text-right font-medium">Valid</th><th className="py-2 text-right font-medium">Failed</th><th className="py-2 text-right font-medium">Pending</th></tr></thead>
                    <tbody>
                      {(s?.hosts ?? []).map((h) => (
                        <tr key={h.host} className="border-b border-ink-700/50">
                          <td className="py-1.5 text-fg-2">{h.host}</td>
                          <td className="py-1.5 text-right font-mono text-xs">{h.total}</td>
                          <td className="py-1.5 text-right font-mono text-xs text-fg">{(h.valid as number) ?? 0}</td>
                          <td className="py-1.5 text-right font-mono text-xs text-danger">{((h.failed as number) ?? 0) + ((h.invalid as number) ?? 0)}</td>
                          <td className="py-1.5 text-right font-mono text-xs text-fg-3">{(h.pending as number) ?? 0}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {s && s.top_errors.length > 0 && (
                  <div className="mt-4">
                    <div className="mb-1 text-xs font-medium text-fg-2">Most common failure reasons</div>
                    <ul className="flex flex-col gap-1 text-xs text-fg-3">{s.top_errors.slice(0, 5).map((e) => <li key={e.error} className="flex justify-between gap-3"><span className="truncate">{e.error || "unknown"}</span><span className="font-mono">{e.count}</span></li>)}</ul>
                  </div>
                )}
              </Card>
              <Card title="Clip duration & sequence length" subtitle="Measured on validated clips (annotated frame range)">
                {s?.duration_histogram ? (
                  <>
                    <ColumnChart ariaLabel="Histogram of clip durations" height={130} valueFormat={(v) => `${v} clips`}
                      bars={s.duration_histogram.map((b) => ({ label: `${b.from.toFixed(1)}–${b.to.toFixed(1)} s`, value: b.count }))} />
                    <div className="mt-1 flex justify-between font-mono text-2xs text-fg-3"><span>0 s</span><span>{s.duration_histogram[s.duration_histogram.length - 1].to.toFixed(1)} s</span></div>
                    {s.sequence_length && (
                      <div className="mt-4">
                        <Kv k="Median clip length" v={`${s.sequence_length.median_frames.toFixed(0)} frames`} />
                        <Kv k="10th–90th percentile" v={`${s.sequence_length.p10_frames.toFixed(0)}–${s.sequence_length.p90_frames.toFixed(0)} frames`} />
                        <Kv k="Recommended sequence length" v={<span className="text-accent-fg">{s.sequence_length.recommended_seq_len} steps</span>} />
                        <p className="mt-2 text-2xs text-fg-3">Basis: {s.sequence_length.basis}. Each clip is resampled to this many steps by timestamp.</p>
                      </div>
                    )}
                  </>
                ) : <Empty title="No validated clips yet">Durations are measured once videos are downloaded and validated.</Empty>}
              </Card>
            </div>

            <div className="grid gap-5 lg:grid-cols-[1fr_1.4fr]">
              <Card title="Signers" subtitle={`${s?.signers.length ?? 0} signer ids · videos per signer (official split)`}>
                <div className="max-h-80 overflow-y-auto pr-1">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-ink-850"><tr className="border-b border-ink-700 text-left text-xs text-fg-3"><th className="py-1.5 font-medium">Signer</th><th className="py-1.5 text-right font-medium">Videos</th><th className="py-1.5 text-right font-medium">train / val / test</th></tr></thead>
                    <tbody>{(s?.signers ?? []).map((g) => (
                      <tr key={g.signer_id} className="border-b border-ink-700/40"><td className="py-1 font-mono text-xs">{g.signer_id}</td><td className="py-1 text-right font-mono text-xs">{g.videos}</td><td className="py-1 text-right font-mono text-xs text-fg-3">{g.split.train ?? 0} / {g.split.val ?? 0} / {g.split.test ?? 0}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
                <p className="mt-3 text-2xs text-fg-3">Signers appearing in several official partitions are why the official split is not signer-independent.</p>
              </Card>
              <Card title="Provenance" subtitle="Annotation files exactly as fetched">
                {prov ? (
                  <div>
                    <Kv k="Ingested" v={dateTime(prov.ingested_at)} />
                    {Object.entries(prov.files).map(([name, f]) => <Kv key={name} k={name} v={<span title={f.sha256}>sha256 {f.sha256.slice(0, 16)}…</span>} />)}
                    {Object.entries(prov.checks).map(([k, v]) => <Kv key={k} k={k.replace(/_/g, " ")} v={String(v)} />)}
                  </div>
                ) : <p className="text-sm text-fg-3">No provenance recorded.</p>}
              </Card>
            </div>

            <SplitsCard splits={splits.data ?? []} />
            <VideoTable inventory={inv.data?.glosses ?? []} />
            {summary.error && <ErrorText error={summary.error} />}
            {(s?.status_counts.failed ?? 0) > 0 && (
              <Notice title="Missing videos"><XCircle className="mr-1 inline size-3.5" />Some official URLs no longer work. The WLASL authors provide missing videos on request (see Research &amp; Docs). Put received files in <code className="font-mono">data/videos/provided/</code> and run “Scan local files”.</Notice>
            )}
          </>
        )}
      </div>
    </div>
  );
}
