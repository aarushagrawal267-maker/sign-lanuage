import { Link } from "react-router-dom";
import { ArrowRight, CheckCircle2, Circle, Cpu, Database, FlaskConical, Video } from "lucide-react";
import { useApi } from "../lib/hooks";
import { dateTime, ms, num, pct, timeAgo } from "../lib/format";
import type { Overview, SystemInfo } from "../api/client";
import { Badge, Card, Empty, Kv, Notice, PageHeader, Progress, Stat, buttonClass } from "../components/ui";

function Step({ done, title, children, to, cta }: { done: boolean; title: string; children: React.ReactNode; to: string; cta: string }) {
  return (
    <li className="flex gap-3 py-3">
      {done ? <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-teal" aria-label="done" /> : <Circle className="mt-0.5 size-5 shrink-0 text-fg-3" aria-label="to do" />}
      <div className="flex min-w-0 flex-1 flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className={done ? "text-sm text-fg-2" : "text-sm font-medium text-fg"}>{title}</div>
          <div className="text-xs text-fg-3">{children}</div>
        </div>
        {!done && <Link to={to} className="inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline">{cta}<ArrowRight className="size-3" /></Link>}
      </div>
    </li>
  );
}

export default function OverviewPage() {
  const { data: o, error } = useApi<Overview>("/api/overview", 5000);
  const { data: sys } = useApi<SystemInfo>("/api/system");

  if (error && !o) {
    return (
      <>
        <PageHeader eyebrow="Sign intelligence overview" title="Overview" />
        <Notice tone="danger" title="Backend not reachable">
          {error}. Start the application with <code>start.ps1</code> (backend on http://localhost:8000).
        </Notice>
      </>
    );
  }
  if (!o) return <PageHeader eyebrow="Sign intelligence overview" title="Overview" description="Loading…" />;

  const st = o.video_status;
  const valid = st.valid ?? 0;
  const unavailable = (st.failed ?? 0) + (st.invalid ?? 0);
  const active = o.active_model;
  const steps = [
    { done: o.total_videos > 0, title: "Load the official WLASL100 annotations", body: "Fetches WLASL_v0.3.json + nslt_100.json and builds signer-grouped splits.", to: "/dataset", cta: "Dataset" },
    { done: valid > 0, title: "Acquire and validate videos", body: `${num(valid)} valid of ${num(o.total_videos)} annotated instances.`, to: "/dataset", cta: "Acquire" },
    { done: o.landmark_sequences > 0, title: "Extract MediaPipe landmarks", body: `${num(o.landmark_sequences)} landmark sequences extracted.`, to: "/dataset", cta: "Extract" },
    { done: (o.experiments.completed ?? 0) > 0, title: "Train a sequence model", body: "GRU, LSTM or temporal CNN on the signer-independent split.", to: "/lab", cta: "Model Laboratory" },
    { done: !!active && !active.is_synthetic, title: "Activate a model and sign live", body: "Use your webcam on the Live Recognition page.", to: active ? "/live" : "/experiments", cta: active ? "Go live" : "Activate" },
  ];

  return (
    <div className="fade-in">
      <PageHeader eyebrow="Sign intelligence overview" title="Overview"
        description="Status of the WLASL100 dataset, landmark extraction, models and recent recognitions. Every number on this page comes from the local database."
        actions={<Link to="/live" className={buttonClass("primary")}><Video className="size-4" aria-hidden />Live recognition</Link>} />

      {active?.is_synthetic && (
        <div className="mb-5"><Notice tone="warn" title="Active model is a synthetic test model">It was trained on generated landmark data to verify the pipeline and is not an ASL recogniser.</Notice></div>
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Vocabulary" value={num(o.vocabulary_size)} sub="WLASL100 glosses" />
        <Stat label="Valid videos" value={num(valid)} sub={`${num(unavailable)} unavailable · ${num((st.pending ?? 0) + (st.downloaded ?? 0))} pending`} />
        <Stat label="Landmark sequences" value={num(o.landmark_sequences)} sub={Object.entries(o.samples_by_official_split).map(([k, v]) => `${k} ${v}`).join(" · ") || "none yet"} />
        <Stat label="Active model" value={active ? `#${active.id}` : "—"} tone={active ? "accent" : undefined}
          sub={active ? `${active.architecture.toUpperCase()} · val top-1 ${pct(active.val?.top1)}` : "train & activate in the Laboratory"} />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1.3fr_1fr]">
        <Card title="Pipeline status" subtitle="From annotations to live recognition">
          <ol className="divide-y divide-ink-700/60">
            {steps.map((s) => <Step key={s.title} done={s.done} title={s.title} to={s.to} cta={s.cta}>{s.body}</Step>)}
          </ol>
          {o.running_jobs.length > 0 && (
            <div className="mt-3 flex flex-col gap-3 rounded-xl border border-ink-700 bg-ink-900 p-3">
              {o.running_jobs.map((j) => (
                <div key={j.id}>
                  <div className="mb-1.5 flex justify-between text-xs"><span className="font-medium text-fg">{j.kind} job #{j.id}</span><span className="text-fg-3">{j.message ?? "starting…"}</span></div>
                  <Progress value={j.progress} label={`${j.kind} job #${j.id} progress`} />
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card title="Latest evaluation" subtitle={o.latest_experiment ? `${o.latest_experiment.name} · finished ${timeAgo(o.latest_experiment.finished_at)}` : undefined}
          actions={o.latest_experiment && <Link to={`/evaluation?exp=${o.latest_experiment.id}`} className="text-xs font-medium text-accent hover:underline">Details</Link>}>
          {o.latest_experiment ? (
            <div>
              <div className="mb-3 flex flex-wrap gap-1.5">
                <Badge tone="accent">{o.latest_experiment.architecture.toUpperCase()}</Badge>
                <Badge>{o.latest_experiment.split_name}</Badge>
                {o.latest_experiment.is_synthetic && <Badge tone="warn">synthetic</Badge>}
              </div>
              <Kv k="Validation top-1" v={pct(o.latest_experiment.val?.top1)} />
              <Kv k="Validation top-5" v={pct(o.latest_experiment.val?.top5)} />
              <Kv k="Validation macro F1" v={pct(o.latest_experiment.val?.macro_f1)} />
              <Kv k="Held-out test top-1" v={o.latest_experiment.test ? pct(o.latest_experiment.test.top1) : "not evaluated"} />
              <Kv k="Model inference (CPU, batch 1)" v={ms(o.latest_experiment.latency?.mean_ms)} />
              <Kv k="Parameters" v={num(o.latest_experiment.n_params)} />
            </div>
          ) : (
            <Empty icon={<FlaskConical className="size-6" />} title="No completed experiments">Train a model in the Model Laboratory to see real evaluation results here.</Empty>
          )}
        </Card>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-3">
        <Card title="Recent recognition activity" className="lg:col-span-2">
          {o.recent_events.length === 0 ? (
            <Empty title="No recognitions yet">Recognised signs from the live page and video analysis are listed here (gloss and score only — no video is stored).</Empty>
          ) : (
            <ul className="grid gap-x-6 sm:grid-cols-2">
              {o.recent_events.map((e, i) => (
                <li key={i} className="flex items-center justify-between border-b border-ink-700/60 py-2 text-sm">
                  <span className="font-mono text-fg">{e.gloss.toUpperCase()}</span>
                  <span className="flex gap-3 font-mono text-xs text-fg-3"><span>{pct(e.score, 0)}</span><span>{e.source}</span><span>{timeAgo(e.created_at)}</span></span>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card title="System" actions={<Cpu className="size-4 text-fg-3" />}>
          {sys ? (
            <div>
              <Kv k="Compute" v={sys.cuda_available ? `CUDA · ${sys.cuda_device}` : "CPU"} />
              <Kv k="PyTorch" v={sys.torch} />
              <Kv k="MediaPipe" v={sys.mediapipe ?? "missing"} />
              <Kv k="Holistic model" v={sys.holistic_model_present ? "present" : <span className="text-danger">missing — run setup.ps1</span>} />
              <Kv k="Python" v={sys.python} />
              <Kv k="Experiments" v={Object.entries(o.experiments).map(([k, v]) => `${v} ${k}`).join(", ") || "none"} />
            </div>
          ) : <p className="text-sm text-fg-3">Loading…</p>}
        </Card>
      </div>
      {active && (
        <p className="mt-5 flex items-center gap-2 text-xs text-fg-3"><Database className="size-3.5" />Active model #{active.id} “{active.name}” · finished {dateTime(active.finished_at)}</p>
      )}
    </div>
  );
}
