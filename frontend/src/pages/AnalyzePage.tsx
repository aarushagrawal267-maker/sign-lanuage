import { useEffect, useRef, useState } from "react";
import { FileVideo, Upload } from "lucide-react";
import { IS_STATIC, api, type ModelInfo, type RawFrame, type TopK } from "../api/client";
import { drawFrame } from "../mediapipe/draw";
import { ms, pct } from "../lib/format";
import { REJECT_REASON } from "../recognition/status";
import { Badge, Button, Card, Empty, ErrorText, Field, Kv, Notice, PageHeader, Segmented, Toggle } from "../components/ui";
import { HBars } from "../components/charts/Bars";

type Segment = { accepted: boolean; reason: string; frames: number; duration_ms?: number; agreement?: string; top?: TopK[]; end_t: number };
type Result = {
  filename: string; frames: number; fps: number; width: number; height: number; duration_s: number;
  extract_seconds: number; extract_ms_per_frame: number;
  quality: { frames: number; pose_rate: number; left_rate: number; right_rate: number; any_hand_rate: number; face_rate: number; usable: boolean };
  landmarks: { frame_step: number; frames: RawFrame[] };
  model: ModelInfo | null;
  whole_clip: { top?: TopK[]; inference_ms?: number; error?: string } | null;
  stream: { segments: Segment[]; commits: { gloss: string; score: number; t: number }[]; threshold: number; mode: string } | null;
  message?: string;
};

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<"segment" | "sliding">("segment");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<Result | null>(null);
  const [overlay, setOverlay] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  const pick = (f: File | null) => {
    setRes(null);
    setErr(null);
    setFile(f);
    if (url) URL.revokeObjectURL(url);
    setUrl(f ? URL.createObjectURL(f) : null);
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("mode", mode);
      setRes(await api.upload<Result>("/api/analyze/video", fd));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Landmark overlay synchronised with playback
  useEffect(() => {
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c || !res) return;
    let h = 0;
    const frames = res.landmarks.frames;
    const tick = () => {
      const w = c.clientWidth, hh = c.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); }
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const t = v.currentTime * 1000;
      let best = frames[0];
      for (const f of frames) { if (f.t <= t) best = f; else break; }
      drawFrame(ctx, overlay ? best : null, w, hh, { pose: true, hands: true, face: true, mirror: false });
      h = requestAnimationFrame(tick);
    };
    h = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(h);
  }, [res, overlay]);

  const dur = res ? res.duration_s * 1000 : 1;

  return (
    <div className="fade-in">
      <PageHeader eyebrow="Offline · not live recognition" title="Video analysis"
        description="Analyse a local video file you are permitted to use. Frames are processed by MediaPipe on the local backend, classified with the active model, and the temporary copy is deleted immediately afterwards." />

      {IS_STATIC && <div className="mb-5"><Notice tone="warn" title="Not available in the online demo">Video-file analysis needs the Python backend. Run the project locally (setup.ps1 / start.ps1) to use it; live webcam recognition works here.</Notice></div>}
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="flex min-w-0 flex-col gap-4">
          <Card>
            <div className="flex flex-col gap-4 md:flex-row md:items-end">
              <label className="flex flex-1 cursor-pointer items-center gap-3 rounded-lg border border-dashed border-ink-600 px-4 py-4 hover:border-accent">
                <Upload className="size-5 text-fg-3" aria-hidden />
                <span className="min-w-0 text-sm">
                  <span className="block truncate font-medium text-fg">{file ? file.name : "Choose a video file"}</span>
                  <span className="text-xs text-fg-3">{file ? `${(file.size / 1e6).toFixed(1)} MB` : "MP4, WebM, MOV… up to 200 MB. One sign per clip works best."}</span>
                </span>
                <input type="file" accept="video/*" className="sr-only" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
              </label>
              <Field label="Mode">
                <Segmented ariaLabel="Analysis mode" value={mode} onChange={setMode} options={[{ value: "segment", label: "Sign boundaries" }, { value: "sliding", label: "Sliding window" }]} />
              </Field>
              <Button variant="primary" size="lg" icon={<FileVideo className="size-4" />} disabled={!file || IS_STATIC} loading={busy} onClick={analyze}>{busy ? "Analysing…" : "Analyse"}</Button>
            </div>
            <div className="mt-3"><ErrorText error={err} /></div>
          </Card>

          {url ? (
            <div className="relative overflow-hidden rounded-xl border border-ink-700 bg-black" style={{ aspectRatio: res ? `${res.width} / ${res.height}` : "16 / 9" }}>
              <video ref={videoRef} src={url} controls playsInline muted className="absolute inset-0 size-full object-contain" />
              <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" aria-hidden />
            </div>
          ) : (
            <Card><Empty icon={<FileVideo className="size-6" />} title="No video selected">The file is read by your browser for preview and sent only to the local backend at localhost for analysis.</Empty></Card>
          )}

          {res && (
            <Card title="Timeline" subtitle={`${res.stream?.mode ?? mode} mode · threshold ${pct(res.stream?.threshold, 0)}`} actions={<Toggle label="Landmark overlay" checked={overlay} onChange={setOverlay} />}>
              {res.stream ? (
                <>
                  <div className="relative h-10 rounded-lg bg-ink-900">
                    {res.stream.commits.map((c, i) => (
                      <button key={i} className="absolute top-1 -translate-x-1/2 rounded-md border border-teal/50 bg-teal-soft px-1.5 py-0.5 font-mono text-2xs text-teal"
                        style={{ left: `${Math.min(97, (c.t / dur) * 100)}%` }} onClick={() => { if (videoRef.current) videoRef.current.currentTime = c.t / 1000; }}>
                        {c.gloss.toUpperCase()}
                      </button>
                    ))}
                  </div>
                  <ul className="mt-4 flex flex-col gap-2">
                    {res.stream.segments.length === 0 && <li className="text-sm text-fg-3">No sign segments detected (hands were not raised long enough).</li>}
                    {res.stream.segments.map((s, i) => (
                      <li key={i} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-ink-700 px-3 py-2 text-sm">
                        <span className="flex items-center gap-2">
                          <span className="font-mono text-xs text-fg-3">{(s.end_t / 1000).toFixed(2)} s</span>
                          <span className="font-mono text-fg">{s.top?.[0]?.gloss.toUpperCase() ?? "—"}</span>
                          <span className="font-mono text-xs text-fg-3">{pct(s.top?.[0]?.prob, 0)}</span>
                        </span>
                        <span className="flex items-center gap-2 text-xs text-fg-3">
                          {s.frames} frames {s.agreement && `· agree ${s.agreement}`}
                          <Badge tone={s.accepted ? "teal" : "warn"}>{s.accepted ? "recognised" : REJECT_REASON[s.reason] ?? s.reason}</Badge>
                        </span>
                      </li>
                    ))}
                  </ul>
                </>
              ) : <p className="text-sm text-fg-3">{res.message}</p>}
            </Card>
          )}
        </div>

        <div className="flex flex-col gap-4">
          {res?.model?.is_synthetic && <Notice tone="warn">Active model is a synthetic test model — predictions are meaningless for real signing.</Notice>}
          <Card title="Whole-clip classification" subtitle="Treats the entire file as one isolated sign">
            {res?.whole_clip?.top ? (
              <>
                <div className="mb-4 font-mono text-3xl font-medium text-fg">{res.whole_clip.top[0].gloss.toUpperCase()}</div>
                <HBars items={res.whole_clip.top.map((t) => ({ label: t.gloss.toUpperCase(), value: t.prob }))} highlightFirst />
                <p className="mt-3 text-xs text-fg-3">Model time {ms(res.whole_clip.inference_ms)}. A low top score means the clip is unlike any training sign.</p>
              </>
            ) : res?.whole_clip?.error ? <p className="text-sm text-danger">{res.whole_clip.error}</p>
              : <p className="text-sm text-fg-3">{res?.message ?? "Run an analysis to see predictions."}</p>}
          </Card>
          {res && (
            <Card title="Processing">
              <Kv k="Frames" v={`${res.frames} @ ${res.fps.toFixed(1)} fps`} />
              <Kv k="Resolution" v={`${res.width}×${res.height}`} />
              <Kv k="Duration" v={`${res.duration_s.toFixed(2)} s`} />
              <Kv k="Landmark extraction" v={`${res.extract_seconds.toFixed(1)} s (${res.extract_ms_per_frame.toFixed(0)} ms/frame)`} />
              <Kv k="Body detected" v={pct(res.quality.pose_rate, 0)} />
              <Kv k="Any hand detected" v={pct(res.quality.any_hand_rate, 0)} />
              <Kv k="Face detected" v={pct(res.quality.face_rate, 0)} />
              {!res.quality.usable && <p className="mt-2 text-xs text-warn">Low detection quality — results are unreliable.</p>}
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
