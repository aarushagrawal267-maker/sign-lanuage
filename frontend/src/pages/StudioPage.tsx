import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Camera, CameraOff, Circle, ListOrdered, Pause, Play } from "lucide-react";
import { api, type SystemInfo, type FeatureSpec, type LandmarkClip, type RawFrame, type TopK, type VideoRow } from "../api/client";
import { DEFAULT_FACE_IDX, loadHolistic, toRawFrame } from "../mediapipe/holistic";
import { COLORS, HAND_CONNECTIONS, drawFrame } from "../mediapipe/draw";
import { useApi } from "../lib/hooks";
import { pct } from "../lib/format";
import { Button, Card, Empty, ErrorText, Field, Notice, PageHeader, Segmented, Toggle, cx, inputCls } from "../components/ui";
import { LineChart } from "../components/charts/LineChart";
import { HBars } from "../components/charts/Bars";

type Clip = { width: number; height: number; frames: RawFrame[]; label: string };
type Comp = "left" | "right" | "pose" | "face";
const POSE_NAMES: Record<number, string> = { 0: "nose", 11: "L shoulder", 12: "R shoulder", 13: "L elbow", 14: "R elbow", 15: "L wrist", 16: "R wrist", 23: "L hip", 24: "R hip" };
const HAND_NAMES = ["wrist", "thumb cmc", "thumb mcp", "thumb ip", "thumb tip", "index mcp", "index pip", "index dip", "index tip", "middle mcp", "middle pip", "middle dip", "middle tip", "ring mcp", "ring pip", "ring dip", "ring tip", "pinky mcp", "pinky pip", "pinky dip", "pinky tip"];

/** Records a short landmark clip from the webcam, entirely in the browser. */
function useRecorder() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<"off" | "starting" | "ready" | "recording" | "error">("off");
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<RawFrame | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const rec = useRef<RawFrame[] | null>(null);
  const loop = useRef<number | null>(null);
  const faceIdx = useRef(DEFAULT_FACE_IDX);
  const lastTs = useRef(0);
  const t0 = useRef(0);

  useEffect(() => { api.get<FeatureSpec>("/api/features/spec").then((s) => (faceIdx.current = s.raw.face_indices)).catch(() => {}); }, []);

  const stop = useCallback(() => {
    if (loop.current) cancelAnimationFrame(loop.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    setState("off");
    setLive(null);
  }, []);
  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    setState("starting");
    setError(null);
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 540 } }, audio: false });
      const v = videoRef.current!;
      v.srcObject = stream.current;
      await v.play();
      const { landmarker } = await loadHolistic();
      setState("ready");
      let last = 0;
      const tick = () => {
        const now = performance.now();
        if (v.readyState >= 2 && now - last >= 38) {
          last = now;
          const ts = Math.max(Math.round(now), lastTs.current + 1);
          lastTs.current = ts;
          const f = toRawFrame(landmarker.detectForVideo(v, ts), ts, faceIdx.current);
          setLive(f);
          if (rec.current) rec.current.push({ ...f, t: ts - t0.current });
        }
        loop.current = requestAnimationFrame(tick);
      };
      tick();
    } catch (e) {
      setState("error");
      setError((e as DOMException)?.name === "NotAllowedError" ? "Camera permission denied." : String(e));
    }
  }, []);

  const record = useCallback((seconds: number, cb: (c: Clip) => void) => {
    rec.current = [];
    t0.current = performance.now();
    setState("recording");
    window.setTimeout(() => {
      const frames = rec.current ?? [];
      rec.current = null;
      setState("ready");
      const v = videoRef.current!;
      cb({ width: v.videoWidth, height: v.videoHeight, frames, label: `webcam recording · ${frames.length} frames` });
    }, seconds * 1000);
  }, []);

  return { videoRef, state, error, live, start, stop, record };
}

function Hand3D({ hand, color, yaw, pitch }: { hand: number[][] | null; color: string; yaw: number; pitch: number }) {
  const size = 220;
  if (!hand) return <div className="flex h-[220px] items-center justify-center rounded-lg border border-ink-700 bg-ink-900 text-xs text-fg-3">hand not detected in this frame</div>;
  const w = hand[0];
  const rel = hand.map((p) => [p[0] - w[0], p[1] - w[1], p[2] - w[2]]);
  const scale = Math.max(1e-6, ...rel.map((p) => Math.hypot(p[0], p[1], p[2])));
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const proj = rel.map(([x, y, z]) => {
    const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
    const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
    return { x: size / 2 + (x1 / scale) * size * 0.4, y: size * 0.75 + (y1 / scale) * size * 0.4 - size * 0.25, z: z2 };
  });
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full rounded-lg border border-ink-700 bg-ink-900" role="img" aria-label="3D hand landmark view">
      {HAND_CONNECTIONS.map(([a, b], i) => <line key={i} x1={proj[a].x} y1={proj[a].y} x2={proj[b].x} y2={proj[b].y} stroke={color} strokeWidth={2} strokeLinecap="round" opacity={0.85} />)}
      {proj.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={i === 0 ? 4.5 : 3} fill={i === 0 ? "#fff" : color} stroke="var(--color-ink-900)" strokeWidth={1.5} />)}
    </svg>
  );
}

function PresenceStrip({ frames, cursor }: { frames: RawFrame[]; cursor: number }) {
  const rows: { key: Comp; label: string; color: string }[] = [
    { key: "pose", label: "body", color: "var(--color-fg-2)" }, { key: "left", label: "left hand", color: COLORS.left },
    { key: "right", label: "right hand", color: COLORS.right }, { key: "face", label: "face", color: COLORS.face },
  ];
  return (
    <div className="flex flex-col gap-1.5">
      {rows.map((r) => (
        <div key={r.key} className="grid grid-cols-[5.5rem_1fr_3rem] items-center gap-3 text-2xs">
          <span className="text-fg-3">{r.label}</span>
          <div className="relative flex h-3 overflow-hidden rounded-sm bg-ink-900">
            {frames.map((f, i) => <span key={i} className="h-full flex-1" style={{ background: f[r.key] ? r.color : "transparent", opacity: 0.85 }} />)}
            <span className="absolute inset-y-0 w-px bg-white" style={{ left: `${(cursor / Math.max(1, frames.length - 1)) * 100}%` }} />
          </div>
          <span className="text-right font-mono text-fg-2">{pct(frames.filter((f) => f[r.key]).length / Math.max(1, frames.length), 0)}</span>
        </div>
      ))}
    </div>
  );
}

export default function StudioPage() {
  const [source, setSource] = useState<"webcam" | "dataset">("webcam");
  const [clip, setClip] = useState<Clip | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [show, setShow] = useState({ pose: true, hands: true, face: true, trail: true });
  const [comp, setComp] = useState<Comp>("right");
  const [yaw, setYaw] = useState(0.4);
  const [pitch, setPitch] = useState(0.2);
  const [seconds, setSeconds] = useState(3);
  const [pred, setPred] = useState<{ top?: TopK[]; error?: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const rec = useRecorder();
  const { data: sys } = useApi<SystemInfo>("/api/system");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vids = useApi<{ items: VideoRow[] }>(source === "dataset" ? "/api/dataset/videos?status=valid&limit=500" : null);
  const withLm = (vids.data?.items ?? []).filter((v) => v.landmarks?.status === "ok");

  const frame = clip ? clip.frames[Math.min(cursor, clip.frames.length - 1)] : rec.live;

  // playback
  useEffect(() => {
    if (!playing || !clip) return;
    const id = window.setInterval(() => setCursor((c) => (c + 1 >= clip.frames.length ? (setPlaying(false), c) : c + 1)), 40);
    return () => window.clearInterval(id);
  }, [playing, clip]);

  // draw
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const w = c.clientWidth, h = c.clientHeight, dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const trail = clip && show.trail
      ? clip.frames.slice(Math.max(0, cursor - 30), cursor + 1).flatMap((f) => (["left", "right"] as const).filter((s) => f[s]).map((s) => ({ x: f[s]![0][0], y: f[s]![0][1], side: s })))
      : undefined;
    drawFrame(ctx, frame ?? null, w, h, { pose: show.pose, hands: show.hands, face: show.face, mirror: false, trail });
  }, [frame, show, clip, cursor]);

  const loadDataset = async (id: string) => {
    setErr(null);
    setPred(null);
    try {
      const d = await api.get<LandmarkClip>(`/api/dataset/videos/${id}/landmarks`);
      setClip({ width: d.width, height: d.height, frames: d.frames, label: `WLASL ${id} · ${d.gloss?.toUpperCase()} · signer ${d.signer_id}` });
      setCursor(0);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const classify = async () => {
    if (!clip) return;
    setPred(null);
    try {
      setPred(await api.post<{ top: TopK[] }>("/api/analyze/landmarks", { width: clip.width, height: clip.height, frames: clip.frames }));
    } catch (e) {
      setPred({ error: e instanceof Error ? e.message : String(e) });
    }
  };

  const traj = useMemo(() => {
    if (!clip) return null;
    const x = clip.frames.map((f) => +(f.t / 1000).toFixed(3));
    const wrist = (side: "left" | "right", f: RawFrame) => f[side]?.[0] ?? null;
    const height = (side: "left" | "right") => clip.frames.map((f) => { const w = wrist(side, f); return w ? 1 - w[1] : null; });
    const horiz = (side: "left" | "right") => clip.frames.map((f) => wrist(side, f)?.[0] ?? null);
    return { x, ly: height("left"), ry: height("right"), lx: horiz("left"), rx: horiz("right") };
  }, [clip]);

  const coords: number[][] | null = frame ? (frame[comp] as number[][] | null) : null;
  const aspect = clip ? `${clip.width} / ${clip.height}` : "16 / 9";

  return (
    <div className="fade-in">
      <PageHeader eyebrow="MediaPipe Holistic" title="Landmark visualisation studio"
        description="Inspect exactly what the model sees: hand, body and facial landmarks, their trajectories and raw coordinate values over time." />

      <div className="mb-5 flex flex-wrap items-center gap-3">
        <Segmented ariaLabel="Source" value={source} onChange={(v) => { setSource(v); setClip(null); setPred(null); }} options={[{ value: "webcam", label: "Webcam recording" }, { value: "dataset", label: "Dataset clip" }]} />
        {source === "dataset" && (
          <select className={inputCls + " w-80"} defaultValue="" onChange={(e) => e.target.value && loadDataset(e.target.value)} aria-label="Dataset clip">
            <option value="">{withLm.length ? "Choose a clip with extracted landmarks" : "No clips with landmarks yet"}</option>
            {withLm.map((v) => <option key={v.video_id} value={v.video_id}>{v.gloss.toUpperCase()} · {v.video_id} · signer {v.signer_id}</option>)}
          </select>
        )}
        {source === "webcam" && (
          <>
            {rec.state === "off" || rec.state === "error" ? (
              <Button variant="secondary" icon={<Camera className="size-4" />} onClick={rec.start}>Start camera</Button>
            ) : (
              <Button variant="ghost" icon={<CameraOff className="size-4" />} onClick={rec.stop}>Stop camera</Button>
            )}
            <Field label=""><select className={inputCls + " w-28"} value={seconds} onChange={(e) => setSeconds(Number(e.target.value))} aria-label="Recording length">{[2, 3, 4, 5].map((s) => <option key={s} value={s}>{s} s</option>)}</select></Field>
            <Button variant="primary" icon={<Circle className="size-4" />} disabled={rec.state !== "ready"} loading={rec.state === "recording"}
              onClick={() => rec.record(seconds, (c) => { setClip(c); setCursor(0); setPred(null); })}>
              {rec.state === "recording" ? "Recording…" : "Record clip"}
            </Button>
            {rec.state !== "off" && <span className="inline-flex items-center gap-1.5 text-xs text-danger"><span className="size-1.5 rounded-full bg-danger pulse-dot" />camera on (local only)</span>}
          </>
        )}
      </div>
      <ErrorText error={err ?? rec.error} />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_400px]">
        <div className="flex min-w-0 flex-col gap-4">
          <div className="relative overflow-hidden rounded-xl border border-ink-700 bg-ink-950" style={{ aspectRatio: aspect }}>
            <video ref={rec.videoRef} playsInline muted className={cx("absolute inset-0 size-full object-contain opacity-30", (clip || rec.state === "off") && "hidden")} />
            <canvas ref={canvasRef} className="absolute inset-0 size-full" aria-label="Landmark skeleton view" />
            {!frame && <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-fg-3">{source === "webcam" ? "Start the camera and record a short clip." : "Choose a dataset clip with extracted landmarks."}</div>}
            {clip && <div className="absolute left-3 top-3 rounded-md bg-black/60 px-2.5 py-1 font-mono text-2xs text-fg-2">{clip.label}</div>}
          </div>

          {clip && (
            <Card>
              <div className="flex items-center gap-3">
                <Button size="sm" variant="secondary" icon={playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />} onClick={() => { if (!playing && cursor >= clip.frames.length - 1) setCursor(0); setPlaying(!playing); }}>{playing ? "Pause" : "Play"}</Button>
                <input type="range" min={0} max={clip.frames.length - 1} value={cursor} onChange={(e) => { setPlaying(false); setCursor(Number(e.target.value)); }} className="flex-1 accent-[var(--color-accent)]" aria-label="Frame" />
                <span className="w-28 text-right font-mono text-xs text-fg-2">{cursor + 1}/{clip.frames.length} · {(clip.frames[cursor].t / 1000).toFixed(2)}s</span>
              </div>
              <div className="mt-4"><PresenceStrip frames={clip.frames} cursor={cursor} /></div>
              <div className="mt-4 flex flex-wrap gap-4">
                {(["pose", "hands", "face", "trail"] as const).map((k) => <Toggle key={k} label={k === "trail" ? "trajectories" : k} checked={show[k]} onChange={(v) => setShow({ ...show, [k]: v })} />)}
              </div>
            </Card>
          )}

          {clip && traj && (
            <Card title="Wrist trajectories" subtitle="Normalised image coordinates over time (height: 1 = top of frame)">
              <div className="grid gap-6 lg:grid-cols-2">
                <div>
                  <div className="mb-1 text-xs text-fg-2">Vertical position</div>
                  <LineChart ariaLabel="Wrist height over time" x={traj.x} xLabel="s" yDomain={[0, 1]} height={170} xFormat={(v) => v.toFixed(1)}
                    series={[{ key: "l", label: "left wrist", color: "var(--color-series-3)", values: traj.ly }, { key: "r", label: "right wrist", color: "var(--color-series-1)", values: traj.ry }]} />
                </div>
                <div>
                  <div className="mb-1 text-xs text-fg-2">Horizontal position</div>
                  <LineChart ariaLabel="Wrist horizontal position over time" x={traj.x} xLabel="s" yDomain={[0, 1]} height={170} xFormat={(v) => v.toFixed(1)}
                    series={[{ key: "l", label: "left wrist", color: "var(--color-series-3)", values: traj.lx }, { key: "r", label: "right wrist", color: "var(--color-series-1)", values: traj.rx }]} />
                </div>
              </div>
            </Card>
          )}
        </div>

        <div className="flex flex-col gap-4">
          <Card title="3D hand view" subtitle="Wrist-centred; MediaPipe z is relative depth">
            <div className="mb-3"><Segmented ariaLabel="Hand" value={comp === "left" ? "left" : "right"} onChange={(v) => setComp(v)} options={[{ value: "left", label: "Left hand" }, { value: "right", label: "Right hand" }]} /></div>
            <Hand3D hand={frame ? (comp === "left" ? frame.left : frame.right) : null} color={comp === "left" ? COLORS.left : COLORS.right} yaw={yaw} pitch={pitch} />
            <div className="mt-3 grid grid-cols-2 gap-3">
              <label className="text-2xs text-fg-3">rotate ↔<input type="range" min={-3.14} max={3.14} step={0.02} value={yaw} onChange={(e) => setYaw(Number(e.target.value))} className="w-full accent-[var(--color-accent)]" /></label>
              <label className="text-2xs text-fg-3">tilt ↕<input type="range" min={-1.5} max={1.5} step={0.02} value={pitch} onChange={(e) => setPitch(Number(e.target.value))} className="w-full accent-[var(--color-accent)]" /></label>
            </div>
          </Card>

          <Card title="Coordinates" subtitle="Current frame · normalised x, y (0–1) and relative z"
            actions={<select className={inputCls + " h-8 w-32 text-xs"} value={comp} onChange={(e) => setComp(e.target.value as Comp)} aria-label="Component">
              <option value="right">right hand</option><option value="left">left hand</option><option value="pose">body pose</option><option value="face">face subset</option></select>}>
            {coords ? (
              <div className="max-h-72 overflow-y-auto">
                <table className="w-full font-mono text-2xs">
                  <thead className="sticky top-0 bg-ink-850 text-fg-3"><tr><th className="py-1 text-left font-medium">#</th><th className="py-1 text-left font-medium">point</th><th className="py-1 text-right font-medium">x</th><th className="py-1 text-right font-medium">y</th><th className="py-1 text-right font-medium">z</th>{comp === "pose" && <th className="py-1 text-right font-medium">vis</th>}</tr></thead>
                  <tbody>{coords.map((p, i) => (comp === "pose" && !(i in POSE_NAMES)) ? null : (
                    <tr key={i} className="border-b border-ink-700/40"><td className="py-0.5 text-fg-3">{i}</td>
                      <td className="py-0.5 text-fg-2">{comp === "pose" ? POSE_NAMES[i] : comp === "face" ? `mesh ${DEFAULT_FACE_IDX[i]}` : HAND_NAMES[i]}</td>
                      <td className="py-0.5 text-right">{p[0].toFixed(3)}</td><td className="py-0.5 text-right">{p[1].toFixed(3)}</td><td className="py-0.5 text-right">{p[2].toFixed(3)}</td>
                      {comp === "pose" && <td className="py-0.5 text-right">{(p[3] ?? 1).toFixed(2)}</td>}</tr>
                  ))}</tbody>
                </table>
              </div>
            ) : <p className="text-sm text-fg-3">Not detected in this frame — stored as missing, never filled with invented values.</p>}
          </Card>

          {clip && (
            <Card title="Classify this clip" subtitle="Uses the active model on the same features as live recognition">
              <Button variant="secondary" icon={<ListOrdered className="size-4" />} onClick={classify}>Classify</Button>
              {pred?.top && <div className="mt-4"><HBars items={pred.top.map((t) => ({ label: t.gloss.toUpperCase(), value: t.prob }))} highlightFirst /></div>}
              {pred?.error && <p className="mt-3 text-sm text-danger">{pred.error}</p>}
            </Card>
          )}
          {source === "dataset" && sys?.public_demo && <Notice>Dataset clips need the locally extracted WLASL landmarks; they are not included in the public demo.</Notice>}
          {source === "dataset" && !withLm.length && vids.data && <Notice>No clips have landmarks yet. Extract landmarks on the Dataset page.</Notice>}
          {!clip && source === "dataset" && <Card><Empty title="No clip loaded" /></Card>}
        </div>
      </div>
    </div>
  );
}
