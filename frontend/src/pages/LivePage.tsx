import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Camera, CameraOff, Copy, Eraser, Hand, Pause, Play, RotateCcw, Settings2, ShieldCheck, Square, User, Smile,
} from "lucide-react";
import { IS_STATIC } from "../api/client";
import { useLocalStorage } from "../lib/hooks";
import { ms, pct } from "../lib/format";
import { DEFAULT_SETTINGS, type LiveSettings } from "../recognition/live";
import { REJECT_REASON, STATUS, TONE_CLASS } from "../recognition/status";
import { useLiveRecognition } from "../recognition/useLiveRecognition";
import { Badge, Button, Card, Field, Notice, Segmented, Toggle, cx, inputCls } from "../components/ui";
import { ConfidenceRing } from "../components/charts/ConfidenceRing";
import { HBars } from "../components/charts/Bars";

function DetectionChip({ on, icon, label }: { on: boolean; icon: React.ReactNode; label: string }) {
  return (
    <span className={cx("inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium",
      on ? "border-teal/40 bg-teal-soft/70 text-teal" : "border-ink-600 bg-ink-900/80 text-fg-3")}
      aria-label={`${label} ${on ? "detected" : "not detected"}`}>
      {icon}
      {label}
    </span>
  );
}

export default function LivePage() {
  const [settings, setSettings] = useLocalStorage<LiveSettings>("live-settings-v1", DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const live = useLiveRecognition(settings);
  const { status, camera, recognizing, paused, model, provisional, lastResult, transcript, detection, metrics, server, actions } = live;
  const meta = STATUS[status];
  const threshold = settings.engine.threshold ?? server?.threshold ?? model?.threshold ?? null;

  const top = provisional?.[0] ?? null;
  const inProgress = status === "collecting" || status === "analyzing";
  const settled = !inProgress && !!lastResult?.top?.length;
  const accepted = settled && !!lastResult?.accepted;
  const headline = settled ? lastResult!.top![0] : top;
  const headLabel = inProgress ? (top ? "provisional — sign in progress" : "collecting") : settled ? (accepted ? (status === "recognized" ? "recognised gloss" : "last recognised") : "last attempt — rejected") : "gloss";
  const toneRing = accepted ? "teal" : settled ? "muted" : "accent";
  const committed = accepted;

  const setEngine = (patch: Partial<LiveSettings["engine"]>) => setSettings({ ...settings, engine: { ...settings.engine, ...patch } });
  const text = transcript.map((t) => t.gloss.toUpperCase()).join(" ");

  return (
    <div className="fade-in">
      <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.14em] text-accent">Live · isolated sign recognition</div>
          <h1 className="text-2xl font-semibold tracking-tight">Live Sign Recognition</h1>
          <p className="mt-1 max-w-2xl text-sm text-fg-2">
            Sign one WLASL100 word at a time: raise your hands, perform the sign, then lower your hands. The output is an ASL gloss label, not a translation.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-fg-3">
          <ShieldCheck className="size-4 text-teal" aria-hidden />
          {IS_STATIC ? "Video, landmarks and the model all stay in your browser" : "Video stays in your browser · only landmark coordinates are sent to the local server"}
        </div>
      </div>

      {model?.is_synthetic && (
        <div className="mb-4">
          <Notice tone="warn" title="The active model was trained on synthetic test data">
            It verifies the pipeline only and cannot recognise real ASL. Train a model on WLASL100 landmarks in the Model Laboratory.
          </Notice>
        </div>
      )}

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
        {/* ------------------------------------------------ camera stage */}
        <div className="flex min-w-0 flex-col gap-3">
          <div className="relative mx-auto w-full overflow-hidden rounded-2xl border border-ink-700 bg-black"
            style={{
              aspectRatio: live.videoSize ? `${live.videoSize.w} / ${live.videoSize.h}` : "16 / 9",
              maxWidth: live.videoSize ? `calc(74vh * ${(live.videoSize.w / live.videoSize.h).toFixed(4)})` : undefined,
            }}>
            <video ref={live.videoRef} playsInline muted className={cx("absolute inset-0 size-full object-contain", settings.mirror && "-scale-x-100", camera !== "active" && "opacity-0")} aria-label="Webcam preview" />
            <canvas ref={live.canvasRef} className="pointer-events-none absolute inset-0 size-full" style={{ objectFit: "contain" }} aria-hidden />

            {camera !== "active" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[radial-gradient(ellipse_at_center,#15203a_0%,#0b0d10_70%)] p-6 text-center">
                <div className="flex size-14 items-center justify-center rounded-2xl border border-ink-600 bg-ink-850">
                  {camera === "error" ? <CameraOff className="size-6 text-danger" /> : <Camera className="size-6 text-accent" />}
                </div>
                <div>
                  <div className="text-base font-medium">{camera === "error" ? "Camera unavailable" : camera === "starting" ? "Waiting for camera permission…" : "Camera is off"}</div>
                  <p className="mx-auto mt-1 max-w-md text-sm text-fg-3">
                    {live.cameraError ?? "Your browser will ask for permission. Frames are processed locally and never saved or uploaded."}
                  </p>
                </div>
                {camera !== "starting" && <Button variant="primary" size="lg" icon={<Camera className="size-4" />} onClick={actions.startCamera}>Start camera</Button>}
              </div>
            )}

            {/* status pill + camera indicator */}
            <div className="absolute left-3 top-3 flex flex-wrap items-center gap-2">
              <span data-testid="live-status" className={cx("inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium backdrop-blur", TONE_CLASS[meta.tone])} role="status" aria-live="polite">
                <span className={cx("size-1.5 rounded-full bg-current", (meta.tone === "busy" || meta.tone === "active") && "pulse-dot")} />
                {meta.label}
              </span>
              {camera === "active" && (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-danger/50 bg-black/60 px-2.5 py-1 text-[11px] font-medium text-danger">
                  <span className="size-1.5 rounded-full bg-danger pulse-dot" /> Camera on
                </span>
              )}
            </div>
            {camera === "active" && (
              <div className="absolute right-3 top-3 flex flex-wrap justify-end gap-1.5">
                <DetectionChip on={detection.pose} icon={<User className="size-3" />} label="Body" />
                <DetectionChip on={detection.left} icon={<Hand className="size-3 -scale-x-100" />} label="Left" />
                <DetectionChip on={detection.right} icon={<Hand className="size-3" />} label="Right" />
                <DetectionChip on={detection.face} icon={<Smile className="size-3" />} label="Face" />
              </div>
            )}
            {camera === "active" && (
              <div className="absolute inset-x-3 bottom-3 flex flex-wrap gap-x-4 gap-y-1 rounded-lg bg-black/55 px-3 py-1.5 font-mono text-[11px] text-fg-2 backdrop-blur">
                <span>landmarks {metrics.fps ? `${metrics.fps.toFixed(1)} fps` : "—"}</span>
                <span>mediapipe {ms(metrics.mpMs)}</span>
                {recognizing && <span>server {ms(metrics.serverMs)}</span>}
                {recognizing && <span>model {ms(metrics.inferenceMs)}</span>}
                {recognizing && <span>round-trip {ms(metrics.e2eMs)}</span>}
                {live.delegate && <span className="text-fg-3">{live.delegate}</span>}
              </div>
            )}
          </div>

          {/* controls */}
          <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-ink-700 bg-ink-850 p-2.5">
            {camera === "active" ? (
              <Button variant="secondary" icon={<CameraOff className="size-4" />} onClick={actions.stopCamera}>Stop camera</Button>
            ) : (
              <Button variant="secondary" icon={<Camera className="size-4" />} onClick={actions.startCamera} disabled={camera === "starting"}>Start camera</Button>
            )}
            <span className="mx-1 h-6 w-px bg-ink-700" aria-hidden />
            {!recognizing ? (
              <Button variant="primary" icon={<Play className="size-4" />} onClick={actions.startRecognition} disabled={camera !== "active" || live.mediapipe !== "ready"}>
                Start recognition
              </Button>
            ) : paused ? (
              <Button variant="primary" icon={<Play className="size-4" />} onClick={actions.resume}>Resume</Button>
            ) : (
              <Button variant="secondary" icon={<Pause className="size-4" />} onClick={actions.pause}>Pause</Button>
            )}
            {recognizing && <Button variant="ghost" icon={<Square className="size-4" />} onClick={actions.stopRecognition}>Stop</Button>}
            <Button variant="ghost" icon={<RotateCcw className="size-4" />} onClick={actions.resetSequence} disabled={!recognizing}>Reset sequence</Button>
            <div className="ml-auto">
              <Button variant={showSettings ? "secondary" : "ghost"} icon={<Settings2 className="size-4" />} onClick={() => setShowSettings((v) => !v)} aria-expanded={showSettings}>
                Settings
              </Button>
            </div>
          </div>

          {showSettings && (
            <Card title="Recognition settings" subtitle="Saved in this browser. Engine changes apply immediately.">
              <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                <Field label="Recognition mode" hint={settings.engine.mode === "segment" ? "Classifies each complete sign between hand raise and hand drop (matches training clips)." : "Classifies a sliding window continuously with smoothing."}>
                  <Segmented ariaLabel="Recognition mode" value={settings.engine.mode} onChange={(v) => setEngine({ mode: v })}
                    options={[{ value: "segment", label: "Sign boundaries" }, { value: "sliding", label: "Sliding window" }]} />
                </Field>
                <Field label={`Confidence threshold: ${settings.engine.threshold === null ? `model default (${pct(model?.threshold ?? null, 0)})` : pct(settings.engine.threshold, 0)}`}
                  hint="Below this the result is shown as uncertain and no text is emitted.">
                  <div className="flex items-center gap-2">
                    <input type="range" min={0.2} max={0.95} step={0.05} value={settings.engine.threshold ?? model?.threshold ?? 0.6}
                      onChange={(e) => setEngine({ threshold: Number(e.target.value) })} className="w-full accent-[var(--color-accent)]" aria-label="Confidence threshold" />
                    <Button size="sm" variant="ghost" onClick={() => setEngine({ threshold: null })}>Default</Button>
                  </div>
                </Field>
                <Field label={`Minimum margin over runner-up: ${pct(settings.engine.min_margin, 0)}`}>
                  <input type="range" min={0} max={0.5} step={0.05} value={settings.engine.min_margin} onChange={(e) => setEngine({ min_margin: Number(e.target.value) })} className="accent-[var(--color-accent)]" aria-label="Minimum margin" />
                </Field>
                <Field label={`Hands-down time that ends a sign: ${settings.engine.end_absent_ms} ms`}>
                  <input type="range" min={200} max={1200} step={50} value={settings.engine.end_absent_ms} onChange={(e) => setEngine({ end_absent_ms: Number(e.target.value) })} className="accent-[var(--color-accent)]" aria-label="Sign end delay" />
                </Field>
                <Field label={`Cooldown between words: ${settings.engine.cooldown_ms} ms`}>
                  <input type="range" min={0} max={3000} step={100} value={settings.engine.cooldown_ms} onChange={(e) => setEngine({ cooldown_ms: Number(e.target.value) })} className="accent-[var(--color-accent)]" aria-label="Cooldown" />
                </Field>
                <Field label="Processing resolution" hint="Lower values are faster on limited hardware.">
                  <select className={inputCls} value={settings.procWidth} onChange={(e) => setSettings({ ...settings, procWidth: Number(e.target.value) })}>
                    <option value={0}>Full camera resolution</option>
                    <option value={960}>960 px wide</option>
                    <option value={640}>640 px wide (recommended)</option>
                    <option value={480}>480 px wide (low-end)</option>
                  </select>
                </Field>
                <Field label="MediaPipe processor" hint="Applies the next time the camera starts. Auto tries the GPU, then falls back to CPU.">
                  <select className={inputCls} value={settings.delegate} onChange={(e) => setSettings({ ...settings, delegate: e.target.value as LiveSettings["delegate"] })}>
                    <option value="auto">Auto (GPU, fallback CPU)</option>
                    <option value="GPU">GPU (WebGL)</option>
                    <option value="CPU">CPU (WebAssembly)</option>
                  </select>
                </Field>
                <Field label="Landmark rate cap">
                  <select className={inputCls} value={settings.targetFps} onChange={(e) => setSettings({ ...settings, targetFps: Number(e.target.value) })}>
                    {[30, 25, 20, 15].map((f) => <option key={f} value={f}>{f} fps</option>)}
                  </select>
                </Field>
                <div className="flex flex-col gap-2.5">
                  <span className="text-xs font-medium text-fg-2">Overlay</span>
                  <Toggle label="Hands" checked={settings.overlay.hands} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, hands: v } })} />
                  <Toggle label="Body pose" checked={settings.overlay.pose} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, pose: v } })} />
                  <Toggle label="Face subset" checked={settings.overlay.face} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, face: v } })} />
                  <Toggle label="Wrist trails" checked={settings.overlay.trail} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, trail: v } })} />
                  <Toggle label="Mirror preview" checked={settings.mirror} onChange={(v) => setSettings({ ...settings, mirror: v })} />
                </div>
              </div>
            </Card>
          )}
        </div>

        {/* ------------------------------------------------ prediction column */}
        <div className="flex flex-col gap-4">
          <Card title="Current sign" subtitle={meta.help}>
            <div className="flex items-center gap-5">
              <ConfidenceRing value={headline ? headline.prob : null} threshold={threshold ?? undefined} tone={toneRing} />
              <div className="min-w-0">
                <div className="text-[11px] uppercase tracking-[0.14em] text-fg-3">{headLabel}</div>
                <div data-testid="current-gloss" className={cx("truncate font-mono text-3xl font-medium tracking-tight", committed ? "text-teal" : settled ? "text-fg-3" : "text-fg")}>
                  {headline ? headline.gloss.toUpperCase() : "—"}
                </div>
                {settled && lastResult && (
                  <div className="mt-1 text-xs text-fg-3">
                    {lastResult.accepted ? "accepted" : `rejected: ${REJECT_REASON[lastResult.reason] ?? lastResult.reason}`}
                    {lastResult.agreement && ` · crops agree ${lastResult.agreement}`}
                    {lastResult.frames ? ` · ${lastResult.frames} frames` : ""}
                  </div>
                )}
              </div>
            </div>
            {provisional && provisional.length > 0 && (
              <div className="mt-5">
                <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-fg-3">Top-5 class scores</div>
                <HBars items={provisional.map((p) => ({ label: p.gloss.toUpperCase(), value: p.prob }))} highlightFirst={!!committed} />
              </div>
            )}
            {recognizing && server && (server.segment_frames ?? 0) > 0 && (
              <div className="mt-4 font-mono text-[11px] text-fg-3">collecting · {server.segment_frames} frames · {((server.segment_ms ?? 0) / 1000).toFixed(1)} s</div>
            )}
          </Card>

          <Card title="Recognised text" subtitle="Space-separated ASL glosses — not an English translation"
            actions={<>
              <Button size="sm" variant="ghost" icon={<Copy className="size-3.5" />} disabled={!transcript.length} onClick={() => navigator.clipboard?.writeText(text)}>Copy</Button>
              <Button size="sm" variant="ghost" icon={<Eraser className="size-3.5" />} disabled={!transcript.length} onClick={actions.clearText}>Clear</Button>
            </>}>
            <div data-testid="transcript" className="flex min-h-16 flex-wrap content-start gap-2" aria-live="polite">
              {transcript.length === 0 ? (
                <span className="text-sm text-fg-3">Recognised signs appear here.</span>
              ) : (
                transcript.map((t) => (
                  <span key={t.id} className="fade-in inline-flex items-baseline gap-1.5 rounded-lg border border-teal/30 bg-teal-soft/50 px-2.5 py-1">
                    <span className="font-mono text-sm font-medium text-teal">{t.gloss.toUpperCase()}</span>
                    <span className="font-mono text-[10px] text-fg-3">{Math.round(t.score * 100)}</span>
                  </span>
                ))
              )}
            </div>
          </Card>

          <Card title="Recent signs">
            {transcript.length === 0 ? (
              <p className="text-sm text-fg-3">No recognitions in this session yet.</p>
            ) : (
              <ul className="flex max-h-56 flex-col gap-1 overflow-y-auto">
                {[...transcript].reverse().slice(0, 20).map((t) => (
                  <li key={t.id} className="flex items-center justify-between rounded-md px-2 py-1.5 text-sm hover:bg-ink-800">
                    <span className="font-mono text-fg">{t.gloss.toUpperCase()}</span>
                    <span className="flex items-center gap-3 font-mono text-xs text-fg-3">
                      {pct(t.score, 0)}
                      <span>{new Date(t.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Model">
            {live.modelAvailable === false ? (
              live.modelMessage && !live.modelMessage.startsWith("No active model") ? (
                <p className="text-sm text-fg-2">{live.modelMessage}</p>
              ) : (
                <p className="text-sm text-fg-2">No active model. <Link className="text-accent underline underline-offset-2" to="/lab">Train one in the Model Laboratory</Link> and activate it.</p>
              )
            ) : model ? (
              <div className="flex flex-col gap-1.5 text-sm">
                <div className="flex items-center justify-between gap-2"><span className="truncate text-fg">{model.name}</span>{model.is_synthetic && <Badge tone="warn">synthetic</Badge>}</div>
                <div className="font-mono text-xs text-fg-3">{model.architecture.toUpperCase()} · {model.num_classes} classes · T={model.seq_len} · split {model.split}</div>
                <div className="font-mono text-xs text-fg-3">validation top-1 {pct(model.val_top1)} · threshold {pct(model.threshold, 0)}</div>
              </div>
            ) : (
              <p className="text-sm text-fg-3">Start recognition to connect to the active model.</p>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
