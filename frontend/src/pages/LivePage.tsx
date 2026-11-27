import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Camera, CameraOff, Hand, Pause, Play, RotateCcw, Settings2, ShieldCheck, Smile, Square, User,
} from "lucide-react";
import { IS_STATIC } from "../api/client";
import { useLocalStorage } from "../lib/hooks";
import { ms, pct } from "../lib/format";
import { useT, type StringKey } from "../i18n";
import { useSignLanguage } from "../state/language";
import {
  SETTINGS_KEY, loadLiveSettings, precisionOutOf100, segmentDecision, slidingStrictCertified, wordText,
  type Certification, type LiveSettings, type Policy,
} from "../recognition/live";
import { STATUS, TONE_CLASS } from "../recognition/status";
import { useLiveRecognition } from "../recognition/useLiveRecognition";
import { Badge, Button, Card, Field, Notice, ScrollRegion, Segmented, Toggle, cx, inputCls } from "../components/ui";
import { BilingualWord } from "../components/BilingualWord";
import { ResultAnnouncer, SignResult } from "../components/SignResult";
import { SentenceBar } from "../components/SentenceBar";
import { HBars } from "../components/charts/Bars";

function DetectionChip({ on, icon, label, onLabel, offLabel }: { on: boolean; icon: React.ReactNode; label: string; onLabel: string; offLabel: string }) {
  return (
    <span className={cx("inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium",
      on ? "border-teal/40 bg-teal-soft/70 text-teal" : "border-ink-600 bg-ink-900/80 text-fg-3")}>
      {icon}
      {label}
      <span className="sr-only">{on ? ` ${onLabel}` : ` ${offLabel}`}</span>
    </span>
  );
}

const POLICY_KEYS: Record<Policy, StringKey> = { strict: "live.policy.strict", balanced: "live.policy.balanced", lenient: "live.policy.lenient" };

export default function LivePage() {
  const { t, reasonText } = useT();
  const sign = useSignLanguage();
  // v2 settings (docs/ISL_DESIGN.md 15.29): ISL starts strict, sign by sign, continuous signing off
  const [initialSettings] = useState(loadLiveSettings);
  const [settings, setSettings] = useLocalStorage<LiveSettings>(SETTINGS_KEY, initialSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [spellOn, setSpellOn] = useState(false);
  const [slidingOffered, setSlidingOffered] = useState(false);
  const pack = !sign.isAsl;
  const live = useLiveRecognition(settings, sign.id, { pack, spell: pack && spellOn, sliding: pack && slidingOffered });
  const { status, camera, recognizing, paused, model, provisional, lastResult, resolved, transcript, detection, metrics, server, actions } = live;
  // Certification of every policy, from the server's hello / config_ack. Sliding mode needs the pack's classifier.
  const certs = pack ? live.certs : null;
  useEffect(() => { setSlidingOffered(!!certs?.sliding); }, [certs]);
  const slidingStrict = slidingStrictCertified(certs);
  const continuous = pack && !spellOn && slidingOffered && !!settings.engine.continuous;
  const policy: Policy = settings.engine.policy ?? "strict";
  const policyLabel = t(POLICY_KEYS[policy]);
  const precisionNote = (c: Certification | undefined | null, label = policyLabel): string | null => {
    if (!c) return null;
    const n = precisionOutOf100(c);
    if (n !== null) return t("live.precisionCertified", { policy: label, n });
    return c.rule === "confirm_only" ? t("live.precisionUncertified", { policy: label }) : null;
  };
  const activeCert = certs ? (continuous ? certs.sliding?.[policy] : certs.segment[policy]) : null;
  const meta = STATUS[status];
  const threshold = settings.engine.threshold ?? server?.threshold ?? model?.threshold ?? null;

  // Capabilities: the socket's hello is authoritative (spelling appears once the alphabet is calibrated).
  const { reload: reloadLanguages } = sign;
  useEffect(() => { reloadLanguages(); }, [reloadLanguages]);
  const caps = live.helloLanguage?.capabilities ?? sign.info?.capabilities ?? [];
  const spellAvailable = pack && caps.includes("spelling");
  useEffect(() => { if (!spellAvailable && spellOn) setSpellOn(false); }, [spellAvailable, spellOn]);
  const components = model?.components;
  const smoke = !!(components?.classifier?.smoke || components?.encoder?.smoke);
  const spellExperimental = components?.spelling?.experimental ?? true;

  const decision = segmentDecision(lastResult);
  const inProgress = status === "collecting" || status === "analyzing" || status === "spelling";
  const top = provisional?.[0] ?? null;
  const acceptedItem = decision === "accept" && lastResult
    ? (lastResult.segment_id !== undefined ? transcript.find((x) => x.segment_id === lastResult.segment_id) : transcript[transcript.length - 1]) ?? null
    : null;

  const setEngine = (patch: Partial<LiveSettings["engine"]>) => setSettings({ ...settings, engine: { ...settings.engine, ...patch } });
  const statusLabel = t(`status.${status}` as StringKey);
  const statusHelp = t(`statusHelp.${status}` as StringKey);
  const spellHint = pack ? (
    <Link to="/my-signs#alphabet" className="text-xs text-accent underline underline-offset-2">{t("live.calibrateToSpell")}</Link>
  ) : undefined;

  return (
    <div className="fade-in">
      <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.14em] text-accent [&:lang(hi)]:font-sans">{t("live.eyebrow", { lang: sign.info?.sign_language ?? "" })}</div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("live.title")}</h1>
          <p className="mt-1 max-w-2xl text-sm text-fg-2">
            {pack ? t("live.introIsl") : t("live.introAsl")}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-fg-3">
          <ShieldCheck className="size-4 text-teal" aria-hidden />
          {IS_STATIC ? t("live.privacyStatic") : t("live.privacy")}
        </div>
      </div>

      {model?.is_synthetic && (
        <div className="mb-4">
          <Notice tone="warn" title="The active model was trained on synthetic test data">
            It verifies the pipeline only and cannot recognise real ASL. Train a model on WLASL100 landmarks in the Model Laboratory.
          </Notice>
        </div>
      )}
      {pack && smoke && (
        <div className="mb-4">
          <Notice tone="warn" title={t("live.smokeTitle")}>{t("live.smokeBody")}</Notice>
        </div>
      )}
      {pack && live.modelMessage && recognizing && (
        <div className="mb-4"><Notice tone="warn">{live.modelMessage}</Notice></div>
      )}
      {live.serverError && (
        <div className="mb-4"><Notice tone="danger">{live.serverError}</Notice></div>
      )}

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
        {/* ------------------------------------------------ camera stage */}
        <div className="flex min-w-0 flex-col gap-3">
          <div className={cx("relative mx-auto w-full overflow-hidden rounded-2xl border border-ink-700 bg-black", camera !== "active" && "min-h-72")}
            style={{
              aspectRatio: live.videoSize ? `${live.videoSize.w} / ${live.videoSize.h}` : "16 / 9",
              maxWidth: live.videoSize ? `calc(74vh * ${(live.videoSize.w / live.videoSize.h).toFixed(4)})` : undefined,
            }}>
            <video ref={live.videoRef} playsInline muted className={cx("absolute inset-0 size-full object-contain", settings.mirror && "-scale-x-100", camera !== "active" && "opacity-0")} aria-label={t("live.webcam")} />
            <canvas ref={live.canvasRef} className="pointer-events-none absolute inset-0 size-full" style={{ objectFit: "contain" }} aria-hidden />

            {camera !== "active" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[radial-gradient(ellipse_at_center,#15203a_0%,#0b0d10_70%)] p-6 text-center">
                <div className="flex size-14 items-center justify-center rounded-2xl border border-ink-600 bg-ink-850">
                  {camera === "error" ? <CameraOff className="size-6 text-danger" /> : <Camera className="size-6 text-accent" />}
                </div>
                <div>
                  <div className="text-base font-medium">{camera === "error" ? t("live.cameraUnavailable") : camera === "starting" ? t("live.cameraWaiting") : t("live.cameraOff")}</div>
                  <p className="mx-auto mt-1 max-w-md text-sm text-fg-3">
                    {live.cameraError ?? t("live.cameraNote")}
                  </p>
                </div>
                {camera !== "starting" && <Button variant="primary" size="lg" icon={<Camera className="size-4" />} onClick={actions.startCamera}>{t("live.startCamera")}</Button>}
              </div>
            )}

            {/* status pill + camera indicator */}
            <div className="absolute left-3 top-3 flex flex-wrap items-center gap-2">
              <span data-testid="live-status" data-status={status} className={cx("inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium backdrop-blur", TONE_CLASS[meta.tone])} role="status" aria-live="polite">
                <span className={cx("size-1.5 rounded-full bg-current", (meta.tone === "busy" || meta.tone === "active") && "pulse-dot")} />
                {statusLabel}
              </span>
              {camera === "active" && (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-danger/50 bg-black/60 px-2.5 py-1 text-[11px] font-medium text-danger">
                  <span className="size-1.5 rounded-full bg-danger pulse-dot" /> {t("live.cameraOn")}
                </span>
              )}
            </div>
            {camera === "active" && (
              <div className="absolute right-3 top-3 flex flex-wrap justify-end gap-1.5">
                <DetectionChip on={detection.pose} icon={<User className="size-3" />} label={t("live.body")} onLabel={t("live.detected")} offLabel={t("live.notDetected")} />
                <DetectionChip on={detection.left} icon={<Hand className="size-3 -scale-x-100" />} label={t("live.left")} onLabel={t("live.detected")} offLabel={t("live.notDetected")} />
                <DetectionChip on={detection.right} icon={<Hand className="size-3" />} label={t("live.right")} onLabel={t("live.detected")} offLabel={t("live.notDetected")} />
                <DetectionChip on={detection.face} icon={<Smile className="size-3" />} label={t("live.face")} onLabel={t("live.detected")} offLabel={t("live.notDetected")} />
              </div>
            )}
            {camera === "active" && (
              <div lang="en" className="absolute inset-x-3 bottom-3 flex flex-wrap gap-x-4 gap-y-1 rounded-lg bg-black/55 px-3 py-1.5 font-mono text-[11px] text-fg-2 backdrop-blur">
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
              <Button variant="secondary" icon={<CameraOff className="size-4" />} onClick={actions.stopCamera}>{t("live.stopCamera")}</Button>
            ) : (
              <Button variant="secondary" icon={<Camera className="size-4" />} onClick={actions.startCamera} disabled={camera === "starting"}>{t("live.startCamera")}</Button>
            )}
            <span className="mx-1 h-6 w-px bg-ink-700" aria-hidden />
            {!recognizing ? (
              <Button variant="primary" icon={<Play className="size-4" />} onClick={actions.startRecognition} disabled={camera !== "active" || live.mediapipe !== "ready" || sign.loading}>
                {t("live.startRecognition")}
              </Button>
            ) : paused ? (
              <Button variant="primary" icon={<Play className="size-4" />} onClick={actions.resume}>{t("live.resume")}</Button>
            ) : (
              <Button variant="secondary" icon={<Pause className="size-4" />} onClick={actions.pause}>{t("live.pause")}</Button>
            )}
            {recognizing && <Button variant="ghost" icon={<Square className="size-4" />} onClick={actions.stopRecognition}>{t("live.stop")}</Button>}
            <Button variant="ghost" icon={<RotateCcw className="size-4" />} onClick={actions.resetSequence} disabled={!recognizing}>{t("live.reset")}</Button>
            {pack && (
              <div className="flex items-center gap-2">
                {spellAvailable ? (
                  <Segmented ariaLabel={t("live.inputMode")} value={spellOn ? "spell" : "words"} onChange={(v) => setSpellOn(v === "spell")}
                    options={[{ value: "words", label: t("live.modeWords") }, { value: "spell", label: t("live.modeSpell") }]} />
                ) : spellHint}
              </div>
            )}
            <div className="ml-auto">
              <Button variant={showSettings ? "secondary" : "ghost"} icon={<Settings2 className="size-4" />} onClick={() => setShowSettings((v) => !v)} aria-expanded={showSettings}>
                {t("live.settings")}
              </Button>
            </div>
          </div>

          {showSettings && (
            <Card title={t("live.settingsTitle")} subtitle={t("live.settingsSubtitle")}>
              <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
                {pack ? (
                  <>
                    <Field label={t("live.policy")} group hint={t("live.policyHint")}>
                      <Segmented<Policy> ariaLabel={t("live.policy")} value={policy} onChange={(v) => setEngine({ policy: v })}
                        options={(["strict", "balanced", "lenient"] as Policy[]).map((p) => ({ value: p, label: t(POLICY_KEYS[p]) }))} />
                      {/* the certified worst-source precision of the active policy, never the 0.98 target (critic 15) */}
                      <p data-testid="policy-precision" className="text-xs text-fg-2">
                        {precisionNote(activeCert) ?? t("live.precisionUnknown")}
                      </p>
                    </Field>
                    <Field label={t("live.continuous")} group
                      hint={!certs ? undefined : !slidingOffered ? t("live.continuousUnavailable")
                        : !settings.engine.continuous ? t("live.continuousOff")
                        : slidingStrict ? t("live.continuousCertified") : t("live.continuousUncertified")}>
                      {(!certs || slidingOffered) && (
                        <Toggle label={t("live.continuousToggle")} checked={!!settings.engine.continuous} onChange={(v) => setEngine({ continuous: v })} />
                      )}
                    </Field>
                    <div className="flex flex-col gap-1.5">
                      <span className="text-xs font-medium text-fg-2">{t("nav.mySigns")}</span>
                      <Toggle label={t("live.useMySigns")} checked={settings.engine.use_my_signs ?? true} onChange={(v) => setEngine({ use_my_signs: v })} />
                    </div>
                  </>
                ) : (
                  <div lang="en" className="contents">
                    <Field label="Recognition mode" group hint={settings.engine.mode === "segment" ? "Classifies each complete sign between hand raise and hand drop (matches training clips)." : "Classifies a sliding window continuously with smoothing."}>
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
                  </div>
                )}
                <Field label={t("live.endDelay", { ms: settings.engine.end_absent_ms })}>
                  <input type="range" min={200} max={1200} step={50} value={settings.engine.end_absent_ms} onChange={(e) => setEngine({ end_absent_ms: Number(e.target.value) })} className="accent-[var(--color-accent)]" aria-label={t("live.endDelayShort")} />
                </Field>
                <Field label={t("live.cooldown", { ms: settings.engine.cooldown_ms })}>
                  <input type="range" min={0} max={3000} step={100} value={settings.engine.cooldown_ms} onChange={(e) => setEngine({ cooldown_ms: Number(e.target.value) })} className="accent-[var(--color-accent)]" aria-label={t("live.cooldownShort")} />
                </Field>
                <Field label={t("live.resolution")} hint={t("live.resolutionHint")}>
                  <select className={inputCls} value={settings.procWidth} onChange={(e) => setSettings({ ...settings, procWidth: Number(e.target.value) })}>
                    <option value={0}>{t("live.resFull")}</option>
                    <option value={960}>960 px</option>
                    <option value={640}>640 px ({t("live.recommended")})</option>
                    <option value={480}>480 px ({t("live.lowEnd")})</option>
                  </select>
                </Field>
                <Field label={t("live.processor")} hint={t("live.processorHint")}>
                  <select className={inputCls} value={settings.delegate} onChange={(e) => setSettings({ ...settings, delegate: e.target.value as LiveSettings["delegate"] })}>
                    <option value="auto">Auto (GPU, fallback CPU)</option>
                    <option value="GPU">GPU (WebGL)</option>
                    <option value="CPU">CPU (WebAssembly)</option>
                  </select>
                </Field>
                <Field label={t("live.fpsCap")}>
                  <select className={inputCls} value={settings.targetFps} onChange={(e) => setSettings({ ...settings, targetFps: Number(e.target.value) })}>
                    {[30, 25, 20, 15].map((f) => <option key={f} value={f}>{f} fps</option>)}
                  </select>
                </Field>
                <div className="flex flex-col gap-2.5">
                  <span className="text-xs font-medium text-fg-2">{t("live.overlay")}</span>
                  <Toggle label={t("live.ovHands")} checked={settings.overlay.hands} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, hands: v } })} />
                  <Toggle label={t("live.ovPose")} checked={settings.overlay.pose} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, pose: v } })} />
                  <Toggle label={t("live.ovFace")} checked={settings.overlay.face} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, face: v } })} />
                  <Toggle label={t("live.ovTrail")} checked={settings.overlay.trail} onChange={(v) => setSettings({ ...settings, overlay: { ...settings.overlay, trail: v } })} />
                  <Toggle label={t("live.ovMirror")} checked={settings.mirror} onChange={(v) => setSettings({ ...settings, mirror: v })} />
                </div>
              </div>
            </Card>
          )}
        </div>

        {/* ------------------------------------------------ prediction column */}
        <div className="flex flex-col gap-4">
          <Card title={t("live.currentSign")} subtitle={statusHelp}>
            <ResultAnnouncer lastResult={lastResult} resolved={resolved} inProgress={inProgress} pending={live.pending} slidingStrict={slidingStrict} />
            <SignResult
              lastResult={lastResult} resolved={resolved} inProgress={inProgress} provisionalTop={top} recognizedNow={status === "recognized"}
              pack={pack} threshold={threshold} spellOn={spellOn} spell={live.spell} spellAvailable={spellAvailable} spellExperimental={spellExperimental}
              acceptedItem={acceptedItem} spellHint={spellHint}
              pending={live.pending} continuous={continuous} slidingStrict={slidingStrict} precisionNote={precisionNote}
              onConfirmPending={actions.confirmPending} onRejectPending={actions.rejectPending}
              onChoose={actions.choose} onDismiss={actions.dismiss} onReject={actions.rejectWord} onSpell={() => setSpellOn(true)} />
            {lastResult && !inProgress && (
              <div className="mt-3 text-xs text-fg-3">
                {lastResult.accepted
                  ? (lastResult.needs_confirm ? (live.pending ? `${t("live.notShown")}: ${t("live.waitingForTap")}` : t("live.notShown")) : t("live.accepted"))
                  : `${t("live.notShown")}: ${reasonText(lastResult.reason)}`}
                {lastResult.agreement && ` · ${t("live.cropsAgree", { a: lastResult.agreement })}`}
                {lastResult.frames ? ` · ${lastResult.frames} frames` : ""}
              </div>
            )}
            {!pack && provisional && provisional.length > 0 && (
              <div className="mt-5" lang="en">
                <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-fg-3">Top-5 class scores</div>
                <HBars items={provisional.map((p) => ({ label: p.gloss.toUpperCase(), value: p.prob }))} highlightFirst={decision === "accept"} />
              </div>
            )}
            {recognizing && server && (server.segment_frames ?? 0) > 0 && (
              <div className="mt-4 font-mono text-[11px] text-fg-3">{t("live.collecting")} · {server.segment_frames} frames · {((server.segment_ms ?? 0) / 1000).toFixed(1)} s</div>
            )}
          </Card>

          <SentenceBar items={transcript} language={sign.id} onClear={actions.clearText} onRemove={actions.removeWord} onWrong={actions.rejectWord} />

          <Card title={t("live.recent")}>
            {transcript.length === 0 ? (
              <p className="text-sm text-fg-3">{t("live.recentEmpty")}</p>
            ) : (
              <ScrollRegion label={t("live.recent")} className="max-h-56 overflow-y-auto">
                <ul className="flex flex-col gap-1">
                  {[...transcript].reverse().slice(0, 20).map((x) => {
                    const w = wordText(x);
                    return (
                      <li key={x.id} className="flex items-center justify-between gap-3 rounded-md px-2 py-1.5 text-sm hover:bg-ink-800">
                        <BilingualWord en={w.en} hi={w.hi} review={x.hi_review} size="sm" inline showMark={false} />
                        <span className="flex shrink-0 items-center gap-3 font-mono text-xs text-fg-3">
                          {x.chosen ? t("live.chosen") : pct(x.score, 0)}
                          <span>{new Date(x.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </ScrollRegion>
            )}
          </Card>

          <Card title={t("live.model")}>
            {live.modelAvailable === false ? (
              live.modelMessage && !live.modelMessage.startsWith("No active model") ? (
                <p className="text-sm text-fg-2">{live.modelMessage}</p>
              ) : (
                <p className="text-sm text-fg-2">No active model. <Link className="text-accent underline underline-offset-2" to="/lab">Train one in the Model Laboratory</Link> and activate it.</p>
              )
            ) : model && pack ? (
              <div className="flex flex-col gap-1.5 text-sm">
                <div className="flex items-center justify-between gap-2"><span className="truncate text-fg">{model.name}</span>{smoke && <Badge tone="warn">{t("badge.testBuild")}</Badge>}</div>
                <div className="font-mono text-xs text-fg-3">
                  {model.version ?? ""} · {model.profile === "commercial-safe" ? t("live.profileSafe") : t("live.profileResearch")}
                </div>
                <div className="font-mono text-xs text-fg-3">{t("live.dictionaryEntries", { n: model.num_classes.toLocaleString() })}</div>
                {activeCert && <div data-testid="model-precision" className="text-xs text-fg-2">{precisionNote(activeCert)}</div>}
              </div>
            ) : model && model.architecture ? (
              <div className="flex flex-col gap-1.5 text-sm">
                <div className="flex items-center justify-between gap-2"><span className="truncate text-fg">{model.name}</span>{model.is_synthetic && <Badge tone="warn">synthetic</Badge>}</div>
                <div className="font-mono text-xs text-fg-3">{model.architecture.toUpperCase()} · {model.num_classes} classes · T={model.seq_len} · split {model.split}</div>
                <div className="font-mono text-xs text-fg-3">validation top-1 {pct(model.val_top1)} · threshold {pct(model.threshold, 0)}</div>
              </div>
            ) : (
              <p className="text-sm text-fg-3">{t("live.connectHint")}</p>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

