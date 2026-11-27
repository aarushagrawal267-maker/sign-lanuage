import { useCallback, useEffect, useRef, useState } from "react";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { LocalRecognitionSocket } from "../static/engine";
import { IS_STATIC, api, wsUrl, type Candidate, type FeatureSpec, type LanguageInfo, type ModelInfo, type RawFrame } from "../api/client";
import { DEFAULT_FACE_IDX, loadHolistic, toRawFrame } from "../mediapipe/holistic";
import { drawFrame } from "../mediapipe/draw";
import {
  RollingStat, bringsWord, deriveStatus, engineConfigPayload, needsConfirm, segmentFromCommit, transcriptFromCandidate, transcriptItem,
  type CertificationByPolicy, type Commit, type LiveSettings, type SegmentResult, type ServerState, type SpellState,
  type TranscriptItem,
} from "./live";

type Metrics = { fps: number | null; mpMs: number | null; serverMs: number | null; inferenceMs: number | null; e2eMs: number | null; sendFps: number | null };

/** Language pack details the server reports in its hello message (authoritative for what this socket can do). */
/** How a shown result was settled by the user: picked from "did you mean", "none of these", or "that word was wrong". */
export type ResolvedHow = "chosen" | "dismissed" | "rejected";

/**
 * A word the server accepted that waits for the user's tap before it is added, spoken or sent (language packs,
 * docs/ISL_DESIGN.md 15.29). `segment` is the result it came from; in continuous mode, where a commit has no result,
 * it is made from the commit so the "added" / "sign again" message has something to refer to.
 */
export type PendingWord = { commit: Commit; segment: SegmentResult; continuous: boolean };

export type HelloLanguage = Pick<LanguageInfo, "id" | "display_name" | "output_languages" | "capabilities" | "profile" | "status" | "sign_language">;

/**
 * Live recognition over the WebSocket. `language` is a language-pack id (docs/ISL_DESIGN.md 4.5); `pack` says it is a
 * language pack with a recogniser (ISL: accept / suggest / abstain, choose / reject, spelling) rather than the classic
 * ASL engine, whose messages are unchanged. `spell` switches a pack to spelling mode.
 */
export function useLiveRecognition(settings: LiveSettings, language: string, opts: { pack: boolean; spell: boolean; sliding?: boolean } = { pack: false, spell: false }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [camera, setCamera] = useState<"inactive" | "starting" | "active" | "error">("inactive");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [mediapipe, setMediapipe] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [delegate, setDelegate] = useState<string | null>(null);
  const [recognizing, setRecognizing] = useState(false);
  const [paused, setPaused] = useState(false);
  const [socket, setSocket] = useState<"closed" | "connecting" | "open" | "error">("closed");
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelAvailable, setModelAvailable] = useState<boolean | null>(null);
  const [modelMessage, setModelMessage] = useState<string | null>(null);
  const [server, setServer] = useState<ServerState | null>(null);
  const [provisional, setProvisional] = useState<Candidate[] | null>(null);
  const [lastResult, setLastResult] = useState<SegmentResult | null>(null);
  const [resolved, setResolved] = useState<{ segment: SegmentResult; how: ResolvedHow } | null>(null);
  const lastResultRef = useRef<SegmentResult | null>(null);
  const [spell, setSpell] = useState<SpellState | null>(null);
  const [helloLanguage, setHelloLanguage] = useState<HelloLanguage | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [pending, setPending] = useState<PendingWord | null>(null);
  const pendingRef = useRef<PendingWord | null>(null);
  pendingRef.current = pending;
  const [certs, setCerts] = useState<CertificationByPolicy | null>(null);
  const [detection, setDetection] = useState({ pose: false, left: false, right: false, face: false });
  const [metrics, setMetrics] = useState<Metrics>({ fps: null, mpMs: null, serverMs: null, inferenceMs: null, e2eMs: null, sendFps: null });
  const [videoSize, setVideoSize] = useState<{ w: number; h: number } | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const lmRef = useRef<HolisticLandmarker | null>(null);
  const lmPref = useRef<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const loopRef = useRef<number | null>(null);
  const lastTsRef = useRef(0);
  const lastProcRef = useRef(0);
  const procCanvas = useRef<HTMLCanvasElement | null>(null);
  const faceIdxRef = useRef<number[]>(DEFAULT_FACE_IDX);
  const sendingRef = useRef(false);
  const inflight = useRef(new Map<number, number>());
  const trail = useRef<{ x: number; y: number; side: "left" | "right" }[]>([]);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const langRef = useRef(language);
  langRef.current = language;
  // Sliding (continuous) mode needs the pack's classifier: unless the caller says, it is available when the server's
  // certification has sliding rows (docs/ISL_DESIGN.md 15.29, 15.30).
  const slidingAvailable = !!opts.pack && (opts.sliding ?? !!certs?.sliding);
  const effOpts = { pack: opts.pack, spell: opts.spell, sliding: slidingAvailable };
  const optsRef = useRef(effOpts);
  optsRef.current = effOpts;
  const lastConfigRef = useRef<string | null>(null);
  const payload = () => {
    const p = engineConfigPayload(settingsRef.current.engine, optsRef.current);
    lastConfigRef.current = JSON.stringify(p);
    return p;
  };
  const stats = useRef({ fps: new RollingStat(), mp: new RollingStat(), srv: new RollingStat(), inf: new RollingStat(4000), e2e: new RollingStat(), send: new RollingStat() });
  const idRef = useRef(1);
  const pausedRef = useRef(false);
  pausedRef.current = paused;

  // Fetch the landmark spec once so the face subset matches the backend exactly.
  useEffect(() => {
    api.get<FeatureSpec>("/api/features/spec").then((s) => (faceIdxRef.current = s.raw.face_indices)).catch(() => {});
  }, []);

  // Periodic metrics publish (avoid re-rendering on every frame).
  useEffect(() => {
    const id = window.setInterval(() => {
      const s = stats.current;
      setMetrics({ fps: s.fps.rate(), mpMs: s.mp.mean(), serverMs: s.srv.mean(), inferenceMs: s.inf.mean(), e2eMs: s.e2e.mean(), sendFps: s.send.rate() });
    }, 500);
    return () => window.clearInterval(id);
  }, []);

  // ---------------------------------------------------------- websocket
  const closeSocket = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    sendingRef.current = false;
    inflight.current.clear();
    setSocket("closed");
  }, []);

  const openSocket = useCallback(() => {
    closeSocket();
    setSocket("connecting");
    setModelAvailable(null);
    setModelMessage(null);
    const lang = langRef.current;
    const ws = (IS_STATIC ? new LocalRecognitionSocket() : new WebSocket(wsUrl(`/api/ws/recognize?language=${encodeURIComponent(lang)}`))) as WebSocket;
    wsRef.current = ws;
    ws.onopen = () => {
      setSocket("open");
      const v = videoRef.current;
      ws.send(JSON.stringify({ type: "config", width: v?.videoWidth || 640, height: v?.videoHeight || 480, engine: payload() }));
    };
    ws.onerror = () => setSocket("error");
    ws.onclose = () => {
      if (wsRef.current === ws) {
        setSocket("closed");
        sendingRef.current = false;
      }
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "hello") {
        setModel(msg.model);
        setModelAvailable(!!msg.model);
        setModelMessage(typeof msg.message === "string" ? msg.message : null);
        setHelloLanguage(msg.language ?? null);
        setCerts(msg.engine?.certification_by_policy ?? null);
        setServerError(null);
        sendingRef.current = !!msg.model;
      } else if (msg.type === "error") {
        setServerError(typeof msg.message === "string" ? msg.message : "error");
      } else if (msg.type === "config_ack") {
        setServerError(null);
        setCerts(msg.engine?.certification_by_policy ?? null);
      } else if (msg.type === "choice_ack") {
        // A repeated choice for the same segment (double tap) comes back marked duplicate: add the word once.
        if (msg.duplicate) return;
        const c = msg.commit as Commit;
        setTranscript((tr) => [...tr, transcriptItem(idRef.current++, c)]);
      } else if (msg.type === "state") {
        const st = msg as ServerState;
        if (inflight.current.has(st.t)) {
          stats.current.e2e.push(performance.now() - st.t);
          inflight.current.delete(st.t);
        }
        stats.current.srv.push(st.server_ms);
        setServer(st);
        if (st.provisional) {
          setProvisional(st.provisional.top);
          stats.current.inf.push(st.provisional.inference_ms + st.provisional.feature_ms);
        }
        if (st.spell) setSpell(st.spell);
        if (st.segment_result) {
          // a new sign that brings a word replaces a word that was never confirmed (it is not added). A too-short
          // movement, an abstain or a "not sure" brings no word, so the waiting word stays (docs/ISL_DESIGN.md 15.30)
          if (bringsWord(st.segment_result)) setPending(null);
          setLastResult(st.segment_result);
          lastResultRef.current = st.segment_result;
          setResolved(null);
          setSpell(null);
          if (st.segment_result.inference_ms !== undefined) stats.current.inf.push(st.segment_result.inference_ms + (st.segment_result.feature_ms ?? 0));
          if (settingsRef.current.engine.mode === "segment") setProvisional(st.segment_result.top ?? null);
        }
        if (st.commit) {
          const c = st.commit;
          if (needsConfirm(c)) {
            // docs/ISL_DESIGN.md 15.29: even a certified word waits for a tap before it joins the sentence
            const continuous = !st.segment_result;
            setPending({ commit: c, segment: st.segment_result ?? segmentFromCommit(c), continuous });
          } else {
            setTranscript((tr) => [...tr, transcriptItem(idRef.current++, c)]);
          }
        }
      }
    };
  }, [closeSocket]);

  // Push engine setting changes (and spelling mode) to an open socket; nothing is sent when the engine settings did not
  // change (e.g. sliding became available but continuous signing is off).
  useEffect(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      const p = engineConfigPayload(settings.engine, effOpts);
      const sig = JSON.stringify(p);
      if (sig !== lastConfigRef.current) {
        lastConfigRef.current = sig;
        ws.send(JSON.stringify({ type: "config", engine: p }));
      }
    }
    setSpell(null);
  }, [settings.engine, opts.pack, opts.spell, slidingAvailable]); // eslint-disable-line react-hooks/exhaustive-deps

  // Switching the sign language: the open socket changes pack (the server answers with a new hello) and the
  // sentence starts again, because words of two sign languages do not form one sentence.
  const firstLang = useRef(true);
  useEffect(() => {
    if (firstLang.current) {
      firstLang.current = false;
      return;
    }
    const ws = wsRef.current;
    setProvisional(null);
    setLastResult(null);
    lastResultRef.current = null;
    setResolved(null);
    setSpell(null);
    setTranscript([]);
    setPending(null);
    setCerts(null);
    setHelloLanguage(null);
    setModel(null);
    setModelMessage(null);
    if (ws && ws.readyState === WebSocket.OPEN && !IS_STATIC) {
      setModelAvailable(null);
      sendingRef.current = false;
      const v = videoRef.current;
      ws.send(JSON.stringify({ type: "config", language, width: v?.videoWidth || 640, height: v?.videoHeight || 480, engine: payload() }));
    }
  }, [language]); // eslint-disable-line react-hooks/exhaustive-deps

  // --------------------------------------------------------------- loop
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const lm = lmRef.current;
    const canvas = canvasRef.current;
    if (!video || !lm || video.readyState < 2 || !video.videoWidth) return;
    const now = performance.now();
    const s = settingsRef.current;
    if (now - lastProcRef.current < 1000 / s.targetFps - 2) return;
    lastProcRef.current = now;

    // Optional downscale before MediaPipe (limited hardware setting).
    let source: HTMLVideoElement | HTMLCanvasElement = video;
    if (s.procWidth && video.videoWidth > s.procWidth) {
      const pc = (procCanvas.current ??= document.createElement("canvas"));
      pc.width = s.procWidth;
      pc.height = Math.round((video.videoHeight / video.videoWidth) * s.procWidth);
      pc.getContext("2d")!.drawImage(video, 0, 0, pc.width, pc.height);
      source = pc;
    }
    const ts = Math.max(Math.round(now), lastTsRef.current + 1);
    lastTsRef.current = ts;
    const t0 = performance.now();
    let frame: RawFrame;
    try {
      frame = toRawFrame(lm.detectForVideo(source, ts), ts, faceIdxRef.current);
    } catch (e) {
      console.error("MediaPipe detection failed", e);
      return;
    }
    stats.current.mp.push(performance.now() - t0);
    if (import.meta.env.DEV) (window as unknown as { __slrLastFrame?: RawFrame }).__slrLastFrame = frame; // e2e parity checks
    stats.current.fps.push(1);
    setDetection({ pose: !!frame.pose, left: !!frame.left, right: !!frame.right, face: !!frame.face });

    // Wrist trail (overlay only)
    for (const side of ["left", "right"] as const) {
      const h = frame[side];
      if (h) trail.current.push({ x: h[0][0], y: h[0][1], side });
    }
    if (trail.current.length > 40) trail.current.splice(0, trail.current.length - 40);
    if (!frame.left && !frame.right && trail.current.length) trail.current.shift();

    if (canvas) {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(w * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      const ctx = canvas.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawFrame(ctx, frame, w, h, { ...s.overlay, mirror: s.mirror, trail: s.overlay.trail ? trail.current : undefined });
    }

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN && sendingRef.current && !pausedRef.current) {
      // Back-pressure: drop frames rather than queue if the server falls behind.
      if (inflight.current.size < 8 && ws.bufferedAmount < 256_000) {
        inflight.current.set(ts, now);
        ws.send(JSON.stringify({ type: "frame", ...frame }));
        stats.current.send.push(1);
      }
    }
  }, []);

  const startLoop = useCallback(() => {
    const video = videoRef.current as (HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number }) | null;
    if (!video) return;
    const tick = () => {
      processFrame();
      loopRef.current = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
    };
    tick();
  }, [processFrame]);

  const stopLoop = useCallback(() => {
    const video = videoRef.current as (HTMLVideoElement & { cancelVideoFrameCallback?: (h: number) => void }) | null;
    if (loopRef.current !== null) {
      if (video?.cancelVideoFrameCallback) video.cancelVideoFrameCallback(loopRef.current);
      cancelAnimationFrame(loopRef.current);
    }
    loopRef.current = null;
  }, []);

  // ------------------------------------------------------------- actions
  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamera("error");
      setCameraError("This browser does not support camera access (use Chrome or Edge on http://localhost).");
      return;
    }
    setCamera("starting");
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      setVideoSize({ w: v.videoWidth, h: v.videoHeight });
      setCamera("active");
    } catch (e) {
      setCamera("error");
      const name = (e as DOMException)?.name;
      setCameraError(name === "NotAllowedError" ? "Camera permission was denied. Allow it in the browser's address bar, then try again."
        : name === "NotFoundError" ? "No camera was found." : name === "NotReadableError" ? "The camera is in use by another application." : String(e));
      return;
    }
    if (!lmRef.current || lmPref.current !== settingsRef.current.delegate) {
      setMediapipe("loading");
      try {
        lmPref.current = settingsRef.current.delegate;
        const { landmarker, delegate } = await loadHolistic(settingsRef.current.delegate);
        lmRef.current = landmarker;
        setDelegate(delegate);
        setMediapipe("ready");
      } catch (e) {
        console.error(e);
        setMediapipe("error");
        return;
      }
    }
    startLoop();
  }, [startLoop]);

  const stopCamera = useCallback(() => {
    stopLoop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const c = canvasRef.current;
    c?.getContext("2d")?.clearRect(0, 0, c.width, c.height);
    closeSocket();
    setRecognizing(false);
    setPaused(false);
    setCamera("inactive");
    setServer(null);
    setDetection({ pose: false, left: false, right: false, face: false });
    trail.current = [];
    Object.values(stats.current).forEach((s) => s.clear());
  }, [closeSocket, stopLoop]);

  const startRecognition = useCallback(() => {
    setRecognizing(true);
    setPaused(false);
    openSocket();
  }, [openSocket]);

  const pause = useCallback(() => setPaused(true), []);
  const resume = useCallback(() => {
    setPaused(false);
    wsRef.current?.send(JSON.stringify({ type: "reset" }));
  }, []);
  const stopRecognition = useCallback(() => {
    closeSocket();
    setRecognizing(false);
    setPaused(false);
    setServer(null);
    setProvisional(null);
    setSpell(null);
    setPending(null);
  }, [closeSocket]);
  const resetSequence = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type: "reset" }));
    setProvisional(null);
    setLastResult(null);
    lastResultRef.current = null;
    setResolved(null);
    setSpell(null);
    setPending(null);
    trail.current = [];
  }, []);
  const clearText = useCallback(() => setTranscript([]), []);

  /** The user picked a word from "did you mean". Packs confirm it on the server (logged as chosen_by_user); the
   *  classic ASL engine has no segment ids, so the word is added here. */
  const choose = useCallback((segment: SegmentResult, cand: Candidate) => {
    const ws = wsRef.current;
    if (segment.segment_id !== undefined && cand.concept_id && ws && ws.readyState === WebSocket.OPEN && !IS_STATIC) {
      ws.send(JSON.stringify({ type: "choose", segment_id: segment.segment_id, concept_id: cand.concept_id }));
    } else {
      setTranscript((tr) => [...tr, transcriptFromCandidate(idRef.current++, cand)]);
    }
    if (pendingRef.current) { // another guess picked instead of the word waiting for a tap: that word is not added
      pendingRef.current = null;
      setPending(null);
    }
    setLastResult(segment);
    lastResultRef.current = segment;
    setResolved({ segment, how: "chosen" });
  }, []);
  /** "None of these": hide the suggestions until the next sign. */
  const dismiss = useCallback((segment: SegmentResult) => setResolved({ segment, how: "dismissed" }), []);
  /** "That was wrong": remove a word from the text; packs log the feedback (never used for training). */
  const rejectWord = useCallback((item: TranscriptItem) => {
    const ws = wsRef.current;
    if (item.segment_id !== null && item.segment_id !== undefined && ws && ws.readyState === WebSocket.OPEN && !IS_STATIC) {
      ws.send(JSON.stringify({ type: "reject", segment_id: item.segment_id }));
    }
    setTranscript((tr) => tr.filter((t) => t.id !== item.id));
    // If the rejected word is the one on show, stop showing it as "recognised" (it was wrong).
    const last = lastResultRef.current;
    if (last && item.segment_id !== null && item.segment_id !== undefined && last.segment_id === item.segment_id) setResolved({ segment: last, how: "rejected" });
  }, []);
  const removeWord = useCallback((id: number) => setTranscript((tr) => tr.filter((t) => t.id !== id)), []);
  /** The user tapped "add" on a word waiting for confirmation: only now does it join the sentence (and may be spoken). */
  const confirmPending = useCallback(() => {
    const p = pendingRef.current;
    if (!p) return;
    pendingRef.current = null;
    setPending(null);
    // packs log the word on the server only now, as confirmed_by_user (docs/ISL_DESIGN.md 15.30)
    const ws = wsRef.current;
    if (p.commit.concept_id && ws && ws.readyState === WebSocket.OPEN && !IS_STATIC) {
      const sid = p.commit.segment_id;
      ws.send(JSON.stringify(sid !== null && sid !== undefined ? { type: "confirm", segment_id: sid } : { type: "confirm", t: p.commit.t }));
    }
    setTranscript((tr) => [...tr, transcriptItem(idRef.current++, { ...p.commit, needs_confirm: false })]);
    setLastResult(p.segment);
    lastResultRef.current = p.segment;
    setResolved({ segment: p.segment, how: "chosen" });
  }, []);
  /** "Wrong word" / "No" on a word waiting for confirmation: it is never added; packs log the feedback. */
  const rejectPending = useCallback(() => {
    const p = pendingRef.current;
    if (!p) return;
    pendingRef.current = null;
    setPending(null);
    const ws = wsRef.current;
    const sid = p.commit.segment_id;
    if (sid !== null && sid !== undefined && ws && ws.readyState === WebSocket.OPEN && !IS_STATIC) {
      ws.send(JSON.stringify({ type: "reject", segment_id: sid }));
    }
    setLastResult(p.segment);
    lastResultRef.current = p.segment;
    setResolved({ segment: p.segment, how: "dismissed" });
  }, []);

  // Stop everything when the page unmounts (camera light must go off).
  useEffect(() => () => {
    stopLoop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    wsRef.current?.close();
  }, [stopLoop]);

  const status = deriveStatus({ camera, mediapipe, recognizing, paused, socket, modelAvailable, serverStatus: server?.status ?? null });

  return {
    videoRef, canvasRef, status, camera, cameraError, mediapipe, delegate, recognizing, paused, socket, model, modelAvailable, modelMessage,
    server, provisional, lastResult, resolved, spell, transcript, detection, metrics, videoSize, helloLanguage, serverError, pending, certs,
    actions: { startCamera, stopCamera, startRecognition, stopRecognition, pause, resume, resetSequence, clearText, choose, dismiss, rejectWord, removeWord, confirmPending, rejectPending },
  };
}
