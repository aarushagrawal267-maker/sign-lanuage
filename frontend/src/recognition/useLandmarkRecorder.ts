// Camera + MediaPipe Holistic in the browser for recording sign takes (My signs) and alphabet letters. Only landmark
// coordinates are kept; video frames never leave the browser and are never stored.
import { useCallback, useEffect, useRef, useState } from "react";
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { api, type FeatureSpec, type RawFrame } from "../api/client";
import { DEFAULT_FACE_IDX, loadHolistic, toRawFrame } from "../mediapipe/holistic";
import { drawFrame } from "../mediapipe/draw";

export type Take = { width: number; height: number; frames: RawFrame[] };

export type RecordMode =
  /** A sign: starts when hands appear, ends when they have been down for `endMs` (or at `maxMs`). */
  | { kind: "sign"; endMs?: number; maxMs?: number; preRollMs?: number }
  /** A letter: records `holdMs` once hands are visible (hold the hand shape still). */
  | { kind: "hold"; holdMs?: number };

export type RecorderPhase = "idle" | "waiting" | "recording";

/** Pure: should a sign recording stop now? (unit-tested) */
export function shouldStopSign(opts: { startT: number; lastHandT: number | null; now: number; endMs: number; maxMs: number }): boolean {
  if (opts.now - opts.startT >= opts.maxMs) return true;
  return opts.lastHandT !== null && opts.now - opts.lastHandT >= opts.endMs;
}

/** Pure: keep the frames of a finished sign (drop the trailing hands-down tail, keep a little after the last hand). */
export function trimTake(frames: RawFrame[], tailMs = 120): RawFrame[] {
  let last = -1;
  for (let i = frames.length - 1; i >= 0; i--) {
    if (frames[i].left || frames[i].right) { last = i; break; }
  }
  if (last < 0) return [];
  const lastT = frames[last].t;
  return frames.filter((f) => f.t <= lastT + tailMs);
}

/** Pure: quick local summary of a take (the server does the real quality check). */
export function takeSummary(t: Take) {
  const n = t.frames.length;
  const hands = t.frames.filter((f) => f.left || f.right).length;
  const dur = n > 1 ? t.frames[n - 1].t - t.frames[0].t : 0;
  return { frames: n, handRate: n ? hands / n : 0, durationMs: dur };
}

export function useLandmarkRecorder() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [camera, setCamera] = useState<"inactive" | "starting" | "active" | "error">("inactive");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [mediapipe, setMediapipe] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [hands, setHands] = useState(false);
  const [body, setBody] = useState(false);
  const [phase, setPhase] = useState<RecorderPhase>("idle");
  const [elapsed, setElapsed] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const lmRef = useRef<HolisticLandmarker | null>(null);
  const loopRef = useRef<number | null>(null);
  const lastTsRef = useRef(0);
  const faceIdxRef = useRef<number[]>(DEFAULT_FACE_IDX);
  const procCanvas = useRef<HTMLCanvasElement | null>(null);
  const ring = useRef<RawFrame[]>([]);
  const rec = useRef<{
    mode: RecordMode; frames: RawFrame[]; startT: number | null; lastHandT: number | null;
    resolve: (t: Take | null) => void;
  } | null>(null);

  useEffect(() => {
    api.get<FeatureSpec>("/api/features/spec").then((s) => (faceIdxRef.current = s.raw.face_indices)).catch(() => {});
  }, []);

  const finish = useCallback((take: Take | null) => {
    const r = rec.current;
    rec.current = null;
    setPhase("idle");
    setElapsed(0);
    r?.resolve(take);
  }, []);

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const lm = lmRef.current;
    if (!video || !lm || video.readyState < 2 || !video.videoWidth) return;
    const now = performance.now();
    if (now - lastTsRef.current < 1000 / 25 - 2) return; // about 25 fps, like live recognition
    const ts = Math.max(Math.round(now), lastTsRef.current + 1);
    lastTsRef.current = ts;
    // Same default as live recognition: MediaPipe runs on a 640 px wide copy (landmarks are normalised).
    let source: HTMLVideoElement | HTMLCanvasElement = video;
    if (video.videoWidth > 640) {
      const pc = (procCanvas.current ??= document.createElement("canvas"));
      pc.width = 640;
      pc.height = Math.round((video.videoHeight / video.videoWidth) * 640);
      pc.getContext("2d")!.drawImage(video, 0, 0, pc.width, pc.height);
      source = pc;
    }
    let frame: RawFrame;
    try {
      frame = toRawFrame(lm.detectForVideo(source, ts), ts, faceIdxRef.current);
    } catch {
      return;
    }
    const hasHands = !!(frame.left || frame.right);
    setHands(hasHands);
    setBody(!!frame.pose);
    ring.current.push(frame);
    while (ring.current.length && ts - ring.current[0].t > 600) ring.current.shift();

    const canvas = canvasRef.current;
    if (canvas) {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(w * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      const ctx = canvas.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawFrame(ctx, frame, w, h, { pose: true, hands: true, face: false, mirror: true });
    }

    const r = rec.current;
    if (!r) return;
    const size = { width: video.videoWidth, height: video.videoHeight };
    if (r.startT === null) {
      if (!hasHands) return;
      r.startT = ts;
      const pre = r.mode.kind === "sign" ? r.mode.preRollMs ?? 200 : 0;
      r.frames = ring.current.filter((f) => f.t >= ts - pre && f.t < ts);
      setPhase("recording");
    }
    r.frames.push(frame);
    if (hasHands) r.lastHandT = ts;
    setElapsed(ts - r.startT);
    if (r.mode.kind === "hold") {
      if (ts - r.startT >= (r.mode.holdMs ?? 1200)) finish({ ...size, frames: r.frames });
    } else if (shouldStopSign({ startT: r.startT, lastHandT: r.lastHandT, now: ts, endMs: r.mode.endMs ?? 450, maxMs: r.mode.maxMs ?? 3800 })) {
      finish({ ...size, frames: trimTake(r.frames) });
    }
  }, [finish]);

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

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamera("error");
      setCameraError("This browser does not support camera access (use Chrome or Edge on http://localhost).");
      return;
    }
    setCamera("starting");
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
      streamRef.current = stream;
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      setCamera("active");
    } catch (e) {
      setCamera("error");
      const name = (e as DOMException)?.name;
      setCameraError(name === "NotAllowedError" ? "Camera permission was denied. Allow it in the browser's address bar, then try again."
        : name === "NotFoundError" ? "No camera was found." : String(e));
      return;
    }
    if (!lmRef.current) {
      setMediapipe("loading");
      try {
        lmRef.current = (await loadHolistic("auto")).landmarker;
        setMediapipe("ready");
      } catch {
        setMediapipe("error");
        return;
      }
    } else {
      setMediapipe("ready");
    }
    startLoop();
  }, [startLoop]);

  const stop = useCallback(() => {
    if (rec.current) finish(null);
    stopLoop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const c = canvasRef.current;
    c?.getContext("2d")?.clearRect(0, 0, c.width, c.height);
    setCamera("inactive");
    setHands(false);
    setBody(false);
  }, [finish, stopLoop]);

  /** Record one take. Resolves with the landmarks, or null when cancelled. */
  const record = useCallback((mode: RecordMode): Promise<Take | null> => {
    if (rec.current) finish(null);
    setPhase("waiting");
    return new Promise((resolve) => {
      rec.current = { mode, frames: [], startT: null, lastHandT: null, resolve };
    });
  }, [finish]);

  const cancel = useCallback(() => finish(null), [finish]);

  useEffect(() => () => {
    stopLoop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, [stopLoop]);

  return { videoRef, canvasRef, camera, cameraError, mediapipe, hands, body, phase, elapsed, start, stop, record, cancel };
}
