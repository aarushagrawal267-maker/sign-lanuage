// Pure helpers for the live recognition loop (unit-tested in live.test.ts).
import type { TopK } from "../api/client";
import type { LiveStatus } from "./status";

export type EngineSettings = {
  mode: "segment" | "sliding";
  threshold: number | null; // null = model's validation-derived threshold
  min_margin: number;
  cooldown_ms: number;
  end_absent_ms: number;
};

export type LiveSettings = {
  engine: EngineSettings;
  targetFps: number;
  procWidth: number;
  overlay: { pose: boolean; hands: boolean; face: boolean; trail: boolean };
  mirror: boolean;
  delegate: "auto" | "GPU" | "CPU";
};

export const DEFAULT_SETTINGS: LiveSettings = {
  engine: { mode: "segment", threshold: null, min_margin: 0.1, cooldown_ms: 700, end_absent_ms: 400 },
  targetFps: 25,
  procWidth: 640,
  overlay: { pose: true, hands: true, face: false, trail: true },
  mirror: true,
  delegate: "auto",
};

export type ServerState = {
  type: "state";
  t: number;
  status: string;
  hands: boolean;
  pose: boolean;
  threshold: number;
  server_ms: number;
  segment_frames?: number;
  segment_ms?: number;
  window_frames?: number;
  provisional?: { top: TopK[]; feature_ms: number; inference_ms: number; streak?: number; reason?: string };
  segment_result?: {
    accepted: boolean; reason: string; frames: number; duration_ms?: number; agreement?: string; top?: TopK[];
    feature_ms?: number; inference_ms?: number;
  };
  commit?: { gloss: string; class_index: number; score: number; t: number };
};

export type Inputs = {
  camera: "inactive" | "starting" | "active" | "error";
  mediapipe: "idle" | "loading" | "ready" | "error";
  recognizing: boolean;
  paused: boolean;
  socket: "closed" | "connecting" | "open" | "error";
  modelAvailable: boolean | null;
  serverStatus: string | null;
};

const ENGINE_STATES = new Set(["no_body", "waiting_for_hands", "collecting", "analyzing", "recognized", "uncertain", "unavailable"]);

/** Single source for the status pill. Order matters: hardware -> model -> engine. */
export function deriveStatus(s: Inputs): LiveStatus {
  if (s.camera === "error") return "camera_error";
  if (s.camera === "starting") return "camera_starting";
  if (s.camera === "inactive") return "camera_inactive";
  if (s.mediapipe === "error") return "mediapipe_error";
  if (s.mediapipe !== "ready") return "mediapipe_loading";
  if (!s.recognizing) return "preview";
  if (s.paused) return "paused";
  if (s.socket === "connecting" || (s.socket === "open" && s.modelAvailable === null)) return "connecting";
  if (s.socket !== "open" || s.modelAvailable === false) return "unavailable";
  if (s.serverStatus && ENGINE_STATES.has(s.serverStatus)) return s.serverStatus as LiveStatus;
  return "waiting_for_hands";
}

export type TranscriptItem = { id: number; gloss: string; score: number; at: number };

/** Rolling mean of numeric samples within a time window. */
export class RollingStat {
  private xs: { t: number; v: number }[] = [];
  constructor(private windowMs = 2000) {}
  push(v: number, t = performance.now()) {
    this.xs.push({ t, v });
    while (this.xs.length && t - this.xs[0].t > this.windowMs) this.xs.shift();
  }
  mean(): number | null {
    return this.xs.length ? this.xs.reduce((a, b) => a + b.v, 0) / this.xs.length : null;
  }
  rate(t = performance.now()): number | null {
    while (this.xs.length && t - this.xs[0].t > this.windowMs) this.xs.shift();
    if (this.xs.length < 2) return null;
    const span = (this.xs[this.xs.length - 1].t - this.xs[0].t) / 1000;
    return span > 0 ? (this.xs.length - 1) / span : null;
  }
  clear() {
    this.xs = [];
  }
}

/** Mean of the per-frame values that came back with the last state message. */
export function engineConfigPayload(e: EngineSettings) {
  return {
    mode: e.mode,
    threshold: e.threshold,
    min_margin: e.min_margin,
    cooldown_ms: e.cooldown_ms,
    end_absent_ms: e.end_absent_ms,
  };
}
