// TypeScript port of backend/app/inference/engine.py + an in-browser ONNX predictor.
// Used by the static build so live recognition runs entirely in the browser.
import * as ort from "onnxruntime-web/wasm";
import type { RawFrame, TopK } from "../api/client";
import { FeatureError, buildFeatures, featureDim, frameMask, type FeatureConfig } from "./features";

const BASE = import.meta.env.BASE_URL;

export type ModelMeta = {
  model: Record<string, unknown> & { experiment_id: number };
  threshold: number;
  feature_config: FeatureConfig;
  class_names: string[];
  feature_dim: number;
};

export class OnnxPredictor {
  private constructor(private session: ort.InferenceSession, public meta: ModelMeta) {}

  static async load(): Promise<OnnxPredictor> {
    ort.env.wasm.wasmPaths = new URL(`${BASE}ort/`, document.baseURI).href; // absolute: ort resolves relative to its own module
    ort.env.wasm.numThreads = 1; // static hosting has no cross-origin isolation
    const [meta, session] = await Promise.all([
      fetch(`${BASE}static-data/model.json`).then((r) => r.json() as Promise<ModelMeta>),
      ort.InferenceSession.create(`${BASE}static-data/model.onnx`, { executionProviders: ["wasm"] }),
    ]);
    if (featureDim(meta.feature_config) !== meta.feature_dim) throw new Error("feature layout mismatch with exported model");
    return new OnnxPredictor(session, meta);
  }

  get cfg() {
    return this.meta.feature_config;
  }

  /** feats: list of (T*D) arrays -> list of probability vectors, plus inference time. */
  async predict(feats: Float32Array[]): Promise<{ probs: Float32Array[]; ms: number }> {
    const { seq_len: T } = this.cfg;
    const D = this.meta.feature_dim;
    const B = feats.length;
    const x = new Float32Array(B * T * D);
    const mask = new Uint8Array(B * T);
    feats.forEach((f, b) => {
      x.set(f, b * T * D);
      frameMask(f, this.cfg).forEach((m, t) => (mask[b * T + t] = m ? 1 : 0));
    });
    const t0 = performance.now();
    const out = await this.session.run({ x: new ort.Tensor("float32", x, [B, T, D]), mask: new ort.Tensor("bool", mask, [B, T]) });
    const ms = performance.now() - t0;
    const data = out.probs.data as Float32Array;
    const C = this.meta.class_names.length;
    return { probs: Array.from({ length: B }, (_, b) => data.slice(b * C, (b + 1) * C)), ms };
  }
}

export type EngineConfig = {
  mode: "segment" | "sliding";
  threshold: number | null;
  min_margin: number;
  min_frames: number;
  min_hand_ratio: number;
  start_frames: number;
  pre_roll_ms: number;
  end_absent_ms: number;
  max_segment_ms: number;
  interval_ms: number;
  window_ms: number;
  ema_alpha: number;
  agree_n: number;
  cooldown_ms: number;
  top_k: number;
};

export const DEFAULT_ENGINE: EngineConfig = {
  mode: "segment", threshold: null, min_margin: 0.1, min_frames: 6, min_hand_ratio: 0.5, start_frames: 3, pre_roll_ms: 240,
  end_absent_ms: 400, max_segment_ms: 4000, interval_ms: 200, window_ms: 2000, ema_alpha: 0.5, agree_n: 3, cooldown_ms: 700, top_k: 5,
};

type Frame = { t: number; data: RawFrame; hands: boolean; pose: boolean };

export class LocalEngine {
  cfg: EngineConfig = { ...DEFAULT_ENGINE };
  width = 640;
  height = 480;
  private buf: Frame[] = [];
  private segment: Frame[] = [];
  private active = false;
  private handStreak = 0;
  private lastHandT: number | null = null;
  private lastInferT = -1e18;
  private lastCommitT = -1e18;
  private lastCommitClass: number | null = null;
  private ema: Float32Array | null = null;
  private streakClass: number | null = null;
  private streak = 0;
  status = "waiting_for_hands";

  constructor(private p: OnnxPredictor) {}

  get threshold() {
    return this.cfg.threshold ?? this.p.meta.threshold;
  }

  reset() {
    this.buf = []; this.segment = []; this.active = false; this.handStreak = 0; this.lastHandT = null; this.lastInferT = -1e18;
    this.lastCommitT = -1e18; this.lastCommitClass = null; this.ema = null; this.streakClass = null; this.streak = 0;
    this.status = "waiting_for_hands";
  }

  private async classify(frames: Frame[], tta: boolean): Promise<
    { error: string } | { probs: Float32Array; agree: number; crops: number; feature_ms: number; inference_ms: number }
  > {
    const t0 = performance.now();
    const crops = [frames];
    if (tta && frames.length >= 12) {
      const k = Math.max(1, Math.floor(frames.length / 10));
      crops.push(frames.slice(k), frames.slice(0, -k));
    }
    let feats: Float32Array[];
    try {
      feats = crops.map((c) => buildFeatures(c.map((f) => f.data), this.width, this.height, this.p.cfg));
    } catch (e) {
      if (e instanceof FeatureError) return { error: e.message };
      throw e;
    }
    const featureMs = performance.now() - t0;
    const { probs, ms } = await this.p.predict(feats);
    const C = probs[0].length;
    const avg = new Float32Array(C);
    probs.forEach((p) => p.forEach((v, i) => (avg[i] += v / probs.length)));
    const top1 = argmax(avg);
    const agree = probs.filter((p) => argmax(p) === top1).length;
    return { probs: avg, agree, crops: crops.length, feature_ms: featureMs, inference_ms: ms };
  }

  private topk(p: Float32Array): TopK[] {
    return [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, this.cfg.top_k)
      .map((i) => ({ class_index: i, gloss: this.p.meta.class_names[i], prob: p[i] }));
  }

  private decision(p: Float32Array, agreeOk: boolean): [boolean, string] {
    const s = [...p].sort((a, b) => b - a);
    if (s[0] < this.threshold) return [false, "low_confidence"];
    if (s.length > 1 && s[0] - s[1] < this.cfg.min_margin) return [false, "ambiguous"];
    if (!agreeOk) return [false, "unstable"];
    return [true, "ok"];
  }

  async push(frame: RawFrame): Promise<Record<string, unknown>> {
    const hands = !!(frame.left || frame.right);
    const pose = !!frame.pose;
    const t = frame.t;
    const f: Frame = { t, data: frame, hands, pose };
    this.buf.push(f);
    const horizon = Math.max(this.cfg.max_segment_ms, this.cfg.window_ms) + this.cfg.pre_roll_ms + 500;
    while (this.buf.length && t - this.buf[0].t > horizon) this.buf.shift();
    if (hands) this.lastHandT = t;
    const msg: Record<string, unknown> = { type: "state", t, hands, pose };
    Object.assign(msg, this.cfg.mode === "segment" ? await this.pushSegment(f) : await this.pushSliding(f));
    msg.status = this.status;
    msg.threshold = this.threshold;
    return msg;
  }

  private async pushSegment(f: Frame) {
    const c = this.cfg;
    const out: Record<string, unknown> = {};
    if (!this.active) {
      this.handStreak = f.hands ? this.handStreak + 1 : 0;
      if (this.handStreak >= c.start_frames) {
        this.active = true;
        this.segment = this.buf.filter((x) => x.t >= f.t - c.pre_roll_ms);
        this.status = "collecting";
      } else this.status = f.pose ? "waiting_for_hands" : "no_body";
      out.segment_frames = this.active ? this.segment.length : 0;
      return out;
    }
    this.segment.push(f);
    const dur = f.t - this.segment[0].t;
    const absent = f.t - (this.lastHandT ?? f.t);
    out.segment_frames = this.segment.length;
    out.segment_ms = dur;
    if (absent >= c.end_absent_ms || dur >= c.max_segment_ms) {
      const frames = [...this.segment];
      while (frames.length > 1 && !frames[frames.length - 1].hands && frames[frames.length - 1].t - (this.lastHandT ?? 0) > 120) frames.pop();
      this.active = false; this.segment = []; this.handStreak = 0;
      Object.assign(out, await this.finalize(frames, f.t));
      return out;
    }
    this.status = "collecting";
    if (this.segment.length >= c.min_frames && f.t - this.lastInferT >= c.interval_ms) {
      this.lastInferT = f.t;
      const r = await this.classify(this.segment, false);
      if (!("error" in r)) {
        this.status = "analyzing";
        out.provisional = { top: this.topk(r.probs), feature_ms: r.feature_ms, inference_ms: r.inference_ms };
      }
    }
    return out;
  }

  private async finalize(frames: Frame[], t: number) {
    const c = this.cfg;
    const handRatio = frames.filter((x) => x.hands).length / Math.max(1, frames.length);
    if (frames.length < c.min_frames || handRatio < c.min_hand_ratio * 0.6) {
      this.status = "waiting_for_hands";
      return { segment_result: { accepted: false, reason: "too_short", frames: frames.length } };
    }
    const r = await this.classify(frames, true);
    if ("error" in r) {
      this.status = "no_body";
      return { segment_result: { accepted: false, reason: r.error } };
    }
    const need = r.crops >= 3 ? 2 : 1;
    let [ok, reason] = this.decision(r.probs, r.agree >= need);
    if (ok && t - this.lastCommitT < c.cooldown_ms) [ok, reason] = [false, "cooldown"];
    const top = this.topk(r.probs);
    const res = { accepted: ok, reason, frames: frames.length, duration_ms: frames[frames.length - 1].t - frames[0].t,
      agreement: `${r.agree}/${r.crops}`, top, feature_ms: r.feature_ms, inference_ms: r.inference_ms };
    const out: Record<string, unknown> = { segment_result: res };
    if (ok) {
      this.lastCommitT = t;
      this.lastCommitClass = top[0].class_index;
      this.status = "recognized";
      out.commit = { gloss: top[0].gloss, class_index: top[0].class_index, score: top[0].prob, t };
    } else this.status = "uncertain";
    return out;
  }

  private async pushSliding(f: Frame) {
    const c = this.cfg;
    const out: Record<string, unknown> = {};
    if (this.lastHandT === null || f.t - this.lastHandT >= c.end_absent_ms) {
      this.ema = null; this.streak = 0; this.streakClass = null; this.lastCommitClass = null;
      this.status = f.pose ? "waiting_for_hands" : "no_body";
      return out;
    }
    if (f.t - this.lastInferT < c.interval_ms) return out;
    const win = this.buf.filter((x) => x.t >= f.t - c.window_ms);
    const ratio = win.filter((x) => x.hands).length / Math.max(1, win.length);
    out.window_frames = win.length;
    if (win.length < c.min_frames || ratio < c.min_hand_ratio) {
      this.status = "collecting";
      return out;
    }
    this.lastInferT = f.t;
    const r = await this.classify(win, false);
    if ("error" in r) {
      this.status = "no_body";
      return out;
    }
    const p = r.probs;
    this.ema = this.ema ? p.map((v, i) => c.ema_alpha * v + (1 - c.ema_alpha) * this.ema![i]) : p;
    const top = this.topk(this.ema);
    const cls = top[0].class_index;
    this.streak = cls === this.streakClass ? this.streak + 1 : 1;
    this.streakClass = cls;
    let [ok, reason] = this.decision(this.ema, this.streak >= c.agree_n);
    if (ok && cls === this.lastCommitClass) [ok, reason] = [false, "repeat_suppressed"];
    if (ok && f.t - this.lastCommitT < c.cooldown_ms) [ok, reason] = [false, "cooldown"];
    out.provisional = { top, feature_ms: r.feature_ms, inference_ms: r.inference_ms, streak: this.streak, reason };
    if (ok) {
      this.lastCommitT = f.t;
      this.lastCommitClass = cls;
      this.status = "recognized";
      out.commit = { gloss: top[0].gloss, class_index: cls, score: top[0].prob, t: f.t };
    } else this.status = reason === "unstable" ? "analyzing" : "uncertain";
    return out;
  }
}

function argmax(p: Float32Array): number {
  let b = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
  return b;
}

/** A WebSocket look-alike that runs the recognition engine locally (same message protocol). */
export class LocalRecognitionSocket {
  static readonly OPEN = 1;
  readyState = 0;
  bufferedAmount = 0;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private engine: LocalEngine | null = null;
  private queue: Promise<void> = Promise.resolve();
  private closed = false;

  constructor() {
    loadPredictor().then((p) => {
      if (this.closed) return;
      this.engine = new LocalEngine(p);
      this.readyState = 1;
      this.onopen?.();
      this.emit({ type: "hello", model: { ...p.meta.model, threshold: p.meta.threshold }, engine: { config: this.engine.cfg }, message: null });
    }).catch((e) => {
      console.error("failed to load the in-browser model", e);
      this.readyState = 3;
      this.onerror?.();
      this.onclose?.();
    });
  }

  private emit(m: unknown) {
    if (!this.closed) this.onmessage?.({ data: JSON.stringify(m) });
  }

  send(raw: string) {
    const msg = JSON.parse(raw);
    this.bufferedAmount += raw.length;
    this.queue = this.queue.then(async () => {
      this.bufferedAmount -= raw.length;
      const e = this.engine;
      if (!e || this.closed) return;
      const t0 = performance.now();
      if (msg.type === "frame") {
        const state = await e.push(msg as RawFrame);
        state.server_ms = performance.now() - t0;
        this.emit(state);
      } else if (msg.type === "config") {
        if (msg.width && msg.height) { e.width = msg.width; e.height = msg.height; }
        if (msg.engine) {
          const patch = Object.fromEntries(Object.entries(msg.engine).filter(([, v]) => v !== undefined));
          e.cfg = { ...e.cfg, ...patch } as EngineConfig;
          e.reset();
        }
        this.emit({ type: "config_ack" });
      } else if (msg.type === "reset") {
        e.reset();
        this.emit({ type: "reset_ack" });
      } else if (msg.type === "ping") this.emit({ type: "pong", t: msg.t });
    }).catch((err) => console.error(err));
  }

  close() {
    this.closed = true;
    this.readyState = 3;
    this.onclose?.();
  }
}

let predictorPromise: Promise<OnnxPredictor> | null = null;
export function loadPredictor(): Promise<OnnxPredictor> {
  predictorPromise ??= OnnxPredictor.load();
  predictorPromise.catch(() => (predictorPromise = null));
  return predictorPromise;
}
