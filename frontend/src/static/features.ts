// TypeScript port of backend/app/features/build.py (inference path, no augmentation).
// Used by the static (fully client-side) build. Parity with Python is enforced by
// features.test.ts against fixtures produced by scripts/export_static.py.
import type { RawFrame } from "../api/client";

export type FeatureConfig = { seq_len: number; use_face: boolean; use_velocity: boolean; max_gap: number };

const POSE_FEATURE_IDX = [0, 11, 12, 13, 14, 15, 16];
const L_SHOULDER = 11;
const R_SHOULDER = 12;
const N_POSE = 33;
const N_HAND = 21;
const MIDDLE_MCP = 9;

export class FeatureError extends Error {}

/** (T, N, D) array of numbers with NaN = missing, as nested arrays. */
type Seq = number[][][];

function toSeq(frames: RawFrame[], key: "pose" | "left" | "right" | "face", n: number, d: number): Seq {
  return frames.map((f) => {
    const a = f[key];
    if (!a) return Array.from({ length: n }, () => new Array(d).fill(NaN));
    return a.map((p) => p.slice(0, d));
  });
}

/** Interpolate interior gaps of at most maxGap frames (missing = NaN in element [0][0]). */
export function fillGaps(arr: Seq, maxGap: number): Seq {
  const out = arr.map((f) => f.map((p) => p.slice()));
  const T = arr.length;
  if (T === 0 || maxGap <= 0) return out;
  const miss = arr.map((f) => Number.isNaN(f[0][0]));
  let i = 0;
  while (i < T) {
    if (!miss[i]) { i++; continue; }
    let j = i;
    while (j < T && miss[j]) j++;
    if (i > 0 && j < T && j - i <= maxGap) {
      const a = arr[i - 1], b = arr[j];
      for (let k = i; k < j; k++) {
        const w = (k - (i - 1)) / (j - (i - 1));
        out[k] = a.map((p, pi) => p.map((v, di) => (1 - w) * v + w * b[pi][di]));
      }
    }
    i = j;
  }
  return out;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export type BodyFrame = {
  pose: Seq; left: Seq; right: Seq; face: Seq;
  present: { pose: boolean[]; left: boolean[]; right: boolean[]; face: boolean[] };
  t: number[];
};

export function toBodyFrame(frames: RawFrame[], width: number, height: number, maxGap = 5): BodyFrame {
  if (!frames.length) throw new FeatureError("empty sequence");
  const asp = height ? width / height : 1;
  let pose = toSeq(frames, "pose", N_POSE, 3);
  let left = toSeq(frames, "left", N_HAND, 3);
  let right = toSeq(frames, "right", N_HAND, 3);
  const nFace = frames.find((f) => f.face)?.face?.length ?? 20;
  let face = toSeq(frames, "face", nFace, 3);
  for (const a of [pose, left, right, face]) for (const f of a) for (const p of f) p[0] *= asp;
  for (const a of [left, right]) for (const f of a) for (const p of f) p[2] *= asp;
  [pose, left, right, face] = [pose, left, right, face].map((a) => fillGaps(a, maxGap));
  const pres = (a: Seq) => a.map((f) => !Number.isNaN(f[0][0]));
  const present = { pose: pres(pose), left: pres(left), right: pres(right), face: pres(face) };
  if (!present.pose.some(Boolean)) throw new FeatureError("no body (pose) detected in any frame");

  const mx: number[] = [], my: number[] = [], sw: number[] = [];
  pose.forEach((f, i) => {
    if (!present.pose[i]) return;
    const l = f[L_SHOULDER], r = f[R_SHOULDER];
    mx.push((l[0] + r[0]) / 2);
    my.push((l[1] + r[1]) / 2);
    sw.push(Math.hypot(l[0] - r[0], l[1] - r[1]));
  });
  const cx = median(mx), cy = median(my), scale = median(sw);
  if (!Number.isFinite(scale) || scale < 1e-3) throw new FeatureError("shoulders not resolved (zero shoulder width)");

  const norm = (a: Seq, presentArr: boolean[], dims: number): Seq =>
    a.map((f, i) => f.map((p) => {
      if (!presentArr[i]) return new Array(dims).fill(0);
      const o = [(p[0] - cx) / scale, (p[1] - cy) / scale];
      if (dims === 3) o.push(p[2] / scale);
      return o;
    }));
  const poseSel = pose.map((f) => POSE_FEATURE_IDX.map((k) => f[k]));
  let t = frames.map((f, i) => (typeof f.t === "number" ? f.t : i * 40));
  if (t.length > 1 && t.some((v, i) => i > 0 && v - t[i - 1] <= 0)) t = t.map((_, i) => i * 40);
  return {
    pose: norm(poseSel, present.pose, 2),
    left: norm(left, present.left, 3),
    right: norm(right, present.right, 3),
    face: norm(face, present.face, 2),
    present,
    t,
  };
}

function searchsortedRight(t: number[], v: number): number {
  let lo = 0, hi = t.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function resample(arr: Seq, present: boolean[], t: number[], tq: number[]): { out: Seq; pres: boolean[] } {
  const T = arr.length;
  if (T === 1) return { out: tq.map(() => arr[0].map((p) => (present[0] ? p.slice() : p.map(() => 0)))), pres: tq.map(() => present[0]) };
  const out: Seq = [];
  const pres: boolean[] = [];
  for (const q of tq) {
    const idx = Math.min(Math.max(searchsortedRight(t, q) - 1, 0), T - 2);
    const t0 = t[idx], t1 = t[idx + 1];
    const w = Math.min(Math.max((q - t0) / Math.max(t1 - t0, 1e-9), 0), 1);
    const both = present[idx] && present[idx + 1];
    const nearest = w < 0.5 ? idx : idx + 1;
    const p = both ? true : present[nearest];
    let f: number[][];
    if (both) f = arr[idx].map((pt, pi) => pt.map((v, di) => v * (1 - w) + arr[idx + 1][pi][di] * w));
    else f = arr[nearest].map((pt) => pt.slice());
    if (!p) f = f.map((pt) => pt.map(() => 0));
    out.push(f);
    pres.push(p);
  }
  return { out, pres };
}

function handShape(h: Seq, pres: boolean[]): { shape: number[][]; pos: number[][] } {
  const shape: number[][] = [], pos: number[][] = [];
  h.forEach((f, i) => {
    const w = f[0];
    const rel = f.map((p) => [p[0] - w[0], p[1] - w[1], p[2] - w[2]]);
    let size = Math.hypot(rel[MIDDLE_MCP][0], rel[MIDDLE_MCP][1]);
    if (!(size > 1e-6)) size = 1;
    const s: number[] = [];
    for (let k = 1; k < rel.length; k++) s.push(rel[k][0] / size, rel[k][1] / size, rel[k][2] / size);
    shape.push(pres[i] ? s : s.map(() => 0));
    pos.push(pres[i] ? [w[0], w[1]] : [0, 0]);
  });
  return { shape, pos };
}

const flat = (s: Seq) => s.map((f) => f.flat());

export function featurize(bf: BodyFrame, cfg: FeatureConfig): Float32Array {
  const L = cfg.seq_len;
  const t = bf.t;
  const tq = t.length > 1 ? Array.from({ length: L }, (_, i) => t[0] + ((t[t.length - 1] - t[0]) * i) / (L - 1)) : new Array(L).fill(t[0]);
  const pose = resample(bf.pose, bf.present.pose, t, tq);
  const left = resample(bf.left, bf.present.left, t, tq);
  const right = resample(bf.right, bf.present.right, t, tq);
  const lh = handShape(left.out, left.pres);
  const rh = handShape(right.out, right.pres);
  const blocks: number[][][] = [flat(pose.out), lh.shape, lh.pos, rh.shape, rh.pos];
  const blockPres: boolean[][] = [pose.pres, left.pres, left.pres, right.pres, right.pres];
  const flags: boolean[][] = [left.pres, right.pres];
  if (cfg.use_face) {
    const face = resample(bf.face, bf.present.face, t, tq);
    blocks.push(flat(face.out));
    blockPres.push(face.pres);
    flags.push(face.pres);
  }
  const vel: number[][][] = [];
  if (cfg.use_velocity) {
    blocks.forEach((b, bi) => {
      const pr = blockPres[bi];
      vel.push(b.map((row, i) => {
        const prev = i === 0 ? row : b[i - 1];
        const ok = pr[i] && (i === 0 ? pr[0] : pr[i - 1]);
        return row.map((v, k) => (ok ? (v - prev[k]) * (L / 16) : 0));
      }));
    });
  }
  const rows: number[][] = [];
  for (let i = 0; i < L; i++) {
    const r: number[] = [];
    for (const b of blocks) r.push(...b[i]);
    for (const b of vel) r.push(...b[i]);
    for (const f of flags) r.push(f[i] ? 1 : 0);
    rows.push(r);
  }
  const D = rows[0].length;
  const outArr = new Float32Array(L * D);
  rows.forEach((r, i) => outArr.set(r, i * D));
  return outArr;
}

export function featureDim(cfg: FeatureConfig): number {
  const stat = 14 + 60 + 2 + 60 + 2 + (cfg.use_face ? 40 : 0);
  return stat * (cfg.use_velocity ? 2 : 1) + 2 + (cfg.use_face ? 1 : 0);
}

export function buildFeatures(frames: RawFrame[], width: number, height: number, cfg: FeatureConfig): Float32Array {
  return featurize(toBodyFrame(frames, width, height, cfg.max_gap), cfg);
}

/** Frames with at least one hand (flag columns); all frames if none. Mirrors models/checkpoint.frame_mask. */
export function frameMask(x: Float32Array, cfg: FeatureConfig): boolean[] {
  const D = featureDim(cfg);
  const flagStart = D - (2 + (cfg.use_face ? 1 : 0));
  const m: boolean[] = [];
  for (let i = 0; i < cfg.seq_len; i++) m.push(x[i * D + flagStart] > 0.5 || x[i * D + flagStart + 1] > 0.5);
  return m.some(Boolean) ? m : m.map(() => true);
}
