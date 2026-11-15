// Thin typed client for the FastAPI backend (proxied at /api by Vite).

export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
/** Static build: no backend. Read-only data comes from JSON snapshots; the model runs in the browser. */
export const IS_STATIC = import.meta.env.VITE_STATIC === "1";
const STATIC_MSG = "Not available in the online demo (it runs entirely in your browser). Run the project locally to download data, train or analyse files.";

const staticCache = new Map<string, Promise<unknown>>();
function staticJson(key: string): Promise<unknown> {
  if (!staticCache.has(key)) {
    staticCache.set(key, fetch(`${import.meta.env.BASE_URL}static-data/api/${key}.json`).then((r) => {
      if (!r.ok) throw new ApiError(404, "Not included in the online demo");
      return r.json();
    }));
  }
  return staticCache.get(key)!;
}

async function staticGet<T>(path: string): Promise<T> {
  const [p, qs] = path.split("?");
  const key = p.replace(/^\/api\//, "").replace(/^\/+|\/+$/g, "").replace(/\//g, "__");
  if (key === "dataset__videos") {
    const q = new URLSearchParams(qs ?? "");
    let rows = (await staticJson(key)) as Record<string, unknown>[];
    for (const f of ["class_index", "signer_id"]) if (q.has(f)) rows = rows.filter((r) => String(r[f]) === q.get(f));
    for (const f of ["status", "split"]) if (q.has(f)) rows = rows.filter((r) => r[f] === q.get(f));
    const off = Number(q.get("offset") ?? 0), lim = Number(q.get("limit") ?? 50);
    return { total: rows.length, items: rows.slice(off, off + lim) } as T;
  }
  if (key.startsWith("dataset__videos__")) throw new ApiError(404, "Not included in the online demo (WLASL data is not redistributed)");
  if (key.startsWith("jobs")) return [] as T;
  return (await staticJson(key)) as T;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

const staticDenied = <T>(): Promise<T> => Promise.reject(new ApiError(403, STATIC_MSG));

export const api = IS_STATIC ? {
  get: staticGet,
  post: <T>(_path: string, _body?: unknown) => staticDenied<T>(),
  del: <T>(_path: string) => staticDenied<T>(),
  upload: <T>(_path: string, _form: FormData) => staticDenied<T>(),
} : {
  get: <T>(path: string) => fetch(API_BASE + path).then((r) => parse<T>(r)),
  post: <T>(path: string, body?: unknown) =>
    fetch(API_BASE + path, {
      method: "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => parse<T>(r)),
  del: <T>(path: string) => fetch(API_BASE + path, { method: "DELETE" }).then((r) => parse<T>(r)),
  upload: <T>(path: string, form: FormData) =>
    fetch(API_BASE + path, { method: "POST", body: form }).then((r) => parse<T>(r)),
};

export function wsUrl(path: string): string {
  if (API_BASE) return API_BASE.replace(/^http/, "ws") + path;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

// ---------------------------------------------------------------- types
export type TopK = { class_index: number; gloss: string; prob: number };

export type ModelInfo = {
  experiment_id: number;
  name: string;
  architecture: string;
  split: string | null;
  is_synthetic: boolean;
  val_top1: number | null;
  threshold: number;
  seq_len: number;
  num_classes: number;
};

export type ExpBrief = {
  id: number;
  name: string;
  architecture: string;
  split_name: string | null;
  finished_at: string | null;
  is_synthetic: boolean;
  n_params: number | null;
  val: { top1: number; top5: number; macro_f1: number; n: number } | null;
  test: { top1: number; top5: number; macro_f1: number; n: number } | null;
  latency?: { mean_ms: number; p95_ms: number; device: string } | null;
};

export type Overview = {
  vocabulary_size: number;
  video_status: Record<string, number>;
  total_videos: number;
  landmark_sequences: number;
  samples_by_official_split: Record<string, number>;
  experiments: Record<string, number>;
  active_model: ExpBrief | null;
  latest_experiment: ExpBrief | null;
  running_jobs: { id: number; kind: string; progress: number; message: string | null }[];
  recent_events: { gloss: string; score: number; source: string; created_at: string }[];
};

export type SystemInfo = {
  python: string;
  platform: string;
  torch: string;
  cuda_available: boolean;
  cuda_device: string | null;
  mediapipe: string | null;
  holistic_model_present: boolean;
  data_dir: string;
  public_demo?: boolean;
  static?: boolean;
};

export type Job = {
  id: number;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string | null;
  params: Record<string, unknown>;
  result: Record<string, unknown>;
  created_at: string;
  finished_at: string | null;
};

export type PerClass = { class_index: number; gloss: string; precision: number; recall: number; f1: number; support: number };

export type Metrics = {
  n: number;
  top1: number;
  top5: number;
  macro_f1: number;
  classes_evaluated: number;
  per_class: PerClass[];
  confusion_matrix: number[][];
  top_confusions: { true: string; pred: string; count: number }[];
  per_signer: Record<string, { n: number; top1: number }>;
  confidence_curve: { threshold: number; coverage: number; accuracy: number | null }[];
  loss?: number;
  latency?: { device: string; mean_ms: number; p95_ms: number };
  evaluated_at?: string;
  recommended_threshold?: number;
  best_epoch?: number;
  train_seconds?: number;
  partition?: string;
  error?: string;
};

export type HistoryRow = {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  val_macro_f1: number;
  lr: number;
  seconds: number;
};

export type ExperimentSummary = {
  id: number;
  name: string;
  status: string;
  architecture: string;
  split_name: string | null;
  dataset_version: string | null;
  created_at: string;
  finished_at: string | null;
  is_active: boolean;
  is_synthetic: boolean;
  device: string | null;
  n_params: number | null;
  seq_len: number | null;
  epochs_run: number;
  epochs_planned: number | null;
  last_epoch: HistoryRow | null;
  error: string | null;
  checkpoint_path: string | null;
  val: { top1: number; top5: number; macro_f1: number; n: number; best_epoch: number; recommended_threshold: number } | null;
  test: { top1: number; top5: number; macro_f1: number; n: number; evaluated_at: string } | null;
};

export type ExperimentDetail = ExperimentSummary & {
  config: Record<string, unknown> & { features: Record<string, unknown> };
  feature_config: Record<string, unknown>;
  data_summary: Record<string, unknown>;
  history: HistoryRow[];
  val_metrics: Metrics | Record<string, never>;
  test_metrics: Metrics | Record<string, never>;
  split: { name: string; kind: string; signer_independent: boolean; leakage_passed: boolean; n_signers: Record<string, number> } | null;
};

export type SplitInfo = {
  id: number;
  name: string;
  kind: string;
  signer_independent: boolean;
  seed: number | null;
  config: Record<string, unknown>;
  leakage_report: {
    n_videos: Record<string, number>;
    n_signers: Record<string, number>;
    signer_overlap: Record<string, { count: number; signers: number[] }>;
    content_hash_overlap: Record<string, number>;
    source_file_overlap: Record<string, number>;
    eval_videos_with_train_signers: Record<string, { videos: number; videos_with_train_signer: number }>;
    classes_missing: Record<string, number[]>;
    signer_independent: boolean;
    passed: boolean;
  };
};

export type VocabEntry = {
  class_index: number;
  gloss: string;
  annotated: number;
  valid: number;
  with_landmarks: number;
  train_official: number;
  signers: number;
  variations: number;
  local_example: string | null;
  val: PerClass | null;
  test: PerClass | null;
};

export type InventoryRow = {
  class_index: number;
  gloss: string;
  annotated: number;
  valid: number;
  unavailable: number;
  pending: number;
  landmarks: number;
  signers: number[];
  split: Record<string, number>;
  variations: number;
  median_duration_s: number | null;
  resolutions: Record<string, number>;
  fps: Record<string, number>;
};

export type DatasetSummary = {
  videos: number;
  status_counts: Record<string, number>;
  official_split: Record<string, number>;
  hosts: ({ host: string; total: number } & Record<string, number | string>)[];
  signers: { signer_id: number; videos: number; split: Record<string, number> }[];
  variation_counts: Record<string, number>;
  duration_histogram: { from: number; to: number; count: number }[] | null;
  sequence_length: { median_frames: number; p10_frames: number; p90_frames: number; recommended_seq_len: number; basis: string } | null;
  landmarks: { status_counts: Record<string, number>; mean_pose_rate: number | null; mean_any_hand_rate: number | null; mean_face_rate: number | null };
  top_errors: { error: string; count: number }[];
};

export type VideoRow = {
  video_id: string;
  gloss: string;
  class_index: number;
  signer_id: number;
  split: string;
  source: string;
  host: string;
  url: string;
  status: string;
  error: string | null;
  frame_start: number;
  frame_end: number;
  variation_id: number;
  fps: number | null;
  width: number | null;
  height: number | null;
  duration_s: number | null;
  clip_frames: number | null;
  has_file: boolean;
  landmarks: { status: string; frames: number; any_hand_rate: number; pose_rate: number; face_rate: number } | null;
};

export type RawFrame = {
  t: number;
  pose: number[][] | null;
  left: number[][] | null;
  right: number[][] | null;
  face: number[][] | null;
};

export type LandmarkClip = {
  video_id?: string;
  gloss?: string | null;
  signer_id?: number | null;
  width: number;
  height: number;
  frame_step: number;
  frames: RawFrame[];
  quality: Record<string, number | boolean>;
};

export type FeatureSpec = {
  raw: {
    extractor_version: string;
    face_indices: number[];
    pose: { count: number };
    hand: { count: number };
    face: { count: number };
  };
  feature_config: { seq_len: number; use_face: boolean; use_velocity: boolean; max_gap: number };
  feature_dim: number;
  layout: { name: string; start: number; size: number }[];
  from_active_model: boolean;
};

export type DatasetStatus = {
  annotations_loaded: boolean;
  total_videos: number;
  status_counts: Record<string, number>;
  landmark_counts: Record<string, number>;
  provenance: { ingested_at: string; checks: Record<string, unknown>; files: Record<string, { url?: string; sha256: string }> } | null;
  synthetic: Record<string, unknown> | null;
};
