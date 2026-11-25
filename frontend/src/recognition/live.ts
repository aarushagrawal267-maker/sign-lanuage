// Pure helpers for the live recognition loop (unit-tested in live.test.ts).
import type { Candidate, HiReview } from "../api/client";
import type { LiveStatus } from "./status";

export type Policy = "strict" | "balanced" | "lenient";

export type EngineSettings = {
  mode: "segment" | "sliding";
  threshold: number | null; // null = model's validation-derived threshold
  min_margin: number;
  cooldown_ms: number;
  end_absent_ms: number;
  /** Language packs (ISL): how readily a word is shown without asking (maps to the pack's fitted thresholds). */
  policy?: Policy;
  /** Language packs (ISL): also search the user's own signs. */
  use_my_signs?: boolean;
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
  engine: { mode: "segment", threshold: null, min_margin: 0.1, cooldown_ms: 700, end_absent_ms: 400, policy: "balanced", use_my_signs: true },
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
  provisional?: { top: Candidate[]; feature_ms: number; inference_ms: number; streak?: number; reason?: string };
  segment_result?: SegmentResult;
  commit?: Commit;
  spell?: SpellState;
};

export type Decision = "accept" | "suggest" | "abstain";

export type SegmentResult = {
  accepted: boolean; reason: string; frames: number; duration_ms?: number; agreement?: string; top?: Candidate[];
  feature_ms?: number; inference_ms?: number;
  // language packs (ISL) only
  segment_id?: number; decision?: Decision; confidence?: number; candidates?: Candidate[]; spell?: SpellState;
};

export type Commit = {
  gloss: string; class_index: number; score: number; t: number;
  // language packs and ASL with a Hindi lexicon
  concept_id?: string; text?: { en?: string | null; hi?: string | null }; hi_review?: HiReview; source?: string;
  segment_id?: number | null; chosen_by_user?: boolean;
};

export type SpellState = {
  letters: { letter: string; prob: number; start?: number; end?: number }[];
  buffer: string;
  suggestions: { word: string; concept_id?: string | null; score: number; in_lexicon?: boolean }[];
  holds?: number;
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

const ENGINE_STATES = new Set(["no_body", "waiting_for_hands", "collecting", "analyzing", "recognized", "uncertain", "not_sure", "spelling", "unavailable"]);

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

export type TranscriptItem = {
  id: number; gloss: string; score: number; at: number;
  concept_id?: string; text?: { en?: string | null; hi?: string | null }; hi_review?: HiReview; source?: string;
  segment_id?: number | null; chosen?: boolean;
};

/** A transcript entry from a server commit (accepted, or chosen by the user from "did you mean"). */
export function transcriptItem(id: number, c: Commit, at = Date.now()): TranscriptItem {
  return {
    id, gloss: c.gloss, score: c.score, at, concept_id: c.concept_id, text: c.text, hi_review: c.hi_review ?? null,
    source: c.source, segment_id: c.segment_id ?? null, chosen: !!c.chosen_by_user,
  };
}

/** A transcript entry for a candidate the user tapped when the server has no segment to confirm (ASL). */
export function transcriptFromCandidate(id: number, c: Candidate, at = Date.now()): TranscriptItem {
  return {
    id, gloss: c.gloss, score: c.prob, at, concept_id: c.concept_id, text: c.text ?? { en: c.gloss }, hi_review: c.hi_review ?? null,
    source: c.source, segment_id: null, chosen: true,
  };
}

/** English and Hindi text of a candidate or transcript item (Hindi null when there is none). */
export function wordText(x: { gloss: string; text?: { en?: string | null; hi?: string | null } }): { en: string; hi: string | null } {
  return { en: (x.text?.en || x.gloss || "").trim(), hi: x.text?.hi?.trim() || null };
}

/**
 * Three-way outcome of a finished sign. Language packs send it; the classic ASL engine only says accepted or not, so
 * a rejected ASL sign that still has ranked guesses becomes "suggest" (the user may pick one) and one without
 * guesses (too short, no body) becomes "abstain".
 */
export function segmentDecision(r: SegmentResult | null | undefined): Decision | null {
  if (!r) return null;
  if (r.decision === "accept" || r.decision === "suggest" || r.decision === "abstain") return r.decision;
  if (r.accepted) return "accept";
  if (r.top?.length && r.reason !== "too_short") return "suggest";
  return "abstain";
}

/** Words for the sentence endpoint (POST /api/languages/{lang}/sentence, "words" form), in signing order. */
export function sentenceWords(items: TranscriptItem[]) {
  return items.map((t) => {
    const w = wordText(t);
    return { concept_id: t.concept_id ?? null, en: w.en, hi: w.hi };
  });
}

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

export type EngineMode = "segment" | "sliding" | "spell";

/**
 * Engine config sent over the socket. The classic ASL engine gets exactly the fields it always got. A language pack
 * (ISL) gets its own decision policy instead of the ASL threshold sliders (its thresholds are fitted per pack), the
 * My signs switch, and "spell" when spelling mode is on.
 */
export function engineConfigPayload(e: EngineSettings, opts: { pack?: boolean; spell?: boolean } = {}) {
  if (!opts.pack) {
    return {
      mode: e.mode,
      threshold: e.threshold,
      min_margin: e.min_margin,
      cooldown_ms: e.cooldown_ms,
      end_absent_ms: e.end_absent_ms,
    };
  }
  const mode: EngineMode = opts.spell ? "spell" : "segment";
  return {
    mode,
    threshold: null,
    cooldown_ms: e.cooldown_ms,
    end_absent_ms: e.end_absent_ms,
    policy: e.policy ?? "balanced",
    use_my_signs: e.use_my_signs ?? true,
  };
}
