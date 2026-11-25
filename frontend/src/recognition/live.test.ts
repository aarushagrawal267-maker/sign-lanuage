import { describe, expect, it } from "vitest";
import { RollingStat, deriveStatus, type Inputs } from "./live";

const base: Inputs = { camera: "active", mediapipe: "ready", recognizing: true, paused: false, socket: "open", modelAvailable: true, serverStatus: "collecting" };

describe("deriveStatus", () => {
  it("prioritises hardware problems over everything else", () => {
    expect(deriveStatus({ ...base, camera: "inactive" })).toBe("camera_inactive");
    expect(deriveStatus({ ...base, camera: "error" })).toBe("camera_error");
    expect(deriveStatus({ ...base, camera: "starting" })).toBe("camera_starting");
    expect(deriveStatus({ ...base, mediapipe: "loading" })).toBe("mediapipe_loading");
    expect(deriveStatus({ ...base, mediapipe: "error" })).toBe("mediapipe_error");
  });
  it("shows preview / paused before engine states", () => {
    expect(deriveStatus({ ...base, recognizing: false })).toBe("preview");
    expect(deriveStatus({ ...base, paused: true })).toBe("paused");
  });
  it("reports unavailable without a model or connection", () => {
    expect(deriveStatus({ ...base, modelAvailable: false })).toBe("unavailable");
    expect(deriveStatus({ ...base, socket: "error" })).toBe("unavailable");
    expect(deriveStatus({ ...base, socket: "connecting" })).toBe("connecting");
    expect(deriveStatus({ ...base, modelAvailable: null })).toBe("connecting");
  });
  it("passes through engine states and ignores unknown ones", () => {
    for (const s of ["no_body", "waiting_for_hands", "collecting", "analyzing", "recognized", "uncertain"]) {
      expect(deriveStatus({ ...base, serverStatus: s })).toBe(s);
    }
    expect(deriveStatus({ ...base, serverStatus: "bogus" })).toBe("waiting_for_hands");
  });
});

describe("RollingStat", () => {
  it("averages within the window and computes a rate", () => {
    const r = new RollingStat(1000);
    r.push(10, 0);
    r.push(20, 500);
    r.push(30, 1000);
    expect(r.mean()).toBe(20);
    expect(r.rate(1000)).toBeCloseTo(2, 5);
    r.push(40, 2200);
    expect(r.mean()).toBe(40);
  });
});

import { engineConfigPayload, segmentDecision, sentenceWords, transcriptFromCandidate, transcriptItem, wordText, DEFAULT_SETTINGS } from "./live";
import { shouldStopSign, takeSummary, trimTake } from "./useLandmarkRecorder";

describe("segmentDecision", () => {
  it("uses the language pack's decision when present", () => {
    expect(segmentDecision({ accepted: false, reason: "ambiguous", frames: 30, decision: "suggest" })).toBe("suggest");
    expect(segmentDecision({ accepted: false, reason: "out_of_vocabulary", frames: 30, decision: "abstain", top: [{ class_index: 1, gloss: "x", prob: 0.2 }] })).toBe("abstain");
    expect(segmentDecision({ accepted: true, reason: "ok", frames: 30, decision: "accept" })).toBe("accept");
  });
  it("maps the classic ASL result: rejected with guesses -> suggest, without -> abstain", () => {
    expect(segmentDecision({ accepted: true, reason: "ok", frames: 30, top: [{ class_index: 1, gloss: "book", prob: 0.9 }] })).toBe("accept");
    expect(segmentDecision({ accepted: false, reason: "low_confidence", frames: 30, top: [{ class_index: 1, gloss: "book", prob: 0.3 }] })).toBe("suggest");
    expect(segmentDecision({ accepted: false, reason: "too_short", frames: 3 })).toBe("abstain");
    expect(segmentDecision(null)).toBeNull();
  });
});

describe("engineConfigPayload", () => {
  it("sends the classic ASL engine exactly the fields it always got", () => {
    expect(engineConfigPayload(DEFAULT_SETTINGS.engine)).toEqual({ mode: "segment", threshold: null, min_margin: 0.1, cooldown_ms: 700, end_absent_ms: 400 });
  });
  it("sends a language pack its policy, My signs switch and spelling mode, never the ASL threshold", () => {
    const e = { ...DEFAULT_SETTINGS.engine, threshold: 0.9, mode: "sliding" as const, policy: "strict" as const };
    expect(engineConfigPayload(e, { pack: true })).toEqual({ mode: "segment", threshold: null, cooldown_ms: 700, end_absent_ms: 400, policy: "strict", use_my_signs: true });
    expect(engineConfigPayload(e, { pack: true, spell: true }).mode).toBe("spell");
    // settings stored by an older version (no policy field) still work
    const old = { mode: "segment" as const, threshold: null, min_margin: 0.1, cooldown_ms: 700, end_absent_ms: 400 };
    expect(engineConfigPayload(old, { pack: true })).toMatchObject({ policy: "balanced", use_my_signs: true });
  });
});

describe("transcript helpers", () => {
  it("keeps English, Hindi and the segment of a commit", () => {
    const it1 = transcriptItem(1, { gloss: "water", class_index: 5, score: 0.9, t: 1, concept_id: "isl.water", text: { en: "water", hi: "पानी" }, hi_review: "curated", segment_id: 7, chosen_by_user: false }, 0);
    expect(it1).toMatchObject({ id: 1, concept_id: "isl.water", segment_id: 7, chosen: false, hi_review: "curated" });
    expect(wordText(it1)).toEqual({ en: "water", hi: "पानी" });
    const it2 = transcriptFromCandidate(2, { class_index: 3, gloss: "book", prob: 0.4 }, 0);
    expect(it2).toMatchObject({ chosen: true, segment_id: null });
    expect(wordText(it2)).toEqual({ en: "book", hi: null });
    expect(sentenceWords([it1, it2])).toEqual([{ concept_id: "isl.water", en: "water", hi: "पानी" }, { concept_id: null, en: "book", hi: null }]);
  });
});

describe("recording helpers (My signs)", () => {
  const f = (t: number, hand: boolean) => ({ t, pose: null, left: hand ? [[0, 0, 0]] : null, right: null, face: null });
  it("stops a sign after the hands are down long enough, or at the time limit", () => {
    expect(shouldStopSign({ startT: 0, lastHandT: 1000, now: 1300, endMs: 450, maxMs: 3800 })).toBe(false);
    expect(shouldStopSign({ startT: 0, lastHandT: 1000, now: 1450, endMs: 450, maxMs: 3800 })).toBe(true);
    expect(shouldStopSign({ startT: 0, lastHandT: 3700, now: 3800, endMs: 450, maxMs: 3800 })).toBe(true);
  });
  it("drops the hands-down tail and summarises a take", () => {
    const frames = [f(0, false), f(40, true), f(80, true), f(120, false), f(200, false), f(400, false)];
    const kept = trimTake(frames);
    expect(kept.map((x) => x.t)).toEqual([0, 40, 80, 120, 200]);
    expect(trimTake([f(0, false)])).toEqual([]);
    const s = takeSummary({ width: 640, height: 480, frames: kept });
    expect(s.frames).toBe(5);
    expect(s.durationMs).toBe(200);
    expect(s.handRate).toBeCloseTo(0.4);
  });
});
