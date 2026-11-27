// The live hook with a fake socket (docs/ISL_DESIGN.md 15.29): a language-pack word that needs a tap waits as
// "pending" and only a tap adds it to the sentence; "wrong" never adds it and logs the feedback; ASL commits are added
// as before; the certification of every policy is read from the server's hello.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { DEFAULT_SETTINGS, type CertificationByPolicy, type LiveSettings } from "./live";
import { useLiveRecognition } from "./useLiveRecognition";

class FakeSocket {
  static last: FakeSocket | null = null;
  static OPEN = 1;
  readyState = 1;
  sent: Record<string, unknown>[] = [];
  bufferedAmount = 0;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) {
    FakeSocket.last = this;
  }
  send(s: string) {
    this.sent.push(JSON.parse(s));
  }
  close() {
    this.readyState = 3;
  }
  receive(msg: unknown) {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }
}

const cert = { rule: "certified", certified_precision: 0.96, worst_source: "include", confirm_tap: true } as const;
const certs: CertificationByPolicy = {
  segment: { strict: cert, balanced: cert, lenient: cert },
  sliding: { strict: { ...cert, rule: "confirm_only", certified_precision: null }, balanced: cert, lenient: cert },
};
const commit = { gloss: "water", class_index: 1, score: 0.97, t: 5, concept_id: "isl.water", text: { en: "water", hi: "पानी" }, segment_id: 3, chosen_by_user: false, certification: cert, needs_confirm: true };
const segResult = { segment_id: 3, accepted: true, decision: "accept", reason: "ok", frames: 30, top: [{ concept_id: "isl.water", class_index: 1, gloss: "water", prob: 0.97 }], certification: cert, needs_confirm: true };

beforeEach(() => {
  FakeSocket.last = null;
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
});
afterEach(() => vi.unstubAllGlobals());

function start(settings: LiveSettings = DEFAULT_SETTINGS, opts = { pack: true, spell: false, sliding: false }) {
  const h = renderHook((p: { s: LiveSettings; o: typeof opts }) => useLiveRecognition(p.s, "isl", p.o), { initialProps: { s: settings, o: opts } });
  act(() => h.result.current.actions.startRecognition());
  const ws = FakeSocket.last!;
  act(() => ws.onopen?.());
  act(() => ws.receive({ type: "hello", model: { name: "isl" }, engine: { config: {}, certification_by_policy: certs }, language: null }));
  return { h, ws };
}

describe("useLiveRecognition: a confirm tap before a word is added", () => {
  it("reads the certification of every policy from the hello", () => {
    const { h } = start();
    expect(h.result.current.certs).toEqual(certs);
  });

  it("a word that needs a tap waits; only 'add' puts it in the sentence", () => {
    const { h } = start();
    act(() => FakeSocket.last!.receive({ type: "state", t: 1, status: "recognized", hands: false, pose: true, threshold: 0.9, server_ms: 1, segment_result: segResult, commit }));
    expect(h.result.current.transcript).toEqual([]);
    expect(h.result.current.pending?.commit.concept_id).toBe("isl.water");
    act(() => h.result.current.actions.confirmPending());
    expect(h.result.current.pending).toBeNull();
    expect(h.result.current.transcript.map((x) => x.concept_id)).toEqual(["isl.water"]);
    expect(h.result.current.resolved?.how).toBe("chosen");
  });

  it("'wrong word' never adds it and sends the feedback for that sign", () => {
    const { h, ws } = start();
    act(() => ws.receive({ type: "state", t: 1, status: "recognized", hands: false, pose: true, threshold: 0.9, server_ms: 1, segment_result: segResult, commit }));
    act(() => h.result.current.actions.rejectPending());
    expect(h.result.current.transcript).toEqual([]);
    expect(h.result.current.pending).toBeNull();
    expect(ws.sent.at(-1)).toEqual({ type: "reject", segment_id: 3 });
  });

  it("the next recognised sign replaces an unconfirmed word; a too-short movement does not", () => {
    const { h, ws } = start();
    act(() => ws.receive({ type: "state", t: 1, status: "recognized", hands: false, pose: true, threshold: 0.9, server_ms: 1, segment_result: segResult, commit }));
    act(() => ws.receive({ type: "state", t: 2, status: "waiting_for_hands", hands: false, pose: true, threshold: 0.9, server_ms: 1, segment_result: { accepted: false, reason: "too_short", frames: 3 } }));
    expect(h.result.current.pending).not.toBeNull();
    act(() => ws.receive({ type: "state", t: 3, status: "not_sure", hands: false, pose: true, threshold: 0.9, server_ms: 1, segment_result: { segment_id: 4, accepted: false, decision: "abstain", reason: "out_of_vocabulary", frames: 30, top: [] } }));
    expect(h.result.current.pending).toBeNull();
    expect(h.result.current.transcript).toEqual([]);
  });

  it("a continuous-mode commit (no sign result) waits too", () => {
    const { h, ws } = start();
    act(() => ws.receive({ type: "state", t: 1, status: "recognized", hands: true, pose: true, threshold: 0.9, server_ms: 1, commit: { ...commit, segment_id: null } }));
    expect(h.result.current.pending?.continuous).toBe(true);
    act(() => h.result.current.actions.rejectPending());
    expect(ws.sent.some((m) => m.type === "reject")).toBe(false); // nothing to reject on the server
  });

  it("a word the user picked from 'did you mean' is added at once", () => {
    const { h, ws } = start();
    act(() => ws.receive({ type: "choice_ack", commit: { ...commit, chosen_by_user: true, needs_confirm: false } }));
    expect(h.result.current.transcript).toHaveLength(1);
    expect(h.result.current.pending).toBeNull();
  });

  it("ASL commits (no certification) are added as before", () => {
    const { h, ws } = start(DEFAULT_SETTINGS, { pack: false, spell: false, sliding: false });
    act(() => ws.receive({ type: "state", t: 1, status: "recognized", hands: false, pose: true, threshold: 0.6, server_ms: 1, commit: { gloss: "book", class_index: 2, score: 0.9, t: 1 } }));
    expect(h.result.current.transcript.map((x) => x.gloss)).toEqual(["book"]);
    expect(h.result.current.pending).toBeNull();
    expect(ws.sent[0]).toMatchObject({ type: "config", engine: { mode: "segment", threshold: null, min_margin: 0.1 } });
    expect(ws.sent[0].engine).not.toHaveProperty("policy");
  });

  it("asks for sliding mode only when continuous signing is on and the pack offers it", () => {
    const on = { ...DEFAULT_SETTINGS, engine: { ...DEFAULT_SETTINGS.engine, continuous: true } };
    const { ws } = start(on, { pack: true, spell: false, sliding: false });
    expect(ws.sent[0]).toMatchObject({ type: "config", engine: { mode: "segment", policy: "strict" } });
    const { ws: ws2 } = start(on, { pack: true, spell: false, sliding: true });
    expect(ws2.sent[0]).toMatchObject({ type: "config", engine: { mode: "sliding", policy: "strict" } });
  });
});
