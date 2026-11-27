// The live page with a fake socket (docs/ISL_DESIGN.md 15.29, 15.30): the settings show the certified precision of the
// active policy (never the 0.98 target), the model card repeats it, a confirm-only policy says it is not tested well
// enough, an ungated pack says it is not calibrated, and the continuous-signing toggle is shown only when the server
// sends sliding rows and then asks the server for sliding mode.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { UiLangProvider } from "../i18n";
import type { Certification, CertificationByPolicy } from "../recognition/live";

vi.mock("../mediapipe/holistic", () => ({
  DEFAULT_FACE_IDX: [],
  loadHolistic: vi.fn(async () => ({ landmarker: { detectForVideo: vi.fn(), close: vi.fn() }, delegate: "CPU" })),
  toRawFrame: vi.fn(),
}));
vi.mock("../state/language", () => {
  const info = {
    id: "isl", kind: "language-pack", sign_language: "ISL", standard: "ISL", display_name: { en: "Indian Sign Language", hi: "भारतीय सांकेतिक भाषा" },
    output_languages: ["en", "hi"], capabilities: ["words"], profile: "commercial-safe", version: "1", licences: [], status: "ready", message: null,
  };
  const ctx = { languages: [info], id: "isl", info, isAsl: false, loading: false, error: null, setId: () => {}, reload: () => {} };
  return { useSignLanguage: () => ctx, ASL_ID: "asl-wlasl100", ISL_ID: "isl" };
});

import LivePage from "./LivePage";

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

const certified: Certification = { rule: "certified", certified_precision: 0.953, worst_source: "include", floor: 0.95, target: 0.98, confirm_tap: true };
const confirmOnly: Certification = { rule: "confirm_only", certified_precision: null, worst_source: "include", floor: 0.95, target: 0.98, confirm_tap: true };
const ungated: Certification = { rule: "ungated", certified_precision: null, confirm_tap: true };
const all = (c: Certification) => ({ strict: c, balanced: c, lenient: c });
const model = { name: "ISL pack", version: "1.0", profile: "commercial-safe", num_classes: 263 };

beforeEach(() => {
  FakeSocket.last = null;
  localStorage.clear();
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [] })) },
  });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function startWith(certs: CertificationByPolicy) {
  const view = render(<MemoryRouter><UiLangProvider initial="en"><LivePage /></UiLangProvider></MemoryRouter>);
  await act(async () => {
    fireEvent.click(screen.getAllByRole("button", { name: "Start camera" })[0]);
  });
  act(() => {
    fireEvent.click(screen.getByRole("button", { name: "Start recognition" }));
  });
  const ws = FakeSocket.last!;
  act(() => ws.onopen?.());
  act(() => ws.receive({ type: "hello", model, engine: { config: {}, certification_by_policy: certs }, language: null }));
  fireEvent.click(screen.getByRole("button", { name: "Settings" }));
  return { ws, view };
}

describe("LivePage: the app default for a language pack", () => {
  it("shows the certified worst-source precision of strict, in the settings and on the model card", async () => {
    const { view } = await startWith({ segment: all(certified), sliding: null });
    expect(screen.getByTestId("policy-precision")).toHaveTextContent("Strict: right about 95 of 100 times when it answers");
    expect(screen.getByTestId("model-precision")).toHaveTextContent("right about 95 of 100 times");
    expect(document.body).not.toHaveTextContent("98 of 100");
    // no classifier: continuous signing is not offered
    expect(screen.queryByRole("switch", { name: "Recognise while I keep signing" })).toBeNull();
    view.unmount();
  });

  it("a confirm-only policy says it is not tested well enough; an ungated pack says it is not calibrated", async () => {
    const a = await startWith({ segment: { ...all(certified), strict: confirmOnly }, sliding: null });
    expect(screen.getByTestId("policy-precision")).toHaveTextContent("Strict: not tested well enough yet");
    a.view.unmount();
    const b = await startWith({ segment: all(ungated), sliding: null });
    expect(screen.getByTestId("policy-precision")).toHaveTextContent("Strict: not calibrated (developer settings)");
    expect(screen.getByTestId("policy-precision")).not.toHaveTextContent("Start recognition");
    b.view.unmount();
  });

  it("the continuous toggle appears with sliding rows and asks the server for sliding mode", async () => {
    const { ws, view } = await startWith({ segment: all(certified), sliding: { ...all(certified), strict: confirmOnly } });
    // the hello alone does not change the engine settings, so no extra config message is sent
    expect(ws.sent.filter((m) => m.type === "config")).toHaveLength(1);
    expect(ws.sent[0]).toMatchObject({ type: "config", engine: { mode: "segment", policy: "strict" } });
    const toggle = screen.getByRole("switch", { name: "Recognise while I keep signing" });
    expect(toggle).toHaveAttribute("aria-checked", "false");
    fireEvent.click(toggle);
    expect(ws.sent.at(-1)).toMatchObject({ type: "config", engine: { mode: "sliding", policy: "strict" } });
    // continuous mode shows the sliding row of the policy: strict is confirm-only there
    expect(screen.getByTestId("policy-precision")).toHaveTextContent("Strict: not tested well enough yet");
    view.unmount();
  });
});
