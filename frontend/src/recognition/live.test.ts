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
