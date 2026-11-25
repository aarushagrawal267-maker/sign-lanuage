import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { speak, speechAvailable, voicesFor } from "./speech";

type V = { lang: string; name: string };
let voices: V[] = [];
const spoken: { text: string; lang: string; voice: V | null }[] = [];

class FakeUtterance {
  lang = "";
  voice: V | null = null;
  rate = 1;
  onend: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public text: string) {}
}

beforeEach(() => {
  spoken.length = 0;
  vi.stubGlobal("SpeechSynthesisUtterance", FakeUtterance);
  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    value: {
      getVoices: () => voices,
      cancel: vi.fn(),
      speak: (u: FakeUtterance) => { spoken.push({ text: u.text, lang: u.lang, voice: u.voice }); u.onend?.(); },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    },
  });
});
afterEach(() => {
  vi.unstubAllGlobals();
  delete (window as unknown as { speechSynthesis?: unknown }).speechSynthesis;
});

describe("speech", () => {
  it("prefers an exact hi-IN voice, then any Hindi voice", () => {
    voices = [{ lang: "en-US", name: "en" }, { lang: "hi", name: "hi-generic" }, { lang: "hi_IN", name: "hi-underscore" }, { lang: "hi-IN", name: "hi" }];
    expect(voicesFor("hi-IN").map((v) => v.name)).toEqual(["hi-underscore", "hi", "hi-generic"]);
    expect(speechAvailable("hi-IN")).toBe(true);
  });

  it("says Hindi is unavailable without a Hindi voice, but English works with any English voice", () => {
    voices = [{ lang: "en-US", name: "en" }];
    expect(speechAvailable("hi-IN")).toBe(false);
    expect(speechAvailable("en-IN")).toBe(true);
  });

  it("speaks with the requested language and voice and resolves when done", async () => {
    voices = [{ lang: "hi-IN", name: "hi" }];
    await speak("  पानी ", "hi-IN");
    expect(spoken).toEqual([{ text: "पानी", lang: "hi-IN", voice: voices[0] }]);
    await speak("   ", "en-IN");
    expect(spoken).toHaveLength(1);
  });

  it("does nothing without speechSynthesis", async () => {
    delete (window as unknown as { speechSynthesis?: unknown }).speechSynthesis;
    expect(speechAvailable("en-IN")).toBe(false);
    await expect(speak("water", "en-IN")).resolves.toBeUndefined();
  });
});
