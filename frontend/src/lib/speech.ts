// Speak-aloud with the browser's own speechSynthesis (docs/ISL_DESIGN.md 8.3). Optional and off by default;
// nothing is sent to a server. Many browsers have no Hindi voice: callers check speechAvailable("hi-IN") and,
// if it is false, explain that and offer English (en-IN) instead.

export type SpeechLang = "hi-IN" | "en-IN";

function synth(): SpeechSynthesis | null {
  return typeof window !== "undefined" && "speechSynthesis" in window ? window.speechSynthesis : null;
}

/** Voices for a language: exact match first (hi-IN), then any voice of the same language (hi, hi_IN). */
export function voicesFor(lang: SpeechLang): SpeechSynthesisVoice[] {
  const s = synth();
  if (!s) return [];
  const want = lang.toLowerCase();
  const base = want.split("-")[0];
  const all = s.getVoices();
  const norm = (v: SpeechSynthesisVoice) => v.lang.toLowerCase().replace("_", "-");
  const exact = all.filter((v) => norm(v) === want);
  const same = all.filter((v) => norm(v) !== want && norm(v).split("-")[0] === base);
  return [...exact, ...same];
}

/** True when the browser can speak this language with a matching voice. English falls back to any English voice. */
export function speechAvailable(lang: SpeechLang): boolean {
  if (!synth()) return false;
  if (voicesFor(lang).length) return true;
  if (lang === "en-IN") return synth()!.getVoices().some((v) => v.lang.toLowerCase().startsWith("en"));
  return false;
}

/** Resolves when speaking ends (or fails). Cancels anything already being spoken. */
export function speak(text: string, lang: SpeechLang, opts: { rate?: number } = {}): Promise<void> {
  const s = synth();
  const t = text.trim();
  if (!s || !t || typeof SpeechSynthesisUtterance === "undefined") return Promise.resolve();
  s.cancel();
  return new Promise((resolve) => {
    const u = new SpeechSynthesisUtterance(t);
    u.lang = lang;
    const v = voicesFor(lang)[0];
    if (v) u.voice = v;
    u.rate = opts.rate ?? 0.95;
    u.onend = () => resolve();
    u.onerror = () => resolve();
    s.speak(u);
  });
}

export function stopSpeaking() {
  synth()?.cancel();
}

/** Voices load asynchronously in Chrome: call back when the list changes. Returns an unsubscribe function. */
export function onVoicesChanged(cb: () => void): () => void {
  const s = synth();
  if (!s || typeof s.addEventListener !== "function") return () => {};
  s.addEventListener("voiceschanged", cb);
  return () => s.removeEventListener("voiceschanged", cb);
}
