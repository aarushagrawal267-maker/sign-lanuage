// Simple interface translation (English / Hindi). Keys live in strings.ts; a missing Hindi string falls back to
// English so the app never shows a raw key. The interface language is separate from the sign language and from
// the output language: recognised words are always shown in English AND Hindi.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { EN, HI, type StringKey } from "./strings";

export type UiLang = "en" | "hi";
const STORAGE_KEY = "ui-lang-v1";

export type Vars = Record<string, string | number>;

/** Pure lookup used by the hook and by tests. `{name}` placeholders are replaced from `vars`. */
export function translate(lang: UiLang, key: StringKey, vars?: Vars): string {
  const raw = (lang === "hi" ? HI[key] : undefined) ?? EN[key] ?? key;
  if (!vars) return raw;
  return raw.replace(/\{(\w+)\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m));
}

function initialLang(): UiLang {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    if (v === "en" || v === "hi") return v;
  } catch {
    /* storage unavailable */
  }
  return typeof navigator !== "undefined" && navigator.language?.toLowerCase().startsWith("hi") ? "hi" : "en";
}

/** Plain-language reason why a sign was not shown as a word (engine reason codes). Unknown codes are shown as is. */
export function reasonFor(lang: UiLang, code: string): string {
  const key = `reason.${code}`;
  return key in EN ? translate(lang, key as StringKey) : code.replace(/_/g, " ");
}

type Ctx = { lang: UiLang; setLang: (l: UiLang) => void; t: (key: StringKey, vars?: Vars) => string; reasonText: (code: string) => string };
const UiLangContext = createContext<Ctx>({ lang: "en", setLang: () => {}, t: (k, v) => translate("en", k, v), reasonText: (c) => reasonFor("en", c) });

export function UiLangProvider({ children, initial }: { children: ReactNode; initial?: UiLang }) {
  const [lang, setLangState] = useState<UiLang>(() => initial ?? initialLang());
  const setLang = useCallback((l: UiLang) => {
    setLangState(l);
    try {
      window.localStorage.setItem(STORAGE_KEY, l);
    } catch {
      /* storage unavailable */
    }
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang === "hi" ? "hi" : "en";
  }, [lang]);
  const value = useMemo<Ctx>(() => ({ lang, setLang, t: (k, v) => translate(lang, k, v), reasonText: (c) => reasonFor(lang, c) }), [lang, setLang]);
  return <UiLangContext.Provider value={value}>{children}</UiLangContext.Provider>;
}

export function useT() {
  return useContext(UiLangContext);
}

export type { StringKey };
