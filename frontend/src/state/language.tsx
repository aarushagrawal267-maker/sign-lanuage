// Which sign language the app recognises (docs/ISL_DESIGN.md 3.2): a language pack id from GET /api/languages.
// ISL is the default for this app (launch in India); ASL stays available as the second option. The choice is
// remembered in this browser. The static (in-browser) demo only has the ASL model.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { IS_STATIC, api, type LanguageInfo } from "../api/client";

export const ASL_ID = "asl-wlasl100";
export const ISL_ID = "isl";
const STORAGE_KEY = "sis.language"; // docs/ISL_DESIGN.md 11, frontend task

export const STATIC_ASL: LanguageInfo = {
  id: ASL_ID, kind: "experiment-registry", sign_language: "ASL", standard: "WLASL100",
  display_name: { en: "American Sign Language (WLASL100)", hi: "अमेरिकी सांकेतिक भाषा" },
  output_languages: ["en"], capabilities: ["words"], profile: "research", version: null,
  licences: [{ source: "wlasl", licence: "C-UDA-1.0", commercial: false }], status: "ready", message: null,
};

/** Order for the switcher: ISL packs first (commercial-safe before research), then ASL, then anything else. */
export function sortLanguages(langs: LanguageInfo[]): LanguageInfo[] {
  const rank = (l: LanguageInfo) => (l.sign_language === "ISL" ? 0 : l.sign_language === "ASL" ? 1 : 2);
  return [...langs].sort((a, b) => rank(a) - rank(b) || (a.profile === "commercial-safe" ? -1 : 0) - (b.profile === "commercial-safe" ? -1 : 0) || a.id.localeCompare(b.id));
}

/** The pack to use: the stored choice if it still exists, else ISL, else the server default, else the first pack. */
export function pickLanguage(langs: LanguageInfo[], stored: string | null, serverDefault: string | null): string {
  const ids = new Set(langs.map((l) => l.id));
  if (stored && ids.has(stored)) return stored;
  if (ids.has(ISL_ID)) return ISL_ID;
  const isl = langs.find((l) => l.sign_language === "ISL");
  if (isl) return isl.id;
  if (serverDefault && ids.has(serverDefault)) return serverDefault;
  return langs[0]?.id ?? ASL_ID;
}

export const isAslPack = (l: LanguageInfo | null | undefined) => !l || l.kind === "experiment-registry";

type Ctx = {
  languages: LanguageInfo[];
  id: string;
  info: LanguageInfo | null;
  isAsl: boolean;
  loading: boolean;
  error: string | null;
  setId: (id: string) => void;
  reload: () => void;
};

const SignLanguageContext = createContext<Ctx>({
  languages: [STATIC_ASL], id: ASL_ID, info: STATIC_ASL, isAsl: true, loading: false, error: null, setId: () => {}, reload: () => {},
});

function readStored(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function SignLanguageProvider({ children }: { children: ReactNode }) {
  const [languages, setLanguages] = useState<LanguageInfo[]>(IS_STATIC ? [STATIC_ASL] : []);
  const [serverDefault, setServerDefault] = useState<string | null>(null);
  const [stored, setStored] = useState<string | null>(() => readStored());
  const [loading, setLoading] = useState(!IS_STATIC);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    if (IS_STATIC) return;
    setLoading(true);
    api.get<{ default: string; languages: LanguageInfo[] }>("/api/languages")
      .then((r) => {
        setLanguages(sortLanguages(r.languages));
        setServerDefault(r.default);
        setError(null);
      })
      .catch((e) => {
        // Older backend without language packs: behave exactly like the ASL-only app.
        setLanguages([STATIC_ASL]);
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => setLoading(false));
  }, []);
  useEffect(reload, [reload]);

  const setId = useCallback((id: string) => {
    setStored(id);
    try {
      window.localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* storage unavailable */
    }
  }, []);

  const value = useMemo<Ctx>(() => {
    const id = loading && !languages.length ? stored ?? ISL_ID : pickLanguage(languages, stored, serverDefault);
    const info = languages.find((l) => l.id === id) ?? null;
    return { languages, id, info, isAsl: info ? isAslPack(info) : id === ASL_ID, loading, error, setId, reload };
  }, [languages, stored, serverDefault, loading, error, setId, reload]);

  return <SignLanguageContext.Provider value={value}>{children}</SignLanguageContext.Provider>;
}

export function useSignLanguage() {
  return useContext(SignLanguageContext);
}
