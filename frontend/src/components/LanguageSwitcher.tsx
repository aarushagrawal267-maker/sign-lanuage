import type { LanguageInfo } from "../api/client";
import { useT, type UiLang } from "../i18n";
import { useSignLanguage } from "../state/language";
import { Segmented } from "./ui";

/** Short label for a sign-language pack ("ISL", "ASL", "ISL research"). */
export function packShortLabel(l: LanguageInfo, all: LanguageInfo[]): string {
  const sameLang = all.filter((x) => x.sign_language === l.sign_language).length > 1;
  return sameLang && l.profile === "research" ? `${l.sign_language} · research` : l.sign_language;
}

/** Which sign language to recognise: ISL (default in India) or ASL. */
export function SignLanguageSwitcher() {
  const { t } = useT();
  const { languages, id, setId, loading } = useSignLanguage();
  if (loading || languages.length < 2) return null;
  return (
    <div className="flex items-center gap-2" title={t("switch.signLanguageHelp")}>
      <span className="hidden text-[11px] text-fg-3 md:inline" id="sign-lang-label">{t("switch.signLanguage")}</span>
      <Segmented
        ariaLabel={t("switch.signLanguage")}
        value={id}
        onChange={setId}
        options={languages.map((l) => ({ value: l.id, label: packShortLabel(l, languages) }))}
      />
    </div>
  );
}

/** Interface language: English or Hindi menus and labels. */
export function UiLanguageSwitcher() {
  const { t, lang, setLang } = useT();
  return (
    <Segmented<UiLang>
      ariaLabel={t("switch.uiLanguage")}
      value={lang}
      onChange={setLang}
      options={[{ value: "en", label: "EN" }, { value: "hi", label: "हिंदी" }]}
    />
  );
}
