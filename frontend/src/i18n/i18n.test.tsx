import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { UiLangProvider, reasonFor, translate, useT } from "./index";
import { EN, HI, type StringKey } from "./strings";

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe("interface strings", () => {
  it("every English string has a Hindi translation with the same placeholders", () => {
    const keys = Object.keys(EN) as StringKey[];
    const missing = keys.filter((k) => !HI[k]);
    expect(missing).toEqual([]);
    for (const k of keys) expect(placeholders(HI[k]!), k).toEqual(placeholders(EN[k]));
    expect(Object.keys(HI).filter((k) => !(k in EN))).toEqual([]);
  });

  it("Hindi strings are written in Devanagari (not copied English)", () => {
    const latinOnly = (Object.keys(HI) as StringKey[]).filter((k) => !/[ऀ-ॿ]/.test(HI[k]!));
    // Only strings that are proper nouns, codes or numbers may stay Latin.
    expect(latinOnly).toEqual([]);
  });

  it("translates, interpolates and falls back", () => {
    expect(translate("en", "live.title")).toBe("Live Sign Recognition");
    expect(translate("hi", "live.title")).toBe("लाइव संकेत पहचान");
    expect(translate("en", "my.takesCount", { n: 2, min: 3, max: 5 })).toBe("2 recorded (need 3 to 5)");
    expect(translate("hi", "my.takesCount", { n: 2, min: 3, max: 5 })).toBe("2 रिकॉर्ड हुईं (3 से 5 चाहिए)");
    expect(translate("en", "origin.region", {})).toBe("Regional sign: {region}");
  });

  it("explains engine reasons in plain words and keeps unknown codes readable", () => {
    expect(reasonFor("en", "out_of_vocabulary")).toBe("looks like no word in the dictionary");
    expect(reasonFor("hi", "low_confidence")).toBe("भरोसा कम है");
    expect(reasonFor("en", "something_new")).toBe("something new");
  });

  it("the provider switches the interface language and the document language", () => {
    function Probe() {
      const { t, lang, setLang } = useT();
      return <button onClick={() => setLang(lang === "en" ? "hi" : "en")}>{t("nav.mySigns")}</button>;
    }
    render(<UiLangProvider initial="en"><Probe /></UiLangProvider>);
    expect(document.documentElement.lang).toBe("en");
    fireEvent.click(screen.getByRole("button", { name: "My signs" }));
    expect(screen.getByRole("button", { name: "मेरे संकेत" })).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("hi");
    expect(window.localStorage.getItem("ui-lang-v1")).toBe("hi");
  });
});
