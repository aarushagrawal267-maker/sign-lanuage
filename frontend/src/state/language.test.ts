import { describe, expect, it } from "vitest";
import type { LanguageInfo } from "../api/client";
import { ASL_ID, STATIC_ASL, isAslPack, pickLanguage, sortLanguages } from "./language";

const isl: LanguageInfo = { ...STATIC_ASL, id: "isl", kind: "isl-hybrid", sign_language: "ISL", standard: "ISLRTC", profile: "commercial-safe", capabilities: ["words", "my_signs"] };
const islResearch: LanguageInfo = { ...isl, id: "isl-research", profile: "research" };

describe("sign language selection", () => {
  it("lists ISL first (commercial-safe before research), then ASL", () => {
    expect(sortLanguages([STATIC_ASL, islResearch, isl]).map((l) => l.id)).toEqual(["isl", "isl-research", ASL_ID]);
  });
  it("defaults to ISL even when the server default is ASL, but keeps the user's choice", () => {
    expect(pickLanguage([STATIC_ASL, isl], null, ASL_ID)).toBe("isl");
    expect(pickLanguage([STATIC_ASL, isl], ASL_ID, ASL_ID)).toBe(ASL_ID);
    expect(pickLanguage([STATIC_ASL, isl], "gone", ASL_ID)).toBe("isl");
    expect(pickLanguage([STATIC_ASL], null, ASL_ID)).toBe(ASL_ID);
    expect(pickLanguage([STATIC_ASL, islResearch], null, ASL_ID)).toBe("isl-research");
    expect(pickLanguage([], null, null)).toBe(ASL_ID);
  });
  it("treats the experiment-registry pack as the classic ASL engine", () => {
    expect(isAslPack(STATIC_ASL)).toBe(true);
    expect(isAslPack(isl)).toBe(false);
  });
});
