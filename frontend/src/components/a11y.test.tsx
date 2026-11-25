// Accessibility regressions found in the ISL audit (docs/ISL_DESIGN.md section 14): clear result states, announcements
// for screen-reader / braille users, focus that never gets lost, language marks for Hindi and English, and axe-core
// (structure rules; colour contrast is checked in the browser by e2e/a11y.spec.ts) on every result state.
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import axe from "axe-core";
import type { Candidate } from "../api/client";
import { UiLangProvider } from "../i18n";
import type { SegmentResult, TranscriptItem } from "../recognition/live";
import { CandidateList } from "./CandidateList";
import { pageIsTranslated } from "./Layout";
import { SentenceBar } from "./SentenceBar";
import { ResultAnnouncer, SignResult, type SignResultProps } from "./SignResult";
import { SkeletonPreview } from "./SkeletonPreview";
import { Segmented } from "./ui";

const water: Candidate = { concept_id: "isl.water", class_index: 1, gloss: "water", prob: 0.91, text: { en: "water", hi: "पानी" }, hi_review: "curated", source: "classifier" };
const drink: Candidate = { concept_id: "isl.drink", class_index: 2, gloss: "drink", prob: 0.2, text: { en: "drink", hi: "पीना" }, hi_review: "machine", source: "retrieval" };
const seg = (decision: "accept" | "suggest" | "abstain", id = 4): SegmentResult =>
  ({ segment_id: id, accepted: decision === "accept", decision, reason: decision === "accept" ? "ok" : "low_confidence", frames: 30, top: decision === "abstain" ? [] : [water, drink], confidence: 0.9 });
const item: TranscriptItem = { id: 1, gloss: "water", score: 0.9, at: 0, segment_id: 4, text: { en: "water", hi: "पानी" } };

const wrap = (ui: React.ReactNode, lang: "en" | "hi" = "en") => render(<MemoryRouter><UiLangProvider initial={lang}>{ui}</UiLangProvider></MemoryRouter>);
function props(over: Partial<SignResultProps>): SignResultProps {
  return {
    lastResult: null, resolved: null, inProgress: false, provisionalTop: null, recognizedNow: true, pack: true, threshold: null,
    spellOn: false, spell: null, spellAvailable: false, spellExperimental: true, acceptedItem: null,
    onChoose: vi.fn(), onDismiss: vi.fn(), onReject: vi.fn(), onSpell: vi.fn(), ...over,
  };
}

async function axeViolations(container: HTMLElement) {
  const r = await axe.run(container, {
    runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] },
    rules: { "color-contrast": { enabled: false } }, // jsdom has no layout; checked in the browser instead
  });
  return r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
}

describe("result states are unmistakable", () => {
  it("recognised, did-you-mean and not-sure each have their own state, icon, border and heading", () => {
    const states: [SegmentResult, string, string][] = [[seg("accept"), "accepted", "border-teal/60"], [seg("suggest"), "suggest", "border-warn/60"], [seg("abstain"), "not-sure", "border-dashed"]];
    const seen = new Set<string>();
    for (const [s, state, border] of states) {
      const { container, unmount } = wrap(<SignResult {...props({ lastResult: s, acceptedItem: s.decision === "accept" ? item : null })} />);
      const box = container.querySelector(`[data-state="${state}"]`) as HTMLElement;
      expect(box, state).not.toBeNull();
      expect(box.className).toContain(border);
      expect(box.querySelector("svg"), `${state} has an icon`).not.toBeNull();
      seen.add(box.className);
      unmount();
    }
    expect(seen.size).toBe(3);
  });

  it("only the recognised state shows a word as the answer; the other two never do", () => {
    wrap(<SignResult {...props({ lastResult: seg("suggest") })} />);
    expect(screen.queryByTestId("accepted")).toBeNull();
    expect(screen.getByRole("heading", { name: "Did you mean…?" })).toBeInTheDocument();
  });

  it("each state passes axe-core (structure rules) in English and Hindi", async () => {
    for (const lang of ["en", "hi"] as const) {
      for (const d of ["accept", "suggest", "abstain"] as const) {
        const { container, unmount } = wrap(<SignResult {...props({ lastResult: seg(d), acceptedItem: d === "accept" ? item : null })} />, lang);
        expect(await axeViolations(container), `${lang} ${d}`).toEqual([]);
        unmount();
      }
    }
  });

  it("a word the user marked as wrong is no longer shown as recognised", () => {
    const s = seg("accept");
    wrap(<SignResult {...props({ lastResult: s, resolved: { segment: s, how: "rejected" } })} />);
    expect(screen.queryByTestId("accepted")).toBeNull();
    expect(screen.getByTestId("resolved")).toHaveTextContent("Removed from your sentence. Please sign the word again.");
  });
});

describe("focus is never lost", () => {
  it("choosing a 'did you mean' word moves focus to the confirmation", () => {
    const s = seg("suggest");
    const p = props({ lastResult: s });
    const { rerender } = wrap(<SignResult {...p} />);
    const btn = screen.getByRole("button", { name: /^Choose drink/ });
    btn.focus();
    fireEvent.click(btn);
    rerender(<MemoryRouter><UiLangProvider initial="en"><SignResult {...p} resolved={{ segment: s, how: "chosen" }} /></UiLangProvider></MemoryRouter>);
    expect(document.activeElement).toBe(screen.getByTestId("resolved"));
  });

  it("'Wrong word? Remove it' moves focus to the 'sign again' message", () => {
    const s = seg("accept");
    const p = props({ lastResult: s, acceptedItem: item });
    const { rerender } = wrap(<SignResult {...p} />);
    const btn = screen.getByRole("button", { name: "Wrong word? Remove it" });
    btn.focus();
    fireEvent.click(btn);
    expect(p.onReject).toHaveBeenCalledWith(item);
    rerender(<MemoryRouter><UiLangProvider initial="en"><SignResult {...p} acceptedItem={null} resolved={{ segment: s, how: "rejected" }} /></UiLangProvider></MemoryRouter>);
    expect(document.activeElement).toBe(screen.getByTestId("resolved"));
  });

  it("removing a word from the sentence keeps focus in the word list", () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    const onRemove = vi.fn();
    wrap(<SentenceBar items={[{ ...item, segment_id: null }]} language="isl" onClear={() => {}} onRemove={onRemove} onWrong={() => {}} />);
    const btn = screen.getByRole("button", { name: "Remove water" });
    btn.focus();
    fireEvent.click(btn);
    expect(onRemove).toHaveBeenCalledWith(1);
    expect(document.activeElement).toBe(screen.getByRole("group", { name: "Signed words" }));
    vi.unstubAllGlobals();
  });
});

describe("announcements for screen-reader and braille users", () => {
  it("announces a recognised word in English and Hindi, with the Hindi marked lang=hi", () => {
    wrap(<ResultAnnouncer lastResult={seg("accept")} resolved={null} inProgress={false} />);
    const log = screen.getByTestId("result-announcer");
    expect(log).toHaveAttribute("aria-live", "polite");
    expect(log).toHaveTextContent("Recognised: water, पानी");
    expect(screen.getByText("पानी")).toHaveAttribute("lang", "hi");
  });

  it("says 'did you mean' and 'not sure' without naming a guess as the answer, and stays quiet while signing", () => {
    const { rerender } = wrap(<ResultAnnouncer lastResult={seg("suggest")} resolved={null} inProgress={false} />);
    const log = screen.getByTestId("result-announcer");
    expect(log).toHaveTextContent("Not sure. 2 possible words are shown: choose the right one.");
    expect(log).not.toHaveTextContent("water");
    const r = (x: React.ReactNode) => rerender(<MemoryRouter><UiLangProvider initial="en">{x}</UiLangProvider></MemoryRouter>);
    r(<ResultAnnouncer lastResult={seg("abstain", 5)} resolved={null} inProgress={false} />);
    expect(log).toHaveTextContent("Not sure – try again or spell it");
    r(<ResultAnnouncer lastResult={seg("abstain", 5)} resolved={null} inProgress />);
    expect(log).toBeEmptyDOMElement();
  });

  it("announces the same word again when it is signed twice (new node per sign)", () => {
    const { rerender } = wrap(<ResultAnnouncer lastResult={seg("accept", 1)} resolved={null} inProgress={false} />);
    const first = screen.getByTestId("result-announcer").firstElementChild;
    rerender(<MemoryRouter><UiLangProvider initial="en"><ResultAnnouncer lastResult={seg("accept", 2)} resolved={null} inProgress={false} /></UiLangProvider></MemoryRouter>);
    expect(screen.getByTestId("result-announcer").firstElementChild).not.toBe(first);
  });

  it("is in Hindi in the Hindi interface", () => {
    wrap(<ResultAnnouncer lastResult={seg("accept")} resolved={null} inProgress={false} />, "hi");
    expect(screen.getByTestId("result-announcer")).toHaveTextContent("पहचाना गया: water, पानी");
    expect(screen.getByText("water")).toHaveAttribute("lang", "en");
  });
});

describe("names and languages", () => {
  it("'did you mean' buttons are named by their visible text (WCAG 2.5.3), Hindi in a lang=hi span", () => {
    wrap(<CandidateList candidates={[water]} onChoose={() => {}} />);
    const btn = screen.getByRole("button", { name: "Choose water पानी 91% sure" });
    expect(btn).not.toHaveAttribute("aria-label");
    expect(btn.querySelector('[lang="hi"]')).toHaveTextContent("पानी");
  });

  it("Segmented options can carry their own language", () => {
    render(<Segmented ariaLabel="Interface language" value="en" onChange={() => {}} options={[{ value: "en", label: "EN", lang: "en" }, { value: "hi", label: "हिंदी", lang: "hi" }]} />);
    expect(screen.getByRole("radio", { name: "हिंदी" })).toHaveAttribute("lang", "hi");
  });

  it("only translated pages are left in the interface language", () => {
    expect(pageIsTranslated("/live", false)).toBe(true);
    expect(pageIsTranslated("/my-signs", false)).toBe(true);
    expect(pageIsTranslated("/vocabulary", false)).toBe(true);
    expect(pageIsTranslated("/vocabulary", true)).toBe(false);
    expect(pageIsTranslated("/evaluation", false)).toBe(true); // ISL report page
    expect(pageIsTranslated("/evaluation", true)).toBe(false); // ASL evaluation is English only
    for (const p of ["/", "/analyze", "/studio", "/dataset", "/lab", "/experiments", "/research"]) expect(pageIsTranslated(p, false), p).toBe(false);
  });
});

describe("moving stick figure (WCAG 2.2.2)", () => {
  afterEach(() => vi.restoreAllMocks());
  const frames = [{}, {}, {}] as never[];
  it("has a pause button", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    wrap(<SkeletonPreview frames={frames} width={4} height={3} label="Sign for water" />);
    expect(screen.getByRole("img", { name: "Sign for water" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Pause the moving figure" }));
    expect(screen.getByRole("button", { name: "Play the moving figure" })).toBeInTheDocument();
  });
  it("starts paused when the system asks for reduced motion", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener() {}, removeEventListener() {} }));
    act(() => { wrap(<SkeletonPreview frames={frames} width={4} height={3} label="Sign for water" />); });
    expect(screen.getByRole("button", { name: "Play the moving figure" })).toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});
