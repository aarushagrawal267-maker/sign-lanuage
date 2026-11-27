// The ISL app default (docs/ISL_DESIGN.md 15.29): a confirm tap before any word is added, the certified precision
// next to it, "did you mean X?" for uncertified policies and dictionary-only words, English + Hindi, keyboard and axe.
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import axe from "axe-core";
import type { Candidate } from "../api/client";
import { UiLangProvider } from "../i18n";
import { segmentFromCommit, type Certification, type Commit, type SegmentResult } from "../recognition/live";
import type { PendingWord } from "../recognition/useLiveRecognition";
import { ConfirmWord } from "./ConfirmWord";
import { ResultAnnouncer, SignResult, type SignResultProps } from "./SignResult";

const water: Candidate = { concept_id: "isl.water", class_index: 1, gloss: "water", prob: 0.97, text: { en: "water", hi: "पानी" }, hi_review: "curated", source: "classifier" };
const drink: Candidate = { concept_id: "isl.drink", class_index: 2, gloss: "drink", prob: 0.02, text: { en: "drink", hi: "पीना" }, hi_review: "machine", source: "retrieval" };
const certified: Certification = { rule: "certified", certified_precision: 0.953, worst_source: "include", floor: 0.95, target: 0.98, confirm_tap: true };
const confirmOnly: Certification = { rule: "confirm_only", certified_precision: null, worst_source: "include", floor: 0.95, target: 0.98, confirm_tap: true };
const commit = (over: Partial<Commit> = {}): Commit => ({
  gloss: "water", class_index: 1, score: 0.97, t: 10, concept_id: "isl.water", text: { en: "water", hi: "पानी" }, hi_review: "curated",
  source: "classifier", segment_id: 4, chosen_by_user: false, certification: certified, needs_confirm: true, ...over,
});
const pendingFrom = (c: Commit, continuous = false): PendingWord => {
  const segment: SegmentResult = continuous ? segmentFromCommit(c)
    : { segment_id: 4, accepted: true, decision: "accept", reason: "ok", frames: 30, top: [water, drink], confidence: 0.97, certification: c.certification, needs_confirm: true };
  return { commit: c, segment, continuous };
};
const note = (c: Certification | undefined) => (c?.rule === "certified" ? `Strict: right about ${Math.floor(c.certified_precision! * 100)} of 100 times when it answers` : c ? "Strict: not tested well enough yet" : null);

const wrap = (ui: React.ReactNode, lang: "en" | "hi" = "en") => render(<MemoryRouter><UiLangProvider initial={lang}>{ui}</UiLangProvider></MemoryRouter>);
function props(over: Partial<SignResultProps>): SignResultProps {
  return {
    lastResult: null, resolved: null, inProgress: false, provisionalTop: null, recognizedNow: true, pack: true, threshold: null,
    spellOn: false, spell: null, spellAvailable: false, spellExperimental: true, acceptedItem: null, precisionNote: note,
    onChoose: vi.fn(), onDismiss: vi.fn(), onReject: vi.fn(), onSpell: vi.fn(), onConfirmPending: vi.fn(), onRejectPending: vi.fn(), ...over,
  };
}
async function axeViolations(container: HTMLElement) {
  const r = await axe.run(container, {
    runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] },
    rules: { "color-contrast": { enabled: false } },
  });
  return r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
}

describe("ConfirmWord", () => {
  it("recognised: the word, the certified precision, Add and Wrong word buttons", () => {
    const onConfirm = vi.fn(), onReject = vi.fn();
    wrap(<ConfirmWord word={water} kind="recognised" note="Strict: right about 95 of 100 times when it answers" onConfirm={onConfirm} onReject={onReject} />);
    const card = screen.getByTestId("confirm-word");
    expect(card).toHaveAttribute("data-state", "confirm-recognised");
    expect(card).toHaveTextContent("water");
    expect(card).toHaveTextContent("पानी");
    expect(screen.getByTestId("confirm-precision")).toHaveTextContent("right about 95 of 100 times");
    expect(card).toHaveTextContent("Nothing is added, spoken or sent until you tap.");
    fireEvent.click(screen.getByRole("button", { name: "Add water" }));
    expect(onConfirm).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Wrong word" }));
    expect(onReject).toHaveBeenCalledOnce();
  });
  it("did you mean: yes / no and the other guesses, all real buttons (keyboard)", () => {
    const onChooseOther = vi.fn();
    wrap(<ConfirmWord word={water} kind="did_you_mean" others={[water, drink]} onConfirm={() => {}} onReject={() => {}} onChooseOther={onChooseOther} />);
    expect(screen.getByRole("region", { name: "Did you mean this word?" })).toHaveAttribute("data-state", "confirm-did-you-mean");
    expect(screen.getByRole("button", { name: "Yes, add water" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "No, sign again" })).toBeInTheDocument();
    // the top word is not offered twice
    expect(screen.getAllByRole("button", { name: /^Choose/ })).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Choose drink पीना" }));
    expect(onChooseOther).toHaveBeenCalledWith(drink);
  });
  it("is translated in the Hindi interface", () => {
    wrap(<ConfirmWord word={water} kind="did_you_mean" onConfirm={() => {}} onReject={() => {}} />, "hi");
    expect(screen.getByText("क्या आपका मतलब यह शब्द था?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "हाँ, water जोड़ें" })).toBeInTheDocument();
  });
  it("passes axe-core (structure rules) in both looks and both languages", async () => {
    for (const lang of ["en", "hi"] as const) {
      for (const kind of ["recognised", "did_you_mean"] as const) {
        const { container, unmount } = wrap(<ConfirmWord word={water} kind={kind} note="x" others={[water, drink]} onConfirm={() => {}} onReject={() => {}} onChooseOther={() => {}} />, lang);
        expect(await axeViolations(container), `${kind} ${lang}`).toEqual([]);
        unmount();
      }
    }
  });
});

describe("SignResult with a word waiting for a tap", () => {
  it("a certified word is 'recognised – add it?', never shown as already written", () => {
    const p = props({ pending: pendingFrom(commit()), lastResult: pendingFrom(commit()).segment });
    wrap(<SignResult {...p} />);
    expect(screen.queryByTestId("accepted")).toBeNull();
    expect(screen.getByTestId("confirm-word")).toHaveAttribute("data-state", "confirm-recognised");
    expect(screen.getByTestId("confirm-precision")).toHaveTextContent("right about 95 of 100");
    fireEvent.click(screen.getByRole("button", { name: "Add water" }));
    expect(p.onConfirmPending).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Wrong word" }));
    expect(p.onRejectPending).toHaveBeenCalledOnce();
  });
  it("an uncertified (confirm-only) policy shows 'did you mean X?'", () => {
    const pw = pendingFrom(commit({ certification: confirmOnly }));
    wrap(<SignResult {...props({ pending: pw, lastResult: pw.segment })} />);
    expect(screen.getByTestId("confirm-word")).toHaveAttribute("data-state", "confirm-did-you-mean");
    expect(screen.getByTestId("confirm-precision")).toHaveTextContent("not tested well enough");
  });
  it("a dictionary-only word is always 'did you mean', without the policy's precision", () => {
    const pw = pendingFrom(commit({ open_vocabulary: true }));
    wrap(<SignResult {...props({ pending: pw, lastResult: pw.segment })} />);
    expect(screen.getByTestId("confirm-word")).toHaveAttribute("data-state", "confirm-did-you-mean");
    expect(screen.queryByTestId("confirm-precision")).toBeNull();
    expect(screen.getByText("Dictionary-only word: the app always asks.")).toBeInTheDocument();
  });
  it("continuous mode: 'did you mean' unless sliding strict is certified; the card stays while signing goes on", () => {
    const pw = pendingFrom(commit({ segment_id: null }), true);
    const { unmount } = wrap(<SignResult {...props({ pending: pw, continuous: true, slidingStrict: false, inProgress: true, provisionalTop: drink })} />);
    expect(screen.getByTestId("confirm-word")).toHaveAttribute("data-state", "confirm-did-you-mean");
    expect(screen.getByText("drink")).toBeInTheDocument(); // the sign in progress, under the card
    unmount();
    wrap(<SignResult {...props({ pending: pw, continuous: true, slidingStrict: true, inProgress: true })} />);
    expect(screen.getByTestId("confirm-word")).toHaveAttribute("data-state", "confirm-recognised");
  });
  it("one of the user's own signs never shows the policy's certified precision, but its own line", () => {
    const pw = pendingFrom(commit({ source: "my_signs" }));
    wrap(<SignResult {...props({ pending: pw, lastResult: pw.segment })} />);
    const line = screen.getByTestId("confirm-precision");
    expect(line).not.toHaveTextContent("95 of 100");
    expect(line).toHaveTextContent("Your own sign: the tested figure does not cover it");
  });
  it("a confirm-only suggestion of a user sign gets the same line", () => {
    const mine: Candidate = { ...water, source: "my_signs" };
    const sugg: SegmentResult = { segment_id: 9, accepted: false, decision: "suggest", reason: "confirm_only", frames: 30, top: [mine, drink], certification: certified, needs_confirm: true };
    wrap(<SignResult {...props({ lastResult: sugg })} />);
    expect(screen.getByTestId("confirm-precision")).toHaveTextContent("Your own sign");
    expect(screen.getByTestId("confirm-precision")).not.toHaveTextContent("95 of 100");
  });
  it("when the next sign replaces the word, new buttons are mounted and a tap meant for the old word is ignored", () => {
    let now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    const first = pendingFrom(commit());
    const p = props({ pending: first, lastResult: first.segment });
    const { rerender } = wrap(<SignResult {...p} />);
    const oldAdd = screen.getByRole("button", { name: "Add water" });
    oldAdd.focus();
    expect(document.activeElement).toBe(oldAdd);
    const pen: Commit = commit({ concept_id: "isl.pen", gloss: "pen", text: { en: "pen", hi: "कलम" }, segment_id: 5, t: 20 });
    const second = pendingFrom(pen);
    rerender(<MemoryRouter><UiLangProvider initial="en"><SignResult {...p} pending={second} lastResult={second.segment} /></UiLangProvider></MemoryRouter>);
    const newAdd = screen.getByRole("button", { name: "Add pen" });
    expect(newAdd).not.toBe(oldAdd); // remounted: Enter on the old button cannot reach the new word
    expect(oldAdd.isConnected).toBe(false);
    // focus was inside the old card: it moves to the new card (not to the Add button), so the word is read first
    expect(document.activeElement).toBe(screen.getByTestId("confirm-word"));
    fireEvent.click(newAdd); // a tap right after the change
    expect(p.onConfirmPending).not.toHaveBeenCalled();
    now += 400;
    fireEvent.click(newAdd);
    expect(p.onConfirmPending).toHaveBeenCalledOnce();
    vi.restoreAllMocks();
  });
  it("the first word of a series is not delayed", () => {
    const pw = pendingFrom(commit());
    const p = props({ pending: pw, lastResult: pw.segment });
    wrap(<SignResult {...p} />);
    fireEvent.click(screen.getByRole("button", { name: "Add water" }));
    expect(p.onConfirmPending).toHaveBeenCalledOnce();
  });
  it("while spelling, the waiting word stays visible above the spelling panel", () => {
    const pw = pendingFrom(commit());
    wrap(<SignResult {...props({ pending: pw, spellOn: true, spellAvailable: true, inProgress: true })} />);
    expect(screen.getByTestId("confirm-word")).toBeInTheDocument();
    expect(screen.getByTestId("spell-panel")).toBeInTheDocument();
  });
  it("choosing another guess sends that word for the same sign", () => {
    const pw = pendingFrom(commit({ certification: confirmOnly }));
    const p = props({ pending: pw, lastResult: pw.segment });
    wrap(<SignResult {...p} />);
    fireEvent.click(screen.getByRole("button", { name: "Choose drink पीना" }));
    expect(p.onChoose).toHaveBeenCalledWith(pw.segment, drink);
  });
});

describe("SignResult: a rule-based suggestion (confirm-only policy)", () => {
  const sugg: SegmentResult = { segment_id: 9, accepted: false, decision: "suggest", reason: "confirm_only", frames: 30, top: [water, drink], certification: confirmOnly, needs_confirm: true };
  it("shows the top word as 'did you mean X?'; yes chooses it, no dismisses", () => {
    const p = props({ lastResult: sugg });
    wrap(<SignResult {...p} />);
    expect(screen.queryByTestId("did-you-mean")).toBeNull(); // not the five-chip list of a doubtful sign
    fireEvent.click(screen.getByRole("button", { name: "Yes, add water" }));
    expect(p.onChoose).toHaveBeenCalledWith(sugg, water);
    fireEvent.click(screen.getByRole("button", { name: "No, sign again" }));
    expect(p.onDismiss).toHaveBeenCalledWith(sugg);
  });
  it("a doubtful sign still gets the usual chips", () => {
    wrap(<SignResult {...props({ lastResult: { ...sugg, reason: "ambiguous" } })} />);
    expect(screen.getByTestId("did-you-mean")).toBeInTheDocument();
    expect(screen.queryByTestId("confirm-word")).toBeNull();
  });
});

describe("announcements", () => {
  it("a word waiting for a tap is announced as 'please confirm', in English and Hindi", () => {
    wrap(<ResultAnnouncer lastResult={null} resolved={null} inProgress pending={pendingFrom(commit())} />);
    const log = screen.getByTestId("result-announcer");
    expect(log).toHaveTextContent("Recognised, please confirm: water, पानी");
    expect(screen.getByText("पानी", { selector: "[lang=hi]" })).toBeInTheDocument();
  });
  it("an uncertified word is announced as 'did you mean'", () => {
    wrap(<ResultAnnouncer lastResult={null} resolved={null} inProgress={false} pending={pendingFrom(commit({ certification: confirmOnly }))} />);
    expect(screen.getByTestId("result-announcer")).toHaveTextContent("Did you mean: water, पानी");
  });
});
