import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { Candidate } from "../api/client";
import { UiLangProvider } from "../i18n";
import type { SegmentResult, TranscriptItem } from "../recognition/live";
import { BilingualWord } from "./BilingualWord";
import { CandidateList } from "./CandidateList";
import { SentenceBar } from "./SentenceBar";
import { SignResult, type SignResultProps } from "./SignResult";

const water: Candidate = { concept_id: "isl.water", class_index: 1, gloss: "water", prob: 0.46, text: { en: "water", hi: "पानी" }, hi_review: "curated", source: "classifier+retrieval" };
const drink: Candidate = { concept_id: "isl.drink", class_index: 2, gloss: "drink", prob: 0.2, text: { en: "drink", hi: "पीना" }, hi_review: "machine", source: "retrieval" };
const myIndore: Candidate = { concept_id: "mysign:abc", class_index: -1, gloss: "Indore", prob: 0.1, text: { en: "Indore", hi: "इंदौर" }, hi_review: "user", source: "my_signs", region: "Indore, MP", origin: "My signs" };

const wrap = (ui: React.ReactNode, lang: "en" | "hi" = "en") => render(<MemoryRouter><UiLangProvider initial={lang}>{ui}</UiLangProvider></MemoryRouter>);

describe("BilingualWord", () => {
  it("shows English and Hindi with the Hindi review marker", () => {
    wrap(<BilingualWord en="water" hi="पानी" review="machine" size="xl" />);
    expect(screen.getByText("water")).toHaveAttribute("lang", "en");
    expect(screen.getByText("पानी")).toHaveAttribute("lang", "hi");
    expect(screen.getByText("machine")).toHaveAttribute("title", "Machine translation, not yet checked");
  });
  it("never invents Hindi: says it is not available", () => {
    wrap(<BilingualWord en="zebra crossing" hi={null} />);
    expect(screen.getByText("Hindi not available yet")).toBeInTheDocument();
    expect(screen.getByText("हिंदी उपलब्ध नहीं")).toBeInTheDocument();
  });
});

describe("CandidateList (did you mean)", () => {
  it("offers at most five bilingual choices and reports the tapped one", () => {
    const onChoose = vi.fn();
    const many = [water, drink, myIndore, { ...drink, concept_id: "a" }, { ...drink, concept_id: "b" }, { ...drink, concept_id: "c" }];
    wrap(<CandidateList candidates={many} onChoose={onChoose} onDismiss={() => {}} />);
    expect(screen.getAllByRole("button", { name: /^Choose/ })).toHaveLength(5);
    expect(screen.getByText("My signs")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Choose water पानी, 46 percent" }));
    expect(onChoose).toHaveBeenCalledWith(water);
  });
  it("is translated in the Hindi interface", () => {
    wrap(<CandidateList candidates={[water]} onChoose={() => {}} onDismiss={() => {}} />, "hi");
    expect(screen.getByText("क्या आपका मतलब था…?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "इनमें से कोई नहीं – फिर से करें" })).toBeInTheDocument();
  });
});

function props(over: Partial<SignResultProps>): SignResultProps {
  return {
    lastResult: null, resolved: null, inProgress: false, provisionalTop: null, recognizedNow: false, pack: true, threshold: null,
    spellOn: false, spell: null, spellAvailable: false, spellExperimental: true, acceptedItem: null,
    onChoose: vi.fn(), onDismiss: vi.fn(), onReject: vi.fn(), onSpell: vi.fn(), ...over,
  };
}

describe("SignResult", () => {
  const seg = (decision: "accept" | "suggest" | "abstain", top: Candidate[] = [water, drink]): SegmentResult =>
    ({ segment_id: 4, accepted: decision === "accept", decision, reason: decision === "accept" ? "ok" : "low_confidence", frames: 30, top, confidence: top[0]?.prob });

  it("accepted: the word in English and Hindi, and a way to say it was wrong", () => {
    const item: TranscriptItem = { id: 1, gloss: "water", score: 0.9, at: 0, segment_id: 4, text: { en: "water", hi: "पानी" } };
    const p = props({ lastResult: seg("accept"), recognizedNow: true, acceptedItem: item });
    wrap(<SignResult {...p} />);
    expect(screen.getByTestId("accepted")).toHaveTextContent("water");
    expect(screen.getByTestId("accepted")).toHaveTextContent("पानी");
    fireEvent.click(screen.getByRole("button", { name: "Wrong word? Remove it" }));
    expect(p.onReject).toHaveBeenCalledWith(item);
  });

  it("suggest: did-you-mean chips; choosing sends the segment and the candidate", () => {
    const s = seg("suggest");
    const p = props({ lastResult: s });
    wrap(<SignResult {...p} />);
    expect(screen.queryByTestId("accepted")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^Choose drink/ }));
    expect(p.onChoose).toHaveBeenCalledWith(s, drink);
  });

  it("after a choice the chips go away", () => {
    const s = seg("suggest");
    wrap(<SignResult {...props({ lastResult: s, resolved: { segment: s, how: "chosen" } })} />);
    expect(screen.queryByTestId("did-you-mean")).toBeNull();
    expect(screen.getByTestId("resolved")).toHaveTextContent("Added to your sentence");
  });

  it("abstain: 'not sure', no word shown, spelling offered only when available", () => {
    const p = props({ lastResult: seg("abstain"), spellHint: <a href="#x">calibrate</a> });
    const { unmount } = wrap(<SignResult {...p} />);
    expect(screen.getByTestId("not-sure")).toHaveTextContent("Not sure – try again or spell it");
    expect(screen.queryByText("water")).toBeNull();
    expect(screen.queryByRole("button", { name: "Spell it" })).toBeNull();
    expect(screen.getByText("calibrate")).toBeInTheDocument();
    unmount();
    const p2 = props({ lastResult: seg("abstain"), spellAvailable: true });
    wrap(<SignResult {...p2} />);
    fireEvent.click(screen.getByRole("button", { name: "Spell it" }));
    expect(p2.onSpell).toHaveBeenCalled();
  });

  it("spelling: shows the live letters", () => {
    wrap(<SignResult {...props({ spellOn: true, inProgress: true, spell: { letters: [{ letter: "I", prob: 0.9 }], buffer: "INDO", suggestions: [{ word: "indore", score: 0.7 }] } })} />);
    expect(screen.getByTestId("spell-panel")).toHaveTextContent("INDO");
    expect(screen.getByTestId("spell-panel")).toHaveTextContent("INDORE");
    expect(screen.getByText("experimental")).toBeInTheDocument();
  });
});

describe("SentenceBar", () => {
  afterEach(() => vi.unstubAllGlobals());
  const items: TranscriptItem[] = [
    { id: 1, gloss: "water", score: 0.9, at: 0, concept_id: "isl.water", text: { en: "water", hi: "पानी" }, segment_id: 3 },
    { id: 2, gloss: "I", score: 0.8, at: 0, concept_id: "isl.i", text: { en: "I", hi: "मैं" }, segment_id: null, chosen: true },
  ];

  it("asks the backend for an approximate English + Hindi sentence and labels it approximate", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      language: "isl", en: { text: "I want water.", approximate: true, rules: ["r1"], missing_hi: [] },
      hi: { text: "मैं पानी चाहता हूँ।", approximate: true, rules: [], missing_hi: [] },
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    const onRemove = vi.fn();
    wrap(<SentenceBar items={items} language="isl" onClear={() => {}} onRemove={onRemove} onWrong={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("sentence-en")).toHaveTextContent("I want water."));
    expect(screen.getByTestId("sentence-hi")).toHaveTextContent("मैं पानी चाहता हूँ।");
    expect(screen.getByText("Approximate sentence (rule-based, may be wrong)")).toBeInTheDocument();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/languages/isl/sentence");
    expect(JSON.parse(init.body)).toEqual({ words: [{ concept_id: "isl.water", en: "water", hi: "पानी" }, { concept_id: "isl.i", en: "I", hi: "मैं" }] });
    fireEvent.click(screen.getByRole("button", { name: "Remove I" }));
    expect(onRemove).toHaveBeenCalledWith(2);
  });

  it("falls back to the words in order when the sentence builder is unreachable, and explains missing voices", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    wrap(<SentenceBar items={items} language="isl" onClear={() => {}} onRemove={() => {}} onWrong={() => {}} />);
    await waitFor(() => expect(screen.getByText(/Sentence builder not reachable/)).toBeInTheDocument());
    expect(screen.getByTestId("sentence-en")).toHaveTextContent("water I");
    expect(screen.getByTestId("sentence-hi")).toHaveTextContent("पानी मैं");
    expect(screen.getByRole("button", { name: "Speak" })).toBeDisabled();
    expect(screen.getByText("This browser has no speech voices.")).toBeInTheDocument();
  });
});
