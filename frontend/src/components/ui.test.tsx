import { describe, expect, it } from "vitest";
import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { Field, Progress, Segmented } from "./ui";
import { ConfidenceRing } from "./charts/ConfidenceRing";
import { HBars } from "./charts/Bars";
import { pct, timeAgo } from "../lib/format";

describe("format", () => {
  it("formats percentages and missing values", () => {
    expect(pct(0.1234)).toBe("12.3%");
    expect(pct(null)).toBe("—");
    expect(pct(Number.NaN)).toBe("—");
    expect(timeAgo(null)).toBe("—");
  });
});

describe("charts", () => {
  it("ConfidenceRing exposes an accessible label", () => {
    render(<ConfidenceRing value={0.82} threshold={0.6} />);
    expect(screen.getByRole("img", { name: "confidence 82 percent" })).toBeInTheDocument();
  });
  it("HBars renders every item with its value", () => {
    render(<HBars items={[{ label: "BOOK", value: 0.7 }, { label: "DRINK", value: 0.2 }]} />);
    expect(screen.getByText("BOOK")).toBeInTheDocument();
    expect(screen.getByText("70.0%")).toBeInTheDocument();
  });
});

describe("accessible controls", () => {
  it("Segmented is a radio group with one tab stop and arrow-key selection", () => {
    function Harness() {
      const [v, setV] = useState<"a" | "b" | "c">("a");
      return <Segmented ariaLabel="Mode" value={v} onChange={setV} options={[{ value: "a", label: "A" }, { value: "b", label: "B" }, { value: "c", label: "C" }]} />;
    }
    render(<Harness />);
    const a = screen.getByRole("radio", { name: "A" });
    expect(a).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("radio", { name: "B" })).toHaveAttribute("tabindex", "-1");
    a.focus();
    fireEvent.keyDown(a, { key: "ArrowRight" });
    expect(screen.getByRole("radio", { name: "B" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: "B" })).toHaveFocus();
    fireEvent.keyDown(screen.getByRole("radio", { name: "B" }), { key: "Home" });
    expect(screen.getByRole("radio", { name: "A" })).toHaveAttribute("aria-checked", "true");
  });

  it("Field group labels a radio group without renaming its first option", () => {
    render(<Field label="Architecture" group><Segmented ariaLabel="Architecture" value="gru" onChange={() => {}} options={[{ value: "gru", label: "GRU" }, { value: "lstm", label: "LSTM" }]} /></Field>);
    expect(screen.getByRole("group", { name: "Architecture" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "GRU" })).toBeInTheDocument();
  });

  it("Progress bars have an accessible name", () => {
    render(<Progress value={0.5} label="Training progress" />);
    expect(screen.getByRole("progressbar", { name: "Training progress" })).toHaveAttribute("aria-valuenow", "50");
  });
});
