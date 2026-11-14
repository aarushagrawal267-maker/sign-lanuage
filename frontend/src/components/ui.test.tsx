import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
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
