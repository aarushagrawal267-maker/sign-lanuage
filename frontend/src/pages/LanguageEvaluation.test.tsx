// The ISL evaluation page renders the phase-2 report (backend/app/isl/evaluation/runners.py, schema 2) completely:
// every protocol E1a/E1b/E1c, E2, E3, E4, E5, E6, E7, E8 with its numbers, its interval or the reason there is none,
// refused / not-run states with their reason, the leakage guards, and a warning banner for smoke and partial-data
// reports. Fixtures are trimmed copies of the real smoke reports (artifacts/isl/phase2-smoke/research/eval/report.json
// and the phase-1 artifacts/isl/commercial-safe/results.json).
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import axe from "axe-core";
import { UiLangProvider } from "../i18n";
import { IslReport, LanguageEvaluation, statusOf, type Protocol, type Results } from "./LanguageEvaluation";
import phase2Json from "./__fixtures__/isl_phase2_smoke_report.json";
import phase1Json from "./__fixtures__/isl_phase1_smoke_results.json";

const phase2 = () => structuredClone(phase2Json) as unknown as Results;
const phase1 = () => structuredClone(phase1Json) as unknown as Results;
const wrap = (r: Results, lang: "en" | "hi" = "en") =>
  render(<MemoryRouter><UiLangProvider initial={lang}><IslReport r={r} /></UiLangProvider></MemoryRouter>);
const row = (id: string) => document.querySelector(`tr[data-protocol="${id}"]`) as HTMLElement;
const card = (id: string) => screen.getAllByRole("heading", { level: 2 }).find((h) => within(h).queryByText(id, { exact: true }))!.closest("section") as HTMLElement;

describe("protocol status", () => {
  it("tells measured, not run, refused and missing apart", () => {
    expect(statusOf({ run: true, metrics: { top1: 0.5 } })).toBe("measured");
    expect(statusOf({ metrics: { p50_ms: 10 } })).toBe("measured"); // E6 has no run flag
    expect(statusOf({ run: false, reason: "no classifier" })).toBe("not_run");
    expect(statusOf({ run: false, n: 0, leakage: "classifier split unknown", reason: "refused: classifier split unknown" })).toBe("refused");
    expect(statusOf(undefined)).toBe("missing");
  });
});

describe("phase-2 smoke report", () => {
  it("lists every protocol E1a to E8 in the overview table", () => {
    wrap(phase2());
    const table = screen.getByRole("table", { name: /Every evaluation protocol/ });
    for (const id of ["E1a", "E1b", "E1c", "E2", "E2_seen", "E3", "E4", "E5", "E6", "E7", "E8"]) {
      expect(within(table).getByRole("rowheader", { name: new RegExp(`^${id} `) }), id).toBeInTheDocument();
    }
    expect(row("E1b")).toHaveTextContent("measured");
    expect(row("E1b")).toHaveTextContent("1.0%"); // top-1 0.0098
    expect(row("E3")).toHaveTextContent("8.2%");
    expect(row("E6")).toHaveTextContent("14 ms"); // p50
    expect(row("E7")).toHaveTextContent("yes");
  });

  it("shows a prominent smoke banner with the report's own label", () => {
    wrap(phase2());
    const alert = screen.getAllByRole("alert").find((a) => a.textContent?.includes("Smoke run"))!;
    expect(alert).toHaveTextContent("Smoke run: these numbers are NOT the app's accuracy");
    expect(alert).toHaveTextContent("SMOKE RUN on partial data with tiny settings");
    expect(alert).toHaveTextContent("The data was also incomplete");
    expect(screen.getByText("PARTIAL DATA")).toBeInTheDocument();
  });

  it("shows intervals with their group count, or the stated reason when there is none", () => {
    wrap(phase2());
    expect(row("E1b")).toHaveTextContent("none: too few signer groups (4)");
    expect(row("E3")).toHaveTextContent("6.1%–9.9% (6 groups)");
    const e1b = card("E1b");
    expect(within(e1b).getAllByText("too few signer groups (4)").length).toBeGreaterThan(0);
    expect(within(e1b).getByText("Cross-validation")).toBeInTheDocument();
    expect(e1b).toHaveTextContent("0.3%–1.1% (8 groups)"); // pooled CV interval
    const e3 = card("E3");
    expect(e3).toHaveTextContent("0.0%–2.0% (6 groups)"); // wrong-accept interval of a policy
    const e4 = card("E4");
    expect(within(e4).getByRole("table", { name: "By number of recordings per word (k)" })).toBeInTheDocument();
    expect(e4).toHaveTextContent("too few signer groups (3)"); // k5
    expect(row("E4")).toHaveTextContent("none: too few signer groups (4)"); // k3 interval reason in the overview
    expect(within(e3).getByRole("table", { name: "E3 metrics with 95% intervals" })).toHaveTextContent("69.8%–80.0% (6 groups)"); // not sure, balanced
  });

  it("shows not-run protocols with their reason", () => {
    wrap(phase2());
    expect(row("E5")).toHaveTextContent("not measured");
    expect(row("E5")).toHaveTextContent("leave-one-signer-out needs at least 2");
    expect(card("E5")).toHaveTextContent("Letter examples");
  });

  it("shows the leakage guards", () => {
    wrap(phase2());
    const sec = screen.getByRole("heading", { name: "Leakage guards" }).closest("section") as HTMLElement;
    expect(sec).toHaveTextContent("Pre-training excludes evaluation clips");
    expect(sec).toHaveTextContent("Computed: 0 of 200 pre-training clips are evaluation clips or their duplicates.");
    expect(sec).toHaveTextContent("not reliable: excluded Sign_Library");
    expect(sec).toHaveTextContent("1 of 400");
    expect(within(sec).getAllByText("yes").length).toBeGreaterThanOrEqual(2);
  });

  it("E1c not run (commercial-safe has no other signer-held-out source)", () => {
    const r = phase2();
    r.protocols!.E1c = { run: false, reason: "no other labelled source has a signer-held-out test part yet (needs a source with real signer ids and >= 6 signers, e.g. ISL500)" };
    wrap(r);
    expect(row("E1c")).toHaveTextContent("not measured");
    expect(row("E1c")).toHaveTextContent("no other labelled source has a signer-held-out test part yet");
  });

  it("marks a protocol that is missing from a phase-2 report", () => {
    const r = phase2();
    delete r.protocols!.E7;
    wrap(r);
    expect(row("E7")).toHaveTextContent("not in this report");
  });

  it("has no axe violations", async () => {
    const { container } = wrap(phase2());
    const res = await axe.run(container, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] },
      rules: { "color-contrast": { enabled: false } }, // checked in the browser
    });
    expect(res.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
  });
});

describe("refusals and leakage", () => {
  const refused = (label: string, why: string): Protocol => ({ label, run: false, n: 0, leakage: why, reason: `refused: ${why}` });

  it("shows protocols refused by the classifier guard and the guard's reason", () => {
    const r = phase2();
    const why = "classifier.pt has no recorded split: it may have seen these test clips";
    r.leakage_guards!.classifier_refused = why;
    r.protocols!.E1b = refused("E1b core words", why);
    r.protocols!.E3 = refused("E3 fused system", why);
    r.protocols!.E4 = refused("My signs few-shot (uses the word classifier)", why);
    wrap(r);
    for (const id of ["E1b", "E3", "E4"]) {
      expect(row(id)).toHaveTextContent("refused by a leakage guard");
      expect(row(id)).toHaveTextContent(`refused: ${why}`);
      expect(card(id)).toHaveTextContent(`refused: ${why}`);
    }
    const sec = screen.getByRole("heading", { name: "Leakage guards" }).closest("section") as HTMLElement;
    expect(sec).toHaveTextContent(why);
  });

  it("E2 whose every fold was refused by the encoder leakage check is shown as refused, not just not measured", () => {
    // runners.py: a LeakageError per fold gives {fold, run: false, leakage}; E2 then has run: false and the joined
    // leakage messages as its reason, but no top-level `leakage` key
    const r = phase2();
    const why = "encoder fold0 was trained on held-out concept clips";
    r.protocols!.E2 = { run: false, n: 0, reason: why, folds: [{ fold: 0, run: false, leakage: why }] };
    expect(statusOf(r.protocols!.E2)).toBe("refused");
    expect(statusOf({ run: false, reason: "empty gallery", folds: [{ fold: 0, run: false, reason: "empty gallery" }] })).toBe("not_run");
    wrap(r);
    expect(row("E2")).toHaveTextContent("refused by a leakage guard");
    expect(row("E2")).toHaveTextContent(why);
  });

  it("warns loudly when pre-training saw evaluation clips", () => {
    const r = phase2();
    r.leakage_guards!.pretraining_excludes_eval_clips = false;
    r.leakage_guards!.pretraining_check = { ...r.leakage_guards!.pretraining_check, excludes_eval_clips: false, overlap: 3, overlap_examples: ["include/abc", "include/def"] };
    wrap(r);
    const alert = screen.getAllByRole("alert").find((a) => a.textContent?.includes("Leakage"))!;
    expect(alert).toHaveTextContent("Leakage: pre-training saw evaluation clips");
    expect(alert).toHaveTextContent("include/abc, include/def");
  });

  it("says unknown, with the reason, when the pre-training check could not run", () => {
    const r = phase2();
    r.leakage_guards!.pretraining_excludes_eval_clips = null;
    r.leakage_guards!.pretraining_check = { excludes_eval_clips: null, reason: "no self-supervised pre-training in this build" };
    wrap(r);
    const sec = screen.getByRole("heading", { name: "Leakage guards" }).closest("section") as HTMLElement;
    expect(within(sec).getByText("unknown")).toBeInTheDocument();
    expect(sec).toHaveTextContent("no self-supervised pre-training in this build");
  });
});

describe("partial data and older reports", () => {
  it("a partial-data (not smoke) report gets its own danger banner", () => {
    const r = phase2();
    r.smoke = false;
    r.label = "PARTIAL DATA: trained and evaluated while ingestion was still running or a source manifest was incomplete (started with --allow-partial). These numbers are NOT the accuracy of the app.";
    wrap(r);
    const alert = screen.getAllByRole("alert").find((a) => a.textContent?.includes("Partial data"))!;
    expect(alert).toHaveTextContent("Partial data: these numbers are NOT the app's accuracy");
    expect(alert).toHaveTextContent("Do not ship this pack.");
    expect(screen.queryByText(/Smoke run/)).toBeNull();
  });

  it("a full report has no warning banner", () => {
    const r = phase2();
    r.smoke = false;
    r.partial_data = false;
    r.label = "Measured on held-out data.";
    wrap(r);
    expect(screen.queryByText(/NOT the app's accuracy/)).toBeNull();
    expect(screen.queryByText("PARTIAL DATA")).toBeNull();
  });

  it("renders a phase-1 results file (schema 1: E1, E2, E3, E6, no guards)", () => {
    wrap(phase1());
    expect(screen.getByText("Older report format (phase 1)")).toBeInTheDocument();
    for (const id of ["E1", "E2", "E3", "E6"]) expect(row(id), id).toBeInTheDocument();
    expect(row("E1b")).toBeNull(); // no "not in this report" rows for an old file
    expect(row("E1")).toHaveTextContent("50.0%");
    expect(row("E1")).toHaveTextContent("39.7%–81.0%"); // no group count in phase 1
    expect(screen.getByText("This older report has no leakage-guard results.")).toBeInTheDocument();
    expect(screen.getByText(/Smoke run/)).toBeInTheDocument();
  });
});

describe("Hindi interface", () => {
  it("translates the page's own words and marks the report's English text", () => {
    wrap(phase2(), "hi");
    expect(screen.getByText("सभी प्रोटोकॉल")).toBeInTheDocument();
    expect(screen.getByText("स्मोक रन: ये आँकड़े ऐप की सटीकता नहीं हैं")).toBeInTheDocument();
    expect(row("E5")).toHaveTextContent("नहीं मापा गया");
    expect(row("E1b")).toHaveTextContent("कोई अंतराल नहीं:");
    const reason = within(row("E1b")).getByText("too few signer groups (4)");
    expect(reason).toHaveAttribute("lang", "en");
    // the smoke badge and the commit label are the page's words too (not hard-coded English)
    expect(screen.getByText("स्मोक रन")).toBeInTheDocument();
    expect(screen.getByText("कमिट b00fe99")).toBeInTheDocument();
    expect(screen.queryByText("SMOKE")).not.toBeInTheDocument();
  });

  it("shows the smoke badge and the commit in English", () => {
    wrap(phase2());
    expect(screen.getByText("SMOKE")).toBeInTheDocument();
    expect(screen.getByText("commit b00fe99")).toBeInTheDocument();
  });
});

describe("LanguageEvaluation page", () => {
  afterEach(() => vi.unstubAllGlobals());
  const page = () => render(<MemoryRouter><UiLangProvider initial="en"><LanguageEvaluation /></UiLangProvider></MemoryRouter>);

  it("fetches the pack's results and renders them", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(phase2Json), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    page();
    await waitFor(() => expect(screen.getByRole("table", { name: /Every evaluation protocol/ })).toBeInTheDocument());
    expect(fetchMock.mock.calls[0][0]).toMatch(/^\/api\/languages\/[^/]+\/results$/);
    expect(screen.getByRole("heading", { level: 1, name: "Model Evaluation" })).toBeInTheDocument();
  });

  it("explains how to get results when the pack has none", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "no evaluation results for this language yet" }), { status: 404, headers: { "Content-Type": "application/json" } })));
    page();
    await waitFor(() => expect(screen.getByText("No evaluation results for this language yet")).toBeInTheDocument());
    expect(screen.getByText(/python -m app.isl.cli evaluate/)).toBeInTheDocument();
  });
});
