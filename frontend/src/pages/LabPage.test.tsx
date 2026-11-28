import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { ExperimentDetail, ExperimentSummary } from "../api/client";
import LabPage from "./LabPage";

// Regression test for the Model Laboratory reload loop: RunMonitor used to call onDone() whenever the
// monitored run was finished, with onDone (a new inline arrow on every render) in the effect's deps.
// With a finished run on screen (on a fresh clone the bundled model is one), every render reloaded the
// experiment list, which re-rendered the page, which reloaded again: hundreds of GET /api/experiments
// per second and "Maximum update depth exceeded".

function summary(status: string): ExperimentSummary {
  return {
    id: 1, name: "bundled", status, architecture: "gru", split_name: "signer-holdout", dataset_version: null,
    created_at: "2026-01-01T00:00:00Z", finished_at: status === "completed" ? "2026-01-01T01:00:00Z" : null,
    is_active: true, is_synthetic: false, device: "cpu", n_params: 1000, seq_len: 48, epochs_run: 0,
    epochs_planned: 20, last_epoch: null, error: null, checkpoint_path: null, val: null, test: null,
  };
}

function detail(status: string): ExperimentDetail {
  return {
    ...summary(status), config: { epochs: 20, features: {} }, feature_config: {}, data_summary: {}, history: [],
    val_metrics: {}, test_metrics: {}, split: null,
  };
}

/** Serves the endpoints LabPage polls; `status()` decides the run's state for each request. */
function mockBackend(status: () => string) {
  const calls: string[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    calls.push(url);
    let body: unknown;
    if (url === "/api/experiments") body = [summary(status())];
    else if (url === "/api/experiments/1") body = detail(status());
    else if (url === "/api/splits") body = [];
    else if (url === "/api/dataset/summary") body = {};
    else return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { listCalls: () => calls.filter((u) => u === "/api/experiments").length };
}

/** Lets fetches resolve and React commit, advancing the (fake) clock by `ms`. */
async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("LabPage run monitor", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("does not reload the experiment list in a loop when the monitored run is already finished", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    const backend = mockBackend(() => "completed");
    render(<MemoryRouter><LabPage /></MemoryRouter>);
    await advance(0);
    await advance(1000); // well before any polling interval (1.5 s detail, 4 s list)
    expect(screen.getByText(/Run #1/)).toBeInTheDocument();
    expect(backend.listCalls()).toBe(1); // only the initial load
    await advance(3200); // detail polls twice more, list polls once (t = 4.2 s)
    expect(backend.listCalls()).toBe(2);
    expect(errors.mock.calls.flat().join(" ")).not.toMatch(/Maximum update depth/);
    errors.mockRestore();
  });

  it("reloads the experiment list once when a running run finishes", async () => {
    let state = "running";
    const backend = mockBackend(() => state);
    render(<MemoryRouter><LabPage /></MemoryRouter>);
    await advance(0);
    expect(screen.getByRole("button", { name: "Run #1 in progress" })).toBeDisabled();
    expect(backend.listCalls()).toBe(1);

    state = "completed";
    await advance(1500); // the monitor's next poll sees the run finished and reloads the list once
    expect(backend.listCalls()).toBe(2);
    expect(screen.getByRole("button", { name: "Start training" })).toBeEnabled();

    await advance(1500); // further polls of the finished run do not reload it again (t = 3 s)
    expect(backend.listCalls()).toBe(2);
  });
});
