import { expect, test } from "@playwright/test";

const PAGES: [string, RegExp][] = [
  ["/", /Overview/],
  ["/live", /Live Sign Recognition/],
  ["/analyze", /Video Analysis/],
  ["/vocabulary", /Sign Vocabulary Explorer/],
  ["/studio", /Landmark Visualisation Studio/],
  ["/dataset", /Dataset Explorer/],
  ["/lab", /Model Laboratory/],
  ["/evaluation", /Model Evaluation/],
  ["/experiments", /Experiment History/],
  ["/research", /Research & Documentation/],
];

test("every page renders without console errors and the backend is reachable", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  for (const [path, heading] of PAGES) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(heading);
  }
  await expect(page.getByText("backend", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("live page: camera permission, MediaPipe in the browser, recognition socket", async ({ page }) => {
  await page.goto("/live");
  const status = page.getByTestId("live-status");
  await expect(status).toHaveText("Camera inactive");
  await page.getByRole("button", { name: "Start camera" }).first().click();
  await expect(page.getByText("Camera on")).toBeVisible();
  // MediaPipe Holistic (WASM) loads locally and starts tracking
  await expect(status).toHaveText(/Detecting landmarks/, { timeout: 60_000 });
  await page.getByRole("button", { name: "Start recognition" }).click();
  // With a real person on the fake camera: body visible, no hands -> waiting for hands.
  // With Chromium's synthetic pattern (no person): body not visible.
  await expect(status).toHaveText(/Waiting for hands|Body not visible/, { timeout: 20_000 });
  await expect(page.getByText(/landmarks \d+(\.\d)? fps/)).toBeVisible();
  await page.getByRole("button", { name: "Pause" }).click();
  await expect(status).toHaveText("Recognition paused");
  await page.getByRole("button", { name: "Resume" }).click();
  await page.getByRole("button", { name: "Stop camera" }).click();
  await expect(status).toHaveText("Camera inactive");
  await expect(page.getByText("Camera on")).toHaveCount(0);
});

test("keyboard: skip link, focus after navigation, experiment details, mobile menu", async ({ page }) => {
  await page.goto("/vocabulary");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to main content" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("main")).toBeFocused();

  await page.getByRole("link", { name: "Experiment History" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveTitle(/Experiment History/);
  await expect(page.locator("main")).toBeFocused();
  const details = page.getByRole("button", { name: /^Details for run/ }).first();
  await details.focus();
  await page.keyboard.press("Enter");
  await expect(details).toHaveAttribute("aria-expanded", "true");

  await page.setViewportSize({ width: 390, height: 844 });
  const menu = page.getByRole("button", { name: "Open navigation" });
  await menu.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("button", { name: "Close navigation" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(menu).toBeFocused();
});
