import { expect, test } from "@playwright/test";

// Indian Sign Language flows. Needs a backend with the ISL language pack (the tracked models/packs/isl is enough:
// vocabulary + Hindi lexicon; without built models live recognition still runs with "My signs" only).

test("ISL is the default sign language and ASL is the second option", async ({ page }) => {
  await page.goto("/vocabulary");
  const switcher = page.getByRole("radiogroup", { name: "Sign language" });
  await expect(switcher.getByRole("radio", { name: "ISL" })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByText(/ISLRTC Indian Sign Language dictionary/)).toBeVisible();
  await switcher.getByRole("radio", { name: "ASL" }).click();
  await expect(page.getByText(/The 100 ASL signs the system can recognise/)).toBeVisible();
  await page.reload(); // the choice is remembered
  await expect(page.getByRole("radiogroup", { name: "Sign language" }).getByRole("radio", { name: "ASL" })).toHaveAttribute("aria-checked", "true");
  await page.getByRole("radiogroup", { name: "Sign language" }).getByRole("radio", { name: "ISL" }).click();
  await expect(page.getByText(/ISLRTC Indian Sign Language dictionary/)).toBeVisible();
});

test("vocabulary search works in English and Hindi and shows both", async ({ page }) => {
  await page.goto("/vocabulary");
  const search = page.getByRole("textbox", { name: "Search words in English or Hindi" });
  await search.fill("पानी");
  const card = page.getByRole("button", { name: /^water\s*पानी/i }).first();
  await expect(card).toBeVisible();
  await card.click();
  await expect(page.getByRole("heading", { name: "Word details" })).toBeVisible();
  await expect(page.getByText("isl.water", { exact: true })).toBeVisible();
  await search.fill("water");
  await expect(page.getByRole("button", { name: /^water\s*पानी/i }).first()).toBeVisible();
});

test("interface language toggle switches menus to Hindi and back", async ({ page }) => {
  await page.goto("/live");
  const ui = page.getByRole("radiogroup", { name: "Interface language" });
  await ui.getByRole("radio", { name: "हिंदी" }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("लाइव संकेत पहचान");
  await expect(page.getByRole("link", { name: "मेरे संकेत" }).first()).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("lang", "hi");
  await page.getByRole("radiogroup", { name: "ऐप की भाषा" }).getByRole("radio", { name: "EN" }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Live Sign Recognition");
});

test("live ISL: camera, MediaPipe and the ISL recognition socket", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/live");
  await expect(page.getByText(/Sign one Indian Sign Language word at a time/)).toBeVisible();
  const status = page.getByTestId("live-status");
  await page.getByRole("button", { name: "Start camera" }).first().click();
  await expect(status).toHaveText(/Detecting landmarks/, { timeout: 60_000 });
  const ws = page.waitForEvent("websocket", (w) => w.url().includes("/api/ws/recognize?language=isl"));
  await page.getByRole("button", { name: "Start recognition" }).click();
  await ws;
  await expect(status).toHaveText(/Waiting for hands|Body not visible/, { timeout: 20_000 });
  // the sentence strip and the speak controls are there before any word is signed
  await expect(page.getByRole("heading", { name: "Your sentence" })).toBeVisible();
  await expect(page.getByRole("radiogroup", { name: "Voice language" })).toBeVisible();
  // switching to ASL on an open socket keeps recognising (the server answers with a new hello)
  await page.getByRole("radiogroup", { name: "Sign language" }).getByRole("radio", { name: "ASL" }).click();
  await expect(status).toHaveText(/Waiting for hands|Body not visible|Recognition unavailable/, { timeout: 20_000 });
  await page.getByRole("radiogroup", { name: "Sign language" }).getByRole("radio", { name: "ISL" }).click();
  await page.getByRole("button", { name: "Stop camera" }).click();
  await expect(status).toHaveText("Camera inactive");
  expect(errors).toEqual([]);
});

test("my signs: recorder, form, sharing and alphabet sections", async ({ page }) => {
  await page.goto("/my-signs");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("My signs");
  await expect(page.getByText("Only body and hand points are saved, on this computer.")).toBeVisible();
  const record = page.getByRole("button", { name: "Record take 1" });
  await expect(record).toBeDisabled();
  await page.getByRole("button", { name: "Start camera" }).click();
  await expect(record).toBeEnabled({ timeout: 60_000 });
  await record.click();
  await expect(page.getByTestId("recorder-status")).toHaveText(/Raise your hands and sign/);
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByTestId("take-count")).toHaveText("0 recorded (need 3 to 5)");
  // saving needs 3 takes and a word
  await page.getByRole("textbox", { name: "English word" }).fill("water");
  await expect(page.getByRole("button", { name: "Check and save" })).toBeDisabled();
  await expect(page.getByRole("button", { name: /^water/ }).first()).toBeVisible(); // link to the dictionary word
  // export needs consent
  await expect(page.getByRole("button", { name: /^Export/ })).toBeDisabled();
  await expect(page.getByRole("button", { name: /Record letter A/ })).toBeEnabled();
});
