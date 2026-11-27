import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// Accessibility audit for deaf and low-literacy users in India (docs/ISL_DESIGN.md section 14). Needs the same servers
// as the other e2e tests (backend with the ISL pack + Vite dev server). Checks WCAG 2.2 A/AA rules axe-core can test,
// plus things axe cannot: every Hindi word is marked lang="hi" and drawn with a Devanagari font without letter
// spacing, English-only pages are marked lang="en" in the Hindi interface, nothing scrolls sideways at 320 px (400%
// zoom), keyboard focus follows the vocabulary detail panel, and the three result states look different.

const WCAG = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];
const PAGES = ["/", "/live", "/my-signs", "/vocabulary", "/evaluation", "/analyze", "/studio", "/dataset", "/lab", "/experiments", "/research"];
const NEW_PAGES = ["/live", "/my-signs", "/vocabulary", "/evaluation"];

async function setPrefs(page: Page, ui: "en" | "hi", sign: "isl" | "asl-wlasl100") {
  await page.addInitScript(([u, s]) => {
    localStorage.setItem("ui-lang-v1", u);
    localStorage.setItem("sis.language", s);
  }, [ui, sign] as const);
}

async function settle(page: Page) {
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 30_000 });
  await page.waitForTimeout(1000); // let data requests fill the page
}

async function axe(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(WCAG).analyze();
  return r.violations.map((v) => `${v.id} (${v.impact}): ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

/** Devanagari text nodes that are not inside a lang="hi*" element, have letter spacing, or are smaller than 12px. */
async function devanagariProblems(page: Page) {
  return page.evaluate(() => {
    const out: string[] = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = n.textContent ?? "";
      if (!/[ऀ-ॿ]/.test(text)) continue;
      const el = n.parentElement!;
      if (el.closest(".sr-only, [aria-hidden=true]") || !el.getClientRects().length) continue;
      const lang = el.closest("[lang]")?.getAttribute("lang") ?? "";
      const cs = getComputedStyle(el);
      const ls = cs.letterSpacing;
      const short = text.trim().slice(0, 30);
      if (!lang.startsWith("hi")) out.push(`no lang=hi: "${short}" (${lang})`);
      if (ls !== "normal" && parseFloat(ls) !== 0) out.push(`letter-spacing ${ls}: "${short}"`);
      if (parseFloat(cs.fontSize) < 12) out.push(`font-size ${cs.fontSize}: "${short}"`);
    }
    return out;
  });
}

for (const ui of ["en", "hi"] as const) {
  for (const sign of ["isl", "asl-wlasl100"] as const) {
    for (const path of PAGES) {
      test(`axe WCAG 2.2 AA and Hindi marking: ${ui} interface, ${sign}, ${path}`, async ({ page }) => {
        await setPrefs(page, ui, sign);
        await page.goto(path);
        await settle(page);
        const problems = [...await axe(page), ...await devanagariProblems(page)];
        if (ui === "hi") {
          // English-only pages are marked lang="en" so screen readers use an English voice for them.
          const lang = await page.getByTestId("page-content").getAttribute("lang");
          const translated = path === "/live" || path === "/my-signs" || (path === "/vocabulary" && sign === "isl");
          if (!translated && lang !== "en") problems.push("untranslated page not marked lang=en");
          if (translated && lang) problems.push(`translated page marked lang=${lang}`);
        }
        expect(problems).toEqual([]);
      });
    }
  }
}

test("Hindi uses the bundled Devanagari font", async ({ page }) => {
  await setPrefs(page, "hi", "isl");
  await page.goto("/live");
  await settle(page);
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => document.fonts.check('16px "Noto Sans Devanagari"', "पानी"))).toBe(true);
  // the eyebrow used letter spacing, which pulled "शब्द" apart
  const eyebrow = page.getByText(/^लाइव · ISL/);
  await expect(eyebrow).toHaveCSS("letter-spacing", "normal");
});

for (const ui of ["en", "hi"] as const) {
  test(`reflow at 320 px (400% zoom), ${ui} interface`, async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 720 });
    await setPrefs(page, ui, "isl");
    for (const path of NEW_PAGES) {
      await page.goto(path);
      await settle(page);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${path} scrolls sideways`).toBeLessThanOrEqual(0);
    }
    // the "camera off" panel is not clipped: its Start camera button is fully inside the camera area
    await page.goto("/live");
    const stage = page.locator("video").locator("..");
    const btn = stage.getByRole("button", { name: ui === "hi" ? "कैमरा चालू करें" : "Start camera" });
    const [s, b] = [await stage.boundingBox(), await btn.boundingBox()];
    expect(s && b && b.y >= s.y && b.y + b.height <= s.y + s.height).toBe(true);
  });
}

test("keyboard: vocabulary detail takes focus and gives it back", async ({ page }) => {
  await setPrefs(page, "en", "isl");
  await page.goto("/vocabulary");
  await page.getByRole("textbox", { name: "Search words in English or Hindi" }).fill("water");
  const card = page.getByRole("button", { name: /^water\s*पानी/i }).first();
  await card.focus();
  await page.keyboard.press("Enter");
  const detail = page.getByTestId("word-detail");
  await expect(detail).toBeFocused();
  await expect(page.getByRole("button", { name: "Pause the moving figure" }).or(page.getByText("No preview on this computer", { exact: false })).first()).toBeVisible();
  await detail.getByRole("button", { name: "Close" }).click();
  await expect(page.getByRole("button", { name: /^water\s*पानी/i }).first()).toBeFocused();
});

test("my signs says why it cannot save yet", async ({ page }) => {
  await setPrefs(page, "en", "isl");
  await page.goto("/my-signs");
  const save = page.getByRole("button", { name: "Check and save" });
  await expect(save).toBeDisabled();
  await expect(page.getByTestId("save-needs")).toContainText("at least 3 times");
  await expect(page.getByTestId("save-needs")).toContainText("Type the English word.");
});

// Recognised / did you mean / not sure, driven by a mocked recognition socket (the fake camera shows no person).
test("live result states are visually distinct, announced, and pass axe", async ({ page }) => {
  test.setTimeout(180_000);
  await setPrefs(page, "en", "isl");
  let server: { send: (m: string) => void } | null = null;
  await page.routeWebSocket(/\/api\/ws\/recognize/, (ws) => {
    server = ws;
    ws.onMessage((m) => {
      if (typeof m === "string" && m.includes('"type":"config"') && m.includes('"width"')) {
        ws.send(JSON.stringify({
          type: "hello",
          model: { name: "ISL test", num_classes: 3, version: "test", profile: "commercial-safe", architecture: "isl-hybrid", components: {} },
          language: { id: "isl", sign_language: "ISL", capabilities: ["words", "my_signs"], output_languages: ["en", "hi"], profile: "commercial-safe", status: "ready", display_name: { en: "ISL", hi: "ISL" } },
        }));
      }
    });
  });
  const cand = (en: string, hi: string, prob: number, id: string) => ({ concept_id: id, class_index: 1, gloss: en, prob, text: { en, hi }, hi_review: "curated" });
  const state = (status: string, segment_result: object, segId: number, commit?: object) => JSON.stringify({
    type: "state", t: 0, status, hands: false, pose: true, threshold: 0.7, server_ms: 1, segment_result: { segment_id: segId, frames: 30, ...segment_result }, ...(commit ? { commit } : {}),
  });

  await page.goto("/live");
  await page.getByRole("button", { name: "Start camera" }).first().click();
  await expect(page.getByTestId("live-status")).toHaveText(/Detecting landmarks|Waiting for hands|Body not visible/, { timeout: 120_000 });
  await page.getByRole("button", { name: "Start recognition" }).click();
  await expect.poll(() => server !== null, { timeout: 20_000 }).toBe(true);

  const water = cand("water", "पानी", 0.93, "isl.water");
  server!.send(state("recognized", { accepted: true, decision: "accept", reason: "ok", top: [water], confidence: 0.93 }, 1,
    { gloss: "water", class_index: 1, score: 0.93, t: 0, concept_id: "isl.water", text: { en: "water", hi: "पानी" }, segment_id: 1 }));
  const accepted = page.locator('[data-state="accepted"]');
  await expect(accepted).toBeVisible();
  await expect(page.getByTestId("result-announcer")).toHaveText("Recognised: water, पानी");
  expect(await axe(page)).toEqual([]);
  const styles: string[] = [await accepted.evaluate((e) => getComputedStyle(e).borderColor + getComputedStyle(e).borderStyle)];

  server!.send(state("uncertain", { accepted: false, decision: "suggest", reason: "low_confidence", top: [cand("drink", "पीना", 0.4, "isl.drink"), water] }, 2));
  const suggest = page.locator('[data-state="suggest"]');
  await expect(suggest).toBeVisible();
  await expect(page.getByTestId("result-announcer")).toContainText("possible words are shown");
  expect(await axe(page)).toEqual([]);
  styles.push(await suggest.evaluate((e) => getComputedStyle(e).borderColor + getComputedStyle(e).borderStyle));
  // choosing by keyboard: focus lands on the confirmation, not on the page body
  await page.getByRole("button", { name: /^Choose drink/ }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("resolved")).toBeFocused();

  server!.send(state("not_sure", { accepted: false, decision: "abstain", reason: "out_of_vocabulary", top: [] }, 3));
  const notSure = page.locator('[data-state="not-sure"]');
  await expect(notSure).toBeVisible();
  await expect(page.getByTestId("result-announcer")).toHaveText("Not sure – try again or spell it");
  expect(await axe(page)).toEqual([]);
  styles.push(await notSure.evaluate((e) => getComputedStyle(e).borderColor + getComputedStyle(e).borderStyle));
  expect(new Set(styles).size).toBe(3);

  // docs/ISL_DESIGN.md 15.29: a certified word still waits for a tap (keyboard here) before it joins the sentence
  const certified = { rule: "certified", certified_precision: 0.953, worst_source: "include", confirm_tap: true };
  server!.send(state("recognized", { accepted: true, decision: "accept", reason: "ok", top: [water], confidence: 0.97, certification: certified, needs_confirm: true }, 4,
    { gloss: "water", class_index: 1, score: 0.97, t: 0, concept_id: "isl.water", text: { en: "water", hi: "पानी" }, segment_id: 4, certification: certified, needs_confirm: true }));
  await expect(page.locator('[data-state="confirm-recognised"]')).toBeVisible();
  await expect(page.getByTestId("result-announcer")).toHaveText("Recognised, please confirm: water, पानी");
  expect(await axe(page)).toEqual([]);
  const words = page.getByTestId("transcript").getByTestId("bilingual-word");
  const before = await words.count();
  await page.getByRole("button", { name: "Add water" }).focus();
  await page.keyboard.press("Enter");
  await expect(words).toHaveCount(before + 1);
  await expect(page.getByTestId("resolved")).toBeFocused();

  await page.getByRole("button", { name: "Stop camera" }).click();
});
