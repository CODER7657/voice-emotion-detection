import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Route } from "@playwright/test";

const fixture = (name: string) => new URL(`./fixtures/${name}`, import.meta.url);
const EXPECTED = new Float32Array(readFileSync(fixture("clip.expected.f32")).buffer.slice(0));

const PREDICTION = {
  emotion: "calm",
  probabilities: {
    calm: 0.58,
    neutral: 0.21,
    sad: 0.09,
    happy: 0.05,
    fearful: 0.03,
    surprised: 0.02,
    angry: 0.01,
    disgust: 0.01,
  },
  inference_ms: 707.2,
  model_sha256: "ea666f45",
  note: "acted English speech",
};

interface Sent {
  bodies: Buffer[];
}

/** Mock the API: /ready answers, /predict records the body and returns `reply`. */
async function mockApi(page: Page, reply: (route: Route) => Promise<void>): Promise<Sent> {
  const sent: Sent = { bodies: [] };
  await page.route("**/ready", (r) => r.fulfill({ json: { ready: true } }));
  await page.route("**/predict", async (route) => {
    const body = route.request().postDataBuffer();
    if (body) sent.bodies.push(body);
    await reply(route);
  });
  return sent;
}

const ok = (route: Route) => route.fulfill({ json: PREDICTION });

function correlation(a: Float32Array, b: Float32Array): number {
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  return ab / Math.sqrt(aa * bb);
}

function samples(body: Buffer): Float32Array {
  return new Float32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength));
}

async function upload(page: Page, name: string) {
  await page.locator("#file").setInputFiles(fixture(name).pathname.replace(/^\/(\w:)/, "$1"));
}

async function expectNoSeriousA11yIssues(page: Page) {
  const { violations } = await new AxeBuilder({ page }).analyze();
  const serious = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(
    serious.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`),
  ).toEqual([]);
}

const watched = new WeakMap<Page, string[]>();
const frozen = new WeakMap<Page, string[]>();

/** Screenshot after freezing the error log: WebKit screenshots inject a stylesheet of
 *  Playwright's own, which the page's CSP (rightly) refuses. */
async function screenshot(page: Page, path: string) {
  frozen.set(page, [...(watched.get(page) ?? [])]);
  await page.screenshot({ path, fullPage: true });
}

const hasWebAudio = (page: Page) => page.evaluate(() => typeof OfflineAudioContext === "function");

/** Playwright's Windows WebKit build ships without Web Audio (Linux WebKit and iOS have it). */
async function needsWebAudio(page: Page) {
  await page.goto("/");
  test.skip(!(await hasWebAudio(page)), "this engine build has no Web Audio");
}

test.beforeEach(({ page }) => {
  const problems: string[] = [];
  page.on("console", (m) => {
    // The browser logs every non-2xx response itself; the mocked 429/503 are on purpose.
    if (m.type() === "error" && !m.text().startsWith("Failed to load resource"))
      problems.push(m.text());
  });
  page.on("pageerror", (e) => problems.push(e.message));
  watched.set(page, problems);
});

test.afterEach(({ page }) => {
  // No CSP violations, no uncaught errors (the API mock answers every request).
  expect(frozen.get(page) ?? watched.get(page)).toEqual([]);
});

test("idle page: Aurora layout, accessible, no overflow, big targets", async ({ page }, info) => {
  await mockApi(page, ok);
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Emotion");
  await expect(page.locator(".glow")).toHaveCount(8);
  await expect(
    page.getByText("(acted English speech; not for decisions about people)"),
  ).toBeVisible();

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
  for (const target of [
    page.locator("#record"),
    page.locator("#file-label"),
    page.getByRole("link", { name: "Source" }),
  ]) {
    const box = await target.boundingBox();
    expect(box?.height ?? 0).toBeGreaterThanOrEqual(44);
    expect(box?.width ?? 0).toBeGreaterThanOrEqual(44);
  }
  const fonts = await page.evaluate(async () => {
    await document.fonts.ready;
    return ["800 1em Syne", "400 1em 'Space Grotesk'"].map((f) => document.fonts.check(f));
  });
  expect(fonts).toEqual([true, true]); // self-hosted fonts loaded under the CSP
  await expectNoSeriousA11yIssues(page);
  await screenshot(page, `test-results/screens/${info.project.name}-idle.png`);
});

test("a chosen file is prepared exactly like the training pipeline and shown", async ({
  page,
}, info) => {
  const sent = await mockApi(page, ok);
  await needsWebAudio(page);
  await upload(page, "clip.wav");

  await expect(page.locator("#verdict")).toHaveText("calm");
  await expect(page.locator("#status")).toHaveText("It sounds calm, 58%. Next: neutral, 21%.");
  await expect(page.locator(".bar")).toHaveCount(8);
  await expect(page.locator(".bar").first()).toContainText("58%");

  expect(sent.bodies).toHaveLength(1);
  const body = sent.bodies[0];
  if (!body) throw new Error("no body");
  expect(body.byteLength).toBe(192_000);
  // Browser decode + resample + trim vs librosa.load + vemo.trim on the same file.
  expect(correlation(samples(body), EXPECTED)).toBeGreaterThan(0.99);

  // Intensity follows probability: calm's glow is brighter than disgust's.
  const level = (i: number) =>
    page
      .locator(".glow")
      .nth(i)
      .evaluate((el) => Number(getComputedStyle(el).getPropertyValue("--level")));
  expect(await level(1)).toBeGreaterThan(await level(2)); // GLOWS order: angry, calm, disgust, ...

  await expectNoSeriousA11yIssues(page);
  await screenshot(page, `test-results/screens/${info.project.name}-result.png`);
});

test("the microphone path records, prepares and sends a real clip", async ({ page }, info) => {
  test.skip(info.project.name !== "mobile-375", "fake microphone is a Chromium flag");
  const sent = await mockApi(page, ok);
  await page.goto("/");
  await page.getByRole("button", { name: "Record 3 seconds" }).click();
  await expect(page.locator("#app")).toHaveAttribute("data-state", "recording");
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  await expect(page.locator("#verdict")).toHaveText("calm", { timeout: 15_000 });
  const body = sent.bodies[0];
  if (!body) throw new Error("no body");
  expect(body.byteLength).toBe(192_000);
  const peak = samples(body).reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  expect(peak).toBeGreaterThan(0.01); // real audio reached the request, not silence
});

test("rate limit: tells the visitor how long to wait", async ({ page }) => {
  await mockApi(page, (r) =>
    r.fulfill({ status: 429, headers: { "Retry-After": "42" }, json: { detail: "x" } }),
  );
  await needsWebAudio(page);
  await upload(page, "clip.wav");
  await expect(page.locator("#status")).toHaveText("Too many tries. Wait 42 s and record again.");
  await expect(page.locator("#result")).toBeHidden();
  await expect(page.locator("#app")).toHaveAttribute("data-state", "error");
});

test("server errors and bad input are explained, and the page recovers", async ({ page }) => {
  let fail = true;
  await mockApi(page, (r) => (fail ? r.fulfill({ status: 503, json: { detail: "x" } }) : ok(r)));
  await needsWebAudio(page);
  await upload(page, "clip.wav");
  await expect(page.locator("#status")).toContainText("starting or unavailable");

  await upload(page, "silence.wav");
  await expect(page.locator("#status")).toHaveText(
    "We could not hear any speech. Move closer and try again.",
  );

  await upload(page, "garbage.wav");
  await expect(page.locator("#status")).toContainText("could not be read as audio");

  fail = false;
  await upload(page, "clip.wav");
  await expect(page.locator("#verdict")).toHaveText("calm");
});

test("reduced motion stops the drifting glows", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockApi(page, ok);
  await page.goto("/");
  const animation = await page
    .locator(".glow")
    .first()
    .evaluate((el) => getComputedStyle(el).animationName);
  expect(animation).toBe("none");
});

test("without Web Audio the page says so instead of failing on every file", async ({ page }) => {
  await mockApi(page, ok);
  await page.addInitScript(() => {
    // @ts-expect-error simulate an engine without Web Audio
    delete window.OfflineAudioContext;
  });
  await page.goto("/");
  await expect(page.locator("#record-hint")).toHaveText(
    "(this browser cannot process audio; try Chrome, Firefox or Safari)",
  );
  await expect(page.locator("#record")).toBeDisabled();
  await expect(page.locator("#file")).toBeDisabled();
});
