/**
 * Against a real deployment, no mocks: VEMO_LIVE_URL=https://... npx playwright test live
 * Runs the real model through the real page (file and fake-microphone paths).
 */
import { expect, test } from "@playwright/test";

const live = process.env.VEMO_LIVE_URL;
test.skip(!live, "set VEMO_LIVE_URL to run against a deployment");
test.use({ baseURL: live ?? "" });

const clip = new URL("./fixtures/clip.wav", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
const EMOTION = /^(angry|calm|disgust|fearful|happy|neutral|sad|surprised)$/;

test("a file goes through the live model", async ({ page }) => {
  await page.goto("/");
  await page.locator("#file").setInputFiles(clip);
  await expect(page.locator("#verdict")).toHaveText(EMOTION, { timeout: 45_000 });
  await expect(page.locator(".bar")).toHaveCount(8);
  await expect(page.locator("#status")).toContainText("It sounds");
});

test("the microphone goes through the live model", async ({ page }, info) => {
  test.skip(info.project.name !== "mobile-375", "fake microphone is a Chromium flag");
  await page.goto("/");
  await page.getByRole("button", { name: "Record 3 seconds" }).click();
  await expect(page.locator("#verdict")).toHaveText(EMOTION, { timeout: 45_000 });
});
