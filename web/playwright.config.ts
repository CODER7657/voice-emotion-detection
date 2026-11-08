import { defineConfig, devices } from "@playwright/test";

const clip = new URL("./e2e/fixtures/clip.wav", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { baseURL: "http://localhost:4317", trace: "retain-on-failure" },
  webServer: {
    command: "npm run build && npm run preview",
    url: "http://localhost:4317",
    reuseExistingServer: false, // always test a fresh build
    timeout: 120_000,
  },
  projects: [
    {
      name: "mobile-375",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 375, height: 812 },
        hasTouch: true,
        launchOptions: {
          // A fake microphone that plays the fixture clip: the record path runs for real.
          args: [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            `--use-file-for-fake-audio-capture=${clip}`,
          ],
        },
      },
    },
    {
      name: "tablet-768",
      use: { ...devices["Desktop Chrome"], viewport: { width: 768, height: 1024 } },
    },
    {
      name: "desktop-1280",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } },
    },
    // WebKit with an iPhone profile: the closest to iOS Safari that runs on CI.
    { name: "iphone-webkit", use: { ...devices["iPhone 13"] } },
  ],
});
