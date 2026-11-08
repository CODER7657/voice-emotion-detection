import { readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vitest/config";

// In development /health, /ready and /predict go to a local API (uvicorn api.index:app);
// in production the site and the API share one Vercel origin, so no CORS is involved.
const api = process.env.VEMO_API ?? "http://127.0.0.1:8000";

// One source for the headers: Vercel serves them (scripts/build_vercel_bundle.py), and the
// built page carries the CSP as a <meta> too, so `vite preview` (and the e2e tests) enforce it.
const headers = JSON.parse(
  readFileSync(new URL("./security-headers.json", import.meta.url), "utf8"),
) as Record<string, string>;
const metaCsp = (headers["Content-Security-Policy"] ?? "")
  .split("; ")
  // frame-ancestors is ignored in <meta>; upgrade-insecure-requests would send WebKit to
  // https://localhost under `vite preview`. Production gets both from the real header (and HSTS).
  .filter((d) => !d.startsWith("frame-ancestors") && d !== "upgrade-insecure-requests")
  .join("; ");

const cspMeta: Plugin = {
  name: "csp-meta",
  apply: "build",
  transformIndexHtml: () => [
    {
      tag: "meta",
      attrs: { "http-equiv": "Content-Security-Policy", content: metaCsp },
      injectTo: "head-prepend",
    },
  ],
};

// Preload the three latin faces the first paint uses, so text does not reflow when they land.
const FIRST_PAINT_FONTS = [
  /syne-latin-800-normal-.*\.woff2$/,
  /space-grotesk-latin-(400|500)-normal-.*\.woff2$/,
];
const preloadFonts: Plugin = {
  name: "preload-fonts",
  apply: "build",
  transformIndexHtml: (_html, ctx) =>
    Object.keys(ctx.bundle ?? {})
      .filter((file) => FIRST_PAINT_FONTS.some((re) => re.test(file)))
      .map((file) => ({
        tag: "link",
        attrs: {
          rel: "preload",
          href: `/${file}`,
          as: "font",
          type: "font/woff2",
          crossorigin: "",
        },
        injectTo: "head" as const,
      })),
};

export default defineConfig({
  plugins: [cspMeta, preloadFonts],
  build: { target: "es2022", assetsInlineLimit: 0, sourcemap: false },
  server: { proxy: { "/predict": api, "/ready": api, "/health": api } },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
